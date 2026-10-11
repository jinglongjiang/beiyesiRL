"""Fixed-root causal contrasts for the repaired GRU; no new navigation algorithm."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = HERE.parent / 'recovery-duration'
FINDINGS = HERE.parent / 'navigation-findings'
OUT = HERE / 'artifacts'
spec = importlib.util.spec_from_file_location('historical_env_helpers', OLD/'run.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
PROTO = json.loads((HERE/'protocol.json').read_text())


def save(name, obj):
    m.save_json(HERE/name, obj)


def setup():
    import configparser
    cfg = configparser.ConfigParser()
    assert cfg.read(OLD/'parent-config.ini')
    m.contracts.init_grid_from_cfg(cfg)
    assert m.contracts.grid_action_dim(m.contracts.GRID) == 80
    for module in [m.crowd_nav,m.crowd_sim,m.mamba_rl]:
        assert Path(module.__file__).resolve().is_relative_to(ROOT)
    torch.set_num_threads(2)
    torch.manual_seed(61)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    device='cuda' if torch.cuda.is_available() else 'cpu'
    p=m.MambaRLPolicy(cfg,device).eval()
    p.multiagent_training=True
    p.use_sarl_predict=True
    p.time_step=.25
    p.build_action_space(1.)
    assert len(p.action_space)==80 and p.temporal_backbone=='gru'
    weight=OLD/'artifacts'/'nominal.pth'
    assert hashlib.sha256(weight.read_bytes()).hexdigest()==PROTO['nominal_sha256']
    p.load_state_dict(torch.load(weight,map_location=device))
    p.set_phase('test')
    OUT.mkdir(exist_ok=True)
    return cfg,p


def load_tape(scene,case,long=False,cfg=None,p=None):
    path=OLD/'artifacts'/'tapes'/f'{scene}-{case}.npz'
    assert path.exists()
    old=np.load(path)['humans']
    if not long:
        return old
    new_path=OUT/'long-tapes'/path.name
    if new_path.exists():
        tape=np.load(new_path)['humans']
        np.testing.assert_allclose(tape[:len(old)],old,rtol=0,atol=1e-9)
        return tape
    env=m.make_env(cfg,p,scene)
    env.reset(options={'test_case':case})
    assert not env.robot.visible
    rows=[]
    for _ in range(203):
        rows.append([[h.px,h.py,h.vx,h.vy,h.radius] for h in env.humans])
        acts=[h.act([o.get_observable_state() for o in env.humans if o is not h]) for h in env.humans]
        for h,a in zip(env.humans,acts):
            h.step(a)
    tape=np.asarray(rows,dtype=np.float64)
    np.testing.assert_allclose(tape[:len(old)],old,rtol=0,atol=1e-9)
    new_path.parent.mkdir(exist_ok=True)
    np.savez_compressed(new_path,humans=tape)
    return tape


def environment(cfg,p,scene,case,tape,snapshot=None,limit=25):
    env=m.tape_env(cfg,p,scene,case,tape,snapshot)
    env.time_limit=float(limit)
    p.set_env(env)
    return env


def decompose(p,env):
    state=m.state(env)
    commands,tok,windows,rewards,clear=p._score_candidates_vectorized(state)
    with torch.no_grad():
        values=p.forward_candidate_values(windows)
        r=torch.tensor(rewards.tolist(),device=p.device)
        q=p.candidate_action_values(state,commands,r,values,env)
        filtering=(p._phase in ['test','val','eval'] and (p.test_min_clearance>0 or p.test_risk_lambda>0))
        margin=p.test_min_clearance if p.test_min_clearance>0 else p.discomfort_dist
        risk=(p.test_risk_lambda*torch.clamp(margin-torch.tensor(clear.tolist(),device=p.device),min=0)
              if filtering else torch.zeros_like(q))
        scores=q-risk
        safe=clear>=p.test_min_clearance
        if filtering and p.test_min_clearance>0 and safe.any():
            scores=scores.masked_fill(torch.tensor(~safe,device=p.device),-1e9)
        scores[0]-=.001
    pos=np.asarray(env.robot.get_position())
    goal=np.asarray(env.robot.get_goal_position())
    progress=np.linalg.norm(pos-goal)-np.linalg.norm(pos+np.asarray(commands)*.25-goal,axis=1)
    allowed=safe if filtering and p.test_min_clearance>0 and safe.any() else np.ones(80,dtype=bool)
    best=int(scores.argmax())
    forward=int(np.where(allowed,progress,-np.inf).argmax())
    terminal,tr=p.candidate_terminal_outcomes(state,commands,env)
    effective=np.where(terminal,tr,rewards)
    flag=bool(progress[best]<=0 and progress[forward]>=.02 and effective[forward]>=effective[best]-1e-8)
    return {'commands':commands,'token':tok,'windows':windows,'reward':effective,'raw_reward':rewards,
            'value':values.cpu().numpy(),'q':q.cpu().numpy(),'risk':risk.cpu().numpy(),
            'scores':scores.cpu().numpy(),'clear':clear,'progress':progress,
            'safe':safe,'best':best,'forward':forward,'flag':flag}


def snapshot(p,env):
    return {'robot':copy.copy(env.robot.get_full_state()),'time':env.global_time,
            'history':[h.copy() for h in p._history],'last_action':p._last_action}


def rollout(cfg,p,scene,case,tape,snap=None,forced=None,variant='deploy',draw=0,keep=True):
    old=(p.test_action_smoothing,p.test_min_clearance,p.test_risk_lambda)
    limit=50 if variant=='deadline50' else 25
    env=environment(cfg,p,scene,case,tape,snap,limit)
    if variant=='no-smoothing':
        p.test_action_smoothing=0.
    if variant=='no-filter':
        p.test_min_clearance=p.test_risk_lambda=0.
    rng=np.random.default_rng(12345+scene*1000000+case*100+draw)
    explore=rng.random(203)
    random_actions=rng.integers(80,size=203)
    records=[]
    sampled=[]
    decision_times=[]
    first=None
    try:
        for j in range(203):
            step=int(round(env.global_time/.25))
            assert step+1<len(tape)
            np.testing.assert_allclose([[h.px,h.py,h.vx,h.vy,h.radius] for h in env.humans],tape[step],rtol=0,atol=1e-8)
            start=time.perf_counter()
            part=decompose(p,env)
            if forced is not None and j==0:
                idx,action=forced
            elif variant=='explore' and explore[step]<.05:
                idx=int(random_actions[step])
                action=part['commands'][idx]
            else:
                idx=part['best']
                action=part['commands'][idx]
            decision_times.append(time.perf_counter()-start)
            if step in PROTO['baseline']['root_steps']:
                sampled.append({'step':step,'position':list(env.robot.get_position()),
                    'best':part['best'],'forward':part['forward'],'flag':part['flag'],
                    'selected_progress':float(part['progress'][part['best']]),
                    'forward_progress':float(part['progress'][part['forward']]),
                    'safe_count':int(part['safe'].sum()),
                    'q_gap_selected_minus_forward':float(part['q'][part['best']]-part['q'][part['forward']]),
                    'value_gap_selected_minus_forward':float(part['value'][part['best']]-part['value'][part['forward']])})
            before=m.before_geometry(env)
            rob=env.robot.get_position()
            m.commit_action(p,part['token'],action,idx)
            _,r,term,trunc,info=env.step(action)
            clearance=m.geo_step(env,before)
            records.append([step,*rob,action.vx,action.vy,idx,part['best'],r,clearance,part['flag']])
            if j==0:
                first={'reward':r,'done':bool(term or trunc),'next_robot':copy.copy(env.robot.get_full_state()),
                       'next_humans':[copy.copy(h.get_observable_state()) for h in env.humans]}
            if term or trunc:
                break
        assert term or trunc
        a=np.asarray(records,dtype=np.float64)
        result={'scene':m.SCENES[scene],'case':case,'variant':variant,'draw':draw,
                'outcome':info['event'],'success':int(info['event']=='reach_goal'),
                'collision':int(info['event']=='collision'),'timeout':int(info['event']=='timeout'),
                'contact':int((a[:,8]<0).any()),'min_clearance':float(a[:,8].min()),
                'duration':len(a)*.25,'steps':len(a),'return':float(a[:,7]@(.99**np.arange(len(a)))),
                'path':float(np.linalg.norm(a[:,3:5],axis=1).sum()*.25),
                'final_distance':float(np.linalg.norm(np.asarray(env.robot.get_position())-env.robot.get_goal_position())),
                'decision_ms_mean':float(np.mean(decision_times)*1000),
                'flagged_roots':sum(r['flag'] for r in sampled),'sampled_roots':sampled}
        return result,a,first
    finally:
        p.test_action_smoothing,p.test_min_clearance,p.test_risk_lambda=old


def frozen_root(cfg,p,root,tape):
    scene=m.SCENES.index(root['scene'])
    case=root['case']
    env=environment(cfg,p,scene,case,tape)
    rows=np.load(OLD/'artifacts'/'development-traces'/'nominal'/f'{scene}-{case}.npz')['records']
    for row in rows:
        step=int(row[0])
        np.testing.assert_allclose(env.robot.get_position(),row[1:3],atol=1e-9,rtol=0)
        if step==root['step']:
            return env,snapshot(p,env)
        tok=m.token(p,m.state(env))
        action=m.ActionXY(*row[3:5])
        m.commit_action(p,tok,action,int(row[5]))
        _,_,term,trunc,_=env.step(action)
        assert not(term or trunc)
    raise AssertionError('Frozen root not reached')


def preflight():
    cfg,p=setup()
    tests=[]
    for scene,case in [(0,100),(0,115),(5,102)]:
        tape=load_tape(scene,case)
        env=environment(cfg,p,scene,case,tape)
        part=decompose(p,env)
        cmd=p.predict_sarl_style(m.state(env))
        assert p._selected_action_index==part['best']
        np.testing.assert_array_equal(cmd,part['commands'][part['best']])
        a,x,_=rollout(cfg,p,scene,case,tape)
        b,y,_=rollout(cfg,p,scene,case,tape)
        np.testing.assert_array_equal(x,y)
        tests.append({'scene':m.SCENES[scene],'case':case,'canonical_match':True,'deterministic':True,
                      'outcome':a['outcome'],'steps':a['steps']})
    save('preflight.json',{'passed':True,'actions':80,'tests':tests})
    print(json.dumps(tests),flush=True)


def baseline():
    cfg,p=setup()
    rows=[]
    old={(r['scene'],r['case']):r for r in json.loads((OLD/'artifacts'/'development-results.json').read_text()) if r['arm']=='nominal'}
    for scene in range(6):
        for case in range(100,140):
            tape=load_tape(scene,case)
            r,a,_=rollout(cfg,p,scene,case,tape)
            r['old_outcome']=old[(r['scene'],case)]['outcome']
            path=OUT/'repaired-traces'/f'{scene}-{case}.npz'
            path.parent.mkdir(exist_ok=True)
            np.savez_compressed(path,records=a)
            rows.append(r)
            save('repaired-baseline.json',rows)
            if len(rows)%10==0:
                print('baseline',len(rows),'success',sum(v['success'] for v in rows),'changed',sum(v['old_outcome']!=v['outcome'] for v in rows),flush=True)


def contrasts(variants=None):
    cfg,p=setup()
    roots=json.loads((FINDINGS/'counterfactual-protocol.json').read_text())['roots']
    rows=[]
    repair={ (r['scene'],r['case']):r for r in json.loads((HERE/'repaired-baseline.json').read_text())}
    for n,root in enumerate(roots):
        scene=m.SCENES.index(root['scene'])
        case=root['case']
        tape=load_tape(scene,case)
        env,snap=frozen_root(cfg,p,root,tape)
        part=decompose(p,env)
        actions=[(part[k],part['commands'][part[k]]) for k in ['best','forward']]
        prefix=list(snap['history'])+[part['token']]
        predicted_humans=[p.propagate(h,m.ActionXY(h.vx,h.vy)) for h in m.state(env).human_states]
        outcomes=[]
        firsts=[]
        for label,forced in zip(['selected','forward'],actions):
            result,trace,first=rollout(cfg,p,scene,case,tape,snap,forced)
            path=OUT/'contrasts'/f'{scene}-{case}-{label}.npz'
            path.parent.mkdir(exist_ok=True)
            np.savez_compressed(path,records=trace)
            outcomes.append(result)
            firsts.append(first)
        np.testing.assert_array_equal(
            [[h.px,h.py,h.vx,h.vy,h.radius] for h in firsts[0]['next_humans']],
            [[h.px,h.py,h.vx,h.vy,h.radius] for h in firsts[1]['next_humans']])
        windows=[]
        for field in ['cv','positions','velocities','actual']:
            for first in firsts:
                humans=[]
                for pred,actual in zip(predicted_humans,first['next_humans']):
                    pos=actual if field in ['positions','actual'] else pred
                    vel=actual if field in ['velocities','actual'] else pred
                    humans.append(m.ObservableState(pos.px,pos.py,vel.vx,vel.vy,actual.radius))
                joint=m.JointState(first['next_robot'],humans)
                windows.append(m.window(prefix+[m.token(p,joint)]))
        with torch.no_grad():
            values=p.forward_candidate_values(torch.as_tensor(np.stack(windows),device=p.device)).cpu().numpy().reshape(4,2)
        np.testing.assert_allclose(values[0],part['value'][[part['best'],part['forward']]],rtol=0,atol=2e-6)
        bq=np.asarray([f['reward']+.99*(not f['done'])*v for f,v in zip(firsts,values[3])])
        record={'scene':root['scene'],'case':case,'step':root['step'],'partition':root['partition'],
                'root_flag_after_repair':part['flag'],
                'historical_outcome':root['outcome'],'repaired_episode_outcome':repair[(root['scene'],case)]['outcome'],
                'selected_index':part['best'],'forward_index':part['forward'],
                'A_values':part['value'][[part['best'],part['forward']]].tolist(),
                'A_q':part['q'][[part['best'],part['forward']]].tolist(),
                'A_rewards':part['reward'][[part['best'],part['forward']]].tolist(),
                'actual_rewards':[f['reward'] for f in firsts],
                'actual_terminal':[f['done'] for f in firsts],
                'B_values':values[3].tolist(),'B_q':bq.tolist(),
                'field_values':dict(zip(['cv','positions','velocities','actual'],values.tolist())),
                'C_returns':[r['return'] for r in outcomes],'C_outcomes':outcomes,
                'progress':part['progress'][[part['best'],part['forward']]].tolist(),
                'risk':part['risk'][[part['best'],part['forward']]].tolist(),
                'human_position_cv_error_max':float(max(np.linalg.norm([a.px-b.px,a.py-b.py]) for a,b in zip(predicted_humans,firsts[0]['next_humans']))),
                'human_velocity_cv_error_max':float(max(np.linalg.norm([a.vx-b.vx,a.vy-b.vy]) for a,b in zip(predicted_humans,firsts[0]['next_humans'])))}
        repaired_trace=np.load(OUT/'repaired-traces'/f'{scene}-{case}.npz')['records']
        historical=np.load(OLD/'artifacts'/'development-traces'/'nominal'/f'{scene}-{case}.npz')['records']
        t=root['step']
        record['repaired_policy_reaches_same_root']=bool(len(repaired_trace)>t and np.array_equal(repaired_trace[:t,1:6],historical[:t,1:6]))
        rows.append(record)
        save('value-contrasts.json',rows)
        if (n+1)%10==0:
            print('contrasts',n+1,'B-rank-reversals',sum((r['B_q'][0]-r['B_q'][1])<-1e-6 for r in rows),
                  'C-forward-better',sum(r['C_returns'][1]>r['C_returns'][0]+1e-6 for r in rows),flush=True)


def upstream(variant):
    cfg,p=setup()
    roots=json.loads((FINDINGS/'counterfactual-protocol.json').read_text())['roots']
    assert len(json.loads((HERE/'value-contrasts.json').read_text()))==len(roots)
    rows=[]
    draws=range(4) if variant=='explore' else range(1)
    for n,root in enumerate(roots):
        scene=m.SCENES.index(root['scene'])
        case=root['case']
        tape=load_tape(scene,case,variant=='deadline50',cfg,p)
        env,snap=frozen_root(cfg,p,root,tape)
        part=decompose(p,env)
        actions=[(part[k],part['commands'][part[k]]) for k in ['best','forward']]
        for draw in draws:
            outcomes=[]
            for label,forced in zip(['selected','forward'],actions):
                result,trace,_=rollout(cfg,p,scene,case,tape,snap,forced,variant,draw)
                result.pop('sampled_roots')
                path=OUT/'upstream'/variant/f'{scene}-{case}-{draw}-{label}.npz'
                path.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(path,records=trace)
                outcomes.append(result)
            rows.append({'scene':root['scene'],'case':case,'step':root['step'],
                         'partition':root['partition'],'variant':variant,'draw':draw,
                         'returns':[r['return'] for r in outcomes],'outcomes':outcomes})
        save(f'upstream-{variant}.json',rows)
        if (n+1)%10==0:
            print('upstream',variant,n+1,'pairs',len(rows),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['preflight','baseline','contrasts','upstream'])
    parser.add_argument('--variant',choices=['deadline50','no-smoothing','no-filter','explore'])
    args=parser.parse_args()
    if args.stage=='upstream':
        assert args.variant
        upstream(args.variant)
    else:
        globals()[args.stage]()
