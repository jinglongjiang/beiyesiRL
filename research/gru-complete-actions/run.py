"""Complete-action fixed-world diagnosis, without modifying the navigation policy."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SPEC = json.loads((HERE / 'protocol.json').read_text())
OLD = HERE.parent / 'gru-root-cause'
spec = importlib.util.spec_from_file_location('previous_root_runner', OLD / 'run.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
OUT = HERE / 'artifacts'


def save(path, value):
    r.m.save_json(HERE / path, value)


def roots():
    result = []
    for group in ['difficult', 'rescues', 'damages', 'successful_controls']:
        for scene, case, step in SPEC[group]:
            result.append({'scene': r.m.SCENES[scene], 'case': case, 'step': step,
                           'group': group, 'partition': 'fit' if case < 120 else 'held-out'})
    assert len(result) == 38
    assert len({(v['scene'], v['case']) for v in result}) == 38
    return result


def key(root):
    return f"{r.m.SCENES.index(root['scene'])}-{root['case']}"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def setup():
    cfg, p = r.setup()
    assert sha(r.OLD / 'artifacts' / 'nominal.pth') == SPEC['nominal_sha256']
    OUT.mkdir(exist_ok=True)
    return cfg, p


def verify():
    cfg, p = setup()
    records = []
    for root in roots():
        scene, case = r.m.SCENES.index(root['scene']), root['case']
        tape = r.load_tape(scene, case)
        env = r.m.make_env(cfg, p, scene)
        env.reset(options={'test_case': case})
        assert not env.robot.visible
        max_error = 0.
        arrived_steps = 0
        goals = [[h.gx, h.gy] for h in env.humans]
        for step in range(len(tape)):
            observed = np.array([[h.px,h.py,h.vx,h.vy,h.radius] for h in env.humans])
            max_error = max(max_error, float(np.abs(observed-tape[step]).max()))
            np.testing.assert_allclose(observed, tape[step], rtol=0, atol=1e-9)
            assert [[h.gx,h.gy] for h in env.humans] == goals
            arrived_steps += sum(h.reached_destination() for h in env.humans)
            if step+1 < len(tape):
                # Native step runs all human policies before any positions update.
                # Robot terminal flags are ignored only in this full-tape check.
                env.step(r.m.ActionXY(0., 0.))
        env, snap = r.frozen_root(cfg, p, root, tape)
        part = r.decompose(p, env)
        assert part['flag'] and part['safe'][part['best']] and part['safe'][part['forward']]
        hist = np.stack(list(snap['history'])+[part['token']])
        path = OUT / 'roots' / f'{key(root)}.npz'
        path.parent.mkdir(exist_ok=True)
        robot = snap['robot']
        np.savez_compressed(path, history=hist,
            robot=np.array([robot.px,robot.py,robot.vx,robot.vy,robot.radius,
                            robot.gx,robot.gy,robot.v_pref,robot.theta]),
            last_action=np.asarray(snap['last_action']), time=snap['time'],
            humans=tape, commands=np.asarray(part['commands']), safe=part['safe'],
            predicted_q=part['q'], predicted_value=part['value'], progress=part['progress'])
        records.append({**root,'frames_verified':len(tape),'max_abs_error':max_error,
                        'arrived_human_frames':int(arrived_steps),'goals_unchanged':True,
                        'safe_count':int(part['safe'].sum()),'root_sha256':sha(path)})
    save('verification.json', {'passed':True,'roots':records,'actions':80,
                             'nominal_sha256':SPEC['nominal_sha256']})
    print('VERIFIED full tapes', len(records), 'frames', sum(v['frames_verified'] for v in records),flush=True)


def actual_windows(cfg, p, root, tape, snap, part, indices):
    prefix = list(snap['history'])+[part['token']]
    result = []
    first = []
    for index in indices:
        env = r.environment(cfg,p,r.m.SCENES.index(root['scene']),root['case'],tape,snap)
        command = part['commands'][index]
        r.m.commit_action(p,part['token'],command,index)
        _,reward,term,trunc,_ = env.step(command)
        result.append(r.m.window(prefix+[r.m.token(p,r.m.state(env))]))
        first.append([reward,int(term or trunc)])
    return np.asarray(result,np.float32), np.asarray(first,np.float64)


def catalogue(timing=False):
    assert json.loads((HERE/'verification.json').read_text())['passed']
    cfg,p = setup()
    prior = {(v['scene'],v['case']):v for v in json.loads((OLD/'value-contrasts.json').read_text())}
    timings = []
    selected = roots()
    if timing:
        ids = {tuple(v) for v in SPEC['timing_roots']}
        selected = [v for v in selected if (r.m.SCENES.index(v['scene']),v['case'],v['step']) in ids]
    global_start = time.perf_counter()
    for root in selected:
        started = time.perf_counter()
        scene,case = r.m.SCENES.index(root['scene']),root['case']
        tape = r.load_tape(scene,case)
        env,snap = r.frozen_root(cfg,p,root,tape)
        part = r.decompose(p,env)
        indices = list(range(80)) if timing else np.flatnonzero(part['safe']).tolist()
        assert part['safe'].any(), 'Native empty-safe fallback needs separate analysis'
        path = OUT/'catalogue'/key(root)
        path.mkdir(parents=True,exist_ok=True)
        windows,first = actual_windows(cfg,p,root,tape,snap,part,list(range(80)))
        with torch.no_grad():
            bvalue = p.forward_candidate_values(torch.as_tensor(windows,device=p.device)).cpu().numpy()
        bq = first[:,0]+.99*(1-first[:,1])*bvalue
        np.savez_compressed(path/'inputs.npz',windows=windows,first=first,A_q=part['q'],B_q=bq,
                            progress=part['progress'],safe=part['safe'],predicted_clearance=part['clear'])
        completed = 0
        for index in indices:
            dest = path/f'action-{index:02d}.json'
            if dest.exists():
                continue
            old = prior[(root['scene'],case)]
            label = ('selected' if index==old['selected_index'] else
                     'forward' if index==old['forward_index'] else None)
            if label:
                k = 0 if label=='selected' else 1
                result = copy.deepcopy(old['C_outcomes'][k])
                result.pop('sampled_roots',None)
                trace = np.load(r.OUT/'contrasts'/f'{scene}-{case}-{label}.npz')['records']
                np.testing.assert_allclose(trace[0,3:5],part['commands'][index],rtol=0,atol=1e-12)
                assert int(trace[0,0])==root['step']
                reused = True
            else:
                result,trace,_ = r.rollout(cfg,p,scene,case,tape,snap,
                                         (index,part['commands'][index]))
                result.pop('sampled_roots',None)
                reused = False
            np.savez_compressed(path/f'action-{index:02d}.npz',records=trace)
            save(str(dest.relative_to(HERE)),{'root':root,'index':index,
                'legal':bool(part['safe'][index]),'reused':reused,
                'executed_first':list(part['commands'][index]),
                'A_q':float(part['q'][index]),'B_q':float(bq[index]),
                'progress':float(part['progress'][index]),'result':result})
            completed += 1
        elapsed = time.perf_counter()-started
        timings.append({**root,'requested_actions':len(indices),'newly_completed':completed,'seconds':elapsed})
        print('CATALOGUE',key(root),root['group'],'actions',len(indices),'new',completed,
              'seconds',round(elapsed,2),'total',round(time.perf_counter()-global_start,1),flush=True)
        save('timing.json' if timing else 'catalogue-runtime.json',timings)
    if timing:
        secs = sum(v['seconds'] for v in timings)
        estimate = secs/400*38*80
        save('budget.json',{'five_root_seconds':secs,'P1_estimate_seconds':estimate,
                            'P3_estimate_seconds':secs/400*38*2*4,
                            'wall_budget_seconds':21600,'within_budget':estimate*1.5<18000})
        assert estimate*1.5<18000, 'Preserve P1 and reduce P3 before proceeding'


def prefix_rollout(cfg,p,root,tape,snap,witness,length):
    scene,case = r.m.SCENES.index(root['scene']),root['case']
    env = r.environment(cfg,p,scene,case,tape,snap)
    rows = []
    validity = []
    for j in range(103):
        step = int(round(env.global_time/.25))
        np.testing.assert_allclose([[h.px,h.py,h.vx,h.vy,h.radius] for h in env.humans],tape[step],rtol=0,atol=1e-8)
        part = r.decompose(p,env)
        if j < length and j < len(witness):
            w = witness[j]
            assert int(w[0])==step
            np.testing.assert_allclose(env.robot.get_position(),w[1:3],rtol=0,atol=1e-9)
            distances = np.linalg.norm(np.asarray(part['commands'])-w[3:5],axis=1)
            idx = int(distances.argmin())
            assert distances[idx]<1e-9, 'Witness command is not a native smoothed candidate'
            allowed = bool(part['safe'][idx])
            validity.append({'step':step,'index':idx,'safe':allowed,'predicted_clearance':float(part['clear'][idx])})
            assert allowed, 'Saved witness violated the original safety mask'
        else:
            idx = part['best']
        action = part['commands'][idx]
        before = r.m.before_geometry(env)
        pos = env.robot.get_position()
        r.m.commit_action(p,part['token'],action,idx)
        _,reward,term,trunc,info = env.step(action)
        clear = r.m.geo_step(env,before)
        rows.append([step,*pos,action.vx,action.vy,idx,part['best'],reward,clear,part['flag']])
        if term or trunc:
            break
    assert term or trunc
    a = np.asarray(rows,np.float64)
    return {'root':root,'requested_prefix':length,'forced_steps':len(validity),
            'outcome':info['event'],'success':int(info['event']=='reach_goal'),
            'collision':int(info['event']=='collision'),'contact':int((a[:,8]<0).any()),
            'min_clearance':float(a[:,8].min()),'return':float(a[:,7]@(.99**np.arange(len(a)))),
            'steps':len(a),'duration':len(a)*.25,'legality':validity},a


def prefixes():
    cfg,p = setup()
    results = []
    for root in roots():
        if root['group']!='difficult':
            continue
        folder=OUT/'catalogue'/key(root)
        actions=[json.loads(v.read_text()) for v in folder.glob('action-*.json')]
        assert len(actions)>=int(np.load(folder/'inputs.npz')['safe'].sum())
        legal=[v for v in actions if v['legal']]
        if any(v['result']['success'] and not v['result']['contact'] for v in legal):
            continue
        scene,case=r.m.SCENES.index(root['scene']),root['case']
        tape=r.load_tape(scene,case)
        _,snap=r.frozen_root(cfg,p,root,tape)
        witness=np.load(r.FINDINGS/'artifacts'/f'counterfactual-value-free-progress-{scene}-{case}.npz')['records']
        for length in SPEC['P1']['prefix_lengths']:
            result,trace=prefix_rollout(cfg,p,root,tape,snap,witness,length)
            folder=OUT/'prefixes'
            folder.mkdir(exist_ok=True)
            np.savez_compressed(folder/f'{key(root)}-{length}.npz',records=trace)
            results.append(result)
            save('prefixes.json',results)
            print('PREFIX',key(root),length,result['outcome'],'contact',result['contact'],flush=True)
    save('prefixes.json',results)


def checkpoint_pairs():
    cfg,p=setup()
    prior={(v['scene'],v['case']):v for v in json.loads((OLD/'value-contrasts.json').read_text())}
    previous=json.loads((OLD/'training-evidence.json').read_text())
    all_roots=json.loads((r.FINDINGS/'counterfactual-protocol.json').read_text())['roots']
    old_index={(v['scene'],v['case']):i for i,v in enumerate(all_roots)}
    results=[]
    started=time.perf_counter()
    for stage,filename in [('IL','gru-il.pth')]+[(f'RL-{n}',f'gru-rl-{n}.pth') for n in [500,1000,1500,2000]]:
        weight=r.OLD/'artifacts'/filename
        p.load_state_dict(torch.load(weight,map_location=p.device))
        p.eval()
        for root in roots():
            scene,case=r.m.SCENES.index(root['scene']),root['case']
            old=prior[(root['scene'],case)]
            indices=[old['selected_index'],old['forward_index']]
            tape=r.load_tape(scene,case)
            env,snap=r.frozen_root(cfg,p,root,tape)
            part=r.decompose(p,env)
            assert all(part['safe'][i] for i in indices)
            vals=previous['checkpoints'][stage]['values'][old_index[(root['scene'],case)]]
            outcomes=[]
            for label,idx in zip(['selected','forward'],indices):
                if stage=='RL-2000':
                    outcome=copy.deepcopy(old['C_outcomes'][0 if label=='selected' else 1])
                else:
                    outcome,trace,_=r.rollout(cfg,p,scene,case,tape,snap,(idx,part['commands'][idx]))
                    folder=OUT/'checkpoints'/stage
                    folder.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(folder/f'{key(root)}-{label}.npz',records=trace)
                outcome.pop('sampled_roots',None)
                outcomes.append(outcome)
            results.append({'root':root,'stage':stage,'sha256':sha(weight),'indices':indices,
                            'B_values':vals,'outcomes':outcomes})
            save('checkpoint-pairs.json',results)
        print('CHECKPOINT',stage,'pairs',len(results),'seconds',round(time.perf_counter()-started,2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['verify','timing','catalogue','prefixes','checkpoint-pairs'])
    args=parser.parse_args()
    if args.stage in ['timing','catalogue']:
        catalogue(args.stage=='timing')
    else:
        globals()[args.stage.replace('-','_')]()
