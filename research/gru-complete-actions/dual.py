"""Explicit 50s two-continuation diagnostic; original 25s outputs stay untouched."""
import argparse
import copy
import json
import time

import numpy as np
import torch

import run as q

HERE=q.HERE
SPEC=json.loads((HERE/'protocol-amendment-v2.json').read_text())
OUT=q.OUT/'v2-50s'
r=q.r


def save(path,value):
    q.save(str((OUT/path).relative_to(HERE)),value)


def setup():
    cfg,p=q.setup()
    OUT.mkdir(exist_ok=True)
    return cfg,p


def tape_for(cfg,p,root):
    return r.load_tape(r.m.SCENES.index(root['scene']),root['case'],True,cfg,p)


def verify():
    cfg,p=setup()
    rows=[]
    for root in q.roots():
        scene,case=r.m.SCENES.index(root['scene']),root['case']
        tape=tape_for(cfg,p,root)
        assert len(tape)==203
        env=r.m.make_env(cfg,p,scene)
        env.time_limit=50.
        assert float(env.time_limit)==50.
        env.reset(options={'test_case':case})
        assert not env.robot.visible
        arrived=[]
        error=[]
        goals=[[h.gx,h.gy] for h in env.humans]
        for step in range(len(tape)):
            state=np.array([[h.px,h.py,h.vx,h.vy,h.radius] for h in env.humans])
            err=float(np.abs(state-tape[step]).max())
            np.testing.assert_allclose(state,tape[step],rtol=0,atol=1e-9)
            assert [[h.gx,h.gy] for h in env.humans]==goals
            error.append(err)
            arrived.append([h.reached_destination() for h in env.humans])
            if step+1<len(tape):
                env.step(r.m.ActionXY(0.,0.))
        arr=np.asarray(arrived,bool)
        transitions=(arr[1:]!=arr[:-1])
        horizon_transitions=transitions[root['step']:201]
        root_path=OUT/'verified-tapes'/f'{q.key(root)}.npz'
        root_path.parent.mkdir(exist_ok=True)
        np.savez_compressed(root_path,humans=tape,arrived=arr,per_step_max_error=error)
        rows.append({**root,'frames':len(tape),'max_abs_error':max(error),
                     'arrival_transition_count_in_root_horizon':int(horizon_transitions.sum()),
                     'has_arrival_transition':bool(horizon_transitions.any()),
                     'first_arrival_transition_step':int(np.where(horizon_transitions.any(1))[0][0]+root['step']+1) if horizon_transitions.any() else None,
                     'goal_reset_count':0,'tape_sha256':q.sha(root_path)})
    strata={}
    for label,flag in [('arrival',True),('no-arrival',False)]:
        subset=[v for v in rows if v['has_arrival_transition']==flag]
        strata[label]={'roots':len(subset),'frames':sum(v['frames'] for v in subset),
                       'max_abs_error':max((v['max_abs_error'] for v in subset),default=None)}
    assert strata['arrival']['roots']>0, 'Arrival-event coverage required'
    save('verification.json',{'passed':True,'time_limit':50,'roots':rows,'arrival_strata':strata})
    print('V2 VERIFIED',json.dumps(strata),flush=True)


def continuation(cfg,p,root,tape,snap,index,command,policy):
    previous=(p.test_min_clearance,p.test_risk_lambda,p.test_action_smoothing)
    setting=SPEC['policies'][policy]
    p.test_min_clearance=setting['margin']
    p.test_risk_lambda=setting['risk_lambda']
    p.test_action_smoothing=setting['smoothing']
    try:
        result,trace,first=r.rollout(cfg,p,r.m.SCENES.index(root['scene']),root['case'],
                                   tape,snap,(index,command),'deadline50')
        result.pop('sampled_roots',None)
        result['continuation_policy']=policy
        assert trace[-1,0]<=200
        return result,trace,first
    finally:
        p.test_min_clearance,p.test_risk_lambda,p.test_action_smoothing=previous


def actual_windows(cfg,p,root,tape,snap,part):
    prefix=list(snap['history'])+[part['token']]
    windows=[]
    first=[]
    for index in range(80):
        env=r.environment(cfg,p,r.m.SCENES.index(root['scene']),root['case'],tape,snap,50)
        assert float(env.time_limit)==50.
        action=part['commands'][index]
        r.m.commit_action(p,part['token'],action,index)
        _,reward,term,trunc,_=env.step(action)
        windows.append(r.m.window(prefix+[r.m.token(p,r.m.state(env))]))
        first.append([reward,int(term or trunc)])
    return np.asarray(windows,np.float32),np.asarray(first,np.float64)


def catalogue(timing=False):
    assert json.loads((OUT/'verification.json').read_text())['passed']
    cfg,p=setup()
    old={(v['scene'],v['case']):v for v in json.loads((q.OLD/'value-contrasts.json').read_text())}
    reuse={(v['scene'],v['case']):v for v in json.loads((q.OLD/'upstream-deadline50.json').read_text())}
    chosen=q.roots()
    if timing:
        ids={tuple(v) for v in q.SPEC['timing_roots']}
        chosen=[v for v in chosen if (r.m.SCENES.index(v['scene']),v['case'],v['step']) in ids]
    runtimes=[]
    started=time.perf_counter()
    for root in chosen:
        began=time.perf_counter()
        scene,case=r.m.SCENES.index(root['scene']),root['case']
        tape=tape_for(cfg,p,root)
        env,snap=r.frozen_root(cfg,p,root,tape)
        env.time_limit=50.
        assert float(env.time_limit)==50.
        part=r.decompose(p,env)
        assert part['safe'].any()
        indices=list(range(80)) if timing else np.flatnonzero(part['safe']).tolist()
        folder=OUT/'catalogue'/q.key(root)
        folder.mkdir(parents=True,exist_ok=True)
        windows,first=actual_windows(cfg,p,root,tape,snap,part)
        with torch.no_grad():
            values=p.forward_candidate_values(torch.as_tensor(windows,device=p.device)).cpu().numpy()
        bq=first[:,0]+.99*(1-first[:,1])*values
        np.savez_compressed(folder/'inputs.npz',windows=windows,first=first,A_q=part['q'],
                            B_q=bq,values=part['value'],progress=part['progress'],safe=part['safe'],
                            predicted_clearance=part['clear'])
        count=0
        for index in indices:
            for policy in SPEC['policies']:
                dest=folder/f'action-{index:02d}-{policy}.json'
                if dest.exists():
                    continue
                row=old[(root['scene'],case)]
                label=('selected' if index==row['selected_index'] else
                       'forward' if index==row['forward_index'] else None)
                if policy=='A_deploy' and label:
                    k=0 if label=='selected' else 1
                    result=copy.deepcopy(reuse[(root['scene'],case)]['outcomes'][k])
                    result['continuation_policy']=policy
                    trace=np.load(r.OUT/'upstream'/'deadline50'/f'{scene}-{case}-0-{label}.npz')['records']
                    reused=True
                else:
                    result,trace,_=continuation(cfg,p,root,tape,snap,index,part['commands'][index],policy)
                    reused=False
                assert int(trace[0,0])==root['step']
                np.testing.assert_allclose(trace[0,3:5],part['commands'][index],rtol=0,atol=1e-12)
                np.testing.assert_allclose(trace[0,7],first[index,0],rtol=0,atol=1e-9)
                assert result['duration']<=50.25-root['step']*.25+1e-9
                np.savez_compressed(folder/f'action-{index:02d}-{policy}.npz',records=trace)
                save(str(dest.relative_to(OUT)),{'root':root,'index':index,'policy':policy,
                    'legal_first_action':bool(part['safe'][index]),'reused':reused,
                    'first_command':list(part['commands'][index]),'Q_pred':float(part['q'][index]),
                    'Q_actual_next':float(bq[index]),'progress':float(part['progress'][index]),'result':result})
                count+=1
        runtimes.append({**root,'actions':len(indices),'branches':len(indices)*2,
                         'new_branches':count,'seconds':time.perf_counter()-began})
        save('timing.json' if timing else 'runtime.json',runtimes)
        print('V2 CATALOGUE',q.key(root),'branches',len(indices)*2,'new',count,
              'seconds',round(runtimes[-1]['seconds'],1),'total',round(time.perf_counter()-started,1),flush=True)
    if timing:
        total=sum(v['seconds'] for v in runtimes)
        estimate=total/800*38*160
        save('budget.json',{'five_roots_two_policies_seconds':total,
             'all_38_roots_estimate_seconds':estimate,'P3_fixed_pairs_estimate_seconds':estimate/80*4,
             'fits_budget':estimate*1.3<14400,'deadline_utc':SPEC['deadline_utc']})
        assert estimate*1.3<14400, 'Reduce roots by frozen group round-robin; never drop a policy'


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['verify','timing','catalogue'])
    args=parser.parse_args()
    if args.stage in ['timing','catalogue']:
        catalogue(args.stage=='timing')
    else:
        verify()
