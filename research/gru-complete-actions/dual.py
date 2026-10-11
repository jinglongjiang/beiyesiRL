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


def catalogue(timing=False,shard=0,shards=1):
    assert json.loads((OUT/'verification.json').read_text())['passed']
    cfg,p=setup()
    old={(v['scene'],v['case']):v for v in json.loads((q.OLD/'value-contrasts.json').read_text())}
    reuse={(v['scene'],v['case']):v for v in json.loads((q.OLD/'upstream-deadline50.json').read_text())}
    chosen=q.roots()
    if timing:
        ids={tuple(v) for v in q.SPEC['timing_roots']}
        chosen=[v for v in chosen if (r.m.SCENES.index(v['scene']),v['case'],v['step']) in ids]
    else:
        assert 0<=shard<shards
        chosen=chosen[shard::shards]
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
        save('timing.json' if timing else f'runtime-shard-{shard}.json',runtimes)
        print('V2 CATALOGUE',q.key(root),'branches',len(indices)*2,'new',count,
              'seconds',round(runtimes[-1]['seconds'],1),'total',round(time.perf_counter()-started,1),flush=True)
    if timing:
        total=sum(v['seconds'] for v in runtimes)
        estimate=total/800*38*160
        save('budget.json',{'five_roots_two_policies_seconds':total,
             'all_38_roots_estimate_seconds':estimate,'P3_fixed_pairs_estimate_seconds':estimate/80*4,
             'fits_budget':estimate*1.3<14400,'deadline_utc':SPEC['deadline_utc']})
        assert estimate*1.3<14400, 'Reduce roots by frozen group round-robin; never drop a policy'


def prefixes():
    cfg,p=setup()
    records=[]
    for root in q.roots():
        if root['group']!='difficult':
            continue
        folder=OUT/'catalogue'/q.key(root)
        branches=[json.loads(v.read_text()) for v in folder.glob('action-*-A_deploy.json')]
        assert len(branches)>=int(np.load(folder/'inputs.npz')['safe'].sum())
        legal=[v for v in branches if v['legal_first_action']]
        if any(v['result']['success'] and not v['result']['contact'] for v in legal):
            continue
        scene,case=r.m.SCENES.index(root['scene']),root['case']
        tape=tape_for(cfg,p,root)
        _,snap=r.frozen_root(cfg,p,root,tape)
        witness=np.load(r.FINDINGS/'artifacts'/f'counterfactual-value-free-progress-{scene}-{case}.npz')['records']
        for length in q.SPEC['P1']['prefix_lengths']:
            env=r.environment(cfg,p,scene,case,tape,snap,50)
            assert float(env.time_limit)==50.
            trace=[]
            checks=[]
            for j in range(203):
                step=int(round(env.global_time/.25))
                np.testing.assert_allclose([[h.px,h.py,h.vx,h.vy,h.radius] for h in env.humans],tape[step],rtol=0,atol=1e-8)
                part=r.decompose(p,env)
                if j<length and j<len(witness):
                    row=witness[j]
                    assert int(row[0])==step
                    np.testing.assert_allclose(env.robot.get_position(),row[1:3],rtol=0,atol=1e-9)
                    distance=np.linalg.norm(np.asarray(part['commands'])-row[3:5],axis=1)
                    index=int(distance.argmin())
                    assert distance[index]<1e-9 and part['safe'][index]
                    checks.append({'step':step,'index':index,'predicted_clearance':float(part['clear'][index])})
                else:
                    index=part['best']
                action=part['commands'][index]
                before=r.m.before_geometry(env)
                pos=env.robot.get_position()
                r.m.commit_action(p,part['token'],action,index)
                _,reward,term,trunc,info=env.step(action)
                clear=r.m.geo_step(env,before)
                trace.append([step,*pos,action.vx,action.vy,index,part['best'],reward,clear,part['flag']])
                if term or trunc:
                    break
            assert term or trunc
            a=np.asarray(trace,np.float64)
            result={'root':root,'prefix':length,'forced_steps':len(checks),'legality':checks,
                    'success':int(info['event']=='reach_goal'),'collision':int(info['event']=='collision'),
                    'contact':int((a[:,8]<0).any()),'min_clearance':float(a[:,8].min()),
                    'return':float(a[:,7]@(.99**np.arange(len(a)))),
                    'duration':len(a)*.25,'steps':len(a),'outcome':info['event']}
            dest=OUT/'prefixes'/f'{q.key(root)}-{length}.npz'
            dest.parent.mkdir(exist_ok=True)
            np.savez_compressed(dest,records=a)
            records.append(result)
            save('prefixes.json',records)
            print('V2 PREFIX',q.key(root),length,result['outcome'],result['contact'],flush=True)
    save('prefixes.json',records)


def checkpoints():
    stats=json.loads((OUT/'statistics.json').read_text())
    relevant=sum(v['category'] in ['VALUE_RANKING_FAILURE_FIXED_WORLD','MULTISTEP_POLICY_TRAP_TESTED_ACTION_SET'] for v in stats['cases'])
    if relevant<2:
        save('checkpoint-pairs.json',{'executed':False,'reason':'No repeated A task loss or tested trap'})
        return
    cfg,p=setup()
    previous=json.loads((q.OLD/'training-evidence.json').read_text())
    old_roots=json.loads((r.FINDINGS/'counterfactual-protocol.json').read_text())['roots']
    old_index={(v['scene'],v['case']):i for i,v in enumerate(old_roots)}
    prior={(v['scene'],v['case']):v for v in json.loads((q.OLD/'value-contrasts.json').read_text())}
    rows=[]
    started=time.perf_counter()
    for stage,filename in [('IL','gru-il.pth')]+[(f'RL-{n}',f'gru-rl-{n}.pth') for n in [500,1000,1500,2000]]:
        weight=r.OLD/'artifacts'/filename
        p.load_state_dict(torch.load(weight,map_location=p.device))
        p.eval()
        for root in q.roots():
            scene,case=r.m.SCENES.index(root['scene']),root['case']
            row=prior[(root['scene'],case)]
            indices=[row['selected_index'],row['forward_index']]
            tape=tape_for(cfg,p,root)
            env,snap=r.frozen_root(cfg,p,root,tape)
            env.time_limit=50.
            part=r.decompose(p,env)
            for policy in SPEC['policies']:
                outcomes=[]
                for label,index in zip(['selected','forward'],indices):
                    if stage=='RL-2000':
                        source=OUT/'catalogue'/q.key(root)/f'action-{index:02d}-{policy}.json'
                        result=json.loads(source.read_text())['result']
                    else:
                        result,trace,_=continuation(cfg,p,root,tape,snap,index,part['commands'][index],policy)
                        folder=OUT/'checkpoints'/stage
                        folder.mkdir(parents=True,exist_ok=True)
                        np.savez_compressed(folder/f'{q.key(root)}-{label}-{policy}.npz',records=trace)
                    outcomes.append(result)
                values=previous['checkpoints'][stage]['values'][old_index[(root['scene'],case)]]
                rows.append({'root':root,'stage':stage,'policy':policy,'sha256':q.sha(weight),
                             'indices':indices,'actual_next_values':values,'outcomes':outcomes})
                save('checkpoint-pairs.json',{'executed':True,'rows':rows})
        print('V2 CHECKPOINT',stage,'pairs',len(rows),'seconds',round(time.perf_counter()-started,1),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['verify','timing','catalogue','prefixes','checkpoints'])
    parser.add_argument('--shard',type=int,default=0)
    parser.add_argument('--shards',type=int,default=1)
    args=parser.parse_args()
    if args.stage in ['timing','catalogue']:
        catalogue(args.stage=='timing',args.shard,args.shards)
    else:
        globals()[args.stage]()
