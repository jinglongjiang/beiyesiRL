"""Fixed-slot complementary human selection with unchanged physical execution."""
import argparse
import importlib.util
import json
from pathlib import Path
import time
import types

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('witness_parent_helpers',HERE.parent/'local-action-gap/run.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
x,h,r,m=old.x,old.h,old.r,old.m
SPEC=json.loads((HERE/'protocol.json').read_text())
OUT=HERE/'artifacts'
x.HERE,x.OUT,x.SPEC,h.OUT=HERE,OUT,SPEC,OUT
x.Controller=old.BaseController


def save(name,data): m.save_json(HERE/name,data)


def freeze():
    OUT.mkdir(exist_ok=True)
    files=[p for d in ['crowd_nav','crowd_sim'] for p in (ROOT/d).rglob('*.py')]
    files+=[HERE/'run.py',HERE/'protocol.json',HERE/'PRIOR.md',r.OLD/'parent-config.ini',r.OLD/'artifacts/nominal.pth']
    files+=[HERE.parent/name for name in ['local-action-gap/run.py','local-action-gap/protocol.json',
        'progress-contract-v2/run.py','progress-contract-v2/controller.py','progress-contract/controller.py',
        'gru-causal-repairs/run.py','gru-causal-repairs/analyze.py','gru-root-cause/run.py',
        'gru-complete-actions/run.py','recovery-duration/run.py']]
    assert old.sha(r.OLD/'artifacts/nominal.pth')==SPEC['parent_sha256']
    save('frozen-manifest.json',{'files':{str(p.relative_to(ROOT)):old.sha(p) for p in sorted(set(files))}})


def verify():
    for name,digest in json.loads((HERE/'frozen-manifest.json').read_text())['files'].items():
        assert digest and old.sha(ROOT/name)==digest,name


def greedy(weights,order,k=5):
    selected=[];covered=np.zeros(weights.shape[0])
    for _ in range(min(k,weights.shape[1])):
        gains=np.maximum(weights-covered[:,None],0).mean(0)
        gains[selected]=-np.inf
        chosen=max((i for i in order if i not in selected),key=lambda i:gains[i])
        selected.append(chosen);covered=np.maximum(covered,weights[:,chosen])
    return selected


def geometry(robot,humans,commands):
    relative=np.asarray([[v.px-robot.px,v.py-robot.py] for v in humans],np.float64)
    velocity=np.asarray([[v.vx,v.vy] for v in humans],np.float64)
    radii=robot.radius+np.asarray([v.radius for v in humans],np.float64)
    dv=velocity[None]-commands[:,None]
    aa=(dv*dv).sum(-1);bb=2*(relative[None]*dv).sum(-1)
    cc=(relative*relative).sum(-1)-radii*radii
    closest=np.clip(-bb/(2*np.maximum(aa,1e-15)),0,4.)
    clearance=np.linalg.norm(relative[None]+closest[...,None]*dv,axis=-1)-radii
    disc=bb*bb-4*aa*cc[None]
    root=(-bb-np.sqrt(np.maximum(disc,0)))/(2*np.maximum(aa,1e-15))
    ttc=np.where((aa>1e-15)&(disc>=0)&(root>=0)&(root<=4),root,np.inf)
    ttc[:,cc<=0]=0.
    return clearance,ttc


def install(p,arm):
    original=p._build_joint_state_34
    diagnostics={'calls':0,'eligible':0,'cover_scalar_different':0,'selected_nearest_different':0,'current_steps':[]}
    def build(self,robot,humans):
        order=sorted(range(len(humans)),key=lambda i:(humans[i].px-robot.px)**2+(humans[i].py-robot.py)**2)
        diagnostics['calls']+=1
        if len(humans)<=5:
            return original(robot,humans)
        raw=np.asarray(self.action_space,np.float64)
        commands=.3*np.asarray([robot.vx,robot.vy])+.7*raw
        current=getattr(self,'env',None)
        is_current=current is not None and np.linalg.norm(
            np.asarray([robot.px,robot.py])-np.asarray(current.robot.get_position()))<1e-12
        initial_current=self._last_action is None and is_current
        if initial_current: commands=raw
        if is_current:
            np.testing.assert_allclose(commands,np.asarray(self.executable_candidates()),rtol=0,atol=1e-15)
        clearance,ttc=geometry(robot,humans,commands)
        weights=.5*np.minimum(1.,.25/np.maximum(ttc,1e-12))+.5*np.clip((.2-clearance)/.2,0,1)
        _,current_ttc=geometry(robot,humans,np.asarray([[robot.vx,robot.vy]]))
        selections={'nearest':order[:5],
            'ttc':sorted(order,key=lambda i:current_ttc[0,i])[:5],
            'scalar':sorted(order,key=lambda i:-weights[:,i].mean())[:5],
            'cover':greedy(weights,order)}
        selected=selections[arm]
        if is_current:
            diagnostics['current_steps'].append({'time':float(current.global_time),'human_indices':selections})
        diagnostics['eligible']+=1
        diagnostics['cover_scalar_different']+=int(set(selections['cover'])!=set(selections['scalar']))
        diagnostics['selected_nearest_different']+=int(set(selected)!=set(order[:5]))
        result=original(robot,[humans[i] for i in selected])
        if arm=='nearest': np.testing.assert_array_equal(result,original(robot,humans))
        return result
    p._build_joint_state_34=types.MethodType(build,p)
    return original,diagnostics


def episode(cfg,p,scene,case,arm,output):
    original,diagnostics=install(p,arm)
    try:
        row=x.episode(cfg,p,scene,case,x.humans(cfg,p,scene,case),'hold8',output)
        row.update(arm=arm,selection=diagnostics)
        m.save_json(OUT/(output+'.json'),row)
        return row
    finally: p._build_joint_state_34=original


def controls():
    verify()
    weights=np.zeros((2,6));weights[0,:5]=1.;weights[1,5]=.9
    selected=greedy(weights,list(range(6)))
    assert 5 in selected and np.max(weights[:,selected],axis=1).mean()>np.max(weights[:,:5],axis=1).mean()
    cfg,p=h.setup();h.weight(p,'nominal');rows=[]
    for scene in range(6):
        # A complete nearest-wrapper run asserts original builder parity on every token.
        nearest=episode(cfg,p,scene,121,'nearest',f'controls/nearest-{scene}-121')
        row=episode(cfg,p,scene,121,'cover',f'controls/cover-{scene}-121')
        if scene==0:
            for key in ['records','candidates']:
                np.testing.assert_array_equal(np.load(OUT/f'controls/nearest-{scene}-121.npz')[key],np.load(OUT/f'controls/cover-{scene}-121.npz')[key])
        if scene==1:
            episode(cfg,p,scene,121,'cover','controls/repeat-cover-1-121')
            for key in ['records','candidates']:
                np.testing.assert_array_equal(np.load(OUT/'controls/cover-1-121.npz')[key],np.load(OUT/'controls/repeat-cover-1-121.npz')[key])
        rows.append(row);print('CONTROL',scene,row['outcome'],row['selection'],flush=True)
    eligible=sum(v['selection']['eligible'] for v in rows)
    different=sum(v['selection']['cover_scalar_different'] for v in rows)
    split_scenes=sum(v['selection']['cover_scalar_different']>0 for v in rows)
    activated=eligible>0 and different/eligible>=.01 and split_scenes>=2
    verify();save('controls.json',{'passed':True,'grid':80,'time_limit':50,'exact_duplicate':True,
        'nearest_token_parity':True,'five_human_parity':True,'synthetic_cover':True,
        'eligible_calls':eligible,'cover_scalar_different':different,'split_scenes':split_scenes,
        'activation_passed':activated,'smoke':rows})


def evaluate(arm,cohort):
    verify();control=json.loads((HERE/'controls.json').read_text())
    assert control['passed'] and control['activation_passed']
    if cohort=='confirmation': assert json.loads((HERE/'development-summary.json').read_text())['gate']['passed']
    cfg,p=h.setup();h.weight(p,'nominal');rows=[];start=time.perf_counter()
    for scene in SPEC[cohort]['scenes']:
        for case in range(SPEC[cohort]['case_first'],SPEC[cohort]['case_last']+1):
            if cohort=='development' and arm in ['nominal','hold8']:
                source=HERE.parent/f'progress-contract-v2/development-{arm}.json'
                row=dict(next(v for v in json.loads(source.read_text())['results'] if v['scene']==m.SCENES[scene] and v['case']==case),reused=True,source_sha256=old.sha(source))
            elif arm in ['nominal','hold8']:
                tape=x.humans(cfg,p,scene,case)
                row=h.run_episode(cfg,p,scene,case,tape,'nominal',output=f'{cohort}/{arm}/{scene}-{case}') if arm=='nominal' else x.episode(cfg,p,scene,case,tape,'hold8',f'{cohort}/{arm}/{scene}-{case}')
            else: row=episode(cfg,p,scene,case,arm,f'{cohort}/{arm}/{scene}-{case}')
            rows.append(row);save(f'{cohort}-{arm}.json',{'results':rows,'seconds':time.perf_counter()-start})
            print(cohort,arm,len(rows),row['outcome'],flush=True)
    verify()


def summarize(cohort):
    rows={a:json.loads((HERE/f'{cohort}-{a}.json').read_text())['results'] for a in SPEC['arms']}
    assert all(len(v)==SPEC[cohort]['n'] for v in rows.values())
    overall={a:x.stats.aggregate(v) for a,v in rows.items()}
    pairs={a:x.stats.pairs(rows['cover'],rows[a]) for a in SPEC['arms'] if a!='cover'}
    ratios={}
    for arm in pairs:
        base={(v['scene'],v['case']):v for v in rows[arm]}
        both=[v for v in rows['cover'] if v['success'] and base[(v['scene'],v['case'])]['success']]
        ratios[arm]=float(np.mean([v['duration'] for v in both])/np.mean([base[(v['scene'],v['case'])]['duration'] for v in both]))
    own=overall['cover']
    changed=[]
    for v in rows['cover']:
        stem=f'{m.SCENES.index(v["scene"])}-{v["case"]}.npz'
        base_path=(HERE.parent/'progress-contract-v2/artifacts/development/hold8'/stem
                   if cohort=='development' else OUT/'confirmation/hold8'/stem)
        own_trace=np.load(OUT/cohort/'cover'/stem)['records']
        base_trace=np.load(base_path)['records']
        n=min(len(own_trace),len(base_trace))
        different=np.flatnonzero((own_trace[:n,3:5]!=base_trace[:n,3:5]).any(1))
        if len(different): changed.append({'scene':v['scene'],'case':v['case'],'first_executed_action_difference_step':int(different[0])})
    checks={'success_nonregression':all(own['success']>=overall[a]['success'] for a in pairs),
        'safety':all(own['contact']<=overall[a]['contact'] and own['collision']<=overall[a]['collision'] for a in pairs),
        'independent_task_gain':all(own['success']>=overall[a]['success']+2 for a in pairs) or all(v<=.95 for v in ratios.values()),
        'activation':len(changed)>0,'compute':own['decision_ms_episode_mean']<250}
    if cohort=='confirmation':
        ids=sorted({v['case'] for v in rows['cover']})
        draws=np.random.default_rng(121012).integers(len(ids),size=(4000,len(ids)))
        intervals={}
        for arm in pairs:
            base={(v['scene'],v['case']):v for v in rows[arm]}
            counts=np.zeros(len(ids));sums=np.zeros(len(ids))
            for v in rows['cover']:
                other=base[(v['scene'],v['case'])]
                if v['success'] and other['success']:
                    j=ids.index(v['case']);counts[j]+=1;sums[j]+=v['duration']-other['duration']
            denominator=counts[draws].sum(1)
            assert (denominator>0).all()
            intervals[arm]=np.quantile(sums[draws].sum(1)/denominator,[.025,.975]).tolist()
        checks['fresh_case_intervals']=all(pairs[a]['delta_SR_case_bootstrap_95'][0]>0 or intervals[a][1]<0 for a in pairs)
        save('confirmation-time-intervals.json',intervals)
    save(f'{cohort}-summary.json',{'overall':overall,'pairs':pairs,'paired_success_time_ratios':ratios,
        'changed_executed_trajectories':changed,'gate':{'checks':checks,'passed':all(checks.values())},
        'verdict':'METHOD_SIGNAL_FOUND' if all(checks.values()) else 'VERSION_NEGATIVE',
        'novelty':'UNVERIFIED; no paper PASS from a generic greedy coverage selector alone.'})
    print('SUMMARY',checks,ratios,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['freeze','verify','controls','evaluate','summarize'])
    parser.add_argument('--arm',choices=SPEC['arms']);parser.add_argument('--cohort',default='development',choices=['development','confirmation'])
    args=parser.parse_args()
    if args.command=='evaluate': evaluate(args.arm,args.cohort)
    elif args.command=='summarize': summarize(args.cohort)
    else: globals()[args.command]()
