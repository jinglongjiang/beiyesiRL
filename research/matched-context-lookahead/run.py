"""Matched current-context offsets for candidate-dependent truncated lookahead."""
import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('matched_context_helpers',HERE.parent/'local-action-gap/run.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
x,h,r,m=old.x,old.h,old.r,old.m
SPEC=json.loads((HERE/'protocol.json').read_text())
OUT=HERE/'artifacts'
x.HERE,x.OUT,x.SPEC,h.OUT=HERE,OUT,SPEC,OUT
x.Controller=old.BaseController
original_decompose=r.decompose


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
    import pytorch_mppi.mppi as upstream
    save('frozen-manifest.json',{'files':{str(p.relative_to(ROOT)):old.sha(p) for p in sorted(set(files))},
        'upstream_mppi_sha256':old.sha(Path(upstream.__file__))})


def verify():
    manifest=json.loads((HERE/'frozen-manifest.json').read_text())
    for name,digest in manifest['files'].items():
        assert digest and old.sha(ROOT/name)==digest,name
    import pytorch_mppi.mppi as upstream
    assert old.sha(Path(upstream.__file__))==manifest['upstream_mppi_sha256']


def nearest(robot,humans):
    return sorted(range(len(humans)),key=lambda i:(humans[i].px-robot.px)**2+(humans[i].py-robot.py)**2)[:5]


def token(p,robot,humans):
    from crowd_nav.contracts import _batch_joint34_to_tokens_vectorized
    return _batch_joint34_to_tokens_vectorized(p._build_joint_state_34(robot,humans).reshape(1,-1))[0]


def current_windows(p,tokens):
    prefix=list(p._history)[-(p.seq_len-1):]
    result=np.zeros((len(tokens),p.seq_len,*tokens[0].shape),np.float32)
    if prefix: result[:,-len(prefix)-1:-1]=np.asarray(prefix)[None]
    result[:,-1]=tokens
    return torch.from_numpy(result).to(p.device)


def decompose(p,env,arm,diagnostics):
    part=original_decompose(p,env)
    state=m.state(env);robot=state.self_state;humans=state.human_states
    root_ids=nearest(robot,humans)
    propagated=[p.propagate(v,m.ActionXY(v.vx,v.vy)) for v in humans]
    next_robots=[p.propagate(robot,a) for a in part['commands']]
    subsets=[nearest(v,propagated) for v in next_robots]
    changed=np.asarray([set(v)!=set(root_ids) for v in subsets])
    diagnostics['decisions']+=1
    diagnostics['changed_support_decisions']+=int(changed.any())
    if not changed.any():
        diagnostics['steps'].append({'step':round(env.global_time/.25),'changed_support_actions':0,'ranking_changed':False,'correction_max':0.})
        return part
    fixed_windows=part['windows'].clone()
    fixed_tokens=[token(p,v,[propagated[i] for i in root_ids]) for v in next_robots]
    fixed_windows[:,-1]=torch.as_tensor(np.asarray(fixed_tokens),device=p.device)
    # Current-time baselines preserve the actual history; no repeated/fake time step.
    baseline_tokens=[token(p,robot,[humans[i] for i in ids]) for ids in subsets]+[part['token']]
    with torch.no_grad():
        fixed_values=p.forward_candidate_values(fixed_windows)
        baselines=p.forward_candidate_values(current_windows(p,baseline_tokens))
        correction=(baselines[:-1]-baselines[-1]).cpu().numpy()
        correction[~changed]=0.
        matched_values=torch.as_tensor(part['value']-correction,device=p.device)
        values={'reference':torch.as_tensor(part['value'],device=p.device),'fixed':fixed_values,'matched':matched_values}[arm]
        terminal,_=p.candidate_terminal_outcomes(state,part['commands'],env)
        q=p.candidate_action_values(state,part['commands'],torch.tensor(part['raw_reward'].tolist(),device=p.device),values,env)
        scores=q-torch.as_tensor(part['risk'],device=p.device)
        if part['safe'].any(): scores=scores.masked_fill(torch.as_tensor(~part['safe'],device=p.device),-1e9)
        scores[0]-=.001
    np.testing.assert_array_equal(q.cpu().numpy()[terminal],part['q'][terminal])
    old_best=part['best'];best=int(scores.argmax())
    if arm=='reference':
        np.testing.assert_array_equal(q.cpu().numpy(),part['q'])
        np.testing.assert_array_equal(scores.cpu().numpy(),part['scores'])
        assert best==old_best
    diagnostics['ranking_changes']+=int(best!=old_best)
    diagnostics['steps'].append({'step':round(env.global_time/.25),'changed_support_actions':int(changed.sum()),
        'ranking_changed':best!=old_best,'before':old_best,'after':best,'root_ids':root_ids,
        'selected_ids':subsets[best],'correction_max':float(np.abs(correction).max()),
        'current_context_offset':correction.tolist()})
    part.update(value=values.cpu().numpy(),q=q.cpu().numpy(),scores=scores.cpu().numpy(),best=best)
    part['flag']=bool(part['progress'][best]<=0 and part['progress'][part['forward']]>=.02
                      and part['reward'][part['forward']]>=part['reward'][best]-1e-8)
    return part


def episode(cfg,p,scene,case,arm,output):
    diagnostics={'decisions':0,'changed_support_decisions':0,'ranking_changes':0,'steps':[]}
    r.decompose=lambda policy,env:decompose(policy,env,arm,diagnostics)
    try:
        row=x.episode(cfg,p,scene,case,x.humans(cfg,p,scene,case),'hold8',output)
        row.update(arm=arm,context=diagnostics)
        m.save_json(OUT/(output+'.json'),row)
        return row
    finally: r.decompose=original_decompose


def controls():
    verify();cfg,p=h.setup();h.weight(p,'nominal');rows=[]
    for scene in range(6):
        reference=episode(cfg,p,scene,121,'reference',f'controls/reference-{scene}-121')
        row=episode(cfg,p,scene,121,'matched',f'controls/matched-{scene}-121')
        if scene==0:
            for key in ['records','candidates']:
                np.testing.assert_array_equal(np.load(OUT/f'controls/reference-{scene}-121.npz')[key],np.load(OUT/f'controls/matched-{scene}-121.npz')[key])
            # All five retained identities make fixed support exactly the original query.
            episode(cfg,p,scene,121,'fixed','controls/fixed-five-0-121')
            for key in ['records','candidates']:
                np.testing.assert_array_equal(np.load(OUT/'controls/fixed-five-0-121.npz')[key],np.load(OUT/'controls/reference-0-121.npz')[key])
        if scene==1:
            episode(cfg,p,scene,121,'matched','controls/repeat-1-121')
            for key in ['records','candidates']:
                np.testing.assert_array_equal(np.load(OUT/'controls/repeat-1-121.npz')[key],np.load(OUT/'controls/matched-1-121.npz')[key])
        rows.append(row)
        print('CONTROL',scene,row['outcome'],row['context']['changed_support_decisions'],row['context']['ranking_changes'],flush=True)
    decisions=sum(v['context']['decisions'] for v in rows)
    changed=sum(v['context']['changed_support_decisions'] for v in rows)
    scenes=sum(v['context']['changed_support_decisions']>0 for v in rows)
    activated=changed/decisions>=.01 and scenes>=2 and sum(v['context']['ranking_changes'] for v in rows)>0
    verify();save('controls.json',{'passed':True,'grid':80,'time_limit':50,'reference_scores_exact':True,
        'five_human_full_trace_exact':True,'terminal_q_exact':True,'exact_duplicate':True,
        'decisions':decisions,'changed_support_decisions':changed,'split_scenes':scenes,
        'activation_passed':activated,'smoke':rows})


def evaluate(arm,cohort):
    verify();c=json.loads((HERE/'controls.json').read_text());assert c['passed'] and c['activation_passed']
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
    pairs={a:x.stats.pairs(rows['matched'],rows[a]) for a in SPEC['arms'] if a!='matched'}
    ratios={}
    for arm in pairs:
        base={(v['scene'],v['case']):v for v in rows[arm]}
        both=[v for v in rows['matched'] if v['success'] and base[(v['scene'],v['case'])]['success']]
        ratios[arm]=float(np.mean([v['duration'] for v in both])/np.mean([base[(v['scene'],v['case'])]['duration'] for v in both]))
    changed=[]
    for v in rows['matched']:
        stem=f'{m.SCENES.index(v["scene"])}-{v["case"]}.npz'
        base_path=HERE.parent/'progress-contract-v2/artifacts/development/hold8'/stem if cohort=='development' else OUT/'confirmation/hold8'/stem
        own=np.load(OUT/cohort/'matched'/stem)['records'];base=np.load(base_path)['records'];n=min(len(own),len(base))
        steps=np.flatnonzero((own[:n,3:5]!=base[:n,3:5]).any(1))
        if len(steps): changed.append({'scene':v['scene'],'case':v['case'],'first_executed_difference':int(steps[0])})
    own=overall['matched']
    checks={'success_nonregression':all(own['success']>=overall[a]['success'] for a in pairs),
        'safety':all(own['contact']<=overall[a]['contact'] and own['collision']<=overall[a]['collision'] for a in pairs),
        'independent_task_gain':all(own['success']>=overall[a]['success']+2 for a in pairs) or all(v<=.95 for v in ratios.values()),
        'activation':len(changed)>0,'compute':own['decision_ms_episode_mean']<250}
    if cohort=='confirmation':
        ids=sorted({v['case'] for v in rows['matched']});draws=np.random.default_rng(121012).integers(len(ids),size=(4000,len(ids)));intervals={}
        for arm in pairs:
            base={(v['scene'],v['case']):v for v in rows[arm]};counts=np.zeros(len(ids));sums=np.zeros(len(ids))
            for v in rows['matched']:
                other=base[(v['scene'],v['case'])]
                if v['success'] and other['success']:
                    j=ids.index(v['case']);counts[j]+=1;sums[j]+=v['duration']-other['duration']
            denominator=counts[draws].sum(1);assert (denominator>0).all()
            intervals[arm]=np.quantile(sums[draws].sum(1)/denominator,[.025,.975]).tolist()
        checks['fresh_intervals']=all(pairs[a]['delta_SR_case_bootstrap_95'][0]>0 or intervals[a][1]<0 for a in pairs)
        save('confirmation-time-intervals.json',intervals)
    save(f'{cohort}-summary.json',{'overall':overall,'pairs':pairs,'paired_success_time_ratios':ratios,
        'changed_executed_trajectories':changed,'gate':{'checks':checks,'passed':all(checks.values())},
        'verdict':'METHOD_SIGNAL_FOUND' if all(checks.values()) else 'VERSION_NEGATIVE','novelty':'UNVERIFIED'})
    print('SUMMARY',checks,ratios,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['freeze','verify','controls','evaluate','summarize'])
    parser.add_argument('--arm',choices=SPEC['arms']);parser.add_argument('--cohort',default='development',choices=['development','confirmation'])
    args=parser.parse_args()
    if args.command=='evaluate': evaluate(args.arm,args.cohort)
    elif args.command=='summarize': summarize(args.cohort)
    else: globals()[args.command]()
