"""Recomputable case-clustered summaries and a matched frozen-head diagnostic."""
import argparse
import json

import numpy as np
import torch

import run as q


def catalogue():
    prior={(v['scene'],v['case']):v for v in json.loads((q.OLD/'value-contrasts.json').read_text())}
    prefixes=json.loads((q.HERE/'prefixes.json').read_text()) if (q.HERE/'prefixes.json').exists() else []
    records=[]
    regression=[]
    for root in q.roots():
        folder=q.OUT/'catalogue'/q.key(root)
        data=np.load(folder/'inputs.npz')
        branches=[json.loads(p.read_text()) for p in sorted(folder.glob('action-*.json'))]
        legal=[v for v in branches if v['legal']]
        assert len(legal)==int(data['safe'].sum()), 'Incomplete legal action catalogue'
        old=prior[(root['scene'],root['case'])]
        chosen=next(v for v in legal if v['index']==old['selected_index'])
        safe_success=[v for v in legal if v['result']['success'] and not v['result']['contact']]
        best=max(legal,key=lambda v:v['result']['return'])
        best_contact_free=max((v for v in legal if not v['result']['contact']),
                              key=lambda v:v['result']['return'])
        improved=[v for v in legal if v['result']['return']>chosen['result']['return']+1e-6]
        ps=[v for v in prefixes if v['root']['scene']==root['scene'] and v['root']['case']==root['case']]
        successful_prefixes=[v for v in ps if v['success'] and not v['contact']]
        if not chosen['result']['success'] and safe_success:
            category='VALUE_RANKING_FAILURE_FIXED_WORLD'
        elif not safe_success and successful_prefixes:
            category='MULTISTEP_POLICY_TRAP_TESTED_ACTION_SET'
        elif not safe_success:
            category='NO_SAFE_SINGLE_STEP_RESCUE_AND_NO_TESTED_PREFIX_RESCUE'
        elif improved:
            category='EFFICIENCY_OR_RETURN_MISRANK_WITH_SUCCESSFUL_NOMINAL'
        else:
            category='NO_REALIZED_RETURN_IMPROVEMENT_FOUND'
        g=np.array([v['result']['return'] for v in legal])
        a=np.array([v['A_q'] for v in legal])
        b=np.array([v['B_q'] for v in legal])
        x=np.array([v['progress'] for v in legal])
        # Each row remains grouped in the same original case in all resampling.
        regression.append((x-x.mean(),(b-g)-(b-g).mean()))
        return_order=np.sign(g[:,None]-g[None,:])
        predicted_order=np.sign(b[:,None]-b[None,:])
        pairmask=np.triu(np.ones_like(return_order,dtype=bool),1)&(np.abs(g[:,None]-g[None,:])>1e-6)
        inverted=(return_order*predicted_order<0)&pairmask
        records.append({**root,'category':category,'legal_actions':len(legal),
            'nominal_index':chosen['index'],'nominal':chosen['result'],
            'forward_index':old['forward_index'],'better_realized_actions':len(improved),
            'safe_success_actions':len(safe_success),
            'success_action_indices':[v['index'] for v in safe_success],
            'safe_success_forward_actions':sum(v['progress']>0 for v in safe_success),
            'safe_success_backward_actions':sum(v['progress']<=0 for v in safe_success),
            'best_realized_index':best['index'],'best_realized':best['result'],
            'best_contact_free_index':best_contact_free['index'],
            'contact_free_realized_regret':float(best_contact_free['result']['return']-chosen['result']['return']),
            'single_branches_contact':sum(v['result']['contact'] for v in legal),
            'single_branches_collision':sum(v['result']['collision'] for v in legal),
            'A_B_rank_disagreements':int((np.sign(a[:,None]-a[None,:])*predicted_order<0).sum()//2),
            'non_tied_realized_pairs':int(pairmask.sum()),'inverted_realized_pairs':int(inverted.sum()),
            'successful_prefix_lengths':[v['requested_prefix'] for v in successful_prefixes],
            'minimum_successful_prefix':min((v['requested_prefix'] for v in successful_prefixes),default=None),
            'root_error_mean':float((b-g).mean()),
            'within_root_error_progress_slope':float(x.dot(x)*0+(x-x.mean()).dot((b-g)-(b-g).mean())/max((x-x.mean()).dot(x-x.mean()),1e-15))})
    numer=np.array([x.dot(y) for x,y in regression])
    denom=np.array([x.dot(x) for x,y in regression])
    slope=float(numer.sum()/denom.sum())
    rng=np.random.default_rng(20261011)
    draw=rng.integers(len(records),size=(2000,len(records)))
    boots=numer[draw].sum(1)/denom[draw].sum(1)
    stages={}
    if (q.HERE/'checkpoint-pairs.json').exists():
        pairs=json.loads((q.HERE/'checkpoint-pairs.json').read_text())
        for stage in q.SPEC['P3']['checkpoints']:
            rows=[v for v in pairs if v['stage']==stage]
            assert len(rows)==38
            err=[]
            for v in rows:
                difference=v['B_values'][0]-v['B_values'][1]
                actual=v['outcomes'][0]['return']-v['outcomes'][1]['return']
                err.append(difference*actual < -1e-8)
            stages[stage]={'roots':len(rows),
                'negative_action_value_higher':sum(v['B_values'][0]>v['B_values'][1]+1e-6 for v in rows),
                'own_continuation_prefers_forward':sum(v['outcomes'][1]['return']>v['outcomes'][0]['return']+1e-6 for v in rows),
                'own_continuation_prefers_selected':sum(v['outcomes'][0]['return']>v['outcomes'][1]['return']+1e-6 for v in rows),
                'strict_own_policy_misranks':int(sum(err)),
                'selected_success':sum(v['outcomes'][0]['success'] for v in rows),
                'forward_success':sum(v['outcomes'][1]['success'] for v in rows),
                'selected_contact':sum(v['outcomes'][0]['contact'] for v in rows),
                'forward_contact':sum(v['outcomes'][1]['contact'] for v in rows)}
    counts={c:sum(v['category']==c for v in records) for c in sorted({v['category'] for v in records})}
    result={'cases':records,'category_counts':counts,
        'legal_actions_total':sum(v['legal_actions'] for v in records),
        'all_timing_branches_including_unsafe':sum(len(list((q.OUT/'catalogue'/q.key(v)).glob('action-*.json'))) for v in q.roots()),
        'regression':{'within_root_error_progress_slope':slope,'case_cluster_bootstrap_95':np.quantile(boots,[.025,.975]).tolist(),
            'cases':len(records),'replicates':2000,'interpretation':'Association on the selected fixed-world cohort, not historical training causality or conditional expected-value bias'},
        'checkpoint_own_continuation':stages,
        'no_new_algorithm':True,
        'limits':['Case list is an intentionally selected developmental diagnostic cohort.',
                  'Hidden-world realized returns are not Bayes-optimal action labels.',
                  'All-single-action claims are limited to the original 80-action support and original safety mask.',
                  'A one-step safe candidate can later lead to contact; these branches are retained and not called safe rescues.']}
    q.save('statistics.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2),flush=True)


def head_fit():
    stats=json.loads((q.HERE/'statistics.json').read_text())
    failures=[v for v in stats['cases'] if v['category']=='VALUE_RANKING_FAILURE_FIXED_WORLD']
    if len(failures)<2:
        q.save('head-fit.json',{'executed':False,'reason':'Fewer than two independent single-action loss cases; no untargeted fitting'})
        return
    cfg,p=q.setup()
    old={(v['scene'],v['case']):v for v in json.loads((q.OLD/'value-contrasts.json').read_text())}
    stages=json.loads((q.HERE/'checkpoint-pairs.json').read_text())
    labels={(v['stage'],v['root']['scene'],v['root']['case']):v for v in stages}
    features=[]
    records=[]
    for root in q.roots():
        data=np.load(q.OUT/'catalogue'/q.key(root)/'inputs.npz')
        row=old[(root['scene'],root['case'])]
        indices=[row['selected_index'],row['forward_index']]
        captured=[]
        handle=p.value_head.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().cpu().numpy()))
        with torch.no_grad():
            v=p.forward_candidate_values(torch.as_tensor(data['windows'][indices],device=p.device)).cpu().numpy()
        handle.remove()
        assert len(captured)==1 and captured[0].shape==(2,256)
        np.testing.assert_allclose(v,row['B_values'],rtol=0,atol=2e-6)
        targets=[]
        for stage in ['RL-500','RL-2000']:
            returns=np.array([v['return'] for v in labels[(stage,root['scene'],root['case'])]['outcomes']])
            rewards=data['first'][indices,0]
            done=data['first'][indices,1].astype(bool)
            assert not done.any(), 'Terminal pair needs separate handling, not V fitting'
            targets.append((returns-rewards)/.99)
        features.extend(captured[0])
        records.append({'root':root,'indices':indices,'V_labels':np.asarray(targets).tolist(),
                        'immediate_rewards':rewards.tolist(),'terminal':done.tolist()})
    f=np.asarray(features,np.float64)
    design=np.c_[f,np.ones(len(f))]
    targets=np.asarray([v['V_labels'] for v in records],np.float64)
    fit_case=np.array([v['root']['case']<120 for v in records])
    fit=np.repeat(fit_case,2)
    base=np.r_[p.value_head.weight.detach().cpu().numpy().reshape(-1),p.value_head.bias.detach().cpu().numpy()]
    pred0=(design@base).reshape(38,2)
    penalty=.01
    models={}
    for k,name in enumerate(q.SPEC['P4']['arms']):
        y=targets[:,k,:].reshape(-1)
        x=design[fit]
        # Solve a single fixed objective for a change from the same original head.
        delta=np.linalg.solve(x.T@x/len(x)+penalty*np.eye(x.shape[1]),x.T@(y[fit]-x@base)/len(x))
        theta=base+delta
        pred=(design@theta).reshape(38,2)
        models[name]={'head_parameter_l2_change':float(np.linalg.norm(delta)),
                      'predictions':pred.tolist(),'weights':theta.tolist()}
    predictions={'original-unfitted-head':pred0,**{k:np.asarray(v['predictions']) for k,v in models.items()}}
    true=targets[:,1,:]
    panels={}
    for split,mask in [('fit',fit_case),('held-out',~fit_case)]:
        panel={}
        for name,pred in predictions.items():
            yd=true[mask,0]-true[mask,1]
            pd=pred[mask,0]-pred[mask,1]
            strict=np.abs(yd)>1e-6
            panel[name]={'cases':int(mask.sum()),'non_tied_pairs':int(strict.sum()),
                'strict_ranking_errors':int(((yd*pd<0)&strict).sum()),
                'pair_advantage_MAE':float(np.abs(yd-pd).mean()),
                'V_MAE':float(np.abs(true[mask]-pred[mask]).mean()),
                'negative_action_preferred':int((pd>1e-6).sum())}
        panels[split]=panel
    q.save('head-fit.json',{'executed':True,'records':records,'models':models,'panels':panels,
        'fit_cases':int(fit_case.sum()),'held_out_cases':int((~fit_case).sum()),
        'parameter_count_each':257,'ridge_penalty':penalty,'exact_solve':True,
        'initial_checkpoint_sha256':q.SPEC['nominal_sha256'],
        'only_fitted_arm_difference':'Continuation-policy source of labels, RL500 versus RL2000',
        'limits':['Neither label source is the unavailable historical RL MC replay.',
                  'Compared to the original head both refits also use new counterfactual coverage. Only the two fitted arms isolate label-source change.',
                  'No new-head navigation rollout was run; ranking correction is not navigation improvement.',
                  'Failure to correct does not identify a raw-input insufficiency.']})
    np.savez_compressed(q.OUT/'head-features.npz',features=f,targets=targets,fit_cases=fit_case,
                        original_head=base)
    print(json.dumps(panels,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['catalogue','head-fit'])
    args=parser.parse_args()
    globals()[args.stage.replace('-','_')]()
