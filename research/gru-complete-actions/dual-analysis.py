"""Two-policy fixed-world attribution; case-clustered, no historical-label claim."""
import argparse
import json

import numpy as np
import torch

import dual as d

q=d.q


def load_pair(root,index):
    folder=d.OUT/'catalogue'/q.key(root)
    return [json.loads((folder/f'action-{index:02d}-{policy}.json').read_text())
            for policy in d.SPEC['policies']]


def summarize():
    original={(v['scene'],v['case']):v for v in json.loads((q.OLD/'value-contrasts.json').read_text())}
    prefix_path=d.OUT/'prefixes.json'
    prefixes=json.loads(prefix_path.read_text()) if prefix_path.exists() else []
    records=[]
    regression=[]
    for root in q.roots():
        folder=d.OUT/'catalogue'/q.key(root)
        x=np.load(folder/'inputs.npz')
        indices=np.flatnonzero(x['safe'])
        pairs=[load_pair(root,int(i)) for i in indices]
        assert all(a['legal_first_action'] and b['legal_first_action'] for a,b in pairs)
        ga=np.array([v[0]['result']['return'] for v in pairs])
        gb=np.array([v[1]['result']['return'] for v in pairs])
        qa=x['Q_pred'][indices] if 'Q_pred' in x else x['A_q'][indices]
        qt=x['B_q'][indices]
        progress=x['progress'][indices]
        residual=qt-gb
        dx=progress-progress.mean()
        dy=residual-residual.mean()
        slope=float(dx.dot(dy)/max(dx.dot(dx),1e-15))
        regression.append({'case':root['case'],'numerator':float(dx.dot(dy)),
                           'denominator':float(dx.dot(dx))})
        chosen=original[(root['scene'],root['case'])]['selected_index']
        forward=original[(root['scene'],root['case'])]['forward_index']
        selected=load_pair(root,chosen)
        safe_success_a=[int(i) for i,v in zip(indices,pairs) if v[0]['result']['success'] and not v[0]['result']['contact']]
        safe_success_b=[int(i) for i,v in zip(indices,pairs) if v[1]['result']['success'] and not v[1]['result']['contact']]
        ps=[v for v in prefixes if q.key(v['root'])==q.key(root)]
        successful_prefixes=[v['prefix'] for v in ps if v['success'] and not v['contact']]
        if not selected[0]['result']['success'] and safe_success_a:
            category='VALUE_RANKING_FAILURE_FIXED_WORLD'
        elif not safe_success_a and successful_prefixes:
            category='MULTISTEP_POLICY_TRAP_TESTED_ACTION_SET'
        elif not safe_success_a:
            category='NO_CONTACT_FREE_RESCUE_IN_TESTED_SINGLE_OR_PREFIX_ACTIONS'
        elif any(v[0]['result']['return']>selected[0]['result']['return']+1e-6 for v in pairs):
            category='EFFICIENCY_OR_RETURN_MISRANK_WITH_SUCCESSFUL_NOMINAL'
        else:
            category='NO_REALIZED_RETURN_IMPROVEMENT_FOUND'
        v=np.sort(x['values'])
        gap=float(v[-1]-v[-2])
        implied=abs(slope)*float(np.ptp(progress))/.99
        # This is an exact numeric identity, not a causal identification of old V labels.
        np.testing.assert_allclose(qa-ga,(qa-qt)+(qt-gb)+(gb-ga),rtol=0,atol=1e-12)
        bpolicy_pairs=(gb[:,None]-gb[None,:])*(qt[:,None]-qt[None,:])
        apolicy_pairs=(ga[:,None]-ga[None,:])*(qt[:,None]-qt[None,:])
        valid=np.triu(np.ones((len(indices),len(indices)),bool),1)
        non_tied_a=valid&(np.abs(ga[:,None]-ga[None,:])>1e-6)
        non_tied_b=valid&(np.abs(gb[:,None]-gb[None,:])>1e-6)
        records.append({**root,'category':category,'legal_actions':len(indices),
            'nominal_index':chosen,'forward_index':forward,'selected_A':selected[0]['result'],
            'selected_B':selected[1]['result'],'A_contact_free_success_indices':safe_success_a,
            'B_contact_free_success_indices':safe_success_b,'successful_prefixes':successful_prefixes,
            'A_success_positive_progress':sum(x['progress'][i]>0 for i in safe_success_a),
            'A_success_nonpositive_progress':sum(x['progress'][i]<=0 for i in safe_success_a),
            'A_contact_branches':sum(v[0]['result']['contact'] for v in pairs),
            'B_contact_branches':sum(v[1]['result']['contact'] for v in pairs),
            'A_collision_branches':sum(v[0]['result']['collision'] for v in pairs),
            'B_collision_branches':sum(v[1]['result']['collision'] for v in pairs),
            'A_realized_regret_no_contact':float(max(v[0]['result']['return'] for v in pairs if not v[0]['result']['contact'])-selected[0]['result']['return']),
            'mean_model_term':float((qa-qt).mean()),
            'mean_raw_reference_residual':float(residual.mean()),
            'mean_policy_bundle_effect':float((gb-ga).mean()),
            'within_root_residual_progress_slope':slope,
            'unmasked_value_top_gap':gap,'slope_implied_V_span':implied,
            'slope_span_exceeds_gap':bool(implied>=gap),
            'A_realized_pair_inversions':int(((apolicy_pairs<0)&non_tied_a).sum()),
            'B_realized_pair_inversions':int(((bpolicy_pairs<0)&non_tied_b).sum()),
            'A_non_tied_pairs':int(non_tied_a.sum()),'B_non_tied_pairs':int(non_tied_b.sum()),
            'selected_policy_return_effect_B_minus_A':float(selected[1]['result']['return']-selected[0]['result']['return']),
            'pair_order_reversals_A_vs_B':int((valid&((ga[:,None]-ga[None,:])*(gb[:,None]-gb[None,:])<0)).sum())})
    ids=sorted({v['case'] for v in records})
    num=np.array([sum(v['numerator'] for v in regression if v['case']==i) for i in ids])
    den=np.array([sum(v['denominator'] for v in regression if v['case']==i) for i in ids])
    rng=np.random.default_rng(20261011)
    draw=rng.integers(len(ids),size=(2000,len(ids)))
    bootstrap=num[draw].sum(1)/den[draw].sum(1)
    checkpoints={}
    cp=d.OUT/'checkpoint-pairs.json'
    if cp.exists():
        c=json.loads(cp.read_text())
        if c['executed']:
            for stage in q.SPEC['P3']['checkpoints']:
                checkpoints[stage]={}
                for policy in d.SPEC['policies']:
                    rows=[v for v in c['rows'] if v['stage']==stage and v['policy']==policy]
                    assert len(rows)==38
                    delta=np.array([v['actual_next_values'][0]-v['actual_next_values'][1] for v in rows])
                    actual=np.array([v['outcomes'][0]['return']-v['outcomes'][1]['return'] for v in rows])
                    checkpoints[stage][policy]={
                        'negative_action_value_preferred':int((delta>1e-6).sum()),
                        'own_return_forward_better':int((actual< -1e-6).sum()),
                        'own_return_selected_better':int((actual>1e-6).sum()),
                        'own_return_tied':int((np.abs(actual)<=1e-6).sum()),
                        'strict_pair_misranks':int((((delta>1e-6)&(actual< -1e-6))|((delta< -1e-6)&(actual>1e-6))).sum()),
                        'selected_success':sum(v['outcomes'][0]['success'] for v in rows),
                        'forward_success':sum(v['outcomes'][1]['success'] for v in rows),
                        'selected_contact':sum(v['outcomes'][0]['contact'] for v in rows),
                        'forward_contact':sum(v['outcomes'][1]['contact'] for v in rows)}
    result={'cases':records,'categories':{k:sum(v['category']==k for v in records) for k in sorted({v['category'] for v in records})},
        'roots':len(records),'original_case_clusters':len(ids),
        'legal_first_actions':sum(v['legal_actions'] for v in records),
        'legal_two_policy_branches':sum(v['legal_actions']*2 for v in records),
        'selected':{policy:{'success':sum(v[f'selected_{policy}']['success'] for v in records),
                           'collision':sum(v[f'selected_{policy}']['collision'] for v in records),
                           'contact':sum(v[f'selected_{policy}']['contact'] for v in records)} for policy in ['A','B']},
        'selected_B_rescues_A':sum(not v['selected_A']['success'] and v['selected_B']['success'] for v in records),
        'selected_B_damages_A':sum(v['selected_A']['success'] and not v['selected_B']['success'] for v in records),
        'slope':{'estimate':float(num.sum()/den.sum()),'case_cluster_bootstrap_95':np.quantile(bootstrap,[.025,.975]).tolist(),
                 'replicates':2000,'case_ids':ids,'span_exceeds_gap_roots':sum(v['slope_span_exceeds_gap'] for v in records),
                 'limits':'Association on deliberately selected fixed worlds; not proof of OOD or historical training cause'},
        'checkpoints':checkpoints,'no_method_level_progress':True,
        'limits':['Historical original RL replay is unavailable.',
                  'B is deterministic final-weight raw greedy, not the historical epsilon behavior mixture.',
                  'A/B effects change safety, risk and smoothing as a bundle and do not isolate individual components.',
                  'The 38 roots are a selected developmental cohort, not a population performance evaluation.',
                  'No fixed-world best-action label is claimed to be the legal-observation conditional expected optimum.']}
    d.save('statistics.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['cases','checkpoints']},indent=2),flush=True)
    if checkpoints:
        print('CHECKPOINTS',json.dumps(checkpoints),flush=True)


def head_fit():
    stats=json.loads((d.OUT/'statistics.json').read_text())
    if stats['categories'].get('VALUE_RANKING_FAILURE_FIXED_WORLD',0)<2:
        d.save('head-fit.json',{'executed':False,'reason':'No repeated contact-free single-step loss; stop untargeted fitting'})
        return
    cfg,p=d.setup()
    original={(v['scene'],v['case']):v for v in json.loads((q.OLD/'value-contrasts.json').read_text())}
    features=[]
    labels=[]
    meta=[]
    for root in q.roots():
        data=np.load(d.OUT/'catalogue'/q.key(root)/'inputs.npz')
        old=original[(root['scene'],root['case'])]
        indices=[old['selected_index'],old['forward_index']]
        captured=[]
        handle=p.value_head.register_forward_pre_hook(lambda mod,args:captured.append(args[0].detach().cpu().numpy()))
        with torch.no_grad():
            value=p.forward_candidate_values(torch.as_tensor(data['windows'][indices],device=p.device)).cpu().numpy()
        handle.remove()
        assert len(captured)==1 and captured[0].shape==(2,256)
        np.testing.assert_allclose(value,old['B_values'],rtol=0,atol=2e-6)
        first=data['first'][indices]
        assert not first[:,1].any(), 'Terminal pairs excluded; this fixed list should have none'
        target=[]
        for policy in ['B_raw_greedy','A_deploy']:
            returns=np.array([load_pair(root,i)[0 if policy=='A_deploy' else 1]['result']['return'] for i in indices])
            target.append((returns-first[:,0])/.99)
        features.extend(captured[0])
        labels.append(target)
        meta.append({'root':root,'indices':indices,'first_rewards':first[:,0].tolist()})
    f=np.asarray(features,np.float64)
    x=np.c_[f,np.ones(len(f))]
    labels=np.asarray(labels,np.float64)
    fit_cases=np.array([v['root']['case']<120 for v in meta])
    fit=np.repeat(fit_cases,2)
    theta0=np.r_[p.value_head.weight.detach().cpu().numpy().ravel(),p.value_head.bias.detach().cpu().numpy()]
    predictions={'original_head':(x@theta0).reshape(38,2)}
    models={}
    for column,name in enumerate(d.SPEC['P4']['arms']):
        y=labels[:,column,:].reshape(-1)
        a=x[fit]
        delta=np.linalg.solve(a.T@a/len(a)+.01*np.eye(x.shape[1]),a.T@(y[fit]-a@theta0)/len(a))
        weights=theta0+delta
        predictions[name]=(x@weights).reshape(38,2)
        models[name]={'weights':weights.tolist(),'change_l2':float(np.linalg.norm(delta))}
    metrics={}
    for split,mask in [('fit',fit_cases),('held-out',~fit_cases)]:
        metrics[split]={}
        truth=labels[mask,1,:]
        td=truth[:,0]-truth[:,1]
        for name,pred in predictions.items():
            values=pred[mask]
            pd=values[:,0]-values[:,1]
            metrics[split][name]={'cases':int(mask.sum()),'non_tied_pairs':int((np.abs(td)>1e-6).sum()),
                'strict_ranking_errors':int((((td>1e-6)&(pd< -1e-6))|((td< -1e-6)&(pd>1e-6))).sum()),
                'pair_advantage_MAE':float(np.abs(td-pd).mean()),'V_MAE':float(np.abs(truth-values).mean()),
                'negative_action_preferred':int((pd>1e-6).sum())}
    d.save('head-fit.json',{'executed':True,'cases':meta,'metrics':metrics,'models':models,
        'predictions':{k:v.tolist() for k,v in predictions.items()},'V_labels':labels.tolist(),
        'matched_initialization_sha256':q.SPEC['nominal_sha256'],
        'only_fitted_arm_difference':'Fixed-final-weight continuation policy producing labels',
        'same_features_and_coverage':True,'same_parameter_count':257,'same_ridge_penalty':.01,
        'fit_case_count':int(fit_cases.sum()),'held_out_case_count':int((~fit_cases).sum()),
        'limits':['Original historical training labels are not recovered.',
                  'Both fitted heads differ in coverage from the original head; only fitted-arm contrast isolates source.',
                  'Held-out cases have no normal-success controls because the frozen case split puts the first controls in fitting.',
                  'No head closed-loop experiment; correction is not navigation gain or confirmed historical cause.']})
    np.savez_compressed(d.OUT/'head-fit-inputs.npz',features=f,labels=labels,fit_cases=fit_cases,original_head=theta0)
    print(json.dumps(metrics,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['summarize','head-fit'])
    args=parser.parse_args()
    globals()[args.stage.replace('-','_')]()
