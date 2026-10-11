"""Recompute cause evidence and paired diagnostics, without models or GPU."""
import json
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'recovery-duration'
SCENES=['baseline_circle','baseline_square','dense_circle','dense_square','large_circle','large_square']


def read(name):
    return json.loads((HERE/name).read_text())


def block_interval(rows, field):
    cases=sorted({r['case'] for r in rows})
    blocks=np.array([[sum(r[field] for r in rows if r['case']==c),
                      sum(r['case']==c for r in rows)] for c in cases])
    draw=np.random.default_rng(20261011).integers(len(cases),size=(10000,len(cases)))
    sampled=blocks[draw].sum(1)
    estimates=sampled[:,0]/sampled[:,1]
    return {'mean':float(np.mean([r[field] for r in rows])),
            'ci95':np.percentile(estimates,[2.5,97.5]).tolist(),
            'case_seed_blocks':len(cases)}


def main():
    ledger=read('nominal-ledger.json')
    scores=np.load(HERE/'artifacts'/'nominal-action-scores.npz')['scores']
    assert len(scores)==len(ledger)
    selection_metadata=[]
    # Preserve the frozen ledger; correct descriptive IDs from saved legal states.
    for row,actions in zip(ledger,scores):
        tape=np.load(OLD/'artifacts'/'tapes'/f'{SCENES.index(row["scene"])}-{row["case"]}.npz')['humans']
        human=tape[row['step']]
        predicted=human[:,:2]+.25*human[:,2:4]
        poses=np.asarray(row['position'])[None,:]+.25*actions[:,:2]
        ids=[tuple(sorted(np.argsort(((predicted-pose)**2).sum(1),kind='stable')[:5].tolist()))
             for pose in poses]
        corrected={'scene':row['scene'],'case':row['case'],'step':row['step'],
                   'support_switch':len(set(ids))>1,
                   'chosen_selected_ids':list(ids[row['chosen_index']]),
                   'progress_selected_ids':list(ids[row['progress_index']])}
        selection_metadata.append(corrected)
        row.update(corrected)
    (HERE/'human-selection-metadata.json').write_text(json.dumps({
        'scope':'Corrected descriptive metadata only: use CV-propagated humans as the unchanged planner does. Original frozen ledger retained; no action/value/outcome was changed.',
        'rows':selection_metadata},indent=2)+'\n')
    first=read('recovery-first-choice.json')
    histories=read('recovery-history-channel.json')['rows']
    collision=read('recovery-collision-choices.json')['rows']
    off=read('recovery-offline.json')
    contract=read('recovery-contract.json')
    c=read('counterfactual-results.json')
    assert len(c)==2*len(read('counterfactual-protocol.json')['roots'])
    assert len({(r['scene'],r['case'],r['variant']) for r in c})==len(c)
    result={'recovery':{},'ledger':{},'counterfactuals':{},'recurrence':{},'terminal':{}}
    nominal_results={(r['scene'],r['case']):r for r in
                     json.loads((OLD/'results'/'development-results.json').read_text()) if r['arm']=='nominal'}
    nominal_metrics=[]
    root_lookup={(r['scene'],r['case'],r['step']):r for r in ledger}
    for row in read('counterfactual-protocol.json')['roots']:
        episode=nominal_results[(row['scene'],row['case'])]
        trace=np.load(OLD/'artifacts'/'development-traces'/'nominal'/f'{SCENES.index(row["scene"])}-{row["case"]}.npz')['records']
        tail=trace[trace[:,0]>=row['step']]
        nominal_metrics.append({'time':len(tail)*.25,
                                'path':float(np.linalg.norm(tail[:,3:5],axis=1).sum()*.25),
                                'min_clearance':float(tail[:,8].min()),
                                'progress':root_lookup[(row['scene'],row['case'],row['step'])]['goal_distance']-episode['final_distance'],
                                'success':episode['success']})
    def metrics(rows):
        return {'episodes':len(rows),
                **{key:float(np.mean([r[key] for r in rows])) for key in ['time','path','min_clearance','progress']},
                'worst_clearance':float(min(r['min_clearance'] for r in rows)),
                'success_only_time':float(np.mean([r['time'] for r in rows if r['success']]))}
    result['navigation_metrics']={'nominal_suffix':metrics(nominal_metrics)}
    for variant in ['one-step-progress','value-free-progress']:
        result['navigation_metrics'][variant]=metrics([r['result'] for r in c if r['variant']==variant])
    for arm in ['ordinary','paired']:
        z=[r for r in first if r['arm']==arm]
        h=[r for r in histories if r['arm']==arm]
        contact=[r for r in collision if r['arm']==arm]
        trace=off['traces'][arm]
        result['recovery'][arm]={
            'first_at_step4':sum(r['first_recovery_step']==4 for r in trace),
            'trace_episodes':len(trace),'countdown_errors':sum(r['countdown_errors'] for r in trace),
            'longest_non_nominal_steps':max(r['longest_changed_run_steps'] for r in trace),
            'one_shot_successes':sum(r['one_shot']['success'] for r in z),
            'repeated_successes':sum(r['repeated_outcome']=='reach_goal' for r in z),
            'nominal_successes':sum(r['nominal_outcome']=='reach_goal' for r in z),
            'first_choice_negative_return_advantage':sum(r['one_shot_advantage']< -1e-6 for r in z),
            'one_shot_success_repeat_failure':sum(r['one_shot']['success'] and r['repeated_outcome']!='reach_goal' for r in z),
            'one_shot_failure_repeat_success':sum(not r['one_shot']['success'] and r['repeated_outcome']=='reach_goal' for r in z),
            'mean_single_minus_repeat_return':float(np.mean([r['one_shot']['discounted_return']-r['repeated_root_return'] for r in z])),
            'history_input_weights_unchanged':sum(r['input_weights_identical_to_initialization'] for r in h),
            'history_channel_choice_changes':sum(r['actual_choice']!=r['age_neutralized_choice'] for r in h),
            'history_channel_roots':len(h),
            'contact_steps':len(contact),'empty_safe_contact_steps':sum(r['safe_count']==0 for r in contact),
            'noncontact_candidate_available':sum(r['noncontact_candidates']>0 for r in contact),
            'nominal_action_avoids_immediate_contact':sum(r['nominal_actual_clearance']>=0 for r in contact)}
    result['recovery']['execution_replayed_episodes']=len(contract['episodes'])
    result['recovery']['execution_checked_roots']=len(contract['decisions_and_collision_roots'])
    result['recovery']['execution_max_replay_error']=max(max(r['replay_errors'].values()) for r in contract['episodes'])
    for partition in ['discovery','mechanism-check']:
        for outcome in ['reach_goal','timeout']:
            z=[r for r in ledger if r['partition']==partition and r['outcome']==outcome]
            result['ledger'][partition+' / '+outcome]={
                'roots':len(z),'episodes':len({(r['scene'],r['case']) for r in z}),
                'value_overrides_progress_roots':sum(r['value_overrides_progress'] for r in z),
                'value_override_fraction':float(np.mean([r['value_overrides_progress'] for r in z])),
                'empty_safe_roots':sum(r['safe_count']==0 for r in z),
                'support_switch_roots':sum(r['support_switch'] for r in z)}
    by_case={}
    for r in ledger:
        by_case.setdefault((r['scene'],r['case']),[]).append(r)
    case_index=[]
    for key,roots in by_case.items():
        roots=sorted(roots,key=lambda r:r['step'])
        pairs=[(a['step'],b['step']) for a,b in zip(roots,roots[1:]) if b['step']-a['step']==8 and
               a['safe_count']==b['safe_count']==80 and a['value_overrides_progress'] and b['value_overrides_progress']]
        case_index.append({'scene':key[0],'case':key[1],'partition':roots[0]['partition'],
                           'outcome':roots[0]['outcome'],'reversal_roots':sum(r['value_overrides_progress'] for r in roots),
                           'recurrence':bool(pairs),'recurrence_steps':pairs})
    for partition in ['discovery','mechanism-check']:
        result['recurrence'][partition]={}
        for bit in [False,True]:
            z=[r for r in case_index if r['partition']==partition and r['recurrence']==bit]
            result['recurrence'][partition][str(bit)]={
                'episodes':len(z),'timeouts':sum(r['outcome']=='timeout' for r in z),
                'timeout_fraction':float(np.mean([r['outcome']=='timeout' for r in z])) if z else None}
    flat=[]
    for r in c:
        flat.append({**r,'success_delta':r['result']['success']-int(r['outcome']=='reach_goal')})
    for partition in ['discovery','mechanism-check','all']:
        for variant in ['one-step-progress','value-free-progress']:
            z=[r for r in flat if r['variant']==variant and (partition=='all' or r['partition']==partition)]
            rescues=[{'scene':r['scene'],'case':r['case'],'step':r['step']} for r in z if r['success_delta']>0]
            damages=[{'scene':r['scene'],'case':r['case'],'step':r['step'],'outcome':r['result']['outcome']}
                     for r in z if r['success_delta']<0]
            result['counterfactuals'][partition+' / '+variant]={
                'episodes':len(z),'nominal_successes':sum(r['outcome']=='reach_goal' for r in z),
                'alternative_successes':sum(r['result']['success'] for r in z),
                'collisions':sum(r['result']['collision'] for r in z),
                'contacts':sum(r['result']['contact'] for r in z),
                'sr_difference':block_interval(z,'success_delta'),
                'return_advantage':block_interval(z,'return_advantage'),
                'rescues':rescues,'damages':damages,
                'first_step_contacts':int(sum(np.load(HERE/'artifacts'/f'counterfactual-{variant}-{SCENES.index(r["scene"])}-{r["case"]}.npz')['records'][0,8]<0 for r in z))}
    # The current planner and native environment disagree on the arrival annulus.
    terminal_rows=[]
    for row,actions in zip(ledger,scores):
        terminal=(row['goal_distance']-actions[:,2])<row['native_radius']
        old_goal=(row['goal_distance']-actions[:,2])<row['policy_success_radius']
        if not np.any(terminal!=old_goal):
            continue
        arm_scene=SCENES.index(row['scene'])
        trace=np.load(OLD/'artifacts'/'development-traces'/'nominal'/f'{arm_scene}-{row["case"]}.npz')['records']
        executed=trace[trace[:,0]==row['step']][0]
        chosen=row['chosen_index']
        terminal_rows.append({'scene':row['scene'],'case':row['case'],'step':row['step'],
                              'candidate_disagreements':int((terminal!=old_goal).sum()),
                              'chosen_in_annulus':bool(terminal[chosen] and not old_goal[chosen]),
                              'chosen_predicted_reward':row['chosen_reward'],
                              'chosen_actual_reward':float(executed[7]),
                              'chosen_next_value':row['chosen_value'],
                              'outcome':row['outcome']})
    result['terminal']={'roots_with_candidate_radius_mismatch':len(terminal_rows),
                        'executed_annulus_mismatches':sum(r['chosen_in_annulus'] for r in terminal_rows),
                        'rows':terminal_rows}
    (HERE/'findings-statistics.json').write_text(json.dumps(result,indent=2)+'\n')
    (HERE/'case-index.json').write_text(json.dumps(case_index,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['counterfactuals','terminal']},indent=2))
    print('counterfactuals', {k:{'n':v['episodes'],'successes':v['alternative_successes'],
                                      'collisions':v['collisions'],'delta':v['sr_difference'],
                                      'return':v['return_advantage']} for k,v in result['counterfactuals'].items()})
    print('terminal', {k:v for k,v in result['terminal'].items() if k!='rows'})


if __name__=='__main__':
    main()
