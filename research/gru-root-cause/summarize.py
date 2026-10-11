"""Recompute episode-level and fixed-root causal contrast counts from saved data."""
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent


def read(name):
    return json.loads((HERE/name).read_text())


def diff(x):
    return x[0]-x[1]


def main():
    baseline=read('repaired-baseline.json')
    roots=read('value-contrasts.json')
    tolerance=1e-6
    wrong=[r for r in roots if diff(r['B_q'])>tolerance and diff(r['C_returns'])<-tolerance]
    tied=[r for r in roots if abs(diff(r['C_returns']))<=tolerance]
    old=json.loads((HERE.parent/'navigation-findings'/'counterfactual-results.json').read_text())
    continuous={(r['scene'],r['case']):r for r in old if r['variant']=='value-free-progress'}
    policy_traps=[r for r in tied if all(o['timeout'] for o in r['C_outcomes'])
                 and continuous[(r['scene'],r['case'])]['result']['success']
                 and not continuous[(r['scene'],r['case'])]['result']['contact']]
    result={'repaired_baseline':{
        'episodes':len(baseline),'successes':sum(r['success'] for r in baseline),
        'collisions':sum(r['collision'] for r in baseline),'contacts':sum(r['contact'] for r in baseline),
        'timeouts':sum(r['timeout'] for r in baseline),'roots':sum(len(r['sampled_roots']) for r in baseline),
        'negative_progress_roots':sum(r['flagged_roots'] for r in baseline),
        'changed_outcomes':[{k:r[k] for k in ['scene','case','old_outcome','outcome']}
                            for r in baseline if r['old_outcome']!=r['outcome']]},
        'three_level':{'roots':len(roots),'historical_root_still_flagged':sum(r['root_flag_after_repair'] for r in roots),
            'repaired_policy_reaches_same_root':sum(r['repaired_policy_reaches_same_root'] for r in roots),
            'B_reverses_A_task_ranking':sum(diff(r['A_q'])*diff(r['B_q'])<0 for r in roots),
            'B_C_strict_realized_misrank':len(wrong),'C_forward_better':sum(diff(r['C_returns'])<-tolerance for r in roots),
            'C_selected_better':sum(diff(r['C_returns'])>tolerance for r in roots),'C_tied':len(tied),
            'selected_successes':sum(r['C_outcomes'][0]['success'] for r in roots),
            'forward_successes':sum(r['C_outcomes'][1]['success'] for r in roots),
            'one_step_rescues':sum(not r['C_outcomes'][0]['success'] and r['C_outcomes'][1]['success'] for r in roots),
            'one_step_damages':sum(r['C_outcomes'][0]['success'] and not r['C_outcomes'][1]['success'] for r in roots),
            'policy_trap_with_saved_safe_alternative':[{k:r[k] for k in ['scene','case','step']} for r in policy_traps]},
        'strata':{},'upstream':{}}
    result['three_level'].update({
        'equal_predicted_immediate_rewards':sum(abs(diff(r['A_rewards']))<=tolerance for r in roots),
        'nonzero_pair_risk_terms':sum(any(abs(x)>tolerance for x in r['risk']) for r in roots),
        'mean_absolute_A_B_value_difference':float(np.mean([np.abs(np.subtract(r['A_values'],r['B_values'])) for r in roots])),
        'native_reward_differs_from_prediction':sum(bool(np.max(np.abs(np.subtract(r['A_rewards'],r['actual_rewards'])))>tolerance) for r in roots),
        'both_timeout_return_ties':sum(all(o['timeout'] for o in r['C_outcomes']) for r in tied),
        'forward_better_both_success':sum(all(o['success'] for o in r['C_outcomes']) for r in wrong),
        'selected_contacts':sum(r['C_outcomes'][0]['contact'] for r in roots),
        'forward_contacts':sum(r['C_outcomes'][1]['contact'] for r in roots),
        'timeout_predicted_positive':sum(r['B_q'][0]>0 for r in roots if r['C_outcomes'][0]['timeout']),
        'timeout_median_B_minus_C':float(np.median([r['B_q'][0]-r['C_returns'][0] for r in roots if r['C_outcomes'][0]['timeout']]))})
    result['scenes']={}
    for scene in sorted({r['scene'] for r in baseline}):
        episodes=[r for r in baseline if r['scene']==scene]
        subset=[r for r in roots if r['scene']==scene]
        result['scenes'][scene]={'episodes':len(episodes),'success':sum(r['success'] for r in episodes),
            'collision':sum(r['collision'] for r in episodes),'contact':sum(r['contact'] for r in episodes),
            'timeout':sum(r['timeout'] for r in episodes),'fixed_roots':len(subset),
            'B_C_misrank':sum(diff(r['B_q'])>tolerance and diff(r['C_returns'])<-tolerance for r in subset)}
    def recurrence(episode):
        points=episode['sampled_roots']
        return any(a['flag'] and b['flag'] and a['safe_count']==b['safe_count']==80
                   and b['step']-a['step']==8 for a,b in zip(points,points[1:]))
    result['reused_recurrence_rule']={}
    for name,subset in [('discovery',[r for r in baseline if r['case']<120]),
                        ('mechanism-check',[r for r in baseline if r['case']>=120])]:
        result['reused_recurrence_rule'][name]={
            'positive_episodes':sum(recurrence(r) for r in subset),
            'positive_timeouts':sum(recurrence(r) and r['timeout'] for r in subset),
            'negative_episodes':sum(not recurrence(r) for r in subset),
            'negative_timeouts':sum(not recurrence(r) and r['timeout'] for r in subset)}
    result['three_level']['both_timeout_ties_equal_pure_terminal_return']=sum(
        all(o['timeout'] for o in r['C_outcomes'])
        and all(abs(v+.5*.99**(o['steps']-1))<1e-9 for v,o in zip(r['C_returns'],r['C_outcomes']))
        for r in tied)
    for label,subset in [('all',roots),('discovery',[r for r in roots if r['partition']=='discovery']),
                        ('mechanism-check',[r for r in roots if r['partition']=='mechanism-check'])]:
        result['strata'][label]={'roots':len(subset),
            'B_C_misrank':sum(diff(r['B_q'])>tolerance and diff(r['C_returns'])<-tolerance for r in subset),
            'selected_success':sum(r['C_outcomes'][0]['success'] for r in subset),
            'selected_timeout':sum(r['C_outcomes'][0]['timeout'] for r in subset)}
    root_lookup={(r['scene'],r['case']):r for r in roots}
    for path in sorted(HERE.glob('upstream-*.json')):
        if path.name=='upstream-protocol.json':
            continue
        rows=json.loads(path.read_text())
        grouped={}
        for row in rows:
            grouped.setdefault((row['scene'],row['case']),[]).append(row)
        metrics=[]
        for key,draws in grouped.items():
            original=root_lookup[key]
            value_difference=diff(original['B_q'])
            actual_difference=diff(original['C_returns'])
            changed_difference=float(np.mean([diff(r['returns']) for r in draws]))
            metrics.append({'scene':key[0],'case':key[1],
                'old_strict_misrank':bool(value_difference>tolerance and actual_difference<-tolerance),
                'new_strict_misrank':bool(value_difference>tolerance and changed_difference<-tolerance),
                'reconciled':bool(value_difference>tolerance and actual_difference<-tolerance and changed_difference>=-tolerance),
                'baseline_selected_success':original['C_outcomes'][0]['success'],
                'selected_success_probability':float(np.mean([r['outcomes'][0]['success'] for r in draws])),
                'selected_contact_probability':float(np.mean([r['outcomes'][0]['contact'] for r in draws])),
                'return_rank_before':actual_difference,'return_rank_after':changed_difference})
        result['upstream'][path.stem[len('upstream-'):]]={
            'roots':len(grouped),'draw_pairs':len(rows),'reconciled_rankings':sum(r['reconciled'] for r in metrics),
            'remaining_misrankings':sum(r['new_strict_misrank'] for r in metrics),
            'selected_successes_or_expected_count':sum(r['selected_success_probability'] for r in metrics),
            'selected_contacts_or_expected_count':sum(r['selected_contact_probability'] for r in metrics),
            'rescues_or_expected_count':sum(r['selected_success_probability'] for r in metrics if not r['baseline_selected_success']),
            'damages_or_expected_count':sum(1-r['selected_success_probability'] for r in metrics if r['baseline_selected_success']),
            'metrics':metrics}
        if path.stem=='upstream-deadline50':
            still_failed=[row for row in rows if row['outcomes'][0]['timeout']]
            result['upstream']['deadline50'].update({
                'selected_timeouts50':len(still_failed),
                'timeouts50_predicted_positive':sum(root_lookup[(row['scene'],row['case'])]['B_q'][0]>0 for row in still_failed),
                'timeouts50_median_B_minus_C':float(np.median([
                    root_lookup[(row['scene'],row['case'])]['B_q'][0]-row['returns'][0] for row in still_failed]))})
    result['causal_training']={}
    evaluations={}
    for arm in ['control','matched-smoothing']:
        directory=HERE/'artifacts'/'causal-evaluation'/arm
        if not (directory/'complete.json').exists():
            continue
        episodes=json.loads((directory/'baseline.json').read_text())
        contrasts=json.loads((directory/'contrasts.json').read_text())
        metadata=json.loads((directory/'complete.json').read_text())
        training=json.loads((HERE/'artifacts'/'causal-training'/arm/'manifest.json').read_text())
        disagreements=[r for r in contrasts if diff(r['fixed_pair_B_q'])*diff(r['C_returns'])<0
                       and min(abs(diff(r['fixed_pair_B_q'])),abs(diff(r['C_returns'])))>tolerance]
        result['causal_training'][arm]={'episodes':len(episodes),'roots':len(contrasts),
            'success':sum(r['success'] for r in episodes),'collision':sum(r['collision'] for r in episodes),
            'contact':sum(r['contact'] for r in episodes),'timeout':sum(r['timeout'] for r in episodes),
            'mean_discounted_return':float(np.mean([r['return'] for r in episodes])),
            'success_mean_duration':float(np.mean([r['duration'] for r in episodes if r['success']])),
            'minimum_clearance':float(min(r['min_clearance'] for r in episodes)),
            'negative_progress_roots':sum(r['flagged_roots'] for r in episodes),
            'sampled_roots':sum(len(r['sampled_roots']) for r in episodes),
            'fixed_pair_B_C_misrank':len(disagreements),
            'fixed_old_selected_preference_removed':sum(diff(r['fixed_pair_B_q'])<=tolerance for r in contrasts),
            'new_choice_nonnegative_progress_at_old_root':sum(r['new_selected_progress']>=0 for r in contrasts),
            'new_action_index_changed_at_old_root':sum(r['new_selected_index']!=r['fixed_indices'][0] for r in contrasts),
            'fixed_pair_B_C_MAE':float(np.mean([np.abs(np.subtract(r['fixed_pair_B_q'],r['C_returns'])) for r in contrasts])),
            'fixed_old_states_still_naturally_reached':sum(r['new_policy_reaches_same_root'] for r in contrasts),
            'training_seconds':training['elapsed_seconds'],'evaluation_seconds':metadata['elapsed_seconds'],
            'per_scene':{scene:{'success':sum(r['success'] for r in episodes if r['scene']==scene),
                               'collision':sum(r['collision'] for r in episodes if r['scene']==scene),
                               'timeout':sum(r['timeout'] for r in episodes if r['scene']==scene),
                               'contact':sum(r['contact'] for r in episodes if r['scene']==scene)}
                         for scene in sorted({r['scene'] for r in episodes})}}
        evaluations[arm]={(r['scene'],r['case']):r for r in episodes}
    if len(evaluations)==2:
        c,t=evaluations['control'],evaluations['matched-smoothing']
        assert set(c)==set(t) and len(c)==240
        result['causal_training']['paired']={
            'rescues':sum(not c[k]['success'] and t[k]['success'] for k in c),
            'damages':sum(c[k]['success'] and not t[k]['success'] for k in c),
            'outcome_changes':[{'scene':k[0],'case':k[1],'control':c[k]['outcome'],'treatment':t[k]['outcome']} for k in c if c[k]['outcome']!=t[k]['outcome']]}
        common=[k for k in c if c[k]['success'] and t[k]['success']]
        result['causal_training']['paired']['common_success_efficiency']={
            'episodes':len(common),'mean_treatment_minus_control_duration':float(np.mean([t[k]['duration']-c[k]['duration'] for k in common])),
            'median_treatment_minus_control_duration':float(np.median([t[k]['duration']-c[k]['duration'] for k in common]))}
        sensitive={(r['scene'],r['case']) for r in result['upstream']['no-smoothing']['metrics'] if r['reconciled']}
        result['causal_training']['paired']['old_smoothing_sensitive_cases']={
            'cases':len(sensitive),
            'rescues':sum(not c[k]['success'] and t[k]['success'] for k in sensitive),
            'damages':sum(c[k]['success'] and not t[k]['success'] for k in sensitive)}
        cm,tm=[result['causal_training'][a] for a in ['control','matched-smoothing']]
        result['causal_training']['frozen_threshold_passed']=bool(
            tm['fixed_pair_B_C_misrank']<=.8*cm['fixed_pair_B_C_misrank']
            and tm['timeout']<=cm['timeout']-4 and tm['contact']<=cm['contact']
            and tm['collision']<=cm['collision'])
    (HERE/'statistics.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='upstream'},indent=2))
    print(json.dumps({k:{a:b for a,b in v.items() if a!='metrics'} for k,v in result['upstream'].items()},indent=2))


if __name__=='__main__':
    main()
