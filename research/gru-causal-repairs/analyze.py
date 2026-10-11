"""Case-paired summaries; candidate and timestep counts are not sample sizes."""
import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'gru-complete-actions/artifacts/v2-50s'


def write(name, data):
    (HERE / name).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def aggregate(rows):
    success = [v for v in rows if v['success']]
    return {'n': len(rows), **{k: sum(v[k] for v in rows) for k in ['success', 'collision', 'timeout', 'contact']},
            'SR': float(np.mean([v['success'] for v in rows])),
            'CR': float(np.mean([v['collision'] for v in rows])),
            'TR': float(np.mean([v['timeout'] for v in rows])),
            'duration_all_mean': float(np.mean([v['duration'] for v in rows])),
            'successful_time_mean': float(np.mean([v['duration'] for v in success])) if success else None,
            'return_mean': float(np.mean([v['return'] for v in rows])),
            'min_clearance_min': float(min(v['min_clearance'] for v in rows)),
            'min_clearance_median': float(np.median([v['min_clearance'] for v in rows])),
            'triggers': sum(v.get('triggers', 0) for v in rows),
            'intervention_steps': sum(v.get('intervention_steps', 0) for v in rows),
            'fallback_steps': sum(v.get('fallback', 0) for v in rows),
            'decision_ms_episode_mean': float(np.mean([v.get('decision_ms_mean', 0) for v in rows]))}


def pairs(first, second):
    left = {(v['scene'], v['case']): v for v in first}
    right = {(v['scene'], v['case']): v for v in second}
    assert left.keys() == right.keys()
    keys = sorted(left)
    case_ids = sorted({k[1] for k in keys})
    rng = np.random.default_rng(121012)
    draw = rng.integers(len(case_ids), size=(4000, len(case_ids)))
    counts = np.array([sum(k[1] == c for k in keys) for c in case_ids])
    delta = np.array([sum(left[k]['success'] - right[k]['success'] for k in keys if k[1] == c) for c in case_ids])
    contact = np.array([sum(left[k]['contact'] - right[k]['contact'] for k in keys if k[1] == c) for c in case_ids])
    rescues = [list(k) for k in keys if left[k]['success'] and not right[k]['success']]
    damages = [list(k) for k in keys if right[k]['success'] and not left[k]['success']]
    both = [k for k in keys if left[k]['success'] and right[k]['success']]
    return {'episodes': len(keys), 'case_clusters': len(case_ids),
            'delta_SR': float(delta.sum() / counts.sum()), 'delta_SR_case_bootstrap_95': np.quantile(delta[draw].sum(1) / counts[draw].sum(1), [.025, .975]).tolist(),
            'delta_contact_rate': float(contact.sum() / counts.sum()),
            'delta_contact_case_bootstrap_95': np.quantile(contact[draw].sum(1) / counts[draw].sum(1), [.025, .975]).tolist(),
            'rescues': rescues, 'damages': damages,
            'rescue_count': len(rescues), 'damage_count': len(damages),
            'both_success_time_delta_mean': float(np.mean([left[k]['duration'] - right[k]['duration'] for k in both])) if both else None}


def summarize(cohort):
    arms = ['nominal', 'behavior-mc', 'deploy-mc', 'progress', 'commit8', 'adaptive']
    rows = {a: json.loads((HERE / f'{cohort}-{a}.json').read_text())['results'] for a in arms}
    output = {'cohort': cohort, 'overall': {a: aggregate(v) for a, v in rows.items()}, 'pairwise': {}}
    for a, b in [('behavior-mc', 'nominal'), ('deploy-mc', 'nominal'), ('deploy-mc', 'behavior-mc'),
                 ('progress', 'nominal'), ('commit8', 'nominal'), ('adaptive', 'nominal'),
                 ('adaptive', 'commit8'), ('adaptive', 'progress')]:
        output['pairwise'][f'{a}-vs-{b}'] = pairs(rows[a], rows[b])
    output['per_scene'] = {scene: {a: aggregate([v for v in rs if v['scene'] == scene]) for a, rs in rows.items()}
                           for scene in sorted({v['scene'] for v in rows['nominal']})}
    if cohort == 'roots':
        output['mechanism_groups'] = {}
        for cat in sorted({v['category'] for v in rows['nominal']}):
            output['mechanism_groups'][cat] = {a: aggregate([v for v in rs if v['category'] == cat]) for a, rs in rows.items()}
    write(f'{cohort}-summary.json', output)
    print(json.dumps(output['overall'], indent=2), flush=True)


def ranking():
    data = json.loads((HERE / 'ranking-probes.json').read_text())
    stats = json.loads((OLD / 'statistics.json').read_text())
    categories = {(v['scene'], v['case']): v['category'] for v in stats['cases']}
    result = {}
    for arm, rows in data.items():
        result[arm] = {}
        for name, group in [('all', rows), ('A10', [v for v in rows if categories[(v['root']['scene'], v['root']['case'])] == 'VALUE_RANKING_FAILURE_FIXED_WORLD']),
                            ('normal', [v for v in rows if v['root']['group'] == 'successful_controls'])]:
            result[arm][name] = {'n': len(group), 'mean_regret': float(np.mean([v['regret_fixed_old_policy'] for v in group])),
                                 'first_then_original_safe_success': sum(v['old_policy_first_action_result']['success'] and not v['old_policy_first_action_result']['contact'] for v in group),
                                 'mean_inversion': float(np.mean([v['pair_inversions'] / max(v['nontied_pairs'], 1) for v in group]))}
    write('ranking-summary.json', result)
    print(json.dumps(result, indent=2))


def failure():
    probes = json.loads((HERE / 'ranking-probes.json').read_text())
    result = {}
    for arm in ['behavior-mc', 'deploy-mc']:
        actual = json.loads((HERE / f'roots-{arm}.json').read_text())['results']
        index = {(v['root']['scene'], v['root']['case']): v for v in probes[arm]}
        rows = []
        for row in actual:
            k = (row['scene'], row['case'])
            first = index[k]
            path = HERE / 'artifacts/roots' / arm / f'{["baseline_circle","baseline_square","dense_circle","dense_square","large_circle","large_square"].index(k[0])}-{k[1]}.npz'
            data = np.load(path)
            trace = data['records']
            assert int(trace[0, 5]) == first['chosen']
            before = first['old_policy_first_action_result']
            rows.append({'scene': k[0], 'case': k[1], 'category': row['category'],
                         'first_action_old_policy_safe_success': bool(before['success'] and not before['contact']),
                         'new_policy_safe_success': bool(row['success'] and not row['contact']),
                         'nonpositive_progress_fraction': float((trace[:, 9] <= 0).mean()),
                         'nominal_veto_fraction': float((trace[:, 5] != trace[:, 6]).mean()),
                         'fallback_steps': row['fallback'], 'outcome': row['outcome'],
                         'final_distance': row['final_distance'], 'duration': row['duration']})
        result[arm] = {'rows': rows,
                      'old_policy_safe_success_lost_under_new_continuation': sum(v['first_action_old_policy_safe_success'] and not v['new_policy_safe_success'] for v in rows),
                      'new_policy_success_absent_under_old_continuation': sum(not v['first_action_old_policy_safe_success'] and v['new_policy_safe_success'] for v in rows),
                      'scope': 'Same forced first command/world, later frozen continuation policy differs. Identifies post-first-action policy degradation, not its unique historical training cause.'}
    write('negative-mechanism.json', result)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'rows'} for k, v in result.items()}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['roots', 'fresh', 'ranking', 'failure'])
    args = parser.parse_args()
    if args.stage == 'failure':
        failure()
    elif args.stage == 'ranking':
        ranking()
    else:
        summarize(args.stage)
