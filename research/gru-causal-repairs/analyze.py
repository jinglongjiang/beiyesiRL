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


def frontier():
    evidence = json.loads((HERE / 'B-evidence.json').read_text())['roots']
    prefixes = json.loads((OLD / 'prefixes.json').read_text())
    records = []
    names = ['baseline_circle', 'baseline_square', 'dense_circle', 'dense_square', 'large_circle', 'large_square']
    for item in evidence:
        root = item['root']
        key = f'{names.index(root["scene"])}-{root["case"]}'
        first_index = item['steps'][0]['witness']
        first = json.loads((OLD / 'catalogue' / key / f'action-{first_index:02d}-A_deploy.json').read_text())
        assert not first['result']['success']
        rows = [v for v in prefixes if v['root']['scene'] == root['scene'] and v['root']['case'] == root['case']]
        successes = sorted(v['prefix'] for v in rows if v['success'] and not v['contact'])
        result = {'root': root, 'prefix1_outcome': first['result']['outcome'],
                  'saved_prefix_outcomes': {str(v['prefix']): v['outcome'] for v in rows},
                  'root_relative_not_architecture_impossibility': True}
        if 2 in successes:
            two = next(v for v in rows if v['prefix'] == 2)
            before = np.load(OLD / 'catalogue' / key / f'action-{first_index:02d}-A_deploy.npz')['records']
            after = np.load(OLD / 'prefixes' / f'{key}-2.npz')['records']
            np.testing.assert_allclose(before[0, :6], after[0, :6], rtol=0, atol=1e-12)
            np.testing.assert_allclose(before[1, :3], after[1, :3], rtol=0, atol=1e-12)
            step = item['steps'][1]
            assert int(before[1, 5]) == step['nominal'] and int(after[1, 5]) == step['witness']
            result['confirmed_downstream_single_action_misrank'] = {
                'step': int(after[1, 0]), 'nominal': step['nominal'], 'alternative': step['witness'],
                'predicted_nominal_Q_advantage': step['nominal_Q_advantage'],
                'actual_nominal_return_from_child': (first['result']['return'] - before[0, 7]) / .99,
                'actual_alternative_return_from_child': (two['return'] - after[0, 7]) / .99,
                'nominal_outcome': first['result']['outcome'], 'alternative_outcome': two['outcome'],
                'alternative_contact': two['contact'], 'same_child_physical_state_and_original_prefix_history': True}
            assert step['nominal_Q_advantage'] > 0 and two['contact'] == 0
        else:
            four = next(v for v in rows if v['prefix'] == 4)
            eight = next(v for v in rows if v['prefix'] == 8)
            assert not four['success'] and eight['success'] and not eight['contact']
            x = np.load(OLD / 'prefixes' / f'{key}-4.npz')['records']
            y = np.load(OLD / 'prefixes' / f'{key}-8.npz')['records']
            np.testing.assert_allclose(x[:4, :6], y[:4, :6], rtol=0, atol=1e-12)
            result['downstream_transition_bound'] = {'prefix4': 'failure', 'prefix8': 'safe success',
                'failure_to_success_extension_exists_between_lengths': [5, 8],
                'exact_transition_step': 'UNRESOLVED; not simulated',
                'reason': 'With identical deterministic world, history and continuation, a finite failure-to-success prefix extension entails at least one downstream state where a single alternative followed by the same GRU improves the outcome. This does not identify its historical training cause.'}
        records.append(result)
    write('B-causal-frontier.json', {'records': records,
          'no_new_simulation': True,
          'conclusion': 'Initial-root multistep trap is not proof of an inherently sequence-only representational defect; at least one is directly a downstream single-action value misrank.'})
    print(json.dumps(records, indent=2))


def coverage():
    names = ['coverage-control', 'targeted-head', 'targeted-full']
    result = {}
    for cohort in ['roots', 'fresh']:
        rows = {a: json.loads((HERE / f'coverage-{cohort}-{a}.json').read_text())['results'] for a in names}
        reference = 'roots-nominal.json' if cohort == 'roots' else 'coverage-fresh-nominal.json'
        rows['nominal'] = json.loads((HERE / reference).read_text())['results']
        out = {'overall': {a: aggregate(v) for a, v in rows.items()},
               'pairwise': {f'{a}-vs-{b}': pairs(rows[a], rows[b]) for a, b in
                            [('coverage-control', 'nominal'), ('targeted-head', 'nominal'),
                             ('targeted-full', 'nominal'), ('targeted-full', 'coverage-control'),
                             ('targeted-full', 'targeted-head')]},
               'per_scene': {s: {a: aggregate([v for v in rs if v['scene'] == s]) for a, rs in rows.items()}
                             for s in sorted({v['scene'] for v in rows['nominal']})}}
        if cohort == 'roots':
            out['mechanism_groups'] = {c: {a: aggregate([v for v in rs if v['category'] == c])
                                          for a, rs in rows.items()}
                                       for c in sorted({v['category'] for v in rows['nominal']})}
        result[cohort] = out
    probes = json.loads((HERE / 'coverage-ranking-probes.json').read_text())
    stats = json.loads((OLD / 'statistics.json').read_text())['cases']
    active = {(v['scene'], v['case']): v for v in stats
              if v['category'] != 'NO_CONTACT_FREE_RESCUE_IN_TESTED_SINGLE_OR_PREFIX_ACTIONS'}
    result['ranking'] = {}
    for arm, records in probes.items():
        result['ranking'][arm] = {}
        for split in ['even-fit', 'odd-heldout']:
            subset = [v for v in records if (v['root']['scene'], v['root']['case']) in active
                      and (v['root']['case'] % 2 == 0) == (split == 'even-fit')]
            result['ranking'][arm][split] = {'n': len(subset),
                'fixed_old_policy_safe_success': sum(v['old_policy_first_action_result']['success']
                                                     and not v['old_policy_first_action_result']['contact'] for v in subset),
                'mean_regret': float(np.mean([v['regret_fixed_old_policy'] for v in subset])),
                'actual_next_V_MAE': float(np.mean([v['actual_next_V_MAE'] for v in subset]))}
    write('coverage-summary.json', result)
    print(json.dumps({c: v['overall'] for c, v in result.items() if c != 'ranking'}, indent=2))


def controllers():
    result = {}
    for arm in ['commit8', 'adaptive']:
        episodes = json.loads((HERE / f'fresh-{arm}.json').read_text())['results']
        rows = []
        for episode in episodes:
            scene, case = episode['scene'], episode['case']
            path = HERE / 'artifacts/fresh' / arm / f'{["baseline_circle", "baseline_square", "dense_circle", "dense_square", "large_circle", "large_square"].index(scene)}-{case}.npz'
            trace = np.load(path)['records']
            rows.append({'scene': scene, 'case': case, 'outcome': episode['outcome'],
                         'triggers': episode['triggers'], 'intervention_steps': episode['intervention_steps'],
                         'nonpositive_executed_progress_fraction': float((trace[:, 9] <= 0).mean()),
                         'nonpositive_nominal_progress_fraction': float((trace[:, 10] <= 0).mean()),
                         'selected_differs_nominal_steps': int((trace[:, 5] != trace[:, 6]).sum()),
                         'final_distance': episode['final_distance']})
        result[arm] = {'rows': rows,
                       'scope': 'Descriptive activation and failure accounting only; divergent trajectories do not identify a unique causal reason for the duration difference.',
                       'triggered_episodes': sum(v['triggers'] > 0 for v in rows),
                       'triggered_timeouts': sum(v['triggers'] > 0 and v['outcome'] == 'timeout' for v in rows)}
    write('controller-accounting.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['roots', 'fresh', 'ranking', 'failure', 'frontier', 'coverage', 'controllers'])
    args = parser.parse_args()
    if args.stage in ('coverage', 'controllers'):
        globals()[args.stage]()
    elif args.stage == 'frontier':
        frontier()
    elif args.stage == 'failure':
        failure()
    elif args.stage == 'ranking':
        ranking()
    else:
        summarize(args.stage)
