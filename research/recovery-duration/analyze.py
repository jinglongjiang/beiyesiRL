"""Recompute paired outcomes and the frozen GO/KILL rule. No model execution."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE/'artifacts'
PROTO = json.loads((HERE/'protocol.json').read_text())
SCENES = PROTO['data']['scenes']
ARMS = ['nominal', 'ordinary', 'paired']
METRICS = ['success', 'collision', 'timeout', 'contact', 'time', 'discounted_return',
           'task_return', 'path', 'progress', 'final_distance', 'stall_fraction',
           'min_clearance', 'mean_decision_ms', 'interventions', 'overrides',
           'progress_efficiency', 'backtracking_ratio', 'low_progress_windows']


def mean_metrics(records):
    r = {k: float(np.mean([e[k] for e in records])) for k in METRICS}
    r['episodes'] = len(records)
    success = [e['time'] for e in records if e['success']]
    r['success_only_time'] = float(np.mean(success)) if success else None
    return r


def paired_compare(records, first, second, draws=10000):
    a = {(r['scene'], r['case']): r for r in records if r['arm']==first}
    b = {(r['scene'], r['case']): r for r in records if r['arm']==second}
    assert a.keys() == b.keys() and a
    keys = sorted(a)
    delta = np.array([[a[k][m]-b[k][m] for m in METRICS] for k in keys])
    rng = np.random.default_rng(20261011)
    # Case IDs share the simulator seed across scene configurations. Resample
    # the whole six-scene case block, preserving both arm and scene coupling.
    cases = sorted({k[1] for k in keys})
    blocks = np.stack([delta[[i for i, k in enumerate(keys) if k[1]==case]].mean(0)
                       for case in cases])
    chosen = rng.integers(len(cases), size=(draws, len(cases)))
    boot = blocks[chosen].mean(1)
    result = {'direction': f'{first} minus {second}', 'episodes': len(keys),
              'independent_case_seed_blocks': len(cases),
              'statistics': {m: {'difference': float(delta[:, j].mean()),
                               'ci95': np.percentile(boot[:, j], [2.5, 97.5]).tolist()}
                             for j, m in enumerate(METRICS)}}
    for name, cond in (
        ('rescues', lambda x, y: x['success'] and not y['success']),
        ('damages', lambda x, y: y['success'] and not x['success']),
        ('new_collisions', lambda x, y: x['collision'] and not y['collision']),
        ('new_contacts', lambda x, y: x['contact'] and not y['contact'])):
        result[name] = [{'scene': k[0], 'case': k[1], 'first': a[k]['outcome'], 'second': b[k]['outcome']}
                        for k in keys if cond(a[k], b[k])]
    result['outcome_matrix'] = {}
    for k in keys:
        pair = a[k]['outcome']+' / '+b[k]['outcome']
        result['outcome_matrix'][pair] = result['outcome_matrix'].get(pair, 0)+1
    return result


def main(prefix):
    records = json.loads((OUT/f'{prefix}-results.json').read_text())
    lo, hi = PROTO['fresh_cases_per_scene_if_signal' if prefix=='fresh' else 'development_cases_per_scene']
    expected = {(a, s, c) for a in ARMS for s in SCENES for c in range(lo, hi+1)}
    actual = {(r['arm'], r['scene'], r['case']) for r in records}
    assert len(records) == len(expected) and actual == expected, 'Incomplete/duplicate evaluation cohort'
    for row in records:
        scene = SCENES.index(row['scene'])
        trace = np.load(OUT/f'{prefix}-traces'/row['arm']/f'{scene}-{row["case"]}.npz')['records']
        goal = np.array([0., 6. if row['scene']=='large_circle' else 4.])
        positions = np.vstack([trace[:, 1:3], trace[-1, 1:3]+.25*trace[-1, 3:5]])
        distance = np.linalg.norm(positions-goal, axis=1)
        row['progress_efficiency'] = row['progress']/max(row['path'], 1e-9)
        row['backtracking_ratio'] = float((np.diff(distance)>.001).mean())
        row['low_progress_windows'] = float(((distance[:-8]-distance[8:])<.2).mean()) if len(distance)>8 else 0.
    grouped = {a: {s: mean_metrics([r for r in records if r['arm']==a and (s=='all' or r['scene']==s)])
                   for s in SCENES+['all']} for a in ARMS}
    comparisons = {f'{a}-vs-{b}': paired_compare(records, a, b)
                   for a, b in [('paired', 'nominal'), ('paired', 'ordinary'), ('ordinary', 'nominal')]}
    checks = {}
    for base in ('nominal', 'ordinary'):
        result = comparisons[f'paired-vs-{base}']['statistics']
        checks[base+'-sr'] = result['success']['difference'] >= .03-1e-10
        checks[base+'-return'] = result['discounted_return']['difference'] > 0
        checks[base+'-collision'] = result['collision']['difference'] <= 1e-10
        checks[base+'-contact'] = result['contact']['difference'] <= 1e-10
        checks[base+'-dense-scenes'] = sum(grouped['paired'][s]['success']>grouped[base][s]['success']
                                          for s in ['dense_circle', 'dense_square', 'large_square']) >= 2
        if prefix == 'fresh':
            checks[base+'-sr-ci'] = result['success']['ci95'][0] > 0
    signal = all(checks.values())
    data = {'cohort': prefix, 'groups': grouped, 'comparisons': comparisons, 'decision_checks': checks,
            'verdict': ('METHOD_ENTRY_FOUND' if prefix=='fresh' else 'METHOD_SIGNAL_FOUND') if signal
                       else ('CONFIRMATION_NEGATIVE' if prefix=='fresh' else 'VERSION_NEGATIVE'),
            'scientific_scope': 'case-paired outcome of frozen models; not training-seed inference'}
    data['descriptive_motion_definitions'] = {
        'progress_efficiency': 'net goal-distance reduction / executed path length',
        'backtracking_ratio': 'fraction of steps increasing goal distance by more than 1mm',
        'low_progress_windows': 'fraction of complete 8-step (2s) windows with <20cm net goal progress; descriptive, not a pathology label or acceptance criterion',
        'stall_fraction': 'fraction of executed steps with speed <0.1m/s'}
    activation = {}
    for arm in ARMS:
        counts = np.zeros(10, dtype=np.int64)
        changes = steps = episodes_changed = 0
        for trace_path in sorted((OUT/f'{prefix}-traces'/arm).glob('*.npz')):
            trace = np.load(trace_path)['records']
            changed = trace[:, 5] != trace[:, 6]
            changes += int(changed.sum())
            steps += len(trace)
            episodes_changed += int(changed.any())
            options = trace[trace[:, 9]>=0, 9].astype(int)
            counts += np.bincount(options, minlength=10)
        activation[arm] = {'steps': steps, 'executed_non_nominal_steps': changes,
                           'episodes_with_non_nominal_action': episodes_changed,
                           'option_decisions_0_to_9': counts.tolist()}
    data['activation'] = activation
    # Descriptive reporting only; these do not alter the frozen acceptance rule.
    data['outcome_motion_medians'] = {
        arm: {outcome: {
            'episodes': len(subset),
            **{metric: float(np.median([r[metric] for r in subset]))
               for metric in ['progress_efficiency', 'backtracking_ratio',
                              'low_progress_windows', 'path', 'final_distance']}}
              for outcome in ['reach_goal', 'collision', 'timeout']
              if (subset := [r for r in records if r['arm']==arm and r['outcome']==outcome])}
        for arm in ARMS}
    data['total_decision_seconds'] = {
        arm: float(sum(r['mean_decision_ms']*r['steps']/1000
                       for r in records if r['arm']==arm)) for arm in ARMS}
    label_path = OUT/'paired-results.json'
    if label_path.exists():
        labels = json.loads(label_path.read_text())
        roots = {}
        for row in labels:
            key = (row['scene'], row['case'], row['root'])
            roots.setdefault(key, {})[row['option']] = row
        classes = {'timeout_rescuable_by_catalog': [], 'nominal_success_harmed_by_catalog': [],
                   'no_catalog_success_from_nominal_failure': []}
        for key, options in roots.items():
            assert set(options)==set(range(10))
            nominal = options[0]
            if nominal['timeout'] and any(options[j]['success'] for j in range(1, 10)):
                classes['timeout_rescuable_by_catalog'].append(key)
            if nominal['success'] and any(not options[j]['success'] for j in range(1, 10)):
                classes['nominal_success_harmed_by_catalog'].append(key)
            if not nominal['success'] and not any(options[j]['success'] for j in range(1, 10)):
                classes['no_catalog_success_from_nominal_failure'].append(key)
        data['training_root_counterfactuals'] = {'root_count': len(roots), 'classes': classes,
            'interpretation': 'Descriptive catalog headroom on training roots, not deployed method improvement or independent episodes.'}
    path = HERE/f'{prefix}-summary.json'
    path.write_text(json.dumps(data, indent=2)+'\n')
    # Small complete CSV and exact rescue/damage lists are tracked; stepwise archives stay local.
    with (HERE/f'{prefix}-episodes.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    for arm in ARMS:
        g = grouped[arm]['all']
        print(arm, 'SR/CR/TO/contact', [round(g[k]*100, 3) for k in ['success', 'collision', 'timeout', 'contact']],
              'return', round(g['discounted_return'], 5))
    print(data['verdict'], checks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cohort', choices=['development', 'fresh'])
    main(parser.parse_args().cohort)
