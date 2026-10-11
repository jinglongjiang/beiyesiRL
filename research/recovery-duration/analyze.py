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
           'min_clearance', 'mean_decision_ms', 'interventions', 'overrides']


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
    boot = np.zeros((draws, len(METRICS)))
    for scene in SCENES:
        idx = [i for i, k in enumerate(keys) if k[0]==scene]
        chosen = rng.choice(idx, size=(draws, len(idx)), replace=True)
        boot += delta[chosen].sum(1)/len(keys)
    result = {'direction': f'{first} minus {second}', 'episodes': len(keys),
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
