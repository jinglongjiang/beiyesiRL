"""Analyze completed saved benchmarks; no model loading or experiment launches."""

import collections
import csv
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'crowd_nav/runs/bayes-fix-20261008'
REMOTE = BASE / 'remote-run'
ARMS = ('gru', 'bayes_mean', 'bayes')
SEEDS = (42, 43)
METRICS = ('SR', 'CR', 'TR', 'geometric_contact')
BOOTSTRAPS = 10000
BOOTSTRAP_SEED = 20261008


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def case_key(row):
    return row['scenario'], int(row['episode']), int(row['seed'])


def load(arm, seed):
    path = REMOTE / 'runs' / f'formal-{arm}-seed{seed}' / 'benchmark-10000.csv'
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = sorted(reader, key=case_key)
    assert len(rows) == len({case_key(r) for r in rows}) == 3000, path
    scenarios = collections.Counter(r['scenario'] for r in rows)
    assert len(scenarios) == 6 and set(scenarios.values()) == {500}, path
    assert all(r['outcome'] in ('success', 'collision', 'timeout') for r in rows)
    numeric = set(fields) - {'run_label', 'scenario', 'outcome'}
    assert all(np.isfinite(float(r[k])) for r in rows for k in numeric), path
    assert all(int(r['steps']) > 0 for r in rows)
    assert all(r['run_label'] == f'{arm}-fix-seed{seed}' for r in rows)
    return rows, path


def aggregate(rows):
    success = [r for r in rows if r['outcome'] == 'success']
    mean = lambda field, subset=rows: float(np.mean([float(r[field]) for r in subset])) if subset else None
    steps = sum(int(r['steps']) for r in rows)
    counts = collections.Counter(r['outcome'] for r in rows)
    contacts = sum(int(r['geometric_contact_steps']) > 0 for r in rows)
    return {
        'n': len(rows), 'success': counts['success'], 'collision': counts['collision'],
        'timeout': counts['timeout'], 'contact_episodes': contacts,
        'rates_percent': dict(zip(METRICS, [100 * x / len(rows) for x in
                             (counts['success'], counts['collision'], counts['timeout'], contacts)])),
        'total_steps': steps,
        'contact_steps': sum(int(r['geometric_contact_steps']) for r in rows),
        'contact_duration_s': sum(float(r['geometric_contact_duration_s']) for r in rows),
        'discomfort_step_fraction': sum(int(r['discomfort_steps']) for r in rows) / steps,
        'minimum_clearance_m': min(float(r['min_clearance_m']) for r in rows),
        'episode_mean_minimum_clearance_m': mean('min_clearance_m'),
        'success_time_s': mean('duration_s', success),
        'success_path_m': mean('path_length_m', success),
        'all_episode_duration_s': mean('duration_s'),
        'all_episode_path_m': mean('path_length_m'),
        'inference_step_weighted_ms': sum(float(r['mean_inference_ms']) * int(r['steps']) for r in rows) / steps,
        'mean_episode_inference_p95_ms': mean('p95_inference_ms'),
        'mean_episode_stall_fraction': mean('stalled_window_ratio'),
        'mean_episode_oscillatory_stall_fraction': mean('oscillatory_stall_window_ratio'),
        'mean_episode_progress_efficiency': mean('goal_progress_efficiency'),
        'mean_episode_heading_flip_hz': mean('heading_flip_hz'),
        'mean_episode_total_turn_rad': mean('total_abs_turn_rad'),
        'mean_episode_curvature_rad_per_m': mean('curvature_rad_per_m'),
    }


def outcomes(rows):
    return np.array([[r['outcome'] == 'success', r['outcome'] == 'collision',
                      r['outcome'] == 'timeout', int(r['geometric_contact_steps']) > 0]
                     for r in rows], dtype=float)


def interval(diff, rows):
    # Seeds share the same evaluation worlds: averaged-seed differences are
    # resampled by CASE, not by independently resampling 6000 observations.
    groups = [np.array([i for i, r in enumerate(rows) if r['scenario'] == s])
              for s in sorted({r['scenario'] for r in rows})]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = []
    for _ in range(BOOTSTRAPS // 250):
        batch = np.zeros((250, len(METRICS)))
        for ids in groups:
            batch += diff[ids][rng.integers(len(ids), size=(250, len(ids)))].mean(axis=1) / len(groups)
        samples.append(batch)
    return dict(zip(METRICS, (100 * np.percentile(np.concatenate(samples), [2.5, 97.5], axis=0).T).tolist()))


def compare(left, right):
    assert [case_key(r) for r in left] == [case_key(r) for r in right]
    diff = outcomes(left) - outcomes(right)
    common = [(l, r) for l, r in zip(left, right) if l['outcome'] == r['outcome'] == 'success']
    common_differences = {
        field: float(np.mean([float(l[field]) - float(r[field]) for l, r in common])) if common else None
        for field in ('duration_s', 'path_length_m', 'min_clearance_m', 'stalled_window_ratio',
                      'heading_flip_hz', 'total_abs_turn_rad', 'curvature_rad_per_m')
    }
    return {
        'difference_pp': dict(zip(METRICS, (100 * diff.mean(axis=0)).tolist())),
        'exploratory_fixed_model_paired_case_95_ci_pp': interval(diff, left),
        'left_only_success': sum(l['outcome'] == 'success' and r['outcome'] != 'success' for l, r in zip(left, right)),
        'right_only_success': sum(r['outcome'] == 'success' and l['outcome'] != 'success' for l, r in zip(left, right)),
        'common_success': len(common), 'common_success_left_minus_right': common_differences,
        'outcome_transition_left_to_right': dict(collections.Counter(l['outcome'] + '->' + r['outcome'] for l, r in zip(left, right))),
    }


def main():
    complete = read_json(REMOTE / 'complete.json')
    assert complete['status'] == 'COMPLETE'
    assert not (REMOTE / 'error.json').exists()
    trained = read_json(REMOTE / 'training-results.json')
    training = {(r['arm'], r['seed']): r for r in trained}
    evaluated = {(r['arm'], r['seed']): r for r in complete['results']}
    expected = {(a, s) for a in ARMS for s in SEEDS}
    assert len(trained) == len(complete['results']) == 6
    assert set(training) == set(evaluated) == expected
    version = read_json(REMOTE / 'formal-version.json')
    for name, sha in version['source_sha256'].items():
        assert digest(ROOT / name) == sha, f'Frozen production source changed: {name}'
    for name, sha in version['configs_sha256'].items():
        assert digest(REMOTE / 'configs' / name) == sha, f'Frozen config changed: {name}'
    result = {
        'analysis_version': 'completed-logvariance-readout-v1',
        'analyzed_at_kst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),
        'completion': 'all six 10000-RL/40000-update runs and six 3000-case benchmarks complete',
        'source_files_verified': len(version['source_sha256']),
        'configs_verified': len(version['configs_sha256']),
        'bootstrap_replicates': BOOTSTRAPS, 'bootstrap_seed': BOOTSTRAP_SEED,
        'inference_limits': [
            'Two online RL seeds share each arm IL seed42; not two independent IL+RL seeds.',
            'Both RL seeds use the same 3000 native evaluation worlds; 6000 records are not 6000 unique worlds.',
            'Intervals describe paired evaluation-case variation conditional on trained models, not training-seed population uncertainty.',
            'Exploratory intervals have no multiple-comparison correction and are not fresh confirmation.',
            'Mean still uses variance in filtering; comparison isolates explicit readout variance channels.',
            'Geometric contacts are sampled-state radius overlaps, not continuous swept collision checks.',
            'Success time/path metrics are conditional on success; all-outcome durations include early collision termination.',
            'No per-decision fallback counter exists in these benchmark CSVs; do not infer zero fallback.',
            'Compute timings contain shared-device contention and are not isolated deployment measurements.',
        ],
        'cohorts': {}, 'descriptive_pooled': {}, 'pairs': {}, 'pooled_paired_cases': {},
        'raw_sha256': {str(p.relative_to(ROOT)): digest(p) for p in
                       (REMOTE / 'complete.json', REMOTE / 'training-results.json', REMOTE / 'formal-version.json')},
    }
    saved = {}
    for seed in SEEDS:
        result['cohorts'][str(seed)] = {}
        for arm in ARMS:
            rows, path = load(arm, seed)
            saved[arm, seed] = rows
            summary = aggregate(rows)
            t, e = training[arm, seed], evaluated[arm, seed]
            assert t['status'] == 'TRAINED' and t['episodes'] == 10000 and t['optimizer_steps'] == 40000
            assert t['success'] + t['collision'] + t['timeout'] == 10000
            assert e == read_json(path.parent / 'benchmark-result.json') and e['status'] == 'EVALUATED'
            for key in ('success', 'collision', 'timeout'):
                assert summary[key] == e[key]
            assert summary['contact_episodes'] == e['geometric_contact_episodes']
            result['raw_sha256'][str(path.relative_to(ROOT))] = digest(path)
            result['cohorts'][str(seed)][arm] = {
                'aggregate': summary, 'training': t, 'evaluation_seconds': e['seconds'],
                'scenarios': {s: aggregate([r for r in rows if r['scenario'] == s]) for s in sorted({r['scenario'] for r in rows})},
            }
    keys = [case_key(r) for r in saved['gru', 42]]
    assert all([case_key(r) for r in rows] == keys for rows in saved.values())
    for arm in ARMS:
        result['descriptive_pooled'][arm] = aggregate(saved[arm, 42] + saved[arm, 43])
    for left, right in (('bayes', 'bayes_mean'), ('bayes', 'gru'), ('bayes_mean', 'gru')):
        name = f'{left}_minus_{right}'
        result['pairs'][name] = {str(seed): compare(saved[left, seed], saved[right, seed]) for seed in SEEDS}
        diff = np.mean([outcomes(saved[left, s]) - outcomes(saved[right, s]) for s in SEEDS], axis=0)
        result['pooled_paired_cases'][name] = {
            'difference_pp': dict(zip(METRICS, (100 * diff.mean(axis=0)).tolist())),
            'exploratory_fixed_models_shared_case_95_ci_pp': interval(diff, saved[left, 42]),
        }
    path = BASE / 'final-analysis.json'
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ('completion', 'source_files_verified', 'configs_verified', 'pooled_paired_cases')}, indent=2))
    print(path)


if __name__ == '__main__':
    main()
