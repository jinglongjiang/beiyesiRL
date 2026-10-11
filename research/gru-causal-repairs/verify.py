"""Freeze model hashes and verify all delivered closed-loop arithmetic."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
OUT = HERE / 'artifacts'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def freeze():
    names = ['behavior-mc.pth', 'deploy-mc.pth', 'matched-data.npz', 'batch-indices.npy']
    files = {name: {'sha256': sha(OUT / name), 'bytes': (OUT / name).stat().st_size} for name in names}
    save('frozen-models.json', {'files': files, 'fresh_cases': [300, 319],
                               'all_parameters_frozen_before_fresh_outcomes': True})


def verify():
    assert (HERE / 'pipeline-complete.json').exists(), 'Do not verify an incomplete run'
    frozen = json.loads((HERE / 'frozen-models.json').read_text())
    for name, record in frozen['files'].items():
        assert sha(OUT / name) == record['sha256'], name
    initial = HERE.parent / 'recovery-duration/artifacts/nominal.pth'
    protocol = json.loads((HERE / 'protocol.json').read_text())
    assert sha(initial) == protocol['parent_sha256']
    root = HERE.parents[1]
    gitdir = Path((root / '.git').read_text().strip().split(': ', 1)[1])
    if not gitdir.exists():
        gitdir = Path('/workspace') / gitdir.relative_to('/home/abc/workspace')
    assert gitdir.is_dir(), 'Worktree Git metadata unavailable'
    upstream = subprocess.run(['git', f'--git-dir={gitdir}', f'--work-tree={root}',
                               '-c', f'safe.directory={root}', 'diff', protocol['parent_commit'],
                               '--', 'crowd_nav', 'crowd_sim'],
                              cwd=root, capture_output=True, text=True)
    assert upstream.returncode == 0, upstream.stderr
    assert not upstream.stdout, 'Upstream source changed'
    training = json.loads((HERE / 'training.json').read_text())
    for arm, record in training['arms'].items():
        assert sha(OUT / f'{arm}.pth') == record['sha256']
    conditional = json.loads((HERE / 'coverage-training.json').read_text())
    for arm, record in conditional.items():
        assert sha(OUT / f'{arm}.pth') == record['sha256']
    parent_state = torch.load(initial, map_location='cpu')
    head_state = torch.load(OUT / 'targeted-head.pth', map_location='cpu')
    assert parent_state.keys() == head_state.keys()
    changed = [k for k in parent_state if not torch.equal(parent_state[k], head_state[k])]
    assert changed and all(k.startswith('value_head.') for k in changed), changed
    np.testing.assert_array_equal(np.load(OUT / 'targeted-head-batch-indices.npy'),
                                  np.load(OUT / 'targeted-full-batch-indices.npy'))
    common = np.load(OUT / 'matched-data.npz')
    assert len(common['windows']) == 337 and int((common['case'] < 2012).sum()) == 255
    for cohort, arms in [('fresh', ['nominal', 'behavior-mc', 'deploy-mc', 'progress', 'commit8', 'adaptive']),
                        ('coverage-fresh', ['nominal', 'coverage-control', 'targeted-head', 'targeted-full'])]:
        expected = {(s, c) for s in ['baseline_circle', 'baseline_square', 'dense_circle',
                                    'dense_square', 'large_circle', 'large_square']
                    for c in range(300, 320) if cohort == 'fresh'}
        if cohort != 'fresh':
            expected = {(s, c) for s in ['baseline_circle', 'baseline_square', 'dense_circle',
                                         'dense_square', 'large_circle', 'large_square'] for c in range(320, 340)}
        for arm in arms:
            rows = json.loads((HERE / f'{cohort}-{arm}.json').read_text())['results']
            assert len(rows) == 120 and {(v['scene'], v['case']) for v in rows} == expected
    counts = {}
    trajectories = steps = 0
    for path in OUT.rglob('*.npz'):
        data = np.load(path)
        if 'records' not in data:
            continue
        meta = path.with_suffix('.json')
        if not meta.exists():
            continue
        row = json.loads(meta.read_text())
        records = data['records']
        assert len(records) == row['steps'] and records.shape[1] == 12, str(path)
        assert np.isfinite(records).all(), str(path)
        assert records[-1, 0] <= 200
        assert row['success'] + row['collision'] + row['timeout'] == 1
        assert row['contact'] == int((records[:, 8] < 0).any())
        assert row['duration'] == .25 * len(records)
        np.testing.assert_allclose(row['return'], records[:, 7] @ (.99 ** np.arange(len(records))), rtol=0, atol=1e-12)
        arrays = data['candidates']
        assert arrays.shape == (len(records), 6, 80)
        selected = records[:, 5].astype(int)
        mask = arrays[:, 5].astype(bool)
        assert np.all(mask[np.arange(len(mask)), selected] | ~mask.any(1)), str(path)
        cohort = path.relative_to(OUT).parts[0]
        counts[cohort] = counts.get(cohort, 0) + 1
        trajectories += 1
        steps += len(records)
    for name in ['fresh', 'roots']:
        summary = json.loads((HERE / f'{name}-summary.json').read_text())
        for arm, values in summary['overall'].items():
            assert values['n'] == (120 if name == 'fresh' else 35)
    checks = {'status': 'PASS', 'trajectories': trajectories, 'executed_steps': steps,
              'cohort_counts': counts, 'frozen_models_unchanged': True, 'actions': 80,
              'time_limit': 50, 'no_upstream_source_change': True,
              'conditional_model_hashes_verified': True, 'paired_targeted_batches_equal': True,
              'targeted_head_only_changed_keys': changed,
              'all_fresh_cases_complete': True, 'parent_sha256': sha(initial)}
    save('verification.json', checks)
    files = []
    for path in sorted(HERE.rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts and path.name != 'evidence-manifest.json':
            files.append({'path': str(path.relative_to(HERE)), 'bytes': path.stat().st_size, 'sha256': sha(path)})
    save('evidence-manifest.json', {'verification': checks, 'files': files})
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['freeze', 'verify'])
    args = parser.parse_args()
    globals()[args.stage]()
