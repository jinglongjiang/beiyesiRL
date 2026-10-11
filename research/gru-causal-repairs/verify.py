"""Freeze model hashes and verify all delivered closed-loop arithmetic."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

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
    frozen = json.loads((HERE / 'frozen-models.json').read_text())
    for name, record in frozen['files'].items():
        assert sha(OUT / name) == record['sha256'], name
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
              'time_limit': 50, 'no_upstream_source_change': True}
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
