"""Validate saved evidence and pin it; never execute or train a policy."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'recovery-duration'


def digest(path):
    value = hashlib.sha256(path.read_bytes()).hexdigest()
    assert value
    return value


def read(path):
    def invalid(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    return json.loads(path.read_text(), parse_constant=invalid)


def main():
    baseline = read(OLD / 'frozen-manifest.json')
    assets = {}
    for filename, key in [('nominal.pth', 'nominal_sha256'),
                          ('ordinary.pth', 'ordinary_checkpoint_sha256'),
                          ('paired.pth', 'paired_checkpoint_sha256')]:
        path = OLD / 'artifacts' / filename
        sha = digest(path)
        assert sha == baseline[key], filename
        assets[str(path.relative_to(HERE.parent))] = sha
    assert digest(OLD / 'parent-config.ini') == baseline['configuration_sha256']
    frozen = read(HERE / 'counterfactual-protocol.json')
    assert digest(HERE / 'nominal-ledger.json') == frozen['source_ledger_sha256']
    ledger = read(HERE / 'nominal-ledger.json')
    scores = np.load(HERE / 'artifacts' / 'nominal-action-scores.npz')['scores']
    assert scores.shape == (len(ledger), 80, 9) and len(ledger) == 1949
    assert np.all(np.isfinite(scores))
    for row, actions in zip(ledger, scores):
        assert actions[:, 7].argmax() == row['chosen_index']
    rows = read(HERE / 'counterfactual-results.json')
    assert len(rows) == 2 * len(frozen['roots']) == 328
    roots = {(r['scene'], r['case']): r for r in frozen['roots']}
    scenes = ['baseline_circle', 'baseline_square', 'dense_circle',
              'dense_square', 'large_circle', 'large_square']
    seen = set()
    for row in rows:
        root = roots[(row['scene'], row['case'])]
        assert root['step'] == row['step']
        key = (row['scene'], row['case'], row['variant'])
        assert key not in seen
        seen.add(key)
        name = 'counterfactual-{}-{}-{}.npz'.format(row['variant'], scenes.index(row['scene']), row['case'])
        data = np.load(HERE / 'artifacts' / name)
        trace, choices = data['records'], data['choices']
        assert len(trace) == row['result']['steps'] == len(choices)
        assert np.all(np.isfinite(trace))
        assert trace[0, 0] == row['step']
        assert np.array_equal(trace[:, 0], choices[:, 0])
        assert np.array_equal(trace[:, 5], choices[:, 2])
        assert int((trace[:, 8] < 0).any()) == row['result']['contact']
        actual_return = float(trace[:, 7] @ (.99 ** np.arange(len(trace))))
        assert abs(actual_return - row['result']['discounted_return']) < 1e-10
        if row['variant'] == 'one-step-progress':
            assert np.array_equal(choices[1:, 1], choices[1:, 2])
    hashes = {}
    for path in sorted(HERE.rglob('*')):
        if not path.is_file() or path.name == 'execution-manifest.json' or '__pycache__' in path.parts:
            continue
        if path.suffix == '.json':
            read(path)
        hashes[str(path.relative_to(HERE))] = digest(path)
    manifest = {
        'verified_utc': datetime.now(timezone.utc).isoformat(),
        'source_baseline_commit': '15086e9',
        'original_project_modified': False,
        'training_executed': False,
        'recovery_repair_or_retraining': False,
        'nominal_episodes_replayed': 240,
        'nominal_scored_roots': len(ledger),
        'one_shot_recovery_continuations': 24,
        'nominal_counterfactual_continuations': len(rows),
        'recovery_contact_roots': 44,
        'baseline_checkpoint_sha256': assets,
        'baseline_config_sha256': baseline['configuration_sha256'],
        'checks': 'JSON finite; checkpoint/config/ledger SHA unchanged; root list and all328 trace lengths, choices, contacts and returns consistent; one-step control returns to nominal after first step',
        'files_sha256': hashes,
    }
    (HERE / 'execution-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('VERIFIED: 1949 roots, 328 complete continuations, 3 unchanged checkpoints, {} evidence files'.format(len(hashes)))


if __name__ == '__main__':
    main()
