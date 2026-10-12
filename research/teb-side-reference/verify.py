"""Recompute outcome, safety and direction-memory invariants from saved traces."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location('paired_reference_runner', HERE / 'run.py')
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)


def verify(cohort='development'):
    run.verify_manifest()
    rows = []
    reused = 0
    if cohort == 'development':
        for arm in ['nominal', 'commit8']:
            baseline = json.loads((HERE / f'{cohort}-{arm}.json').read_text())['results']
            assert len(baseline) == 120
            for item in baseline:
                source = ROOT / item['source']
                assert item['reused'] and run.sha(source) == item['source_sha256']
                original = json.loads(source.read_text())
                assert all(item[key] == value for key, value in original.items())
                reused += 1
    for arm in ['side-reset', 'side-latched']:
        results = json.loads((HERE / f'{cohort}-{arm}.json').read_text())['results']
        assert len(results) == 120
        assert len({(v['scene'], v['case']) for v in results}) == 120
        for item in results:
            scene = run.m.SCENES.index(item['scene'])
            path = HERE / 'artifacts' / cohort / arm / f"{scene}-{item['case']}.npz"
            with np.load(path) as data:
                a, c = data['records'], data['candidates']
            assert a.shape == (item['steps'], 16) and c.shape == (item['steps'], 8, 80)
            assert np.isfinite(a).all() and np.isfinite(c).all()
            indices, nominal = a[:, 5].astype(int), a[:, 6].astype(int)
            safe = c[:, 5].astype(bool)
            active = a[:, 12].astype(bool)
            assert ((indices >= 0) & (indices < 80)).all()
            assert (safe[np.arange(len(a)), indices] | ~safe.any(axis=1)).all()
            assert item['contact'] == int((a[:, 8] < 0).any())
            assert item['min_clearance'] == float(a[:, 8].min())
            assert item['duration'] == len(a) * .25
            assert item['intervention_steps'] == int(active.sum())
            assert item['action_changes'] == int((indices != nominal).sum())
            assert not (indices[~active] != nominal[~active]).any()
            assert item['fallback'] == int((~safe.any(axis=1)).sum())
            assert not c[~active, 6].any()
            # Serialized Float32 operands can round differently from the online
            # Float64 subtraction; raw actions/determinism are checked exactly.
            np.testing.assert_allclose(c[:, 7], c[:, 2] - c[:, 6], rtol=2e-7, atol=1e-7)
            if arm == 'side-latched':
                held = None
                for row in a:
                    if row[12]:
                        if int(row[11]) == 7:
                            held = row[13:16].copy()
                        assert held is not None
                        np.testing.assert_array_equal(row[13:16], held)
                    else:
                        held = None
            rows.append({'arm': arm, 'scene': scene, 'case': item['case'],
                         'steps': len(a), 'action_changes': item['action_changes'],
                         'trace_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    run.verify_manifest()
    result = {'passed': True, 'cohort': cohort, 'new_complete_episodes': len(rows),
              'reused_complete_episodes_verified': reused,
              'new_executed_steps': sum(v['steps'] for v in rows), 'rows': rows,
              'checkpoint_and_sources_unchanged': True, 'all_action_and_safety_assertions_passed': True}
    (HERE / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    verify()
