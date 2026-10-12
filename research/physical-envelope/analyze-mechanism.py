"""Recompute clipping activation and the bound-only control's score plateaus."""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
out = {}
for arm in ['envelope', 'distance-cap', 'free']:
    rows, plateau, active, distinct = [], [], [], []
    paths = sorted((HERE / 'artifacts/development' / arm).glob('*.npz'))
    assert len(paths) == 120
    for path in paths:
        with np.load(path) as data:
            a, c = data['records'], data['candidates']
        rows.append(a)
        valid = c[:, 5, :].astype(bool)
        selected = np.max(np.where(valid, c[:, 9, :], -1e9), axis=1)
        plateau.extend(((np.abs(c[:, 9, :] - selected[:, None]) < 1e-8) & valid).sum(1).tolist())
        active.extend((c[:, 0, :] > c[:, 6, :]).any(1).tolist())
        distinct.extend((np.abs(c[:, 6, :] - c[:, 7, :]) > 1e-8).any(1).tolist())
    a = np.concatenate(rows)
    episodes = json.loads((HERE / f'development-{arm}.json').read_text())['results']
    out[arm] = dict(episodes=len(paths), steps=len(a),
        speed_median=float(np.median(np.linalg.norm(a[:, 3:5], axis=1))),
        nonpositive_progress_fraction=float((a[:, 9] <= 0).mean()),
        top_score_tie_count_median=float(np.median(plateau)),
        top_score_tie_fraction=float((np.asarray(plateau) > 1).mean()),
        upper_bound_clip_step_fraction=float(np.mean(active)),
        envelope_differs_distance_bound_step_fraction=float(np.mean(distinct)),
        action_changes=sum(v['action_changes'] for v in episodes),
        envelope_distance_disagreements=sum(v['envelope_distance_disagreements'] for v in episodes))
(HERE / 'mechanism.json').write_text(json.dumps(out, indent=2, allow_nan=False) + '\n')
print(json.dumps(out, indent=2))
