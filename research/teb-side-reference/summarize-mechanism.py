"""Post-outcome descriptive mechanism checks; not a new causal experiment."""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SCENES = ['baseline_circle', 'baseline_square', 'dense_circle', 'dense_square', 'large_circle', 'large_square']


def analyze(arm):
    results = json.loads((HERE / f'development-{arm}.json').read_text())['results']
    active_rows, candidate_rows, phases, cases = [], [], [], []
    for result in results:
        scene = SCENES.index(result['scene'])
        with np.load(HERE / f"artifacts/development/{arm}/{scene}-{result['case']}.npz") as data:
            rows, candidates = data['records'], data['candidates']
        active = rows[:, 12].astype(bool)
        if active.any():
            active_rows.append(rows[active])
            candidate_rows.append(candidates[active])
            cases.append({'scene': result['scene'], 'case': result['case'],
                          'outcome': result['outcome'], 'triggers': result['triggers'],
                          'active_steps': int(active.sum()), 'changed_actions': result['action_changes']})
        current = []
        for row in rows:
            if row[12]:
                if int(row[11]) == 7 and current:
                    phases.append(current)
                    current = []
                current.append(float(row[9]))
            elif current:
                phases.append(current)
                current = []
        if current:
            phases.append(current)
    a = np.concatenate(active_rows)
    c = np.concatenate(candidate_rows)
    changed = a[:, 5] != a[:, 6]
    positive_available = ((c[:, 3] > 0) & c[:, 5].astype(bool)).any(axis=1)
    return {'activated_episodes': len(cases), 'active_steps': len(a), 'changed_actions': int(changed.sum()),
            'fraction_active_executed_nonpositive_progress': float((a[:, 9] <= 0).mean()),
            'fraction_active_nominal_nonpositive_progress_same_states': float((a[:, 10] <= 0).mean()),
            'fraction_active_with_safe_positive_alternative': float(positive_available.mean()),
            'nonpositive_selection_despite_positive_alternative_steps': int(((a[:, 9] <= 0) & positive_available).sum()),
            'mean_executed_progress_per_active_step_m': float(a[:, 9].mean()),
            'mean_nominal_progress_per_active_step_same_states_m': float(a[:, 10].mean()),
            'intervention_phases': len(phases),
            'nonpositive_net_progress_phases': sum(sum(p) <= 0 for p in phases),
            'median_net_progress_per_phase_m': float(np.median([sum(p) for p in phases])),
            'phase_lengths': {'min': min(map(len, phases)), 'max': max(map(len, phases))},
            'cases': cases}


if __name__ == '__main__':
    result = {'analysis_registered_after_outcomes': True,
              'scope': 'Descriptive trace accounting; no assertion of a unique cause, no changed controller or new simulator experiment.',
              'arms': {arm: analyze(arm) for arm in ['side-reset', 'side-latched']}}
    (HERE / 'mechanism.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({arm: {k: v for k, v in data.items() if k != 'cases'} for arm, data in result['arms'].items()}, indent=2))
