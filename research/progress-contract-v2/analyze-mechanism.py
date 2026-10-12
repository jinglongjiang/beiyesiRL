"""Post-outcome mechanism accounting, not extra gates or causal attribution."""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
names = ['baseline_circle', 'baseline_square', 'dense_circle', 'dense_square', 'large_circle', 'large_square']
output = {}
for arm in ['mpc', 'hold8', 'deadline']:
    rows = json.loads((HERE / f'development-{arm}.json').read_text())['results']
    episodes, obligations = [], []
    for row in rows:
        stem = HERE / f'artifacts/development/{arm}/{names.index(row["scene"])}-{row["case"]}'
        with np.load(stem.with_suffix('.npz')) as data:
            trace = data['records'].copy()
        diagnostics = json.loads(stem.with_name(stem.name + '-mechanism.json').read_text())
        assert len(trace) == len(diagnostics)
        active = np.asarray([v['active'] for v in diagnostics])
        for step, item in enumerate(diagnostics):
            if item['triggered'] and item['active']:
                interval = diagnostics[step:step + 16]
                complete = len(interval) == 16 and all(v['active'] for v in interval)
                if complete:
                    assert [v['remaining'] for v in interval] == list(range(15, -1, -1))
                    obligations.append({'scene': row['scene'], 'case': row['case'], 'step': step,
                                        'predicted_repayment': item['predicted_repayment'],
                                        'actual_repayment': float(trace[step:step + 16, 9].sum()),
                                        'repaid_half_metre': bool(trace[step:step + 16, 9].sum() >= .5),
                                        'contact_in_interval': bool((trace[step:step + 16, 8] < 0).any())})
        episodes.append({'scene': row['scene'], 'case': row['case'], 'success': row['success'],
                         'contact': row['contact'], 'triggers': row['triggers'],
                         'planning_calls': row['intervention_steps'] + row['invalidations'],
                         'active_steps': int(active.sum()),
                         'active_nonpositive_steps': int((active & (trace[:, 9] <= 0)).sum()),
                         'changed_executed_actions': int((active & (trace[:, 5] != trace[:, 6])).sum()),
                         'retained_plan_steps': row['retained_plan_steps'],
                         'invalidations': row['invalidations'], 'rejected_plan_count': row['rejected_plan_count']})
    output[arm] = {'episodes': episodes, 'complete_intervals': obligations,
                   'total_planning_calls': sum(v['planning_calls'] for v in episodes),
                   'activated_episodes': sum(v['active_steps'] > 0 for v in episodes),
                   'active_nonpositive_fraction': sum(v['active_nonpositive_steps'] for v in episodes) / max(sum(v['active_steps'] for v in episodes), 1),
                   'complete_interval_count': len(obligations),
                   'actually_repaid_count': sum(v['repaid_half_metre'] for v in obligations),
                   'scope': 'Approximate legal CV forecast certificate is not a true-world safety/liveness guarantee. Predictions versus realized progress are descriptive; do not uniquely attribute forecast error versus replanning without another controlled test.'}
(HERE / 'mechanism.json').write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
print({a: {k: v for k, v in data.items() if k not in ['episodes', 'complete_intervals']} for a, data in output.items()})
