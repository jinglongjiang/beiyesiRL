"""Show that the reference-gate correction changes no scientific computation."""
import json
from pathlib import Path
import numpy as np

base = Path(__file__).resolve().parent
episodes, steps = 0, 0
for old in sorted((base / 'smoke-before-reference-correction').glob('seed*-case*.npz')):
    new = base / 'smoke' / old.name
    a, b = np.load(old), np.load(new)
    assert a.files == b.files
    for key in a.files:
        np.testing.assert_array_equal(a[key], b[key], err_msg=old.name+'/'+key)
    first = json.loads(old.with_suffix('.json').read_text())
    second = json.loads(new.with_suffix('.json').read_text())
    assert first['rows'] == second['rows']
    episodes += 1
    steps += len(first['rows'])
assert episodes == 36
result = {'status': 'VERIFIED', 'episodes': episodes, 'steps': steps,
          'all_arrays_bitwise_identical': True, 'all_step_records_identical': True,
          'scope': 'Only external benchmark acceptance changed, not actions/scores/interventions/metrics'}
(base / 'engineering-smoke-r1-r2-parity.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result))
