"""Single-case engineering replay; never part of the scientific cohort."""
import sys
sys.dont_write_bytecode = True
import json
import types
from pathlib import Path
import numpy as np
import torch
import importlib.util

spec = importlib.util.spec_from_file_location('archived_diagnostic',
       Path(__file__).resolve().parent / 'decompose-external-reference-stop.py')
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)

base = Path(__file__).resolve().parent
d.pinned()
seed, scene, case = 42, 4, 34
m, policy, references = d.setup(seed)
reference = references[d.SCENARIOS[scene][0], case]
fixed = d.load(base / 'fixed-seed42.json')['mean_logvar']
instrumented = []
for attempt in range(2):
    try:
        rows, trace, _, meta = d.episode(m, policy, references, seed, scene, case, 'engineering', fixed, True)
        instrumented.append({'passed_external_reference': True, 'outcome': meta['outcome'], 'steps': meta['steps']})
    except AssertionError as error:
        tb = error.__traceback__
        frame = None
        while tb is not None:
            if tb.tb_frame.f_code.co_name == 'episode':
                frame = tb.tb_frame
            tb = tb.tb_next
        assert str(error) == 'Benchmark outcome/step mismatch', 'Do not suppress another failure'
        assert frame is not None
        local = frame.f_locals
        rows, trace = local['rows'], local['trace']
        instrumented.append({'passed_external_reference': False, 'outcome': local['outcome'], 'steps': local['steps']})
    np.savez_compressed(base / ('engineering-case34-attempt%d.npz' % attempt),
                        actions=np.asarray([t['action'] for t in trace]),
                        positions=np.asarray([t['position'] for t in trace]),
                        values=np.stack([t['real_values'] for t in trace]))
    d.save(base / ('engineering-case34-attempt%d.json' % attempt), {'engineering_only': True, 'rows': rows})
    if attempt == 0:
        first_trace = trace
    else:
        d.identical_trace(first_trace, trace)

env, robot = d.environment(m, policy, seed, scene)
original_predict = policy.predict
actions = []
def observe(state):
    action = original_predict(state)
    actions.append((action.vx, action.vy))
    return action
policy.predict = observe
with torch.no_grad():
    try:
        baseline = m.test_episode(env, robot, policy, int(reference['seed']), case_desc=d.SCENARIOS[scene][0])
    finally:
        policy.predict = original_predict
np.testing.assert_array_equal(np.asarray(actions), np.asarray([t['action'] for t in first_trace]))
np.testing.assert_array_equal(np.asarray(baseline[2][1:]), np.asarray([t['position'] for t in first_trace]))
assert baseline[0] == instrumented[0]['outcome'] and baseline[1] == instrumented[0]['steps']
result = {'status': 'ENGINEERING_REPLAY_COMPLETE_NOT_SCIENTIFIC_OUTCOME',
          'training_seed': seed, 'scene': d.SCENARIOS[scene][0], 'case': case,
          'reference_outcome': reference['outcome'], 'reference_steps': int(reference['steps']),
          'instrumented_attempts': instrumented,
          'unpatched_local_parent_outcome': baseline[0], 'unpatched_local_parent_steps': baseline[1],
          'instrumented_repeat_bitwise_equal': True, 'unpatched_actions_positions_bitwise_equal': True,
          'conclusion': 'No instrumentation effect on this local replay. External benchmark trajectory is not identical; exact cross-runtime cause remains unverified.',
          'formal_run_stopped': True, 'new_scientific_observations': 0}
d.save(base / 'engineering-case34.json', result)
d.pinned()
print(json.dumps(result, indent=2))
