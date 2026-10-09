"""Freeze an external-reference audit correction, retaining interrupted evidence."""
import hashlib
import json
from pathlib import Path

base = Path(__file__).resolve().parent
manifest_path = base / 'frozen-manifest.json'
manifest = json.loads(manifest_path.read_text())
main = base / 'decompose.py'
archived = base / 'decompose-external-reference-stop.py'
assert hashlib.sha256(archived.read_bytes()).hexdigest() == manifest[str(main)]['sha256']
assert not (base / 'frozen-manifest-before-reference-correction.json').exists()
for name, entry in manifest.items():
    if name != str(main):
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == entry['sha256'], name
diagnosis = json.loads((base / 'engineering-case34.json').read_text())
assert diagnosis['instrumented_repeat_bitwise_equal'] and diagnosis['unpatched_actions_positions_bitwise_equal']
assert diagnosis['unpatched_local_parent_steps'] == 67 and diagnosis['reference_steps'] == 66
assert len(list((base / 'stepwise-interrupted').glob('seed*-case*.json'))) == 434
manifest_path.rename(base / 'frozen-manifest-before-reference-correction.json')
manifest[str(archived)] = manifest[str(main)]
manifest[str(main)] = {'sha256': hashlib.sha256(main.read_bytes()).hexdigest(), 'bytes': main.stat().st_size}
manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
protocol_path = base / 'protocol.json'
protocol = json.loads(protocol_path.read_text())
protocol_path.rename(base / 'protocol-before-reference-correction.json')
protocol['engineering_revision'] = 'Cross-environment exact benchmark trajectory is AUDIT_ONLY, not internal real determinism'
protocol['controls']['real_twice'] = 'UNCHANGED: two local replays must have bitwise identical inputs, values, actions, positions, rewards, outcomes, steps'
protocol['controls']['smoke'] = '6scenes x3cases x2seeds, GRID80, deterministic flips0, tensor replacement verified. External benchmark parity recorded separately, not used to remove cases'
protocol['correction_scope'] = 'No intervention/seed/metric/threshold/dynamics/action/weight changes; interrupted cohort archived, complete cohort rerun from case0'
protocol['external_reference'] = 'Case34 unpatched and instrumented local replay success67 versus prior success66; exact cause unverified. All mismatches saved without exclusion'
protocol_path.write_text(json.dumps(protocol, indent=2)+'\n')
analysis_path = base / 'analysis-freeze.json'
analysis = json.loads(analysis_path.read_text())
assert hashlib.sha256((base / 'analyze-before-reference-correction.py').read_bytes()).hexdigest() == analysis['sha256']
analysis_path.rename(base / 'analysis-freeze-before-reference-correction.json')
analysis['sha256'] = hashlib.sha256((base / 'analyze.py').read_bytes()).hexdigest()
analysis['phase'] = 'Before corrected smoke/formal outcomes; aggregate benefit outcomes not inspected'
analysis['amendment'] = 'Record external benchmark mismatch counts and engineering provenance; all metrics/statistics/directional criteria unchanged'
analysis_path.write_text(json.dumps(analysis, indent=2)+'\n')
(base / 'engineering-reference-correction.json').write_text(json.dumps({
    'status': 'CORRECTED_OVERSTRICT_DIAGNOSTIC_REFERENCE_GATE',
    'failed_formal_exit': 1, 'interrupted_episodes_preserved': 434,
    'instrumentation_effect_found': False, 'local_real_determinism_failed': False,
    'legacy_reference_mismatch': diagnosis,
    'formal_restart_from_beginning': True, 'old_new_scientific_data_not_merged': True,
    'outcome_benefit_aggregates_inspected': False,
    'algorithm_or_training_changes': False, 'original_project_changes': False
}, indent=2)+'\n')
print('Reference correction frozen; science and required deterministic gate unchanged.')
