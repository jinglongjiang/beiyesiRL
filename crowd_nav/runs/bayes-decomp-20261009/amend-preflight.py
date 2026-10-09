"""Preserve the invalid pre-outcome version and freeze an interface-only fix."""
import hashlib
import json
from pathlib import Path

base = Path(__file__).resolve().parent
manifest_path = base / 'frozen-manifest.json'
manifest = json.loads(manifest_path.read_text())
script = base / 'decompose.py'
archived = base / 'decompose-preflight-invalid.py'
script_key = next(name for name in manifest if Path(name).resolve() == script)
assert not list(base.glob('fixed-seed*.json'))
assert not (base / 'controls.json').exists()
assert not (base / 'frozen-manifest-initial.json').exists()
assert hashlib.sha256(archived.read_bytes()).hexdigest() == manifest[script_key]['sha256']
for name, entry in manifest.items():
    if name != script_key:
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == entry['sha256'], name
manifest_path.rename(base / 'frozen-manifest-initial.json')
manifest[str(archived)] = manifest.pop(script_key)
manifest[str(script)] = {'sha256': hashlib.sha256(script.read_bytes()).hexdigest(), 'bytes': script.stat().st_size}
manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
record = {'status': 'ENGINEERING_INVALID_BEFORE_FIRST_EPISODE',
          'error': 'AttributeError: optional set_env does not exist on inherited policy',
          'scientific_observations': 0, 'log': 'calibration.log',
          'repair': 'Guard optional set_env exactly as existing validated read-only evaluator',
          'algorithm_changes': False, 'old_source': str(archived),
          'new_source_sha256': manifest[str(script)]['sha256'],
          'protocol_unchanged': True, 'failed_job_stopped': True}
(base / 'engineering-manifest-key.json').write_text(json.dumps({
    'status': 'ENGINEERING_INVALID', 'scientific_observations': 0,
    'error': 'KeyError from Python3.8 relative __file__ key in freeze manifest',
    'repair': 'Resolve manifest key, preserving original hash and all frozen protocol definitions'
}, indent=2) + '\n')
(base / 'engineering-preflight.json').write_text(json.dumps(record, indent=2) + '\n')
print('Pre-outcome interface amendment frozen; original evidence retained.')
