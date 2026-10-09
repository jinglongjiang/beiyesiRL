"""Reuse completed nested tests explicitly, never rerun them as new evidence."""
import hashlib
import json
from pathlib import Path

base = Path(__file__).resolve().parent
source = Path('/home/abc/workspace/CrowdNav(20270731_backup2)/CrowdNav/crowd_nav/runs/bayes-diagnostic-20261009')
results = {}
for tag in ('old', 'seed42', 'seed43'):
    path = source / ('nested-'+tag+'.json')
    obj = json.loads(path.read_text())
    results[tag] = {'source': str(path), 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                    'status': 'EXISTING_RESULT_REUSE_NOT_A_NEW_RUN',
                    'results': obj['results']}
(base / 'information-layer-existing.json').write_text(json.dumps({
    'status': 'EXISTING_RESULT_REUSE_NOT_A_NEW_RUN', 'data': results,
    'scope': 'Original b1 is pre-repair. Seed42/43 are repaired filters sampled earlier; incremental associations beyond two linear geometry descriptors only.',
    'not_tested': 'Candidate-local logvar contrast and action-specific decision information are NOT identified by current-state sigma probes.'
}, indent=2) + '\n')
print('Reused three frozen nested-test results with source hashes; no new experiment.')
