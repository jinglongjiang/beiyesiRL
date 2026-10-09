"""Hash only after all log-writing jobs have finished."""
import sys
sys.dont_write_bytecode = True
import hashlib
import json
from pathlib import Path

base = Path(__file__).resolve().parent
from decompose import pinned
pinned()
summary = json.loads((base / 'stepwise/summary.json').read_text())
assert summary['status'] == 'COMPLETE' and summary['episodes'] == 1200
assert not Path('/proc/%s' % (base / 'pids.txt').read_text().strip()).exists()
analysis_path = base / 'analysis-manifest.json'
analysis = json.loads(analysis_path.read_text())
analysis['report_before_interpretation_edit_sha256'] = analysis['report_sha256']
analysis['report_sha256'] = hashlib.sha256((base / 'REPORT.md').read_bytes()).hexdigest()
analysis['interpretation_edit'] = 'Summarize existing frozen results and posthoc terminal-reward attribution; no statistics or verdict changes'
analysis_path.write_text(json.dumps(analysis, indent=2)+'\n')
path = base / 'final-manifest.json'
manifest = json.loads(path.read_text())
manifest['files'] = {str(p.relative_to(base)): {'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'bytes': p.stat().st_size}
                     for p in sorted(base.rglob('*')) if p.is_file() and p != path}
path.write_text(json.dumps(manifest, indent=2)+'\n')
assert all(hashlib.sha256((base / name).read_bytes()).hexdigest() == entry['sha256']
           for name, entry in manifest['files'].items())
print(json.dumps({'status': 'ALL_HASHES_VERIFIED', 'episodes': 1200, 'steps': summary['steps'],
                  'files': len(manifest['files']), 'verdict': summary['verdict'],
                  'next_route': summary['next_route'], 'report': str(base / 'REPORT.md')}))
