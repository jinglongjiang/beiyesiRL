"""Exact argv matching, not pgrep/substring self matching."""
from pathlib import Path
import json

base = Path(__file__).resolve().parent
expected = [b'python3', b'-B', b'decompose.py', b'full']
pids = []
for directory in Path('/proc').iterdir():
    if not directory.name.isdigit():
        continue
    try:
        args = (directory / 'cmdline').read_bytes().rstrip(b'\x00').split(b'\x00')
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
    if args == expected:
        pids.append(int(directory.name))
print(json.dumps({'exact_full_process_pids': pids, 'liveness': [Path('/proc/%d' % pid).exists() for pid in pids]}))
if pids:
    (base / 'pids.txt').write_text(''.join(str(pid)+'\n' for pid in pids))
