"""Three independent case shards; fail fast and keep every child's output."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
OUT=HERE/'artifacts'/'v2-50s'
assert json.loads((OUT/'budget.json').read_text())['fits_budget']
assert not (OUT/'launch.json').exists(), 'Do not duplicate an existing launch'
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
children=[]
logs=[]
metadata={'started_unix':time.time(),'workers':[],'complete':False}
for shard in range(3):
    log=(OUT/f'worker-{shard}.log').open('w')
    command=[sys.executable,str(HERE/'dual.py'),'catalogue','--shard',str(shard),'--shards','3']
    process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
    children.append(process)
    logs.append(log)
    metadata['workers'].append({'shard':shard,'pid':process.pid,'command':command})
(OUT/'launch.json').write_text(json.dumps(metadata,indent=2)+'\n')
try:
    while True:
        codes=[p.poll() for p in children]
        if any(c is not None and c!=0 for c in codes):
            raise RuntimeError(f'Worker failed: {codes}; see preserved worker logs')
        if all(c==0 for c in codes):
            break
        time.sleep(2)
    metadata.update(complete=True,elapsed_seconds=time.time()-metadata['started_unix'],exit_codes=codes)
    (OUT/'launch.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('ALL CATALOGUE SHARDS COMPLETE',json.dumps(metadata),flush=True)
finally:
    for p in children:
        if p.poll() is None:
            p.terminate()
    for p in children:
        p.wait()
    for log in logs:
        log.close()
