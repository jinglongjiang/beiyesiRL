"""Save the isolated server run locally without copying the shared IL corpus."""

import fcntl
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'crowd_nav/runs/bayes-optimized-20261007/accepted-run'
REMOTE = 'root@45.126.120.6:/root/bayes-vl-optimized-20261007/crowd_nav/runs/optimized-10000-r1/'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / 'mirror.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    command = ['rsync', '-az', '--timeout=60', '-e', 'ssh -o BatchMode=yes -o ConnectTimeout=15',
               REMOTE, str(OUT) + '/']
    while True:
        try:
            subprocess.run(command, check=True, timeout=180)
            print(time.strftime('%Y-%m-%d %H:%M:%S'), 'synchronized', flush=True)
            if (OUT / 'complete.json').exists() or (OUT / 'error.json').exists():
                return
        except Exception as error:
            print('synchronization error:', error, flush=True)
        time.sleep(60)


if __name__ == '__main__':
    main()
