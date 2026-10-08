"""Serialize memory-heavy corpus conversion without changing training budgets."""

import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--pids', type=int, nargs=4, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run = root / 'crowd_nav/runs/bayes-matched-v1'
    arms = ('mamba', 'gru', 'bayes_mean', 'bayes')
    for pid in args.pids:
        command = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
        if str(run) not in command or 'crowd_nav.train' not in command:
            raise RuntimeError('Refusing to signal a process outside this experiment')

    def record(arm, event):
        item = {'utc': datetime.datetime.utcnow().isoformat(), 'arm': arm, 'event': event}
        with (run / 'resource-events.jsonl').open('a') as stream:
            stream.write(json.dumps(item) + '\n')
        print(json.dumps(item), flush=True)

    for arm, pid in zip(arms[1:], args.pids[1:]):
        os.kill(pid, signal.SIGSTOP)
        record(arm, 'preprocessing-paused-for-host-memory')
    for arm, pid in zip(arms, args.pids):
        if arm != 'mamba':
            os.kill(pid, signal.SIGCONT)
            record(arm, 'preprocessing-resumed')
        log = run / f'{arm}-seed{args.seed}' / 'train.log'
        while '[IL-BC] Prepared dataset:' not in log.read_text(errors='replace'):
            if not Path(f'/proc/{pid}').exists():
                record(arm, 'worker-exited-before-preprocessing-completed')
                break
            time.sleep(10)
        else:
            record(arm, 'preprocessing-completed-training-may-overlap')


if __name__ == '__main__':
    main()
