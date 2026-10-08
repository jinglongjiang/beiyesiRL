"""Mirror overnight artifacts and refresh one bounded section of the existing report."""

import csv
import fcntl
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'crowd_nav/runs/bayes-matched-v1'
REPORT = ROOT / 'BAYES-TEMPORAL.md'
ARMS = ('gru', 'bayes_mean', 'bayes')
START = '<!-- overnight-status-start -->'
END = '<!-- overnight-status-end -->'


def metrics(path):
    if not path.exists():
        return 'not started'
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 3000:
        return f'evaluation {len(rows)}/3000; no aggregate verdict'
    scenarios = {row['scenario'] for row in rows}
    if len(scenarios) != 6 or any(sum(row['scenario'] == s for row in rows) != 500 for s in scenarios):
        return 'INVALID scenario counts; see raw CSV'
    counts = {key: sum(row['outcome'] == key for row in rows) for key in ('success', 'collision', 'timeout')}
    if sum(counts.values()) != 3000:
        return 'INVALID outcomes; see raw CSV'
    contact = sum(int(row['geometric_contact_steps']) > 0 for row in rows)
    return 'SR {:.2f}%, CR {:.2f}%, TR {:.2f}%; geometric-contact episodes {}/3000'.format(
        *(100 * counts[key] / 3000 for key in ('success', 'collision', 'timeout')), contact)


def refresh():
    rows = ['## Overnight continuation (automatically refreshed)', '',
            'Last local synchronization: ' + time.strftime('%Y-%m-%d %H:%M:%S %Z'), '',
            'User request: finish episode-3000 evaluation, then resume GRU, Bayes-Mean,',
            'and Bayes-Full to total episode 8000 and automatically evaluate. No Mamba runs.', '',
            '| Arm | Episode-3000 benchmark | Continuation | Episode-8000 benchmark |',
            '|---|---|---|---|']
    for arm in ARMS:
        dest = RUN / f'{arm}-seed42-to8000'
        state = json.loads((dest / 'state.json').read_text()) if (dest / 'state.json').exists() else {}
        status = state.get('status', 'QUEUED_AFTER_3000_EVALUATIONS')
        checkpoints = [int(p.stem.split('ep')[-1]) for p in dest.glob('rl_model_ep*.pth')]
        if checkpoints:
            status += '; last saved episode ' + str(max(checkpoints))
        if state.get('error'):
            status += ': ' + state['error'].replace('|', '/')
        rows.append('| {} | {} | {} | {} |'.format(
            arm, metrics(RUN / f'{arm}-seed42/benchmark-3000.csv'), status,
            metrics(dest / 'benchmark-8000.csv')))
    controller_log = RUN / 'controller-8000.log'
    if controller_log.exists() and 'Traceback (most recent call last)' in controller_log.read_text(errors='replace'):
        rows += ['', 'CONTROLLER ERROR: inspect controller-8000.log; queued status does not imply a running worker.']
    rows += ['', 'Resume restores policy, target network, optimizer, and statistics; no IL retraining.',
             'Native checkpoints do NOT contain online replay, RNG state, or environment case counter.',
             'All three arms use native frozen-ORCA replay prefill and the same matched RNG reset.',
             'Earlier training cases may repeat. This is a matched warm continuation, NOT an exact',
             'uninterrupted 8000-episode reproduction. GRU steps beyond original 3000 are not reused.', '',
             'Episode-8000 evaluation reuses the same six scenario/case blocks, 500 each.',
             'It is NOT fresh confirmation. Only one training seed is available. Full versus Mean',
             'is capacity-matched; GRU is not. Uncertainty remains value-MSE-trained, uncalibrated.',
             'No METHOD_ENTRY_FOUND or posterior calibration claim follows from these numbers.', '',
             'Artifacts: crowd_nav/runs/bayes-matched-v1/budget-8000-amendment.json;',
             'budget-8000-events.jsonl; controller-8000.log; *-seed42-to8000/state.json;',
             'training-8000.json; benchmark-8000.csv/log; complete-8000.json.',
             'All checkpoints and negative outcomes are retained. Hardware is shared; reported',
             'wall/latency measurements are contended, not isolated inference speed.', '']
    text = REPORT.read_text()
    if START not in text or END not in text:
        raise RuntimeError('Expected report status markers missing; refusing to rewrite user text')
    before, rest = text.split(START, 1)
    _, after = rest.split(END, 1)
    updated = before + START + '\n' + '\n'.join(rows) + END + after
    temporary = REPORT.with_suffix('.md.tmp')
    temporary.write_text(updated)
    temporary.replace(REPORT)


def main():
    lock = (RUN / 'overnight-mirror.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    command = ['rsync', '-az', '--timeout=90', '-e', 'ssh -o BatchMode=yes -o ConnectTimeout=15',
               '--exclude=mamba-*', '--exclude=orca-shared.pth', '--exclude=collection/',
               'root@45.126.120.6:/root/bayes-vl-matched-20261006/crowd_nav/runs/bayes-matched-v1/',
               str(RUN) + '/']
    while True:
        try:
            subprocess.run(command, check=True, timeout=240)
            refresh()
            print(time.strftime('%Y-%m-%d %H:%M:%S'), 'mirror refreshed', flush=True)
            if (RUN / 'complete-8000-all.json').exists():
                return
        except Exception as error:
            print(time.strftime('%Y-%m-%d %H:%M:%S'), 'mirror error:', error, flush=True)
        time.sleep(120)


if __name__ == '__main__':
    main()
