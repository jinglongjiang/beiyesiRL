"""Operational cutoff and evaluation; leaves the frozen scientific sources intact."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time

import torch


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'crowd_nav/runs/bayes-matched-v1'
spec = importlib.util.spec_from_file_location('matched_runner', ROOT / 'scripts/run-bayes-matched.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
ARMS = ('gru', 'bayes_mean', 'bayes')


def record(event, **values):
    item = dict(time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                event=event, **values)
    with (RUN / 'budget-3000-events.jsonl').open('a') as stream:
        stream.write(json.dumps(item) + '\n')
    print(json.dumps(item), flush=True)


def checkpoint(arm):
    return RUN / f'{arm}-seed42' / 'rl_model_ep3000.pth'


def validate(arm):
    path = checkpoint(arm)
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if saved.get('episode') != 3000 or not str(saved.get('stage', '')).startswith('rl_training'):
        raise RuntimeError(f'Wrong/incomplete checkpoint: {path}')
    if saved.get('config', {}).get('temporal_backbone') != arm:
        raise RuntimeError(f'Wrong temporal architecture: {path}')
    states = saved.get('policy_state', {})
    optimizer = saved.get('optim_value_state', {}).get('state', {})
    if not states or not optimizer or 'target_value_net_state' not in saved:
        raise RuntimeError(f'Partial checkpoint: {path}')
    if not all(torch.isfinite(t).all().item() for t in states.values() if torch.is_tensor(t)):
        raise RuntimeError(f'Nonfinite checkpoint: {path}')
    return runner.digest(path)


def command_line(pid):
    try:
        return (Path('/proc') / str(pid) / 'cmdline').read_bytes().split(b'\0')
    except FileNotFoundError:
        return []


def owned(pid, arm):
    args = [s.decode() for s in command_line(pid) if s]
    expected = ['--outdir', str(RUN / f'{arm}-seed42')]
    return (args and all(value in args for value in expected)
            and '--temporal-backbone' in args
            and args[args.index('--temporal-backbone') + 1] == arm)


def finish(arm, pid):
    if not owned(pid, arm):
        raise RuntimeError(f'Worker ownership mismatch: {arm} pid={pid}')
    while not checkpoint(arm).exists():
        if not owned(pid, arm):
            raise RuntimeError(f'Worker ended before checkpoint 3000: {arm}')
        time.sleep(1)
    sha = validate(arm)
    if owned(pid, arm):
        os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + 15
        while owned(pid, arm) and time.monotonic() < deadline:
            time.sleep(0.2)
        if owned(pid, arm):
            os.kill(pid, signal.SIGKILL)
    record('USER_BUDGET_CUTOFF', arm=arm, pid=pid, checkpoint_sha256=sha,
           evaluated_episode=3000, note='Post-checkpoint work, if any, excluded; not a scientific failure')
    runner.write_json(RUN / f'{arm}-seed42' / 'state.json',
                      dict(status='TRAINING_COMPLETE_AT_USER_BUDGET', arm=arm, seed=42, episode=3000))


def evaluate(arm):
    sha = validate(arm)
    output = RUN / f'{arm}-seed42'
    done = output / 'complete-3000.json'
    if done.exists():
        record('ALREADY_EVALUATED', arm=arm)
        return
    csv = output / 'benchmark-3000.csv'
    if csv.exists():
        raise RuntimeError(f'Existing unfinished result preserved, not overwritten: {csv}')
    config = RUN / f'{arm}-seed42.eval.ini'
    command = [sys.executable, '-m', 'crowd_nav.test', '--policy', 'mamba',
               '--model_dir', output, '--weights', checkpoint(arm).name,
               '--env_config', config, '--policy_config', config,
               '--temporal-backbone', arm, '--episodes', 500,
               '--time-limit', 25, '--seed', 20261006, '--no_progress',
               '--measure-latency', '--oscillation_csv', csv,
               '--run_label', f'{arm}-seed42-rl3000', '--gpu']
    record('EVALUATION_STARTED', arm=arm, checkpoint_sha256=sha)
    seconds = runner.invoke(command, output / 'benchmark-3000.log', f'{arm}-seed42-benchmark-3000')
    runner.write_json(done, dict(arm=arm, seed=42, rl_episodes=3000,
                                checkpoint_sha256=sha, eval_wall_seconds=seconds,
                                results=runner.benchmark_summary(csv),
                                training_compute_note='Contended training; GRU ran beyond 3000 before user cutoff, evaluated only episode 3000'))
    runner.write_json(output / 'state.json', dict(status='COMPLETE_AT_USER_BUDGET',
                                                arm=arm, seed=42, episode=3000))
    record('EVALUATION_COMPLETE', arm=arm, wall_seconds=seconds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mean-pid', type=int, default=1609725)
    parser.add_argument('--full-pid', type=int, default=1609723)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    runner.check_sources()
    if args.validate_only:
        print(validate('gru'), flush=True)
        for arm, pid in [('bayes_mean', args.mean_pid), ('bayes', args.full_pid)]:
            if not owned(pid, arm):
                raise RuntimeError(f'Ownership mismatch: {pid}')
        return
    lock = (RUN / 'budget-3000-controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    amendment = dict(reason='Explicit user request; operational budget change, not outcome-selected stopping',
                     original_protocol_sha256=runner.digest(RUN / 'protocol.json'),
                     rl_budget=3000, retained_il_epochs=50, arms=ARMS, seed=42,
                     checkpoint='rl_model_ep3000.pth', evaluation='Original six scenarios x 500',
                     no_mamba_training_or_evaluation=True,
                     gru='Use existing episode-3000 weights; later training is not used for selection',
                     seed43='Not launched under revised initial three-arm evaluation schedule',
                     cutoff='Monitor complete atomic checkpoint, validate then stop existing workers; do not resume/reseed',
                     source_sha256=runner.digest(Path(__file__)))
    path = RUN / 'budget-3000-amendment.json'
    if path.exists():
        if json.loads(path.read_text()) != json.loads(json.dumps(amendment)):
            raise RuntimeError('Existing amendment mismatch')
    else:
        runner.write_json(path, amendment)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(finish, 'bayes_mean', args.mean_pid),
                   executor.submit(finish, 'bayes', args.full_pid)]
        failures = []
        try:
            evaluate('gru')
        except Exception as error:
            record('EVALUATION_ERROR', arm='gru', error=str(error))
            failures.append(str(error))
        for arm, future in zip(('bayes_mean', 'bayes'), futures):
            try:
                future.result()
                evaluate(arm)
            except Exception as error:
                record('ARM_ERROR', arm=arm, error=str(error))
                failures.append(str(error))
        if failures:
            raise RuntimeError('; '.join(failures))


if __name__ == '__main__':
    main()
