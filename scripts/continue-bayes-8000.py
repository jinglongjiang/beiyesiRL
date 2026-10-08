"""User-requested warm continuation; never overwrites the 3000-run artifacts."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import configparser
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'crowd_nav/runs/bayes-matched-v1'
ARMS = ('gru', 'bayes_mean', 'bayes')
spec = importlib.util.spec_from_file_location('matched', ROOT / 'scripts/run-bayes-matched.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def output(arm):
    return RUN / f'{arm}-seed42-to8000'


def record(event, **values):
    row = dict(time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), event=event, **values)
    with (RUN / 'budget-8000-events.jsonl').open('a') as stream:
        stream.write(json.dumps(row) + '\n')
    print(json.dumps(row), flush=True)


def validate(path, arm, episode):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if saved.get('episode') != episode or saved.get('stage') != 'rl_training_sarl':
        raise RuntimeError(f'Wrong checkpoint episode/stage: {path}')
    if saved.get('config', {}).get('temporal_backbone') != arm:
        raise RuntimeError(f'Wrong backbone: {path}')
    if not saved.get('target_value_net_state') or not saved.get('policy_state'):
        raise RuntimeError(f'Incomplete model/target checkpoint: {path}')
    optimizer = saved.get('optim_value_state', {}).get('state', {})
    if not optimizer or any(float(state['step']) != episode * 4 for state in optimizer.values()):
        raise RuntimeError(f'Wrong optimizer update count: {path}')
    if not all(torch.isfinite(t).all().item() for t in saved['policy_state'].values() if torch.is_tensor(t)):
        raise RuntimeError(f'Nonfinite model: {path}')
    return runner.digest(path)


def prepare():
    runner.check_sources()
    if runner.digest(RUN / 'orca-shared.pth') != json.loads((RUN / 'dataset.json').read_text())['sha256']:
        raise RuntimeError('Frozen corpus hash mismatch')
    hashes = {arm: validate(RUN / f'{arm}-seed42/rl_model_ep3000.pth', arm, 3000) for arm in ARMS}
    protocol = dict(
        version='user-budget-8000-warm-continuation-v1', arms=ARMS, training_seed=42,
        original_protocol_sha256=runner.digest(RUN / 'protocol.json'),
        starting_checkpoint_sha256=hashes, start_episode=3001, last_episode=8000,
        additional_rl_episodes=5000, il_retraining=False,
        restored=['policy', 'target_value_network', 'optimizer', 'statistics_history'],
        not_restored=['online_replay', 'RNG_state', 'environment_case_counter'],
        replay='Native replay prefill from identical frozen ORCA corpus; old online replay unavailable',
        rng='Native matched reset seed+200003; this is NOT an exact uninterrupted run',
        environment='Native restart; training case sequence may repeat earlier cases',
        no_mamba=True, model_objective_reward_action_history_unchanged=True,
        evaluation='Same six native scenarios x 500 at fixed episode-8000 weights; not fresh confirmation',
        evaluation_seed=20261006, evaluation_time_limit=25,
        launch_condition='All three complete-3000.json exist and validate; never interrupt their evaluations',
        resource_guard='At least 27 GiB available host RAM and 4 GiB disk before starting 3 workers',
        failure_rule='No silent fresh training, partial restore, overwrite, or automatic rescue',
        script_sha256=runner.digest(Path(__file__)))
    path = RUN / 'budget-8000-amendment.json'
    if path.exists() and json.loads(path.read_text()) != json.loads(json.dumps(protocol)):
        raise RuntimeError('Existing 8000 protocol differs')
    runner.write_json(path, protocol)
    for arm in ARMS:
        dest = output(arm)
        dest.mkdir(exist_ok=True)
        il = RUN / f'{arm}-seed42/il_policy.pth'
        if (dest / 'il_policy.pth').exists():
            if runner.digest(dest / 'il_policy.pth') != runner.digest(il):
                raise RuntimeError('Copied IL checkpoint mismatch')
        else:
            shutil.copy2(il, dest / 'il_policy.pth')
        cfg = configparser.RawConfigParser(inline_comment_prefixes=('#', ';'))
        cfg.read(RUN / f'{arm}-seed42.ini')
        cfg.set('train', 'train_episodes', '8000')
        cfg.set('train', 'il_ckpt', str(dest / 'il_policy.pth'))
        with (RUN / f'{arm}-seed42.8000.ini').open('w') as stream:
            cfg.write(stream)
    record('PROTOCOL_FROZEN', checkpoint_sha256=hashes)


def run_training(arm):
    dest = output(arm)
    log = dest / 'stdout.log'
    if log.exists():
        raise RuntimeError(f'Interrupted output preserved, requires explicit recovery: {dest}')
    command = [sys.executable, '-m', 'crowd_nav.train', '--config', RUN / f'{arm}-seed42.8000.ini',
               '--outdir', dest, '--temporal-backbone', arm, '--seed', '42', '--gpu',
               '--resume', RUN / f'{arm}-seed42/rl_model_ep3000.pth']
    env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
               MPLBACKEND='Agg', PYTHONUNBUFFERED='1')
    start = time.time()
    with log.open('w') as stream:
        proc = subprocess.Popen(list(map(str, command)), cwd=ROOT, env=env,
                                stdout=stream, stderr=subprocess.STDOUT)
        runner.write_json(dest / 'state.json', dict(status='RESUMING', pid=proc.pid, arm=arm, target=8000))
        record('CONTINUATION_STARTED', arm=arm, pid=proc.pid, command=list(map(str, command)))
        verified = False
        try:
            while proc.poll() is None:
                text = log.read_text(errors='replace')
                forbidden = ('Starting fresh training', 'Failed to load checkpoint',
                             'Failed to load some components', 'trying partial load',
                             '[RESUME] Missing keys', '[RESUME] Unexpected keys',
                             'No BC checkpoint found', 'Failed to prefill buffer',
                             'Will train from scratch')
                if any(token in text for token in forbidden):
                    raise RuntimeError(f'Invalid restore detected; stopping owned worker {proc.pid}')
                if not verified and all(token in text for token in
                                        ('resumed from episode 3001', 'Target network loaded', 'Optimizer loaded')):
                    verified = True
                    record('RESTORE_VERIFIED', arm=arm, pid=proc.pid)
                    runner.write_json(dest / 'state.json', dict(status='TRAINING', arm=arm, pid=proc.pid,
                                                               start_episode=3001, target=8000))
                time.sleep(10)
            if proc.returncode or not verified:
                raise RuntimeError(f'Training exit={proc.returncode}, restore_verified={verified}: {arm}')
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
    seconds = time.time() - start
    sha = validate(dest / 'rl_model_ep8000.pth', arm, 8000)
    runner.write_json(dest / 'training-8000.json', dict(arm=arm, train_wall_seconds=seconds,
                                                      checkpoint_sha256=sha, warm_continuation=True))
    record('TRAINING_COMPLETE_8000', arm=arm, wall_seconds=seconds, checkpoint_sha256=sha)


def finish_arm(arm):
    dest = output(arm)
    if (dest / 'complete-8000.json').exists():
        return
    if not (dest / 'training-8000.json').exists():
        run_training(arm)
    sha = validate(dest / 'rl_model_ep8000.pth', arm, 8000)
    csv = dest / 'benchmark-8000.csv'
    if csv.exists():
        raise RuntimeError(f'Unfinished evaluation preserved: {csv}')
    cfg = RUN / f'{arm}-seed42.eval.ini'
    command = [sys.executable, '-m', 'crowd_nav.test', '--policy', 'mamba', '--model_dir', dest,
               '--weights', 'rl_model_ep8000.pth', '--env_config', cfg, '--policy_config', cfg,
               '--temporal-backbone', arm, '--episodes', '500', '--time-limit', '25',
               '--seed', '20261006', '--no_progress', '--measure-latency', '--oscillation_csv', csv,
               '--run_label', f'{arm}-seed42-rl8000', '--gpu']
    runner.write_json(dest / 'state.json', dict(status='EVALUATING', arm=arm, episode=8000))
    record('EVALUATION_STARTED_8000', arm=arm)
    seconds = runner.invoke(command, dest / 'benchmark-8000.log', f'{arm}-seed42-benchmark-8000')
    runner.write_json(dest / 'complete-8000.json', dict(
        arm=arm, seed=42, rl_episodes=8000, warm_continuation=True, checkpoint_sha256=sha,
        eval_wall_seconds=seconds, results=runner.benchmark_summary(csv)))
    runner.write_json(dest / 'state.json', dict(status='COMPLETE', arm=arm, episode=8000))
    record('EVALUATION_COMPLETE_8000', arm=arm)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    lock = (RUN / 'budget-8000-controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    prepare()
    if args.prepare_only:
        return
    record('WAITING_FOR_3000_EVALUATIONS')
    while not all((RUN / f'{arm}-seed42/complete-3000.json').exists() for arm in ARMS):
        time.sleep(30)
    for arm in ARMS:
        data = json.loads((RUN / f'{arm}-seed42/complete-3000.json').read_text())
        if data['checkpoint_sha256'] != validate(RUN / f'{arm}-seed42/rl_model_ep3000.pth', arm, 3000):
            raise RuntimeError('Completed 3000 evaluation checkpoint mismatch')
        runner.benchmark_summary(RUN / f'{arm}-seed42/benchmark-3000.csv')
    available_kib = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                             if line.startswith('MemAvailable:')))
    if available_kib < 27 * 1024 ** 2 or shutil.disk_usage(ROOT).free < 4 * 1024 ** 3:
        raise RuntimeError('Resource guard: insufficient memory/disk; no worker launched')
    failures = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(finish_arm, arm): arm for arm in ARMS}
        for future in as_completed(futures):
            arm = futures[future]
            try:
                future.result()
            except Exception as error:
                record('ARM_ERROR', arm=arm, error=str(error))
                runner.write_json(output(arm) / 'state.json', dict(status='ERROR', arm=arm, error=str(error)))
                failures.append(arm)
    runner.write_json(RUN / 'complete-8000-all.json', dict(
        status='ERROR' if failures else 'COMPLETE', failures=failures,
        arms={arm: json.loads((output(arm) / 'state.json').read_text()) for arm in ARMS}))
    if failures:
        raise RuntimeError('Failed arms: ' + ', '.join(failures))


if __name__ == '__main__':
    main()
