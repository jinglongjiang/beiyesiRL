"""Reuse strict IL weights, accept a 200-episode smoke, then run native RL."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import configparser
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ARMS = ('gru', 'bayes_mean', 'bayes')
SOURCE = ('crowd_nav/policy/mamba_rl.py', 'crowd_nav/policy/bayes_temporal.py',
          'crowd_nav/utils/ppo_buffer.py', 'crowd_nav/train.py', 'crowd_nav/contracts.py',
          'crowd_nav/utils/explorer.py', 'crowd_nav/test.py',
          'scripts/verify-bayes-optimization.py', 'scripts/run-bayes-optimized.py')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def run(command, directory, label):
    log = directory / ('stdout.log' if label == 'train' else label + '.log')
    if log.exists():
        raise RuntimeError('Existing run preserved: ' + str(log))
    env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
               MPLBACKEND='Agg', PYTHONUNBUFFERED='1', TF_CPP_MIN_LOG_LEVEL='3')
    started = time.time()
    with log.open('w') as stream:
        proc = subprocess.Popen(list(map(str, command)), cwd=ROOT, env=env,
                                stdout=stream, stderr=subprocess.STDOUT)
        save(directory / 'state.json', dict(status=label.upper(), pid=proc.pid, command=list(map(str, command))))
        try:
            while proc.poll() is None:
                text = log.read_text(errors='replace')
                if label == 'train':
                    forbidden = ('Will train from scratch', 'trying partial load',
                                 'No BC checkpoint found', 'Failed to load checkpoint',
                                 'Failed to prefill buffer', 'Missing action_indices',
                                 'action_indices contains None', 'action_indices length mismatch')
                    if any(term in text for term in forbidden):
                        raise RuntimeError('Invalid IL/replay contract: ' + str(log))
                if shutil.disk_usage(ROOT).free < 1024 ** 3:
                    raise RuntimeError('Disk guard: less than 1 GiB free')
                time.sleep(10)
            if proc.returncode:
                raise RuntimeError(f'{label} exited {proc.returncode}: {log}')
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    proc.kill(); proc.wait()
    return time.time() - started


def configuration(assets, arm, directory, smoke):
    cfg = configparser.RawConfigParser(inline_comment_prefixes=('#', ';'))
    cfg.read(assets / f'{arm}-seed42.ini')
    cfg.set('train', 'train_episodes', '200' if smoke else '10000')
    cfg.set('train', 'save_every', '50' if smoke else '500')
    cfg.set('train', 'eval_every', '100' if smoke else '1000')
    cfg.set('train', 'eval_episodes', '5' if smoke else '50')
    cfg.set('train', 'il_ckpt', str(assets / f'{arm}-seed42/il_policy.pth'))
    cfg.set('train', 'offline_il_dataset', str(assets / 'orca-shared.pth'))
    cfg.set('train', 'il_force_retrain', 'false')
    cfg.set('train', 'cuda_memory_fraction', '0.25')
    cfg.set('imitation_learning', 'post_il_eval_episodes', '5' if smoke else '100')
    path = directory / 'config.ini'
    shutil.copy2(assets / 'env.config', directory / 'env.config')
    with path.open('w') as stream:
        cfg.write(stream)
    return path


def validate_checkpoint(path, arm, episode):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    assert saved['episode'] == episode and saved['stage'] == 'rl_training_sarl'
    assert saved['config']['temporal_backbone'] == arm
    optimizer = saved['optim_value_state']['state']
    assert optimizer and all(float(x['step']) == episode * 4 for x in optimizer.values())
    assert all(torch.isfinite(x).all().item() for x in saved['policy_state'].values() if torch.is_tensor(x))
    return digest(path)


def train_arm(assets, output, arm, smoke):
    directory = output / (('smoke-' if smoke else 'formal-') + arm)
    directory.mkdir(exist_ok=False)
    config = configuration(assets, arm, directory, smoke)
    seconds = run([sys.executable, '-m', 'crowd_nav.train', '--config', config,
                   '--outdir', directory, '--temporal-backbone', arm, '--seed', '42', '--gpu'],
                  directory, 'train')
    episode = 200 if smoke else 10000
    sha = validate_checkpoint(directory / f'rl_model_ep{episode}.pth', arm, episode)
    text = (directory / 'train.log').read_text(errors='replace')
    assert 'Loaded all parameters (strict mode)' in text
    assert 'Successfully loaded BC weights' in text
    assert '[IL-VALUE] epoch=' not in text
    assert 'episodes injected=5000, skipped=0' in text
    result = dict(status='PASS' if smoke else 'TRAINED', arm=arm, episode=episode,
                  optimizer_steps=episode * 4, il_retrained=False,
                  wall_seconds=seconds, checkpoint_sha256=sha)
    save(directory / 'training-result.json', result)
    save(directory / 'state.json', result)
    print(json.dumps(result), flush=True)
    return result


def evaluate_arm(assets, output, arm):
    directory = output / ('formal-' + arm)
    config = assets / f'{arm}-seed42.eval.ini'
    result_csv = directory / 'benchmark-10000.csv'
    seconds = run([sys.executable, '-m', 'crowd_nav.test', '--policy', 'mamba',
                   '--model_dir', directory, '--weights', 'rl_model_ep10000.pth',
                   '--env_config', config, '--policy_config', config, '--temporal-backbone', arm,
                   '--episodes', '500', '--time-limit', '25', '--seed', '20261006',
                   '--no_progress', '--measure-latency', '--oscillation_csv', result_csv,
                   '--run_label', f'{arm}-optimized-rl10000', '--gpu'], directory, 'benchmark')
    with result_csv.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 3000 and len({(r['scenario'], r['episode']) for r in rows}) == 3000
    scenarios = {r['scenario'] for r in rows}
    assert len(scenarios) == 6 and all(sum(r['scenario'] == s for r in rows) == 500 for s in scenarios)
    result = dict(status='COMPLETE', arm=arm, evaluation_wall_seconds=seconds,
                  success=sum(r['outcome'] == 'success' for r in rows),
                  collision=sum(r['outcome'] == 'collision' for r in rows),
                  timeout=sum(r['outcome'] == 'timeout' for r in rows),
                  contact_episodes=sum(int(r['geometric_contact_steps']) > 0 for r in rows))
    assert result['success'] + result['collision'] + result['timeout'] == 3000
    save(directory / 'complete.json', result); save(directory / 'state.json', result)
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--assets-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=('all', 'smoke', 'formal'), default='all')
    args = parser.parse_args()
    assets = args.assets_root.resolve() / 'crowd_nav/runs/bayes-matched-v1'
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    from crowd_nav.contracts import init_grid_from_cfg
    from crowd_nav.policy.mamba_rl import MambaRLPolicy
    assert Path(sys.modules[MambaRLPolicy.__module__].__file__).resolve() == ROOT / 'crowd_nav/policy/mamba_rl.py'
    il = {}
    for arm in ARMS:
        cfg = configparser.RawConfigParser(inline_comment_prefixes=('#', ';'))
        cfg.read(assets / f'{arm}-seed42.ini'); init_grid_from_cfg(cfg)
        checkpoint = assets / f'{arm}-seed42/il_policy.pth'
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert saved['meta']['stage'] == 'bc_policy_only' and saved['meta']['objective'] == 'value_regression'
        MambaRLPolicy(cfg).load_state_dict(saved['value'], strict=True)
        il[arm] = dict(sha256=digest(checkpoint), path=str(checkpoint), metadata=saved['meta'])
    protocol = dict(arms=ARMS, seed=42, target_episodes=10000, smoke_episodes=200,
                    batch_size=256, updates_per_episode=4, no_mamba=True, il=il,
                    data_sha256=digest(assets / 'orca-shared.pth'),
                    sources={name: digest(ROOT / name) for name in SOURCE},
                    initialized_from='original matched IL; not smoke or old RL weights',
                    evaluation='same six native scenarios x 500; reused development test set, not fresh confirmation')
    path = output / 'protocol.json'
    if path.exists():
        assert json.loads(path.read_text()) == json.loads(json.dumps(protocol))
    else:
        save(path, protocol)
    available_kib = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                             if line.startswith('MemAvailable:')))
    if available_kib < 24 * 1024 ** 2 or shutil.disk_usage(ROOT).free < 3 * 1024 ** 3:
        raise RuntimeError('Resource guard: need 24 GiB available RAM and 3 GiB disk for three workers')
    try:
        if args.phase != 'formal':
            verify = [sys.executable, ROOT / 'scripts/verify-bayes-optimization.py']
            run(verify, output, 'parity')
            with ThreadPoolExecutor(max_workers=3) as pool:
                results = list(pool.map(lambda arm: train_arm(assets, output, arm, True), ARMS))
            save(output / 'smoke-results.json', results)
        if args.phase != 'smoke':
            assert (output / 'smoke-results.json').exists()
            assert all(row['status'] == 'PASS' for row in json.loads((output / 'smoke-results.json').read_text()))
            assert all(digest(ROOT / name) == sha for name, sha in protocol['sources'].items())
            with ThreadPoolExecutor(max_workers=3) as pool:
                results = list(pool.map(lambda arm: train_arm(assets, output, arm, False), ARMS))
            save(output / 'training-results.json', results)
            with ThreadPoolExecutor(max_workers=3) as pool:
                results = list(pool.map(lambda arm: evaluate_arm(assets, output, arm), ARMS))
            save(output / 'complete.json', dict(status='COMPLETE', results=results))
    except Exception as error:
        save(output / 'error.json', dict(status='ERROR', error=str(error)))
        raise


if __name__ == '__main__':
    main()
