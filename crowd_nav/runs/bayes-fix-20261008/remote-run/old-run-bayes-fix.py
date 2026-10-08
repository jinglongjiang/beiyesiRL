"""Pinned IL, smoke, six matched RL workers, then the native 3000-case test."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import configparser
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import shutil
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ARMS = ('gru', 'bayes_mean', 'bayes')
SEEDS = (42, 43)
BASE = ROOT.parent
OUT = BASE / 'runs'
CONFIGS = BASE / 'configs'
DATA = Path('/root/bayes-assets/crowd_nav/runs/bayes-matched-v1/orca-shared.pth')
ABORT = threading.Event()


def digest(path):
    value = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    assert value
    return value


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def identity(pid):
    try:
        fields = Path('/proc/{}/stat'.format(pid)).read_text().rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except FileNotFoundError:
        return None


def pins(config):
    import crowd_nav
    import crowd_sim
    import crowd_nav.policy.mamba_rl as policy
    from crowd_nav import contracts
    for module in (crowd_nav, crowd_sim, policy):
        Path(module.__file__).resolve().relative_to(ROOT)
    cfg = configparser.RawConfigParser(inline_comment_prefixes=('#', ';'))
    assert cfg.read(str(config))
    contracts.init_grid_from_cfg(cfg)
    expected = cfg.getint('policy', 'n_speeds') * cfg.getint('policy', 'n_headings')
    expected += int(cfg.getboolean('policy', 'include_stop'))
    assert contracts.grid_action_dim(contracts.GRID) == expected == 80
    assert contracts.GRID['sampling'] == 'exponential' and contracts.GRID['v_min'] == .05
    print('[GRID-GUARD] num_actions=80 project={}'.format(ROOT), flush=True)
    return cfg


def guarded_worker(mode, command):
    key = '--config' if mode == 'train' else '--policy_config'
    pins(command[command.index(key) + 1])
    import torch
    torch.set_num_threads(2)
    sys.argv = ['crowd_nav.' + mode] + command
    runpy.run_module('crowd_nav.' + mode, run_name='__main__')


def run(directory, label, mode, command):
    directory.mkdir(parents=True, exist_ok=True)
    log = directory / (label + '.stdout.log')
    assert not log.exists(), 'Do not overwrite an old run: ' + str(log)
    env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
               MPLBACKEND='Agg', PYTHONUNBUFFERED='1', TF_CPP_MIN_LOG_LEVEL='3')
    invocation = [sys.executable, str(Path(__file__).resolve()), 'worker', mode] + list(map(str, command))
    started = time.time()
    with log.open('w') as stream:
        proc = subprocess.Popen(invocation, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        record = dict(status=label.upper(), pid=proc.pid, proc_start_ticks=identity(proc.pid),
                      started_unix=started, command=invocation)
        assert record['proc_start_ticks'] is not None
        save(directory / 'process.json', record)
        if label == 'formal':
            with (BASE / 'pids.txt').open('a') as pid_stream:
                pid_stream.write(str(proc.pid) + '\n')
        try:
            while proc.poll() is None:
                if ABORT.is_set():
                    raise RuntimeError('A sibling worker failed its engineering contract')
                text = log.read_text(errors='replace')
                forbidden = ('trying partial load', 'Will train from scratch', 'Failed to load checkpoint',
                             'Failed to prefill buffer', 'Failed to store trajectory',
                             'using online', 'Missing action_indices', 'Traceback (most recent call last)')
                if any(term in text for term in forbidden):
                    raise RuntimeError('Engineering contract failed: ' + str(log))
                if shutil.disk_usage(BASE).free < 1024 ** 3:
                    raise RuntimeError('Disk guard: less than 1 GiB free')
                available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                                     if line.startswith('MemAvailable:'))) * 1024
                if available < 2 * 1024**3:
                    raise RuntimeError('Memory guard: less than 2 GiB available')
                save(directory / 'live.json', dict(record, elapsed_seconds=time.time()-started,
                    disk_free_bytes=shutil.disk_usage(BASE).free))
                time.sleep(15)
            if proc.returncode:
                raise RuntimeError('{} exited {}: {}'.format(label, proc.returncode, log))
        except Exception:
            ABORT.set()
            raise
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
    save(directory / 'process.json', dict(record, status='EXITED_OK', returncode=0))
    return time.time() - started


def config_for(arm, seed, phase):
    cfg = configparser.RawConfigParser(inline_comment_prefixes=('#', ';'))
    assert cfg.read(str(CONFIGS / '{}-seed{}.ini'.format(arm, seed)))
    cfg.set('train', 'seed', str(seed))
    cfg.set('train', 'offline_il_dataset', str(DATA))
    cfg.set('train', 'il_ckpt', str(OUT / 'il-{}-seed{}'.format(arm, seed) / 'il_policy.pth'))
    cfg.set('train', 'il_force_retrain', 'false')
    cfg.set('train', 'cuda_memory_fraction', '0.15')
    cfg.set('train', 'train_episodes', '200' if phase == 'smoke' else '10000')
    cfg.set('train', 'save_every', '50' if phase == 'smoke' else '500')
    cfg.set('train', 'eval_every', '100' if phase == 'smoke' else '1000')
    cfg.set('train', 'eval_episodes', '5' if phase == 'smoke' else '50')
    directory = OUT / '{}-{}-seed{}'.format(phase, arm, seed)
    directory.mkdir(parents=True, exist_ok=False)
    with (directory / 'config.ini').open('w') as stream:
        cfg.write(stream)
    shutil.copy2(CONFIGS / 'env.config', directory / 'env.config')
    return directory


def il_check(arm, seed):
    import torch
    from crowd_nav.policy.mamba_rl import MambaRLPolicy
    directory = OUT / 'il-{}-seed{}'.format(arm, seed)
    cfg = pins(directory / 'config.ini')
    path = directory / 'il_policy.pth'
    saved = torch.load(path, map_location='cpu', weights_only=False)
    assert saved['meta']['stage'] == 'bc_policy_only'
    assert saved['meta']['objective'] == 'value_regression'
    assert 1 <= saved['meta']['epochs'] <= 50
    assert '[IL-VALUE] epoch=50/50' in (directory / 'train.log').read_text(errors='replace')
    MambaRLPolicy(cfg, device='cpu').load_state_dict(saved['value'], strict=True)
    assert all(torch.isfinite(v).all() for v in saved['value'].values() if torch.is_tensor(v))
    return dict(checkpoint_sha256=digest(path), metadata=saved['meta'])


def checkpoint_check(directory, arm, episodes):
    import torch
    saved = torch.load(directory / 'rl_model_ep{}.pth'.format(episodes), map_location='cpu', weights_only=False)
    assert saved['episode'] == episodes and saved['stage'] == 'rl_training_sarl'
    assert saved['config']['temporal_backbone'] == arm
    opt = saved['optim_value_state']['state']
    assert opt and all(float(v['step']) == episodes * 4 for v in opt.values())
    assert all(torch.isfinite(v).all() for v in saved['policy_state'].values() if torch.is_tensor(v))
    text = (directory / 'train.log').read_text(errors='replace')
    for term in ('num_actions=80', 'Built 80 actions', 'matched-native-orca-v1',
                 'Loaded all parameters (strict mode)', 'episodes injected=5000, skipped=0'):
        assert term in text, 'Missing ' + term
    assert '[IL-VALUE] epoch=' not in text, 'RL must reuse the new matched IL checkpoint'
    events = re.findall(r'RESULT=(SUCCESS|COLLISION|TIMEOUT)\b', (directory / 'trajectory.log').read_text())
    assert len(events) == episodes
    counts = {event.lower(): events.count(event) for event in ('SUCCESS', 'COLLISION', 'TIMEOUT')}
    if episodes == 200:
        assert 0 < counts['success'] < episodes, 'Degenerate smoke success rate'
    return dict(arm=arm, episodes=episodes, optimizer_steps=episodes*4, **counts)


def train_one(arm, seed, phase):
    directory = config_for(arm, seed, phase)
    command = ['--config', directory / 'config.ini', '--outdir', directory,
               '--temporal-backbone', arm, '--seed', seed, '--gpu']
    if phase == 'il':
        command.append('--pretrain-only')
    seconds = run(directory, phase, 'train', command)
    if phase == 'il':
        result = dict(status='IL_VALID', arm=arm, seed=seed, **il_check(arm, seed))
    else:
        result = dict(status='PASS' if phase == 'smoke' else 'TRAINED', seed=seed,
                      **checkpoint_check(directory, arm, 200 if phase == 'smoke' else 10000))
    result['wall_seconds'] = seconds
    save(directory / 'result.json', result)
    print(json.dumps(result), flush=True)
    return result


def evaluate_one(pair):
    arm, seed = pair
    directory = OUT / 'formal-{}-seed{}'.format(arm, seed)
    config = CONFIGS / '{}-seed{}.eval.ini'.format(arm, seed)
    result_csv = directory / 'benchmark-10000.csv'
    seconds = run(directory, 'benchmark', 'test', [
        '--policy', 'mamba', '--model_dir', directory, '--weights', 'rl_model_ep10000.pth',
        '--env_config', config, '--policy_config', config, '--temporal-backbone', arm,
        '--episodes', '500', '--time-limit', '25', '--seed', '20261006', '--no_progress',
        '--measure-latency', '--oscillation_csv', result_csv,
        '--run_label', '{}-fix-seed{}'.format(arm, seed), '--gpu'])
    with result_csv.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 3000 and len({(r['scenario'], r['episode']) for r in rows}) == 3000
    scenes = {r['scenario'] for r in rows}
    assert len(scenes) == 6 and all(sum(r['scenario'] == s for r in rows) == 500 for s in scenes)
    result = dict(status='EVALUATED', arm=arm, seed=seed, seconds=seconds,
        success=sum(r['outcome'] == 'success' for r in rows),
        collision=sum(r['outcome'] == 'collision' for r in rows),
        timeout=sum(r['outcome'] == 'timeout' for r in rows),
        geometric_contact_episodes=sum(int(r['geometric_contact_steps']) > 0 for r in rows))
    assert result['success'] + result['collision'] + result['timeout'] == 3000
    save(directory / 'benchmark-result.json', result)
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert identity(os.getpid()) is not None and identity(2147483647) is None
    assert json.loads((BASE / 'b0-native.json').read_text())['status'] == 'PASS'
    assert json.loads((BASE / 'self-tests.json').read_text())['status'] == 'PASS'
    preregistered = json.loads((BASE / 'protocol.json').read_text())
    assert digest(DATA) == preregistered['dataset_sha256']
    import torch
    assert torch.load(DATA, map_location='cpu', weights_only=False)['version'] == 'matched-native-orca-v1'
    assert digest(ROOT / 'crowd_nav/policy/bayes_temporal.py') == json.loads(
        (BASE / 'self-tests.json').read_text())['readout_source_sha256']
    files = list((ROOT / 'crowd_nav').rglob('*.py')) + list((ROOT / 'crowd_sim').rglob('*.py'))
    files.append(Path(__file__).resolve())
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in files}
    configs = {str(p.name): digest(p) for p in CONFIGS.iterdir() if p.is_file()}
    common = None
    for seed in SEEDS:
        for arm in ARMS:
            cfg = pins(CONFIGS / '{}-seed{}.ini'.format(arm, seed))
            normalized = {section: dict(cfg.items(section)) for section in cfg.sections()}
            normalized['train'].pop('seed')
            normalized['mamba'].pop('temporal_backbone')
            # The old laptop templates have a different allocator cap. Every
            # generated server run below explicitly uses the same 0.15 cap.
            normalized['train']['cuda_memory_fraction'] = '0.15'
            if common is None:
                common = normalized
            assert normalized == common, 'Unmatched scientific hyperparameters'
    version = dict(source_sha256=hashes, configs_sha256=configs, protocol=preregistered,
                   formal_init='new 50-epoch matched IL, never smoke RL', seeds=SEEDS, arms=ARMS)
    assert not (BASE / 'run-version.json').exists(), 'Do not restart/overwrite an old night run'
    save(BASE / 'run-version.json', version)
    if shutil.disk_usage(BASE).free < 3 * 1024**3:
        raise RuntimeError('Need at least 3 GiB free before startup')
    available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:'))) * 1024
    assert available > 28 * 1024**3, 'Insufficient RAM for the planned six workers'
    try:
        for seed in SEEDS:
            save(BASE / 'controller-state.json', dict(stage='IL', seed=seed, pid=os.getpid()))
            with ThreadPoolExecutor(max_workers=3) as pool:
                list(pool.map(lambda arm: train_one(arm, seed, 'il'), ARMS))
            if seed == 42:
                save(BASE / 'controller-state.json', dict(stage='SMOKE', seed=42, pid=os.getpid()))
                with ThreadPoolExecutor(max_workers=3) as pool:
                    results = list(pool.map(lambda arm: train_one(arm, 42, 'smoke'), ARMS))
                save(BASE / 'smoke-results.json', results)
        assert all(digest(ROOT / name) == value for name, value in hashes.items())
        assert all(row['status'] == 'PASS' for row in json.loads((BASE / 'smoke-results.json').read_text()))
        save(BASE / 'controller-state.json', dict(stage='FORMAL', seeds=SEEDS, pid=os.getpid()))
        with ThreadPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(lambda pair: train_one(*pair, 'formal'),
                                    [(arm, seed) for seed in SEEDS for arm in ARMS]))
        save(BASE / 'training-results.json', results)
        assert all(digest(ROOT / name) == value for name, value in hashes.items())
        save(BASE / 'controller-state.json', dict(stage='BENCHMARK_3000', pid=os.getpid()))
        with ThreadPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(evaluate_one, [(arm, seed) for seed in SEEDS for arm in ARMS]))
        save(BASE / 'complete.json', dict(status='COMPLETE', results=results))
    except Exception as error:
        save(BASE / 'error.json', dict(status='ENGINEERING_STOP', error=str(error)))
        raise


def mirror(destination):
    import fcntl
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    lock = (destination / 'mirror.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    command = ['rsync', '-az', '--timeout=60', '--exclude=code/', '--exclude=scripts/',
               '-e', 'ssh -o BatchMode=yes -o ConnectTimeout=15',
               'root@45.126.120.6:/root/bayes-fix-20261008/', str(destination) + '/']
    while True:
        try:
            subprocess.run(command, check=True, timeout=300)
            print(time.strftime('%Y-%m-%d %H:%M:%S'), 'synchronized', flush=True)
            if (destination / 'complete.json').exists() or (destination / 'error.json').exists():
                return
        except Exception as error:
            print('Mirror error:', error, flush=True)
        time.sleep(120)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'worker':
        guarded_worker(sys.argv[2], sys.argv[3:])
    elif len(sys.argv) > 1 and sys.argv[1] == 'mirror':
        mirror(sys.argv[2])
    else:
        main()
