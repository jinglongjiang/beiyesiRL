"""Frozen four-arm training/evaluation using the native CrowdNav entry points."""

import argparse
import configparser
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'crowd_nav/runs/bayes-matched-v1'
ARMS = ('mamba', 'gru', 'bayes_mean', 'bayes')
SEEDS = (42, 43)
DATA = RUN / 'orca-shared.pth'
SOURCE_FILES = (
    'crowd_nav/policy/mamba_rl.py', 'crowd_nav/policy/bayes_temporal.py',
    'crowd_nav/train.py', 'crowd_nav/test.py', 'crowd_nav/utils/ppo_buffer.py',
    'crowd_nav/contracts.py', 'crowd_nav/utils/explorer.py',
    'crowd_nav/configs/env.config', 'crowd_nav/configs/policy.config',
    'crowd_nav/configs/train.config', 'scripts/run-bayes-matched.py',
)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def configuration(arm, seed, collect=False):
    cfg = configparser.RawConfigParser(inline_comment_prefixes=('#', ';'))
    cfg.read([str(ROOT / 'crowd_nav/configs' / name)
              for name in ('env.config', 'policy.config', 'train.config')])
    values = {
        'temporal': {'history_contract': 'legal-prefix'},
        'mamba': {'temporal_backbone': arm},
        'train': {
            'seed': str(seed), 'matched_rng': 'true', 'compile_policy': 'false',
            'il_data_on_cpu': 'true', 'cuda_memory_fraction': '0.15' if seed == 42 else '0.8',
            'train_episodes': '10000', 'il_epochs': '50', 'plot_every': '1000',
            'offline_il_dataset': str(DATA.relative_to(ROOT)),
        },
        'imitation_learning': {
            'force_online': 'true' if collect else 'false',
            'success_target': '5000', 'max_il_prefill': '50000',
            'max_prefill_episodes': '12000', 'prefill_batch_episodes': '64',
        },
    }
    for section, options in values.items():
        if not cfg.has_section(section):
            cfg.add_section(section)
        for key, value in options.items():
            cfg.set(section, key, value)
    return cfg


def prepare():
    RUN.mkdir(parents=True, exist_ok=True)
    for name in ('env.config', 'policy.config', 'train.config'):
        shutil.copy2(ROOT / 'crowd_nav/configs' / name, RUN / name)
    protocol = {
        'version': 'bayes-temporal-v1-matched', 'arms': ARMS, 'paired_seeds': SEEDS,
        'parent': 'user-supplied CrowdNav Mamba-VL backup; no external Bayesian Parent',
        'dataset_seed': 271828, 'dataset_success_target': 5000,
        'il_epochs': 50, 'il_batch_size': 256, 'rl_episodes': 10000,
        'development_checkpoint': 3000, 'save_every': 500, 'updates_per_episode': 4,
        'rl_batch_size': 256, 'replay_capacity': 200000, 'gamma': 0.99,
        'eval_every': 1000, 'validation_episodes': 50,
        'final_checkpoint': 'rl_model_ep10000.pth, not best-on-test checkpoint',
        'final_scenarios': ['baseline_circle', 'baseline_square', 'dense_circle',
                            'dense_square', 'large_circle', 'large_square'],
        'final_episodes_per_scenario': 500, 'final_time_limit_s': 25,
        'final_eval_seed': 20261006, 'final_test_pool_size': 1000,
        'final_case_rule': 'native indices 412+3*scenario_id through 911+3*scenario_id; disjoint from IL quick-test selection roots',
        'human_policy': 'original native ORCA',
        'observation_noise': 0, 'behavior_profile': 'nominal',
        'history_contract': 'right-aligned real suffix; zero prefix is excluded before temporal encoding',
        'exploration_contract': 'append one legal current observation even for epsilon actions',
        'inherited_test_safety_filter': True,
        'shared_changes': ['legal-prefix', 'exploration-clock', 'action-index return',
                           'fixed IL epoch count', 'independent common IL shuffle RNG',
                           'common post-IL RNG reset', 'compile disabled on all arms'],
        'online_replay': 'same protocol, not identical divergent on-policy trajectories',
        'capacity_matched': False, 'uncertainty_calibrated': False,
        'il_storage': 'CPU corpus; identical per-minibatch transfer to the training device',
        'memory_cap_policy': 'same cap for all four arms within a host/seed',
        'platform_policy': 'all four arms per paired seed on one common host/runtime; report seeds separately',
        'stage_a_rule': 'save/evaluate at 3000; no stop solely because V1 trails Mamba',
        'v2_rule': 'no automatic hyperparameter search; one evidence-based structural repair after complete V1 comparison',
        'stop_rules': ['engineering invalidity', 'nonfinite loss/parameters', 'actual memory/disk risk'],
        'sources': {name: digest(ROOT / name) for name in SOURCE_FILES},
    }
    path = RUN / 'protocol.json'
    if path.exists():
        old = json.loads(path.read_text())
        if old != json.loads(json.dumps(protocol)):
            raise RuntimeError('Frozen protocol/source mismatch; do not overwrite a live experiment')
    else:
        write_json(path, protocol)
    paths = [('collect.ini', configuration('bayes_mean', 271828, collect=True))]
    paths += [(f'{arm}-seed{seed}.ini', configuration(arm, seed))
              for seed in SEEDS for arm in ARMS]
    for seed in SEEDS:
        for arm in ARMS:
            evaluation = configuration(arm, seed)
            evaluation.set('env', 'time_limit', '25')
            evaluation.set('env', 'test_size', '1000')
            paths.append((f'{arm}-seed{seed}.eval.ini', evaluation))
    for name, cfg in paths:
        path = RUN / name
        with path.open('w') as stream:
            cfg.write(stream)
    print('Frozen protocol:', RUN / 'protocol.json', flush=True)


def check_sources():
    protocol = json.loads((RUN / 'protocol.json').read_text())
    for name, expected in protocol['sources'].items():
        if digest(ROOT / name) != expected:
            raise RuntimeError('Source changed after freeze: ' + name)
    return protocol


def invoke(command, log, label):
    start = time.time()
    print(label, ' '.join(map(str, command)), flush=True)
    env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='2', MPLBACKEND='Agg', PYTHONUNBUFFERED='1')
    with Path(log).open('a') as stream:
        result = subprocess.run(list(map(str, command)), cwd=ROOT, env=env,
                                stdout=stream, stderr=subprocess.STDOUT)
    elapsed = time.time() - start
    record = {'label': label, 'exit_code': result.returncode, 'wall_seconds': elapsed,
              'command': list(map(str, command)), 'log': str(Path(log).relative_to(ROOT))}
    with (RUN / 'commands.jsonl').open('a') as stream:
        stream.write(json.dumps(record) + '\n')
    if result.returncode:
        raise RuntimeError(f'{label} failed with exit {result.returncode}; see {log}')
    return elapsed


def collect():
    check_sources()
    if DATA.exists():
        raise FileExistsError('Shared data already exists; refusing recollection')
    invoke([sys.executable, '-m', 'crowd_nav.train', '--config', RUN / 'collect.ini',
            '--outdir', RUN / 'collection', '--temporal-backbone', 'bayes_mean',
            '--seed', 271828, '--collect-il-only', '--cpu'],
           RUN / 'collection.log', 'shared-native-orca-collection')
    write_json(RUN / 'dataset.json', {'sha256': digest(DATA), 'path': str(DATA.relative_to(ROOT))})


def benchmark_summary(csv_path):
    with Path(csv_path).open() as stream:
        rows = list(csv.DictReader(stream))
    summary = {}
    for scenario in sorted({r['scenario'] for r in rows}):
        group = [r for r in rows if r['scenario'] == scenario]
        successful = [r for r in group if r['outcome'] == 'success']
        n = len(group)
        summary[scenario] = {
            'episodes': n,
            'success': len(successful),
            'collision': sum(r['outcome'] == 'collision' for r in group),
            'timeout': sum(r['outcome'] == 'timeout' for r in group),
            'sr': len(successful) / n,
            'cr': sum(r['outcome'] == 'collision' for r in group) / n,
            'tr': sum(r['outcome'] == 'timeout' for r in group) / n,
            'mean_success_time_s': (sum(float(r['duration_s']) for r in successful) / len(successful)
                                    if successful else None),
            'mean_all_episode_path_m': sum(float(r['path_length_m']) for r in group) / n,
            'minimum_clearance_m': min(float(r['min_clearance_m']) for r in group),
            'geometric_contact_episodes': sum(int(r['geometric_contact_steps']) > 0 for r in group),
            'geometric_contact_steps': sum(int(r['geometric_contact_steps']) for r in group),
            'discomfort_steps': sum(int(r['discomfort_steps']) for r in group),
            'total_steps': sum(int(r['steps']) for r in group),
        }
    if len(rows) != 3000 or len(summary) != 6 or any(g['episodes'] != 500 for g in summary.values()):
        raise RuntimeError('Incomplete benchmark; do not report it as 3000 episodes')
    return summary


def train(seed, device, arms):
    protocol = check_sources()
    if not DATA.exists() or digest(DATA) != json.loads((RUN / 'dataset.json').read_text())['sha256']:
        raise RuntimeError('Shared IL dataset missing or hash mismatch')
    import torch
    metadata = {'seed': seed, 'device': device, 'hostname': os.uname().nodename,
                'python': sys.version, 'torch': torch.__version__,
                'cuda': torch.version.cuda,
                'gpu': torch.cuda.get_device_name(0) if device == 'cuda' else None}
    try:
        import mamba_ssm
        metadata['mamba_ssm'] = mamba_ssm.__version__
    except ImportError:
        metadata['mamba_ssm'] = None
    for arm in arms:
        write_json(RUN / f'platform-{arm}-seed{seed}.json', metadata)
        output = RUN / f'{arm}-seed{seed}'
        done = output / 'complete.json'
        if done.exists():
            print('Already complete:', arm, seed, flush=True)
            continue
        output.mkdir(exist_ok=True)
        config = RUN / f'{arm}-seed{seed}.ini'
        checkpoint = output / 'rl_model_ep10000.pth'
        train_seconds = None
        if not checkpoint.exists():
            # Restoring only weights would not restore replay/RNG; never claim exact resume.
            if (output / 'il_policy.pth').exists():
                raise RuntimeError('Interrupted arm requires explicit recovery; do not silently restart/reseed')
            write_json(output / 'state.json', {'status': 'TRAINING', 'arm': arm, 'seed': seed,
                                              'protocol_sha256': digest(RUN / 'protocol.json')})
            train_seconds = invoke([sys.executable, '-m', 'crowd_nav.train', '--config', config,
                                    '--outdir', output, '--temporal-backbone', arm, '--seed', seed,
                                    '--gpu' if device == 'cuda' else '--cpu'],
                                   output / 'stdout.log', f'{arm}-seed{seed}-training')
            if not checkpoint.exists():
                raise RuntimeError('Training returned without the full-budget checkpoint')
        result_csv = output / 'benchmark.csv'
        eval_config = RUN / f'{arm}-seed{seed}.eval.ini'
        eval_seconds = invoke([sys.executable, '-m', 'crowd_nav.test', '--policy', 'mamba',
                               '--model_dir', output, '--weights', checkpoint.name,
                               '--env_config', eval_config, '--policy_config', eval_config,
                               '--temporal-backbone', arm, '--episodes', 500,
                               '--time-limit', protocol['final_time_limit_s'],
                               '--seed', protocol['final_eval_seed'], '--no_progress',
                               '--measure-latency', '--oscillation_csv', result_csv,
                               '--run_label', f'{arm}-seed{seed}'] + (['--gpu'] if device == 'cuda' else []),
                              output / 'benchmark.log', f'{arm}-seed{seed}-benchmark')
        write_json(done, {'arm': arm, 'seed': seed, 'train_wall_seconds': train_seconds,
                          'eval_wall_seconds': eval_seconds, 'checkpoint_sha256': digest(checkpoint),
                          'platform': metadata, 'results': benchmark_summary(result_csv)})
        write_json(output / 'state.json', {'status': 'COMPLETE', 'arm': arm, 'seed': seed})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('prepare', 'collect', 'train'))
    parser.add_argument('--seed', type=int, choices=SEEDS, default=42)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    parser.add_argument('--parallel', type=int, choices=(1, 2, 4), default=1)
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare()
    elif args.mode == 'collect':
        collect()
    else:
        if args.parallel == 1:
            train(args.seed, args.device, args.arms)
        else:
            with ThreadPoolExecutor(max_workers=args.parallel) as executor:
                futures = [executor.submit(train, args.seed, args.device, [arm]) for arm in args.arms]
                for future in futures:
                    future.result()


if __name__ == '__main__':
    main()
