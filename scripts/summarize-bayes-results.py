"""Analyze saved three-arm results only; never launch training or evaluation."""

import configparser
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RUN = ROOT / 'crowd_nav/runs/bayes-matched-v1'
ARMS = ('gru', 'bayes_mean', 'bayes')
BOOTSTRAPS = 10000


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(arm, stage):
    directory = RUN / (f'{arm}-seed42' if stage == 3000 else f'{arm}-seed42-to8000')
    path = directory / f'benchmark-{stage}.csv'
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    rows.sort(key=lambda row: (row['scenario_index'], int(row['episode'])))
    keys = [(r['scenario'], r['episode'], r['seed']) for r in rows]
    assert len(rows) == len(set(keys)) == 3000
    assert all(sum(r['scenario'] == s for r in rows) == 500 for s in {r['scenario'] for r in rows})
    assert all(r['outcome'] in ('success', 'collision', 'timeout') for r in rows)
    assert all(np.isfinite(float(r[k])) for r in rows for k in
               ('min_clearance_m', 'duration_s', 'path_length_m', 'mean_inference_ms'))
    return rows, digest(path)


def aggregate(rows):
    successes = [r for r in rows if r['outcome'] == 'success']
    mean = lambda field, subset=rows: float(np.mean([float(r[field]) for r in subset])) if subset else None
    steps = sum(int(r['steps']) for r in rows)
    return dict(
        n=len(rows), success=len(successes), collision=sum(r['outcome'] == 'collision' for r in rows),
        timeout=sum(r['outcome'] == 'timeout' for r in rows),
        contact_episodes=sum(int(r['geometric_contact_steps']) > 0 for r in rows),
        contact_steps=sum(int(r['geometric_contact_steps']) for r in rows),
        contact_duration_s=sum(float(r['geometric_contact_duration_s']) for r in rows),
        discomfort_steps=sum(int(r['discomfort_steps']) for r in rows), total_steps=steps,
        discomfort_step_fraction=sum(int(r['discomfort_steps']) for r in rows) / steps,
        min_clearance_m=min(float(r['min_clearance_m']) for r in rows),
        success_time_s=mean('duration_s', successes), all_episode_duration_s=mean('duration_s'),
        all_episode_path_m=mean('path_length_m'), success_path_m=mean('path_length_m', successes),
        episode_mean_clearance_m=mean('mean_clearance_m'),
        inference_step_weighted_ms=sum(float(r['mean_inference_ms']) * int(r['steps']) for r in rows) / steps,
        mean_episode_inference_p95_ms=mean('p95_inference_ms'),
        mean_episode_stall_fraction=mean('stalled_window_ratio'),
        mean_episode_backtracking_fraction=mean('backtracking_ratio'),
        mean_episode_progress_efficiency=mean('goal_progress_efficiency'),
        mean_episode_turn_reversal_hz=mean('turn_reversal_hz'))


def compare(left, right):
    assert [(r['scenario'], r['episode'], r['seed']) for r in left] == [
        (r['scenario'], r['episode'], r['seed']) for r in right]
    features = lambda rows: np.array([
        [r['outcome'] == 'success', r['outcome'] == 'collision',
         int(r['geometric_contact_steps']) > 0] for r in rows], dtype=float)
    diff = features(left) - features(right)
    # Resample paired cases within each fixed scenario, not individual frames.
    rng = np.random.default_rng(20261007)
    samples = []
    for _ in range(BOOTSTRAPS // 250):
        means = np.zeros((250, 3))
        for scenario in sorted({r['scenario'] for r in left}):
            indices = [i for i, r in enumerate(left) if r['scenario'] == scenario]
            means += diff[indices][rng.integers(0, 500, (250, 500))].mean(axis=1) / 6
        samples.append(means)
    intervals = np.percentile(np.concatenate(samples), [2.5, 97.5], axis=0).T * 100
    common = [(l, r) for l, r in zip(left, right) if l['outcome'] == r['outcome'] == 'success']
    transition = {f'{a}->{b}': sum(l['outcome'] == a and r['outcome'] == b for l, r in zip(left, right))
                  for a in ('success', 'collision', 'timeout') for b in ('success', 'collision', 'timeout')}
    return dict(
        difference_pp=dict(zip(('SR', 'CR', 'geometric_contact'), (diff.mean(axis=0) * 100).tolist())),
        exploratory_case_bootstrap_95_ci_pp=dict(zip(('SR', 'CR', 'geometric_contact'), intervals.tolist())),
        left_only_success=sum(l['outcome'] == 'success' and r['outcome'] != 'success' for l, r in zip(left, right)),
        right_only_success=sum(r['outcome'] == 'success' and l['outcome'] != 'success' for l, r in zip(left, right)),
        common_success=len(common),
        common_success_time_difference_s=float(np.mean([float(l['duration_s']) - float(r['duration_s']) for l, r in common])),
        common_success_path_difference_m=float(np.mean([float(l['path_length_m']) - float(r['path_length_m']) for l, r in common])),
        outcome_transition_left_to_right=transition)


def training(arm):
    result = {}
    for stage, suffix in ((3000, ''), (8000, '-to8000')):
        text = (RUN / f'{arm}-seed42{suffix}/train.log').read_text(errors='replace')
        validation = re.findall(
            r'\[EVAL\] Starting evaluation: ep=(\d+).*?\[EVAL\] Completed evaluation: succ=([\d.]+) coll=([\d.]+) timeout=([\d.]+)',
            text, flags=re.S)
        losses = re.findall(r'\[SARL-LOSS\] ep=(\d+) value_loss=([\d.eE+-]+)', text)
        result[str(stage)] = dict(
            validation=[dict(episode=int(ep), sr=float(s), cr=float(c), tr=float(t)) for ep, s, c, t in validation
                        if int(ep) <= stage],
            value_loss=[dict(episode=int(ep), mse=float(loss)) for ep, loss in losses if int(ep) <= stage],
            il_value_losses=[dict(epoch=int(ep), mse=float(loss)) for ep, loss in re.findall(
                r'\[IL-VALUE\] epoch=(\d+)/50 value_loss=([\d.eE+-]+)', text)],
            best_il_epoch=re.findall(r'Using best checkpoint from epoch (\d+)', text))
    result['additional_training'] = json.loads((RUN / f'{arm}-seed42-to8000/training-8000.json').read_text())
    result['evaluation'] = json.loads((RUN / f'{arm}-seed42-to8000/complete-8000.json').read_text())['eval_wall_seconds']
    return result


def checkpoint_audit(arm):
    from crowd_nav.contracts import init_grid_from_cfg
    from crowd_nav.policy.mamba_rl import MambaRLPolicy
    assert Path(sys.modules[MambaRLPolicy.__module__].__file__).resolve() == ROOT / 'crowd_nav/policy/mamba_rl.py'
    config = configparser.RawConfigParser(inline_comment_prefixes=('#', ';'))
    config.read(RUN / f'{arm}-seed42.ini')
    assert config.get('mamba', 'temporal_backbone') == arm
    init_grid_from_cfg(config)
    policy = MambaRLPolicy(config)
    path = RUN / f'{arm}-seed42-to8000/rl_model_ep8000.pth'
    saved = torch.load(path, map_location='cpu', weights_only=False)
    assert saved['episode'] == 8000 and saved['config']['temporal_backbone'] == arm
    policy.load_state_dict(saved['policy_state'], strict=True)
    assert all(torch.isfinite(v).all().item() for v in saved['policy_state'].values() if torch.is_tensor(v))
    assert all(torch.isfinite(v).all().item() for v in saved['target_value_net_state'].values() if torch.is_tensor(v))
    optimizer = saved['optim_value_state']['state']
    assert all(float(v['step']) == 32000 for v in optimizer.values())
    expected = json.loads((path.parent / 'complete-8000.json').read_text())['checkpoint_sha256']
    assert digest(path) == expected
    return dict(sha256=expected, bytes=path.stat().st_size,
                registered_parameters=sum(p.numel() for p in policy.parameters()),
                temporal_parameters=sum(p.numel() for p in policy.temporal_encoder.parameters()),
                optimizer_updated_parameters=sum(v['exp_avg'].numel() for v in optimizer.values()),
                optimizer_parameter_states=len(optimizer), optimizer_steps=32000,
                strict_loading=True, policy_and_target_finite=True)


def main():
    torch.set_num_threads(2)
    protocol = json.loads((RUN / 'protocol.json').read_text())
    assert all(digest(ROOT / name) == sha for name, sha in protocol['sources'].items())
    result = dict(version='saved-three-arm-analysis-v1', training_seed=42,
                  uncertainty_interval='Exploratory paired scenario-stratified case bootstrap; not training-seed variability or fresh confirmation',
                  bootstrap_replicates=BOOTSTRAPS, bootstrap_seed=20261007,
                  cohorts={}, pairs={}, stage_changes={}, training={}, checkpoints={})
    all_rows = {}
    for stage in (3000, 8000):
        result['cohorts'][str(stage)] = {}
        for arm in ARMS:
            rows, sha = load(arm, stage)
            all_rows[stage, arm] = rows
            result['cohorts'][str(stage)][arm] = dict(
                csv_sha256=sha, aggregate=aggregate(rows),
                scenarios={s: aggregate([r for r in rows if r['scenario'] == s]) for s in sorted({r['scenario'] for r in rows})})
        result['pairs'][str(stage)] = {
            f'{left}_minus_{right}': compare(all_rows[stage, left], all_rows[stage, right])
            for left, right in (('bayes_mean', 'gru'), ('bayes', 'gru'), ('bayes', 'bayes_mean'))}
    for arm in ARMS:
        result['stage_changes'][arm] = compare(all_rows[8000, arm], all_rows[3000, arm])
        result['training'][arm] = training(arm)
        result['checkpoints'][arm] = checkpoint_audit(arm)
    path = RUN / 'comprehensive-analysis.json'
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)
    print(path)


if __name__ == '__main__':
    main()
