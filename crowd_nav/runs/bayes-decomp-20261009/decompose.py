"""Read-only runtime interventions in the repaired Bayesian readout."""

import os
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys
sys.dont_write_bytecode = True
import argparse
import collections
import configparser
import csv
import hashlib
import importlib.util
import json
import logging
import time
import types
from pathlib import Path

import numpy as np
import torch

BASE = Path(__file__).resolve().parent
ROOT = Path('/home/abc/workspace/CrowdNav(20270731_backup2)/CrowdNav')
BACKUP = Path('/home/abc/4090_backup_20261009/root/bayes-fix-20261008/runs')
CONFIG = ROOT / 'crowd_nav/runs/bayes-fix-20261008/remote-run/configs'
CONDITIONS = ('real', 'fixed', 'shuffle_within', 'ctrl_shuffle_mean')
SCENARIOS = (
    ('baseline_circle', 'circle_crossing', 5, 'circle_radius', 4),
    ('baseline_square', 'square_crossing', 10, 'square_width', 10),
    ('dense_circle', 'circle_crossing', 10, 'circle_radius', 4),
    ('dense_square', 'square_crossing', 20, 'square_width', 10),
    ('large_circle', 'circle_crossing', 12, 'circle_radius', 6),
    ('large_square', 'square_crossing', 20, 'square_width', 14),
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def load(path):
    return json.loads(Path(path).read_text())


def source_files():
    return sorted(p for folder in ('crowd_nav', 'crowd_sim', 'scripts')
                  for p in (ROOT / folder).rglob('*')
                  if p.is_file() and p.suffix in ('.py', '.config', '.ini')
                  and 'runs' not in p.relative_to(ROOT).parts
                  and '__pycache__' not in p.parts)


def freeze():
    assert not (BASE / 'protocol.json').exists(), 'Frozen protocol cannot be overwritten'
    assets = []
    for seed in (42, 43):
        assets.extend([BACKUP / ('formal-bayes-seed%d' % seed) / 'rl_model_ep10000.pth',
                       BACKUP / ('formal-bayes-seed%d' % seed) / 'benchmark-10000.csv',
                       CONFIG / ('bayes-seed%d.eval.ini' % seed)])
    paths = source_files() + assets + [Path(__file__)]
    manifest = {str(p): {'sha256': sha(p), 'bytes': p.stat().st_size} for p in paths}
    for seed in (42, 43):
        checkpoint = BACKUP / ('formal-bayes-seed%d' % seed) / 'rl_model_ep10000.pth'
        assert manifest[str(checkpoint)]['sha256'], 'Empty SHA not allowed'
    save(BASE / 'frozen-manifest.json', manifest)
    save(BASE / 'protocol.json', {
        'version': 'bayes-decomp-v1-20261009', 'created_unix': time.time(),
        'root_read_only': str(ROOT), 'device': 'local RTX3060, no remote access, no training',
        'training_seeds': [42, 43], 'scenarios': SCENARIOS,
        'evaluation_cases': list(range(100)), 'smoke_cases': list(range(3)),
        'calibration_cases': list(range(400, 416)),
        'episode_seed': 'Exact seed from existing benchmark CSV per scene/case; base 1540568412+',
        'fixed': 'Per-training-seed 64D mean of logvar AFTER clamp/log, over ALL candidate rows on independent calibration steps; never average variance then log',
        'shuffle_within': 'One shared random permutation of 80 entire logvar vectors within current decision; mean unchanged',
        'ctrl_shuffle_mean': 'Permute 80 entire mean vectors; logvar unchanged; positive control only',
        'randomness': 'Independent numpy Generator keyed by training seed, scene index, case index, tick, condition; never global env RNG',
        'executed_policy': 'Always unmodified real policy. Intervention first actions are shadows on same root; no intervention history writes',
        'ranking': 'Identical native reward, gamma, safety mask/risk penalty, index0 tie penalty. Original action smoothing preserved',
        'metrics': {
            'flip': 'candidate argmax differs from real; also save pre/post safety-filter ranking',
            'primary_clearance': 'All-human actual next-position endpoint clearance under same preceding action smoothing; human action computed before ego execution, independent of candidate',
            'reward': 'Exact native immediate reward: swept current human velocity geometry, timeout/collision/goal priorities, original shaping. Must match real env.step reward',
            'additional': 'Candidate dmins from inherited CV lookahead; native swept clearance; no neural value used to judge benefit',
            'windows': ['all', 'collision_pre4', 'collision_pre8', 'timeout_pre4', 'timeout_pre8', 'success_all', 'success_last8'],
            'strata': ['training seed', 'six scenarios', 'human count 5/10/12/20'],
            'ties': 'Absolute difference <=1e-10 treated as tie; win rate among nonties and all flipped steps both reported',
            'inference': 'Exact two-sided binomial against0.5 descriptive only, frames not iid; primary uncertainty episode-cluster paired bootstrap2000 draws per seed/condition/metric',
            'multiplicity': 'Primary 2 variance conditions x2 metrics x2 seeds=8 comparisons; report nominal and Bonferroni8 p. Strata descriptive, no cherry-picking',
        },
        'controls': {
            'real_twice': 'Each episode replayed twice; exact equality of all real values/actions/logvars/rewards/dmins plus executed trajectories/outcomes/steps',
            'tensor': 'Noop expected_features bitwise equal original on every real call; swapped channel permutation exact, nonswapped channel exact; fixed vector equals held-out constant',
            'positive': 'ctrl_shuffle_mean flip >=1% separately for both seeds, smoke before full and then full gate',
            'smoke': '6scenes x3cases x2seeds, GRID80, benchmark outcome/steps parity, deterministic flips0, actual replacement printed',
        },
        'interpretation': {
            'real_favorable': 'Both seeds: at least one SAME variance condition has positive flip, primary clearance win clusterCI entirely above0.5 plus Bonferroni binomial significance, and reward does not show harm. Only next step closed-loop confirmation, not method success',
            'otherwise': 'Information-layer follow-up only; positive controls required for interpretation of near-zero variance flips',
            'near_zero': 'Descriptive flip <1%; exact counts/clusterCI retained; no universal absence claim',
            'no_full_vs_mean': 'Never infer Full vs Mean; mean initialization/fan-in confounded',
            'ood': 'Permutation preserves marginal logvar vectors but can break mean-logvar/candidate joint relations. Fixed global vector also may be out-of-distribution',
        },
        'stops': ['Any exception/assertion stops run, artifacts retained', 'Real replay flip must be exactly0',
                  'Positive control under1% stops this round', 'No original project writes',
                  'No training, 4090 access, reward/action/filter/history/time-limit changes'],
    })
    print('FROZEN', len(manifest), 'inputs before any new outcomes', flush=True)


def pinned():
    for name, entry in load(BASE / 'frozen-manifest.json').items():
        assert entry['sha256'] and sha(name) == entry['sha256'], 'Changed input: ' + name


def module():
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location('readonly_audit', ROOT / 'scripts/audit-bayes-fix.py')
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    for imported in (result.crowd_nav, result.crowd_sim, result.policy_module):
        Path(imported.__file__).resolve().relative_to(ROOT)
    return result


def setup(seed):
    torch.set_num_threads(2)
    assert torch.cuda.is_available() and '3060' in torch.cuda.get_device_name(0)
    torch.use_deterministic_algorithms(True)
    m = module()
    cfg = configuration(seed)
    m.contracts.init_grid_from_cfg(cfg)
    assert m.contracts.grid_action_dim(m.contracts.GRID) == 80
    assert m.contracts.GRID['sampling'] == 'exponential' and m.contracts.GRID['v_min'] == .05
    saved = torch.load(BACKUP / ('formal-bayes-seed%d' % seed) / 'rl_model_ep10000.pth',
                       map_location='cpu', weights_only=False)
    assert saved['episode'] == 10000 and saved['algo'] == 'sarl'
    policy = m.policy_module.MambaRLPolicy(cfg, device='cuda').eval()
    sd = {k[len('_orig_mod.'):] if k.startswith('_orig_mod.') else k: v
          for k, v in saved['policy_state'].items()}
    policy.load_state_dict(sd, strict=True)
    assert policy.temporal_encoder.readout[0].in_features == 128
    assert not policy.temporal_encoder.mean_only
    assert policy.history_contract == 'legal-prefix' and policy.seq_len == 24
    policy.use_sarl_predict = True
    policy.set_phase('test')
    policy.epsilon = 0
    policy._test_args = types.SimpleNamespace(gating=False, discrete_search=False,
         mamba_bias=False, mamba_rescue=False, behavior_profile='nominal', measure_latency=False)
    references = {(r['scenario'], int(r['episode'])): r for r in csv.DictReader(
        (BACKUP / ('formal-bayes-seed%d' % seed) / 'benchmark-10000.csv').open())}
    return m, policy, references


def configuration(seed):
    cfg = configparser.RawConfigParser(inline_comment_prefixes=('#', ';'))
    assert cfg.read(str(CONFIG / ('bayes-seed%d.eval.ini' % seed)))
    assert cfg.get('mamba', 'temporal_backbone') == 'bayes'
    assert cfg.getfloat('env', 'time_limit') == 25
    assert cfg.getfloat('env', 'time_step') == .25
    return cfg


def environment(m, policy, seed, scene):
    cfg = configuration(seed)
    desc, sim, count, dimension, size = SCENARIOS[scene]
    m.contracts.init_grid_from_cfg(cfg)
    assert m.contracts.grid_action_dim(m.contracts.GRID) == 80
    cfg.set('sim', 'test_sim', sim)
    cfg.set('sim', 'human_num', str(count))
    cfg.set('sim', dimension, str(size))
    env = m.CrowdSim()
    env.configure(cfg)
    env.phase = 'test'
    robot = m.Robot(cfg, 'robot')
    assert not robot.visible, 'Counterfactual response independence requires native robot invisible'
    robot.set_policy(policy)
    robot.env = env
    env.set_robot(robot)
    if hasattr(policy, 'set_env'):
        policy.set_env(env)
    policy.set_env_dt(.25)
    return env, robot


def score(policy, values, rewards, dmins, safety=True):
    rt = torch.tensor(rewards.tolist(), device=policy.device)
    if policy.lookahead_ablation_mode == 'reward_only':
        total = rt
    elif policy.lookahead_ablation_mode == 'value_only':
        total = policy.gamma * values
    else:
        total = rt + policy.gamma * values
    if safety:
        dt = torch.tensor(dmins.tolist(), device=policy.device)
        if policy.test_min_clearance > 0:
            safe = dt >= policy.test_min_clearance
            if safe.any():
                total = total.masked_fill(~safe, -1e9)
        if policy.test_risk_lambda > 0:
            margin = policy.test_min_clearance if policy.test_min_clearance > 0 else policy.discomfort_dist
            total = total - policy.test_risk_lambda * torch.clamp(margin - dt, min=0)
    total[0] -= 1e-3
    return total


class Intervention:
    def __init__(self, encoder):
        self.encoder = encoder
        self.original = encoder.expected_features
        self.mode = 'real'
        self.permutation = None
        self.constant = None
        self.captured = None
        self.calls = 0
        self.changed = False
        self.digest = None
        encoder.expected_features = types.MethodType(self.call, encoder)

    def call(self, encoder, mean, variance):
        assert mean.shape == (80, 64), 'Every candidate must be in SAME grouped readout call'
        logvar = variance.clamp_min(1e-6).log()
        assert torch.isfinite(mean).all() and torch.isfinite(logvar).all()
        self.captured = (mean.detach().clone(), logvar.detach().clone())
        transformed_mean, transformed_logvar = mean, logvar
        if self.mode == 'fixed':
            transformed_logvar = self.constant[None].expand_as(logvar)
            assert torch.equal(transformed_logvar, self.constant[None].expand_as(logvar))
        elif self.mode == 'shuffle_within':
            transformed_logvar = logvar[self.permutation]
            assert torch.equal(torch.sort(transformed_logvar, dim=0).values,
                               torch.sort(logvar, dim=0).values)
        elif self.mode == 'ctrl_shuffle_mean':
            transformed_mean = mean[self.permutation]
            assert torch.equal(torch.sort(transformed_mean, dim=0).values,
                               torch.sort(mean, dim=0).values)
        else:
            assert self.mode == 'real'
        if self.mode != 'ctrl_shuffle_mean':
            assert torch.equal(transformed_mean, mean)
        if self.mode not in ('fixed', 'shuffle_within'):
            assert torch.equal(transformed_logvar, logvar)
        result = encoder.readout(torch.cat((transformed_mean, transformed_logvar), dim=-1))
        if self.mode == 'real':
            assert torch.equal(result, self.original(mean, variance)), 'Noop not bitwise identical'
        self.calls += 1
        original_channel = mean if self.mode == 'ctrl_shuffle_mean' else logvar
        new_channel = transformed_mean if self.mode == 'ctrl_shuffle_mean' else transformed_logvar
        difference = (new_channel - original_channel).abs()
        self.changed = bool(difference.max().item() > 0)
        self.digest = {'mode': self.mode, 'mean_before': float(original_channel.mean()),
             'mean_after': float(new_channel.mean()), 'std_before': float(original_channel.std()),
             'std_after': float(new_channel.std()), 'max_abs_change': float(difference.max()),
             'per_dimension_candidate_std_before_mean': float(original_channel.std(0).mean()),
             'per_dimension_candidate_std_after_mean': float(new_channel.std(0).mean())}
        return result

    def close(self):
        self.encoder.expected_features = self.original


def native_counterfactual(env, before, commands, human_positions):
    # Shadow actions share the real preceding action and smoothing state.
    dt = env.time_step
    rp, goal, radius, old_hpos, old_hvel, hr, global_time = before
    nrp = rp[None] + commands * dt
    endpoint = np.min(np.linalg.norm(nrp[:, None] - human_positions[None], axis=2) - radius - hr[None], axis=1)
    relative = old_hpos[None] - rp[None, None]
    segment = (old_hvel[None] - commands[:, None]) * dt
    length = np.sum(segment * segment, axis=2)
    t = np.zeros_like(length)
    np.divide(-np.sum(relative * segment, axis=2), length, out=t, where=length > 0)
    t = np.clip(t, 0, 1)
    swept = np.min(np.linalg.norm(relative + t[..., None] * segment, axis=2) - radius - hr[None], axis=1)
    distance = np.linalg.norm(nrp - goal[None], axis=1)
    progress = np.linalg.norm(rp - goal) - distance
    reward = env.progress_reward * progress + env.time_penalty
    reward += np.where(np.linalg.norm(commands, axis=1) < .05, env.stand_penalty, 0)
    reward -= np.where(swept < env.discomfort_dist,
                       env.discomfort_penalty_factor * (env.discomfort_dist - swept) * dt, 0)
    reward = np.where(distance < radius, env.success_reward, reward)
    reward = np.where(swept < 0, env.collision_penalty, reward)
    if global_time >= env.time_limit - 1e-6:
        reward[:] = env.timeout_penalty
    return endpoint, swept, reward


def episode(m, policy, references, training_seed, scene, case, phase, constant=None, capture=False):
    env, robot = environment(m, policy, training_seed, scene)
    desc = SCENARIOS[scene][0]
    reference = references[desc, case]
    hook = Intervention(policy.temporal_encoder)
    if constant is not None:
        hook.constant = torch.tensor(constant, dtype=torch.float32, device='cuda')
    original_forward = policy.forward_value
    original_score = policy._score_candidates_vectorized
    original_step = env.step
    original_predict = policy.predict
    rows, trace, calibration = [], [], []
    pending = {}
    first_digests = {}

    def observe_score(state, current_token=None):
        result = original_score(state, current_token=current_token)
        commands, current, windows, rewards, dmins = result
        assert len(commands) == 80 and windows.shape[0] == 80
        pending.update(commands=np.asarray([(a.vx, a.vy) for a in commands]), rewards=rewards.copy(),
                       dmins=dmins.copy(), human_count=len(state.human_states),
                       previous_action=policy._last_action)
        return result

    def observe_forward(windows, mask=None):
        hook.mode = 'real'
        hook.calls = 0
        real = original_forward(windows, mask)
        assert hook.calls == 1
        mean, logvar = hook.captured
        if phase == 'calibration':
            calibration.append(logvar.cpu().numpy())
            return real
        tick = len(rows)
        totals = score(policy, real, pending['rewards'], pending['dmins'])
        real_idx = int(totals.argmax())
        record = {'tick': tick, 'human_count': pending['human_count'], 'real_idx': real_idx,
                  'logvar_mean': float(logvar.mean()), 'logvar_min': float(logvar.min()),
                  'logvar_max': float(logvar.max()),
                  'candidate_logvar_std': float(logvar.std(0).mean()), 'conditions': {}}
        values = [real.cpu().numpy()]
        if capture:
            for condition_index, condition in enumerate(CONDITIONS[1:], start=1):
                rng = np.random.default_rng(np.random.SeedSequence(
                    [20261009, training_seed, scene, case, tick, condition_index]))
                hook.permutation = torch.tensor(rng.permutation(80), device='cuda', dtype=torch.long)
                hook.mode = condition
                hook.calls = 0
                intervention_values = original_forward(windows, mask)
                assert hook.calls == 1
                assert torch.equal(hook.captured[0], mean) and torch.equal(hook.captured[1], logvar)
                post = score(policy, intervention_values, pending['rewards'], pending['dmins'])
                pre = score(policy, intervention_values, pending['rewards'], pending['dmins'], safety=False)
                idx = int(post.argmax())
                record['conditions'][condition] = {'idx': idx, 'flip': idx != real_idx,
                    'pre_safety_idx': int(pre.argmax()), 'input_changed': hook.changed,
                    'max_abs_value_change': float((intervention_values - real).abs().max()),
                    'permutation': hook.permutation.cpu().tolist() if condition != 'fixed' else None}
                if condition not in first_digests:
                    first_digests[condition] = dict(hook.digest)
                values.append(intervention_values.cpu().numpy())
            hook.mode = 'real'
        record['pre_safety_real_idx'] = int(score(policy, real, pending['rewards'], pending['dmins'], safety=False).argmax())
        pending['record'] = record
        pending['real_values'] = real.cpu().numpy()
        pending['logvar'] = logvar.cpu().numpy()
        pending['mean'] = mean.cpu().numpy()
        pending['value_matrix'] = np.stack(values)
        return real

    def observe_step(action, update=True):
        assert update, 'No shadow env.step calls allowed'
        before = (np.asarray([robot.px, robot.py]), np.asarray([robot.gx, robot.gy]), robot.radius,
                  np.asarray([(h.px, h.py) for h in env.humans]),
                  np.asarray([(h.vx, h.vy) for h in env.humans]),
                  np.asarray([h.radius for h in env.humans]), env.global_time)
        output = original_step(action, update=update)
        if phase == 'calibration':
            return output
        row = pending['record']
        assert policy._selected_action_index == row['real_idx'], 'Shadow ranking differs from native selected action'
        commands = pending['commands'].copy()
        prev = pending['previous_action']
        if policy.test_action_smoothing > 0 and prev is not None:
            alpha = policy.test_action_smoothing
            commands = alpha * np.asarray([prev.vx, prev.vy])[None] + (1-alpha) * commands
        actual = np.asarray([action.vx, action.vy])
        assert np.array_equal(actual, commands[row['real_idx']]), 'Executed action mismatch'
        endpoints, swept, native_rewards = native_counterfactual(env, before, commands,
             np.asarray([(h.px, h.py) for h in env.humans]))
        assert abs(native_rewards[row['real_idx']] - output[1]) < 1e-10, 'Native reward contract mismatch'
        row.update(real_clearance=float(endpoints[row['real_idx']]),
                   real_reward=float(native_rewards[row['real_idx']]),
                   real_swept_clearance=float(swept[row['real_idx']]),
                   executed_action=actual.tolist(),
                   native_event=output[4]['event'],
                   safe_candidate_count=int((pending['dmins'] >= policy.test_min_clearance).sum()))
        for condition, item in row['conditions'].items():
            idx = item['idx']
            item.update(clearance=float(endpoints[idx]), reward=float(native_rewards[idx]),
                        swept_clearance=float(swept[idx]), predicted_clearance=float(pending['dmins'][idx]),
                        executed_shadow=commands[idx].tolist(),
                        clearance_delta_real=float(endpoints[row['real_idx']] - endpoints[idx]),
                        reward_delta_real=float(native_rewards[row['real_idx']] - native_rewards[idx]))
        rows.append(row)
        trace.append({'real_values': pending['real_values'], 'logvar': pending['logvar'],
                      'mean': pending['mean'], 'rewards': pending['rewards'], 'dmins': pending['dmins'],
                      'value_matrix': pending['value_matrix'], 'actual_clearance_all': endpoints,
                      'native_reward_all': native_rewards, 'selected': row['real_idx'],
                      'action': actual, 'position': np.asarray([robot.px, robot.py])})
        return output

    policy._score_candidates_vectorized = observe_score
    policy.forward_value = observe_forward
    env.step = observe_step
    with torch.no_grad():
        try:
            result = m.test_episode(env, robot, policy, int(reference['seed']), case_desc=desc)
        finally:
            policy._score_candidates_vectorized = original_score
            policy.forward_value = original_forward
            policy.predict = original_predict
            env.step = original_step
            hook.close()
    outcome, steps = result[0:2]
    external_parity = outcome == reference['outcome'] and steps == int(reference['steps'])
    if phase != 'calibration':
        assert len(rows) == steps
        for row in rows:
            row.update(outcome=outcome, episode_steps=steps, training_seed=training_seed,
                       scene=desc, scene_index=scene, case_index=case, episode_seed=int(reference['seed']))
    return rows, trace, calibration, {'outcome': outcome, 'steps': steps, 'tensor_digests': first_digests,
         'external_benchmark_parity': external_parity,
         'external_benchmark_outcome': reference['outcome'], 'external_benchmark_steps': int(reference['steps'])}


def calibrate():
    pinned()
    protocol = load(BASE / 'protocol.json')
    for seed in protocol['training_seeds']:
        assert not (BASE / ('fixed-seed%d.json' % seed)).exists()
        m, policy, references = setup(seed)
        accumulator, count, episodes = np.zeros(64, dtype=np.float64), 0, []
        start = time.time()
        for scene in range(6):
            for case in protocol['calibration_cases']:
                _, _, candidate_logvars, meta = episode(m, policy, references, seed, scene, case, 'calibration')
                batch = np.concatenate(candidate_logvars).astype(np.float64)
                accumulator += batch.sum(0)
                count += len(batch)
                episodes.append({'scene': SCENARIOS[scene][0], 'case': case, **meta})
            print('CALIBRATION', seed, SCENARIOS[scene][0], 'candidate rows', count, flush=True)
        save(BASE / ('fixed-seed%d.json' % seed), {'mean_logvar': (accumulator/count).tolist(),
             'candidate_rows': count, 'episodes': episodes, 'seconds': time.time()-start,
             'sha_checkpoint': sha(BACKUP / ('formal-bayes-seed%d' % seed) / 'rl_model_ep10000.pth')})
        print('FIXED', seed, (accumulator/count).tolist(), flush=True)
    pinned()


def identical_trace(first, second):
    assert len(first) == len(second)
    for a, b in zip(first, second):
        for name in ('real_values', 'logvar', 'mean', 'rewards', 'dmins', 'actual_clearance_all',
                     'native_reward_all', 'selected', 'action', 'position'):
            assert np.array_equal(a[name], b[name]), 'Determinism failed: ' + name


def run(phase):
    pinned()
    protocol = load(BASE / 'protocol.json')
    if phase == 'full':
        assert load(BASE / 'controls.json')['smoke_passed']
    case_list = protocol['smoke_cases'] if phase == 'smoke' else protocol['evaluation_cases']
    directory = BASE / ('smoke' if phase == 'smoke' else 'stepwise')
    directory.mkdir(parents=True, exist_ok=True)
    counts, controls, start = {}, {}, time.time()
    for seed in (42, 43):
        m, policy, references = setup(seed)
        fixed = load(BASE / ('fixed-seed%d.json' % seed))['mean_logvar']
        totals = collections.Counter()
        controls[str(seed)] = []
        for scene in range(6):
            for case in case_list:
                prefix = directory / ('seed%d-scene%d-case%03d' % (seed, scene, case))
                assert not prefix.with_suffix('.json').exists(), 'Do not overwrite outcome'
                rows, trace, _, meta = episode(m, policy, references, seed, scene, case, phase, fixed, True)
                other_rows, other_trace, _, other_meta = episode(m, policy, references, seed, scene, case, phase, fixed, False)
                identical_trace(trace, other_trace)
                assert meta['outcome'] == other_meta['outcome'] and meta['steps'] == other_meta['steps']
                totals['steps'] += len(rows)
                totals['episodes'] += 1
                totals['outcome-'+meta['outcome']] += 1
                totals['external-benchmark-mismatches'] += int(not meta['external_benchmark_parity'])
                for condition in CONDITIONS[1:]:
                    totals[condition+'-flips'] += sum(r['conditions'][condition]['flip'] for r in rows)
                    totals[condition+'-changed'] += sum(r['conditions'][condition]['input_changed'] for r in rows)
                save(prefix.with_suffix('.json'), {'meta': meta, 'deterministic': True, 'rows': rows})
                np.savez_compressed(prefix.with_suffix('.npz'),
                    values=np.stack([t['value_matrix'] for t in trace]),
                    all_candidate_clearance=np.stack([t['actual_clearance_all'] for t in trace]),
                    all_candidate_reward=np.stack([t['native_reward_all'] for t in trace]),
                    predicted_clearance=np.stack([t['dmins'] for t in trace]),
                    native_lookahead_reward=np.stack([t['rewards'] for t in trace]),
                    actions=np.asarray([t['action'] for t in trace]), positions=np.asarray([t['position'] for t in trace]))
                if phase == 'smoke':
                    print('TENSOR', seed, scene, case, meta['tensor_digests'], flush=True)
                controls[str(seed)].append({'scene': SCENARIOS[scene][0], 'case': case,
                                           'deterministic_flip_count': 0, **meta})
                save(directory / 'progress.json', {'seed': seed, 'counts': dict(totals), 'seconds': time.time()-start})
            print(phase.upper(), seed, SCENARIOS[scene][0], dict(totals), 'elapsed', time.time()-start, flush=True)
        counts[str(seed)] = dict(totals)
        assert totals['ctrl_shuffle_mean-flips'] / totals['steps'] >= .01, 'Positive control flip below1%; STOP'
    status = {'smoke_passed': True, 'grid': 80, 'deterministic_flips': 0, 'counts': counts,
              'controls': controls, 'seconds': time.time()-start, 'device': torch.cuda.get_device_name(0),
              'no_original_edits': True, 'remote_access': False, 'training': False}
    if phase == 'smoke':
        save(BASE / 'controls.json', status)
    else:
        save(directory / 'run-complete.json', status)
    pinned()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('freeze', 'calibrate', 'smoke', 'full'))
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    if args.command == 'freeze':
        freeze()
    elif args.command == 'calibrate':
        calibrate()
    else:
        run(args.command)


if __name__ == '__main__':
    main()
