"""Bounded native IL/RL and matched recovery study. Run from this worktree."""
import argparse
from collections import deque
import configparser
import copy
import csv
import hashlib
import json
import logging
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import crowd_nav
import crowd_sim
from crowd_nav import contracts
from crowd_nav.policy import mamba_rl
from crowd_nav.policy.mamba_rl import MambaRLPolicy
from crowd_sim.envs.crowd_sim import CrowdSim
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.state import JointState, FullState, ObservableState
from crowd_sim.envs.utils.action import ActionXY
from crowd_sim.envs.policy.orca import ORCA

HERE = Path(__file__).resolve().parent
OUT = HERE / 'artifacts'
PROTO = json.loads((HERE / 'protocol.json').read_text())
SCENES = PROTO['data']['scenes']
SPECS = [('circle', 4, 5), ('square', 10, 10), ('circle', 4, 10),
         ('square', 10, 20), ('circle', 6, 12), ('square', 14, 20)]
OPTIONS = [(0, 0)] + [(angle, duration) for angle in (90, -90, 180)
                      for duration in (2, 4, 8)]


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def log(message):
    print(time.strftime('%Y-%m-%d %H:%M:%S'), message, flush=True)


def setup(device='cuda'):
    for module in (crowd_nav, crowd_sim, mamba_rl):
        assert Path(module.__file__).resolve().is_relative_to(ROOT)
    cfg = configparser.ConfigParser()
    cfg.read(ROOT / 'crowd_nav/runs/bayes-fix-20261008/remote-run/configs/gru-seed42.ini')
    contracts.init_grid_from_cfg(cfg)
    assert contracts.grid_action_dim(contracts.GRID) == 80
    torch.set_num_threads(2)
    torch.manual_seed(61)
    random.seed(61)
    np.random.seed(61)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    p = MambaRLPolicy(cfg, device)
    p.multiagent_training = True
    p.use_sarl_predict = True
    p.time_step = .25
    p.build_action_space(1.)
    assert len(p.action_space) == 80
    p.eval()
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'parent-config.ini').open('w') as f:
        cfg.write(f)
    return cfg, p


def make_env(cfg, policy, scene=0, phase='test'):
    c = copy.deepcopy(cfg)
    if phase == 'test':
        kind, size, humans = SPECS[scene]
        c.set('env', 'time_limit', '25')
        c.set('sim', 'human_num', str(humans))
        c.set('sim', 'test_sim', kind + '_crossing')
        c.set('sim', 'circle_radius' if kind == 'circle' else 'square_width', str(size))
    env = CrowdSim()
    env.configure(c)
    robot = Robot(c, 'robot')
    robot.set_policy(policy)
    env.set_robot(robot)
    robot.env = env
    env.phase = phase
    return env


def state(env):
    return JointState(env.robot.get_full_state(), [h.get_observable_state() for h in env.humans])


def token(policy, s):
    flat = policy._build_joint_state_34(s.self_state, s.human_states)
    return contracts._batch_joint34_to_tokens_vectorized(flat[None])[0]


def window(tokens, length=24):
    a = np.asarray(tokens[-length:], dtype=np.float32)
    result = np.zeros((length, 8, 13), np.float32)
    result[-len(a):] = a
    return result


def geo_step(env, before):
    # Actual simultaneous motions, independent of the environment's CV contact test.
    r0, h0 = before
    r1 = np.array(env.robot.get_position())
    h1 = np.array([h.get_position() for h in env.humans])
    a = h0 - r0
    b = (h1 - r1) - a
    frac = np.clip(-(a*b).sum(1) / np.maximum((b*b).sum(1), 1e-15), 0, 1)
    radii = np.array([h.radius + env.robot.radius for h in env.humans])
    return float((np.linalg.norm(a + frac[:, None]*b, axis=1)-radii).min())


def before_geometry(env):
    return np.array(env.robot.get_position()), np.array([h.get_position() for h in env.humans])


def score(policy, env):
    s = state(env)
    commands, tok, windows, rew, dmin = policy._score_candidates_vectorized(s)
    with torch.no_grad():
        vals = policy.forward_candidate_values(windows).cpu().numpy()
    scores = rew + policy.gamma * vals
    if policy._phase in ('test', 'val', 'eval'):
        mask = dmin >= policy.test_min_clearance
        if mask.any():
            scores[~mask] = -1e9
        scores -= policy.test_risk_lambda * np.maximum(policy.test_min_clearance-dmin, 0)
    scores[0] -= .001
    return commands, tok, dmin, int(scores.argmax())


def commit_action(policy, tok, action, index):
    policy._history.append(tok)
    policy._last_action = action
    policy._selected_action_index = index


class EpisodeMemory:
    """MC windows assembled on sampling; no giant duplicated window allocation."""
    def __init__(self, capacity=1024, seed=61):
        self.episodes = deque(maxlen=capacity)
        self.rng = np.random.default_rng(seed)

    def add(self, tokens, rewards, gamma=.99):
        returns = np.zeros(len(rewards), np.float32)
        g = 0.
        for i in range(len(rewards)-1, -1, -1):
            g = float(rewards[i]) + gamma*g
            returns[i] = g
        self.episodes.append((np.asarray(tokens, np.float32), returns))

    def sample(self, batch, device):
        lengths = np.array([len(e[0]) for e in self.episodes])
        cumul = lengths.cumsum()
        idx = self.rng.integers(int(cumul[-1]), size=batch)
        epi = np.searchsorted(cumul, idx, side='right')
        starts = np.r_[0, cumul[:-1]]
        xs, ys = [], []
        for e, ix in zip(epi, idx):
            ts, ret = self.episodes[int(e)]
            t = int(ix-starts[e])
            xs.append(window(ts[:t+1]))
            ys.append(ret[t])
        return torch.as_tensor(np.stack(xs), device=device), torch.as_tensor(np.array(ys), device=device)


def update(policy, memory, opt, batch=256):
    x, y = memory.sample(batch, policy.device)
    policy.train()
    opt.zero_grad(set_to_none=True)
    loss = (policy.forward_value(x)-y).square().mean()
    assert torch.isfinite(loss)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.)
    opt.step()
    policy.eval()
    return float(loss.detach())


def train_nominal(cfg, policy):
    spec = PROTO['nominal']
    memory = EpisodeMemory()
    teacher = ORCA()
    teacher.configure(cfg)
    teacher.safety_space = .25
    teacher.neighbor_dist = 6.
    teacher.max_neighbors = 15
    teacher.multiagent_training = True
    env = make_env(cfg, teacher, phase='train')
    successes = 0
    curves = []
    started = time.perf_counter()
    for ep in range(spec['il_max_episodes']):
        env.reset(options={'test_case': ep})
        teacher.sim = None
        teacher._last_pref_vel = None
        ts, rs = [], []
        for _ in range(202):
            s = state(env)
            ts.append(token(policy, s))
            action = teacher.predict(s)
            _, rew, term, trunc, info = env.step(action)
            rs.append(rew)
            if term or trunc:
                break
        curves.append({'stage': 'collect', 'episode': ep, 'outcome': info['event'], 'steps': len(rs)})
        if info['event'] == 'reach_goal':
            memory.add(ts, rs)
            successes += 1
        if (ep+1) % 64 == 0:
            log(f'ORCA collection {ep+1}, successes={successes}')
        if successes == spec['il_successes']:
            break
    assert successes == spec['il_successes'], 'ORCA collection budget exhausted'
    torch.save(list(memory.episodes), OUT / 'corrected-orca.pth')
    steps_per_epoch = math.ceil(sum(len(e[0]) for e in memory.episodes)/spec['batch'])
    optimizer = torch.optim.Adam(policy.parameters(), lr=spec['il_lr'])
    for epoch in range(spec['il_epochs']):
        losses = [update(policy, memory, optimizer) for _ in range(steps_per_epoch)]
        curves.append({'stage': 'IL', 'epoch': epoch+1, 'loss': float(np.mean(losses))})
        log(f'IL {epoch+1}/{spec["il_epochs"]} loss={np.mean(losses):.6f}')
        save_json(OUT / 'training-curves.json', curves)
    torch.save(policy.state_dict(), OUT / 'gru-il.pth')
    env = make_env(cfg, policy, phase='train')
    optimizer = torch.optim.Adam(policy.parameters(), lr=spec['rl_lr'])
    exploration = np.random.default_rng(spec['seed'])
    policy.set_phase('train')
    for ep in range(spec['rl_episodes']):
        env.reset(options={'test_case': 10000+ep})
        policy.reset_episode_stats()
        ts, rs = [], []
        eps = .3 + (.05-.3)*min(ep/1500., 1.)
        for _ in range(202):
            if exploration.random() < eps:
                tok = token(policy, state(env))
                idx = int(exploration.integers(80))
                action = policy.action_space[idx]
            else:
                commands, tok, _, idx = score(policy, env)
                action = commands[idx]
            ts.append(tok)
            commit_action(policy, tok, action, idx)
            _, rew, term, trunc, info = env.step(action)
            rs.append(rew)
            if term or trunc:
                break
        memory.add(ts, rs)
        losses = [update(policy, memory, optimizer) for _ in range(spec['updates_per_episode'])]
        curves.append({'stage': 'RL', 'episode': ep+1, 'outcome': info['event'],
                       'steps': len(rs), 'return': float(sum(rs)), 'loss': float(np.mean(losses))})
        if (ep+1) % 25 == 0:
            last = curves[-25:]
            sr = np.mean([r.get('outcome') == 'reach_goal' for r in last])
            log(f'RL {ep+1}/{spec["rl_episodes"]} recent_SR={sr:.3f}, loss={np.mean(losses):.5f}')
            save_json(OUT / 'training-curves.json', curves)
        if (ep+1) % 500 == 0:
            torch.save(policy.state_dict(), OUT / f'gru-rl-{ep+1}.pth')
    torch.save(policy.state_dict(), OUT / 'nominal.pth')
    save_json(OUT / 'training-compute.json', {'seconds': time.perf_counter()-started,
              'IL_steps': steps_per_epoch*spec['il_epochs'], 'RL_steps': spec['rl_episodes']*4,
              'parameters': sum(p.numel() for p in policy.parameters()), 'old_weights_reused': False})


def make_tape(cfg, policy, scene, case):
    path = OUT / 'tapes' / f'{scene}-{case}.npz'
    if path.exists():
        return np.load(path)['humans']
    env = make_env(cfg, policy, scene)
    env.reset(options={'test_case': case})
    assert not env.robot.visible
    tape = []
    for _ in range(103):
        tape.append([[h.px, h.py, h.vx, h.vy, h.radius] for h in env.humans])
        acts = [h.act([o.get_observable_state() for o in env.humans if o is not h])
                for h in env.humans]
        for h, a in zip(env.humans, acts):
            h.step(a)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.array(tape, np.float64)
    np.savez_compressed(path, humans=arr)
    return arr


def tape_env(cfg, policy, scene, case, tape, snapshot=None):
    env = make_env(cfg, policy, scene)
    env.reset(options={'test_case': case})
    policy.reset_episode_stats()
    policy.set_phase('test')
    t = 0
    if snapshot is not None:
        s = snapshot['robot']
        env.robot.set(s.px, s.py, s.gx, s.gy, s.vx, s.vy, s.theta,
                      radius=s.radius, v_pref=s.v_pref)
        env.global_time = snapshot['time']
        t = int(round(env.global_time/.25))
        policy._history.extend([a.copy() for a in snapshot['history']])
        policy._last_action = snapshot['last_action']
    for i, h in enumerate(env.humans):
        h.px, h.py, h.vx, h.vy, h.radius = tape[t, i]
        h.act = lambda obs, i=i: ActionXY(*tape[int(round(env.global_time/.25))+1, i, 2:4])
    return env


def feature(policy, env, tok, positions):
    hist = list(policy._history)+[tok]
    x = torch.as_tensor(window(hist)[None], device=policy.device)
    valid = int(np.any(np.asarray(window(hist)) != 0, axis=(1, 2)).sum())
    with torch.no_grad():
        z = policy.temporal_encoder(policy.spatial_encoder(x[:, -valid:]))[:, -1].cpu().numpy()[0]
    p = np.array(positions[-8:]+[env.robot.get_position()])
    goal = np.array(env.robot.get_goal_position())
    distances = np.linalg.norm(p-goal, axis=1)
    path = np.linalg.norm(np.diff(p, axis=0), axis=1).sum() if len(p)>1 else 0.
    progress = distances[0]-distances[-1]
    hs = np.array([[h.px-env.robot.px, h.py-env.robot.py, h.vx, h.vy] for h in env.humans])
    clearance = np.linalg.norm(hs[:, :2], axis=1)-.6
    desc = [distances[-1], (env.time_limit-env.global_time)/env.time_limit,
            env.robot.vx, env.robot.vy, progress, path, progress/(path+.01),
            float((clearance<1.).sum()), float(clearance.min()),
            len(p)/9., float(np.linalg.norm(p[-1]-p[0])),
            float(np.mean(np.linalg.norm(np.diff(p, axis=0), axis=1)<.025)) if len(p)>1 else 0.]
    return np.r_[z, desc].astype(np.float32)


class RecoveryNet(nn.Module):
    def __init__(self, input_dim=268):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 10))

    def forward(self, x):
        return self.net(x)


def recovery_index(policy, env, option):
    angle = math.atan2(env.robot.gy-env.robot.py, env.robot.gx-env.robot.px)
    angle += math.radians(OPTIONS[option][0])
    desired = np.array([math.cos(angle), math.sin(angle)])
    raw = np.array(policy.action_space)
    return int(((raw-desired)**2).sum(1).argmin())


def run_episode(cfg, policy, scene, case, tape, snapshot=None, first_option=0,
                model=None, norm=None, collect_roots=False):
    env = tape_env(cfg, policy, scene, case, tape, snapshot)
    positions = [] if snapshot is None else list(snapshot['positions'])
    roots, records, timings = [], [], []
    option, remaining = first_option, OPTIONS[first_option][1]
    root_time = env.global_time
    initial_distance = np.linalg.norm(np.array(env.robot.get_position())-env.robot.get_goal_position())
    interventions = overrides = 0
    for _ in range(102):
        t = int(round(env.global_time/.25))
        started = time.perf_counter()
        commands, tok, dmin, nominal = score(policy, env)
        if collect_roots and t in PROTO['data']['roots_steps']:
            snap = {'robot': copy.copy(env.robot.get_full_state()), 'time': env.global_time,
                    'history': [a.copy() for a in policy._history], 'last_action': policy._last_action,
                    'positions': list(positions)}
            roots.append((t, snap, feature(policy, env, tok, positions)))
        chosen = nominal
        decision = -1
        if model is not None and remaining == 0 and t >= 4 and t % 4 == 0:
            f = feature(policy, env, tok, positions)
            with torch.no_grad():
                q = model(torch.as_tensor((f-norm[0])/norm[1], device=policy.device))
                option = int(q.argmax())
            remaining = OPTIONS[option][1]
            decision = option
            interventions += int(option != 0)
        if remaining > 0:
            chosen = recovery_index(policy, env, option)
            safe = dmin >= policy.test_min_clearance
            if safe.any() and not safe[chosen]:
                chosen = nominal
                remaining = 0
                overrides += 1
            else:
                remaining -= 1
        action = commands[chosen]
        timings.append(time.perf_counter()-started)
        positions.append(env.robot.get_position())
        before = before_geometry(env)
        commit_action(policy, tok, action, chosen)
        _, reward, term, trunc, info = env.step(action)
        assert env.robot.vx == action.vx and env.robot.vy == action.vy
        clearance = geo_step(env, before)
        records.append([t, before[0][0], before[0][1], action.vx, action.vy, chosen,
                        nominal, reward, clearance, decision, remaining])
        if term or trunc:
            break
    else:
        raise RuntimeError('Native episode did not terminate within its deadline')
    a = np.asarray(records, np.float64)
    total_path = float(np.linalg.norm(a[:, 3:5], axis=1).sum()*.25)
    final_distance = float(np.linalg.norm(np.array(env.robot.get_position())-env.robot.get_goal_position()))
    result = {'scene': SCENES[scene], 'case': case, 'outcome': info['event'],
              'success': int(info['event']=='reach_goal'), 'collision': int(info['event']=='collision'),
              'timeout': int(info['event']=='timeout'), 'contact': int((a[:, 8]<0).any()),
              'min_clearance': float(a[:, 8].min()), 'time': env.global_time-root_time,
              'task_return': float(a[:, 7].sum()),
              'discounted_return': float(a[:, 7] @ (.99**np.arange(len(a)))),
              'steps': len(a), 'path': total_path, 'progress': initial_distance-final_distance,
              'final_distance': final_distance, 'stall_fraction': float((np.linalg.norm(a[:, 3:5], axis=1)<.1).mean()),
              'interventions': interventions, 'overrides': overrides,
              'mean_decision_ms': float(np.mean(timings)*1000)}
    return result, a, roots


def collect_recovery(cfg, policy):
    features, returns, identifiers, outcomes, raw = [], [], [], [], []
    start = time.perf_counter()
    target = OUT / 'paired-continuations'
    target.mkdir(parents=True, exist_ok=True)
    for scene in range(6):
        for case in range(16):
            tape = make_tape(cfg, policy, scene, case)
            nominal_result, nominal_records, roots = run_episode(cfg, policy, scene, case, tape, collect_roots=True)
            traces, offsets = [], [0]
            for step, snap, feat in roots:
                gs, os = [], []
                for option in range(10):
                    res, trace, _ = run_episode(cfg, policy, scene, case, tape, snapshot=snap, first_option=option)
                    if option == 0:
                        expected = nominal_records[nominal_records[:, 0]>=step]
                        np.testing.assert_allclose(trace[:, :9], expected[:, :9], atol=1e-9, rtol=0)
                        assert res['outcome'] == nominal_result['outcome']
                    gs.append(res['discounted_return'])
                    os.append(res['outcome'])
                    raw.append({'scene': SCENES[scene], 'case': case, 'root': step, 'option': option, **res})
                    traces.append(trace)
                    offsets.append(offsets[-1]+len(trace))
                features.append(feat)
                returns.append(gs)
                outcomes.append(os)
                identifiers.append([scene, case, step])
            if traces:
                np.savez_compressed(target/f'{scene}-{case}.npz', traces=np.concatenate(traces), offsets=offsets)
            np.savez_compressed(OUT/'recovery-data.npz', x=features, returns=returns, ids=identifiers, outcomes=outcomes)
            save_json(OUT/'paired-results.json', raw)
            log(f'Paired recovery labels scene={SCENES[scene]} case={case}, roots={len(features)}')
    save_json(OUT/'recovery-data-compute.json', {'seconds': time.perf_counter()-start, 'roots': len(features),
              'continuations': len(raw), 'continuation_steps': sum(r['steps'] for r in raw)})


def train_recovery(device):
    d = np.load(OUT/'recovery-data.npz')
    x = torch.as_tensor(d['x'], dtype=torch.float32, device=device)
    y = torch.as_tensor(d['returns'], dtype=torch.float32, device=device)
    mu, sd = x.mean(0), x.std(0).clamp_min(.01)
    x = (x-mu)/sd
    spec = PROTO['recovery_training']
    logs = {}
    for arm in ('ordinary', 'paired'):
        torch.manual_seed(spec['seed'])
        model = RecoveryNet(x.shape[1]).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=spec['lr'])
        rng = np.random.default_rng(spec['seed'])
        losses = []
        started = time.perf_counter()
        for step in range(spec['steps']):
            ix = rng.integers(len(x), size=spec['batch'])
            q, g = model(x[ix]), y[ix]
            if arm == 'ordinary':
                loss = (q-g).square().mean()
            else:
                loss = (q[:, 0]-g[:, 0]).square().mean()
                loss = loss + ((q[:, 1:]-q[:, :1])-(g[:, 1:]-g[:, :1])).square().mean()
                qs, gs = q[:, 1:].reshape(-1, 3, 3), g[:, 1:].reshape(-1, 3, 3)
                pairs = [(0, 1), (0, 2), (1, 2)]
                loss = loss + torch.stack([((qs[:, :, i]-qs[:, :, j])-(gs[:, :, i]-gs[:, :, j])).square().mean()
                                          for i, j in pairs]).mean()
            assert torch.isfinite(loss)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step % 100 == 0:
                losses.append(float(loss.detach()))
        torch.save({'state': model.state_dict(), 'mean': mu.cpu(), 'std': sd.cpu()}, OUT/f'{arm}.pth')
        logs[arm] = {'seconds': time.perf_counter()-started, 'losses': losses,
                     'parameters': sum(p.numel() for p in model.parameters()), 'steps': spec['steps']}
        log(f'Recovery trained {arm}, loss={float(loss):.6f}')
    save_json(OUT/'recovery-training.json', logs)


def evaluate(cfg, policy, fresh=False):
    lo, hi = PROTO['fresh_cases_per_scene_if_signal' if fresh else 'development_cases_per_scene']
    prefix = 'fresh' if fresh else 'development'
    records = []
    for arm in ('nominal', 'ordinary', 'paired'):
        model = norm = None
        if arm != 'nominal':
            ckpt = torch.load(OUT/f'{arm}.pth', map_location=policy.device)
            model = RecoveryNet().to(policy.device)
            model.load_state_dict(ckpt['state'])
            model.eval()
            norm = (ckpt['mean'].cpu().numpy(), ckpt['std'].cpu().numpy())
        trace_dir = OUT/f'{prefix}-traces'/arm
        trace_dir.mkdir(parents=True, exist_ok=True)
        for scene in range(6):
            for case in range(lo, hi+1):
                tape = make_tape(cfg, policy, scene, case)
                res, trace, _ = run_episode(cfg, policy, scene, case, tape, model=model, norm=norm)
                res['arm'] = arm
                records.append(res)
                np.savez_compressed(trace_dir/f'{scene}-{case}.npz', records=trace)
                save_json(OUT/f'{prefix}-results.json', records)
            recent = [r for r in records if r['arm']==arm and r['scene']==SCENES[scene]]
            log(f'{prefix} {arm} {SCENES[scene]} SR={np.mean([r["success"] for r in recent]):.3f}')
    with (OUT/f'{prefix}-results.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def tests(cfg, policy):
    results = {}
    s = FullState(1, -2, .1, .2, .3, -3, 4, 1.2, .5)
    hs = [ObservableState(100+i, 20, 0, 0, .3) for i in range(5)]
    hs += [ObservableState(1.1, -2, .4, .2, .3)]
    js = policy._build_joint_state_34(s, hs)
    tok = contracts._batch_joint34_to_tokens_vectorized(js[None])[0]
    assert tok[0, 11] == np.float32(1.2)
    np.testing.assert_allclose(tok[1, :2], [-4, 6])
    np.testing.assert_allclose(js[9:11], [1.1, -2])
    np.testing.assert_allclose(tok, contracts._single_joint34_to_tokens(js), atol=2e-5)
    results['fields_selection_scalar_batch'] = True
    env = make_env(cfg, policy, 3)
    env.reset(options={'test_case': 900})
    policy.set_phase('test')
    policy.reset_episode_stats()
    policy._last_action = ActionXY(.7, -.2)
    for length in (1, 3, 23):
        policy._history.clear()
        policy._history.extend([token(policy, state(env))]*length)
        commands, tok, windows, _, _ = policy._score_candidates_vectorized(state(env))
        assert commands == policy.executable_candidates()
        np.testing.assert_allclose([commands[0].vx, commands[0].vy],
                                   .3*np.array([.7, -.2])+.7*np.array(policy.action_space[0]))
        with torch.no_grad():
            ref = policy.forward_value(windows)
            fast = policy.forward_candidate_values(windows)
        torch.testing.assert_close(ref, fast, atol=2e-6, rtol=2e-5)
        scalar = np.stack([policy._build_joint_state_34(policy.propagate(env.robot.get_full_state(), a),
                    [policy.propagate(h.get_observable_state(), ActionXY(h.vx, h.vy)) for h in env.humans])
                         for a in commands])
        np.testing.assert_allclose(windows[:, -1].cpu().numpy(),
                                   contracts._batch_joint34_to_tokens_vectorized(scalar), atol=1e-6)
    results['candidate_execution_prefix_parity'] = True
    maxerr = 0.
    for scene in range(6):
        tape = make_tape(cfg, policy, scene, 900)
        env = make_env(cfg, policy, scene)
        env.reset(options={'test_case': 900})
        for t in range(12):
            np.testing.assert_allclose(np.array([[h.px, h.py, h.vx, h.vy, h.radius] for h in env.humans]), tape[t], atol=1e-12)
            env.step(ActionXY(.1, .2))
        policy.reset_episode_stats()
        result, trace, roots = run_episode(cfg, policy, scene, 900, tape, collect_roots=True)
        for step, snap, _ in roots:
            again, other, _ = run_episode(cfg, policy, scene, 900, tape, snapshot=snap)
            expected = trace[trace[:, 0]>=step]
            np.testing.assert_allclose(other[:, :9], expected[:, :9], atol=1e-9, rtol=0)
            maxerr = max(maxerr, float(np.abs(other[:, :9]-expected[:, :9]).max()))
            assert again['outcome'] == result['outcome']
    results['native_tape_and_same_root_continuation'] = True
    results['max_replay_error'] = maxerr
    save_json(HERE/'tests.json', results)
    log(f'TESTS PASSED {results}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['tests', 'train', 'collect', 'fit', 'evaluate', 'fresh'])
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    cfg, policy = setup(args.device)
    if args.command not in ('tests', 'train'):
        policy.load_state_dict(torch.load(OUT/'nominal.pth', map_location=policy.device))
        policy.eval()
        policy.set_phase('test')
    if args.command == 'tests':
        tests(cfg, policy)
    elif args.command == 'train':
        train_nominal(cfg, policy)
    elif args.command == 'collect':
        collect_recovery(cfg, policy)
    elif args.command == 'fit':
        train_recovery(args.device)
    elif args.command in ('evaluate', 'fresh'):
        evaluate(cfg, policy, args.command == 'fresh')
