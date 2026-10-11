"""Read frozen results and replay only the states needed for causal diagnosis."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'recovery-duration'
OUT = HERE / 'artifacts'
OUT.mkdir(exist_ok=True)
spec = importlib.util.spec_from_file_location('frozen_recovery', OLD / 'run.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
PROTO = json.loads((HERE / 'protocol.json').read_text())


def write(name, obj):
    (HERE / name).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def setup():
    manifest = json.loads((OLD / 'frozen-manifest.json').read_text())
    for filename, key in [('nominal.pth', 'nominal_sha256'),
                          ('ordinary.pth', 'ordinary_checkpoint_sha256'),
                          ('paired.pth', 'paired_checkpoint_sha256')]:
        actual = hashlib.sha256((m.OUT / filename).read_bytes()).hexdigest()
        assert actual and actual == manifest[key], (filename, actual)
    cfg, p = m.setup('cuda')
    p.load_state_dict(torch.load(m.OUT / 'nominal.pth', map_location=p.device))
    p.eval()
    p.set_phase('test')
    return cfg, p


def model(arm, device):
    ck = torch.load(m.OUT / f'{arm}.pth', map_location=device)
    net = m.RecoveryNet().to(device).eval()
    net.load_state_dict(ck['state'])
    return net, (ck['mean'].cpu().numpy(), ck['std'].cpu().numpy())


def offline_recovery():
    d = np.load(m.OUT / 'recovery-data.npz')
    x, g = d['x'], d['returns']
    result = {'roots': len(x), 'arms': {}, 'traces': {}}
    for arm in ['ordinary', 'paired']:
        net, norm = model(arm, 'cpu')
        with torch.no_grad():
            q = net(torch.as_tensor((x-norm[0])/norm[1])).numpy()
        chosen = q.argmax(1)
        regret = g.max(1) - g[np.arange(len(g)), chosen]
        result['arms'][arm] = {
            'train_return_mse': float(((q-g)**2).mean()),
            'chosen_option_counts': np.bincount(chosen, minlength=10).tolist(),
            'empirical_best_option_counts': np.bincount(g.argmax(1), minlength=10).tolist(),
            'mean_chosen_actual_advantage': float((g[np.arange(len(g)), chosen]-g[:,0]).mean()),
            'mean_chosen_predicted_advantage': float((q[np.arange(len(g)), chosen]-q[:,0]).mean()),
            'negative_actual_advantage_roots': int((g[np.arange(len(g)), chosen]<g[:,0]-1e-6).sum()),
            'mean_selection_regret': float(regret.mean()),
            'median_selection_regret': float(np.median(regret)),
            'regret_max': float(regret.max())}
        all_runs = []
        for f in sorted((m.OUT/'development-traces'/arm).glob('*.npz')):
            a = np.load(f)['records']
            decisions = a[a[:,9]>=0]
            active = a[:,5]!=a[:,6]
            longest = now = 0
            for bit in active:
                now = now+1 if bit else 0
                longest = max(longest, now)
            previous = 0
            countdown_errors = 0
            for row in a:
                if previous>0 and row[9]<0 and row[10] not in (previous-1, 0):
                    countdown_errors += 1
                previous = row[10]
            all_runs.append({'file': f.name, 'first_recovery_step':
                             int(a[a[:,9]>0,0][0]) if (a[:,9]>0).any() else None,
                             'recovery_decisions': int((decisions[:,9]>0).sum()),
                             'nominal_decisions': int((decisions[:,9]==0).sum()),
                             'changed_steps': int(active.sum()), 'steps': len(a),
                             'longest_changed_run_steps': longest,
                             'countdown_errors': countdown_errors})
        result['traces'][arm] = all_runs
    write('recovery-offline.json', result)
    print(json.dumps(result['arms'], indent=2), flush=True)


def decompose(p, env):
    commands, tok, windows, rew, clear = p._score_candidates_vectorized(m.state(env))
    with torch.no_grad():
        values = p.forward_candidate_values(windows).cpu().numpy()
    risk = p.test_risk_lambda*np.maximum(p.test_min_clearance-clear, 0)
    raw = rew + p.gamma*values-risk
    raw[0] -= .001
    valid = clear>=p.test_min_clearance
    scores = raw.copy()
    if valid.any():
        scores[~valid] = -1e9
    best = int(scores.argmax())
    return commands, tok, rew, values, clear, risk, scores, best


def snap(p, env, positions):
    return {'robot': copy.copy(env.robot.get_full_state()), 'time': env.global_time,
            'history': [h.copy() for h in p._history], 'last_action': p._last_action,
            'positions': list(positions)}


def replay(cfg, p, arm, scene, case, root_steps, callback):
    tape = m.make_tape(cfg, p, scene, case)
    env = m.tape_env(cfg, p, scene, case, tape)
    a = np.load(m.OUT/'development-traces'/arm/f'{scene}-{case}.npz')['records']
    positions = []
    errors = {'position': 0., 'action': 0., 'reward': 0., 'clearance': 0.}
    for row in a:
        t = int(row[0])
        np.testing.assert_allclose(env.robot.get_position(), row[1:3], atol=1e-10, rtol=0)
        if t in root_steps:
            parts = decompose(p, env)
            assert parts[-1] == int(row[6]), (arm, scene, case, t, parts[-1], row[6])
            cmd = parts[0][int(row[5])]
            np.testing.assert_allclose(cmd, row[3:5], atol=1e-10, rtol=0)
            callback(env, positions, row, parts, snap(p, env, positions), tape)
        tok = m.token(p, m.state(env))
        action = m.ActionXY(*row[3:5])
        positions.append(env.robot.get_position())
        before = m.before_geometry(env)
        m.commit_action(p, tok, action, int(row[5]))
        _, reward, term, trunc, info = env.step(action)
        errors['reward'] = max(errors['reward'], abs(reward-row[7]))
        errors['clearance'] = max(errors['clearance'], abs(m.geo_step(env, before)-row[8]))
        np.testing.assert_allclose([reward,m.geo_step(env,before)], row[7:9], atol=1e-9, rtol=0)
    assert term or trunc
    return errors


def recovery_roots():
    cfg, p = setup()
    results = []
    maps = {(r['arm'],r['scene'],r['case']):r for r in
            json.loads((m.OUT/'development-results.json').read_text())}
    for scene in range(6):
        for case in [100,101]:
            # Same unmodified nominal prefix for both frozen recovery networks.
            roots = {}
            def capture(env, positions, row, parts, snapshot, tape):
                roots['snapshot'], roots['tape'] = snapshot, tape
                roots['x'] = m.feature(p, env, parts[1], positions)
            replay(cfg,p,'nominal',scene,case,{4},capture)
            for arm in ['ordinary','paired']:
                net, norm = model(arm,p.device)
                with torch.no_grad():
                    q = net(torch.as_tensor((roots['x']-norm[0])/norm[1],device=p.device)).cpu().numpy()
                choice = int(q.argmax())
                res, trace, _ = m.run_episode(cfg,p,scene,case,roots['tape'],
                    snapshot=roots['snapshot'],first_option=choice)
                nominal = maps[('nominal',m.SCENES[scene],case)]
                original = maps[(arm,m.SCENES[scene],case)]
                original_trace = np.load(m.OUT/'development-traces'/arm/f'{scene}-{case}.npz')['records']
                first = original_trace[original_trace[:,9]>=0][0]
                assert int(first[0])==4 and int(first[9])==choice
                suffix = np.load(m.OUT/'development-traces'/'nominal'/f'{scene}-{case}.npz')['records'][4:]
                nominal_suffix_return = float(suffix[:,7]@(.99**np.arange(len(suffix))))
                repeat_suffix = original_trace[4:]
                repeat_suffix_return = float(repeat_suffix[:,7]@(.99**np.arange(len(repeat_suffix))))
                results.append({'scene':m.SCENES[scene],'case':case,'arm':arm,'option':choice,
                                'predicted_q':q.tolist(),'predicted_advantage':float(q[choice]-q[0]),
                                'one_shot':res,'nominal_outcome':nominal['outcome'],
                                'repeated_outcome':original['outcome'],
                                'nominal_root_return':nominal_suffix_return,
                                'repeated_root_return':repeat_suffix_return,
                                'one_shot_advantage':res['discounted_return']-nominal_suffix_return})
                np.savez_compressed(OUT/f'one-shot-{arm}-{scene}-{case}.npz',records=trace)
                write('recovery-first-choice.json',results)
            print('first-choice diagnostic',scene,case,flush=True)
    print('completed one-shot contrasts',len(results),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('command',choices=['recovery-offline','recovery-roots'])
    args=ap.parse_args()
    if args.command=='recovery-offline':
        offline_recovery()
    elif args.command=='recovery-roots':
        recovery_roots()
