"""Matched return-source fitting and separate bounded continuous-control repairs."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
loader = importlib.util.spec_from_file_location('complete_action_evidence', HERE.parent / 'gru-complete-actions/run.py')
q = importlib.util.module_from_spec(loader)
loader.loader.exec_module(q)
r, m = q.r, q.r.m
OLD = HERE.parent / 'gru-complete-actions/artifacts/v2-50s'
OUT = HERE / 'artifacts'
SPEC = json.loads((HERE / 'protocol.json').read_text())
ARMS = SPEC['evaluation']['arms']


def save(name, data):
    m.save_json(HERE / name, data)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def setup():
    cfg, p = r.setup()
    OUT.mkdir(exist_ok=True)
    assert sha(r.OLD / 'artifacts/nominal.pth') == SPEC['parent_sha256']
    assert p.test_min_clearance == .2 and p.test_risk_lambda == .8 and p.test_action_smoothing == .3
    return cfg, p


def tape(cfg, p, scene, case):
    path = OUT / 'tapes' / f'{scene}-{case}.npz'
    if path.exists():
        return np.load(path)['humans']
    env = m.make_env(cfg, p, scene)
    env.time_limit = 50.
    env.reset(options={'test_case': case})
    assert not env.robot.visible and env.time_step == .25
    rows = []
    for _ in range(203):
        rows.append([[h.px, h.py, h.vx, h.vy, h.radius] for h in env.humans])
        actions = [h.act([o.get_observable_state() for o in env.humans if o is not h]) for h in env.humans]
        for h, action in zip(env.humans, actions):
            h.step(action)
    path.parent.mkdir(exist_ok=True)
    result = np.asarray(rows, np.float64)
    np.savez_compressed(path, humans=result)
    return result


def old_tape(cfg, p, root):
    return r.load_tape(m.SCENES.index(root['scene']), root['case'], True, cfg, p)


def weight(p, arm):
    source = OUT / f'{arm}.pth' if arm in ('behavior-mc', 'deploy-mc') else r.OLD / 'artifacts/nominal.pth'
    p.load_state_dict(torch.load(source, map_location=p.device))
    p.eval()
    p.set_phase('test')
    return source


def run_episode(cfg, p, scene, case, humans, arm='nominal', snap=None, output=None):
    env = r.environment(cfg, p, scene, case, humans, snap, 50)
    assert env.time_limit == 50 and len(p.action_space) == 80 and not env.robot.visible
    records, arrays, timings = [], [], []
    distances = []
    negative = remaining = triggers = interventions = fallback = 0
    anchor = 0.
    for _ in range(203):
        step = int(round(env.global_time / .25))
        assert step + 1 < len(humans)
        start = time.perf_counter()
        part = r.decompose(p, env)
        index = part['best']
        distance = float(np.linalg.norm(np.array(env.robot.get_position()) - env.robot.get_goal_position()))
        distances.append(distance)
        negative = negative + 1 if part['progress'][index] <= 0 else 0
        allowed = part['safe']
        fallback += int(not allowed.any())
        if arm == 'progress' and allowed.any():
            index = part['forward']
        if arm in ('commit8', 'adaptive'):
            if remaining and not allowed.any():
                remaining = 0
            if arm == 'adaptive' and remaining and anchor - distance >= .5 and part['progress'][index] >= .02:
                remaining = 0
            if not remaining and negative >= 8 and len(distances) >= 9 and distances[-9] <= distance:
                if allowed.any():
                    remaining = 8 if arm == 'commit8' else 16
                    anchor = distance
                    triggers += 1
                    negative = 0
            if remaining:
                index = part['forward']
                remaining -= 1
                interventions += 1
        action = part['commands'][index]
        timings.append(time.perf_counter() - start)
        assert allowed[index] or not allowed.any(), 'Intervention violated unchanged safety mask'
        before = m.before_geometry(env)
        position = env.robot.get_position()
        m.commit_action(p, part['token'], action, index)
        _, reward, term, trunc, info = env.step(action)
        np.testing.assert_allclose([[h.px, h.py, h.vx, h.vy, h.radius] for h in env.humans], humans[step + 1], rtol=0, atol=1e-8)
        clearance = m.geo_step(env, before)
        records.append([step, *position, action.vx, action.vy, index, part['best'], reward, clearance,
                        part['progress'][index], part['progress'][part['best']], remaining])
        arrays.append(np.stack([part['value'], part['q'], part['scores'], part['progress'], part['clear'], allowed.astype(float)]))
        if term or trunc:
            break
    assert term or trunc
    a = np.asarray(records, np.float64)
    result = {'arm': arm, 'scene': m.SCENES[scene], 'case': case, 'outcome': info['event'],
              'success': int(info['event'] == 'reach_goal'), 'collision': int(info['event'] == 'collision'),
              'timeout': int(info['event'] == 'timeout'), 'contact': int((a[:, 8] < 0).any()),
              'min_clearance': float(a[:, 8].min()), 'duration': len(a) * .25,
              'steps': len(a), 'return': float(a[:, 7] @ (.99 ** np.arange(len(a)))),
              'path': float(np.linalg.norm(a[:, 3:5], axis=1).sum() * .25),
              'final_distance': float(np.linalg.norm(np.array(env.robot.get_position()) - env.robot.get_goal_position())),
              'triggers': triggers, 'intervention_steps': interventions, 'fallback': fallback,
              'decision_ms_mean': float(np.mean(timings) * 1000)}
    if output:
        path = OUT / output
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path.with_suffix('.npz'), records=a, candidates=np.asarray(arrays, np.float32))
        m.save_json(path.with_suffix('.json'), result)
    return result


def evidence():
    cfg, p = setup()
    stats = json.loads((OLD / 'statistics.json').read_text())
    wanted = {tuple(v) for v in SPEC['B_evidence']['roots']}
    rows = []
    for root in q.roots():
        scene, case = m.SCENES.index(root['scene']), root['case']
        if (scene, case) not in wanted:
            continue
        humans = old_tape(cfg, p, root)
        _, snap = r.frozen_root(cfg, p, root, humans)
        env = r.environment(cfg, p, scene, case, humans, snap, 50)
        witness = np.load(r.FINDINGS / 'artifacts' / f'counterfactual-value-free-progress-{scene}-{case}.npz')['records']
        steps, candidates, windows = [], [], []
        for row in witness[:16]:
            part = r.decompose(p, env)
            np.testing.assert_allclose(env.robot.get_position(), row[1:3], rtol=0, atol=1e-9)
            delta = np.linalg.norm(np.asarray(part['commands']) - row[3:5], axis=1)
            index = int(delta.argmin())
            assert delta[index] < 1e-9 and part['safe'][index]
            raw = part['q'].copy()
            raw[0] -= .001
            best = part['best']
            steps.append({'step': int(row[0]), 'nominal': best, 'witness': index,
                          'witness_safe': True, 'nominal_progress': float(part['progress'][best]),
                          'witness_progress': float(part['progress'][index]),
                          'nominal_Q_advantage': float(part['q'][best] - part['q'][index]),
                          'raw_argmax': int(raw.argmax()), 'mask_changes_raw_argmax': bool(raw.argmax() != best),
                          'nominal_would_cancel_witness': bool(best != index)})
            candidates.append(np.stack([part[k] for k in ['value', 'q', 'scores', 'progress', 'clear']]))
            windows.append(part['windows'].cpu().numpy())
            action = part['commands'][index]
            m.commit_action(p, part['token'], action, index)
            _, _, term, trunc, _ = env.step(action)
            assert not (term or trunc)
        base = next(v for v in stats['cases'] if q.key(v) == q.key(root))
        assert base['category'] == 'MULTISTEP_POLICY_TRAP_TESTED_ACTION_SET'
        np.savez_compressed(OUT / f'trap-evidence-{scene}-{case}.npz', candidates=np.asarray(candidates), windows=np.asarray(windows))
        rows.append({'root': root, 'steps': steps, 'existing_successful_prefixes': base['successful_prefixes'],
                     'existing_A_return_range': base['A_return_range'],
                     'value_interrupts': sum(v['nominal_would_cancel_witness'] and v['nominal_Q_advantage'] > 0 for v in steps),
                     'safe_witness_steps': len(steps), 'mask_interrupts': sum(v['mask_changes_raw_argmax'] for v in steps)})
    gate = sum(v['value_interrupts'] > 0 for v in rows) >= 2
    save('B-evidence.json', {'roots': rows, 'prototype_gate_passed': gate,
                            'scope': 'Saved prefix success reused; no claim of training cause or novel commitment algorithm'})
    assert gate, 'Do not develop B prototype without repeated value interruption evidence'
    print('B EVIDENCE', [(q.key(v['root']), v['value_interrupts'], v['mask_interrupts']) for v in rows], flush=True)


def collect():
    cfg, p = setup()
    windows, behavior, deploy, metadata = [], [], [], []
    started = time.perf_counter()
    for scene in range(6):
        for case in range(2000, 2016):
            humans = tape(cfg, p, scene, case)
            env = r.environment(cfg, p, scene, case, humans, None, 50)
            p.set_phase('train')
            ordinal = scene * 16 + case - 2000
            eps = .30 + (.05 - .30) * ordinal / 95
            rng = np.random.default_rng(121012 + ordinal)
            tokens, rewards, actions, snapshots = [], [], [], []
            for step in range(203):
                if step in SPEC['A_training']['capture_steps']:
                    snapshots.append((step, r.snapshot(p, env)))
                commands, tok, _, index = m.score(p, env)
                if rng.random() < eps:
                    index = int(rng.integers(80))
                action = commands[index]
                tokens.append(tok)
                actions.append([action.vx, action.vy, index])
                m.commit_action(p, tok, action, index)
                _, reward, term, trunc, info = env.step(action)
                rewards.append(reward)
                if term or trunc:
                    break
            assert term or trunc
            returns = np.zeros(len(rewards), np.float64)
            g = 0.
            for j in range(len(rewards) - 1, -1, -1):
                g = rewards[j] + .99 * g
                returns[j] = g
            folder = OUT / 'behavior'
            folder.mkdir(exist_ok=True)
            np.savez_compressed(folder / f'{scene}-{case}.npz', tokens=np.asarray(tokens), rewards=rewards,
                                actions=actions, returns=returns, epsilon=eps)
            for step, snap in snapshots:
                x = m.window(tokens[:step + 1])
                check = r.environment(cfg, p, scene, case, humans, snap, 50)
                np.testing.assert_array_equal(x, m.window(list(p._history) + [m.token(p, m.state(check))]))
                result = run_episode(cfg, p, scene, case, humans, snap=snap,
                                     output=f'deploy-labels/{scene}-{case}-{step}')
                windows.append(x)
                behavior.append(float(returns[step]))
                deploy.append(result['return'])
                metadata.append({'scene': scene, 'case': case, 'step': step, 'epsilon': eps,
                                 'behavior_outcome': info['event'], 'deploy_outcome': result['outcome'],
                                 'behavior_return': float(returns[step]), 'deploy_return': result['return'],
                                 'robot': vars(snap['robot']), 'last_action': list(snap['last_action'])})
            save('collection-progress.json', {'episodes': ordinal + 1, 'states': len(windows),
                                            'seconds': time.perf_counter() - started})
            print('COLLECT', ordinal + 1, '/96 states', len(windows), 'seconds', round(time.perf_counter() - started, 1), flush=True)
    np.savez_compressed(OUT / 'matched-data.npz', windows=np.asarray(windows, np.float32),
                        behavior=np.asarray(behavior, np.float32), deploy=np.asarray(deploy, np.float32),
                        case=np.asarray([v['case'] for v in metadata], np.int64),
                        scene=np.asarray([v['scene'] for v in metadata], np.int64))
    save('matched-data.json', {'states': metadata, 'seconds': time.perf_counter() - started,
                             'sha256': sha(OUT / 'matched-data.npz'), 'historical_replay_restored': False})


def train():
    cfg, p = setup()
    data = np.load(OUT / 'matched-data.npz')
    fit = data['case'] < 2012
    assert fit.any() and (~fit).any()
    x = torch.as_tensor(data['windows'], device=p.device)
    initial = copy.deepcopy(p.state_dict())
    rng = np.random.default_rng(121012)
    batches = rng.choice(np.flatnonzero(fit), size=(2000, 128))
    np.save(OUT / 'batch-indices.npy', batches)
    records = {}
    for arm, column in [('behavior-mc', 'behavior'), ('deploy-mc', 'deploy')]:
        p.load_state_dict(initial)
        optimizer = torch.optim.Adam(p.parameters(), lr=1e-5)
        y = torch.as_tensor(data[column], device=p.device)
        curves = []
        start = time.perf_counter()
        for j, indices in enumerate(batches):
            torch.manual_seed(121012 + j)
            p.train()
            optimizer.zero_grad(set_to_none=True)
            pred = p.forward_value(x[indices])
            loss = (pred - y[indices]).square().mean()
            assert torch.isfinite(loss), 'Nonfinite training loss'
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(p.parameters(), 10.)
            assert torch.isfinite(norm)
            optimizer.step()
            if (j + 1) % 100 == 0:
                p.eval()
                with torch.no_grad():
                    prediction = torch.cat([p.forward_value(x[k:k + 128]) for k in range(0, len(x), 128)]).cpu().numpy()
                curves.append({'update': j + 1, 'batch_MSE': float(loss),
                               'fit_MAE': float(np.abs(prediction[fit] - data[column][fit]).mean()),
                               'heldout_source_MAE': float(np.abs(prediction[~fit] - data[column][~fit]).mean()),
                               'heldout_deploy_MAE': float(np.abs(prediction[~fit] - data['deploy'][~fit]).mean())})
                save(f'{arm}-curves.json', curves)
                print('TRAIN', arm, j + 1, 'MSE', round(float(loss), 6), flush=True)
        torch.save(p.state_dict(), OUT / f'{arm}.pth')
        records[arm] = {'updates': 2000, 'seconds': time.perf_counter() - start,
                        'sha256': sha(OUT / f'{arm}.pth'), 'curves': curves}
    save('training.json', {'arms': records, 'shared_initialization': SPEC['parent_sha256'],
                           'batch_sha256': sha(OUT / 'batch-indices.npy'), 'same_dropout_rng': True,
                           'same_optimizer_initial_state': True, 'same_state_inputs': True,
                           'fit_states': int(fit.sum()), 'heldout_states': int((~fit).sum())})


def evaluate(cohort, arm):
    assert arm in ARMS
    cfg, p = setup()
    source = weight(p, arm)
    cases = []
    if cohort == 'roots':
        stats = json.loads((OLD / 'statistics.json').read_text())
        for item in stats['cases']:
            if item['category'] == 'NO_CONTACT_FREE_RESCUE_IN_TESTED_SINGLE_OR_PREFIX_ACTIONS':
                continue
            root = {k: item[k] for k in ('scene', 'case', 'step', 'group', 'partition')}
            if arm == 'nominal':
                result = copy.deepcopy(item['selected_A'])
                result.update({'arm': arm, 'root': root, 'reused': True, 'category': item['category']})
            else:
                humans = old_tape(cfg, p, root)
                _, snap = r.frozen_root(cfg, p, root, humans)
                # Restore the original nominal pre-root history, not an invented new-model trajectory.
                result = run_episode(cfg, p, m.SCENES.index(root['scene']), root['case'], humans,
                                     arm, snap, f'roots/{arm}/{q.key(root)}')
                result.update({'root': root, 'category': item['category'], 'reused': False})
            cases.append(result)
            save(f'{cohort}-{arm}.json', {'weight_sha256': sha(source), 'results': cases})
    elif cohort == 'fresh':
        assert (HERE / 'training.json').exists()
        for scene in range(6):
            for case in range(300, 320):
                dest = OUT / f'fresh/{arm}/{scene}-{case}.json'
                if dest.exists():
                    result = json.loads(dest.read_text())
                else:
                    humans = tape(cfg, p, scene, case)
                    result = run_episode(cfg, p, scene, case, humans, arm,
                                         output=f'fresh/{arm}/{scene}-{case}')
                cases.append(result)
                save(f'{cohort}-{arm}.json', {'weight_sha256': sha(source), 'results': cases})
                if len(cases) % 10 == 0:
                    print('EVAL', cohort, arm, len(cases), 'success', sum(v['success'] for v in cases), flush=True)
    else:
        raise ValueError(cohort)


def probe():
    cfg, p = setup()
    results = {}
    for arm in ('nominal', 'behavior-mc', 'deploy-mc'):
        weight(p, arm)
        rows = []
        for root in q.roots():
            inputs = np.load(OLD / 'catalogue' / q.key(root) / 'inputs.npz')
            humans = old_tape(cfg, p, root)
            env, _ = r.frozen_root(cfg, p, root, humans)
            env.time_limit = 50.
            part = r.decompose(p, env)
            with torch.no_grad():
                values = p.forward_candidate_values(torch.as_tensor(inputs['windows'], device=p.device)).cpu().numpy()
            qt = inputs['first'][:, 0] + .99 * (1 - inputs['first'][:, 1]) * values
            chosen = int(part['best'])
            branches = {}
            for i in np.flatnonzero(inputs['safe']):
                branch = OLD / 'catalogue' / q.key(root) / f'action-{i:02d}-A_deploy.json'
                branches[int(i)] = json.loads(branch.read_text())['result']
            assert chosen in branches
            ga = np.asarray([branches[int(i)]['return'] for i in np.flatnonzero(inputs['safe'])])
            pred = qt[inputs['safe']]
            pairs = np.triu(np.ones((len(ga), len(ga)), bool), 1) & (np.abs(ga[:, None] - ga[None, :]) > 1e-6)
            inversion = ((ga[:, None] - ga[None, :]) * (pred[:, None] - pred[None, :]) < 0) & pairs
            rows.append({'root': root, 'chosen': chosen, 'old_policy_first_action_result': branches[chosen],
                         'actual_next_V_MAE': float(np.abs((inputs['first'][inputs['safe'], 0] * -1 + ga) / .99 - values[inputs['safe']]).mean()),
                         'pair_inversions': int(inversion.sum()), 'nontied_pairs': int(pairs.sum()),
                         'regret_fixed_old_policy': float(ga.max() - branches[chosen]['return'])})
        results[arm] = rows
    save('ranking-probes.json', results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['evidence', 'collect', 'train', 'evaluate', 'probe'])
    parser.add_argument('--cohort', choices=['roots', 'fresh'], default='roots')
    parser.add_argument('--arm', choices=ARMS, default='nominal')
    args = parser.parse_args()
    try:
        if args.stage == 'evaluate':
            evaluate(args.cohort, args.arm)
        else:
            globals()[args.stage]()
    except Exception as exc:
        save(f'engineering-{args.stage}-{args.arm}.json', {'status': 'STOPPED', 'error': repr(exc)})
        raise
