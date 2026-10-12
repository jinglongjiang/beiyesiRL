"""Frozen existing-method adaptation, complete paired navigation, and stop gates."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from controller import SideController


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


previous = HERE.parent / 'gru-causal-repairs'
h = load('frozen_causal_runner', previous / 'run.py')
analysis = load('frozen_paired_statistics', previous / 'analyze.py')
r, m = h.r, h.m
SPEC = json.loads((HERE / 'protocol.json').read_text())
OUT = HERE / 'artifacts'
h.OUT = OUT


def save(path, data):
    m.save_json(HERE / path, data)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze():
    paths = [p for folder in ['crowd_nav', 'crowd_sim'] for p in (ROOT / folder).rglob('*.py')]
    paths += [r.OLD / 'artifacts/nominal.pth', r.OLD / 'parent-config.ini']
    paths += [previous / f'fresh-{arm}.json' for arm in ['nominal', 'commit8']]
    paths += [HERE / name for name in ['protocol.json', 'controller.py', 'run.py', 'THIRD-PARTY-NOTICES.txt']]
    manifest = {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}
    assert manifest[str((r.OLD / 'artifacts/nominal.pth').relative_to(ROOT))] == SPEC['parent_sha256']
    save('frozen-manifest.json', {'files': manifest, 'generated_before_outcomes': True})
    print('FROZEN', len(manifest), 'files', flush=True)


def verify_manifest():
    manifest = json.loads((HERE / 'frozen-manifest.json').read_text())
    for name, digest in manifest['files'].items():
        assert digest and sha(ROOT / name) == digest, f'Frozen source changed:{name}'


def humans(cfg, p, scene, case):
    old = previous / 'artifacts/tapes' / f'{scene}-{case}.npz'
    if old.exists():
        with np.load(old) as data:
            return data['humans'].copy()
    return h.tape(cfg, p, scene, case)


def side_episode(cfg, p, scene, case, tape, arm, output):
    env = r.environment(cfg, p, scene, case, tape, None, 50)
    assert env.time_limit == 50 and env.time_step == .25 and not env.robot.visible
    assert p.temporal_backbone == 'gru' and len(p.action_space) == 80
    controller = SideController(latched=arm == 'side-latched')
    records, candidates, timings = [], [], []
    fallbacks = 0
    for _ in range(203):
        step = int(round(env.global_time / .25))
        assert step + 1 < len(tape)
        start = time.perf_counter()
        part = r.decompose(p, env)
        position = np.asarray(env.robot.get_position())
        velocity = np.array([env.robot.vx, env.robot.vy])
        index, penalty, active = controller.select(part, position, env.robot.get_goal_position(), velocity)
        fallbacks += int(not part['safe'].any())
        assert part['safe'][index] or not part['safe'].any()
        action = part['commands'][index]
        timings.append(time.perf_counter() - start)
        before = m.before_geometry(env)
        m.commit_action(p, part['token'], action, index)
        assert p._last_action == action
        _, reward, term, trunc, info = env.step(action)
        np.testing.assert_allclose([[v.px, v.py, v.vx, v.vy, v.radius] for v in env.humans],
                                   tape[step + 1], rtol=0, atol=1e-8)
        np.testing.assert_allclose([env.robot.vx, env.robot.vy], action, rtol=0, atol=1e-12)
        np.testing.assert_allclose(env.robot.get_position(), position + np.asarray(action) * .25,
                                   rtol=0, atol=1e-12)
        clearance = m.geo_step(env, before)
        records.append([step, *position, action.vx, action.vy, index, part['best'], reward,
                        clearance, part['progress'][index], part['progress'][part['best']],
                        controller.remaining, int(active), controller.side, *controller.axis])
        candidates.append(np.stack([part['value'], part['q'], part['scores'], part['progress'],
                                   part['clear'], part['safe'].astype(float), penalty, part['scores'] - penalty]))
        if term or trunc:
            break
    assert term or trunc
    a = np.asarray(records, np.float64)
    result = {'arm': arm, 'scene': m.SCENES[scene], 'case': case, 'outcome': info['event'],
              'success': int(info['event'] == 'reach_goal'), 'collision': int(info['event'] == 'collision'),
              'timeout': int(info['event'] == 'timeout'), 'contact': int((a[:, 8] < 0).any()),
              'min_clearance': float(a[:, 8].min()), 'duration': len(a) * .25, 'steps': len(a),
              'return': float(a[:, 7] @ (.99 ** np.arange(len(a)))),
              'path': float(np.linalg.norm(a[:, 3:5], axis=1).sum() * .25),
              'final_distance': float(np.linalg.norm(np.asarray(env.robot.get_position()) - env.robot.get_goal_position())),
              'triggers': controller.triggers, 'intervention_steps': controller.active_steps,
              'action_changes': controller.changes, 'fallback': fallbacks,
              'decision_ms_mean': float(np.mean(timings) * 1000)}
    target = OUT / output
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target.with_suffix('.npz'), records=a, candidates=np.asarray(candidates, np.float32))
    m.save_json(target.with_suffix('.json'), result)
    return result


def units():
    from controller import directional_penalty, side_geometry
    angles = np.arange(80) * (2 * np.pi / 80)
    commands = np.c_[np.cos(angles), np.sin(angles)]
    safe = np.ones(80, bool)
    scores = np.linspace(0, .1, 80)
    part = {'commands': commands, 'safe': safe, 'scores': scores, 'progress': -np.ones(80), 'best': int(scores.argmax())}
    c = SideController()
    for _ in range(8):
        _, penalty, active = c.select(part, np.zeros(2), np.array([1., 0]), np.array([0., 1.]))
        assert not active and not penalty.any()
    _, penalty, active = c.select(part, np.zeros(2), np.array([1., 0]), np.array([0., 1.]))
    assert active and c.remaining == 7 and c.side == 1 and c.changes > 0
    axis, side = c.axis.copy(), c.side
    c.select(part, np.zeros(2), np.array([0., 1.]), np.array([0., -1.]))
    np.testing.assert_array_equal(c.axis, axis)
    assert c.side == side
    bad = dict(part, safe=np.zeros(80, bool))
    chosen, penalty, active = c.select(bad, np.zeros(2), np.ones(2), np.ones(2))
    assert chosen == bad['best'] and c.remaining == 0 and not active and not penalty.any()
    mask = safe.copy()
    mask[:20] = False
    masked = scores.copy()
    masked[~mask] = -1e9
    np.testing.assert_array_equal(directional_penalty(axis, side, commands, masked, mask),
                                  directional_penalty(axis, side, commands, scores, mask))
    angle = .431
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    pos, goal, vel = np.array([2., 3]), np.array([3., 3]), np.array([0., 1.])
    a1, s1 = side_geometry(pos, goal, vel, commands, scores, safe)
    a2, s2 = side_geometry(rotation @ pos, rotation @ goal, rotation @ vel, commands @ rotation.T, scores, safe)
    assert s1 == s2
    np.testing.assert_allclose(directional_penalty(a1, s1, commands, scores, safe),
                               directional_penalty(a2, s2, commands @ rotation.T, scores, safe), rtol=0, atol=1e-14)
    np.testing.assert_allclose(side_geometry(pos + 7, goal + 7, vel, commands, scores, safe)[0], a1)
    assert np.any(directional_penalty(a1, 1, commands, scores, safe)
                  != directional_penalty(a1, -1, commands, scores, safe))
    return {'trigger_clock': True, 'retained_direction': True, 'native_fallback': True,
            'no_masked_score_scaling': True, 'rotation_translation_equivariance': True,
            'direction_sensitivity': True}


def controls():
    verify_manifest()
    unit = units()
    cfg, p = h.setup()
    h.weight(p, 'nominal')
    parity, smoke = [], []
    for scene in range(6):
        tape = humans(cfg, p, scene, 300)
        result = h.run_episode(cfg, p, scene, 300, tape, output=f'controls/nominal/{scene}-300')
        old = previous / f'artifacts/fresh/nominal/{scene}-300'
        current = OUT / f'controls/nominal/{scene}-300'
        with np.load(old.with_suffix('.npz')) as expected, np.load(current.with_suffix('.npz')) as actual:
            np.testing.assert_array_equal(actual['records'], expected['records'])
            np.testing.assert_array_equal(actual['candidates'], expected['candidates'])
        for key, value in json.loads(old.with_suffix('.json').read_text()).items():
            if key != 'decision_ms_mean':
                assert result[key] == value, f'Parent parity:{key}'
        parity.append({'scene': scene, 'case': 300, 'exact_actions_and_candidates': True})
        tape = humans(cfg, p, scene, 301)
        first = side_episode(cfg, p, scene, 301, tape, 'side-latched', f'controls/first/{scene}-301')
        second = side_episode(cfg, p, scene, 301, tape, 'side-latched', f'controls/second/{scene}-301')
        for name in ['records', 'candidates']:
            with np.load(OUT / f'controls/first/{scene}-301.npz') as x, np.load(OUT / f'controls/second/{scene}-301.npz') as y:
                np.testing.assert_array_equal(x[name], y[name])
        assert {k: v for k, v in first.items() if k != 'decision_ms_mean'} == {k: v for k, v in second.items() if k != 'decision_ms_mean'}
        smoke.append({'scene': scene, 'case': 301, 'exact_repeat': True, 'outcome': first['outcome'],
                      'changes': first['action_changes']})
    verify_manifest()
    save('controls.json', {'passed': True, 'units': unit, 'parent_parity': parity, 'determinism': smoke,
                           'grid': 80, 'time_limit': 50, 'checkpoint_sha256': SPEC['parent_sha256']})
    print('CONTROLS PASSED', flush=True)


def evaluate(cohort, arm):
    verify_manifest()
    assert json.loads((HERE / 'controls.json').read_text())['passed']
    assert arm in SPEC['arms']
    if cohort == 'confirmation':
        assert json.loads((HERE / 'development-summary.json').read_text())['development_gate']['passed']
    cfg, p = h.setup()
    h.weight(p, 'nominal')
    spec = SPEC[cohort]
    results = []
    started = time.perf_counter()
    for scene in spec['scenes']:
        for case in range(spec['case_first'], spec['case_last'] + 1):
            output = f'{cohort}/{arm}/{scene}-{case}'
            if cohort == 'development' and arm in ['nominal', 'commit8']:
                source = previous / f'artifacts/fresh/{arm}/{scene}-{case}.json'
                result = json.loads(source.read_text())
                result = dict(result, reused=True, source=str(source.relative_to(ROOT)), source_sha256=sha(source))
            else:
                target = OUT / (output + '.json')
                if target.exists():
                    result = json.loads(target.read_text())
                else:
                    tape = humans(cfg, p, scene, case)
                    if arm in ['nominal', 'commit8']:
                        result = h.run_episode(cfg, p, scene, case, tape, arm=arm, output=output)
                    else:
                        result = side_episode(cfg, p, scene, case, tape, arm, output)
            results.append(result)
            save(f'{cohort}-{arm}.json', {'results': results, 'seconds': time.perf_counter() - started,
                                        'checkpoint_sha256': SPEC['parent_sha256']})
            print(cohort, arm, len(results), '/120', result['scene'], case, result['outcome'], flush=True)
    assert len(results) == 120
    verify_manifest()


def summarize(cohort):
    rows = {arm: json.loads((HERE / f'{cohort}-{arm}.json').read_text())['results'] for arm in SPEC['arms']}
    assert all(len(v) == 120 for v in rows.values())
    overall = {arm: analysis.aggregate(v) for arm, v in rows.items()}
    comparisons = {f'side-latched-vs-{arm}': analysis.pairs(rows['side-latched'], rows[arm])
                   for arm in ['nominal', 'commit8', 'side-reset']}
    ours = overall['side-latched']
    refs = [overall[a] for a in ['nominal', 'commit8']]
    pair = comparisons['side-latched-vs-commit8']
    both = [(a, b) for a, b in zip(rows['side-latched'], rows['commit8']) if a['success'] and b['success']]
    time_ratio = float(np.mean([a['duration'] for a, _ in both]) / np.mean([b['duration'] for _, b in both])) if both else None
    checks = {'at_least_two_successes_over_commit8': ours['success'] >= overall['commit8']['success'] + 2,
              'beats_nominal_success': ours['success'] > overall['nominal']['success'],
              'collision_nonregression': all(ours['collision'] <= a['collision'] for a in refs),
              'contact_nonregression': all(ours['contact'] <= a['contact'] for a in refs),
              'changes_executed_decisions': sum(v['action_changes'] for v in rows['side-latched']) > 0,
              'paired_time_nonregression': time_ratio is not None and time_ratio <= 1.10}
    passed = all(checks.values())
    if cohort == 'confirmation':
        checks['paired_lower_bound_positive_over_commit8'] = pair['delta_SR_case_bootstrap_95'][0] > 0
        checks['not_absorbed_by_side_reset'] = ours['success'] > overall['side-reset']['success'] and ours['contact'] <= overall['side-reset']['contact']
        passed = all(checks.values())
    output = {'cohort': cohort, 'overall': overall, 'pairwise': comparisons,
              'per_scene': {s: {a: analysis.aggregate([v for v in rs if v['scene'] == s]) for a, rs in rows.items()} for s in m.SCENES},
              'development_gate' if cohort == 'development' else 'confirmation_gate': {'passed': passed, 'checks': checks, 'paired_success_time_ratio': time_ratio},
              'decision': 'GO_CONFIRMATION' if passed and cohort == 'development' else 'RETAIN_EXISTING_METHOD_ADAPTATION' if passed else 'KILL_CURRENT_ADAPTATION',
              'new_algorithmic_contribution_established': False}
    save(f'{cohort}-summary.json', output)
    print(json.dumps(output['overall'], indent=2), flush=True)
    print('DECISION', output['decision'], checks, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['freeze', 'controls', 'evaluate', 'summarize'])
    parser.add_argument('--cohort', choices=['development', 'confirmation'], default='development')
    parser.add_argument('--arm', choices=SPEC['arms'], default='side-latched')
    args = parser.parse_args()
    try:
        if args.stage == 'freeze':
            freeze()
        elif args.stage == 'controls':
            controls()
        elif args.stage == 'evaluate':
            evaluate(args.cohort, args.arm)
        else:
            summarize(args.cohort)
    except Exception as exc:
        save(f'engineering-{args.stage}-{args.cohort}-{args.arm}.json', {'status': 'STOPPED', 'error': repr(exc)})
        raise
