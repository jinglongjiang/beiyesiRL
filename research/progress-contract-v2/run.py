"""Frozen paired closed-loop tests; all privileged tapes stay outside the controller."""
import argparse
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
sys.path.insert(0, str(HERE))
from controller import Controller


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


h = load('causal_parent_progress_contract', HERE.parent / 'gru-causal-repairs/run.py')
stats = load('case_statistics_progress_contract', HERE.parent / 'gru-causal-repairs/analyze.py')
r, m = h.r, h.m
SPEC = json.loads((HERE / 'protocol.json').read_text())
OUT = HERE / 'artifacts'
h.OUT = OUT


def save(name, value):
    m.save_json(HERE / name, value)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze():
    paths = [p for folder in ['crowd_nav', 'crowd_sim'] for p in (ROOT / folder).rglob('*.py')]
    paths += [r.OLD / 'artifacts/nominal.pth', r.OLD / 'parent-config.ini']
    paths += list(HERE.glob('*.py')) + [HERE / 'protocol.json']
    paths += [HERE.parent / 'progress-contract/controller.py', HERE.parent / 'progress-contract/development-low-commit8.json']
    paths += [HERE.parent / f'gru-causal-repairs/fresh-{a}.json' for a in ['nominal', 'commit8']]
    manifest = {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}
    assert sha(r.OLD / 'artifacts/nominal.pth') == SPEC['parent_sha256']
    import pytorch_mppi.mppi as upstream
    installed = Path(upstream.__file__)
    source = Path('/workspace/research-deps/pytorch-mppi-20261012/src/pytorch_mppi/mppi.py')
    assert sha(installed) == sha(source)
    save('frozen-manifest.json', {'files': manifest, 'upstream_mppi_sha256': sha(installed),
                                  'upstream_commit': SPEC['upstream']['commit']})
    print('FROZEN', len(manifest), flush=True)


def verify():
    manifest = json.loads((HERE / 'frozen-manifest.json').read_text())
    for name, digest in manifest['files'].items():
        assert digest and sha(ROOT / name) == digest, name
    import pytorch_mppi.mppi as upstream
    assert sha(upstream.__file__) == manifest['upstream_mppi_sha256']


def humans(cfg, p, scene, case):
    old = HERE.parent / f'gru-causal-repairs/artifacts/tapes/{scene}-{case}.npz'
    if old.exists():
        with np.load(old) as data:
            return data['humans'].copy()
    return h.tape(cfg, p, scene, case)


def episode(cfg, p, scene, case, tape, arm, output):
    env = r.environment(cfg, p, scene, case, tape, None, 50)
    assert env.time_limit == 50 and env.time_step == .25 and not env.robot.visible
    assert len(p.action_space) == 80 and p.temporal_backbone == 'gru'
    controller = Controller(arm, p)
    rows, candidates, diagnostics, times = [], [], [], []
    fallback = 0
    for _ in range(203):
        step = round(env.global_time / .25)
        start = time.perf_counter()
        part = r.decompose(p, env)
        pos = np.asarray(env.robot.get_position())
        legal = [[v.px, v.py, v.vx, v.vy, v.radius] for v in env.humans]
        last = None if p._last_action is None else list(p._last_action)
        index, diagnostic = controller.select(part, pos, env.robot.get_goal_position(), legal,
                                               last, env.robot.radius,
                                               812121 + scene * 100000 + case * 300 + step,
                                               env.global_time)
        assert part['safe'][index] or not part['safe'].any()
        fallback += int(not part['safe'].any())
        action = part['commands'][index]
        times.append(time.perf_counter() - start)
        before = m.before_geometry(env)
        m.commit_action(p, part['token'], action, index)
        _, reward, term, trunc, info = env.step(action)
        np.testing.assert_allclose([[v.px, v.py, v.vx, v.vy, v.radius] for v in env.humans], tape[step + 1], rtol=0, atol=1e-8)
        np.testing.assert_allclose(env.robot.get_position(), pos + .25 * np.asarray(action), rtol=0, atol=1e-12)
        clear = m.geo_step(env, before)
        rows.append([step, *pos, *action, index, part['best'], reward, clear,
                     part['progress'][index], part['progress'][part['best']], controller.remaining])
        candidates.append(np.stack([part[k] for k in ['value', 'q', 'scores', 'progress', 'clear', 'safe']]))
        diagnostics.append(diagnostic)
        if term or trunc:
            break
    assert term or trunc
    a = np.asarray(rows, np.float64)
    result = {'arm': arm, 'scene': m.SCENES[scene], 'case': case, 'outcome': info['event'],
              'success': int(info['event'] == 'reach_goal'), 'collision': int(info['event'] == 'collision'),
              'timeout': int(info['event'] == 'timeout'), 'contact': int((a[:, 8] < 0).any()),
              'min_clearance': float(a[:, 8].min()), 'duration': len(a) * .25, 'steps': len(a),
              'return': float(a[:, 7] @ (.99 ** np.arange(len(a)))),
              'path': float(np.linalg.norm(a[:, 3:5], axis=1).sum() * .25),
              'final_distance': float(np.linalg.norm(np.asarray(env.robot.get_position()) - env.robot.get_goal_position())),
              'triggers': controller.triggers, 'intervention_steps': controller.active_steps,
              'action_changes': controller.changes, 'fallback': fallback,
              'rejected_plan_count': controller.rejections, 'retained_plan_steps': controller.retained,
              'invalidations': controller.invalidations, 'completed_obligations': controller.completed,
              'decision_ms_mean': float(np.mean(times) * 1000),
              'decision_ms_p95': float(np.quantile(times, .95) * 1000)}
    target = OUT / output
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target.with_suffix('.npz'), records=a, candidates=np.asarray(candidates, np.float32))
    m.save_json(target.with_suffix('.json'), result)
    m.save_json(target.with_name(target.name + '-mechanism.json'), diagnostics)
    return result


def controls():
    verify()
    cfg, p = h.setup()
    h.weight(p, 'nominal')
    results = []
    for scene in range(6):
        tape = humans(cfg, p, scene, 301)
        env = r.environment(cfg, p, scene, 301, tape, None, 50)
        part = r.decompose(p, env)
        c = Controller('deadline', p)
        c.goal = torch.tensor(env.robot.get_goal_position(), dtype=torch.float64)
        c.humans = torch.tensor([[v.px, v.py, v.vx, v.vy, v.radius] for v in env.humans], dtype=torch.float64)
        c.radius = env.robot.radius
        assert c.free_value(torch.tensor([1., 2.], dtype=torch.float64))[0] > c.free_value(torch.tensor([1., 2.], dtype=torch.float64))[1]
        for prior in [None, np.array([.2, -.1])]:
            p._last_action = None if prior is None else m.ActionXY(*prior)
            part = r.decompose(p, env)
            state = torch.tensor([*env.robot.get_position(), *(prior if prior is not None else [0., 0.]),
                                  -1. if prior is None else 0.], dtype=torch.float64)
            predicted = c.dynamics(state.repeat(80, 1), c.raw)
            np.testing.assert_allclose(predicted[:, 2:4].numpy(), np.asarray(part['commands']), rtol=0, atol=1e-15)
            np.testing.assert_allclose(c.clearance(predicted).numpy(), part['clear'], rtol=0, atol=1e-12)
        first = episode(cfg, p, scene, 301, tape, 'deadline', f'controls/first/{scene}-301')
        second = episode(cfg, p, scene, 301, tape, 'deadline', f'controls/second/{scene}-301')
        with np.load(OUT / f'controls/first/{scene}-301.npz') as x, np.load(OUT / f'controls/second/{scene}-301.npz') as y:
            np.testing.assert_array_equal(x['records'], y['records'])
            np.testing.assert_array_equal(x['candidates'], y['candidates'])
        assert {k: v for k, v in first.items() if not k.startswith('decision_ms')} == {k: v for k, v in second.items() if not k.startswith('decision_ms')}
        results.append(first)
        print('CONTROL', scene, first['outcome'], first['triggers'], first['action_changes'], flush=True)
    verify()
    save('controls.json', {'passed': True, 'grid': 80, 'time_limit': 50,
                          'executed_projection_and_smoothing': True, 'first_clearance_matches_native': True,
                          'exact_repeat': True, 'results': results})


def evaluate(cohort, arm):
    verify()
    assert json.loads((HERE / 'controls.json').read_text())['passed']
    assert arm in SPEC['arms']
    if cohort == 'confirmation':
        assert json.loads((HERE / 'development-summary.json').read_text())['gate']['passed']
    cfg, p = h.setup()
    h.weight(p, 'nominal')
    cohort_spec = SPEC[cohort]
    rows = []
    start = time.perf_counter()
    for scene in cohort_spec['scenes']:
        for case in range(cohort_spec['case_first'], cohort_spec['case_last'] + 1):
            output = f'{cohort}/{arm}/{scene}-{case}'
            if cohort == 'development' and arm == 'low-commit8':
                source = HERE.parent / 'progress-contract/development-low-commit8.json'
                entries = json.loads(source.read_text())['results']
                row = dict(next(v for v in entries if v['scene'] == m.SCENES[scene] and v['case'] == case),
                           reused=True, source_sha256=sha(source))
            elif cohort == 'development' and arm in ['nominal', 'commit8']:
                source = HERE.parent / f'gru-causal-repairs/artifacts/fresh/{arm}/{scene}-{case}.json'
                row = dict(json.loads(source.read_text()), reused=True, source_sha256=sha(source))
            else:
                existing = OUT / (output + '.json')
                if existing.exists():
                    row = json.loads(existing.read_text())
                else:
                    tape = humans(cfg, p, scene, case)
                    if arm in ['nominal', 'commit8']:
                        row = h.run_episode(cfg, p, scene, case, tape, arm=arm, output=output)
                    else:
                        row = episode(cfg, p, scene, case, tape, arm, output)
            rows.append(row)
            save(f'{cohort}-{arm}.json', {'results': rows, 'seconds': time.perf_counter() - start})
            print(cohort, arm, len(rows), '/', cohort_spec['n'], scene, case, row['outcome'], flush=True)
    assert len(rows) == cohort_spec['n']
    verify()


def summarize(cohort):
    rows = {a: json.loads((HERE / f'{cohort}-{a}.json').read_text())['results'] for a in SPEC['arms']}
    assert all(len(v) == SPEC[cohort]['n'] for v in rows.values())
    overall = {a: stats.aggregate(v) for a, v in rows.items()}
    strongest = max((a for a in SPEC['arms'] if a != 'deadline'), key=lambda a: (overall[a]['success'], -overall[a]['contact']))
    pair = {f'deadline-vs-{a}': stats.pairs(rows['deadline'], rows[a]) for a in SPEC['arms'] if a != 'deadline'}
    left = {(v['scene'], v['case']): v for v in rows['deadline']}
    right = {(v['scene'], v['case']): v for v in rows[strongest]}
    both = [k for k in left if left[k]['success'] and right[k]['success']]
    time_ratio = np.mean([left[k]['duration'] for k in both]) / np.mean([right[k]['duration'] for k in both])
    own = overall['deadline']
    checks = {'beats_all_by_two': all(own['success'] >= overall[a]['success'] + 2 for a in overall if a != 'deadline'),
              'safety_nonregression': all(own['contact'] <= overall[a]['contact'] and own['collision'] <= overall[a]['collision'] for a in overall if a != 'deadline'),
              'time_nonregression': bool(time_ratio <= 1.10), 'mechanism_activates': sum(v['action_changes'] for v in rows['deadline']) > 0,
              'compute': own['decision_ms_episode_mean'] < 250}
    passed = all(checks.values())
    if cohort == 'confirmation':
        passed &= pair[f'deadline-vs-{strongest}']['delta_SR_case_bootstrap_95'][0] > 0
    result = {'overall': overall, 'pairs': pair, 'strongest_simple': strongest,
              'paired_time_ratio': float(time_ratio), 'gate': {'checks': checks, 'passed': bool(passed)},
              'verdict': 'METHOD_SIGNAL_FOUND' if passed else 'VERSION_NEGATIVE',
              'scientific_scope': 'Tests this candidate, not general impossibility; no paper PASS without novelty review and fresh confirmation.'}
    save(f'{cohort}-summary.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['freeze', 'controls', 'evaluate', 'summarize', 'verify'])
    parser.add_argument('--arm', choices=SPEC['arms'])
    parser.add_argument('--cohort', choices=['development', 'confirmation'], default='development')
    args = parser.parse_args()
    if args.command == 'evaluate':
        evaluate(args.cohort, args.arm)
    elif args.command == 'summarize':
        summarize(args.cohort)
    else:
        globals()[args.command]()
