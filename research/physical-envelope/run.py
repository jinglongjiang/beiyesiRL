"""Executable reachability caps on a frozen critic; actual native closed loops."""
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
sys.path.insert(0, str(ROOT))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


h = load('envelope_parent', HERE.parent / 'gru-causal-repairs/run.py')
stats = load('envelope_statistics', HERE.parent / 'gru-causal-repairs/analyze.py')
r, m = h.r, h.m
SPEC = json.loads((HERE / 'protocol.json').read_text())
OUT = HERE / 'artifacts'
h.OUT = OUT


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name, data):
    m.save_json(HERE / name, data)


def freeze():
    paths = [p for folder in ['crowd_nav', 'crowd_sim'] for p in (ROOT / folder).rglob('*.py')]
    paths += list(HERE.glob('*.py')) + [HERE / 'protocol.json', r.OLD / 'parent-config.ini',
        r.OLD / 'artifacts/nominal.pth', HERE.parent / 'gru-causal-repairs/run.py',
        HERE.parent / 'gru-root-cause/run.py', HERE.parent / 'recovery-duration/run.py',
        HERE.parent / 'gru-causal-repairs/analyze.py']
    paths += [HERE.parent / f'policy-handoff/development-{a}.json' for a in ['nominal', 'hold8']]
    assert sha(r.OLD / 'artifacts/nominal.pth') == SPEC['parent_sha256']
    save('frozen-manifest.json', {'files': {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(paths))}})


def verify():
    for path, digest in json.loads((HERE / 'frozen-manifest.json').read_text())['files'].items():
        assert digest and sha(ROOT / path) == digest, path


def caps(p, env, part):
    commands = np.asarray(part['commands'], np.float64)
    positions = np.asarray(env.robot.get_position())[None] + .25 * commands
    offset = np.asarray(env.robot.get_goal_position())[None] - positions
    distance = np.linalg.norm(offset, axis=1)
    direction = offset / np.maximum(distance[:, None], 1e-15)
    raw = np.asarray(p.action_space, np.float64)
    support = (direction @ raw.T).max(1)
    velocity = (commands * direction).sum(1)
    n = np.arange(1, 201, dtype=np.float64)
    alpha = p.test_action_smoothing
    projected = .25 * (support[:, None] * n + (velocity-support)[:, None] *
                        alpha * (1-alpha**n) / (1-alpha))
    feasible = projected >= np.maximum(distance-env.robot.radius, 0)[:, None]
    earliest = np.where(feasible.any(1), feasible.argmax(1) + 1, 201)
    max_speed = np.linalg.norm(raw, axis=1).max()
    euclidean = np.maximum(1, np.ceil(np.maximum(distance-env.robot.radius, 0) / (.25*max_speed))).astype(int)
    remaining = max(0, 200 - (round(env.global_time/.25) + 1))
    upper = np.where(earliest <= remaining, p.success_reward * p.gamma ** (earliest-1), 0.)
    simple = np.where(euclidean <= remaining, p.success_reward * p.gamma ** (euclidean-1), 0.)
    assert (upper <= simple + 1e-12).all()
    return upper, simple, earliest, positions


def selection(p, env, part, arm):
    upper, simple, earliest, positions = caps(p, env, part)
    outputs = {}
    state = m.state(env)
    for name, values in [('envelope', np.minimum(part['value'], upper)),
                         ('distance-cap', np.minimum(part['value'], simple)), ('free', simple)]:
        with torch.no_grad():
            q = p.candidate_action_values(state, part['commands'],
                    torch.tensor(part['raw_reward'], device=p.device, dtype=torch.float32),
                    torch.tensor(values, device=p.device, dtype=torch.float32))
            scores = q.cpu().numpy() - part['risk']
        if part['safe'].any():
            scores[~part['safe']] = -1e9
        scores[0] -= .001
        outputs[name] = (int(scores.argmax()), scores, values)
    return outputs[arm][0], outputs, upper, simple, earliest


def humans(cfg, p, scene, case):
    path = HERE.parent / f'policy-handoff/artifacts/tapes/{scene}-{case}.npz'
    if path.exists():
        return np.load(path)['humans'].copy()
    return h.tape(cfg, p, scene, case)


def episode(cfg, p, scene, case, tape, arm, output):
    env = r.environment(cfg, p, scene, case, tape, None, 50)
    assert len(p.action_space) == 80 and env.time_limit == 50 and not env.robot.visible
    assert env.progress_reward == env.time_penalty == env.stand_penalty == 0
    assert env.discomfort_penalty_factor >= 0 and env.timeout_penalty <= 0 and env.collision_penalty <= 0
    rows, arrays, times = [], [], []
    active = changed = distinct = fallback = 0
    for _ in range(203):
        step = round(env.global_time / .25)
        started = time.perf_counter()
        part = r.decompose(p, env)
        index, outputs, upper, simple, earliest = selection(p, env, part, arm)
        times.append(time.perf_counter() - started)
        assert part['safe'][index] or not part['safe'].any()
        active += int((part['value'] > upper).any())
        changed += int(index != part['best'])
        distinct += int(outputs['envelope'][0] != outputs['distance-cap'][0])
        fallback += int(not part['safe'].any())
        position = np.asarray(env.robot.get_position())
        action = part['commands'][index]
        before = m.before_geometry(env)
        m.commit_action(p, part['token'], action, index)
        _, reward, term, trunc, info = env.step(action)
        np.testing.assert_allclose([[v.px,v.py,v.vx,v.vy,v.radius] for v in env.humans], tape[step+1], rtol=0, atol=1e-8)
        np.testing.assert_allclose(env.robot.get_position(), position+.25*np.asarray(action), rtol=0, atol=1e-12)
        clear = m.geo_step(env, before)
        rows.append([step,*position,*action,index,part['best'],reward,clear,part['progress'][index],part['progress'][part['best']]])
        arrays.append(np.stack([part['value'],part['q'],part['scores'],part['progress'],part['clear'],
            part['safe'],upper,simple,earliest,outputs[arm][1]]))
        if term or trunc:
            break
    assert term or trunc
    a = np.asarray(rows, np.float64)
    result = dict(arm=arm,scene=m.SCENES[scene],case=case,outcome=info['event'],
        success=int(info['event']=='reach_goal'),collision=int(info['event']=='collision'),timeout=int(info['event']=='timeout'),
        contact=int((a[:,8]<0).any()),min_clearance=float(a[:,8].min()),duration=len(a)*.25,steps=len(a),
        path=float(np.linalg.norm(a[:,3:5],axis=1).sum()*.25),
        final_distance=float(np.linalg.norm(np.asarray(env.robot.get_position())-env.robot.get_goal_position())),
        **{'return':float(a[:,7] @ (.99**np.arange(len(a))))},
        clipped_steps=active,action_changes=changed,envelope_distance_disagreements=distinct,fallback=fallback,
        decision_ms_mean=float(np.mean(times)*1000),decision_ms_p95=float(np.quantile(times,.95)*1000))
    target = OUT/output
    target.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(target.with_suffix('.npz'), records=a,candidates=np.asarray(arrays,np.float32))
    m.save_json(target.with_suffix('.json'),result)
    return result


def controls():
    verify()
    cfg,p = h.setup()
    h.weight(p,'nominal')
    checks=[]
    for scene in range(6):
        tape=humans(cfg,p,scene,121)
        env=r.environment(cfg,p,scene,121,tape,None,50)
        part=r.decompose(p,env)
        upper,simple,earliest,positions=caps(p,env,part)
        # The support recurrence agrees exactly with repeatedly choosing its maximizing raw action.
        for i in [0,39,79]:
            direction=np.asarray(env.robot.get_goal_position())-positions[i]
            direction/=np.linalg.norm(direction)
            raw=np.asarray(p.action_space)
            best=raw[np.argmax(raw@direction)]
            v=np.asarray(part['commands'][i]); delta=np.zeros(2)
            for n in range(1,17):
                v=.3*v+.7*best; delta+=.25*v
                formula=.25*(n*np.dot(best,direction)+(np.dot(part['commands'][i],direction)-np.dot(best,direction))*.3*(1-.3**n)/.7)
                np.testing.assert_allclose(delta@direction,formula,rtol=0,atol=1e-12)
        first=episode(cfg,p,scene,121,tape,'envelope',f'controls/first/{scene}-121')
        second=episode(cfg,p,scene,121,tape,'envelope',f'controls/second/{scene}-121')
        for key in ['records','candidates']:
            np.testing.assert_array_equal(np.load(OUT/f'controls/first/{scene}-121.npz')[key],np.load(OUT/f'controls/second/{scene}-121.npz')[key])
        checks.append(first)
        print('CONTROL',scene,first['outcome'],flush=True)
    verify()
    save('controls.json',dict(passed=True,grid=80,time_limit=50,repeat_exact=True,support_recurrence=True,results=checks))


def evaluate(arm,cohort):
    verify()
    assert json.loads((HERE/'controls.json').read_text())['passed']
    if cohort=='confirmation':
        assert json.loads((HERE/'development-summary.json').read_text())['gate']['passed']
    cfg,p=h.setup(); h.weight(p,'nominal')
    rows=[]; started=time.perf_counter(); spec=SPEC[cohort]
    for scene in spec['scenes']:
        for case in range(spec['case_first'],spec['case_last']+1):
            if cohort=='development' and arm in ['nominal','hold8']:
                source=HERE.parent/f'policy-handoff/development-{arm}.json'
                data=json.loads(source.read_text())['results']
                row=dict(next(v for v in data if v['scene']==m.SCENES[scene] and v['case']==case),reused=True,source_sha256=sha(source))
            else:
                row=episode(cfg,p,scene,case,humans(cfg,p,scene,case),arm,f'{cohort}/{arm}/{scene}-{case}')
            rows.append(row)
            save(f'{cohort}-{arm}.json',dict(results=rows,seconds=time.perf_counter()-started))
            print(cohort,arm,len(rows),'/',spec['n'],scene,case,row['outcome'],flush=True)
    assert len(rows)==spec['n']; verify()


def summarize(cohort):
    rows={a:json.loads((HERE/f'{cohort}-{a}.json').read_text())['results'] for a in SPEC['arms']}
    assert all(len(v)==SPEC[cohort]['n'] for v in rows.values())
    overall={a:stats.aggregate(v) for a,v in rows.items()}
    simple=max((a for a in rows if a!='envelope'),key=lambda a:(overall[a]['success'],-overall[a]['contact']))
    pair={a:stats.pairs(rows['envelope'],rows[a]) for a in rows if a!='envelope'}
    left={(v['scene'],v['case']):v for v in rows['envelope']}; right={(v['scene'],v['case']):v for v in rows[simple]}
    both=[k for k in left if left[k]['success'] and right[k]['success']]
    ratio=float(np.mean([left[k]['duration'] for k in both])/np.mean([right[k]['duration'] for k in both]))
    ids=sorted({k[1] for k in both})
    delta=np.array([sum(left[k]['duration']-right[k]['duration'] for k in both if k[1]==c) for c in ids])
    counts=np.array([sum(k[1]==c for k in both) for c in ids])
    draw=np.random.default_rng(12012).integers(len(ids),size=(4000,len(ids)))
    time_ci=np.quantile(delta[draw].sum(1)/counts[draw].sum(1),[.025,.975]).tolist()
    own=overall['envelope']
    checks=dict(outcome_nonregression=all(own['success']>=v['success'] for v in overall.values()),
        safety=all(own['collision']<=v['collision'] and own['contact']<=v['contact'] for v in overall.values()),
        independent_gain=all(own['success']>=v['success']+2 for a,v in overall.items() if a!='envelope') or ratio<=.95,
        mechanism=sum(v['envelope_distance_disagreements'] for v in rows['envelope'])>0,
        compute=own['decision_ms_episode_mean']<50)
    passed=all(checks.values())
    if cohort=='confirmation':
        passed &= bool(pair[simple]['delta_SR_case_bootstrap_95'][0]>0 or (ratio<=.95 and time_ci[1]<0))
    save(f'{cohort}-summary.json',dict(overall=overall,pairs=pair,strongest_simple=simple,
        paired_time_ratio=ratio,paired_time_delta_bootstrap_95=time_ci,gate=dict(checks=checks,passed=bool(passed)),
        verdict='METHOD_SIGNAL_FOUND' if passed else 'VERSION_NEGATIVE'))
    print(json.dumps(json.loads((HERE/f'{cohort}-summary.json').read_text()),indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['freeze','verify','controls','evaluate','summarize'])
    parser.add_argument('--arm',choices=SPEC['arms'])
    parser.add_argument('--cohort',choices=['development','confirmation'],default='development')
    args=parser.parse_args()
    if args.command=='evaluate': evaluate(args.arm,args.cohort)
    elif args.command=='summarize': summarize(args.cohort)
    else: globals()[args.command]()
