"""Paired outcomes for an explicit policy-tail handoff candidate."""
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

spec = importlib.util.spec_from_file_location('shared_handoff_test', HERE.parent / 'progress-contract-v2/run.py')
x = importlib.util.module_from_spec(spec)
spec.loader.exec_module(x)
SPEC = json.loads((HERE / 'protocol.json').read_text())
x.HERE, x.OUT, x.SPEC, x.Controller = HERE, HERE / 'artifacts', SPEC, Controller
x.h.OUT = x.OUT
h, r, m = x.h, x.r, x.m


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze():
    files = [v for folder in ['crowd_nav', 'crowd_sim'] for v in (ROOT / folder).rglob('*.py')]
    files += list(HERE.glob('*.py')) + [HERE / 'protocol.json', r.OLD / 'artifacts/nominal.pth', r.OLD / 'parent-config.ini']
    files += [HERE.parent / f for f in ['progress-contract/controller.py', 'progress-contract-v2/controller.py',
        'progress-contract-v2/run.py', 'gru-causal-repairs/run.py', 'gru-causal-repairs/analyze.py',
        'gru-root-cause/run.py', 'gru-complete-actions/run.py', 'recovery-duration/run.py']]
    import pytorch_mppi.mppi as upstream
    assert sha(upstream.__file__) == sha('/workspace/research-deps/pytorch-mppi-20261012/src/pytorch_mppi/mppi.py')
    assert sha(r.OLD / 'artifacts/nominal.pth') == SPEC['parent_sha256']
    m.save_json(HERE / 'frozen-manifest.json', {'files': {str(v.relative_to(ROOT)): sha(v) for v in sorted(files)},
        'upstream_mppi_sha256': sha(upstream.__file__), 'upstream_commit': SPEC['upstream']['commit']})
    print('FROZEN', len(files), flush=True)


def controls():
    x.verify()
    cfg, p = h.setup()
    h.weight(p, 'nominal')
    cases = []
    for scene in range(6):
        tape = x.humans(cfg, p, scene, 121)
        env = r.environment(cfg, p, scene, 121, tape, None, 50)
        part = r.decompose(p, env)
        c = Controller('handoff', p)
        c.goal = torch.tensor(env.robot.get_goal_position(), dtype=torch.float64)
        c.humans = torch.tensor([[v.px, v.py, v.vx, v.vy, v.radius] for v in env.humans], dtype=torch.float64)
        c.radius = env.robot.radius
        state = torch.tensor([*env.robot.get_position(), 0., 0., -1.], dtype=torch.float64)
        predicted = c.dynamics(state.repeat(80, 1), c.raw)
        np.testing.assert_allclose(predicted[:, 2:4], part['commands'], atol=1e-15, rtol=0)
        np.testing.assert_allclose(c.clearance(predicted), part['clear'], atol=1e-12, rtol=0)
        legal = c.legal_state(state)
        np.testing.assert_array_equal([[v.px,v.py,v.vx,v.vy,v.radius] for v in legal.human_states], c.humans)
        hist, previous = p._history, p._last_action
        # First nominal-tail action must match the real current GRU action.
        _, _, _, commands, _ = c.tail(c.raw[0].repeat(16,1), 0, state)
        assert int(c.project(commands[:1])[0]) == part['best']
        assert p._history is hist and p._last_action is previous and p.env is env
        first = x.episode(cfg,p,scene,121,tape,'handoff',f'controls/first/{scene}-121')
        second = x.episode(cfg,p,scene,121,tape,'handoff',f'controls/second/{scene}-121')
        with np.load(x.OUT / f'controls/first/{scene}-121.npz') as a, np.load(x.OUT / f'controls/second/{scene}-121.npz') as b:
            np.testing.assert_array_equal(a['records'], b['records'])
            np.testing.assert_array_equal(a['candidates'], b['candidates'])
        cases.append(first)
        print('CONTROL',scene,first['outcome'],first['triggers'],flush=True)
    x.verify()
    m.save_json(HERE / 'controls.json', {'passed':True, 'grid':80, 'time_limit':50,
        'history_and_env_restored':True, 'nominal_first_action_matches':True,
        'deterministic_repeat':True, 'results':cases})


def evaluate(cohort, arm):
    x.verify()
    assert json.loads((HERE / 'controls.json').read_text())['passed']
    if cohort == 'confirmation':
        assert json.loads((HERE / 'development-summary.json').read_text())['gate']['passed']
    cfg,p = h.setup()
    h.weight(p,'nominal')
    rows=[]
    start=time.perf_counter()
    for scene in SPEC[cohort]['scenes']:
        for case in range(SPEC[cohort]['case_first'],SPEC[cohort]['case_last']+1):
            output=f'{cohort}/{arm}/{scene}-{case}'
            existing=x.OUT/(output+'.json')
            if existing.exists():
                row=json.loads(existing.read_text())
            else:
                tape=x.humans(cfg,p,scene,case)
                row=(h.run_episode(cfg,p,scene,case,tape,arm='nominal',output=output)
                     if arm=='nominal' else x.episode(cfg,p,scene,case,tape,arm,output))
            rows.append(row)
            m.save_json(HERE/f'{cohort}-{arm}.json',{'results':rows,'seconds':time.perf_counter()-start})
            print(cohort,arm,len(rows),'/',SPEC[cohort]['n'],scene,case,row['outcome'],flush=True)
    x.verify()


def summarize(cohort):
    rows={a:json.loads((HERE/f'{cohort}-{a}.json').read_text())['results'] for a in SPEC['arms']}
    assert all(len(v)==SPEC[cohort]['n'] for v in rows.values())
    overall={a:x.stats.aggregate(v) for a,v in rows.items()}
    pair={a:x.stats.pairs(rows['handoff'],rows[a]) for a in SPEC['arms'] if a!='handoff'}
    strongest=max(pair,key=lambda a:(overall[a]['success'],-overall[a]['contact']))
    base={ (v['scene'],v['case']):v for v in rows[strongest] }
    own=rows['handoff']
    both=[v for v in own if v['success'] and base[(v['scene'],v['case'])]['success']]
    ratio=float(np.mean([v['duration'] for v in both])/np.mean([base[(v['scene'],v['case'])]['duration'] for v in both]))
    activated=0
    for row in own:
        scene=m.SCENES.index(row['scene'])
        diagnostics=json.loads((x.OUT/f'{cohort}/handoff/{scene}-{row["case"]}-mechanism.json').read_text())
        activated+=sum(v.get('admission_changed_choice',False) for v in diagnostics)
    checks={'no_success_regression':all(overall['handoff']['success']>=overall[a]['success'] for a in pair),
            'safety':all(overall['handoff']['contact']<=overall[a]['contact'] and overall['handoff']['collision']<=overall[a]['collision'] for a in pair),
            'independent_task_gain':all(overall['handoff']['success']>=overall[a]['success']+2 for a in pair) or ratio<=.95,
            'mechanism':activated>0,'compute':overall['handoff']['decision_ms_episode_mean']<250}
    passed=all(checks.values())
    m.save_json(HERE/f'{cohort}-summary.json',{'overall':overall,'pairs':pair,'strongest_simple':strongest,
        'paired_time_ratio':ratio,'admission_changed_decisions':activated,
        'gate':{'checks':checks,'passed':passed},'verdict':'METHOD_SIGNAL_FOUND' if passed else 'VERSION_NEGATIVE',
        'scope':'Known rollout/shielding ingredients are not novelty. No paper PASS without meaningful mechanism and fresh confirmation.'})
    print('SUMMARY',json.dumps(checks),strongest,ratio,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['freeze','verify','controls','evaluate','summarize'])
    parser.add_argument('--arm',choices=SPEC['arms'])
    parser.add_argument('--cohort',choices=['development','confirmation'],default='development')
    args=parser.parse_args()
    if args.command=='verify':x.verify()
    elif args.command=='evaluate':evaluate(args.cohort,args.arm)
    elif args.command=='summarize':summarize(args.cohort)
    else:globals()[args.command]()
