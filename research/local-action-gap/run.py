"""Whole-case episodic residual controls with actual native navigation outcomes."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial import cKDTree
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


x = load('local_gap_loop', HERE.parent / 'progress-contract-v2/run.py')
x.Controller = load('local_gap_mppi', HERE.parent / 'progress-contract-v2/controller.py').Controller
h, r, m = x.h, x.r, x.m
SPEC = json.loads((HERE / 'protocol.json').read_text())
x.HERE, x.OUT, x.SPEC = HERE, HERE / 'artifacts', SPEC
h.OUT = x.OUT
BaseController = x.Controller


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name, data):
    m.save_json(HERE / name, data)


def freeze():
    files = [f for d in ['crowd_nav', 'crowd_sim'] for f in (ROOT / d).rglob('*.py')]
    files += list(HERE.glob('*.py')) + [HERE / 'protocol.json', HERE / 'PRIOR.md',
        r.OLD / 'parent-config.ini', r.OLD / 'artifacts/nominal.pth']
    files += [HERE.parent / name for name in ['progress-contract-v2/run.py',
        'progress-contract-v2/controller.py', 'progress-contract/controller.py',
        'gru-causal-repairs/run.py', 'gru-causal-repairs/analyze.py',
        'gru-complete-actions/run.py', 'gru-root-cause/run.py', 'recovery-duration/run.py',
        'progress-contract-v2/development-hold8.json', 'progress-contract-v2/development-nominal.json']]
    assert sha(r.OLD / 'artifacts/nominal.pth') == SPEC['parent_sha256']
    save('frozen-manifest.json', {'files': {str(f.relative_to(ROOT)): sha(f) for f in sorted(set(files))}})


def verify():
    for name, digest in json.loads((HERE / 'frozen-manifest.json').read_text())['files'].items():
        assert digest and sha(ROOT / name) == digest, name
    if (HERE / 'bank.json').exists():
        assert sha(x.OUT / 'bank.npz') == json.loads((HERE / 'bank.json').read_text())['sha256']


def features(p, windows):
    captured = []
    hook = p.value_head.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach().cpu().numpy()))
    try:
        with torch.no_grad():
            value = p.forward_candidate_values(torch.as_tensor(windows, device=p.device)).cpu().numpy()
    finally:
        hook.remove()
    f = np.concatenate(captured)
    assert f.shape == (80, 256)
    np.testing.assert_allclose(f @ p.value_head.weight.detach().cpu().numpy().T +
        p.value_head.bias.detach().cpu().numpy(), value[:, None], atol=1e-6, rtol=1e-5)
    return f.astype(np.float64)


def fit():
    verify()
    cfg, p = h.setup(); h.weight(p, 'nominal')
    paused = {h.q.key(v) for v in json.loads((h.OLD / 'statistics.json').read_text())['cases']
              if v['category'] == 'NO_CONTACT_FREE_RESCUE_IN_TESTED_SINGLE_OR_PREFIX_ACTIONS'}
    keys, centered, residuals, cases, meta = [], [], [], [], []
    inputs = {}
    for root in h.q.roots():
        if h.q.key(root) in paused:
            continue
        tape = h.old_tape(cfg, p, root)
        _, snap = r.frozen_root(cfg, p, root, tape)
        env = r.environment(cfg, p, m.SCENES.index(root['scene']), root['case'], tape, snap, 50)
        part = r.decompose(p, env)
        f = features(p, part['windows'])
        terminal, _ = p.candidate_terminal_outcomes(m.state(env), part['commands'], env)
        safe = part['safe'] & ~terminal
        indices = np.flatnonzero(safe)
        source = h.OLD / 'catalogue' / h.q.key(root)
        inputs[str((source / 'inputs.npz').relative_to(ROOT))] = sha(source / 'inputs.npz')
        ret = []
        for i in indices:
            path = source / f'action-{i:02d}-A_deploy.json'
            row = json.loads(path.read_text())
            inputs[str(path.relative_to(ROOT))] = sha(path)
            ret.append(row['result']['return'])
        delta = np.asarray(ret) - part['q'][indices]
        keys.extend(f[indices]); centered.extend((f-f[indices].mean(0))[indices])
        residuals.extend(np.stack([delta, delta-delta.mean()], 1))
        cases.extend([root['case']]*len(indices))
        meta.extend([{'root': root, 'action': int(i)} for i in indices])
    keys, centered, y, cases = map(np.asarray, (keys, centered, residuals, cases))
    fit_mask = cases % 2 == 0
    assert len(set(cases[fit_mask])) > 1 and set(cases).isdisjoint(range(300,320))
    radii, errors = [], []
    for column, f in enumerate([keys, centered]):
        distances = []
        for case in sorted(set(cases[fit_mask])):
            keep = fit_mask & (cases != case)
            assert keep.sum() >= 8
            distances.extend(cKDTree(f[keep]).query(f[fit_mask & (cases == case)], k=8)[0][:,-1])
        radius = float(np.quantile(distances, .95)); radii.append(radius)
        d, ix = cKDTree(f[fit_mask]).query(f[~fit_mask], k=8)
        w = 1/(d*d+.001); pred = (w*y[fit_mask,column][ix]).sum(1)/w.sum(1)
        covered = d[:,-1] <= radius
        errors.append({'radius': radius, 'heldout_coverage': float(covered.mean()),
            'heldout_residual_MAE': float(np.abs(pred-y[~fit_mask,column]).mean())})
    np.savez_compressed(x.OUT / 'bank.npz', absolute=keys[fit_mask], relative=centered[fit_mask],
        target=y[fit_mask], cases=cases[fit_mask], radius=np.asarray(radii))
    save('bank.json', {'sha256': sha(x.OUT / 'bank.npz'), 'fit_rows': int(fit_mask.sum()),
        'heldout_rows': int((~fit_mask).sum()), 'fit_cases': sorted(map(int,set(cases[fit_mask]))),
        'heldout_cases': sorted(map(int,set(cases[~fit_mask]))), 'diagnostics': errors,
        'input_sha256': inputs, 'samples': meta, 'labels': 'Existing fixed-world deployment continuation returns; not historical training labels.'})
    print('BANK', int(fit_mask.sum()), errors, flush=True)


class Controller(BaseController):
    def __init__(self, arm, policy):
        super().__init__('hold8', policy)
        self.memory_arm = arm
        self.memory = np.load(x.OUT / 'bank.npz')
        self.trees = [cKDTree(self.memory[name]) for name in ['absolute', 'relative']]

    def corrections(self, part):
        f = features(self.policy, part['windows'])
        terminal, _ = self.policy.candidate_terminal_outcomes(m.state(self.policy.env), part['commands'], self.policy.env)
        allowed = part['safe'] & ~terminal
        if not allowed.any():
            return np.zeros((2,80)), np.zeros((2,80), bool)
        targets = [f, f-f[allowed].mean(0)]
        corrections, coverage = [], []
        for column, query in enumerate(targets):
            d, ix = self.trees[column].query(query, k=8)
            w = 1/(d*d+.001)
            prediction = (w*self.memory['target'][:,column][ix]).sum(1)/w.sum(1)
            valid = (d[:,-1] <= self.memory['radius'][column]) & allowed
            corrections.append(np.where(valid, prediction, 0.))
            coverage.append(valid)
        return np.asarray(corrections), np.asarray(coverage)

    def select(self, part, *args, **kwargs):
        corr, covered = self.corrections(part)
        which = ['absolute','relative'].index(self.memory_arm)
        scores = part['scores'].copy() + corr[which]
        if part['safe'].any():
            scores[~part['safe']] = -1e9
        query = dict(part, scores=scores, best=int(scores.argmax()))
        index, diag = super().select(query, *args, **kwargs)
        diag.update(memory_changed=int(query['best'] != part['best']),
                    executed_memory_change=int(not diag['active'] and index != part['best']),
                    covered=int(covered[which].sum()), correction_range=float(np.ptp(corr[which])))
        return index, diag


x.Controller = Controller


def controls():
    verify()
    cfg,p = h.setup(); h.weight(p,'nominal')
    for scene in range(6):
        tape = x.humans(cfg,p,scene,301)
        env = r.environment(cfg,p,scene,301,tape,None,50)
        part = r.decompose(p,env)
        c = Controller('relative',p)
        first, cover = c.corrections(part); second, cover2 = c.corrections(part)
        np.testing.assert_array_equal(first,second); np.testing.assert_array_equal(cover,cover2)
        assert np.isfinite(first).all() and np.all(first[:,~part['safe']]==0)
        a = x.episode(cfg,p,scene,301,tape,'relative',f'controls/a/{scene}-301')
        b = x.episode(cfg,p,scene,301,tape,'relative',f'controls/b/{scene}-301')
        for key in ['records','candidates']:
            np.testing.assert_array_equal(np.load(x.OUT/f'controls/a/{scene}-301.npz')[key],
                                          np.load(x.OUT/f'controls/b/{scene}-301.npz')[key])
        print('CONTROL',scene,a['outcome'],flush=True)
    verify()
    save('controls.json', {'passed':True,'deterministic_repeat':True,'grid':80,'time_limit':50,
        'same_frozen_weights':True,'unsafe_and_terminal_correction_zero':True})


def evaluate(arm,cohort):
    verify(); assert json.loads((HERE/'controls.json').read_text())['passed']
    if cohort=='confirmation':
        assert json.loads((HERE/'development-summary.json').read_text())['gate']['passed']
    cfg,p = h.setup(); h.weight(p,'nominal')
    rows=[]; started=time.perf_counter()
    for scene in SPEC[cohort]['scenes']:
        for case in range(SPEC[cohort]['case_first'],SPEC[cohort]['case_last']+1):
            if cohort=='development' and arm in ['nominal','hold8']:
                source=HERE.parent/f'progress-contract-v2/development-{arm}.json'
                row=dict(next(v for v in json.loads(source.read_text())['results'] if v['scene']==m.SCENES[scene] and v['case']==case),reused=True,source_sha256=sha(source))
            elif arm=='nominal':
                row=h.run_episode(cfg,p,scene,case,x.humans(cfg,p,scene,case),arm,output=f'{cohort}/{arm}/{scene}-{case}')
            else:
                original=x.Controller
                if arm=='hold8': x.Controller=BaseController
                try:
                    row=x.episode(cfg,p,scene,case,x.humans(cfg,p,scene,case),arm,f'{cohort}/{arm}/{scene}-{case}')
                finally:
                    x.Controller=original
            rows.append(row); save(f'{cohort}-{arm}.json',{'results':rows,'seconds':time.perf_counter()-started})
            print(cohort,arm,len(rows),row['outcome'],flush=True)
    verify()


def summarize(cohort):
    rows={a:json.loads((HERE/f'{cohort}-{a}.json').read_text())['results'] for a in SPEC['arms']}
    assert all(len(v)==SPEC[cohort]['n'] for v in rows.values())
    overall={a:x.stats.aggregate(v) for a,v in rows.items()}
    pairs={a:x.stats.pairs(rows['relative'],rows[a]) for a in SPEC['arms'] if a!='relative'}
    strongest=max(pairs,key=lambda a:(overall[a]['success'],-overall[a]['contact']))
    base={(v['scene'],v['case']):v for v in rows[strongest]}
    both=[v for v in rows['relative'] if v['success'] and base[(v['scene'],v['case'])]['success']]
    ratio=float(np.mean([v['duration'] for v in both])/np.mean([base[(v['scene'],v['case'])]['duration'] for v in both]))
    activated=0
    for v in rows['relative']:
        path=x.OUT/f'{cohort}/relative/{m.SCENES.index(v["scene"])}-{v["case"]}-mechanism.json'
        activated+=sum(d['executed_memory_change'] for d in json.loads(path.read_text()))
    checks={'no_success_regression':all(overall['relative']['success']>=overall[a]['success'] for a in pairs),
        'safety':all(overall['relative']['contact']<=overall[a]['contact'] and overall['relative']['collision']<=overall[a]['collision'] for a in pairs),
        'independent_task_gain':all(overall['relative']['success']>=overall[a]['success']+2 for a in pairs) or ratio<=.95,
        'mechanism':activated>0,'compute':overall['relative']['decision_ms_episode_mean']<250}
    save(f'{cohort}-summary.json',{'overall':overall,'pairs':pairs,'strongest_simple':strongest,
        'paired_time_ratio':ratio,'executed_memory_changes':activated,'gate':{'checks':checks,'passed':all(checks.values())},
        'verdict':'METHOD_SIGNAL_FOUND' if all(checks.values()) else 'VERSION_NEGATIVE',
        'novelty':'UNVERIFIED; no independent algorithmic contribution established by this experiment alone.'})
    print('SUMMARY',checks,ratio,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('command',choices=['freeze','verify','fit','controls','evaluate','summarize'])
    parser.add_argument('--arm',choices=SPEC['arms']); parser.add_argument('--cohort',default='development',choices=['development','confirmation'])
    args=parser.parse_args()
    if args.command=='evaluate': evaluate(args.arm,args.cohort)
    elif args.command=='summarize': summarize(args.cohort)
    else: globals()[args.command]()
