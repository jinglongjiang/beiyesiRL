"""Matched residual decision objectives and complete native navigation."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location('censored_parent_helpers', HERE.parent/'local-action-gap/run.py')
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
x, h, r, m = old.x, old.h, old.r, old.m
SPEC = json.loads((HERE/'protocol.json').read_text())
OUT = HERE/'artifacts'
x.HERE, x.OUT, x.SPEC, h.OUT = HERE, OUT, SPEC, OUT
ARMS = ['pair', 'set', 'regret']


def save(name, data):
    m.save_json(HERE/name, data)


def freeze():
    OUT.mkdir(exist_ok=True)
    paths = [p for d in ['crowd_nav','crowd_sim'] for p in (ROOT/d).rglob('*.py')]
    paths += [HERE/'run.py', HERE/'protocol.json', HERE/'PRIOR.md',
              r.OLD/'artifacts/nominal.pth', r.OLD/'parent-config.ini']
    paths += [HERE.parent/name for name in ['local-action-gap/run.py',
        'local-action-gap/protocol.json', 'progress-contract-v2/run.py',
        'progress-contract-v2/controller.py', 'progress-contract/controller.py',
        'gru-causal-repairs/run.py', 'gru-causal-repairs/analyze.py',
        'gru-root-cause/run.py', 'gru-complete-actions/run.py', 'recovery-duration/run.py']]
    assert old.sha(r.OLD/'artifacts/nominal.pth') == SPEC['parent_sha256']
    save('frozen-manifest.json', {'files': {str(p.relative_to(ROOT)):old.sha(p) for p in sorted(set(paths))}})


def verify():
    for name, digest in json.loads((HERE/'frozen-manifest.json').read_text())['files'].items():
        assert digest and old.sha(ROOT/name) == digest, name
    if (HERE/'training.json').exists():
        training = json.loads((HERE/'training.json').read_text())
        assert old.sha(OUT/'normalization.npz') == training['normalization_sha256']
        for arm, entry in training['arms'].items():
            assert old.sha(OUT/f'{arm}.pth') == entry['sha256']


class Residual(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(512,64), nn.SiLU(), nn.Linear(64,1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features):
        return .1*torch.tanh(self.net(features).squeeze(-1))


def objective(arm, score, good, returns, base):
    bad = ~good
    if not good.any() or not bad.any():
        return score.sum()*0
    margin = torch.clamp(returns[good].max()-returns[bad], min=0., max=.05)
    if arm == 'pair':
        return F.softplus((margin[:,None]+score[bad,None]-score[good][None,:])/.05).mean()
    if arm == 'set':
        target = torch.softmax(base[good]/.05, dim=0).detach()
        logp = F.log_softmax(score/.05, dim=0)[good]
        return (target*(target.log()-logp)).sum()
    assert arm == 'regret'
    return F.relu((score[bad]+margin).max()-score[good].max())/.05


def data(cfg, p):
    paused = {h.q.key(v) for v in json.loads((h.OLD/'statistics.json').read_text())['cases']
              if v['category']=='NO_CONTACT_FREE_RESCUE_IN_TESTED_SINGLE_OR_PREFIX_ACTIONS'}
    rows, manifest = [], {}
    for root in h.q.roots():
        if h.q.key(root) in paused:
            continue
        tape = h.old_tape(cfg,p,root)
        _, snap = r.frozen_root(cfg,p,root,tape)
        env = r.environment(cfg,p,m.SCENES.index(root['scene']),root['case'],tape,snap,50)
        part = r.decompose(p,env)
        f = old.features(p,part['windows'])
        terminal,_ = p.candidate_terminal_outcomes(m.state(env),part['commands'],env)
        ix = np.flatnonzero(part['safe'] & ~terminal)
        assert len(ix)>0
        center = f[ix].mean(0)
        features = np.concatenate([np.broadcast_to(center, (len(ix),256)), f[ix]-center],1)
        outcomes, returns = [], []
        source = h.OLD/'catalogue'/h.q.key(root)
        for i in ix:
            path = source/f'action-{i:02d}-A_deploy.json'
            item = json.loads(path.read_text())['result']
            manifest[str(path.relative_to(ROOT))] = old.sha(path)
            outcomes.append(bool(item['success'] and not item['contact']))
            returns.append(item['return'])
        rows.append({'root':root, 'features':features, 'base':part['scores'][ix],
                     'good':np.asarray(outcomes), 'returns':np.asarray(returns), 'actions':ix})
    assert len(rows)==35 and all(not 300<=v['root']['case']<380 for v in rows)
    save('data.json', {'input_sha256':manifest, 'roots':[{
        k:(v.tolist() if isinstance(v,np.ndarray) else v)
        for k,v in row.items() if k!='features'} for row in rows]})
    return rows


def fit():
    verify()
    cfg,p=h.setup(); h.weight(p,'nominal')
    rows=data(cfg,p)
    fit_rows=[v for v in rows if v['root']['case']%2==0]
    raw=np.concatenate([v['features'] for v in fit_rows])
    center, scale=raw.mean(0), np.maximum(raw.std(0), .001)
    np.savez_compressed(OUT/'normalization.npz', center=center, scale=scale)
    for row in rows:
        for key in ['base','returns','good']:
            row[key]=torch.as_tensor(row[key],device=p.device)
        row['features']=torch.as_tensor((row['features']-center)/scale,dtype=torch.float32,device=p.device)
    torch.manual_seed(SPEC['model']['initial_seed'])
    initial=Residual().to(p.device).state_dict()
    results={}
    for arm in ARMS:
        model=Residual().to(p.device); model.load_state_dict(copy.deepcopy(initial))
        optimizer=torch.optim.Adam(model.parameters(),lr=SPEC['fit']['lr'])
        start=time.perf_counter(); curve=[]
        for step in range(1000):
            losses=[]
            for row in fit_rows:
                delta=model(row['features'])
                score=row['base']+delta
                loss=objective(arm,score,row['good'],row['returns'],row['base'])
                losses.append(loss+.01*(delta/.1).square().mean())
            loss=torch.stack(losses).mean()
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            norm=nn.utils.clip_grad_norm_(model.parameters(),10.)
            assert torch.isfinite(norm)
            optimizer.step()
            if (step+1)%100==0:
                curve.append({'update':step+1,'loss':float(loss)})
                print('FIT',arm,step+1,float(loss),flush=True)
        model.eval(); diagnostics=[]
        with torch.no_grad():
            for row in rows:
                delta=model(row['features']); score=row['base']+delta
                diagnostics.append({'root':row['root'],'fit':row['root']['case']%2==0,
                    'contrast':bool(row['good'].any() and (~row['good']).any()),
                    'selected_safe_success_label':bool(row['good'][score.argmax()]),
                    'old_selected_safe_success_label':bool(row['good'][row['base'].argmax()]),
                    'changed':bool(score.argmax()!=row['base'].argmax()),
                    'residual_range':float(delta.max()-delta.min())})
        torch.save(model.cpu().state_dict(),OUT/f'{arm}.pth')
        results[arm]={'sha256':old.sha(OUT/f'{arm}.pth'),'seconds':time.perf_counter()-start,
                      'updates':1000,'curve':curve,'diagnostics':diagnostics,
                      'parameters':sum(v.numel() for v in model.parameters())}
    save('training.json',{'arms':results,'normalization_sha256':old.sha(OUT/'normalization.npz'),
        'fit_roots':len(fit_rows),'heldout_roots':len(rows)-len(fit_rows),
        'fit_case_ids':sorted({v['root']['case'] for v in fit_rows})})
    verify()


class Controller(old.BaseController):
    def __init__(self, arm, policy):
        super().__init__('hold8',policy)
        self.arm=arm
        self.models={}
        for name in ARMS:
            model=Residual().to(policy.device)
            model.load_state_dict(torch.load(OUT/f'{name}.pth',map_location=policy.device))
            self.models[name]=model.eval()
        normalization=np.load(OUT/'normalization.npz')
        self.center,self.scale=normalization['center'],normalization['scale']

    def corrections(self,part):
        f=old.features(self.policy,part['windows'])
        terminal,_=self.policy.candidate_terminal_outcomes(m.state(self.policy.env),part['commands'],self.policy.env)
        allowed=part['safe'] & ~terminal
        if not allowed.any():
            return {arm:np.zeros(80) for arm in ARMS}
        center=f[allowed].mean(0)
        feat=np.concatenate([np.broadcast_to(center,(80,256)),f-center],1)
        tensor=torch.as_tensor((feat-self.center)/self.scale,dtype=torch.float32,device=self.policy.device)
        with torch.no_grad():
            return {arm:np.where(allowed,model(tensor).cpu().numpy(),0.) for arm,model in self.models.items()}

    def select(self,part,*args,**kwargs):
        corr=self.corrections(part)[self.arm]
        score=part['scores'].copy()+corr
        if part['safe'].any(): score[~part['safe']]=-1e9
        query=dict(part,scores=score,best=int(score.argmax()))
        index,diag=super().select(query,*args,**kwargs)
        diag.update(residual_changed=int(query['best']!=part['best']),
            executed_residual_change=int(not diag['active'] and index!=part['best']),
            correction_range=float(np.ptp(corr)))
        return index,diag


x.Controller=Controller


def controls():
    verify()
    for arm in ARMS:
        score=torch.tensor([.01,.02,.06],requires_grad=True)
        loss=objective(arm,score,torch.tensor([True,True,False]),torch.tensor([.8,.7,-.2]),score.detach())
        loss.backward()
        assert score.grad[2]>0 and score.grad[:2].sum()<0
        if arm=='regret': assert score.grad[0]==0 and score.grad[1]<0
        loss=objective(arm,score,torch.zeros(3,dtype=torch.bool),score.detach(),score.detach())
        assert loss==0
    cfg,p=h.setup();h.weight(p,'nominal')
    zero=Residual().to(p.device)
    assert torch.count_nonzero(zero(torch.randn(80,512,device=p.device)))==0
    smoke=[]
    for scene in range(6):
        tape=x.humans(cfg,p,scene,121)
        env=r.environment(cfg,p,scene,121,tape,None,50)
        part=r.decompose(p,env)
        corr=Controller('regret',p).corrections(part)
        for arm in ARMS:
            assert np.isfinite(corr[arm]).all() and (corr[arm][~part['safe']]==0).all()
        row=x.episode(cfg,p,scene,121,tape,'regret',f'controls/{scene}-121')
        if scene==1:
            x.episode(cfg,p,scene,121,tape,'regret','controls/repeat-1-121')
            for key in ['records','candidates']:
                np.testing.assert_array_equal(np.load(OUT/'controls/1-121.npz')[key],np.load(OUT/'controls/repeat-1-121.npz')[key])
        smoke.append(row); print('CONTROL',scene,row['outcome'],flush=True)
    verify()
    save('controls.json',{'passed':True,'grid':80,'time_limit':50,'exact_duplicate':True,
        'zero_output_initialization':True,'synthetic_gradients':True,'smoke':smoke})


def evaluate(arm,cohort):
    verify();assert json.loads((HERE/'controls.json').read_text())['passed']
    if cohort=='confirmation': assert json.loads((HERE/'development-summary.json').read_text())['gate']['passed']
    cfg,p=h.setup();h.weight(p,'nominal');rows=[];start=time.perf_counter()
    for scene in SPEC[cohort]['scenes']:
        for case in range(SPEC[cohort]['case_first'],SPEC[cohort]['case_last']+1):
            if cohort=='development' and arm in ['nominal','hold8']:
                source=HERE.parent/f'progress-contract-v2/development-{arm}.json'
                row=dict(next(v for v in json.loads(source.read_text())['results'] if v['scene']==m.SCENES[scene] and v['case']==case),reused=True,source_sha256=old.sha(source))
            elif arm=='nominal':
                row=h.run_episode(cfg,p,scene,case,x.humans(cfg,p,scene,case),arm,output=f'{cohort}/{arm}/{scene}-{case}')
            elif arm=='hold8':
                x.Controller=old.BaseController
                try: row=x.episode(cfg,p,scene,case,x.humans(cfg,p,scene,case),arm,f'{cohort}/{arm}/{scene}-{case}')
                finally: x.Controller=Controller
            else:
                row=x.episode(cfg,p,scene,case,x.humans(cfg,p,scene,case),arm,f'{cohort}/{arm}/{scene}-{case}')
            rows.append(row);save(f'{cohort}-{arm}.json',{'results':rows,'seconds':time.perf_counter()-start})
            print(cohort,arm,len(rows),row['outcome'],flush=True)
    verify()


def summarize(cohort):
    rows={a:json.loads((HERE/f'{cohort}-{a}.json').read_text())['results'] for a in SPEC['arms']}
    assert all(len(v)==SPEC[cohort]['n'] for v in rows.values())
    overall={a:x.stats.aggregate(v) for a,v in rows.items()}
    pairs={a:x.stats.pairs(rows['regret'],rows[a]) for a in SPEC['arms'] if a!='regret'}
    ratios={}
    for arm in pairs:
        base={(v['scene'],v['case']):v for v in rows[arm]}
        both=[v for v in rows['regret'] if v['success'] and base[(v['scene'],v['case'])]['success']]
        ratios[arm]=float(np.mean([v['duration'] for v in both])/np.mean([base[(v['scene'],v['case'])]['duration'] for v in both]))
    activated=0
    for v in rows['regret']:
        path=OUT/f'{cohort}/regret/{m.SCENES.index(v["scene"])}-{v["case"]}-mechanism.json'
        activated+=sum(d['executed_residual_change'] for d in json.loads(path.read_text()))
    own=overall['regret']
    checks={'success_nonregression':all(own['success']>=overall[a]['success'] for a in pairs),
        'safety':all(own['contact']<=overall[a]['contact'] and own['collision']<=overall[a]['collision'] for a in pairs),
        'independent_task_gain':all(own['success']>=overall[a]['success']+2 for a in pairs) or all(v<=.95 for v in ratios.values()),
        'activation':activated>0,'compute':own['decision_ms_episode_mean']<250}
    if cohort=='confirmation':
        rng=np.random.default_rng(121012)
        ids=sorted({v['case'] for v in rows['regret']})
        draws=rng.integers(len(ids),size=(4000,len(ids)))
        time_intervals={}
        for arm in pairs:
            base={(v['scene'],v['case']):v for v in rows[arm]}
            counts=np.zeros(len(ids));sums=np.zeros(len(ids))
            for v in rows['regret']:
                other=base[(v['scene'],v['case'])]
                if v['success'] and other['success']:
                    j=ids.index(v['case']);counts[j]+=1;sums[j]+=v['duration']-other['duration']
            denominator=counts[draws].sum(1)
            assert (denominator>0).all()
            ci=np.quantile(sums[draws].sum(1)/denominator,[.025,.975]).tolist()
            time_intervals[arm]=ci
        checks['fresh_intervals']=all(pairs[a]['delta_SR_case_bootstrap_95'][0]>0 or time_intervals[a][1]<0 for a in pairs)
        save('confirmation-time-intervals.json',time_intervals)
    save(f'{cohort}-summary.json',{'overall':overall,'pairs':pairs,'paired_success_time_ratios':ratios,
        'executed_residual_changes':activated,'gate':{'checks':checks,'passed':all(checks.values())},
        'verdict':'METHOD_SIGNAL_FOUND' if all(checks.values()) else 'VERSION_NEGATIVE',
        'novelty':'UNVERIFIED: structured ranking loss alone is not a contribution.'})
    print('SUMMARY',checks,ratios,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['freeze','verify','fit','controls','evaluate','summarize'])
    parser.add_argument('--arm',choices=SPEC['arms']);parser.add_argument('--cohort',default='development',choices=['development','confirmation'])
    args=parser.parse_args()
    if args.command=='evaluate': evaluate(args.arm,args.cohort)
    elif args.command=='summarize': summarize(args.cohort)
    else: globals()[args.command]()
