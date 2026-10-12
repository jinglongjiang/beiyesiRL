"""Registered activation and closed-loop tests for a distinct sampling hypothesis."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0,str(HERE))
from controller import Controller


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


x=load('component_loop',HERE.parent/'progress-contract-v2/run.py')
SPEC=json.loads((HERE/'protocol.json').read_text())
x.HERE,x.OUT,x.SPEC,x.Controller=HERE,HERE/'artifacts',SPEC,Controller
h,r,m=x.h,x.r,x.m
h.OUT=x.OUT


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(name,data):m.save_json(HERE/name,data)


def freeze():
    files=[p for d in ['crowd_nav','crowd_sim'] for p in (ROOT/d).rglob('*.py')]
    files+=list(HERE.glob('*.py'))+[HERE/'protocol.json',HERE/'PRIOR.md',r.OLD/'parent-config.ini',r.OLD/'artifacts/nominal.pth']
    files+=[HERE.parent/p for p in ['progress-contract-v2/run.py','progress-contract-v2/controller.py',
        'progress-contract/controller.py','gru-causal-repairs/run.py','gru-causal-repairs/analyze.py',
        'gru-root-cause/run.py','gru-complete-actions/run.py','recovery-duration/run.py']]
    files+=[HERE.parent/f'policy-handoff/development-{a}.json' for a in ['nominal','hold8']]
    import pytorch_mppi.mppi as upstream
    assert sha(r.OLD/'artifacts/nominal.pth')==SPEC['parent_sha256']
    save('frozen-manifest.json',{'files':{str(p.relative_to(ROOT)):sha(p) for p in sorted(set(files))},
        'upstream_mppi_sha256':sha(upstream.__file__)})


def controls():
    x.verify(); cfg,p=h.setup(); h.weight(p,'nominal')
    c=Controller('components',p)
    ids,labels,count=c.components(np.ones(80,bool)); assert count==1
    for draw in np.linspace(0,1,1001,endpoint=False):
        assert c.balanced(ids,labels,count,draw)==ids[int(draw*len(ids))]
    # A deliberately disconnected graph mask checks allocation only, not navigation performance.
    ids=np.array([0,1,2,3,4,5]); labels=np.array([0,0,0,0,0,1])
    picks=np.array([c.balanced(ids,labels,2,u) for u in (np.arange(1000)+.5)/1000])
    assert (picks==5).sum()==500
    rows=[]; q=s=0; split_scenes=0
    for scene in range(6):
        tape=x.humans(cfg,p,scene,121)
        row=x.episode(cfg,p,scene,121,tape,'components',f'controls/first/{scene}-121')
        diag=json.loads((x.OUT/f'controls/first/{scene}-121-mechanism.json').read_text())
        queries=sum(v['sample_queries'] for v in diag); split=sum(v['split_queries'] for v in diag)
        q+=queries; s+=split; split_scenes+=int(split>0); rows.append(row)
        if scene==1:
            x.episode(cfg,p,scene,121,tape,'components',f'controls/duplicate/{scene}-121')
            for key in ['records','candidates']:
                np.testing.assert_array_equal(np.load(x.OUT/f'controls/first/{scene}-121.npz')[key],
                    np.load(x.OUT/f'controls/duplicate/{scene}-121.npz')[key])
        print('CONTROL',scene,row['outcome'],'split',split,'/',queries,flush=True)
    active=q>0 and s/q>=.01 and split_scenes>=2
    save('controls.json',{'passed':True,'grid':80,'time_limit':50,'repeat_exact':True,
        'single_component_identical':True,'synthetic_component_quota':True,
        'sample_queries':q,'split_queries':s,'split_scenes':split_scenes,
        'activation_passed':active,'scientific_verdict':'ACTIVE_REFERENCE' if active else 'VERSION_NEGATIVE_INACTIVE',
        'results':rows})
    x.verify()


def evaluate(arm,cohort):
    x.verify(); control=json.loads((HERE/'controls.json').read_text())
    assert control['passed'] and control['activation_passed']
    assert cohort=='development','Fresh confirmation requires a separate frozen gate/prior review.'
    cfg,p=h.setup(); h.weight(p,'nominal'); rows=[]
    for scene in SPEC[cohort]['scenes']:
        for case in range(SPEC[cohort]['case_first'],SPEC[cohort]['case_last']+1):
            if arm in ['nominal','hold8']:
                source=HERE.parent/f'policy-handoff/development-{arm}.json'
                row=dict(next(v for v in json.loads(source.read_text())['results'] if v['scene']==m.SCENES[scene] and v['case']==case),reused=True,source_sha256=sha(source))
            else:
                row=x.episode(cfg,p,scene,case,x.humans(cfg,p,scene,case),arm,f'{cohort}/{arm}/{scene}-{case}')
            rows.append(row); save(f'{cohort}-{arm}.json',{'results':rows})
            print(cohort,arm,len(rows),row['outcome'],flush=True)
    x.verify()


def summarize():
    rows={a:json.loads((HERE/f'development-{a}.json').read_text())['results'] for a in SPEC['arms']}
    assert all(len(v)==120 for v in rows.values())
    overall={a:x.stats.aggregate(v) for a,v in rows.items()}
    pairs={a:x.stats.pairs(rows['components'],rows[a]) for a in rows if a!='components'}
    strongest=max(pairs,key=lambda a:(overall[a]['success'],-overall[a]['contact']))
    base={(v['scene'],v['case']):v for v in rows[strongest]}
    both=[v for v in rows['components'] if v['success'] and base[(v['scene'],v['case'])]['success']]
    ratio=float(np.mean([v['duration'] for v in both])/np.mean([base[(v['scene'],v['case'])]['duration'] for v in both]))
    checks={'no_success_regression':all(overall['components']['success']>=overall[a]['success'] for a in pairs),
        'safety':all(overall['components']['contact']<=overall[a]['contact'] and overall['components']['collision']<=overall[a]['collision'] for a in pairs),
        'independent_gain':all(overall['components']['success']>=overall[a]['success']+2 for a in pairs) or ratio<=.95,
        'compute':overall['components']['decision_ms_episode_mean']<250,
        'activation':json.loads((HERE/'controls.json').read_text())['activation_passed']}
    save('development-summary.json',{'overall':overall,'pairs':pairs,'strongest_simple':strongest,
        'paired_time_ratio':ratio,'gate':{'checks':checks,'passed':all(checks.values())},
        'verdict':'METHOD_SIGNAL_FOUND' if all(checks.values()) else 'VERSION_NEGATIVE',
        'contribution':'UNVERIFIED; no paper PASS without exact prior difference and fresh confirmation.'})
    print('SUMMARY',checks,ratio,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('command',choices=['freeze','controls','verify','evaluate','summarize'])
    parser.add_argument('--arm',choices=SPEC['arms']); parser.add_argument('--cohort',default='development')
    args=parser.parse_args()
    if args.command=='verify':x.verify()
    elif args.command=='evaluate':evaluate(args.arm,args.cohort)
    else:globals()[args.command]()
