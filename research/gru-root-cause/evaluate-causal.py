"""Evaluate matched causal checkpoints on unchanged episodes and fixed action pairs."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

import run as r


def evaluate(arm):
    started = time.perf_counter()
    cfg, policy = r.setup()
    trained = r.HERE / 'artifacts' / 'causal-training' / arm
    manifest = json.loads((trained / 'manifest.json').read_text())
    assert manifest['complete'] and manifest['episodes'] == 2000
    weight = trained / 'nominal.pth'
    assert hashlib.sha256(weight.read_bytes()).hexdigest() == manifest['final_sha256']
    policy.load_state_dict(torch.load(weight, map_location=policy.device))
    policy.eval()
    r.OUT = r.HERE / 'artifacts' / 'causal-evaluation' / arm
    assert not r.OUT.exists(), 'Do not overwrite evaluation evidence'
    r.OUT.mkdir(parents=True)
    baseline = []
    for scene in range(6):
        for case in range(100,140):
            tape = r.load_tape(scene, case)
            result, trace, _ = r.rollout(cfg, policy, scene, case, tape)
            path = r.OUT / 'baseline' / f'{scene}-{case}.npz'
            path.parent.mkdir(exist_ok=True)
            np.savez_compressed(path, records=trace)
            baseline.append(result)
            r.m.save_json(r.OUT / 'baseline.json', baseline)
            if len(baseline)%20 == 0:
                print(arm, 'baseline',len(baseline),'success',sum(v['success'] for v in baseline),flush=True)
    original = json.loads((r.HERE/'value-contrasts.json').read_text())
    roots = json.loads((r.FINDINGS/'counterfactual-protocol.json').read_text())['roots']
    assert len(original) == len(roots) == 164
    contrasts = []
    for number, (root, historical) in enumerate(zip(roots, original)):
        assert (root['scene'],root['case'],root['step']) == (historical['scene'],historical['case'],historical['step'])
        scene, case = r.m.SCENES.index(root['scene']), root['case']
        tape = r.load_tape(scene,case)
        env, snap = r.frozen_root(cfg, policy, root, tape)
        part = r.decompose(policy,env)
        indices = [historical['selected_index'],historical['forward_index']]
        assert part['safe'][indices].all()
        prefix = list(snap['history'])+[part['token']]
        outcomes, windows = [], []
        for label, index in zip(['selected','forward'],indices):
            result, trace, first = r.rollout(cfg,policy,scene,case,tape,snap,
                                            (index,part['commands'][index]))
            path = r.OUT / 'contrasts' / f'{scene}-{case}-{label}.npz'
            path.parent.mkdir(exist_ok=True)
            np.savez_compressed(path,records=trace)
            joint = r.m.JointState(first['next_robot'],first['next_humans'])
            windows.append(r.m.window(prefix+[r.m.token(policy,joint)]))
            outcomes.append((result,first))
        with torch.no_grad():
            values = policy.forward_candidate_values(torch.as_tensor(np.stack(windows),device=policy.device)).cpu().numpy()
        bq = [f['reward']+.99*(not f['done'])*v for (_,f),v in zip(outcomes,values)]
        owntrace = np.load(r.OUT/'baseline'/f'{scene}-{case}.npz')['records']
        oldtrace = np.load(r.OLD/'artifacts'/'development-traces'/'nominal'/f'{scene}-{case}.npz')['records']
        t = root['step']
        contrasts.append({'scene':root['scene'],'case':case,'step':t,'partition':root['partition'],
                          'fixed_indices':indices,'new_selected_index':part['best'],
                          'new_selected_progress':float(part['progress'][part['best']]),
                          'fixed_pair_A_q':part['q'][indices].tolist(),
                          'fixed_pair_B_q':[float(x) for x in bq],
                          'C_returns':[o['return'] for o,_ in outcomes],
                          'C_outcomes':[o for o,_ in outcomes],
                          'new_policy_reaches_same_root':bool(len(owntrace)>t and np.array_equal(owntrace[:t,1:6],oldtrace[:t,1:6]))})
        r.m.save_json(r.OUT/'contrasts.json',contrasts)
        if (number+1)%20 == 0:
            print(arm,'fixed-root',number+1,flush=True)
    assert hashlib.sha256(weight.read_bytes()).hexdigest() == manifest['final_sha256']
    r.m.save_json(r.OUT/'complete.json',{'arm':arm,'episodes':240,'roots':164,
                                        'checkpoint_sha256':manifest['final_sha256'],
                                        'test_smoothing':policy.test_action_smoothing,
                                        'test_filter':policy.test_min_clearance,'test_risk':policy.test_risk_lambda,
                                        'elapsed_seconds':time.perf_counter()-started})
    print('COMPLETE evaluation',arm,flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('arm',choices=['control','matched-smoothing'])
    evaluate(parser.parse_args().arm)
