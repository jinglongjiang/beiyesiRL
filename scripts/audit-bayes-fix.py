"""CPU-only preregistered bias and sigma-information diagnostics."""

import argparse
import configparser
import csv
import hashlib
import importlib.util
import json
import logging
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import crowd_nav
import crowd_sim
import crowd_nav.policy.mamba_rl as policy_module
from crowd_nav import contracts
from crowd_nav.test import test_episode
from crowd_sim.envs.crowd_sim import CrowdSim
from crowd_sim.envs.utils.robot import Robot

OUTPUT = ROOT / 'crowd_nav/runs/bayes-fix-20261008'
ASSETS = ROOT / 'crowd_nav/runs/bayes-matched-v1'
BACKUP = Path('/home/abc/4090_backup_20261008/root')
CHECKPOINT = BACKUP / 'bayes-vl-optimized-20261007/crowd_nav/runs/optimized-10000-r1/formal-bayes/rl_model_ep10000.pth'
DATASET = BACKUP / 'bayes-vl-matched-20261006/crowd_nav/runs/bayes-matched-v1/orca-shared.pth'
SCENARIOS = (
    ('baseline_circle','circle_crossing',5,'circle_radius',4),
    ('baseline_square','square_crossing',10,'square_width',10),
    ('dense_circle','circle_crossing',10,'circle_radius',4),
    ('dense_square','square_crossing',20,'square_width',10),
    ('large_circle','circle_crossing',12,'circle_radius',6),
    ('large_square','square_crossing',20,'square_width',14),
)


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def pinned_model():
    for module in (crowd_nav, crowd_sim, policy_module):
        Path(module.__file__).resolve().relative_to(ROOT)
    protocol = json.loads((OUTPUT/'protocol.json').read_text())
    for path, field in ((CHECKPOINT,'old_full_checkpoint_sha256'),(DATASET,'dataset_sha256')):
        expected = protocol[field]
        assert expected and hashlib.sha256(path.read_bytes()).hexdigest() == expected
    cfg = configparser.RawConfigParser(inline_comment_prefixes=('#',';'))
    cfg.read(ASSETS/'bayes-seed42.eval.ini')
    contracts.init_grid_from_cfg(cfg)
    assert contracts.grid_action_dim(contracts.GRID) == 80
    assert contracts.GRID['sampling'] == 'exponential' and contracts.GRID['v_min'] == .05
    # Reproduce the old diagnostic even after the production encoder is repaired.
    import crowd_nav.policy.bayes_temporal as encoder_module
    spec=importlib.util.spec_from_file_location('frozen_old_bayes',OUTPUT/'old-bayes-temporal.py')
    old_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(old_module)
    current_class=encoder_module.BayesianTemporalEncoder
    try:
        encoder_module.BayesianTemporalEncoder=old_module.BayesianTemporalEncoder
        policy = policy_module.MambaRLPolicy(cfg,device='cpu').eval()
    finally:
        encoder_module.BayesianTemporalEncoder=current_class
    saved = torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    policy.load_state_dict(saved['policy_state'],strict=True)
    assert policy.temporal_encoder.readout[0].in_features == 64, 'B0 must use old unmodified encoder'
    policy.use_sarl_predict=True; policy.set_phase('test'); policy.epsilon=0
    return policy,cfg


def b0():
    policy,_=pinned_model()
    data=torch.load(DATASET,map_location='cpu',weights_only=False)
    assert data['version']=='matched-native-orca-v1'
    eligible=[i for i,e in enumerate(data['trajectories']) if len(e['states'])>=24]
    selected=[eligible[i] for i in np.linspace(0,len(eligible)-1,512,dtype=int)]
    assert len(set(selected))==512
    means,variances,gaps,readouts=[],[],[],[]
    with torch.no_grad():
        for start in range(0,512,16):
            raw=np.asarray([data['trajectories'][i]['states'][:24] for i in selected[start:start+16]],dtype=np.float32)
            tokens=contracts.joint34_to_tokens(raw.reshape(-1,34))
            tokens=torch.as_tensor(tokens,dtype=torch.float32).reshape(-1,24,8,13)
            spatial=policy.spatial_encoder(tokens)
            mu,var=policy.temporal_encoder.posterior(spatial)
            mu=mu.flatten(0,1);var=var.flatten(0,1)
            mean_features=policy.temporal_encoder.readout(mu)
            full_features=policy.temporal_encoder.expected_features(mu,var)
            means.append(mu);variances.append(var);readouts.append(mean_features);gaps.append(full_features-mean_features)
    mu=torch.cat(means);var=torch.cat(variances);gap=torch.cat(gaps);mean_features=torch.cat(readouts)
    constant=gap.mean(0);residual=gap-constant
    ratio=float(constant.norm()/gap.norm(dim=-1).mean())
    residual_ratio=float(residual.norm(dim=-1).mean()/mean_features.norm(dim=-1).mean())
    cosine=F.cosine_similarity(gap,constant[None],dim=-1)
    passed=ratio>.95 and residual_ratio<.02 and float(cosine.median())>.99
    result=dict(status='PASS' if passed else 'B0_DISCREPANCY_STOP',states=len(mu),real_episode_frames=True,
        dataset='first 24 observed ORCA frames, 512 distinct original episodes',episode_indices=selected,
        constant_ratio=ratio,residual_ratio=residual_ratio,cosine_median=float(cosine.median()),
        cosine_p05=float(torch.quantile(cosine,.05)),
        gap_relative_to_mean=float(gap.norm(dim=-1).mean()/mean_features.norm(dim=-1).mean()),
        sigma_median=float(var.sqrt().median()),
        limitation='Training-corpus cohort, not the previous diagnostic cohort; dominant constant component is not an exact constant function.')
    np.savez_compressed(OUTPUT/'b0-latents.npz',mean=mu.numpy(),variance=var.numpy(),gap=gap.numpy())
    save(OUTPUT/'b0.json',result);print(json.dumps(result,indent=2),flush=True)
    if not passed:raise RuntimeError('B0 failed: production encoder must remain unchanged')


def auc(y,score):
    y=np.asarray(y,dtype=bool);score=np.asarray(score)
    if not y.any() or y.all():return None
    values,inverse,counts=np.unique(score,return_inverse=True,return_counts=True)
    average_rank=np.cumsum(counts)-(counts-1)/2
    positive=int(y.sum());negative=len(y)-positive
    return float((average_rank[inverse][y].sum()-positive*(positive+1)/2)/(positive*negative))


def probe(sigma,labels,valid,episodes,case_indices,descriptors):
    holdout=case_indices%4==0
    train=valid & ~holdout;test=valid & holdout
    result=dict(train_steps=int(train.sum()),test_steps=int(test.sum()),train_positives=int(labels[train].sum()),
        test_positives=int(labels[test].sum()),train_episodes=len(set(episodes[train])),test_episodes=len(set(episodes[test])))
    if len(np.unique(labels[train]))<2 or len(np.unique(labels[test]))<2:
        return dict(result,status='EVIDENCE_INSUFFICIENT_SINGLE_CLASS')
    scores={}
    for name,raw in (('sigma_mean',sigma.mean(1)),('sigma_max',sigma.max(1)),
                     ('current_clearance',descriptors[:,0]),('history_age',descriptors[:,1])):
        training_auc=auc(labels[train],raw[train]);direction=1 if training_auc>=.5 else -1
        scores[name]=raw*direction
        result[name]=dict(direction_selected_on_training=direction,raw_test_auc=auc(labels[test],raw[test]),test_auc=auc(labels[test],scores[name][test]))
    x=sigma.astype(np.float64)
    center=x[train].mean(0);scale=np.maximum(x[train].std(0),1e-6)
    x=(x-center)/scale
    train_x=torch.tensor(x[train]);train_y=torch.tensor(labels[train],dtype=torch.float64)
    weights=torch.zeros(64,dtype=torch.float64,requires_grad=True)
    bias=torch.tensor(0.,dtype=torch.float64,requires_grad=True)
    optimizer=torch.optim.LBFGS([weights,bias],max_iter=100,line_search_fn='strong_wolfe')
    def closure():
        optimizer.zero_grad()
        loss=F.binary_cross_entropy_with_logits(train_x@weights+bias,train_y)+.001*weights.square().mean()
        loss.backward();return loss
    optimizer.step(closure)
    scores['sigma_linear']=(torch.tensor(x)@weights.detach()+bias.detach()).numpy()
    result['sigma_linear']=dict(test_auc=auc(labels[test],scores['sigma_linear'][test]))
    rng=np.random.default_rng(20261008)
    ids=sorted(set(episodes[test]));clusters={e:np.flatnonzero(test & (episodes==e)) for e in ids}
    intervals={name:[] for name in scores}
    for _ in range(300):
        sample=np.concatenate([clusters[e] for e in rng.choice(ids,len(ids),replace=True)])
        for name,score in scores.items():
            value=auc(labels[sample],score[sample])
            if value is not None:intervals[name].append(value)
    for name,values in intervals.items():
        result[name]['episode_bootstrap_95_ci']=np.quantile(values,[.025,.975]).tolist() if values else None
    result['status']='ASSOCIATION_DIAGNOSTIC_ONLY'
    return result


def b1():
    # Retain the initial ORCA-prefix failure. The native cohort is the already
    # preregistered B1 cohort and diagnoses distribution/clock differences.
    assert (OUTPUT/'b0.json').exists()
    policy,_=pinned_model()
    with (ROOT/'crowd_nav/runs/bayes-optimized-20261007/accepted-run/formal-bayes/benchmark-10000.csv').open() as stream:
        cases={(r['scenario'],int(r['episode'])):r for r in csv.DictReader(stream)}
    records=[];start=time.time();feature_gaps=[];feature_means=[];history_lengths=[]
    with torch.no_grad():
        for scene_id,(desc,sim,count,dimension,size) in enumerate(SCENARIOS):
            cfg=configparser.RawConfigParser(inline_comment_prefixes=('#',';'));cfg.read(ASSETS/'bayes-seed42.eval.ini')
            contracts.init_grid_from_cfg(cfg);assert contracts.grid_action_dim(contracts.GRID)==80
            cfg.set('sim','test_sim',sim);cfg.set('sim','human_num',str(count));cfg.set('sim',dimension,str(size))
            env=CrowdSim();env.configure(cfg);env.phase='test'
            robot=Robot(cfg,'robot');robot.set_policy(policy);robot.env=env;env.set_robot(robot)
            if hasattr(policy,'set_env'):policy.set_env(env)
            policy.set_env_dt(.25);policy.set_phase('test')
            policy._test_args=SimpleNamespace(gating=False,discrete_search=False,mamba_bias=False,
                mamba_rescue=False,behavior_profile='nominal',measure_latency=False)
            original=policy.predict
            current=[]
            def observed_predict(state):
                raw=policy._build_joint_state_34(state.self_state,state.human_states)
                token=contracts._batch_joint34_to_tokens_vectorized(raw.reshape(1,-1))[0]
                history=(list(policy._history)+[token])[-24:]
                tensor=torch.tensor(np.asarray(history),dtype=torch.float32)[None]
                mean,variance=policy.temporal_encoder.posterior(policy.spatial_encoder(tensor))
                mean_features=policy.temporal_encoder.readout(mean[:,-1])
                full_features=policy.temporal_encoder.expected_features(mean[:,-1],variance[:,-1])
                feature_gaps.append((full_features-mean_features)[0].numpy())
                feature_means.append(mean_features[0].numpy());history_lengths.append(len(history))
                clearance=min(np.hypot(robot.px-h.px,robot.py-h.py)-robot.radius-h.radius for h in env.humans)
                current.append((variance[0,-1].sqrt().numpy(),float(clearance),len(history)))
                return original(state)
            policy.predict=observed_predict
            try:
                for episode in range(32):
                    current=[];reference=cases[(desc,episode)]
                    outcome,steps,positions,clearances,_,_,_=test_episode(env,robot,policy,int(reference['seed']),case_desc=desc)
                    assert len(current)==steps
                    assert outcome==reference['outcome'] and steps==int(reference['steps']), 'Old-policy replay differs; investigate before inference'
                    records.append(dict(scene=desc,scene_id=scene_id,episode=episode,seed=int(reference['seed']),outcome=outcome,
                        sigma=np.asarray([r[0] for r in current]),clearance=np.asarray(clearances),
                        descriptors=np.asarray([[r[1],r[2]] for r in current])))
                print(json.dumps(dict(stage='B1',scene=desc,episodes=len(records),seconds=time.time()-start)),flush=True)
            finally:policy.predict=original
    sigma=np.concatenate([r['sigma'] for r in records]);descriptors=np.concatenate([r['descriptors'] for r in records])
    episodes=np.concatenate([np.full(len(r['sigma']),i) for i,r in enumerate(records)])
    case_indices=np.concatenate([np.full(len(r['sigma']),r['episode']) for r in records])
    collision=np.concatenate([np.full(len(r['sigma']),r['outcome']=='collision') for r in records])
    labels={};validity={}
    for horizon in (4,8):
        for threshold in (0.,.2):
            name=f'future-{horizon}-clearance-{threshold}'
            ys=[];vs=[]
            for record in records:
                clearance=record['clearance']
                for tick in range(len(clearance)):
                    future=clearance[tick:tick+horizon]
                    dangerous=bool(future.min()<threshold)
                    ys.append(dangerous);vs.append(len(future)==horizon or dangerous)
            labels[name]=np.asarray(ys,dtype=bool);validity[name]=np.asarray(vs,dtype=bool)
    labels['episode-collision']=collision;validity['episode-collision']=np.ones(len(sigma),dtype=bool)
    results={name:probe(sigma,y,validity[name],episodes,case_indices,descriptors) for name,y in labels.items()}
    np.savez_compressed(OUTPUT/'b1-rollout.npz',sigma=sigma,descriptors=descriptors,episode=episodes,case_index=case_indices,
        **{f'label-{k}':v for k,v in labels.items()},**{f'valid-{k}':v for k,v in validity.items()})
    gap=np.asarray(feature_gaps);mean_features=np.asarray(feature_means);constant=gap.mean(0)
    cosine=(gap@constant)/(np.linalg.norm(gap,axis=1)*np.linalg.norm(constant))
    ratio=float(np.linalg.norm(constant)/np.linalg.norm(gap,axis=1).mean())
    residual=float(np.linalg.norm(gap-constant,axis=1).mean()/np.linalg.norm(mean_features,axis=1).mean())
    passed=len(gap)>=10000 and ratio>.95 and residual<.02 and float(np.median(cosine))>.99
    native=dict(status='PASS' if passed else 'B0_NATIVE_DISCREPANCY_STOP',states=len(gap),constant_ratio=ratio,
        residual_ratio=residual,cosine_median=float(np.median(cosine)),cosine_p05=float(np.quantile(cosine,.05)),
        gap_relative_to_mean=float(np.linalg.norm(gap,axis=1).mean()/np.linalg.norm(mean_features,axis=1).mean()),
        sigma_median=float(np.median(sigma)),initial_orca_prefix_b0='failed, retained unchanged',
        explanation='Native deployed Full current-history states, rather than a sample restricted to the first 24 ORCA observations.')
    save(OUTPUT/'b0-native.json',native)
    np.savez_compressed(OUTPUT/'b0-native-features.npz',gap=gap,mean_features=mean_features,history_length=np.asarray(history_lengths))
    save(OUTPUT/'b1.json',dict(status='COMPLETE',steps=len(sigma),episodes=len(records),cpu_only=True,
        original_outcome_step_parity=True,collision_episodes=sum(r['outcome']=='collision' for r in records),results=results,
        episode_metadata=[{k:v for k,v in r.items() if k not in ('sigma','clearance','descriptors')} for r in records],
        limitation='AUC is association on reused diagnostic cases, not posterior calibration, causal gain or fresh confirmation.'))
    print(json.dumps(dict(b0_native=native,b1=results),indent=2),flush=True)
    if not passed:raise RuntimeError('Native B0 failed; do not modify production encoder')


def register_b2():
    metadata=json.loads((OUTPUT/'b1.json').read_text())['episode_metadata']
    with (ROOT/'crowd_nav/runs/bayes-optimized-20261007/accepted-run/formal-bayes/benchmark-10000.csv').open() as stream:
        cases={(r['scenario'],int(r['episode'])):r for r in csv.DictReader(stream)}
    roots=[]
    for record in [r for r in metadata if r['outcome']!='success'][:12]:
        steps=int(cases[record['scene'],record['episode']]['steps'])
        roots.append(dict(record,steps=steps,root_step=min(23,steps-2)))
    assert len(roots)==12 and all(r['root_step']>=0 for r in roots)
    path=OUTPUT/'b2-protocol.json';assert not path.exists()
    save(path,dict(roots=roots,registered_unix=time.time(),wall_budget_seconds=1800,
        candidates='All 80 native actions subject to the unchanged Parent one-step safe mask, if any safe candidate exists.',
        intervention='One forced selection at the fixed root, original smoothing and last-action state, then original Parent continuation.',
        oracle='Offline true continuation outcomes; choose contact-free success with highest native discounted return.',
        replay='Reset/replay, no deepcopy; exact root hash and baseline outcome/steps required.',
        claim='Conditional single-action headroom on failure-selected development roots, not population SR upper bound or Bayesian-specific gain.'))
    print(json.dumps(roots,indent=2),flush=True)


def b2():
    protocol=json.loads((OUTPUT/'b2-protocol.json').read_text())
    policy,_=pinned_model();started=time.time();summaries=[]
    log=OUTPUT/'b2-trials.jsonl';assert not log.exists()
    with log.open('w') as stream,torch.no_grad():
        for root in protocol['roots']:
            desc,sim,count,dimension,size=SCENARIOS[root['scene_id']]
            cfg=configparser.RawConfigParser(inline_comment_prefixes=('#',';'));cfg.read(ASSETS/'bayes-seed42.eval.ini')
            contracts.init_grid_from_cfg(cfg);assert contracts.grid_action_dim(contracts.GRID)==80
            cfg.set('sim','test_sim',sim);cfg.set('sim','human_num',str(count));cfg.set('sim',dimension,str(size))
            env=CrowdSim();env.configure(cfg);env.phase='test';robot=Robot(cfg,'robot');env.set_robot(robot)
            robot.set_policy(policy);robot.env=env
            if hasattr(policy,'set_env'):policy.set_env(env)
            policy.set_env_dt(.25)
            policy._test_args=SimpleNamespace(gating=False,discrete_search=False,mamba_bias=False,
                mamba_rescue=False,behavior_profile='nominal',measure_latency=False)
            original_predict=policy.predict;original_step=env.step
            def rollout(forced):
                from crowd_sim.envs.utils.action import ActionXY
                tick=0;executed=0;total_return=0.;capture={};override=None
                def predict(state):
                    nonlocal tick,override
                    previous=policy._last_action
                    if tick==root['root_step']:
                        commands,token,_,_,dmins=policy._score_candidates_vectorized(state)
                        legal=np.asarray(dmins)>=policy.test_min_clearance
                        if not legal.any():legal[:]=True
                        physical=np.asarray([[h.px,h.py,h.vx,h.vy,h.radius] for h in env.humans],dtype=np.float64)
                        state_bytes=policy._build_joint_state_34(state.self_state,state.human_states).tobytes()
                        history_bytes=np.asarray(list(policy._history),dtype=np.float32).tobytes()
                        last_bytes=np.asarray([previous.vx,previous.vy] if previous else [0.,0.],dtype=np.float64).tobytes()
                        capture['root_sha256']=hashlib.sha256(state_bytes+history_bytes+physical.tobytes()+last_bytes).hexdigest()
                        capture['allowed']=np.flatnonzero(legal).tolist()
                    action=original_predict(state)
                    if tick==root['root_step']:
                        capture['parent_index']=int(policy._selected_action_index)
                        if forced is not None:
                            assert forced in capture['allowed']
                            action=commands[forced]
                            if previous is not None and policy.test_action_smoothing>0:
                                alpha=float(policy.test_action_smoothing)
                                action=ActionXY(alpha*previous.vx+(1-alpha)*action.vx,
                                                alpha*previous.vy+(1-alpha)*action.vy)
                            policy._last_action=action;policy._selected_action_index=forced
                            override=(action.vx,action.vy)
                        capture['executed_root_action']=[float(action.vx),float(action.vy)]
                    tick+=1;return action
                def step(action,*args,**kwargs):
                    nonlocal executed,total_return
                    if executed==root['root_step'] and override is not None:
                        assert (action.vx,action.vy)==override
                    result=original_step(action,*args,**kwargs)
                    total_return+=policy.gamma**executed*float(result[1]);executed+=1
                    return result
                policy.predict=predict;env.step=step
                try:
                    outcome,steps,_,clearance,_,_,_=test_episode(env,robot,policy,root['seed'],case_desc=desc)
                finally:policy.predict=original_predict;env.step=original_step
                assert capture and tick==steps
                return dict(root=root,forced_index=forced,outcome=outcome,steps=steps,
                    native_discounted_return=total_return,min_clearance=float(np.min(clearance)),
                    geometric_contact_steps=int(np.sum(np.asarray(clearance)<0)),**capture)
            baseline=rollout(None)
            assert baseline['outcome']==root['outcome'] and baseline['steps']==root['steps']
            stream.write(json.dumps(baseline)+'\n');stream.flush()
            control=rollout(baseline['parent_index'])
            assert control['root_sha256']==baseline['root_sha256']
            assert control['outcome']==baseline['outcome'] and control['steps']==baseline['steps']
            assert control['executed_root_action']==baseline['executed_root_action']
            stream.write(json.dumps(control)+'\n');stream.flush()
            trials=[control];complete=True
            for index in baseline['allowed']:
                if index==baseline['parent_index']:continue
                if time.time()-started>protocol['wall_budget_seconds']:
                    complete=False;break
                trial=rollout(index);assert trial['root_sha256']==baseline['root_sha256']
                stream.write(json.dumps(trial)+'\n');stream.flush();trials.append(trial)
            rescued=[r for r in trials if r['outcome']=='success' and r['geometric_contact_steps']==0]
            best=max(rescued,key=lambda r:r['native_discounted_return']) if rescued else None
            summary=dict(root=root,allowed=len(baseline['allowed']),tested=len(trials),complete=complete,
                parent_index=baseline['parent_index'],safe_goal_rescue=best is not None,best=best)
            summaries.append(summary);print(json.dumps(summary),flush=True)
            save(OUTPUT/'b2-progress.json',dict(roots=summaries,wall_seconds=time.time()-started))
            if not complete:break
    result=dict(status='COMPLETE' if len(summaries)==12 and all(r['complete'] for r in summaries) else 'BUDGET_LIMITED_PARTIAL',
        roots=summaries,wall_seconds=time.time()-started,
        contact_free_goal_rescues=sum(r['safe_goal_rescue'] for r in summaries),
        observed_192_case_single_override_gain_pp=100*sum(r['safe_goal_rescue'] for r in summaries)/192,
        population_sr_upper_bound=None,
        limitation='Known-future single-root interventions with unchanged old Parent continuation. Failure-selected development cohort, not a deployable policy, perfect all-step ranking or Bayesian-specific benefit.')
    save(OUTPUT/'b2.json',result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('b0','b1','b2-register','b2'));args=parser.parse_args()
    logging.basicConfig(level=logging.WARNING);torch.set_num_threads(2)
    torch.manual_seed(42);np.random.seed(42)
    {'b0':b0,'b1':b1,'b2-register':register_b2,'b2':b2}[args.stage]()
