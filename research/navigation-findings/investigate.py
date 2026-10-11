"""Read frozen results and replay only the states needed for causal diagnosis."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'recovery-duration'
OUT = HERE / 'artifacts'
OUT.mkdir(exist_ok=True)
spec = importlib.util.spec_from_file_location('frozen_recovery', OLD / 'run.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
PROTO = json.loads((HERE / 'protocol.json').read_text())


def write(name, obj):
    (HERE / name).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def setup():
    manifest = json.loads((OLD / 'frozen-manifest.json').read_text())
    for filename, key in [('nominal.pth', 'nominal_sha256'),
                          ('ordinary.pth', 'ordinary_checkpoint_sha256'),
                          ('paired.pth', 'paired_checkpoint_sha256')]:
        actual = hashlib.sha256((m.OUT / filename).read_bytes()).hexdigest()
        assert actual and actual == manifest[key], (filename, actual)
    cfg, p = m.setup('cuda')
    p.load_state_dict(torch.load(m.OUT / 'nominal.pth', map_location=p.device))
    p.eval()
    p.set_phase('test')
    return cfg, p


def model(arm, device):
    ck = torch.load(m.OUT / f'{arm}.pth', map_location=device)
    net = m.RecoveryNet().to(device).eval()
    net.load_state_dict(ck['state'])
    return net, (ck['mean'].cpu().numpy(), ck['std'].cpu().numpy())


def offline_recovery():
    d = np.load(m.OUT / 'recovery-data.npz')
    x, g = d['x'], d['returns']
    result = {'roots': len(x), 'arms': {}, 'traces': {}}
    for arm in ['ordinary', 'paired']:
        net, norm = model(arm, 'cpu')
        with torch.no_grad():
            q = net(torch.as_tensor((x-norm[0])/norm[1])).numpy()
        chosen = q.argmax(1)
        regret = g.max(1) - g[np.arange(len(g)), chosen]
        result['arms'][arm] = {
            'train_return_mse': float(((q-g)**2).mean()),
            'chosen_option_counts': np.bincount(chosen, minlength=10).tolist(),
            'empirical_best_option_counts': np.bincount(g.argmax(1), minlength=10).tolist(),
            'mean_chosen_actual_advantage': float((g[np.arange(len(g)), chosen]-g[:,0]).mean()),
            'mean_chosen_predicted_advantage': float((q[np.arange(len(g)), chosen]-q[:,0]).mean()),
            'negative_actual_advantage_roots': int((g[np.arange(len(g)), chosen]<g[:,0]-1e-6).sum()),
            'mean_selection_regret': float(regret.mean()),
            'median_selection_regret': float(np.median(regret)),
            'regret_max': float(regret.max())}
        all_runs = []
        for f in sorted((m.OUT/'development-traces'/arm).glob('*.npz')):
            a = np.load(f)['records']
            decisions = a[a[:,9]>=0]
            active = a[:,5]!=a[:,6]
            longest = now = 0
            for bit in active:
                now = now+1 if bit else 0
                longest = max(longest, now)
            previous = 0
            countdown_errors = 0
            for row in a:
                if previous>0 and row[9]<0 and row[10] not in (previous-1, 0):
                    countdown_errors += 1
                previous = row[10]
            all_runs.append({'file': f.name, 'first_recovery_step':
                             int(a[a[:,9]>0,0][0]) if (a[:,9]>0).any() else None,
                             'recovery_decisions': int((decisions[:,9]>0).sum()),
                             'nominal_decisions': int((decisions[:,9]==0).sum()),
                             'changed_steps': int(active.sum()), 'steps': len(a),
                             'longest_changed_run_steps': longest,
                             'countdown_errors': countdown_errors})
        result['traces'][arm] = all_runs
    write('recovery-offline.json', result)
    print(json.dumps(result['arms'], indent=2), flush=True)


def decompose(p, env):
    commands, tok, windows, rew, clear = p._score_candidates_vectorized(m.state(env))
    with torch.no_grad():
        values = p.forward_candidate_values(windows).cpu().numpy()
    risk = p.test_risk_lambda*np.maximum(p.test_min_clearance-clear, 0)
    raw = rew + p.gamma*values-risk
    raw[0] -= .001
    valid = clear>=p.test_min_clearance
    scores = raw.copy()
    if valid.any():
        scores[~valid] = -1e9
    best = int(scores.argmax())
    return commands, tok, rew, values, clear, risk, scores, best


def snap(p, env, positions):
    return {'robot': copy.copy(env.robot.get_full_state()), 'time': env.global_time,
            'history': [h.copy() for h in p._history], 'last_action': p._last_action,
            'positions': list(positions)}


def replay(cfg, p, arm, scene, case, root_steps, callback):
    tape = m.make_tape(cfg, p, scene, case)
    env = m.tape_env(cfg, p, scene, case, tape)
    a = np.load(m.OUT/'development-traces'/arm/f'{scene}-{case}.npz')['records']
    positions = []
    errors = {'position': 0., 'action': 0., 'reward': 0., 'clearance': 0.}
    for row in a:
        t = int(row[0])
        np.testing.assert_allclose(env.robot.get_position(), row[1:3], atol=1e-10, rtol=0)
        if t in root_steps:
            parts = decompose(p, env)
            assert parts[-1] == int(row[6]), (arm, scene, case, t, parts[-1], row[6])
            cmd = parts[0][int(row[5])]
            np.testing.assert_allclose(cmd, row[3:5], atol=1e-10, rtol=0)
            callback(env, positions, row, parts, snap(p, env, positions), tape)
        tok = m.token(p, m.state(env))
        action = m.ActionXY(*row[3:5])
        positions.append(env.robot.get_position())
        before = m.before_geometry(env)
        m.commit_action(p, tok, action, int(row[5]))
        _, reward, term, trunc, info = env.step(action)
        errors['reward'] = max(errors['reward'], abs(reward-row[7]))
        errors['clearance'] = max(errors['clearance'], abs(m.geo_step(env, before)-row[8]))
        np.testing.assert_allclose([reward,m.geo_step(env,before)], row[7:9], atol=1e-9, rtol=0)
    assert term or trunc
    return errors


def recovery_roots():
    cfg, p = setup()
    results = []
    maps = {(r['arm'],r['scene'],r['case']):r for r in
            json.loads((m.OUT/'development-results.json').read_text())}
    for scene in range(6):
        for case in [100,101]:
            # Same unmodified nominal prefix for both frozen recovery networks.
            roots = {}
            def capture(env, positions, row, parts, snapshot, tape):
                roots['snapshot'], roots['tape'] = snapshot, tape
                roots['x'] = m.feature(p, env, parts[1], positions)
            replay(cfg,p,'nominal',scene,case,{4},capture)
            for arm in ['ordinary','paired']:
                net, norm = model(arm,p.device)
                with torch.no_grad():
                    q = net(torch.as_tensor((roots['x']-norm[0])/norm[1],device=p.device)).cpu().numpy()
                choice = int(q.argmax())
                res, trace, _ = m.run_episode(cfg,p,scene,case,roots['tape'],
                    snapshot=roots['snapshot'],first_option=choice)
                nominal = maps[('nominal',m.SCENES[scene],case)]
                original = maps[(arm,m.SCENES[scene],case)]
                original_trace = np.load(m.OUT/'development-traces'/arm/f'{scene}-{case}.npz')['records']
                first = original_trace[original_trace[:,9]>=0][0]
                assert int(first[0])==4 and int(first[9])==choice
                suffix = np.load(m.OUT/'development-traces'/'nominal'/f'{scene}-{case}.npz')['records'][4:]
                nominal_suffix_return = float(suffix[:,7]@(.99**np.arange(len(suffix))))
                repeat_suffix = original_trace[4:]
                repeat_suffix_return = float(repeat_suffix[:,7]@(.99**np.arange(len(repeat_suffix))))
                results.append({'scene':m.SCENES[scene],'case':case,'arm':arm,'option':choice,
                                'predicted_q':q.tolist(),'predicted_advantage':float(q[choice]-q[0]),
                                'one_shot':res,'nominal_outcome':nominal['outcome'],
                                'repeated_outcome':original['outcome'],
                                'nominal_root_return':nominal_suffix_return,
                                'repeated_root_return':repeat_suffix_return,
                                'one_shot_advantage':res['discounted_return']-nominal_suffix_return})
                np.savez_compressed(OUT/f'one-shot-{arm}-{scene}-{case}.npz',records=trace)
                write('recovery-first-choice.json',results)
            print('first-choice diagnostic',scene,case,flush=True)
    print('completed one-shot contrasts',len(results),flush=True)


def history_channel():
    cfg,p=setup()
    d=np.load(m.OUT/'recovery-data.npz')
    col=256+9
    assert np.all(d['x'][:,col]==1), 'History age not constant in training'
    records=[]
    for scene in range(6):
        for case in [100,101]:
            def capture(env,positions,row,parts,snapshot,tape):
                x=m.feature(p,env,parts[1],positions)
                for arm in ['ordinary','paired']:
                    net,norm=model(arm,p.device)
                    torch.manual_seed(71)
                    initial=m.RecoveryNet()
                    untouched=torch.equal(initial.net[0].weight[:,col],net.net[0].weight[:,col].cpu())
                    changed=x.copy()
                    changed[col]=1.
                    with torch.no_grad():
                        original=net(torch.as_tensor((x-norm[0])/norm[1],device=p.device)).cpu().numpy()
                        neutral=net(torch.as_tensor((changed-norm[0])/norm[1],device=p.device)).cpu().numpy()
                    records.append({'scene':m.SCENES[scene],'case':case,'arm':arm,
                                    'history_age':float(x[col]),'train_mean':float(norm[0][col]),
                                    'train_std_clamped':float(norm[1][col]),
                                    'normalized_age':float((x[col]-norm[0][col])/norm[1][col]),
                                    'input_weights_identical_to_initialization':untouched,
                                    'actual_choice':int(original.argmax()),
                                    'age_neutralized_choice':int(neutral.argmax()),
                                    'actual_q':original.tolist(),'age_neutralized_q':neutral.tolist()})
            replay(cfg,p,'nominal',scene,case,{4},capture)
    write('recovery-history-channel.json',{'training_column':col,'training_min_max':[1.,1.],
          'rows':records,'scope':'Fixed-root input intervention only. No modified controller is executed or retrained.'})
    print('history channel confirmed',len(records),flush=True)


def nominal_ledger():
    cfg,p=setup()
    outcome={(r['scene'],r['case']):r['outcome'] for r in
             json.loads((m.OUT/'development-results.json').read_text()) if r['arm']=='nominal'}
    result=[]
    action_rows=[]
    for scene in range(6):
        for case in range(100,140):
            def capture(env,positions,row,parts,snapshot,tape):
                commands,tok,rew,values,clear,risk,scores,best=parts
                command=np.array(commands)
                goal=np.array(env.robot.get_goal_position())
                pos=np.array(env.robot.get_position())
                dist=np.linalg.norm(pos-goal)
                successor=pos+command*.25
                progress=dist-np.linalg.norm(successor-goal,axis=1)
                valid=clear>=p.test_min_clearance
                allowed=valid if valid.any() else np.ones(80,dtype=bool)
                alternative=int(np.where(allowed,progress,-np.inf).argmax())
                opposite=(progress[best]<=0 and progress[alternative]>=.02 and
                          rew[alternative]>=rew[best]-1e-8)
                humans=np.array([[h.px,h.py] for h in env.humans])
                selected=[tuple(sorted(np.argsort(((humans-(pos+cmd*.25))**2).sum(1),
                                  kind='stable')[:5].tolist())) for cmd in command]
                record={'scene':m.SCENES[scene],'case':case,'step':int(row[0]),
                        'partition':'discovery' if case<120 else 'mechanism-check',
                        'outcome':outcome[(m.SCENES[scene],case)],
                        'position':pos.tolist(),'goal_distance':float(dist),
                        'actual_action':command[best].tolist(),'chosen_index':best,
                        'progress_index':alternative,'chosen_progress':float(progress[best]),
                        'max_safe_progress':float(progress[alternative]),
                        'chosen_reward':float(rew[best]),'progress_reward':float(rew[alternative]),
                        'chosen_value':float(values[best]),'progress_value':float(values[alternative]),
                        'chosen_risk':float(risk[best]),'progress_risk':float(risk[alternative]),
                        'safe_count':int(valid.sum()),'value_overrides_progress':bool(opposite),
                        'strict_immediate_reversal':bool(opposite and rew[alternative]>=rew[best]+.01),
                        'support_switch':len(set(selected))>1,
                        'chosen_selected_ids':list(selected[best]),
                        'progress_selected_ids':list(selected[alternative]),
                        'native_radius':float(env.robot.radius),
                        'policy_success_radius':float(p.success_radius),
                        'terminal_radius_mismatch_candidates':int(((np.linalg.norm(successor-goal,axis=1)<env.robot.radius)&
                                                   (np.linalg.norm(successor-goal,axis=1)>=p.success_radius)).sum())}
                result.append(record)
                action_rows.append(np.stack([command[:,0],command[:,1],progress,rew,values,
                                             clear,risk,scores,allowed.astype(float)],axis=1))
            replay(cfg,p,'nominal',scene,case,set(PROTO['nominal_discovery']['root_steps']),capture)
            if (case-99)%10==0:
                write('nominal-ledger.json',result)
                print('nominal ledger',m.SCENES[scene],case,'roots',len(result),flush=True)
    write('nominal-ledger.json',result)
    np.savez_compressed(OUT/'nominal-action-scores.npz',scores=np.stack(action_rows))
    print('nominal ledger complete',len(result),flush=True)


def recovery_contract():
    cfg,p=setup()
    summaries=[]
    roots=[]
    rows=json.loads((m.OUT/'development-results.json').read_text())
    for arm in ['ordinary','paired']:
        net,norm=model(arm,p.device)
        for scene in range(6):
            cases=sorted({100,101}|{r['case'] for r in rows if r['arm']==arm and
                         r['scene']==m.SCENES[scene] and r['collision']})
            for case in cases:
                trace=np.load(m.OUT/'development-traces'/arm/f'{scene}-{case}.npz')['records']
                times=set(trace[trace[:,9]>=0,0].astype(int))|{int(trace[-1,0])}
                def capture(env,positions,row,parts,snapshot,tape):
                    commands,tok,rew,values,clear,risk,scores,nominal=parts
                    record={'arm':arm,'scene':m.SCENES[scene],'case':case,
                            'step':int(row[0]),'chosen':int(row[5]),'nominal':nominal,
                            'safe_count':int((clear>=p.test_min_clearance).sum()),
                            'chosen_predicted_clearance':float(clear[int(row[5])]),
                            'recorded_actual_clearance':float(row[8]),
                            'last_step':bool(row[0]==trace[-1,0])}
                    if row[9]>=0:
                        x=m.feature(p,env,tok,positions)
                        with torch.no_grad():
                            q=net(torch.as_tensor((x-norm[0])/norm[1],device=p.device)).cpu().numpy()
                        assert q.argmax()==int(row[9]), (arm,scene,case,row[0],'choice mismatch')
                        record['q']=q.tolist()
                        record['option']=int(row[9])
                    roots.append(record)
                errors=replay(cfg,p,arm,scene,case,times,capture)
                summaries.append({'arm':arm,'scene':m.SCENES[scene],'case':case,
                                  'replay_errors':errors,'verified_decision_roots':len(times)})
            print('recovery contract',arm,scene,flush=True)
    write('recovery-contract.json',{'episodes':summaries,'decisions_and_collision_roots':roots,
          'checks':'asserted recorded/recomputed option, nominal argmax, executable command, physical position, native reward and independent geometry; no controller changes'})


def counterfactuals():
    cfg,p=setup()
    ledger=json.loads((HERE/'nominal-ledger.json').read_text())
    selected={}
    for r in ledger:
        key=(r['scene'],r['case'])
        if r['value_overrides_progress'] and key not in selected:
            selected[key]=r
    selection=[{'scene':s,'case':c,'step':r['step'],'partition':r['partition'],
                'outcome':r['outcome']} for (s,c),r in selected.items()]
    expected=json.loads((HERE/'counterfactual-protocol.json').read_text())['roots']
    assert selection==expected, 'Frozen selection changed'
    results=[]
    for item in selection:
        scene=m.SCENES.index(item['scene'])
        case=item['case']
        root=item['step']
        saved={}
        def capture(env,positions,row,parts,snapshot,tape):
            saved['snapshot'],saved['tape']=snapshot,tape
        replay(cfg,p,'nominal',scene,case,{root},capture)
        parent_trace=np.load(m.OUT/'development-traces'/'nominal'/f'{scene}-{case}.npz')['records']
        tail=parent_trace[parent_trace[:,0]>=root]
        nominal_return=float(tail[:,7]@(.99**np.arange(len(tail))))
        for variant in ['one-step-progress','value-free-progress']:
            original_score=m.score
            choices=[]
            def changed_score(policy,env):
                commands,tok,dmin,nominal=original_score(policy,env)
                now=int(round(env.global_time/.25))
                chosen=nominal
                if variant=='value-free-progress' or now==root:
                    mask=dmin>=policy.test_min_clearance
                    if mask.any():
                        pos=np.array(env.robot.get_position())
                        goal=np.array(env.robot.get_goal_position())
                        distance=np.linalg.norm(pos+np.array(commands)*.25-goal,axis=1)
                        chosen=int(np.where(mask,distance,np.inf).argmin())
                choices.append([now,nominal,chosen])
                return commands,tok,dmin,chosen
            try:
                m.score=changed_score
                res,trace,_=m.run_episode(cfg,p,scene,case,saved['tape'],snapshot=saved['snapshot'])
            finally:
                m.score=original_score
            results.append({**item,'variant':variant,'nominal_root_return':nominal_return,
                            'return_advantage':res['discounted_return']-nominal_return,
                            'result':res})
            np.savez_compressed(OUT/f'counterfactual-{variant}-{scene}-{case}.npz',
                                records=trace,choices=choices)
        write('counterfactual-results.json',results)
        print('counterfactual',item['scene'],case,root,'completed',len(results),flush=True)


def freeze_counterfactuals():
    ledger=json.loads((HERE/'nominal-ledger.json').read_text())
    selected={}
    for r in ledger:
        key=(r['scene'],r['case'])
        if r['value_overrides_progress'] and key not in selected:
            selected[key]=r
    roots=[{'scene':s,'case':c,'step':r['step'],'partition':r['partition'],
            'outcome':r['outcome']} for (s,c),r in selected.items()]
    write('counterfactual-protocol.json',{
        'source_ledger_sha256':hashlib.sha256((HERE/'nominal-ledger.json').read_bytes()).hexdigest(),
        'selection':'First preregistered value-overrides-progress root per episode, all qualifying successful and failed episodes; no winner selection',
        'roots':roots,
        'arms':['saved nominal suffix','one-step-progress','value-free-progress'],
        'one-step-progress':'Change only the first primitive to maximum goal progress within the SAME current safe candidate mask; then exact frozen nominal',
        'value-free-progress':'At every remaining step, maximum goal progress within SAME current safe candidate mask; if mask empty, exact nominal fallback; no training and no future/goal-of-human information',
        'unchanged':['80 actions','executed smoothing','native reward','time remaining','human dynamics','observation','native termination','geometry accounting'],
        'decisions':{
            'single_action_loss':'Better complete native return/outcome under one-step replacement supplies a same-root counterexample to current action ranking; repeated cases required, not proof the replacement is globally better',
            'repeated_policy_loss':'Value-free improvement without single-step improvement does NOT prove one-step action misranking; replacement changes the whole continuation policy',
            'simple_explanation':'If value-free geometry absorbs useful headroom, do not recommend a complex new algorithm on this evidence',
            'unresolved':'If both controls damage waiting or no consistent task benefit, preserve value/progress disagreement as unresolved rather than claim a defect',
            'no_method_development':'These are causal diagnostics and known simple controls, not Ours'},
        'recurrence_rule_before_case_level_check':'At least two adjacent sampled roots, exactly8 native steps apart, both value-overrides-progress and all80 candidates safe. Test whether it concentrates timeouts in discovery and mechanism-check subsets; descriptive rule, not a learned detector.',
        'limits':'All cases are the already observed development cohort. Mechanism-check is not fresh independent navigation confirmation.'})
    print('frozen counterfactual roots',len(roots),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('command',choices=['recovery-offline','recovery-roots','history-channel','nominal-ledger',
                                    'recovery-contract','freeze-counterfactuals','counterfactuals'])
    args=ap.parse_args()
    if args.command=='recovery-offline':
        offline_recovery()
    elif args.command=='recovery-roots':
        recovery_roots()
    elif args.command=='history-channel':
        history_channel()
    elif args.command=='nominal-ledger':
        nominal_ledger()
    elif args.command=='recovery-contract':
        recovery_contract()
    elif args.command=='counterfactuals':
        counterfactuals()
    elif args.command=='freeze-counterfactuals':
        freeze_counterfactuals()
