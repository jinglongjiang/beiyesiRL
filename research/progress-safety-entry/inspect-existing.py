"""Read existing roots and continuations; no new rollout, predictor or learner."""
import configparser
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from crowd_nav import contracts
from crowd_nav.policy.mamba_rl import MambaRLPolicy
from crowd_sim.envs.utils.action import ActionXY
from crowd_sim.envs.utils.state import FullState, ObservableState, JointState

HERE = Path(__file__).resolve().parent
DATA = HERE.parent/'navigation-findings'
OLD = HERE.parent/'recovery-duration'
SCENES = ['baseline_circle','baseline_square','dense_circle','dense_square','large_circle','large_square']


def read(path):
    return json.loads(path.read_text())


def distribution(x):
    return {'n':len(x),'mean':float(np.mean(x)),'p25_p50_p75':np.percentile(x,[25,50,75]).tolist()} if len(x) else {'n':0}


def main():
    cfg = configparser.ConfigParser()
    assert cfg.read(OLD/'parent-config.ini')
    contracts.init_grid_from_cfg(cfg)
    assert contracts.grid_action_dim(contracts.GRID) == 80
    torch.set_num_threads(2)
    policy = MambaRLPolicy(cfg,'cpu').eval()
    policy.time_step = .25
    policy.lookahead_ablation_mode = 'full'
    ledger = read(DATA/'nominal-ledger.json')
    matrix = np.load(DATA/'artifacts'/'nominal-action-scores.npz')['scores']
    continuations = read(DATA/'counterfactual-results.json')
    frozen = read(DATA/'counterfactual-protocol.json')
    assert hashlib.sha256((DATA/'nominal-ledger.json').read_bytes()).hexdigest()==frozen['source_ledger_sha256']
    lookup = {(r['scene'],r['case'],r['step']):i for i,r in enumerate(ledger)}
    original = {(r['scene'],r['case']):r for r in read(OLD/'results'/'development-results.json') if r['arm']=='nominal'}
    cached = {}
    roots = []
    fixed_scores = {}
    for i,(row,actions) in enumerate(zip(ledger,matrix)):
        scene_id=SCENES.index(row['scene'])
        key=row['scene'],row['case']
        if key not in cached:
            tape=np.load(OLD/'artifacts'/'tapes'/f'{scene_id}-{row["case"]}.npz')['humans']
            trace=np.load(OLD/'artifacts'/'development-traces'/'nominal'/f'{scene_id}-{row["case"]}.npz')['records']
            cached[key]=tape,trace
        tape,trace=cached[key]
        goal=-trace[0,1:3]
        assert abs(np.linalg.norm(np.array(row['position'])-goal)-row['goal_distance'])<1e-9
        state=JointState(FullState(*row['position'],0.,0.,row['native_radius'],*goal,1.,0.),
                         [ObservableState(*h) for h in tape[row['step']]])
        context=SimpleNamespace(global_time=row['step']*.25,time_limit=25.,timeout_penalty=-.5)
        commands=[ActionXY(*a[:2]) for a in actions]
        with torch.no_grad():
            # Cached values originated as float32; retain the historical evaluator's arithmetic.
            repaired=policy.candidate_action_values(state,commands,actions[:,3],
                         torch.tensor(actions[:,4],dtype=torch.float32),env=context).numpy()
        repaired-=actions[:,6]
        repaired[0]-=.001
        if actions[:,8].sum()<80:
            repaired[actions[:,8]==0]=-1e9
        fixed_scores[i]=repaired
        tail=trace[trace[:,0]>=row['step']]
        observed=float(tail[:,7]@(.99**np.arange(len(tail))))
        chosen=row['chosen_index']
        roots.append({'scene':row['scene'],'case':row['case'],'step':row['step'],
                      'outcome':row['outcome'],'safe_count':row['safe_count'],
                      'selected_progress':row['chosen_progress'],
                      'value_overrides_progress':row['value_overrides_progress'],
                      'original_index':chosen,'repaired_index':int(repaired.argmax()),
                      'original_task_score':float(actions[chosen,3]+np.float32(.99)*np.float32(actions[chosen,4])),
                      'realized_nominal_suffix_return':observed})
    cases=[]
    by_variant={(r['scene'],r['case'],r['variant']):r for r in continuations}
    for root in frozen['roots']:
        scene,case,step=root['scene'],root['case'],root['step']
        i=lookup[(scene,case,step)]
        row,actions=ledger[i],matrix[i]
        one=by_variant[(scene,case,'one-step-progress')]
        full=by_variant[(scene,case,'value-free-progress')]
        traces={v:np.load(DATA/'artifacts'/f'counterfactual-{v}-{SCENES.index(scene)}-{case}.npz')['records']
                for v in ['one-step-progress','value-free-progress']}
        np.testing.assert_array_equal(traces['one-step-progress'][0,:6],traces['value-free-progress'][0,:6])
        label='unchanged_outcome'
        if not original[(scene,case)]['success'] and one['result']['success']:
            label='one_step_rescue'
        elif original[(scene,case)]['success'] and not one['result']['success']:
            label='one_step_damage'
        cases.append({'scene':scene,'case':case,'step':step,'partition':root['partition'],
                      'label':label,'nominal_outcome':root['outcome'],
                      'one_step_outcome':one['result']['outcome'],'continuous_outcome':full['result']['outcome'],
                      'continuous_contact':full['result']['contact'],
                      'same_first_executed_action':True,
                      'one_step_first_clearance':float(traces['one-step-progress'][0,8]),
                      'predicted_alternative_minus_selected_score':float(actions[row['progress_index'],7]-actions[row['chosen_index'],7]),
                      'realized_one_step_return_advantage':one['return_advantage'],
                      'terminal_repair_changes_root_choice':int(fixed_scores[i].argmax())!=row['chosen_index'],
                      'safe_count':row['safe_count'],'selected_progress':row['chosen_progress'],
                      'alternative_progress':row['max_safe_progress']})
    groups={}
    for outcome in ['reach_goal','timeout']:
        for flag in [False,True]:
            z=[r for r in roots if r['outcome']==outcome and r['value_overrides_progress']==flag]
            groups[f'{outcome} / progress_override={flag}']={
                'roots':len(z),
                'score_minus_realized_return':distribution([r['original_task_score']-r['realized_nominal_suffix_return'] for r in z])}
    rescues=[r for r in cases if r['label']=='one_step_rescue']
    damages=[r for r in cases if r['label']=='one_step_damage']
    changed=[r for r in roots if r['original_index']!=r['repaired_index']]
    summaries={}
    for variant in ['one-step-progress','value-free-progress']:
        selected=[r for r in continuations if r['variant']==variant]
        summaries[variant]={
            'roots':len(selected),
            'successes':sum(r['result']['success'] for r in selected),
            'collisions':sum(r['result']['collision'] for r in selected),
            'contacts':sum(r['result']['contact'] for r in selected),
            'timeouts':sum(r['result']['timeout'] for r in selected),
            'rescues':sum(r['outcome']!='reach_goal' and r['result']['success'] for r in selected),
            'damages':sum(r['outcome']=='reach_goal' and not r['result']['success'] for r in selected)}
    flagged=[r for r in ledger if r['value_overrides_progress']]
    value_comparisons={
        'flagged_roots':len(flagged),
        'equal_immediate_reward':sum(r['chosen_reward']==r['progress_reward'] for r in flagged),
        'zero_risk_both':sum(r['chosen_risk']==0 and r['progress_risk']==0 for r in flagged),
        'strictly_higher_selected_value':sum(r['chosen_value']>r['progress_value'] for r in flagged),
        'all_80_candidates_safe':sum(r['safe_count']==80 for r in flagged)}
    sources=[DATA/'nominal-ledger.json',DATA/'artifacts'/'nominal-action-scores.npz',
             DATA/'counterfactual-results.json',DATA/'counterfactual-protocol.json',
             OLD/'results'/'development-results.json',OLD/'parent-config.ini',
             ROOT/'crowd_nav'/'policy'/'mamba_rl.py']
    result={
        'scope':'Existing saved development cohort only; no new rollout, training or prediction experiment',
        'root_count':len(roots),'case_roots':len(cases),
        'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        'saved_counterfactual_outcomes':summaries,
        'saved_value_progress_comparisons':value_comparisons,
        'terminal_repair_cached_choice_changes':len(changed),
        'terminal_repair_change_rows':changed,
        'terminal_repair_changes_counterfactual_roots':sum(r['terminal_repair_changes_root_choice'] for r in cases),
        'realized_one_step_better_count':sum(r['realized_one_step_return_advantage']>1e-8 for r in cases),
        'realized_one_step_worse_count':sum(r['realized_one_step_return_advantage']<-1e-8 for r in cases),
        'realized_one_step_equal_count':sum(abs(r['realized_one_step_return_advantage'])<=1e-8 for r in cases),
        'one_step_rescues':rescues,'one_step_damages':damages,
        'all_one_step_damage_cases_continuous_success':all(r['continuous_outcome']=='reach_goal' for r in damages),
        'first_step_contacts_in_all_paired_cases':sum(r['one_step_first_clearance']<0 for r in cases),
        'continuous_contacts_from_original_success':sum(r['continuous_contact'] and r['nominal_outcome']=='reach_goal' for r in cases),
        'continuous_contacts_from_original_timeout':sum(r['continuous_contact'] and r['nominal_outcome']=='timeout' for r in cases),
        'root_score_residuals':groups,
        'cases':cases,
        'limitations':[
            'Observed suffix returns condition on the simulator world; they are not expected-value calibration under an inferred distribution.',
            'The two alternative policies agree on their first executed action, then differ for the entire continuation.',
            'One harmful forward alternative does not prove every forward action is unsafe or that retreat is physically necessary.',
            'Cached repaired ranking holds learned successor values fixed. Historical complete outcomes remain pre-repair outcomes, not outcomes of the repaired policy.',
            'No intervention isolated value training from successor-model or history-representation error.'
        ]}
    (HERE/'existing-mechanism-evidence.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['cases','terminal_repair_change_rows','one_step_rescues','one_step_damages']},indent=2))


if __name__=='__main__':
    main()
