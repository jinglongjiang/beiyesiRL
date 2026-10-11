"""Trace fixed-action value preference through saved learning checkpoints."""
import hashlib
import json

import numpy as np
import torch

import run as r


def main():
    cfg, policy = r.setup()
    roots = json.loads((r.FINDINGS/'counterfactual-protocol.json').read_text())['roots']
    contrast = json.loads((r.HERE/'value-contrasts.json').read_text())
    windows, metadata = [], []
    for root, fixed in zip(roots,contrast):
        scene, case = r.m.SCENES.index(root['scene']), root['case']
        tape = r.load_tape(scene,case)
        env,snap = r.frozen_root(cfg,policy,root,tape)
        part = r.decompose(policy,env)
        prefix = list(snap['history'])+[part['token']]
        pairs = []
        for index in [fixed['selected_index'],fixed['forward_index']]:
            one = r.environment(cfg,policy,scene,case,tape,snap)
            action = part['commands'][index]
            r.m.commit_action(policy,part['token'],action,index)
            _,reward,term,trunc,_ = one.step(action)
            pairs.append(r.m.window(prefix+[r.m.token(policy,r.m.state(one))]))
        windows.append(pairs)
        robot = snap['robot']
        metadata.append({'scene':root['scene'],'case':case,'step':root['step'],
                         'goal':[robot.gx,robot.gy],'position':[robot.px,robot.py],
                         'overshot_goal_y':bool(robot.py>robot.gy),
                         'humans':len(env.humans)})
    x = np.asarray(windows,np.float32)
    path = r.OUT/'fixed-actual-next-windows.npz'
    np.savez_compressed(path,windows=x)
    stages = {}
    checkpoints = [('IL','gru-il.pth')]+[(f'RL-{n}',f'gru-rl-{n}.pth') for n in [500,1000,1500,2000]]
    for name,filename in checkpoints:
        weight = r.OLD/'artifacts'/filename
        policy.load_state_dict(torch.load(weight,map_location=policy.device))
        policy.eval()
        values=[]
        with torch.no_grad():
            for pair in x:
                values.append(policy.forward_candidate_values(torch.as_tensor(pair,device=policy.device)).cpu().numpy())
        values = np.asarray(values)
        stages[name] = {'checkpoint_sha256':hashlib.sha256(weight.read_bytes()).hexdigest(),
                        'selected_value_above_forward':int((values[:,0]>values[:,1]+1e-6).sum()),
                        'median_gap':float(np.median(values[:,0]-values[:,1])),
                        'values':values.tolist()}
    demonstrations = torch.load(r.OLD/'artifacts'/'corrected-orca.pth',map_location='cpu')
    robot=np.concatenate([np.asarray(t)[:,0,:] for t,_ in demonstrations])
    goal_vector=robot[:,5:7]-robot[:,:2]
    progress_velocity=(goal_vector*robot[:,2:4]).sum(1)/np.maximum(np.linalg.norm(goal_vector,axis=1),1e-12)
    baseline=json.loads((r.HERE/'repaired-baseline.json').read_text())
    flagged=[(episode,root) for episode in baseline for root in episode['sampled_roots'] if root['flag']]
    overshot=[(episode,root) for episode,root in flagged
              if root['position'][1]>(6 if episode['scene']=='large_circle' else 4)]
    curves=json.loads((r.OLD/'artifacts'/'training-curves.json').read_text())
    recent=[c for c in curves if c.get('stage')=='RL'][-200:]
    evidence={'fixed_root_metadata':metadata,'checkpoints':stages,
              'IL_support':{'episodes':len(demonstrations),'frames':len(robot),
                            'unique_robot_goal_coordinates':np.unique(robot[:,5:7],axis=0).tolist(),
                            'robot_above_own_goal_frames':int((robot[:,1]>robot[:,6]).sum()),
                            'negative_goal_progress_velocity_frames':int((progress_velocity<0).sum()),
                            'y_range':[float(robot[:,1].min()),float(robot[:,1].max())]},
              'historical_RL_last200':{'successes':sum(c['outcome']=='reach_goal' for c in recent),
                                      'collisions':sum(c['outcome']=='collision' for c in recent),
                                      'timeouts':sum(c['outcome']=='timeout' for c in recent),
                                      'mean_MSE':float(np.mean([c['loss'] for c in recent]))},
              'goal_overshoot_in_repaired_baseline':{
                  'negative_progress_roots':len(flagged),'above_goal_negative_roots':len(overshot),
                  'affected_episodes':len({(e['scene'],e['case']) for e,_ in overshot}),
                  'affected_timeout_episodes':len({(e['scene'],e['case']) for e,_ in overshot if e['timeout']})},
              'limits':['Checkpoint acquisition timing is not a causal test of any particular training factor.',
                        'IL support is not the whole RL support. Historical RL replay tokens were not saved.',
                        'No time or hidden future state was added to the policy.']}
    r.save('training-evidence.json',evidence)
    print(json.dumps({k:v for k,v in evidence.items() if k not in ['checkpoints','fixed_root_metadata']},indent=2),flush=True)
    print({k:{a:b for a,b in v.items() if a!='values'} for k,v in stages.items()},flush=True)


if __name__=='__main__':
    main()
