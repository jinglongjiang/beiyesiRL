"""CPU-only unit checks for the terminal repair; no training or episode study."""
import configparser
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import crowd_nav
import crowd_sim
from crowd_nav import contracts
from crowd_nav.policy import mamba_rl
from crowd_sim.envs.crowd_sim import CrowdSim
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.state import FullState, JointState, ObservableState
from crowd_sim.envs.utils.action import ActionXY

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'recovery-duration'
FINDINGS = HERE.parent / 'navigation-findings'


def main():
    for module in [crowd_nav, crowd_sim, mamba_rl]:
        assert Path(module.__file__).resolve().is_relative_to(ROOT)
    cfg = configparser.ConfigParser()
    assert cfg.read(OLD / 'parent-config.ini')
    contracts.init_grid_from_cfg(cfg)
    assert contracts.grid_action_dim(contracts.GRID) == 80
    torch.set_num_threads(2)
    policy = mamba_rl.MambaRLPolicy(cfg, 'cpu').eval()
    policy.set_phase('test')
    policy.time_step = .25
    policy.build_action_space(1.)
    assert len(policy.action_space) == 80
    weight = OLD / 'artifacts' / 'nominal.pth'
    sha = hashlib.sha256(weight.read_bytes()).hexdigest()
    assert sha == json.loads((OLD / 'frozen-manifest.json').read_text())['nominal_sha256']
    policy.load_state_dict(torch.load(weight, map_location='cpu'))
    env = CrowdSim()
    env.configure(cfg)
    robot = Robot(cfg, 'robot')
    robot.set_policy(policy)
    env.set_robot(robot)
    policy.set_env(env)
    env.phase = 'test'
    env.reset(options={'test_case':100})
    checks = []

    def fixture(goal, human=None, time=0.):
        robot.set(0, 0, goal, 0, 0, 0, 0, radius=.3, v_pref=1.)
        env.global_time = time
        for i, h in enumerate(env.humans):
            h.set(50+i, 50, 60+i, 50, 0, 0, 0, radius=.3)
            h.act = lambda observation: ActionXY(0, 0)
        if human is not None:
            env.humans[0].set(*human, radius=.3)
            velocity = env.humans[0].vx, env.humans[0].vy
            env.humans[0].act = lambda observation: ActionXY(*velocity)
        return JointState(robot.get_full_state(), [h.get_observable_state() for h in env.humans])

    for name, goal, human, time, expected in [
        ('arrival_annulus', .275, None, 0., 'reach_goal'),
        ('strict_radius_boundary', .3, None, 0., 'nothing'),
        ('inside_radius_boundary', .299999, None, 0., 'reach_goal'),
        ('swept_collision_endpoint_clear', 10., (.8,0.,10.,0.,-6.4,0.,0.), 0., 'collision'),
        ('collision_precedes_goal', .275, (.4,0.,10.,0.,0.,0.,0.), 0., 'collision'),
        ('timeout_precedes_goal_and_collision', .275, (.4,0.,10.,0.,0.,0.,0.), env.time_limit, 'timeout'),
    ]:
        state = fixture(goal, human, time)
        command = [ActionXY(0,0)]
        terminal, rewards = policy.candidate_terminal_outcomes(state, command)
        _, native_reward, term, trunc, info = env.step(command[0], update=False)
        assert info['event'] == expected, (name, info)
        assert terminal[0] == (term or trunc), name
        if terminal[0]:
            assert rewards[0] == native_reward, name
            policy.lookahead_ablation_mode = 'full'
            for value in [-1e12, 1e12]:
                score = policy.candidate_action_values(state, command, np.array([123.]), torch.tensor([value]))
                assert score.item() == native_reward, name
        checks.append({'name':name,'passed':True,'native_event':info['event']})

    state = fixture(10.)
    commands = [ActionXY(.05,0), ActionXY(.2,.1), ActionXY(-.1,0)]
    rewards, values = torch.tensor([-.01,0.,-.02]), torch.tensor([.3,.2,.1])
    for mode in ['full','reward_only','value_only']:
        policy.lookahead_ablation_mode = mode
        expected = rewards if mode == 'reward_only' else policy.gamma*values
        if mode == 'full':
            expected = rewards+policy.gamma*values
        assert torch.equal(policy.candidate_action_values(state, commands, rewards, values),expected)
    checks.append({'name':'all_nonterminal_scores_bitwise_unchanged_in_three_modes','passed':True})
    policy.lookahead_ablation_mode = 'full'
    ledger = json.loads((FINDINGS/'nominal-ledger.json').read_text())
    scores = np.load(FINDINGS/'artifacts'/'nominal-action-scores.npz')['scores']
    for scene, case, step, scene_id, goal in [('baseline_circle',100,56,0,4.),
                                             ('large_circle',128,96,4,6.)]:
        index = next(i for i,r in enumerate(ledger) if (r['scene'],r['case'],r['step'])==(scene,case,step))
        row, actions = ledger[index], scores[index]
        human = np.load(OLD/'artifacts'/'tapes'/f'{scene_id}-{case}.npz')['humans'][step]
        state = JointState(FullState(*row['position'],0.,0.,.3,0.,goal,1.,0.),
                           [ObservableState(*h) for h in human])
        env.global_time = step*.25
        command = [ActionXY(*a[:2]) for a in actions]
        new = policy.candidate_action_values(state,command,actions[:,3],torch.tensor(actions[:,4]))
        chosen = row['chosen_index']
        assert actions[chosen,3] == 0. and new[chosen] == 1.
        checks.append({'name':'saved_arrival_annulus_root','scene':scene,'case':case,'step':step,
                       'old_reward':0.,'new_terminal_score':float(new[chosen]),'passed':True})
    assert hashlib.sha256(weight.read_bytes()).hexdigest() == sha
    result = {'scope':'Targeted CPU fixtures and two saved root checks, not a benchmark or new algorithm',
              'grid_actions':80,'checkpoint_sha256':sha,'checkpoint_unchanged':True,
              'tests':checks,'passed':all(r['passed'] for r in checks),
              'not_changed':['reward config','actions','smoothing','endpoint safety filter','observation','model input','weights','training'],
              'limit':'Nonterminal discomfort still uses the existing endpoint approximation. This patch aligns terminal semantics only; it does not establish future safety or task benefit.'}
    (HERE/'terminal-contract-checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
