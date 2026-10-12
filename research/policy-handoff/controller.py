"""Policy-specific re-entry test, with an ordinary rollout ablation."""
from collections import deque
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

path = Path(__file__).resolve().parents[1] / 'progress-contract-v2/controller.py'
spec = importlib.util.spec_from_file_location('shared_handoff_planner', path)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


class Controller(base.Controller):
    def __init__(self, arm, policy):
        super().__init__('hold8' if arm == 'hold8' else arm, policy)
        self.cooldown = 0
        self.handoff_rejections = self.tail_queries = self.handoffs = 0

    def token_at(self, state):
        from crowd_nav.contracts import _batch_joint34_to_tokens_vectorized
        return _batch_joint34_to_tokens_vectorized(
            self.policy._build_joint_state_34(state.self_state, state.human_states)[None])[0]

    def legal_state(self, state):
        from crowd_sim.envs.utils.state import FullState, ObservableState, JointState
        p = self.policy
        robot = FullState(*state[:4].tolist(), self.radius, *self.goal.tolist(),
                          p.env.robot.v_pref, p.env.robot.theta)
        hs = self.humans.clone()
        hs[:, :2] += max(float(state[4]), 0.) * .25 * hs[:, 2:4]
        return JointState(robot, [ObservableState(*v) for v in hs.tolist()])

    def tail(self, plan, length, initial):
        p = self.policy
        history, previous, env = p._history, p._last_action, p.env
        from crowd_sim.envs.utils.action import ActionXY
        current = initial.clone()
        cost, safe, done = 0., True, False
        tail_start = None
        commands = []
        p._history = deque((v.copy() for v in history), maxlen=history.maxlen)
        try:
            for j in range(length + 8):
                legal = self.legal_state(current)
                p._last_action = None if current[4] < 0 else ActionXY(*current[2:4].tolist())
                if j < length:
                    raw = plan[j]
                    token = self.token_at(legal)
                else:
                    if tail_start is None:
                        tail_start = float(torch.linalg.vector_norm(current[:2] - self.goal))
                    p.env = SimpleNamespace(global_time=self.start_time + j * .25,
                                            time_limit=50., timeout_penalty=env.timeout_penalty,
                                            robot=env.robot)
                    candidates, token, windows, reward, clear = p._score_candidates_vectorized(legal)
                    with torch.no_grad():
                        values = p.forward_candidate_values(windows)
                        q = p.candidate_action_values(legal, candidates,
                            torch.tensor(reward.tolist(), device=p.device), values)
                        d = torch.tensor(clear.tolist(), device=p.device)
                        scores = q - .8 * torch.clamp(.2 - d, min=0)
                        if (d >= .2).any():
                            scores = scores.masked_fill(d < .2, -1e9)
                        scores[0] -= .001
                    raw = self.raw[int(scores.argmax())]
                    self.tail_queries += 1
                p._history.append(token)
                current = self.dynamics(current[None], raw[None])[0]
                clear = float(self.clearance(current[None])[0])
                safe &= clear >= .2
                cost += float(self.cost(current[None], raw[None])[0])
                commands.append(raw.clone())
                distance = float(torch.linalg.vector_norm(current[:2] - self.goal))
                done = distance < self.radius or self.start_time + j * .25 >= 50. - 1e-6
                if done:
                    break
            if not done:
                cost -= float(.99 ** len(commands) * self.free_value(torch.tensor(distance)))
            gain = 0. if tail_start is None else tail_start - distance
            admitted = safe and (gain >= .2 or distance < self.radius)
            return cost, admitted, gain, torch.stack(commands), safe
        finally:
            p._history, p._last_action, p.env = history, previous, env

    def select(self, part, position, goal, humans, previous, radius, seed, now=0.):
        if self.arm == 'hold8':
            return super().select(part, position, goal, humans, previous, radius, seed, now)
        native = part['best']
        distance = float(np.linalg.norm(np.asarray(position) - goal))
        self.distances.append(distance)
        self.start_time = float(now)
        self.goal = torch.as_tensor(goal, dtype=torch.float64)
        self.humans = torch.as_tensor(humans, dtype=torch.float64)
        self.radius = float(radius)
        state = torch.tensor([*position, *(previous if previous is not None else [0., 0.]),
                              0. if previous is not None else -1.], dtype=torch.float64)
        diagnostic = {'active': False, 'triggered': False, 'remaining': self.remaining,
                      'handoff_options': [], 'admission_changed_choice': False, 'fallback': False}
        if not part['safe'].any():
            self.plan, self.remaining, self.cooldown = None, 0, 0
            return native, diagnostic
        if self.remaining:
            padded = torch.cat((self.plan, self.plan[-1:].repeat(16 - len(self.plan), 1)))
            _, clear, _, indices = self.evaluate(padded[None], state)
            index = int(indices[0, 0])
            if (clear[0, :self.remaining] >= .2).all() and part['safe'][index]:
                self.plan = self.plan[1:]
                self.remaining -= 1
                self.active_steps += 1
                self.changes += int(index != native)
                self.retained += 1
                diagnostic.update(active=True, remaining=self.remaining)
                return index, diagnostic
            self.invalidations += 1
            self.plan, self.remaining = None, 0
        if self.cooldown:
            self.cooldown -= 1
            return native, diagnostic
        if len(self.distances) < 9 or self.distances[-9] - distance >= .2:
            return native, diagnostic
        self.triggers += 1
        diagnostic['triggered'] = True
        direction = self.raw[torch.argmax(self.raw @ (self.goal - state[:2]))]
        self.optimizer.U = direction.repeat(16, 1)
        torch.manual_seed(int(seed))
        with torch.no_grad():
            self.optimizer.command(state, shift_nominal_trajectory=False)
            plans = torch.cat((self.optimizer.perturbed_action, self.optimizer.U[None]))
            costs, clear, _, indices = self.evaluate(plans, state)
        valid = (clear >= .2).all(1) & torch.from_numpy(part['safe'])[indices[:, 0]]
        if not valid.any():
            self.invalidations += 1
            return native, diagnostic
        chosen_plan = self.raw[indices[int(torch.where(valid, costs, torch.inf).argmin())]]
        options = [self.tail(chosen_plan, d, state) for d in [0, 4, 8, 16]]
        ordinary = min(range(4), key=lambda k: options[k][0] if options[k][4] else float('inf'))
        admissible = [k for k in range(4) if options[k][1]]
        choice = ordinary
        if self.arm == 'handoff':
            self.handoff_rejections += 4 - len(admissible)
            choice = min(admissible, key=lambda k: options[k][0]) if admissible else 2
            diagnostic['fallback'] = not bool(admissible)
        elif not any(v[4] for v in options):
            choice = 2
            diagnostic['fallback'] = True
        diagnostic['handoff_options'] = [dict(prefix=d, cost=float(v[0]), admitted=bool(v[1]),
            tail_progress=float(v[2]), safe=bool(v[4])) for d, v in zip([0, 4, 8, 16], options)]
        diagnostic['admission_changed_choice'] = bool(choice != ordinary)
        length = [0, 4, 8, 16][choice]
        diagnostic['chosen_prefix'] = length
        if not length:
            self.handoffs += 1
            self.cooldown = 7
            return native, diagnostic
        index = int(self.project(chosen_plan[:1])[0])
        self.plan = chosen_plan[1:length].clone()
        self.remaining = length - 1
        self.active_steps += 1
        self.changes += int(index != native)
        diagnostic.update(active=True, remaining=self.remaining)
        assert part['safe'][index]
        return index, diagnostic
