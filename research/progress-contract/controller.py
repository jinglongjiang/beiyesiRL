"""Pinned MPPI reuse with discrete executed-action and progress-deadline contracts."""
import numpy as np
import torch
from pytorch_mppi import MPPI
from pytorch_mppi.mppi import SpecificActionSampler


class Library(SpecificActionSampler):
    def __init__(self, owner):
        super().__init__()
        self.owner = owner

    def sample_trajectories(self, state, info):
        o = self.owner
        constant = o.raw[:, None, :].repeat(1, 16, 1)
        turn = constant.clone()
        delta = o.goal - state[:2]
        idx = torch.argmax(o.raw @ delta)
        turn[:, 4:] = o.raw[idx]
        return torch.cat((constant, turn))


class Controller:
    def __init__(self, arm, policy):
        self.arm, self.policy = arm, policy
        self.raw = torch.tensor(np.asarray(policy.action_space), dtype=torch.float64)
        assert self.raw.shape == (80, 2)
        self.goal = torch.zeros(2, dtype=torch.float64)
        self.humans = None
        self.radius = None
        self.start_time = 0.
        self.distances = []
        self.remaining = self.triggers = self.active_steps = self.changes = 0
        self.rejections = self.retained = self.invalidations = self.completed = 0
        self.anchor = 0.
        self.plan = None
        self.held = 0
        self.optimizer = MPPI(self.dynamics, self.cost, nx=5,
                              noise_sigma=torch.eye(2, dtype=torch.float64) * .25,
                              num_samples=256, horizon=16, lambda_=.01,
                              device='cpu', u_min=self.raw.min(0).values,
                              u_max=self.raw.max(0).values,
                              terminal_state_cost=lambda states, actions: torch.zeros(states.shape[1], dtype=states.dtype),
                              specific_action_sampler=Library(self))

    def project(self, action):
        return ((action[:, None] - self.raw[None]) ** 2).sum(-1).argmin(-1)

    def dynamics(self, state, action):
        index = self.project(action)
        executed = .3 * state[:, 2:4] + .7 * self.raw[index]
        # At the initial frame the native controller has no previous action.
        executed = torch.where((state[:, 4:5] < 0), self.raw[index], executed)
        clock = torch.clamp_min(state[:, 4:5], 0) + 1
        return torch.cat((state[:, :2] + .25 * executed, executed, clock), -1)

    def clearance(self, state):
        predicted = self.humans[None, :, :2] + state[:, 4:5, None] * .25 * self.humans[None, :, 2:4]
        return (torch.linalg.vector_norm(state[:, None, :2] - predicted, dim=-1)
                - self.radius - self.humans[None, :, 4]).min(-1).values

    def cost(self, state, action):
        p = self.policy
        distance = torch.linalg.vector_norm(state[:, :2] - self.goal, dim=-1)
        olddistance = torch.linalg.vector_norm(state[:, :2] - .25 * state[:, 2:4] - self.goal, dim=-1)
        old_robot = state[:, :2] - .25 * state[:, 2:4]
        relative = (self.humans[None, :, :2] + (state[:, 4:5, None] - 1) * .25
                    * self.humans[None, :, 2:4] - old_robot[:, None])
        segment = .25 * (self.humans[None, :, 2:4] - state[:, None, 2:4])
        fraction = torch.clamp(-(relative * segment).sum(-1) / torch.clamp_min((segment ** 2).sum(-1), 1e-15), 0, 1)
        clear = (torch.linalg.vector_norm(relative + fraction[..., None] * segment, dim=-1)
                 - self.radius - self.humans[None, :, 4]).min(-1).values
        reward = p.progress_reward * (olddistance - distance) + p.time_penalty
        reward -= torch.clamp(p.discomfort_dist - clear, min=0) * p.discomfort_penalty_factor * .25
        reward += torch.where(torch.linalg.vector_norm(state[:, 2:4], dim=-1) < .05, p.stand_penalty, 0.)
        reward = torch.where(distance < self.radius, p.success_reward, reward)
        reward = torch.where(clear < 0, p.collision_penalty, reward)
        reward = torch.where(self.start_time + (state[:, 4] - 1) * .25 >= 50. - 1e-6,
                             float(p.env.timeout_penalty), reward)
        # Large rejection cost assists search; final admission uses a hard mask.
        return -reward * (.99 ** (state[:, 4] - 1)) + (self.clearance(state) < .2).double() * 1000

    def evaluate(self, plans, state):
        current = state[None].repeat(len(plans), 1)
        costs, clearances, distances, indices = [], [], [], []
        reached = torch.zeros(len(plans), dtype=torch.bool)
        for t in range(16):
            action = plans[:, t]
            index = self.project(action)
            current = self.dynamics(current, action)
            clear = self.clearance(current)
            distance = torch.linalg.vector_norm(current[:, :2] - self.goal, dim=-1)
            costs.append(torch.where(reached, 0., self.cost(current, action)))
            clearances.append(torch.where(reached, torch.inf, clear))
            distances.append(torch.where(reached, torch.clamp_max(distance, self.radius), distance))
            indices.append(index)
            reached |= ((distance < self.radius) & (clear >= 0)) | (self.start_time + t * .25 >= 50. - 1e-6)
        return (torch.stack(costs, 1).sum(1), torch.stack(clearances, 1),
                torch.stack(distances, 1), torch.stack(indices, 1))

    def select(self, part, position, goal, humans, previous, radius, seed, now=0.):
        distance = float(np.linalg.norm(np.asarray(position) - goal))
        self.distances.append(distance)
        native = part['best']
        diagnostic = {'active': False, 'remaining': self.remaining, 'certified': 0,
                      'retained': False, 'predicted_repayment': None, 'triggered': False}
        if not part['safe'].any():
            self.remaining = self.held = 0
            self.plan = None
            return native, diagnostic
        if not self.remaining and len(self.distances) >= 9 and self.distances[-9] - distance < .2:
            self.remaining = 8 if self.arm == 'low-commit8' else 16
            self.anchor = distance
            self.triggers += 1
            self.plan = None
            self.held = 0
            diagnostic['triggered'] = True
        if not self.remaining:
            return native, diagnostic
        if self.arm == 'low-commit8':
            index = part['forward']
        else:
            self.start_time = float(now)
            self.goal = torch.as_tensor(goal, dtype=torch.float64)
            self.humans = torch.as_tensor(humans, dtype=torch.float64)
            self.radius = float(radius)
            state = torch.tensor([*position, *(previous if previous is not None else [0., 0.]),
                                  0. if previous is not None else -1.], dtype=torch.float64)
            prior = None
            if self.plan is not None and len(self.plan):
                prior = torch.cat((self.plan, self.plan[-1:].repeat(16 - len(self.plan), 1)))
            torch.manual_seed(int(seed))
            direction = self.raw[torch.argmax(self.raw @ (self.goal - state[:2]))]
            self.optimizer.U = prior.clone() if prior is not None else direction.repeat(16, 1)
            with torch.no_grad():
                self.optimizer.command(state, shift_nominal_trajectory=False)
                plans = torch.cat((self.optimizer.perturbed_action, self.optimizer.U[None]))
                if prior is not None:
                    plans = torch.cat((plans, prior[None]))
                costs, clear, distances, indices = self.evaluate(plans, state)
            valid = (clear >= .2).all(1)
            valid &= torch.from_numpy(part['safe'])[indices[:, 0]]
            if self.arm == 'deadline':
                valid &= distances[:, self.remaining - 1] <= max(self.anchor - .5, self.radius)
                self.rejections += int((clear >= .2).all(1).sum() - valid.sum())
            diagnostic['certified'] = int(valid.sum())
            if not valid.any():
                self.invalidations += 1
                self.remaining = self.held = 0
                self.plan = None
                return native, diagnostic
            chosen = int(torch.where(valid, costs, torch.inf).argmin())
            if self.arm == 'hold8' and self.held and prior is not None and valid[-1]:
                chosen = len(plans) - 1
            if chosen == len(plans) - 1 and prior is not None:
                self.retained += 1
                diagnostic['retained'] = True
            selected = self.raw[indices[chosen]]
            index = int(indices[chosen, 0])
            self.plan = selected[1:].clone()
            if self.arm == 'hold8':
                self.held = self.held - 1 if self.held else 7
            diagnostic['predicted_repayment'] = float(distance - distances[chosen, min(self.remaining, 16) - 1])
        assert part['safe'][index]
        self.remaining -= 1
        if not self.remaining:
            self.completed += int(self.arm == 'deadline' and self.anchor - distance >= .5)
            self.plan = None
        self.active_steps += 1
        self.changes += int(index != native)
        diagnostic.update(active=True, remaining=self.remaining)
        return index, diagnostic
