"""One shared terminal-value correction; the planner and controller are unchanged."""
import importlib.util
from pathlib import Path

import torch

path = Path(__file__).resolve().parents[1] / 'progress-contract/controller.py'
spec = importlib.util.spec_from_file_location('original_progress_contract_controller', path)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


class Controller(base.Controller):
    def __init__(self, arm, policy):
        super().__init__(arm, policy)
        self.optimizer.terminal_state_cost = self.terminal_value_cost

    def free_value(self, distance):
        return self.policy.success_reward * .99 ** (torch.clamp_min(distance - self.radius, 0)
                                                     / (self.policy.env.robot.v_pref * .25))

    def terminal_value_cost(self, states, actions):
        positions = states[0, :, :, :2]
        distances = torch.linalg.vector_norm(positions - self.goal, dim=-1)
        done = (distances < self.radius).any(1)
        done |= self.start_time + 15 * .25 >= 50. - 1e-6
        return -(.99 ** 16) * self.free_value(distances[:, -1]) * (~done).double()

    def evaluate(self, plans, state):
        cost, clear, distance, indices = super().evaluate(plans, state)
        done = (distance < self.radius).any(1)
        done |= self.start_time + 15 * .25 >= 50. - 1e-6
        cost -= (.99 ** 16) * self.free_value(distance[:, -1]) * (~done).double()
        return cost, clear, distance, indices
