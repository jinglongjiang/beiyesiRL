"""TEB-inspired side preference adapted to the existing holonomic action grid.

The one-sided direction penalty follows EdgePreferRotDir. This is not the
TEB optimizer or a new algorithm; see THIRD-PARTY-NOTICES.txt and protocol.json.
"""
import numpy as np


def side_geometry(position, goal, velocity, commands, scores, safe):
    delta = np.asarray(goal) - np.asarray(position)
    axis = delta / max(float(np.linalg.norm(delta)), 1e-12)
    cross = axis[0] * velocity[1] - axis[1] * velocity[0]
    if abs(cross) <= 1e-6:
        lateral = axis[0] * commands[:, 1] - axis[1] * commands[:, 0]
        eligible = safe & (np.abs(lateral) > 1e-6)
        if eligible.any():
            index = int(np.where(eligible, scores, -np.inf).argmax())
            cross = lateral[index]
    side = float(np.sign(cross)) if abs(cross) > 1e-6 else 0.
    return axis, side


def directional_penalty(axis, side, commands, scores, safe):
    commands = np.asarray(commands)
    speed = np.linalg.norm(commands, axis=1)
    lateral = axis[0] * commands[:, 1] - axis[1] * commands[:, 0]
    angle = np.arcsin(np.clip(lateral / np.maximum(speed, 1e-12), -1., 1.))
    scale = float(np.std(scores[safe])) if safe.any() else 0.
    assert np.isfinite(scale)
    return scale * np.maximum(0., -side * angle) ** 2


class SideController:
    def __init__(self, latched=True):
        self.latched = latched
        self.negative = self.remaining = self.triggers = self.active_steps = self.changes = 0
        self.distances = []
        self.axis = np.zeros(2)
        self.side = 0.

    def select(self, part, position, goal, velocity):
        commands = np.asarray(part['commands'], dtype=np.float64)
        scores = np.asarray(part['scores'], dtype=np.float64)
        safe = part['safe']
        assert commands.shape == (80, 2) and safe.shape == (80,)
        assert np.isfinite(commands).all() and np.isfinite(scores).all()
        best = part['best']
        self.negative = self.negative + 1 if part['progress'][best] <= 0 else 0
        distance = float(np.linalg.norm(np.asarray(position) - goal))
        self.distances.append(distance)
        active = False
        penalty = np.zeros(80)
        if not safe.any():
            self.remaining = 0
            return best, penalty, active
        if not self.remaining and self.negative >= 8 and len(self.distances) >= 9 and self.distances[-9] <= distance:
            self.axis, self.side = side_geometry(position, goal, velocity, commands, scores, safe)
            self.remaining = 8
            self.negative = 0
            self.triggers += 1
        if self.remaining:
            if not self.latched:
                self.axis, self.side = side_geometry(position, goal, velocity, commands, scores, safe)
            penalty = directional_penalty(self.axis, self.side, commands, scores, safe)
            best = int(np.where(safe, scores - penalty, -np.inf).argmax())
            self.remaining -= 1
            self.active_steps += 1
            self.changes += int(best != part['best'])
            active = True
        assert safe[best]
        return best, penalty, active
