"""Check the merged optimization, without installing temporary monkey patches."""

import ast
import configparser
import copy
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crowd_nav.contracts import init_grid_from_cfg
from crowd_nav.policy.mamba_rl import MambaRLPolicy
from crowd_nav.train import build_env_and_robot, bind_policy
from crowd_nav.utils.ppo_buffer import ReplayBufferIQL, ReplayBufferMC
from crowd_sim.envs.utils.state import JointState


def check_legacy_source():
    def body(path, name):
        tree = ast.parse(path.read_text())
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
        statements = list(node.body)
        while statements and isinstance(statements[0], ast.Expr) and isinstance(statements[0].value, ast.Str):
            statements.pop(0)
        return [ast.dump(n, include_attributes=False) for n in statements]
    path = ROOT / 'crowd_nav/policy/mamba_rl.py'
    assert body(path, '_predict_sarl_style_legacy') == body(
        Path(str(path) + '.bak_20261007_162833'), 'predict_sarl_style')


def rng_state():
    return np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all()


def restore(state):
    np.random.set_state(state[0]); torch.set_rng_state(state[1]); torch.cuda.set_rng_state_all(state[2])


def check_actions():
    results = {}
    for arm in ('bayes', 'bayes_mean', 'gru'):
        cfg = configparser.RawConfigParser(inline_comment_prefixes=('#', ';'))
        cfg.read(ROOT / f'crowd_nav/runs/bayes-matched-v1/{arm}-seed42.ini')
        init_grid_from_cfg(cfg)
        counts = dict(roots=0, action_mismatches=0, max_window_error=0.0, max_value_error=0.0)
        for people in (5, 20):
            cfg.set('sim', 'human_num', str(people))
            fast = MambaRLPolicy(cfg, device='cuda')
            saved = torch.load(ROOT / f'crowd_nav/runs/bayes-matched-v1/{arm}-seed42/il_policy.pth', map_location='cpu')
            fast.load_state_dict(saved['value'], strict=True)
            fast.multiagent_training = True; fast.use_sarl_predict = True
            fast.set_training_mode('rl')
            env, robot = build_env_and_robot(cfg)
            bind_policy(robot, fast, env, epsilon=0.0)
            for case in (412, 413):
                env.phase = 'test'; env.reset(seed=case, options={'test_case': case})
                fast.reset_episode_stats()
                for step in range(32):
                    state = JointState(robot.get_full_state(), [h.get_observable_state() for h in env.humans])
                    # cuDNN GRU has internal dropout state not restored by torch's
                    # public RNG snapshot. Bayesian dropout is fully replayed here.
                    training = case == 413 and arm != 'gru'
                    fast.train(training); fast.set_phase('train' if training else 'test')
                    fast.epsilon = 1.0 if training and step % 7 == 0 else 0.0
                    slow = copy.deepcopy(fast)
                    captured = []
                    for model in (fast, slow):
                        original = model.forward_value
                        def capture(x, original=original):
                            y = original(x)
                            captured.append((x.detach().cpu(), y.detach().cpu()))
                            return y
                        model.forward_value = capture
                    before = rng_state()
                    a = fast.predict_sarl_style(state)
                    after = rng_state()
                    restore(before)
                    b = slow._predict_sarl_style_legacy(state)
                    replayed = rng_state()
                    np.testing.assert_equal(after[0], replayed[0])
                    assert torch.equal(after[1], replayed[1])
                    assert all(torch.equal(x, y) for x, y in zip(after[2], replayed[2]))
                    assert fast._selected_action_index == slow._selected_action_index
                    np.testing.assert_array_equal(a, b)
                    np.testing.assert_array_equal(np.asarray(fast._history), np.asarray(slow._history))
                    if captured:
                        assert len(captured) == 2
                        torch.testing.assert_close(captured[0][0], captured[1][0], rtol=0, atol=0)
                        torch.testing.assert_close(captured[0][1], captured[1][1], rtol=0, atol=0)
                        counts['max_window_error'] = max(counts['max_window_error'], (captured[0][0] - captured[1][0]).abs().max().item())
                        counts['max_value_error'] = max(counts['max_value_error'], (captured[0][1] - captured[1][1]).abs().max().item())
                    fast.forward_value = fast.__class__.forward_value.__get__(fast)
                    counts['roots'] += 1
                    out = env.step(a)
                    if out[2] or out[3]:
                        break
        results[arm] = counts
    return results


def check_replay():
    results = []
    for contract in ('legal-prefix', 'legacy'):
        for n_step in (1, 3):
            for per in (False, True):
                options = dict(capacity=97, seq_len=24, n_step=n_step, use_per=per, history_contract=contract)
                old = ReplayBufferIQL(**options); new = ReplayBufferMC(**options)
                rng = np.random.RandomState(17)
                for length in (2, 5, 23, 24, 25, 70):
                    data = dict(states=rng.randn(length, 34).astype(np.float32),
                                actions=rng.randn(length, 2).astype(np.float32),
                                action_indices=rng.randint(0, 80, length).tolist(),
                                rewards=rng.randn(length).astype(np.float32), dones=np.zeros(length, bool))
                    data['dones'][-1] = True
                    old.store_episode(data); new.store_episode(data)
                assert old.ptr == new.ptr and old.size == new.size
                for field in ('states', 'returns', 'actions', 'action_indices', 'rewards', 'dones', 'priorities'):
                    np.testing.assert_array_equal(getattr(old, field), getattr(new, field))
                np.random.seed(7); a = old.sample(32, 'cpu')
                np.random.seed(7); b = new.sample(32, 'cpu')
                np.testing.assert_array_equal(a['indices'], b['indices'])
                for field in ('states', 'returns'):
                    torch.testing.assert_close(a[field], b[field], rtol=0, atol=0)
                if per:
                    torch.testing.assert_close(a['weights'], b['weights'], rtol=0, atol=0)
                assert new.next_states is None
                results.append(dict(contract=contract, n_step=n_step, per=per, passed=True))
    return results


if __name__ == '__main__':
    torch.set_num_threads(2)
    assert Path(sys.modules[MambaRLPolicy.__module__].__file__).resolve() == ROOT / 'crowd_nav/policy/mamba_rl.py'
    start = time.time()
    check_legacy_source()
    result = dict(legacy_source_matches_backup=True, replay=check_replay(), actions=check_actions(),
                  wall_seconds=time.time() - start, status='PASS')
    print(json.dumps(result, indent=2), flush=True)
