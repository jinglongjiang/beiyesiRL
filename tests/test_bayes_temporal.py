import configparser
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from crowd_nav.policy.bayes_temporal import BayesianTemporalEncoder
from crowd_nav.policy.mamba_rl import MambaRLPolicy


ROOT = Path(__file__).resolve().parents[1]


def policy_config(kind):
    cfg = configparser.RawConfigParser(inline_comment_prefixes=("#", ";"))
    cfg.read([str(ROOT / "crowd_nav/configs" / name)
              for name in ("env.config", "policy.config", "train.config")])
    cfg.set("mamba", "temporal_backbone", kind)
    from crowd_nav.contracts import init_grid_from_cfg
    init_grid_from_cfg(cfg)
    return cfg


class BayesianTemporalTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)

    def test_matched_train_config_reaches_il_optimizer(self):
        from crowd_nav.train import TrainConfig, _run_policy_il_pretrain, safe_torch_load
        cfg = policy_config('bayes')
        cfg.set('temporal', 'history_contract', 'legal-prefix')
        cfg.set('train', 'matched_rng', 'true')
        cfg.set('train', 'il_data_on_cpu', 'true')
        cfg.set('train', 'seed', '43')
        model = MambaRLPolicy(cfg)
        states = np.zeros((8, 34), dtype=np.float32)
        states[:, 4] = 0.3
        states[:, 6:9] = (4, 1, 0)
        states[:, 9:14] = (2, 0, 0, 0, 0.3)
        trajectories = [((states, np.zeros((8, 2)), [0.0] * 7 + [1.0]), {})]
        before = {key: value.clone() for key, value in model.state_dict().items()}
        with tempfile.TemporaryDirectory() as directory:
            train_cfg = TrainConfig(cfg, directory)
            self.assertTrue(train_cfg.matched_rng)
            self.assertTrue(train_cfg.il_data_on_cpu)
            checkpoint = str(Path(directory) / 'il_policy.pth')
            _run_policy_il_pretrain(model, torch.device('cpu'), 24, 1, 8,
                                   checkpoint, None, 0.99, trajectories, train_cfg)
            saved = safe_torch_load(checkpoint)
            self.assertEqual(saved['meta']['dataset_size'], 8)
            self.assertTrue(any(not torch.equal(value, before[key])
                                for key, value in model.state_dict().items()))

    def test_gaussian_update_matches_scalar_analysis(self):
        model = BayesianTemporalEncoder(1, 1)
        with torch.no_grad():
            model.observation.weight.fill_(1)
            model.observation.bias.zero_()
        inputs = torch.tensor([[[2.0], [1.0]]])
        means, variances = model.posterior(inputs)
        prior = torch.nn.functional.softplus(model.prior_logits)[0].item() + 1e-6
        noise = torch.nn.functional.softplus(model.observation_noise.bias)[0].item() + 1e-6
        m = prior / (prior + noise) * 2
        p = prior * noise / (prior + noise)
        self.assertAlmostEqual(means[0, 0, 0].item(), m, places=5)
        self.assertAlmostEqual(variances[0, 0, 0].item(), p, places=5)
        a = model.retention_logits.sigmoid()[0].item()
        q = torch.nn.functional.softplus(model.process_logits)[0].item() + 1e-6
        m, p = a * m, a * a * p + q
        gain = p / (p + noise)
        self.assertAlmostEqual(means[0, 1, 0].item(), m + gain * (1 - m), places=5)
        self.assertAlmostEqual(variances[0, 1, 0].item(), p * (1 - gain), places=5)

    def test_masked_frame_only_predicts(self):
        model = BayesianTemporalEncoder(8, 3)
        x = torch.randn(2, 4, 8)
        mask = torch.tensor([[True, True, False, True]]).expand(2, -1)
        mean, var = model.posterior(x, mask)
        expected_mean = model.retention_logits.sigmoid() * mean[:, 1] + model.drift
        expected_var = model.retention_logits.sigmoid().square() * var[:, 1]
        expected_var += torch.nn.functional.softplus(model.process_logits) + 1e-6
        torch.testing.assert_close(mean[:, 2], expected_mean)
        torch.testing.assert_close(var[:, 2], expected_var)
        x[:, 2] += 1000
        changed = model.posterior(x, mask)
        torch.testing.assert_close(mean, changed[0])
        torch.testing.assert_close(var, changed[1])

    def test_causal_prefix_and_no_cross_call_state(self):
        model = BayesianTemporalEncoder(8, 3).eval()
        x = torch.randn(2, 24, 8)
        original = model.posterior(x)
        changed = x.clone()
        changed[:, 12:] += 10
        modified = model.posterior(changed)
        for lhs, rhs in zip(original, modified):
            torch.testing.assert_close(lhs[:, :12], rhs[:, :12])
        model(torch.randn_like(x))
        for lhs, rhs in zip(original, model.posterior(x)):
            torch.testing.assert_close(lhs, rhs)

    def test_readout_receives_mean_and_logvariance(self):
        model = BayesianTemporalEncoder(8, 3)
        mean, var = torch.randn(2, 3), torch.rand(2, 3)
        expected = model.readout(torch.cat((mean, var.clamp_min(1e-6).log()), dim=-1))
        torch.testing.assert_close(model.expected_features(mean, var), expected, atol=0, rtol=0)

    def test_uncertainty_reaches_readout_not_just_mean(self):
        model = BayesianTemporalEncoder(8, 3).eval()
        mean = torch.zeros(2, 3)
        low = model.expected_features(mean, torch.full_like(mean, 0.01))
        high = model.expected_features(mean, torch.full_like(mean, 1.0))
        self.assertGreater((low - high).abs().max().item(), 1e-5)
        model.mean_only = True
        self.assertTrue(torch.equal(model.expected_features(mean, torch.ones_like(mean)),
                                    model.expected_features(mean, torch.zeros_like(mean))))

    def test_zero_variance_columns_recover_old_mean_readout(self):
        model = BayesianTemporalEncoder(8, 3, mean_only=True)
        old = torch.nn.Sequential(torch.nn.Linear(3, 8), torch.nn.SiLU(), torch.nn.Linear(8, 8))
        with torch.no_grad():
            model.readout[0].weight[:, :3].copy_(old[0].weight)
            model.readout[0].weight[:, 3:].zero_()
            model.readout[0].bias.copy_(old[0].bias)
            model.readout[2].load_state_dict(old[2].state_dict())
        mean, var = torch.randn(2, 3), torch.rand(2, 3)
        torch.testing.assert_close(model.expected_features(mean, var), old(mean))

    def test_posterior_has_positive_variance_and_all_gradients(self):
        model = BayesianTemporalEncoder(8, 3)
        x = torch.randn(2, 24, 8, requires_grad=True)
        _, var = model.posterior(x)
        self.assertTrue((var > 0).all().item())
        model.forward_last(x).square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all().item())
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all().item(), name)

    def test_full_and_mean_control_have_identical_parameters(self):
        torch.manual_seed(123)
        full = BayesianTemporalEncoder(8, 3)
        torch.manual_seed(123)
        mean = BayesianTemporalEncoder(8, 3, mean_only=True)
        for key, value in full.state_dict().items():
            torch.testing.assert_close(value, mean.state_dict()[key])
        self.assertEqual(sum(p.numel() for p in full.parameters()), sum(p.numel() for p in mean.parameters()))

    def test_parent_gru_value_is_unchanged(self):
        model = MambaRLPolicy(policy_config("gru")).eval()
        x = torch.randn(2, 24, 8, 13)
        with torch.no_grad():
            legacy = model.value_head(model.temporal_encoder(model.spatial_encoder(x))[:, -1]).squeeze(-1)
            torch.testing.assert_close(model.forward_value(x), legacy, atol=0, rtol=0)

    def test_real_lookahead_writes_one_observation_not_candidates(self):
        from crowd_sim.envs.crowd_sim import CrowdSim
        from crowd_sim.envs.utils.robot import Robot
        from crowd_sim.envs.utils.state import JointState
        cfg = policy_config("bayes")
        env = CrowdSim()
        env.configure(cfg)
        robot = Robot(cfg, "robot")
        env.set_robot(robot)
        model = MambaRLPolicy(cfg).eval()
        model.set_phase("test")
        model.use_sarl_predict = True
        robot.set_policy(model)
        env.phase = "test"
        env.reset(options={"test_case": 0})
        state = JointState(robot.get_full_state(), [h.get_observable_state() for h in env.humans])
        with torch.no_grad():
            selected, index = model.act(state)
        self.assertEqual(len(model.action_space), 80)
        self.assertEqual(len(model._history), 1)
        self.assertIsInstance(index, int)
        self.assertEqual(model.action_space[index], selected)
        env.step(selected)
        self.assertAlmostEqual(robot.vx, selected.vx, places=6)
        self.assertAlmostEqual(robot.vy, selected.vy, places=6)

    def test_last_readout_matches_sequence_last(self):
        model = BayesianTemporalEncoder(8, 3).eval()
        x = torch.randn(2, 24, 8)
        torch.testing.assert_close(model.forward_last(x), model(x)[:, -1])

    def test_exploration_index_preserves_original_rng_choice(self):
        from crowd_sim.envs.utils.state import FullState, ObservableState, JointState
        model = MambaRLPolicy(policy_config("bayes"))
        model.use_sarl_predict = True
        model.set_phase("train")
        model.set_epsilon(1.0)
        model.build_action_space(1.0)
        state = JointState(FullState(0, 0, 0, 0, 0.3, 0, 4, 1, 0),
                           [ObservableState(2, 0, 0, 0, 0.3)])
        rng = np.random.RandomState(123)
        rng.random_sample()
        expected = int(rng.choice(80))
        np.random.seed(123)
        action, index = model.predict(state, deterministic=False, return_idx=True)
        self.assertEqual(index, expected)
        self.assertEqual(action, model.action_space[expected])

    def test_policy_backward_checkpoint_and_candidate_independence(self):
        model = MambaRLPolicy(policy_config("bayes")).eval()
        x = torch.randn(2, 24, 8, 13)
        values = model.forward_value(x)
        self.assertEqual(tuple(values.shape), (2,))
        with torch.no_grad():
            solo = torch.cat([model.forward_value(x[i:i+1]) for i in range(2)])
            torch.testing.assert_close(solo, values)
        values.square().mean().backward()
        self.assertIsNotNone(model.temporal_encoder.observation_noise.weight.grad)
        checkpoint = io.BytesIO()
        torch.save(model.state_dict(), checkpoint)
        checkpoint.seek(0)
        restored = MambaRLPolicy(policy_config("bayes")).eval()
        restored.load_state_dict(torch.load(checkpoint, weights_only=False), strict=True)
        torch.testing.assert_close(restored.forward_value(x), values.detach())

    def test_legal_padding_matches_only_real_history_for_all_backbones(self):
        for kind in ('gru', 'bayes_mean', 'bayes'):
            cfg = policy_config(kind)
            cfg.set('temporal', 'history_contract', 'legal-prefix')
            model = MambaRLPolicy(cfg).eval()
            real = torch.randn(2, 3, 8, 13)
            padded = torch.cat((torch.zeros(2, 21, 8, 13), real), dim=1)
            with torch.no_grad():
                torch.testing.assert_close(model.forward_value(padded), model.forward_value(real))
            model.forward_value(padded).square().mean().backward()
            self.assertTrue(any(p.grad is not None for p in model.temporal_encoder.parameters()))

    def test_mixed_legal_lengths_and_invalid_holes(self):
        cfg = policy_config('bayes')
        cfg.set('temporal', 'history_contract', 'legal-prefix')
        model = MambaRLPolicy(cfg).eval()
        x = torch.randn(2, 24, 8, 13)
        x[0, :23] = 0
        x[1, :21] = 0
        with torch.no_grad():
            expected = torch.cat((model.forward_value(x[0:1, -1:]),
                                  model.forward_value(x[1:2, -3:])))
            torch.testing.assert_close(model.forward_value(x), expected)
        x[1, 22] = 0
        with self.assertRaises(ValueError):
            model.forward_value(x)

    def test_legal_exploration_updates_history_and_reset_clears_it(self):
        from crowd_sim.envs.utils.state import FullState, ObservableState, JointState
        cfg = policy_config('bayes')
        cfg.set('temporal', 'history_contract', 'legal-prefix')
        model = MambaRLPolicy(cfg)
        model.use_sarl_predict = True
        model.set_phase('train')
        model.set_epsilon(1.0)
        state = JointState(FullState(0, 0, 0, 0, 0.3, 0, 4, 1, 0),
                           [ObservableState(2, 0, 0, 0, 0.3)])
        model.act(state)
        model.act(state)
        self.assertEqual(len(model._history), 2)
        model.reset_episode_stats()
        self.assertEqual(len(model._history), 0)

    def test_legal_replay_padding_and_episode_boundaries(self):
        from crowd_nav.utils.ppo_buffer import ReplayBufferIQL
        buffer = ReplayBufferIQL(capacity=8, seq_len=4, history_contract='legal-prefix')
        for offset in (1, 10):
            buffer.store_episode({'tokens': np.full((3, 8, 13), offset, dtype=np.float32),
                                  'states': np.zeros((3, 34), dtype=np.float32),
                                  'actions': np.zeros((3, 2)), 'action_indices': [0, 1, 2],
                                  'rewards': [0, 0, 1], 'dones': [False, False, True]})
        self.assertEqual(len(buffer), 6)
        for index in (0, 3):
            self.assertFalse(buffer.states[index, :3].any())
        self.assertTrue((buffer.states[3, -1] == 10).all())

    @unittest.skipUnless(torch.cuda.is_available(), 'Mamba kernel requires CUDA')
    def test_mamba_legal_prefix_on_cuda(self):
        try:
            import mamba_ssm
        except ImportError:
            self.skipTest('Official Mamba runtime unavailable')
        cfg = policy_config('mamba')
        cfg.set('temporal', 'history_contract', 'legal-prefix')
        model = MambaRLPolicy(cfg).cuda().eval()
        real = torch.randn(2, 3, 8, 13, device='cuda')
        padded = torch.cat((torch.zeros(2, 21, 8, 13, device='cuda'), real), dim=1)
        with torch.no_grad():
            torch.testing.assert_close(model.forward_value(padded), model.forward_value(real))


if __name__ == "__main__":
    unittest.main()
