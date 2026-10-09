import sys
sys.dont_write_bytecode = True
import unittest
from types import SimpleNamespace
import numpy as np
import torch
from decompose import ROOT, Intervention, native_counterfactual

sys.path.insert(0, str(ROOT))
from crowd_nav.policy.bayes_temporal import BayesianTemporalEncoder


class Contract(unittest.TestCase):
    def test_noop_and_channels(self):
        torch.manual_seed(20261009)
        encoder = BayesianTemporalEncoder().eval()
        mean = torch.randn(80, 64)
        variance = torch.rand(80, 64) + .1
        original = encoder.expected_features(mean, variance)
        hook = Intervention(encoder)
        try:
            self.assertTrue(torch.equal(original, encoder.expected_features(mean, variance)))
            hook.permutation = torch.arange(79, -1, -1)
            logvar = variance.log()
            for condition in ('shuffle_within', 'ctrl_shuffle_mean'):
                hook.mode = condition
                actual = encoder.expected_features(mean, variance)
                expected = encoder.readout(torch.cat(
                    (mean, logvar.flip(0)) if condition == 'shuffle_within' else (mean.flip(0), logvar), -1))
                self.assertTrue(torch.equal(actual, expected))
            hook.mode = 'fixed'
            hook.constant = logvar.mean(0)
            actual = encoder.expected_features(mean, variance)
            expected = encoder.readout(torch.cat((mean, hook.constant.expand(80, -1)), -1))
            self.assertTrue(torch.equal(actual, expected))
        finally:
            hook.close()
        self.assertTrue(torch.equal(original, encoder.expected_features(mean, variance)))

    def test_timeout_priority_and_geometry(self):
        env = SimpleNamespace(time_step=.25, progress_reward=0, time_penalty=0,
              stand_penalty=0, discomfort_dist=.2, discomfort_penalty_factor=.5,
              success_reward=1, collision_penalty=-.5, timeout_penalty=-.5, time_limit=25)
        commands = np.array([[1., 0.], [-1., 0.]])
        before = (np.array([0., 0.]), np.array([4., 0.]), .3,
                  np.array([[.8, 0.]]), np.array([[0., 0.]]), np.array([.3]), 0)
        endpoint, swept, reward = native_counterfactual(env, before, commands, np.array([[.8, 0.]]))
        np.testing.assert_allclose(endpoint, [-.05, .45], atol=1e-12)
        np.testing.assert_allclose(swept, [-.05, .2], atol=1e-12)
        np.testing.assert_allclose(reward, [-.5, 0.], atol=1e-12)
        before = before[:-1] + (25,)
        _, _, reward = native_counterfactual(env, before, commands, np.array([[.8, 0.]]))
        np.testing.assert_array_equal(reward, [-.5, -.5])

    def test_shuffle_does_not_touch_global_rng(self):
        np.random.seed(123)
        expected = np.random.random(4)
        np.random.seed(123)
        rng = np.random.default_rng(np.random.SeedSequence([20261009, 42, 0, 0, 0, 2]))
        permutation = rng.permutation(80)
        self.assertEqual(sorted(permutation.tolist()), list(range(80)))
        np.testing.assert_array_equal(np.random.random(4), expected)


if __name__ == '__main__':
    unittest.main()
