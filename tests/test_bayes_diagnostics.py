"""Diagnostic-label and paired-bootstrap integrity, without changing policy."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('bayes_diagnostic', ROOT / 'scripts/diagnose-bayes-variance.py')
DIAGNOSTIC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DIAGNOSTIC)


class DiagnosticContracts(unittest.TestCase):
    def test_incomplete_safe_future_is_censored(self):
        labels, valid = DIAGNOSTIC.build_labels([
            {'clearance': np.array([1.0, 1.0, 1.0]), 'sigma': np.ones((3, 64)), 'outcome': 'success'}])
        self.assertFalse(valid['future-4-clearance-0.2'].any())
        self.assertFalse(labels['future-4-clearance-0.2'].any())

    def test_incomplete_danger_is_valid_positive(self):
        labels, valid = DIAGNOSTIC.build_labels([
            {'clearance': np.array([1.0, -.1]), 'sigma': np.ones((2, 64)), 'outcome': 'collision'}])
        self.assertTrue(valid['future-4-clearance-0.0'].all())
        self.assertTrue(labels['future-4-clearance-0.0'].all())
        self.assertTrue(labels['episode-collision'].all())

    def test_future_does_not_cross_episode_boundary(self):
        labels, valid = DIAGNOSTIC.build_labels([
            {'clearance': np.ones(4), 'sigma': np.ones((4, 64)), 'outcome': 'success'},
            {'clearance': np.array([-.1]), 'sigma': np.ones((1, 64)), 'outcome': 'collision'}])
        self.assertEqual(labels['future-4-clearance-0.0'].tolist(), [False] * 4 + [True])
        self.assertEqual(valid['future-4-clearance-0.0'].tolist(), [True, False, False, False, True])

    def test_paired_identical_predictions_have_zero_delta(self):
        y = np.array([False, True, False, True])
        prediction = np.array([.1, .8, .3, .7])
        episodes = np.array([0, 0, 1, 1])
        result = DIAGNOSTIC.paired_bootstrap(y, prediction, prediction, episodes,
                                            {'statistics': {'bootstrap_seed': 20261009}})
        self.assertEqual(result['delta_auc_95_ci'], [0., 0.])

    def test_legacy_split_counts(self):
        with np.load(ROOT / 'crowd_nav/runs/bayes-fix-20261008/b1-rollout.npz') as data:
            target = 'future-4-clearance-0.2'
            holdout = data['case_index'] % 4 == 0
            self.assertEqual(int((data['valid-' + target] & ~holdout).sum()), 7573)
            self.assertEqual(int((data['valid-' + target] & holdout).sum()), 2581)
            self.assertEqual(len(set(data['episode'][holdout])), 48)
            self.assertEqual(len(set(data['episode'][~holdout])), 144)


if __name__ == '__main__':
    unittest.main()
