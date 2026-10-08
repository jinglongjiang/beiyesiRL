"""Acceptance checks for explicit variance consumption; no performance claims."""

import argparse
import configparser
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crowd_nav import contracts
from crowd_nav.policy.bayes_temporal import BayesianTemporalEncoder
from crowd_nav.policy.mamba_rl import MambaRLPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--configs', type=Path, required=True)
    args = parser.parse_args()
    artifacts = args.artifacts
    assert json.loads((artifacts / 'b0-native.json').read_text())['status'] == 'PASS'
    torch.set_num_threads(2)
    torch.manual_seed(42)
    spec = importlib.util.spec_from_file_location('old_bayes', artifacts / 'old-bayes-temporal.py')
    old_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old_module)
    old = old_module.BayesianTemporalEncoder(mean_only=True).eval()
    old_weights = Path('/home/abc/4090_backup_20261008/root/bayes-vl-optimized-20261007/crowd_nav/runs/optimized-10000-r1/formal-bayes/rl_model_ep10000.pth')
    if old_weights.exists():
        saved = torch.load(old_weights, map_location='cpu', weights_only=False)['policy_state']
        old.load_state_dict({k[len('temporal_encoder.'):]: v for k, v in saved.items()
                            if k.startswith('temporal_encoder.')}, strict=True)
    mean = BayesianTemporalEncoder(mean_only=True).eval()
    copied = {k: v for k, v in old.state_dict().items()
              if k not in ('sigma_directions', 'readout.0.weight')}
    weight = torch.zeros_like(mean.readout[0].weight)
    weight[:, :64] = old.readout[0].weight
    copied['readout.0.weight'] = weight
    mean.load_state_dict(copied, strict=True)
    mu = torch.randn(32, 64)
    variance = torch.rand_like(mu) + .01
    delta = float((mean.expected_features(mu, variance) - old.readout(mu)).abs().max())
    torch.testing.assert_close(mean.expected_features(mu, variance), old.readout(mu), atol=2e-6, rtol=1e-5)
    assert torch.equal(mean.expected_features(mu, variance), mean.expected_features(mu, variance * 100))
    torch.manual_seed(42)
    full = BayesianTemporalEncoder().eval()
    torch.manual_seed(42)
    matched_mean = BayesianTemporalEncoder(mean_only=True).eval()
    assert full.state_dict().keys() == matched_mean.state_dict().keys()
    assert all(torch.equal(v, matched_mean.state_dict()[k]) for k, v in full.state_dict().items())
    change = float((full.expected_features(mu, variance) - full.expected_features(mu, variance * 2)).abs().max())
    assert change > 1e-5
    heads = {}
    for arm in ('gru', 'bayes_mean', 'bayes'):
        cfg = configparser.RawConfigParser(inline_comment_prefixes=('#', ';'))
        assert cfg.read(str(args.configs / '{}-seed42.ini'.format(arm)))
        contracts.init_grid_from_cfg(cfg)
        assert contracts.grid_action_dim(contracts.GRID) == 80
        policy = MambaRLPolicy(cfg, device='cpu').eval()
        heads[arm] = dict(weight_shape=list(policy.value_head.weight.shape),
                          parameters=sum(p.numel() for p in policy.value_head.parameters()))
        assert heads[arm] == dict(weight_shape=[1, 256], parameters=257)
        x = torch.randn(2, 24, 8, 13)
        with torch.no_grad():
            values = policy.forward_value(x)
            assert tuple(values.shape) == (2,) and torch.isfinite(values).all()
            padded = x.clone()
            padded[0, :23] = 0
            padded[1, :21] = 0
            expected = torch.cat((policy.forward_value(padded[:1, -1:]),
                                  policy.forward_value(padded[1:, -3:])))
            torch.testing.assert_close(policy.forward_value(padded), expected)
    result = dict(status='PASS', old_mean_mapping_max_abs=delta,
        mapped_trained_old_weights=old_weights.exists(), ordinary_and_legal_prefix='PASS',
        mean_variance_invariance='bitwise PASS', full_variance_sensitivity=change,
        identical_mean_full_state_dict='PASS', value_heads=heads,
        readout_source_sha256=hashlib.sha256((ROOT / 'crowd_nav/policy/bayes_temporal.py').read_bytes()).hexdigest(),
        claim='Engineering acceptance only; no trained navigation improvement yet.')
    (artifacts / 'self-tests.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
