"""Gaussian temporal replacement with an expected nonlinear value feature.

The filter is Bayesian under its learned, diagonal Gaussian measurement model.
This does not assert a calibrated density over raw crowd observations or intents.
Every call starts from the prior: counterfactual windows never mutate real memory.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class BayesianTemporalEncoder(nn.Module):
    def __init__(self, d_model=256, latent_dim=64, mean_only=False):
        super().__init__()
        if d_model < 1 or latent_dim < 1:
            raise ValueError("Feature and latent dimensions must be positive")
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.mean_only = mean_only
        self.backbone_name = "bayes_mean" if mean_only else "bayes"
        self.observation = nn.Linear(d_model, latent_dim)
        self.observation_noise = nn.Linear(d_model, latent_dim)
        nn.init.zeros_(self.observation_noise.weight)
        nn.init.constant_(self.observation_noise.bias, math.log(math.expm1(0.2)))
        self.retention_logits = nn.Parameter(torch.full((latent_dim,), 3.0))
        self.drift = nn.Parameter(torch.zeros(latent_dim))
        self.process_logits = nn.Parameter(
            torch.full((latent_dim,), math.log(math.expm1(0.05)))
        )
        self.prior_mean = nn.Parameter(torch.zeros(latent_dim))
        self.prior_logits = nn.Parameter(
            torch.full((latent_dim,), math.log(math.expm1(1.0)))
        )
        self.readout = nn.Sequential(
            nn.Linear(latent_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        basis = math.sqrt(latent_dim) * torch.eye(latent_dim)
        self.register_buffer("sigma_directions", torch.cat((basis, -basis), dim=0))

    def posterior(self, x, mask=None):
        """Return filtered means and diagonal variances, each [B,T,L].

        A false mask means no measurement, not a repeated observation. Prediction
        still advances. The inherited Parent supplies repeat-first-frame padding;
        no padding policy is silently changed by selecting this backbone.
        """
        if x.ndim != 3 or x.shape[-1] != self.d_model or x.shape[1] == 0:
            raise ValueError("Expected a nonempty [B,T,d_model] input")
        batch, length, _ = x.shape
        if mask is None:
            mask = torch.ones((batch, length), device=x.device, dtype=torch.bool)
        elif mask.shape != (batch, length):
            raise ValueError("Measurement mask must have shape [B,T]")
        else:
            mask = mask.to(device=x.device, dtype=torch.bool)
        measurements = self.observation(x)
        noise = F.softplus(self.observation_noise(x)) + 1e-6
        retention = self.retention_logits.sigmoid()
        process = F.softplus(self.process_logits) + 1e-6
        mean = self.prior_mean.expand(batch, -1)
        variance = (F.softplus(self.prior_logits) + 1e-6).expand(batch, -1)
        means, variances = [], []
        for step in range(length):
            # The first frame observes the initial prior; later frames transition.
            if step:
                mean = retention * mean + self.drift
                variance = retention.square() * variance + process
            gain = variance / (variance + noise[:, step])
            updated_mean = mean + gain * (measurements[:, step] - mean)
            updated_variance = variance * noise[:, step] / (variance + noise[:, step])
            valid = mask[:, step, None]
            mean = torch.where(valid, updated_mean, mean)
            variance = torch.where(valid, updated_variance, variance)
            means.append(mean)
            variances.append(variance)
        return torch.stack(means, dim=1), torch.stack(variances, dim=1)

    def expected_features(self, mean, variance):
        """Spherical-radial cubature for E[readout(z)] (an approximation).

        The unchanged linear value head can consume this expectation. Reading
        just E[z] through a linear head would otherwise discard all variance.
        """
        if self.mean_only:
            return self.readout(mean)
        points = mean.unsqueeze(-2) + variance.clamp_min(0).sqrt().unsqueeze(-2) * self.sigma_directions
        return self.readout(points).mean(dim=-2)

    def forward_last(self, x, mask=None):
        mean, variance = self.posterior(x, mask)
        return self.expected_features(mean[:, -1], variance[:, -1])

    def forward(self, x):
        mean, variance = self.posterior(x)
        return self.expected_features(mean, variance)
