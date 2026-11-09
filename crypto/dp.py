"""Distributed Gaussian mechanism + Renyi-DP accountant (ARCHITECTURE.md §6.5).

PURE-MATH MODULE — verified in-sandbox, see tests/test_dp.py.

Clipping is free (it's R_norm, already proved in zero knowledge on the
EC side). What's new here is: (a) splitting one round's worth of
Gaussian noise into n_real additive shares that sum back to the
intended total noise, and (b) accounting (eps, delta) across many
rounds with subsampling amplification from VRF self-sampling (p=0.25).

This is ordinary numpy + closed-form RDP math — no group operations
needed, hence testable here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def split_noise(total_sigma: float, n_shares: int, dim: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Split N(0, total_sigma^2 I) into n_shares additive Gaussian shares
    that sum to one draw from the target distribution: each share is
    N(0, (total_sigma^2 / n_shares) I), independently drawn.
    """
    per_share_sigma = total_sigma / np.sqrt(n_shares)
    return [rng.normal(0.0, per_share_sigma, size=dim) for _ in range(n_shares)]


@dataclass
class RDPAccountant:
    """Renyi-DP accountant for the subsampled Gaussian mechanism.

    Uses the standard moment bound for Gaussian RDP composed over
    `rounds` rounds, each round touching the client via a Bernoulli(p)
    subsample (VRF self-sampling in JANUS-FL plays exactly this role —
    R_vrf, ARCHITECTURE.md §5.5 — which is where the amplification
    factor comes from).

    This matches the closed-form (not the tightest numerically-optimized
    RDP-for-subsampled-Gaussian bound from the literature, which requires
    numerical integration) — documented explicitly so nobody mistakes
    this for the camera-ready-grade accountant. Swap in a vetted library
    (e.g. Opacus' accountant) on Colab for the paper's final numbers;
    this closed form is for structural testing and sanity-checking orders
    of magnitude in this sandbox.
    """

    sigma: float           # noise multiplier (noise_std / clip_norm)
    sampling_rate: float   # p, from VRF self-sampling
    delta: float = 1e-5

    def rdp_gaussian(self, alpha: float) -> float:
        """RDP of the (non-subsampled) Gaussian mechanism at order alpha."""
        return alpha / (2.0 * self.sigma ** 2)

    def rdp_subsampled(self, alpha: float) -> float:
        """Simple subsampling amplification bound (Wang et al. style,
        loose but directionally correct and closed-form):
        eps_subsampled(alpha) <= log(1 + p^2 * (e^{eps_gauss(alpha)} - 1) * C(alpha,2)/alpha )
        Here we use the commonly cited simplified linear-in-p^2 regime
        bound valid for small p and alpha, which holds for our p=0.25,
        alpha in [2,64] operating range:
            rdp_sub(alpha) <= 2 * p^2 * rdp_gaussian(alpha)
        """
        return min(self.rdp_gaussian(alpha), 2 * self.sampling_rate ** 2 * alpha / (2 * self.sigma ** 2))

    def eps_for_rounds(self, rounds: int, alphas: list[float] | None = None) -> float:
        """Compose over `rounds` rounds and convert the best RDP order to
        (eps, delta)-DP via the standard RDP-to-DP conversion.
        """
        if alphas is None:
            alphas = [1.5, 2, 3, 4, 6, 8, 16, 32, 64]
        best_eps = float("inf")
        for alpha in alphas:
            rdp_total = rounds * self.rdp_subsampled(alpha)
            # RDP -> (eps, delta)-DP conversion (Mironov 2017):
            eps = rdp_total + np.log1p(-1.0 / alpha) - np.log(self.delta * alpha) / (alpha - 1)
            if alpha > 1 and eps > 0:
                best_eps = min(best_eps, eps)
        return best_eps

    def curve(self, max_rounds: int, step: int = 5) -> list[tuple[int, float]]:
        """(round, eps) pairs for plotting the privacy-utility curve
        (ROADMAP.md figure F9)."""
        return [(r, self.eps_for_rounds(r)) for r in range(step, max_rounds + 1, step)]
