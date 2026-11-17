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


def dp_noise_share(
    sigma_dp: float,
    sampling_rate: float,
    n_clients: int,
    model_dim: int,
    b_k: bool,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """One client's share nu_k of the round's distributed Gaussian noise.

    Lives here rather than in fl/client.py (its only caller) so it can be
    unit-tested without dragging in the group backend: fl.client imports
    crypto.pedersen, which does a hash-to-point at module import time and
    needs rbcl. This function is ordinary numpy, like the rest of this
    module.

    Returns the EXACT zero vector in two distinct cases, which must not
    be confused with each other:

    1. `b_k` is False -- this client was not VRF-sampled this round, so
       it is a dummy submitting q_k = 0 (FIX.md F14, ARCHITECTURE.md
       line 201). Noise from dummies would inflate the aggregate past
       the configured sigma_dp, because all n=50 submissions are summed
       server-side while the sqrt(p*n) scaling assumes only the ~p*n
       sampled clients contribute noise.

    2. `sigma_dp <= 0` -- the DP layer is DISABLED by configuration
       (FIX.md F19). Then NO client adds noise, sampled or not, and the
       framework's privacy claim rests entirely on the cryptographic
       layer rather than on any (eps, delta) guarantee. This is an
       explicit configured state; see configs/mvp_3day.yaml's own `dp`
       block for the reasoning and for how to re-enable it.

    Otherwise draws N(0, per_client_sigma^2 I) with
    per_client_sigma = sigma_dp / sqrt(p * n), so that the ~p*n sampled
    clients' shares sum to one draw at the configured aggregate sigma_dp.
    """
    dp_enabled = sigma_dp > 0.0
    if not (b_k and dp_enabled):
        return np.zeros(model_dim, dtype=np.float64)
    per_client_sigma = sigma_dp / np.sqrt(max(sampling_rate * n_clients, 1e-9))
    if rng is None:
        rng = np.random.default_rng()
    return split_noise(per_client_sigma, n_shares=1, dim=model_dim, rng=rng)[0]


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
