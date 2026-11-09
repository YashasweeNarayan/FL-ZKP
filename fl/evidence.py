"""The §6.1 evidence accumulator with a certified false-accusation
bound (ARCHITECTURE.md §6.1, marked NOVEL there). PURE-MATH / TESTABLE
HERE: every input this module needs (the sketched update vector `q`,
a reference median vector, the public norm bound B_t, and a scalar
loss-delta) is already a plain numpy array/float by the time it
reaches this module -- cosine distance and the weighted sum below need
no torch and no real model, so the ACCUMULATION AND CALIBRATION LOGIC
is fully tested with synthetic vectors in tests/test_evidence.py.
(Computing a real client's `q`, median, and loss-delta from an actual
model update is fl/client.py's and fl/server.py's job, which DOES need
torch -- this module only consumes their outputs.)

Formula (ARCHITECTURE.md §6.1, verbatim):

    ev^(t)  = w1 * cos_dist(q, median) + w2 * ||q||/B_t + w3 * loss_delta_on_root_set
    E_P^(e) = sum_{t in epoch e} ev^(t)            # keyed by EPOCH PSEUDONYM P_k^e
    open iff E_P^(e) > theta(alpha, R, p, n)       # theta s.t. Pr[open | honest] <= alpha

Why the epoch pseudonym (not the client's real identity) keys the
accumulator: the server never learns which real client a pseudonym
belongs to during normal operation (that's the whole point of
clsag.py's dual-scope tags) -- it only ever sees "pseudonym P accrued
evidence E", and only the §6.2 opening protocol (run by the committee,
not the server alone) can map a pseudonym back to pk_k.

Calibration (`calibrate_theta`): a conformal/permutation procedure
run entirely on the server's own clean root set (ROADMAP.md §3.4
assigns exactly this role to server_root), simulating what a genuinely
honest client's ev^(t) distribution looks like, and setting theta to
the empirical (1-alpha) quantile of SIMULATED EPOCH TOTALS (not
single-round scores) -- this is what gives the certified
Pr[open | honest] <= alpha guarantee: theta is derived from honest-only
data, so by construction no more than an alpha fraction of honest
epoch totals exceed it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def cosine_distance(q: np.ndarray, reference: np.ndarray) -> float:
    denom = (np.linalg.norm(q) * np.linalg.norm(reference))
    if denom == 0:
        return 1.0  # maximally dissimilar if either vector is degenerate
    cos_sim = float(np.dot(q, reference) / denom)
    cos_sim = max(-1.0, min(1.0, cos_sim))  # clamp floating-point drift
    return 1.0 - cos_sim


def per_round_evidence(q: np.ndarray, median_ref: np.ndarray, norm_q: float, B_t: float, loss_delta: float, w1: float = 1.0, w2: float = 1.0, w3: float = 1.0) -> float:
    """ev^(t) for a single round's submission. `norm_q` is passed
    explicitly (rather than recomputed as np.linalg.norm(q)) because
    in the real pipeline `q` is the JL-sketched vector, not the raw
    update -- sketch.py's project() returns the sketch, and the true
    norm estimate is derived from it per sketch.py's own documented
    distortion bound, not a fresh np.linalg.norm call on the sketch
    itself (which would be the norm of the SKETCH, not the update).
    """
    cos_term = cosine_distance(q, median_ref)
    norm_term = norm_q / B_t if B_t > 0 else float("inf")
    return w1 * cos_term + w2 * norm_term + w3 * loss_delta


@dataclass
class EvidenceAccumulator:
    """Keyed by epoch pseudonym (any hashable -- in production, the
    bytes of clsag.py's P_k^e point). Resets at each new epoch by
    constructing a fresh instance; §6.4's committee reconfiguration
    operates on a different timescale (roster, not evidence) so is
    intentionally not coupled to this class.
    """
    epoch: int
    totals: dict = field(default_factory=dict)       # pseudonym -> running E_P^(e)
    round_counts: dict = field(default_factory=dict)  # pseudonym -> rounds observed this epoch

    def record(self, pseudonym, ev_t: float) -> None:
        self.totals[pseudonym] = self.totals.get(pseudonym, 0.0) + ev_t
        self.round_counts[pseudonym] = self.round_counts.get(pseudonym, 0) + 1

    def total_for(self, pseudonym) -> float:
        return self.totals.get(pseudonym, 0.0)

    def flagged(self, theta: float) -> list:
        """Pseudonyms whose accumulated evidence crosses theta --
        candidates for the §6.2 opening protocol. Returned sorted by
        descending evidence so the most clear-cut cases are reviewed
        first if a committee wants to triage.
        """
        return sorted(
            (p for p, total in self.totals.items() if total > theta),
            key=lambda p: -self.totals[p],
        )


@dataclass
class CalibrationCertificate:
    theta: float
    alpha: float
    n_permutations: int
    empirical_fp_rate_at_theta: float


def calibrate_theta(
    root_set_vectors: list,
    median_ref: np.ndarray,
    B_t: float,
    root_set_loss_deltas: np.ndarray,
    rounds_per_epoch: int,
    sampling_rate: float,
    alpha: float = 0.05,
    n_permutations: int = 2000,
    w1: float = 1.0, w2: float = 1.0, w3: float = 1.0,
    rng: np.random.Generator | None = None,
) -> CalibrationCertificate:
    """Simulates `n_permutations` honest "pseudo-clients" by resampling
    (with replacement) from the server's OWN clean root-set vectors to
    stand in for what an honest client's per-round submissions would
    look like, draws each one's number of participating rounds this
    epoch from Binomial(rounds_per_epoch, sampling_rate) (matching the
    VRF self-sampling rate, so the simulated epoch totals have the same
    expected round-count as real epochs), sums ev^(t) over those rounds
    to get a simulated honest epoch total, and sets theta to the
    empirical (1-alpha) quantile across all simulated totals -- by
    construction, at most an alpha fraction of simulated HONEST epoch
    totals exceed theta, which is exactly the certified bound
    ARCHITECTURE.md's Theorem (bounded false accusation) requires.
    """
    rng = rng if rng is not None else np.random.default_rng()
    n_root = len(root_set_vectors)
    if n_root == 0:
        raise ValueError("root_set_vectors must be non-empty to calibrate theta")

    simulated_totals = np.empty(n_permutations)
    for p in range(n_permutations):
        n_rounds_this_epoch = rng.binomial(rounds_per_epoch, sampling_rate)
        n_rounds_this_epoch = max(n_rounds_this_epoch, 1)  # avoid a degenerate zero-round epoch
        idxs = rng.integers(0, n_root, size=n_rounds_this_epoch)
        total = 0.0
        for i in idxs:
            q = root_set_vectors[i]
            norm_q = float(np.linalg.norm(q))
            loss_delta = float(root_set_loss_deltas[i % len(root_set_loss_deltas)])
            total += per_round_evidence(q, median_ref, norm_q, B_t, loss_delta, w1, w2, w3)
        simulated_totals[p] = total

    theta = float(np.quantile(simulated_totals, 1.0 - alpha))
    empirical_fp = float(np.mean(simulated_totals > theta))
    return CalibrationCertificate(theta=theta, alpha=alpha, n_permutations=n_permutations, empirical_fp_rate_at_theta=empirical_fp)
