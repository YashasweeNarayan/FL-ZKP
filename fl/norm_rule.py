"""The public update-norm anomaly threshold B_t (ARCHITECTURE.md's
check V3 / the R_norm bound that rangeproof.py's bit-decomposition
proof enforces in zero-knowledge). PURE-MATH / TESTABLE HERE: pure
numpy over arbitrary arrays of (sketched) update norms -- no torch, no
real model needed to test the threshold RULE itself.

Why median/MAD, not mean/std: ARCHITECTURE.md's limitations review
(REVIEW.md) flagged that a mean/std-based threshold is itself
poisonable -- a coordinated set of malicious clients submitting
large-norm updates can drag the mean and inflate the std enough to
raise B_t and let themselves pass. Median and MAD (median absolute
deviation) are the standard robust-statistics fix: up to just under
50% of the population can be adversarial before they can meaningfully
shift the median, which is the honest bound given this repo's attacker
model (ARCHITECTURE.md A1: "up to f < n/2 malicious clients, bounded
by the VSS/committee threshold assumptions elsewhere" -- see A2's
loss-delta term for a complementary check this module doesn't replace).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# consistency constant so MAD approximates std under Gaussian noise
# (standard statistics convention: 1.4826 = 1/Phi^{-1}(3/4))
_MAD_TO_STD = 1.4826


@dataclass
class NormThreshold:
    median: float
    mad: float
    B_t: float
    k: float  # the multiplier actually used


def compute_threshold(norms: np.ndarray, k: float = 3.0, min_mad_floor: float = 1e-6) -> NormThreshold:
    """B_t = median(norms) + k * 1.4826 * MAD(norms). `min_mad_floor`
    guards the degenerate all-identical-norms case (MAD=0 would make
    B_t == median exactly, rejecting every nonzero deviation including
    benign ones from quantization noise).
    """
    if len(norms) == 0:
        raise ValueError("cannot compute a threshold from an empty norm array")
    med = float(np.median(norms))
    mad = float(np.median(np.abs(norms - med)))
    mad_scaled = max(mad * _MAD_TO_STD, min_mad_floor)
    B_t = med + k * mad_scaled
    return NormThreshold(median=med, mad=mad_scaled, B_t=B_t, k=k)


def flag_anomalous(norms: np.ndarray, threshold: NormThreshold) -> np.ndarray:
    """Boolean mask, True where a client's update norm exceeds B_t --
    this is the PUBLIC check (anyone can recompute it from the
    published norms / range-proof public outputs), distinct from the
    zero-knowledge range proof itself, which proves "my norm is below
    B_t" without revealing the exact norm.
    """
    return norms > threshold.B_t


def bootstrap_from_server_root(root_set_norms: np.ndarray, k: float = 3.0) -> NormThreshold:
    """Round-0 bootstrap (ROADMAP.md §3.4: server root set is used for
    "B_t bootstrap at round 0, evidence calibration"): compute an
    initial threshold from benign reference norms (computed by the
    server itself on its own root-set data, never seen by clients)
    before any real client submissions exist to calibrate against.
    """
    return compute_threshold(root_set_norms, k=k)


def update_threshold_ema(previous: NormThreshold, new_round_norms: np.ndarray, k: float = 3.0, alpha: float = 0.3) -> NormThreshold:
    """Subsequent rounds: exponential moving average between the
    previous round's (median, mad) and the current round's, so B_t
    adapts as training progresses (update norms shrink as the model
    converges) without being fully re-derived from a single round's
    (possibly adversarial-influenced) submissions.
    """
    current = compute_threshold(new_round_norms, k=k)
    blended_median = alpha * current.median + (1 - alpha) * previous.median
    blended_mad = alpha * current.mad + (1 - alpha) * previous.mad
    B_t = blended_median + k * blended_mad
    return NormThreshold(median=blended_median, mad=blended_mad, B_t=B_t, k=k)
