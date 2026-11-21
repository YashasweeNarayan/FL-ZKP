"""Standard robust-aggregation baselines for comparison against JANUS-FL's
own crypto-verified aggregate (V1-V5 + the double-masked secure sum).

**Why this file exists**: a 2026-11-21 brutal-reviewer self-check on this
project's own results found the single most common gap for a paper
claiming Byzantine-robustness is a missing head-to-head comparison
against established robust-aggregation rules (Krum, trimmed-mean,
coordinate-median) -- "our system with the attack vs. our system
without it" is not the same claim as "our system vs. a competing
defense under the same attack." This file supplies that comparison's
defenses; `baselines/plain_fedavg_loop.py` supplies the training loop
that runs them under attack; the notebook that uses both lives at
`notebooks/RUN_MVP_BASELINE_DEFENSES_COMPARISON.ipynb`.

**Deliberately standalone, zero coupling to fl/ or crypto/**: these are
PLAINTEXT aggregation rules over real `np.ndarray` deltas -- the
comparison this file exists for is specifically "defense mechanism vs.
defense mechanism," not "crypto pipeline vs. crypto pipeline." Wiring
Krum/trimmed-mean/median INTO the real masked-and-verified aggregate
would require the server to see each client's individual plaintext
delta before combining them, which defeats the whole point of secure
aggregation -- that tension (robust aggregation needs per-client
visibility; secure aggregation hides it) is itself a known, citable
limitation of combining the two naively, not something to paper over
by faking a combined system that doesn't really hide anything. These
baselines intentionally run WITHOUT secure aggregation, exactly as
their own original papers describe them, so the comparison is honest:
"what would a non-anonymous robust defense have done here," not "we
bolted their formula onto our protocol."

Every function below is pure NumPy, takes a list of real 1-D delta
arrays (same `np.float64`, same `model_dim` as JANUS-FL's own pipeline
operates on after `flatten_state_dict`), and returns the aggregated
delta the server would apply. No mocking, no synthetic shortcuts --
these are the real published formulas.
"""
from __future__ import annotations

import numpy as np


def mean_aggregate(deltas: list[np.ndarray]) -> np.ndarray:
    """The "no defense" baseline: plain FedAvg averaging (McMahan et al.
    2017). Exactly what every attack in this repo targets -- a single
    unbounded malicious delta can drag this arbitrarily far off, which
    is the point of including it as the bottom-line comparison.
    """
    if not deltas:
        raise ValueError("mean_aggregate: empty deltas list")
    return np.mean(np.stack(deltas, axis=0), axis=0)


def coordinate_median(deltas: list[np.ndarray]) -> np.ndarray:
    """Coordinate-wise median (Yin et al. 2018, "Byzantine-Robust
    Distributed Learning"). Breaks the per-coordinate link between
    clients entirely -- cheap, and robust to up to ~50% malicious
    coordinates per dimension, at the cost of discarding legitimate
    cross-client correlation structure the mean would have kept.
    """
    if not deltas:
        raise ValueError("coordinate_median: empty deltas list")
    return np.median(np.stack(deltas, axis=0), axis=0)


def trimmed_mean(deltas: list[np.ndarray], trim_fraction: float = 0.2) -> np.ndarray:
    """Coordinate-wise trimmed mean (Yin et al. 2018, same paper as
    coordinate_median above). Sorts each coordinate across clients,
    drops the top and bottom `trim_fraction` of values at that
    coordinate, averages what's left. `trim_fraction=0.2` with n
    clients drops floor(0.2*n) from each tail -- matches this repo's
    own attack-fraction grid's largest tested fraction (30%), so a
    30%-malicious attack is NOT fully covered by the trim margin at the
    default setting; that tradeoff (bigger trim = more robust but more
    honest signal discarded) is the real, citable property being
    measured, not a bug in this implementation.
    """
    if not deltas:
        raise ValueError("trimmed_mean: empty deltas list")
    if not (0.0 <= trim_fraction < 0.5):
        raise ValueError(f"trimmed_mean: trim_fraction must be in [0, 0.5), got {trim_fraction}")
    stacked = np.stack(deltas, axis=0)  # (n_clients, model_dim)
    n = stacked.shape[0]
    k = int(np.floor(trim_fraction * n))
    sorted_vals = np.sort(stacked, axis=0)
    if k == 0:
        return np.mean(sorted_vals, axis=0)
    trimmed = sorted_vals[k:n - k]
    if trimmed.shape[0] == 0:
        raise ValueError(
            f"trimmed_mean: trim_fraction={trim_fraction} with n={n} clients leaves "
            f"nothing after trimming both tails -- reduce trim_fraction or increase n"
        )
    return np.mean(trimmed, axis=0)


def krum(deltas: list[np.ndarray], n_byzantine: int) -> tuple[np.ndarray, int]:
    """Krum (Blanchard et al. 2017, "Machine Learning with Adversaries:
    Byzantine Tolerant Gradient Descent"). For each client i, sums the
    squared Euclidean distances to its (n - n_byzantine - 2) CLOSEST
    other clients' deltas (deliberately excluding the farthest
    n_byzantine + 1, since those are exactly the ones Krum is designed
    to not be fooled by); returns the single delta with the smallest
    such score -- i.e. Krum selects ONE client's delta outright rather
    than averaging, by design (an outlier delta can never be chosen,
    but neither can any information from the clients NOT selected be
    used, which is the real, citable cost of this defense vs. an
    averaging-based one).

    `n_byzantine` must satisfy `n < 2*n_byzantine + 3` NOT holding --
    i.e. this needs `n >= 2*n_byzantine + 3` to have >=1 neighbor left
    per client's score; raises clearly rather than returning a
    degenerate result if the caller's assumed malicious count doesn't
    leave Krum enough honest majority to work with.

    Returns (selected_delta, selected_index) -- the index is returned
    too since which specific client "won" is itself part of what a
    robustness evaluation reports (e.g. "did Krum ever select an
    actual attacker's delta").
    """
    n = len(deltas)
    if n < 2 * n_byzantine + 3:
        raise ValueError(
            f"krum: n={n} clients with n_byzantine={n_byzantine} assumed -- Krum requires "
            f"n >= 2*n_byzantine + 3 (={2 * n_byzantine + 3}) to have a well-defined score"
        )
    n_closest = n - n_byzantine - 2
    stacked = np.stack(deltas, axis=0)
    # pairwise squared Euclidean distances, (n, n)
    diffs = stacked[:, None, :] - stacked[None, :, :]
    sq_dists = np.sum(diffs * diffs, axis=-1)
    scores = np.zeros(n, dtype=np.float64)
    for i in range(n):
        others = np.delete(sq_dists[i], i)
        closest = np.sort(others)[:n_closest]
        scores[i] = np.sum(closest)
    best_idx = int(np.argmin(scores))
    return deltas[best_idx].copy(), best_idx


def multi_krum(deltas: list[np.ndarray], n_byzantine: int, m: int) -> np.ndarray:
    """Multi-Krum (same Blanchard et al. 2017 paper, §4.3): repeats
    Krum's selection m times (removing the previously-selected client
    each time before re-scoring the remainder), then averages the m
    selected deltas -- recovers some of plain averaging's use of
    multiple honest signals while keeping Krum's rejection behavior.
    `m` must be <= n - n_byzantine (can't select more clients than are
    honest, or Multi-Krum degenerates into including an attacker by
    construction once honest candidates run out).
    """
    n = len(deltas)
    max_m = n - 2 * n_byzantine - 2  # each of the m krum() calls needs >= 2*n_byz+3 remaining
    if m > max_m:
        raise ValueError(
            f"multi_krum: m={m} exceeds the max {max_m} selections krum() can make from "
            f"n={n} clients with n_byzantine={n_byzantine} assumed (each selection needs "
            f">= 2*n_byzantine + 3 candidates remaining) -- reduce m"
        )
    remaining_idx = list(range(len(deltas)))
    remaining = list(deltas)
    selected = []
    for _ in range(m):
        _, local_best = krum(remaining, n_byzantine)
        selected.append(remaining[local_best])
        del remaining[local_best]
        del remaining_idx[local_best]
    return np.mean(np.stack(selected, axis=0), axis=0)


AGGREGATION_RULES = {
    "mean_no_defense": lambda deltas, n_byz: mean_aggregate(deltas),
    "coordinate_median": lambda deltas, n_byz: coordinate_median(deltas),
    "trimmed_mean": lambda deltas, n_byz: trimmed_mean(deltas, trim_fraction=0.2),
    "krum": lambda deltas, n_byz: krum(deltas, n_byz)[0],
    "multi_krum": lambda deltas, n_byz: multi_krum(deltas, n_byz, m=max(1, len(deltas) - 2 * n_byz - 2)),
}
