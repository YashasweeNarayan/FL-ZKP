"""Baseline Byzantine-FL attacks from the recent robustness-benchmark
literature (2026-11-20) -- NOT in ARCHITECTURE.md's own A1-A7 table,
added so the paper covers the standard baselines a reviewer familiar
with that literature expects to see, alongside the paper's own novel
attack suite.

Unlike A1 (`attacks/a1_scaling.py`'s `craft_scaled_update`), which
REPLACES a client's real trained delta outright with a hand-picked
vector (via `fl/client.py`'s `_force_delta`), every attack here
TRANSFORMS that client's own real Step-1 training output (via the new
`_delta_transform` hook, `fl/round_loop.py`'s `cheater_delta_transform`
param, 2026-11-20) -- sign-flip and noise are defined in the
literature this way, and scale-to-threshold/low-and-slow need the
real delta's direction to be a meaningful attack at all (negating or
scaling a zero vector is a no-op either way).

Four attacks, in two groups:

- `craft_sign_flip`, `craft_gaussian_noise`: cheap, standard baselines
  (Shi et al. 2021 "Challenges and Approaches for Mitigating Byzantine
  Attacks in FL"; used as sanity-check rows in nearly every Byzantine-
  FL benchmark paper found). Expected to be caught easily by V3 (a
  flipped-sign or noisy delta is usually still a large-norm outlier) --
  included for completeness, not because they're expected to be hard.

- `craft_scale_to_threshold`, `craft_low_and_slow`: the attacks that
  actually matter for this paper's credibility. Lycklama et al. 2025
  ("Hammer and Anvil: Toward a Theory of Backdoors in FL") show a
  SINGLE adaptive client scaling its update to exactly a norm-bounding
  defense's threshold is enough to break norm-bounding outright, for
  ANY functional threshold -- unlike A1's deliberately naive gamma=100
  (which overshoots the bound so far V3 catches it trivially), this is
  the realistic version of a scaling attack: scale to (just under) the
  bound, not wildly past it. `craft_low_and_slow` is this project's own
  self-devised extension, not from any paper: deliberately scale even
  smaller, far enough under B_t that a single round looks innocuous,
  testing whether the attack can also stay under the §6.1 evidence
  threshold `theta` across a whole epoch -- i.e. whether patience (not
  size) can evade BOTH defenses layered together. If an attacker can
  evade V3 by staying under B_t, the real question this paper's own
  traceability system has to answer is whether it still shows up in
  the accumulated evidence the ZK norm bound can't suppress.
"""
from __future__ import annotations

import numpy as np


def craft_sign_flip(real_delta: np.ndarray) -> np.ndarray:
    """g_m = -g_b: negate the client's own real trained delta.

    A no-op on an all-zero delta (not VRF-sampled this round) -- that
    is the honest, correct outcome: there is nothing to flip.
    """
    return -real_delta


def craft_gaussian_noise(real_delta: np.ndarray, sigma: float, rng: "np.random.Generator | None" = None) -> np.ndarray:
    """g_m = g_b + N(0, sigma^2 I): additive Gaussian noise on top of
    the client's own real trained delta (Shi et al. 2021's "Noise
    Attack" -- as opposed to their "Random Attack", which replaces the
    delta with pure noise and is better expressed via `cheater_force_delta`
    directly, since it doesn't need the real delta at all).
    """
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.normal(0.0, sigma, size=real_delta.shape)
    return real_delta + noise


def _scale_to_norm(real_delta: np.ndarray, target_norm: float, rng: "np.random.Generator | None" = None) -> np.ndarray:
    """Shared helper: rescale `real_delta` to exactly `target_norm`,
    preserving its direction. Falls back to a random unit vector (then
    scaled) if `real_delta` is exactly zero (not VRF-sampled this
    round) -- there is no direction to preserve, and submitting the
    zero vector instead of a legitimate-looking attack would make the
    client both honest-looking (good cover) AND attack-free (useless
    for this attack's purpose), so this models "the attacker still
    tries to attack even on a round it wasn't really sampled," a
    strictly stronger (more adversarial) assumption than silently
    reverting to zero.
    """
    norm = float(np.linalg.norm(real_delta))
    if norm == 0.0:
        if rng is None:
            rng = np.random.default_rng()
        direction = rng.normal(0.0, 1.0, size=real_delta.shape)
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm == 0.0:  # astronomically unlikely, but stay exact
            direction = np.ones_like(real_delta)
            direction_norm = float(np.linalg.norm(direction))
        return direction * (target_norm / direction_norm)
    return real_delta * (target_norm / norm)


def craft_scale_to_threshold(real_delta: np.ndarray, B_t: float, safety_margin: float = 0.95, rng: "np.random.Generator | None" = None) -> np.ndarray:
    """Scale the client's own real trained delta to `safety_margin *
    B_t` -- just under this round's active R_norm bound, not wildly
    past it like A1's gamma=100. `safety_margin < 1.0` leaves headroom
    against quantization/centering rounding at the boundary (see
    `attacks/a1_scaling.py::run_attack`'s own FIX.md note on
    `quantize()`'s mod-P_MASK wraparound for why an exact boundary
    value is fragile to round off in the wrong direction); 0.95 is a
    deliberately conservative default, not tuned to any one model/dim.
    """
    return _scale_to_norm(real_delta, safety_margin * B_t, rng=rng)


def craft_low_and_slow(real_delta: np.ndarray, B_t: float, target_ratio: float = 0.3, rng: "np.random.Generator | None" = None) -> np.ndarray:
    """Scale the client's own real trained delta to `target_ratio *
    B_t`, deliberately smaller than `craft_scale_to_threshold`'s
    near-boundary value -- the bet this attack makes is that staying
    well under B_t every round also keeps the per-round §6.1 evidence
    term (`norm_q / B_t`, `fl/evidence.py::norm_only_evidence`) small
    enough that the EPOCH-accumulated total never crosses theta
    either, evading both defenses through patience rather than size.
    `target_ratio` is this attack's one real parameter -- 0.3 is a
    starting point, not a value derived from this project's own
    calibration; a real run may need to sweep it to find where (if
    anywhere) the evidence accumulator stops catching it.
    """
    return _scale_to_norm(real_delta, target_ratio * B_t, rng=rng)
