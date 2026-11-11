"""Randomized linear-fingerprint vector-commitment digest (ARCHITECTURE.md
§4 step 5 / the V5 aggregation-binding check). PURE-MATH MODULE -- no EC
dependency, fully testable in this sandbox (unlike pedersen.py itself,
which only wraps this digest in a single Pedersen scalar commitment and
IS EC-dependent).

THE PROBLEM THIS REPLACES
--------------------------
Committing to a d~23.5M-dimensional masked update vector with one
Pedersen commitment PER COORDINATE is impractical (d scalar
multiplications, per client, per round). An earlier placeholder in
this repo instead committed to a single "digest" scalar equal to the
vector's plain coordinate SUM (sum_j vec[j]). That is unsound: it is a
linear check with every weight fixed at 1, which is public and known
in advance -- an aggregator could substitute any OTHER d-dimensional
vector with the same coordinate sum (trivial to construct: e.g. move
some amount from one coordinate to another) and this digest would
never notice.

THE FIX
-------
Weight each coordinate by a FRESH, round-specific pseudorandom field
element before summing: digest(vec) = sum_j w_j * vec[j] mod P_MASK,
where w is derived deterministically from a per-round public seed
(rho_t -- already revealed to every client at the start of the round,
see fl/client.py's round_step_phase1 signature) via a seeded PRG.

This turns the digest into a Schwartz-Zippel / randomized
polynomial-identity-testing fingerprint: for any two DIFFERENT
d-dimensional vectors u != v, a uniformly random weight vector w makes
digest(u) == digest(v) happen with probability at most d/P_MASK
(P_MASK ~ 2^61, so astronomically small for any realistic d), REGARDLESS
of how u and v were chosen -- AS LONG AS w was fixed independently of
which (u, v) pair an adversary is trying to collide.

That independence holds for V5's actual threat model (see
verifier/audit.py's docstring): the party being checked is a
potentially-dishonest SERVER/AGGREGATOR, substituting a different
*aggregate* vector than the one honestly implied by the already-public
per-client commitments. rho_t is fixed and public from the START of
the round -- before the server has aggregated anything, let alone
decided whether to misreport the result -- so the server has no
freedom to choose a favorable w after the fact. (It is NOT a defense
against a malicious client's own choice of what to train/submit --
that is covered by the norm-bound range proof and R_cons, not V5.)

LINEARITY (why the Pedersen homomorphism still lines up)
----------------------------------------------------------
Because every client's digest in a given round uses the SAME weight
vector w (same seed, same dim), the digest function is linear:
    digest(u) + digest(v) == digest(u + v)   (mod P_MASK)
and secagg.py's aggregate_masked() sums client vectors coordinatewise
mod P_MASK too -- the SAME reduction -- so
    sum_i digest(masked_i) mod P_MASK == digest(aggregate_masked_vector) mod P_MASK
exactly. Each digest_i < P_MASK (~2^61); summing up to n_clients=50 of
them never approaches L (~2^252, ristretto255's group order), so the
Pedersen commitments' GROUP-level sum (mod L, via repeated
crypto_core_ristretto255_scalar_add) equals the same integer exactly,
not merely congruent mod L. That is what makes
tests/test_vector_commit.py's linearity check a faithful stand-in for
the full Com(a,r)+Com(b,s) == Com(a+b,r+s) property without needing
pynacl to verify it.

*** PERFORMANCE NOTE, same caveat as sketch.py's SketchProjector ***
This reference does the weighted sum with numpy using Python-object
(arbitrary-precision) arithmetic for the final accumulation, since
weight*value products exceed int64/float64 range. That is correctness-
tested here, at moderate d, but NOT the fast path for d~23.5M -- same
GPU/torch reimplementation note as sketch.py applies (see STATUS.md):
the production client.py should do this with the same chunked
PRG-plus-batched-modmul pattern, on GPU, not this reference loop.
"""
from __future__ import annotations

import hashlib

import numpy as np

P_MASK = 2 ** 61 - 1  # MUST match secagg.py's masking modulus -- the vectors
                      # this digest is applied to already live in this domain


def _weight_stream(seed: bytes, dim: int) -> np.ndarray:
    """Deterministic stream of `dim` field elements in [0, P_MASK),
    derived from `seed` via chunked SHA-256 expansion -- the identical
    construction to secagg.py's _prg (same chunk size, same
    uint64-then-reduce approach), reused here under a distinct domain-
    separation tag ("vc-weight") so these weights never collide with
    any masking PRG output even if a seed were accidentally reused.
    """
    out = np.empty(dim, dtype=np.int64)
    counter = 0
    filled = 0
    buf = b""
    while filled < dim:
        if len(buf) < 8 * 4096:
            buf += hashlib.sha256(seed + b"||vc-weight||" + counter.to_bytes(8, "little")).digest()
            counter += 1
            continue
        take = min(4096, dim - filled)
        chunk_bytes = buf[: take * 8]
        buf = buf[take * 8 :]
        raw = np.frombuffer(chunk_bytes, dtype=np.uint64)[:take]
        vals = (raw % np.uint64(P_MASK)).astype(np.int64)
        out[filled : filled + take] = vals
        filled += take
    return out


def weighted_digest(vec: np.ndarray, seed: bytes) -> int:
    """sum_j w_j * vec[j] mod P_MASK, where w is derived deterministically
    from `seed` (the same seed, e.g. rho_t, used by both the committing
    client and the verifying server/auditor, so both sides derive the
    identical weight stream independently).

    `vec` must already be a 1-D integer array with entries in
    [0, P_MASK) -- exactly the domain fl/client.py's double-masking step
    (and secagg.py's aggregate_masked) already produces.
    """
    vec = np.asarray(vec)
    dim = vec.shape[0]
    weights = _weight_stream(seed, dim)
    total = 0
    chunk = 1 << 16
    for start in range(0, dim, chunk):
        end = min(start + chunk, dim)
        # object dtype: weight*value can exceed 2^122, past int64/float64 range
        prod = weights[start:end].astype(object) * vec[start:end].astype(object)
        total = (total + int(prod.sum())) % P_MASK
    return total
