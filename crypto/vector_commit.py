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

PERFORMANCE (FIX.md F2)
-----------------------
The weights come from the AES-256-CTR expander in crypto/prg.py and the
weighted sum is done in uint64 with an exact Mersenne-prime reduction
(_dot_mod_p), chunk by chunk. The earlier version expanded the weights
with a Python SHA-256 loop and multiplied in Python big integers; the
result is the same number for the same weights, tested against the
big-integer computation in tests/test_prg.py.
"""
from __future__ import annotations

import numpy as np

from .prg import derive_stream_key, expand_field_stream, iter_field_stream

P_MASK = 2 ** 61 - 1  # MUST match secagg.py's masking modulus -- the vectors
                      # this digest is applied to already live in this domain


def _weight_key(seed: bytes) -> bytes:
    # Distinct domain tag: these weights can never collide with a masking
    # stream even if a seed were reused.
    return derive_stream_key(b"vc-weight", seed)


def _weight_stream(seed: bytes, dim: int) -> np.ndarray:
    """Deterministic stream of `dim` field elements in [0, P_MASK),
    derived from `seed` with the AES-256-CTR expander in crypto/prg.py
    (FIX.md F2)."""
    return expand_field_stream(_weight_key(seed), dim)


_M31 = np.uint64((1 << 31) - 1)
_M30 = np.uint64((1 << 30) - 1)
_M32 = np.uint64((1 << 32) - 1)


def _dot_mod_p(w: np.ndarray, v: np.ndarray) -> int:
    """Exact sum_j w_j * v_j mod P_MASK for int64 arrays with entries in
    [0, P_MASK), using only uint64 arithmetic.

    Split a = a_hi*2^31 + a_lo and b likewise (a_hi < 2^30, a_lo < 2^31):

        a*b = a_hi*b_hi*2^62 + (a_hi*b_lo + a_lo*b_hi)*2^31 + a_lo*b_lo

    P_MASK = 2^61 - 1, so 2^61 = 1 and 2^62 = 2 (mod P_MASK). Writing the
    cross term as c = c_hi*2^30 + c_lo gives c*2^31 = c_hi + c_lo*2^31.
    Every partial result stays below 2^63, so nothing wraps.
    """
    a = w.astype(np.uint64)
    b = v.astype(np.uint64)
    a_lo, a_hi = a & _M31, a >> np.uint64(31)
    b_lo, b_hi = b & _M31, b >> np.uint64(31)
    cross = a_hi * b_lo + a_lo * b_hi
    term = (a_lo * b_lo + np.uint64(2) * (a_hi * b_hi)
            + (cross >> np.uint64(30)) + ((cross & _M30) << np.uint64(31)))
    # term < 2^63 per element; add the two 32-bit halves separately so the
    # running sums cannot overflow, then finish in Python integers.
    lo = int((term & _M32).sum(dtype=np.uint64))
    hi = int((term >> np.uint64(32)).sum(dtype=np.uint64))
    return (lo + (hi << 32)) % P_MASK


def weighted_digest(vec: np.ndarray, seed: bytes) -> int:
    """sum_j w_j * vec[j] mod P_MASK, where w is derived deterministically
    from `seed` (the same seed, e.g. rho_t, used by both the committing
    client and the verifying server/auditor, so both sides derive the
    identical weight stream independently).

    `vec` is a 1-D integer array; entries are reduced into [0, P_MASK)
    before use. The weight stream is consumed chunk by chunk and never
    held in full.
    """
    vec = np.asarray(vec)
    dim = vec.shape[0]
    total = 0
    start = 0
    for weights in iter_field_stream(_weight_key(seed), dim, chunk=1 << 18):
        end = start + weights.shape[0]
        part = vec[start:end]
        if part.dtype == object:
            # arbitrary-precision input: exact reference path
            total = (total + int((weights.astype(object) * part).sum())) % P_MASK
        else:
            total = (total + _dot_mod_p(weights, np.mod(part.astype(np.int64), P_MASK))) % P_MASK
        start = end
    return total
