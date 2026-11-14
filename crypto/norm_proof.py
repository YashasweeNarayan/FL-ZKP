"""Hidden-sketch norm proof -- FIX.md F7 (V3 redesign). EC-DEPENDENT.

ARCHITECTURE.md's V3 check is supposed to be a ZK verification,
`Verify(pi_norm, c_k, B_t)`, that never sees the client's sketch. The
MVP as shipped instead has the server call `np.linalg.norm` on the
CLEARTEXT sketch (`fl/server.py::check_V3_norm_bound`). This module
closes that gap for real: a committed sketch, a proof that the
committed value is the sketch's squared norm, and an aggregate check
that never requires any client's individual sketch or blinding factor
to be revealed.

*** DESIGN NOTE, written during F7's implementation (2026-11-13),
replacing part of FIX.md's original F7 plan -- read this before
touching F7b/c. ***

The original plan said "vector Pedersen commitment to [the sketch]"
(one point, m=8192 custom generators, see `ristretto.py::msm`) and
"Sigma protocol for <y,y>=v on the committed sketch". Those two don't
fit together. Proving an inner product of a vector held inside ONE
multi-generator commitment needs an inner-product argument -- that is
what Bulletproofs' O(log m) argument is FOR, and `schnorr_nizk.py`'s
own docstring already made the call, under the 3-day deadline, to stay
inside plain discrete-log Sigma protocols and accept O(witnesses)
proof size instead of building that toolchain. Trying to prove <y,y>=v
over the multi-generator commitment without it would mean inventing an
inner-product argument from scratch, under this project's exact
"never fabricate crypto" rule -- not safe to do under time pressure,
so it's not done here.

What IS a genuine Sigma protocol, buildable from the existing
`schnorr_nizk.py` toolkit: proving a SINGLE committed value is the
square of another SINGLE committed value (`SquareProof` below -- the
standard Pedersen "proof of a multiplicative relation": pick random
(k, s1, s2), let D = k*G + s1*H and E = k*C_y + s2*H, answer
u = k + e*y, v1 = s1 + e*r_y, v2 = s2 + e*(r_c - y*r_y); the verifier's
two checks fall out of the commitment algebra -- see `prove_square`'s
docstring for the worked identity). So the sketch is committed
COORDINATE BY COORDINATE: m=8192 ordinary Pedersen commitments to
y_1..y_m (not one multi-generator point), each paired with a
commitment to y_i^2 and a `SquareProof` tying the two together. The
sum of the y_i^2 commitments is revealed only as its OPENING TO v
(the aggregate blinding, not any individual commitment's blinding) --
exactly the "only the sum is ever revealed" pattern V5 already uses,
and exactly what keeps y itself hidden while making v = <y,y> public
(v has to be public: the server's accept/reject decision is a
function of it).

Honest cost: m=8192 square proofs is roughly 8x the scalar
multiplications of the existing per-client rangeproof.py bit proofs on
one 32-bit value. That is a real increase to phase-2 crypto time, not
a free lunch -- see FIX.md's F7 entry for the measured number once
this runs on Colab. It keeps this module inside the project's already
-declared "O(witnesses), not O(log)" tradeoff; it does not reopen the
Bulletproofs/circuits scope decision.

F7d (subset-sum wraparound test), F7e (aggregate-opens-to-aggregate
check) and F7f (group-testing blame) are below the square proof.
"""
from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field

from .field import ShamirShare, reconstruct as field_reconstruct
from .pedersen import H as PEDERSEN_H, Commitment, commit, commit_sum, verify_opening
from .ristretto import Point, Scalar, fiat_shamir_challenge, group_order, scalar_mul_base

G = Point.base()


def _scalar_signed(x: int) -> Scalar:
    """Scalar.from_int only accepts x >= 0 (see its docstring); sketch
    coordinates here are SIGNED (y = S*q, q already centered), so
    reduce explicitly first. x % L is always in [0, L) in Python
    regardless of x's sign."""
    return Scalar.from_int(x % group_order())


# ---------------------------------------------------------------------------
# F7c: proof that a committed value is the square of another committed
# value, without revealing either.
# ---------------------------------------------------------------------------

@dataclass
class SquareProof:
    D: Point
    E: Point
    u: Scalar
    v1: Scalar
    v2: Scalar


def prove_square(y: Scalar, r_y: Scalar, r_c: Scalar, transcript: bytes,
                  k: Scalar | None = None, s1: Scalar | None = None, s2: Scalar | None = None) -> SquareProof:
    """Proves knowledge of (y, r_y, r_c) with C_y = Com(y; r_y) and
    C_c = Com(y*y; r_c), for the SAME y in both, without revealing y.

    Worked identity (why the verifier's two checks are implied by a
    true statement, and only by one -- special-soundness against a
    cheating prover follows the usual two-transcript Schnorr argument):

      C_y^u = (y*G + r_y*H)^u = (u*y)*G + (u*r_y)*H
      u*y = (k + e*y)*y = k*y + e*y^2
      so  C_y^u = (k*y)*G + (e*y^2)*G + (u*r_y)*H
                = (k*y)*G + e*(C_c - r_c*H) + (u*r_y)*H          [y^2*G = C_c - r_c*H]
                = (k*y)*G + e*C_c + (u*r_y - e*r_c)*H
      E = k*C_y + s2*H = (k*y)*G + (k*r_y + s2)*H
      so (k*y)*G = E - (k*r_y + s2)*H, giving
         C_y^u = E + e*C_c + (u*r_y - e*r_c - k*r_y - s2)*H
      u*r_y - k*r_y = e*y*r_y, so the H-exponent is
         e*(y*r_y - r_c) - s2 = -v2   (since v2 = s2 + e*(r_c - y*r_y))
      i.e.  C_y^u + v2*H == E + e*C_c.   <- check (ii) below
      The symmetric, simpler identity C_y^e * D == G^u * H^v1 is the
      ordinary Schnorr check that (D, u, v1) opens C_y to y.          <- check (i)
    """
    k = k if k is not None else Scalar.random()
    s1 = s1 if s1 is not None else Scalar.random()
    s2 = s2 if s2 is not None else Scalar.random()
    C_y = commit(y, r_y).point
    D = scalar_mul_base(k) + PEDERSEN_H.scalar_mul(s1)
    E = C_y.scalar_mul(k) + PEDERSEN_H.scalar_mul(s2)
    e = fiat_shamir_challenge(transcript, C_y.b, D.b, E.b)
    u = k + e * y
    v1 = s1 + e * r_y
    v2 = s2 + e * (r_c - y * r_y)
    return SquareProof(D, E, u, v1, v2)


def verify_square(proof: SquareProof, C_y: Point, C_c: Point, transcript: bytes) -> bool:
    e = fiat_shamir_challenge(transcript, C_y.b, proof.D.b, proof.E.b)
    check1 = scalar_mul_base(proof.u) + PEDERSEN_H.scalar_mul(proof.v1) == proof.D + C_y.scalar_mul(e)
    check2 = C_y.scalar_mul(proof.u) + PEDERSEN_H.scalar_mul(proof.v2) == proof.E + C_c.scalar_mul(e)
    return check1 and check2


# ---------------------------------------------------------------------------
# Per-client sketch commitment: m coordinate commitments + m square
# proofs + the one public scalar v = <y,y>.
# ---------------------------------------------------------------------------

@dataclass
class SketchCommitment:
    y_commitments: list[Commitment]     # C_{y_i} = Com(y_i; r_i),      i = 1..m
    sq_commitments: list[Commitment]    # C_{y_i^2} = Com(y_i^2; r'_i), i = 1..m
    square_proofs: list[SquareProof]
    v: int                              # claimed <y,y>, PUBLIC
    aggregate_blinding: Scalar          # sum_i r'_i -- opens sum_i C_{y_i^2} to v


def commit_sketch(y: list[int], transcript: bytes) -> tuple[SketchCommitment, list[Scalar], list[Scalar]]:
    """Commits an exact-integer sketch (crypto/sketch.py's
    `project_int`/`project_int_batch` output) coordinate by coordinate.
    Returns the public SketchCommitment plus the two blinding-factor
    lists (y_blindings, sq_blindings) the CLIENT keeps secret -- F7e
    needs their per-coordinate SUMS across clients, routed through
    secure aggregation, never any one client's own list.
    """
    y_blindings = [Scalar.random() for _ in y]
    sq_blindings = [Scalar.random() for _ in y]
    y_commitments = []
    sq_commitments = []
    square_proofs = []
    v = 0
    for i, (yi, r_y, r_c) in enumerate(zip(y, y_blindings, sq_blindings)):
        y_scalar = _scalar_signed(yi)
        sq_scalar = _scalar_signed(yi * yi)
        c_y = commit(y_scalar, r_y)
        c_c = commit(sq_scalar, r_c)
        proof = prove_square(y_scalar, r_y, r_c, transcript + i.to_bytes(4, "little"))
        y_commitments.append(c_y)
        sq_commitments.append(c_c)
        square_proofs.append(proof)
        v += yi * yi
    aggregate_blinding = sq_blindings[0]
    for r_c in sq_blindings[1:]:
        aggregate_blinding = aggregate_blinding + r_c
    sc = SketchCommitment(y_commitments, sq_commitments, square_proofs, v, aggregate_blinding)
    return sc, y_blindings, sq_blindings


def verify_sketch_commitment(sc: SketchCommitment, transcript: bytes, norm_bound_sq: int) -> bool:
    """Server-side check: every per-coordinate square proof is valid,
    the revealed v is consistent with the revealed aggregate_blinding
    opening sum_i C_{y_i^2} (NOT any individual commitment -- v is the
    only thing about y this ever has to learn), and v is within the
    public integer bound (`crypto.sketch.int_sketch_norm_sq_bound`).
    Does NOT see any y_i, y_i^2, or individual blinding factor.
    """
    m = len(sc.y_commitments)
    if not (len(sc.sq_commitments) == len(sc.square_proofs) == m):
        return False
    if sc.v < 0 or sc.v > norm_bound_sq:
        return False
    for i, (c_y, c_c, proof) in enumerate(zip(sc.y_commitments, sc.sq_commitments, sc.square_proofs)):
        if not verify_square(proof, c_y.point, c_c.point, transcript + i.to_bytes(4, "little")):
            return False
    total = commit_sum(sc.sq_commitments)
    if not verify_opening(total, _scalar_signed(sc.v), sc.aggregate_blinding):
        return False
    return True


# ---------------------------------------------------------------------------
# F7d: cheap probabilistic wraparound test, replacing the original
# plan's bit-decomposition range proof.
#
# `fl/client.py` centers the quantized update mod P_MASK (~2^61) before
# sketching; a malicious client can choose a representative that wraps
# (adds/subtracts a multiple of P_MASK coordinate-wise) so the exact
# integer sketch is computed over a DIFFERENT integer vector than the
# one that will actually be reconstructed after unmasking, while still
# passing every check above (those only constrain the sketch's OWN
# algebra, not its relationship to P_MASK). ristretto255's group order
# L is ~2^252; P_MASK is ~2^61 -- about 2^191 of slack. That slack is
# exactly what makes a cheap test possible instead of a full per-bit
# range proof: flood a random LINEAR COMBINATION of the (secret,
# committed) coordinates with enough entropy that a coordinate sitting
# one P_MASK-multiple off is caught with overwhelming probability, and
# reuse the SAME square-proof machinery above to prove the flooded
# combination's value without revealing it.
# ---------------------------------------------------------------------------

def subset_sum_challenge(seed: bytes, m: int, n_tests: int = 8) -> list[list[int]]:
    """Derives `n_tests` public +-1 weight vectors (length m) from
    `seed` (the already-public sigma_t / round seed -- never chosen by
    the client, so it cannot be gamed). Each test checks
    sum_i w_i * y_i for one weight vector w.
    """
    import hashlib
    tests = []
    for t in range(n_tests):
        h = hashlib.sha256(seed + b":wrap:" + t.to_bytes(4, "little")).digest()
        rng_state = int.from_bytes(h, "little")
        weights = []
        for i in range(m):
            rng_state = (rng_state * 6364136223846793005 + 1442695040888963407) % (2 ** 64)
            weights.append(1 if (rng_state >> 63) else -1)
        tests.append(weights)
    return tests


def wraparound_test_values(y: list[int], weights_list: list[list[int]]) -> list[int]:
    """TEST/DIAGNOSTIC helper only -- returns the flooded sums in the
    CLEAR. The real client path (`prove_wraparound_binding` below)
    computes the same sums internally and never exposes them; this
    function exists so a test can sanity-check the catching property
    (section below) without needing the EC-dependent commitment
    machinery. A wrapped coordinate shifts its test sums by a multiple
    of P_MASK, which `wraparound_window`'s quant_bits-derived (not
    P_MASK-derived) bound is tight enough to actually notice -- see
    that function's docstring for why the bound has to be tight.
    """
    out = []
    for weights in weights_list:
        s = sum(w * yi for w, yi in zip(weights, y))
        out.append(s)
    return out


def wraparound_window(m: int, model_dim: int, density: float, quant_bits: int) -> int:
    """Half-width of the window a HONEST client's flooded test sums
    must land in -- MUST be a PUBLIC quantity, computed the same way by
    client and server, from parameters neither side can choose
    unilaterally (m, model_dim, density, quant_bits are all config,
    known before the round starts).

    *** Caught during implementation: an earlier draft of this
    function took `max_coord_abs` from the CLIENT's own array
    (np.max(np.abs(signed_q))) and bounded it by P_MASK//2 (the
    structural ceiling center_mod_p_mask's unwrap CAN produce). Both
    of those are wrong: the first can't be independently recomputed by
    the server at all (it depends on secret data), and the second is
    far too loose -- an HONEST q_j is nowhere near P_MASK//2 (~2^61);
    it is bounded by the QUANTIZER's own range, ~2^(quant_bits-1)
    (~2^15 for the default 16-bit quantization), tens of orders of
    magnitude smaller. Using the loose P_MASK//2 bound would make a
    P_MASK-sized wraparound attack (~2^61) invisible against an
    "honest" window that already allows values that large or larger.
    Using the tight, quant_bits-derived bound instead makes a
    P_MASK-sized shift in even ONE coordinate dwarf the honest window
    (by roughly 2^61 / (model_dim*density*2^15), still >> 1 for any
    realistic model_dim), which is what makes this test catch anything
    at all. ***

    max|q_j| <= 2**(quant_bits-1)  (the quantizer's own representable range)
    max|y_i| <= ceil(model_dim*density) * max|q_j|   (at most that many nonzero S-row entries)
    window    = m * max|y_i|                          (+-1 weighted sum of m such terms)
    """
    import math
    max_q_abs = 2 ** (quant_bits - 1)
    max_y_abs = math.ceil(model_dim * density) * max_q_abs
    return m * max_y_abs


# -- binding a hidden test value to the already-published per-coordinate
# commitments (caught, during implementation, the same way F7b/c's
# original plan was caught: `wraparound_test_values` above returns the
# flooded sums in the CLEAR, which leaks up to n_tests extra linear
# combinations of y about q -- small but real, and avoidable. y_i's
# range is bounded by the norm check itself, i.e. small enough to be
# recoverable by brute-force discrete log, so this is NOT the same as
# V5's existing "reveal r_k in the clear" pattern (digest_k there lives
# in the full ~2^61 P_MASK range, where that brute force is infeasible)
# -- it is the SAME oracle-attack shape F7e's blinding reveal has to
# avoid, just for a different value. Fix: never reveal a test value or
# its blinding directly. The server already holds C_{y_i} for every i
# (published in SketchCommitment), so for public weights w it can
# compute W = sum_i w_i*C_{y_i} = Com(s; R) for s = sum_i w_i*y_i and
# R = sum_i w_i*r_i, WITHOUT the client doing anything -- `W` is a
# homomorphic sum of ALREADY-PUBLISHED points. The client separately
# commits to the SAME s under a FRESH, unrelated blinding (so this
# commitment's own blinding, if later revealed for a range proof,
# reveals nothing about any r_i), and proves in zero-knowledge that
# its fresh commitment and W commit to the same value -- a single
# Schnorr proof of knowledge of (r' - R) as the discrete log, base H,
# of (C' - W): if C' and W open to the same value, that difference is
# exactly (r' - R)*H and nothing else cancels. ***

from .schnorr_nizk import SchnorrProof, schnorr_prove, schnorr_verify


def _weighted_commitment_sum(weights: list[int], commitments: list[Commitment]) -> Point:
    """W = sum_i weights[i] * C_i, weights public and in {-1,0,1} here
    (subset_sum_challenge only emits +-1, but this does not assume
    that). A plain loop, not `ristretto.msm` on C_i.point -- msm takes
    raw Points, Commitments wrap one, so unwrap first."""
    from .ristretto import msm
    return msm(weights, [c.point for c in commitments])


def prove_wraparound_binding(y: list[int], y_blindings: list[Scalar], weights: list[int],
                              window: int, transcript: bytes,
                              shifted_value_commitment: Point, r_prime: Scalar) -> SchnorrProof:
    """Client side, for ONE test. `shifted_value_commitment` and
    `r_prime` MUST be `pedersen_commit(Scalar(s + window), r_prime)`
    for the same `r_prime` the caller is about to hand to (or just got
    from) `rangeproof.prove_range` as its `aggregate_blinding` -- NOT a
    freshly generated blinding of this function's own choosing.

    That requirement is exactly what makes the three pieces
    (this Schnorr proof, the range proof, and `y_commitments`) tie
    together into one statement instead of three independent ones: the
    range proof's own `verify_recomposition` already requires an
    external commitment built from ITS `aggregate_blinding`
    (rangeproof.py's own docstring), so if this function picked its
    OWN fresh blinding instead, the range proof would attest to the
    range of a value with no verified connection to the sketch at all
    -- a client could pass V3's wraparound test with an arbitrary
    in-range number, defeating the test. Caught and fixed before this
    was wired into fl/client.py: see that call site for the required
    order (prove_range first, then this).
    """
    # Scalar has no int-mult operator, so build R = sum w_i*r_i via repeated
    # addition/negation of r_i (weights are in {-1,0,1} here).
    R = None
    for w, r in zip(weights, y_blindings):
        term = r if w == 1 else (r.negate() if w == -1 else None)
        if term is None:
            continue
        R = term if R is None else R + term
    if R is None:
        R = Scalar.from_int(0)
    d = r_prime - R
    return schnorr_prove(d, PEDERSEN_H, transcript)


def verify_wraparound_binding(C_prime: Point, proof: SchnorrProof, weights: list[int],
                               y_commitments: list[Commitment], window: int, transcript: bytes) -> bool:
    """Server side: recomputes W from the already-published
    `y_commitments` and `weights` (both public), then checks the
    Schnorr proof against P = C' - W - window*G. No test value, no
    blinding, and no y_i is used or learned here."""
    W = _weighted_commitment_sum(weights, y_commitments)
    P = C_prime - W - scalar_mul_base(Scalar.from_int(window % group_order()))
    return schnorr_verify(proof, P, PEDERSEN_H, transcript)


# ---------------------------------------------------------------------------
# F7e: aggregate check -- the per-coordinate sum of accepted clients'
# sketches must equal the sketch of the aggregate quantized update,
# WITHOUT any accepted client's individual blinding (or sketch) ever
# being revealed on its own.
#
# This mirrors the existing V5 pattern in fl/server.py / crypto/secagg.py
# exactly: a blinding factor is only ever revealed as part of a SUM that
# the double-masking mechanism (secagg.py's self-mask / pairwise-mask
# shares) assembles, never individually. The function below is the
# verification half; `mask_blinding_vector` / `reconstruct_blinding_sum`
# below are the mechanism that produces the sum.
#
# THE MECHANISM (and why the obvious shortcut is wrong). The server needs
# sum_k r_{k,i} for each coordinate i, to open the product of the accepted
# clients' per-coordinate commitments. It must NOT learn any single
# r_{k,i}: a sketch coordinate's value range is bounded by the norm check
# itself, so C_{y_i} together with that client's own r_{k,i} gives
# y_i*G with y_i small enough to recover by brute-force discrete log.
# (V5's existing per-client `r_k` reveal is safe only because
# `weighted_digest`'s range is the full ~2^61 P_MASK space -- that
# argument does not transfer here, which is exactly the trap the first
# draft of this module fell into.)
#
# So the blinding vector gets the SAME Bonawitz pairwise masking the
# quantized update already gets, one field up: each client publishes
#     rho_{k,i} = r_{k,i} + sum_{j != k} sign(k,j) * PRF(pair_seed_kj, i)   (mod L)
# in phase 2, BEFORE the accepted set is known (same ordering as the
# masked vectors). Summed over all clients the pairwise terms cancel;
# summed over a strict subset, the pairs straddling the boundary do not,
# and are stripped with the SAME `pairwise_round_seed` values accepted
# clients already release about excluded ones for `unmask_aggregate`.
# The mask stream comes from `crypto.prg.expand_scalar_stream` under its
# own domain tag, so it is a different keystream from the vector mask
# derived from the same pair seed.
#
# What this deliberately does NOT have, and the consequence: no
# self-mask, and therefore no Shamir dropout recovery. For the UPDATE
# vector, secagg's self-mask is what stops a server from isolating one
# client by claiming its partners dropped. Here the blinding sum is
# taken over the accepted set, which is fixed and published BEFORE any
# rho_{k,i} is used, and k_min (=8) floors its size -- so the smallest
# sum the server can obtain through the honest path already covers 8
# clients. The one place smaller sums are legitimately computed is F7f's
# localization below, which is bounded by its own `min_group_size`
# floor.
#
# DROPOUT RESILIENCE (added 2026-11-14, closing a gap FIX.md had logged
# as open): the blinding vector originally had ONLY the pairwise mask
# above, so a client that dropped after acceptance made that round's
# F7e check incomputable -- there was nothing else keeping an
# individual r_{k,i} hidden if the pairwise terms alone were stripped
# via excluded-pair seeds, so the design simply never asked a dropped
# client for anything and reported `None` instead. secagg.py's update
# vector never had this problem because a SELF-mask, Shamir-shared
# among the OTHER clients at the start of the round, backs every
# accepted client's contribution -- unmasking only ever needs
# `share_t` of those PRE-DISTRIBUTED shares, never the client itself
# being reachable at verify time. `blinding_self_mask` /
# `recover_blinding_self_mask` below add the exact same layer, one
# field up (Z_L): each client draws a second self-mask secret (over
# Z_L, so Shamir-sharing it reconstructs exactly), Shamir-shares it the
# same way at the start of the round, and adds `blinding_self_mask` on
# top of its own r_{k,i} values before masking. The server recovers it
# from `share_t` shares via `reconstruct_blinding_sum`'s new
# `self_mask_shares` argument, exactly mirroring `unmask_aggregate`'s
# per-accepted-client loop. A post-acceptance dropout no longer makes
# F7e un-runnable, for the same reason the update path was never
# affected by one.
# ---------------------------------------------------------------------------

from .prg import derive_stream_key, expand_scalar_stream


def blinding_pairwise_mask(pair_round_seed: bytes, m: int, sign: int) -> list[int]:
    """One pair's blinding mask, as m ints mod L. `sign` is +1 for the
    lower-indexed party and -1 for the higher, matching
    `secagg.pairwise_mask`'s convention exactly -- the two paths must
    agree on sign or neither cancels.

    Derived from the pair's ROUND seed (not the long-term secret), for
    the same reason `secagg.pairwise_mask` does it that way: the server
    can reproduce this exact stream from a released round seed alone
    when stripping an excluded partner's mask, without ever seeing the
    long-term secret.
    """
    L = group_order()
    stream = expand_scalar_stream(derive_stream_key(b"f7e-blinding-mask", pair_round_seed, b""), m)
    if sign == 1:
        return stream
    return [(-x) % L for x in stream]


def blinding_self_mask(self_secret_bytes: bytes, round_nonce: bytes, m: int) -> list[int]:
    """A client's own private mask on its blinding vector -- m ints mod
    L, summed INTO `mask_blinding_vector`'s output but never cancelled
    by any other client's contribution (mirrors `secagg.self_mask`, one
    field up). `self_secret_bytes` must be 32 bytes; recoverable only
    via `share_t` Shamir shares of the secret integer it came from
    (`recover_blinding_self_mask` below), never by combining other
    clients' pairwise data.
    """
    return expand_scalar_stream(
        derive_stream_key(b"f7e-blinding-self-mask", self_secret_bytes, round_nonce), m,
    )


def recover_blinding_self_mask(shares: list[ShamirShare], round_nonce: bytes, m: int) -> list[int]:
    """Server side. Reconstructs one client's blinding self-mask secret
    from >= share_t of its Shamir shares (same shares and threshold the
    client used for its UPDATE self-mask secret are never reused here --
    this is a second, distinct secret, Shamir-shared separately -- see
    `fl/client.py`'s wiring) and expands it back to the mask vector.
    """
    secret_int = field_reconstruct(shares)
    secret_bytes = secret_int.to_bytes(32, "little")
    return blinding_self_mask(secret_bytes, round_nonce, m)


def mask_blinding_vector(y_blindings: list[Scalar], own_index: int,
                          pairwise_round_seeds: dict[int, bytes],
                          self_secret_bytes: bytes, round_nonce: bytes) -> list[int]:
    """Client side, phase 2. `pairwise_round_seeds` is
    {other_client_index: pairwise_round_seed(shared_secret, rho_t)} --
    the same per-round seeds secagg's vector path uses.
    `self_secret_bytes` is this client's OWN blinding self-mask secret
    (32 bytes, from a fresh `crypto.secagg.new_self_mask_secret()` drawn
    for the blinding vector specifically, not reused from the update
    vector's secret). Returns the masked blinding vector (m ints mod L)
    that goes on the submission; the raw `y_blindings` never leave the
    client.
    """
    L = group_order()
    acc = [r.to_int() % L for r in y_blindings]
    m = len(acc)
    self_mask_vec = blinding_self_mask(self_secret_bytes, round_nonce, m)
    acc = [(a + b) % L for a, b in zip(acc, self_mask_vec)]
    for other_idx, seed in pairwise_round_seeds.items():
        sign = 1 if own_index < other_idx else -1
        mask = blinding_pairwise_mask(seed, m, sign)
        acc = [(a + b) % L for a, b in zip(acc, mask)]
    return acc


def reconstruct_blinding_sum(masked_by_client: dict, accepted: list[int],
                              excluded_pair_seeds: dict, m: int,
                              self_mask_shares: dict, share_threshold: int,
                              round_nonce: bytes) -> list[Scalar]:
    """Server side. Sums the accepted clients' published masked blinding
    vectors, removes each accepted client's blinding self-mask (via
    `share_threshold` Shamir shares of it -- pre-distributed at the
    start of the round, so this works even if that client has since
    dropped out), and strips the pairwise terms that did not cancel
    because one party was excluded -- mirroring `secagg.unmask_aggregate`
    exactly (same two-step structure, same seeds, same sign rule, same
    validation that a seed came from an accepted client about an
    excluded one).

    self_mask_shares -- {k: [ShamirShare, ...]} for every k in
                        `accepted`, at least `share_threshold` distinct
                        shares each (fewer silently reconstructs a WRONG
                        secret, which would corrupt the sum with no
                        error -- so this raises instead, same as
                        `secagg.unmask_aggregate`).

    Returns sum_k r_{k,i} as m `Scalar`s -- the ONLY thing about any
    client's blindings that ever becomes visible.
    """
    L = group_order()
    accepted_set = set(accepted)
    acc = [0] * m
    for k in accepted:
        row = masked_by_client[k]
        if len(row) != m:
            raise ValueError(f"client {k}: masked blinding vector has length {len(row)}, expected {m}")
        acc = [(a + b) % L for a, b in zip(acc, row)]
    for k in accepted:
        distinct = list({sh.index: sh for sh in self_mask_shares.get(k, [])}.values())
        if len(distinct) < share_threshold:
            raise ValueError(
                f"client {k}: only {len(distinct)} distinct blinding self-mask shares, "
                f"need {share_threshold} -- cannot reconstruct this round"
            )
        self_mask_vec = recover_blinding_self_mask(distinct[:share_threshold], round_nonce, m)
        acc = [(a - b) % L for a, b in zip(acc, self_mask_vec)]
    for (j, k), seed in excluded_pair_seeds.items():
        if j not in accepted_set or k in accepted_set:
            raise ValueError(f"pair ({j}, {k}): seed must come from an accepted client about an excluded one")
        sign = 1 if j < k else -1
        mask = blinding_pairwise_mask(seed, m, sign)
        acc = [(a - b) % L for a, b in zip(acc, mask)]
    return [Scalar.from_int(x) for x in acc]


def verify_aggregate_sketch(y_commitments_by_client: list[list[Commitment]],
                             revealed_sum_blindings: list[Scalar],
                             true_aggregate_sketch: list[int]) -> tuple[bool, list[int]]:
    """revealed_sum_blindings[i] is sum_k r_{k,i} for accepted clients k
    (released via secagg, never any one client's r_{k,i} alone).
    true_aggregate_sketch is S(sum_k q_k) recomputed by the server from
    the real post-unmask aggregate (available once V5 passes -- this
    check is downstream of F1/F5, same dependency the original plan
    already noted). Returns (ok, bad_coordinate_indices) -- the indices
    are PUBLIC regardless of outcome (they are just coordinates of a
    public sketch vector, not client identities), and are the input to
    F7f's localization below.
    """
    m = len(true_aggregate_sketch)
    bad = []
    for i in range(m):
        per_client_c = [cs[i] for cs in y_commitments_by_client]
        total = commit_sum(per_client_c)
        if not verify_opening(total, _scalar_signed(true_aggregate_sketch[i]), revealed_sum_blindings[i]):
            bad.append(i)
    return (len(bad) == 0), bad


# ---------------------------------------------------------------------------
# F7f: group-testing blame, replacing the original plan's
# "abort without attribution" (a live DoS: one anonymous malicious
# client could fail F7e every round forever with no way to exclude it).
#
# When F7e fails, bisect the accepted-client set using a PUBLIC,
# round-seeded (not client-chosen) partition, and re-run the same
# per-coordinate opening check on each half using that half's own
# blinding sum (also secagg-routed). Recurse into whichever half(s)
# fail, like combinatorial group testing. Stop and report a SET, not a
# single client, once the live set reaches `min_group_size` --
# consistent with ARCHITECTURE.md §6.3's accountability-anonymity
# budget: this localizes a fault to a small group, not to one identity,
# and the server decides what to do with that group (e.g. require them
# to re-submit individually revealed openings next round, outside this
# module's scope).
# ---------------------------------------------------------------------------

def partition_group(client_ids: list[int], seed: bytes, min_group_size: int = 4) -> tuple[list[int], list[int]] | None:
    """Public, round-seeded 50/50 split of `client_ids`. None if the
    group is already at or below the privacy floor (caller should stop
    bisecting)."""
    if len(client_ids) <= min_group_size:
        return None
    import hashlib
    ordered = sorted(client_ids)
    keyed = []
    for cid in ordered:
        h = hashlib.sha256(seed + b":part:" + cid.to_bytes(8, "little")).digest()
        keyed.append((h, cid))
    keyed.sort()
    half = len(keyed) // 2
    left = [cid for _, cid in keyed[:half]]
    right = [cid for _, cid in keyed[half:]]
    return left, right


@dataclass
class BlameResult:
    suspect_groups: list[list[int]]   # disjoint groups, each >= min_group_size (or the whole input)
    rounds_used: int


@dataclass
class BisectionSession:
    """Resumable form of the group-testing bisection (FIX.md F7f "still
    open" item: this used to be ONE blocking Python call,
    `localize_fault` below, run synchronously inside `run_round` --
    fine for this simulation, where every "client" is reachable
    in-process, but not a faithful model of a real deployment, where
    EACH step is a genuine network round trip: the server asks one
    group's members to release their straddling pair seeds and
    blinding shares (`fl/client.py::release_straddling_pair_seeds`),
    which can take real wall-clock time, time out, or be refused.

    This class pulls the bisection's control-flow state (which groups
    are still queued, which are already suspect, how many rounds have
    run) out of the call stack and into ordinary, inspectable,
    trivially-serializable fields, so a caller can suspend between
    steps for as long as a real round actually takes -- persist this
    object, come back later with the query's answer, and resume -- and
    is never left re-deriving where the bisection was partway through.

    Usage, one step at a time:
        session = BisectionSession.start(accepted_indices, coordinate, seed)
        while not session.is_done():
            group = session.pending_group()
            # <-- a real protocol round happens here: ask `group`'s
            #     members for their straddling material, verify it,
            #     however long that actually takes.
            session.submit_result(group, opened)
        blame = session.result

    `localize_fault` is the synchronous convenience wrapper for a
    caller that doesn't need to suspend (every existing caller in this
    repo, since the simulation has no real transport to suspend
    across) -- it drives this exact state machine to completion in one
    call, so the two give IDENTICAL results
    (test_bisection_session_matches_localize_fault_step_by_step in
    tests/test_norm_proof_bisection.py drives them side by side, out of
    order, to confirm it).
    """
    coordinate: int
    seed: bytes
    min_group_size: int = 4
    max_rounds: int = 10
    frontier: list = field(default_factory=list)
    suspects: list = field(default_factory=list)
    rounds_used: int = 0
    _awaiting: list | None = field(default=None, repr=False)

    @classmethod
    def start(cls, client_ids: list[int], coordinate: int, seed: bytes,
              min_group_size: int = 4, max_rounds: int = 10) -> "BisectionSession":
        return cls(coordinate=coordinate, seed=seed, min_group_size=min_group_size,
                   max_rounds=max_rounds, frontier=[list(client_ids)])

    def is_done(self) -> bool:
        return self._awaiting is None and (not self.frontier or self.rounds_used >= self.max_rounds)

    def pending_group(self) -> list[int] | None:
        """The group this session is waiting on an answer for -- query
        it (in a real deployment: send the request, wait for the
        reply, however long that takes) and pass the boolean result to
        `submit_result`. Returns None once `is_done()`. Idempotent:
        calling this again before `submit_result` returns the SAME
        group rather than popping a new one, so a caller that crashes
        or is replaced mid-round-trip can safely re-ask."""
        if self.is_done():
            return None
        if self._awaiting is None:
            self._awaiting = self.frontier.pop()
        return list(self._awaiting)

    def submit_result(self, group: list[int], opened: bool) -> None:
        """`opened=True`: this group's sub-sum matched (cleared, not
        suspect). `opened=False`: split it (queueing both halves) or,
        if it is already at `min_group_size`, record it as a suspect
        group -- the exact logic `localize_fault`'s loop body used to
        run inline. `group` must be exactly the group `pending_group()`
        last returned, order aside -- answering any other group is a
        caller bug, not a valid protocol step, so this raises rather
        than silently accepting it."""
        if self._awaiting is None or sorted(self._awaiting) != sorted(group):
            raise ValueError(
                f"submit_result got group={sorted(group)}, but the outstanding query "
                f"was for {sorted(self._awaiting) if self._awaiting is not None else None} "
                f"-- call pending_group() first and answer exactly that group"
            )
        self.rounds_used += 1
        if not opened:
            split = partition_group(
                self._awaiting, self.seed + self.coordinate.to_bytes(4, "little"), self.min_group_size,
            )
            if split is None:
                self.suspects.append(self._awaiting)
            else:
                self.frontier.extend(split)
        self._awaiting = None

    @property
    def result(self) -> BlameResult:
        """Valid at any point, not just once `is_done()` -- mid-session
        it additionally reports whatever is still queued or
        outstanding as suspect, the same "max_rounds hit, whatever's
        left is still only suspected" rule `localize_fault` always
        applied."""
        leftover = list(self.frontier) + ([self._awaiting] if self._awaiting is not None else [])
        return BlameResult(suspect_groups=self.suspects + leftover, rounds_used=self.rounds_used)


def localize_fault(client_ids: list[int], coordinate: int, seed: bytes,
                    check_group_opens_correctly, min_group_size: int = 4, max_rounds: int = 10) -> BlameResult:
    """Group-testing bisection for ONE bad coordinate (call once per
    entry of F7e's `bad` list; different coordinates can implicate
    different groups).

    `check_group_opens_correctly(group: list[int]) -> bool` is supplied
    by the caller (fl/server.py): it asks the clients in `group` to
    reveal (via secagg) their SUM-only blinding for `coordinate` and
    checks that sub-sum against the corresponding slice of the true
    aggregate sketch. This function only does the bisection control
    flow; it never touches a blinding or commitment directly, so it
    carries no crypto of its own to get wrong.

    Synchronous convenience wrapper around `BisectionSession` above --
    drives it to completion in one blocking call, for a caller (every
    one in this repo today) that doesn't need to suspend BETWEEN
    rounds. A caller that does (a real deployment, where each round is
    an actual network round trip) drives `BisectionSession` directly
    instead of calling this.
    """
    session = BisectionSession.start(client_ids, coordinate, seed, min_group_size, max_rounds)
    while True:
        group = session.pending_group()
        if group is None:
            break
        session.submit_result(group, check_group_opens_correctly(group))
    return session.result
