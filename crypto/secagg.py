"""Secure aggregation with double-masking (ARCHITECTURE.md §3 P9, REVIEW.md R5).

PURE-MATH for the masking algebra (verified in-sandbox, see
tests/test_secagg.py) — the one EC-dependent piece (deriving a pairwise
shared secret via X25519 ECDH) is isolated into
`derive_pairwise_secret_via_x25519` at the bottom, clearly marked, and
is NOT needed to test the masking math itself: that only needs 32
arbitrary bytes per pair, however they were derived.

R5's fix, concretely: single-masking means "declare client k dropped"
lets the server combine the OTHER clients' pairwise-mask shares to
unmask k's update -- a malicious server can just LIE about a dropout.
Double-masking adds a PRIVATE self-mask that is only ever revealed via
Shamir shares of random value; recovering a client's self-mask (to
"legitimately" handle a real dropout) requires t-of-n of its
previously-distributed self-mask shares, which a server can't forge or
extract from other clients' data alone -- see
test_false_dropout_does_not_unmask in tests/test_secagg.py, which is
the actual guardrail test REVIEW.md calls for.
"""
from __future__ import annotations

import hashlib

import numpy as np

from .field import L, ShamirPolynomial, ShamirShare, reconstruct
from .prg import derive_stream_key, expand_field_stream

# Masking modulus: a 61-bit Mersenne prime, NOT the Ed25519 group order L.
# L (~2^252) is used for Shamir-sharing curve scalars (field.py) and does
# not fit numpy's fixed-width integer types. Model-weight masking only
# needs to hide quantized gradient coordinates, so a modulus that fits
# safely inside int64/uint64 arithmetic (with headroom so additions don't
# overflow before the `% P_MASK` reduction) is the standard choice for
# secure-aggregation masking (distinct from the group-order field used for
# key material elsewhere in this repo).
P_MASK = 2**61 - 1


def _prg(key: bytes, round_nonce: bytes, dim: int, seed_salt: bytes = b"") -> np.ndarray:
    """Deterministic pseudorandom vector in Z_{P_MASK}^dim from (key, round_nonce).
    Both pairwise masks and self-masks use this -- same primitive,
    different key material. AES-256-CTR expansion, see crypto/prg.py
    (FIX.md F2).
    """
    return expand_field_stream(derive_stream_key(b"secagg-mask", key, round_nonce, seed_salt), dim)


def pairwise_round_seed(shared_secret: bytes, round_nonce: bytes) -> bytes:
    """Round-specific seed for one pair's mask, derived one-way from the
    pair's LONG-TERM shared secret. This is the value a surviving client
    hands to the server when its partner's submission was dropped or
    rejected (see `unmask_aggregate`): it lets the server strip that one
    pair's mask for that one round, and reveals nothing about the
    long-term secret or about any other round's mask (SHA-256 preimage
    resistance). Without this indirection, removing a dropped client's
    pairwise masks would mean disclosing the long-term secret itself and
    with it every past and future round's mask for that pair.
    """
    return hashlib.sha256(b"janus-fl-pairwise-round-seed" + shared_secret + round_nonce).digest()


def pairwise_mask_from_round_seed(round_seed: bytes, dim: int, sign: int) -> np.ndarray:
    v = _prg(round_seed, b"", dim, seed_salt=b"pairwise")
    return (sign * v) % P_MASK


def pairwise_mask(shared_secret: bytes, round_nonce: bytes, dim: int, sign: int) -> np.ndarray:
    """sign = +1 for the lower-indexed party in a pair, -1 for the
    higher-indexed -- this is what makes every pairwise mask cancel
    exactly once summed over all clients (standard Bonawitz construction).
    Expanded from the pair's round seed (not the long-term secret
    directly) so the server-side removal path in `unmask_aggregate`
    reproduces exactly the same vector from the round seed alone.
    """
    return pairwise_mask_from_round_seed(pairwise_round_seed(shared_secret, round_nonce), dim, sign)


def pairwise_mask_cached(
    shared_secret: bytes, round_nonce: bytes, dim: int, sign: int,
    cache: dict | None, pair_key,
) -> np.ndarray:
    """Same result as `pairwise_mask` (FIX.md F3). Both parties in a pair
    derive the identical shared_secret (X25519 ECDH) and therefore the
    identical round seed and expanded magnitude vector -- only `sign`
    differs between them. A real client has no access to the other
    party's process and must still expand its own copy; this
    simulation runs every client in one process, so when the caller
    passes a `cache` dict it owns for exactly one round, the expansion
    is done once per unordered pair (`pair_key`, e.g. `(min(i,j),
    max(i,j))`) and the stored vector is reused for the second party
    with the opposite sign -- a cheap O(dim) negation instead of a
    second AES-CTR expansion.

    `cache=None` recovers the exact per-party cost (calls `pairwise_mask`
    directly) -- this is what any real-cost measurement must use.
    """
    if cache is None:
        return pairwise_mask(shared_secret, round_nonce, dim, sign)
    magnitude = cache.get(pair_key)
    if magnitude is None:
        magnitude = pairwise_mask_from_round_seed(pairwise_round_seed(shared_secret, round_nonce), dim, 1)
        cache[pair_key] = magnitude
    return (sign * magnitude) % P_MASK


def self_mask(self_secret: bytes, round_nonce: bytes, dim: int) -> np.ndarray:
    """A client's own private mask -- summed INTO its masked update but
    never cancelled by any other client's contribution. Recoverable only
    via Shamir shares of `self_secret` (see shamir_share_self_secret /
    recover_self_mask below), never by combining other clients' pairwise
    data.
    """
    return _prg(self_secret, round_nonce, dim, seed_salt=b"self")


def shamir_share_self_secret(self_secret_int: int, t: int, holder_ids: list[int]) -> dict[int, ShamirShare]:
    """Each client Shamir-shares its OWN self-mask secret across (a
    subset of) the other clients, so a genuine dropout can still be
    unmasked by the cooperating remainder -- but only with t real shares
    of THAT client's own secret, not by the server unilaterally
    fabricating a 'dropout'.
    """
    poly = ShamirPolynomial(t, secret=self_secret_int)
    return {i: poly.share_for(i) for i in holder_ids}


def recover_self_mask(shares: list[ShamirShare], round_nonce: bytes, dim: int) -> np.ndarray:
    secret_int = reconstruct(shares)
    secret_bytes = secret_int.to_bytes(32, "little")
    return self_mask(secret_bytes, round_nonce, dim)


def aggregate_masked(masked_updates: list[np.ndarray]) -> np.ndarray:
    acc = np.zeros_like(masked_updates[0])
    for u in masked_updates:
        acc = (acc + u) % P_MASK
    return acc


# ---------------------------------------------------------------------------
# Unmasking round (Bonawitz et al. CCS'17, "Unmasking" step). Summing the
# accepted clients' masked vectors is NOT the aggregate yet:
#
#   sum_{k in A} x_k  =  sum_{k in A} q_k
#                      + sum_{k in A} selfmask_k                 (never cancels)
#                      + sum_{k in A, j not in A} +-PRG(seed_kj)  (partner missing)
#
# Pairwise masks cancel only between two clients that are BOTH in the
# accepted set A. The other two terms must be removed explicitly, using
# material the clients release only after the accepted set is fixed.
# ---------------------------------------------------------------------------

def new_self_mask_secret() -> int:
    """A fresh self-mask secret, drawn from Z_L so that Shamir sharing
    over Z_L (field.py) reconstructs it EXACTLY. A raw 256-bit value can
    exceed L (~2^252); it would be reduced mod L inside the polynomial
    and the reconstructed secret would then expand to a different mask
    than the one the client actually applied.
    """
    from .field import rand_scalar
    return rand_scalar()


def self_secret_bytes(self_secret_int: int) -> bytes:
    return self_secret_int.to_bytes(32, "little")


class ShareReleasePolicy:
    """What an honest client is willing to release during unmasking, for
    ONE round. This is the rule that makes double-masking mean anything
    (ARCHITECTURE.md C7 / REVIEW.md R5): about any given client k, an
    honest holder releases EITHER its share of k's self-mask secret (k's
    submission is being aggregated) OR its pairwise round seed with k
    (k's submission is excluded) -- never both. A server holding both
    for the same k could strip every mask from k's vector and read the
    update in the clear.
    """

    def __init__(self):
        self._self_share_released_about: set = set()
        self._pairwise_seed_released_about: set = set()

    def release_self_share(self, about_client: int) -> None:
        if about_client in self._pairwise_seed_released_about:
            raise PermissionError(
                f"refusing to release a self-mask share about client {about_client}: "
                f"pairwise seeds with that client were already released this round"
            )
        self._self_share_released_about.add(about_client)

    def release_pairwise_seed(self, about_client: int) -> None:
        if about_client in self._self_share_released_about:
            raise PermissionError(
                f"refusing to release a pairwise seed with client {about_client}: "
                f"self-mask shares about that client were already released this round"
            )
        self._pairwise_seed_released_about.add(about_client)


def unmask_aggregate(
    masked_sum: np.ndarray,
    accepted: list[int],
    self_mask_shares: dict,
    share_threshold: int,
    excluded_pair_seeds: dict,
    round_nonce: bytes,
    dim: int,
) -> np.ndarray:
    """Turns sum_{k in A} x_k (mod P_MASK) into sum_{k in A} q_k (mod P_MASK).

    masked_sum          -- aggregate_masked() over the accepted clients' vectors.
    accepted            -- client indices in A.
    self_mask_shares    -- {k: [ShamirShare, ...]} for every k in A; at least
                           `share_threshold` shares each, or this raises (fewer
                           shares reconstruct a wrong secret silently, which
                           would corrupt the aggregate with no error).
    excluded_pair_seeds -- {(j, k): round_seed} for every accepted j and every
                           excluded k (rejected or dropped), as released by j.
    """
    acc = masked_sum.copy()
    accepted_set = set(accepted)
    for k in accepted:
        distinct = list({sh.index: sh for sh in self_mask_shares.get(k, [])}.values())
        if len(distinct) < share_threshold:
            raise ValueError(
                f"client {k}: only {len(distinct)} distinct self-mask shares, "
                f"need {share_threshold} -- cannot unmask this round"
            )
        acc = (acc - recover_self_mask(distinct[:share_threshold], round_nonce, dim)) % P_MASK
    for (j, k), seed in excluded_pair_seeds.items():
        if j not in accepted_set or k in accepted_set:
            raise ValueError(f"pair ({j}, {k}): seed must come from an accepted client about an excluded one")
        sign = 1 if j < k else -1
        acc = (acc - pairwise_mask_from_round_seed(seed, dim, sign)) % P_MASK
    return acc


# ---------------------------------------------------------------------------
# EC-DEPENDENT: the one-time pairwise key agreement. NOT needed to test
# the masking algebra above -- only needed to bootstrap real
# `shared_secret` bytes between two real clients. See ristretto.py's
# header for the Colab-validation note (this specifically needs
# nacl.public's X25519, a different curve encoding than ristretto255,
# which is why it's a separate tiny function rather than using
# crypto/ristretto.py).
# ---------------------------------------------------------------------------

def derive_pairwise_secret_via_x25519(my_private_key, their_public_key) -> bytes:
    import nacl.public
    import nacl.bindings
    box = nacl.public.Box(my_private_key, their_public_key)
    raw = nacl.bindings.crypto_scalarmult(bytes(my_private_key), bytes(their_public_key))
    return hashlib.sha256(raw + b"janus-fl-pairwise-secret").digest()
