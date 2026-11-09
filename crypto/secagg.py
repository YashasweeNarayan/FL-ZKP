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
    different key material.
    """
    out = np.empty(dim, dtype=np.int64)
    counter = 0
    filled = 0
    buf = b""
    while filled < dim:
        if len(buf) < 8 * 4096:
            buf += hashlib.sha256(key + round_nonce + seed_salt + counter.to_bytes(8, "little")).digest()
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


def pairwise_mask(shared_secret: bytes, round_nonce: bytes, dim: int, sign: int) -> np.ndarray:
    """sign = +1 for the lower-indexed party in a pair, -1 for the
    higher-indexed -- this is what makes every pairwise mask cancel
    exactly once summed over all clients (standard Bonawitz construction).
    """
    v = _prg(shared_secret, round_nonce, dim, seed_salt=b"pairwise")
    return (sign * v) % P_MASK


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
