"""Fast expansion of a short seed into a long vector over Z_{P_MASK}.

Used for every O(model_dim) pseudorandom vector in the protocol: the
pairwise masks and self-masks (secagg.py) and the public digest weights
(vector_commit.py).

Construction (FIX.md F2):

    key    = SHA-256(domain tag || length-prefixed seed parts)   32 bytes
    stream = AES-256-CTR(key, counter block starting at 0)
    each 8 keystream bytes, read little-endian, shifted right by 3 -> 61 bits
    the single value 2^61 - 1 is mapped to 0

AES in counter mode is the mask PRG used by Bonawitz et al. (CCS 2017).
Each derived key is used for exactly one stream, so the fixed starting
counter is safe. The 61-bit value is uniform on [0, 2^61 - 1] before the
last step, so the output's distance from uniform on [0, P_MASK) is 2^-61
per element.

The length prefixes matter: without them ("ab", "c") and ("a", "bc")
would give the same stream.

This replaced a Python loop that called SHA-256 once per 32 bytes of
output. Same interface, same security assumption (a PRG), about 50x
faster at model_dim = 11.2M.
"""
from __future__ import annotations

import hashlib

import numpy as np

try:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
except ImportError as e:  # pragma: no cover
    raise RuntimeError(
        "crypto/prg.py needs the 'cryptography' package for AES-CTR "
        "(pip install cryptography). There is deliberately no slow "
        "fallback: two parties using different expanders would derive "
        "different masks and the aggregate would not unmask."
    ) from e

P_MASK = 2 ** 61 - 1
_CHUNK = 1 << 20  # elements per keystream request (8 MiB)
_ZEROS = bytes(8 * _CHUNK)


def derive_stream_key(domain: bytes, *parts: bytes) -> bytes:
    h = hashlib.sha256()
    h.update(b"janus-fl-prg-v2")
    for p in (domain,) + parts:
        h.update(len(p).to_bytes(8, "little"))
        h.update(p)
    return h.digest()


def iter_field_stream(key: bytes, dim: int, chunk: int = _CHUNK):
    """Yields consecutive int64 arrays with entries in [0, P_MASK), `dim`
    elements in total. Lets a caller consume the stream without ever
    holding all of it."""
    if len(key) != 32:
        raise ValueError("stream key must be 32 bytes (use derive_stream_key)")
    if chunk > _CHUNK:
        chunk = _CHUNK
    enc = Cipher(algorithms.AES(key), modes.CTR(bytes(16))).encryptor()
    done = 0
    while done < dim:
        take = min(chunk, dim - done)
        raw = np.frombuffer(enc.update(_ZEROS[: 8 * take]), dtype="<u8")
        vals = (raw >> np.uint64(3)).astype(np.int64)
        vals[vals == P_MASK] = 0
        yield vals
        done += take


def expand_field_stream(key: bytes, dim: int) -> np.ndarray:
    out = np.empty(dim, dtype=np.int64)
    pos = 0
    for vals in iter_field_stream(key, dim):
        out[pos : pos + vals.shape[0]] = vals
        pos += vals.shape[0]
    return out


# ---------------------------------------------------------------------------
# Scalar-field stream (FIX.md F7e).
#
# The stream above produces elements of Z_{P_MASK} (~2^61), which is the
# right field for masking a QUANTIZED UPDATE. F7e needs to mask something
# else: a vector of Pedersen BLINDING FACTORS, which live in Z_L where L
# is the ristretto255 scalar group order (~2^252). Those masks have to
# cancel under addition mod L, not mod P_MASK, so they need their own
# expander rather than four stitched-together P_MASK elements.
#
# Same construction, wider output: 64 keystream bytes per scalar, read
# little-endian and reduced mod L. 64 bytes is 512 bits against a ~252-bit
# modulus, so the reduction's distance from uniform on [0, L) is below
# 2^-250 per element -- the same wide-reduction argument crypto/ristretto.py's
# `_wide_scalar_reduce` already relies on for hashing into a scalar.
# ---------------------------------------------------------------------------

_SCALAR_L = 2 ** 252 + 27742317777372353535851937790883648493  # ristretto255/ed25519 group order
_SCALAR_WIDE_BYTES = 64


def expand_scalar_stream(key: bytes, dim: int) -> list[int]:
    """`dim` pseudorandom integers in [0, L), as Python ints.

    Returned as ints rather than `Scalar` objects on purpose: the caller
    (crypto/norm_proof.py) adds and subtracts these mod L in bulk, which
    is far cheaper on Python ints than on 32-byte `Scalar` wrappers, and
    converts to `Scalar` only at the point a commitment is actually
    computed.
    """
    if len(key) != 32:
        raise ValueError("stream key must be 32 bytes (use derive_stream_key)")
    if dim < 0:
        raise ValueError("dim must be non-negative")
    enc = Cipher(algorithms.AES(key), modes.CTR(bytes(16))).encryptor()
    nbytes = _SCALAR_WIDE_BYTES * dim
    raw = enc.update(bytes(nbytes)) if nbytes else b""
    out = []
    for i in range(dim):
        chunk = raw[i * _SCALAR_WIDE_BYTES : (i + 1) * _SCALAR_WIDE_BYTES]
        out.append(int.from_bytes(chunk, "little") % _SCALAR_L)
    return out
