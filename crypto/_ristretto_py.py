"""Pure-Python ristretto255 -- a TEST-ONLY stand-in for `rbcl`.

*** NOT FOR REAL RUNS. *** Roughly 1000x slower than libsodium and not
constant-time. It exists so the EC-dependent protocol code (Pedersen,
CLSAG, DKG, the F7 norm proof, a whole verification round) can be
exercised in an environment with no network to install `rbcl`.
crypto/ristretto.py only selects it when `rbcl` is missing AND the
environment variable JANUS_ALLOW_PURE_PYTHON_GROUP=1 is set, which only
the test files do.

Implements RFC 9496 (ristretto255) encode/decode/equality over the
Edwards25519 curve in extended coordinates, and exposes the same
function names `rbcl` does, so crypto/ristretto.py needs no second code
path.

What is verified, and what is not:
  * encode/decode, addition and scalar multiplication are checked in
    tests/test_ristretto_backend.py against the RFC 9496 test vectors
    for small multiples of the generator, and against group laws.
  * `crypto_core_ristretto255_from_hash` follows the RFC's
    one-way map and always returns a valid group element, but it has
    NOT been checked byte-for-byte against libsodium. Nothing here
    needs it to agree: generators only have to be the same for every
    party inside one process.
"""
from __future__ import annotations

import secrets

P = 2**255 - 19
L = 2**252 + 27742317777372353535851937790883648493
D = (-121665 * pow(121666, -1, P)) % P
SQRT_M1 = pow(2, (P - 1) // 4, P)


def _is_neg(x: int) -> bool:
    return (x % P) & 1 == 1


def _abs(x: int) -> int:
    x %= P
    return P - x if x & 1 else x


def _sqrt_ratio_m1(u: int, v: int):
    """RFC 9496 SQRT_RATIO_M1: (was_square, sqrt(u/v)) or sqrt(i*u/v)."""
    u %= P
    v %= P
    v3 = v * v % P * v % P
    v7 = v3 * v3 % P * v % P
    r = u * v3 % P * pow(u * v7 % P, (P - 5) // 8, P) % P
    check = v * r % P * r % P
    correct_sign = check == u
    flipped = check == (-u) % P
    flipped_i = check == (-u * SQRT_M1) % P
    if flipped or flipped_i:
        r = r * SQRT_M1 % P
    return (correct_sign or flipped), _abs(r)


_ok, INVSQRT_A_MINUS_D = _sqrt_ratio_m1(1, (-1 - D) % P)
assert _ok
# RFC 9496 fixes this root of a*d - 1 (a = -1); it is the odd one.
SQRT_AD_MINUS_ONE = 25063068953384623474111414158702152701244531502492656460079210482610430750235
assert SQRT_AD_MINUS_ONE * SQRT_AD_MINUS_ONE % P == (-D - 1) % P
ONE_MINUS_D_SQ = (1 - D * D) % P
D_MINUS_ONE_SQ = (D - 1) * (D - 1) % P

# Edwards25519 base point: y = 4/5, x the even root.
_BY = 4 * pow(5, -1, P) % P
_ok, _BX = _sqrt_ratio_m1((_BY * _BY - 1) % P, (D * _BY * _BY + 1) % P)
assert _ok
_BASE = (_BX, _BY, 1, _BX * _BY % P)
_IDENT = (0, 1, 1, 0)


def _add(p, q):
    x1, y1, z1, t1 = p
    x2, y2, z2, t2 = q
    a = (y1 - x1) * (y2 - x2) % P
    b = (y1 + x1) * (y2 + x2) % P
    c = t1 * 2 * D % P * t2 % P
    d = z1 * 2 * z2 % P
    e, f, g, h = b - a, d - c, d + c, b + a
    return (e * f % P, g * h % P, f * g % P, e * h % P)


def _neg(p):
    x, y, z, t = p
    return ((-x) % P, y, z, (-t) % P)


def _mul(k: int, p):
    k %= L
    # 4-bit fixed window
    table = [_IDENT, p]
    for _ in range(14):
        table.append(_add(table[-1], p))
    acc = _IDENT
    for shift in range(252, -1, -4):
        for _ in range(4):
            acc = _add(acc, acc)
        acc = _add(acc, table[(k >> shift) & 15])
    return acc


def _decode(s_bytes: bytes):
    if len(s_bytes) != 32:
        return None
    s = int.from_bytes(s_bytes, "little")
    if s >= P or s & 1:
        return None
    ss = s * s % P
    u1 = (1 - ss) % P
    u2 = (1 + ss) % P
    u2_sqr = u2 * u2 % P
    v = (-(D * u1 % P * u1) - u2_sqr) % P
    was_square, invsqrt = _sqrt_ratio_m1(1, v * u2_sqr % P)
    den_x = invsqrt * u2 % P
    den_y = invsqrt * den_x % P * v % P
    x = _abs(2 * s * den_x % P)
    y = u1 * den_y % P
    t = x * y % P
    if (not was_square) or _is_neg(t) or y == 0:
        return None
    return (x, y, 1, t)


def _encode(p) -> bytes:
    x0, y0, z0, t0 = p
    u1 = (z0 + y0) * (z0 - y0) % P
    u2 = x0 * y0 % P
    _, invsqrt = _sqrt_ratio_m1(1, u1 * u2 % P * u2 % P)
    den1 = invsqrt * u1 % P
    den2 = invsqrt * u2 % P
    z_inv = den1 * den2 % P * t0 % P
    ix0 = x0 * SQRT_M1 % P
    iy0 = y0 * SQRT_M1 % P
    enchanted = den1 * INVSQRT_A_MINUS_D % P
    if _is_neg(t0 * z_inv % P):
        x, y, den_inv = iy0, ix0, enchanted
    else:
        x, y, den_inv = x0, y0, den2
    if _is_neg(x * z_inv % P):
        y = (-y) % P
    return _abs(den_inv * ((z0 - y) % P) % P).to_bytes(32, "little")


def _map(t: int):
    """RFC 9496 MAP (Elligator 2 for ristretto255)."""
    r = SQRT_M1 * t % P * t % P
    u = (r + 1) * ONE_MINUS_D_SQ % P
    v = (-1 - r * D) % P * ((r + D) % P) % P
    was_square, s = _sqrt_ratio_m1(u, v)
    s_prime = (-_abs(s * t % P)) % P
    if not was_square:
        s = s_prime
        c = r
    else:
        c = P - 1
    n = (c * ((r - 1) % P) % P * D_MINUS_ONE_SQ - v) % P
    w0 = 2 * s * v % P
    w1 = n * SQRT_AD_MINUS_ONE % P
    w2 = (1 - s * s) % P
    w3 = (1 + s * s) % P
    return (w0 * w3 % P, w2 * w1 % P, w1 * w3 % P, w0 * w2 % P)


def _require(p_bytes: bytes):
    p = _decode(p_bytes)
    if p is None:
        raise ValueError("invalid ristretto255 point encoding")
    return p


def _scalar(b: bytes) -> int:
    if len(b) != 32:
        raise TypeError("scalar must be 32 bytes")
    return int.from_bytes(b, "little")


def _sb(x: int) -> bytes:
    return (x % L).to_bytes(32, "little")


_ZERO_POINT = b"\x00" * 32

# --- the rbcl-compatible surface -------------------------------------------

crypto_core_ristretto255_BYTES = 32
crypto_core_ristretto255_SCALARBYTES = 32
crypto_core_ristretto255_HASHBYTES = 64


def crypto_core_ristretto255_is_valid_point(p: bytes) -> bool:
    return _decode(p) is not None


def crypto_core_ristretto255_add(p: bytes, q: bytes) -> bytes:
    return _encode(_add(_require(p), _require(q)))


def crypto_core_ristretto255_sub(p: bytes, q: bytes) -> bytes:
    return _encode(_add(_require(p), _neg(_require(q))))


def crypto_core_ristretto255_from_hash(h: bytes) -> bytes:
    if len(h) != 64:
        raise TypeError("hash must be 64 bytes")
    mask = (1 << 255) - 1
    t1 = (int.from_bytes(h[:32], "little") & mask) % P
    t2 = (int.from_bytes(h[32:], "little") & mask) % P
    return _encode(_add(_map(t1), _map(t2)))


def crypto_core_ristretto255_random() -> bytes:
    return crypto_core_ristretto255_from_hash(secrets.token_bytes(64))


def crypto_core_ristretto255_scalar_random() -> bytes:
    while True:
        x = int.from_bytes(secrets.token_bytes(40), "little") % L
        if x:
            return _sb(x)


def crypto_core_ristretto255_scalar_add(a: bytes, b: bytes) -> bytes:
    return _sb(_scalar(a) + _scalar(b))


def crypto_core_ristretto255_scalar_sub(a: bytes, b: bytes) -> bytes:
    return _sb(_scalar(a) - _scalar(b))


def crypto_core_ristretto255_scalar_mul(a: bytes, b: bytes) -> bytes:
    return _sb(_scalar(a) * _scalar(b))


def crypto_core_ristretto255_scalar_negate(a: bytes) -> bytes:
    return _sb(-_scalar(a))


def crypto_core_ristretto255_scalar_invert(a: bytes) -> bytes:
    x = _scalar(a) % L
    if x == 0:
        raise ValueError("cannot invert the zero scalar")
    return _sb(pow(x, -1, L))


def crypto_scalarmult_ristretto255(n: bytes, p: bytes) -> bytes:
    out = _encode(_mul(_scalar(n), _require(p)))
    if out == _ZERO_POINT:
        # libsodium returns an error when the result is the identity;
        # mirror that so a caller relying on it fails here too.
        raise ValueError("scalar multiplication produced the identity element")
    return out


def crypto_scalarmult_ristretto255_base(n: bytes) -> bytes:
    out = _encode(_mul(_scalar(n), _BASE))
    if out == _ZERO_POINT:
        raise ValueError("scalar multiplication produced the identity element")
    return out
