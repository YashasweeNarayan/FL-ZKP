"""ristretto255 group operations — thin, explicit wrapper over pynacl's
low-level libsodium bindings.

*** EC-DEPENDENT MODULE — NOT RUNNABLE IN THIS SANDBOX (no network to
install pynacl). Validate on Colab: notebooks/00_Setup_and_Crypto_Tests.ipynb,
cell 1 does `pip install pynacl` then runs tests/colab/test_ristretto.py.
Do not trust any result from this module, or anything built on it
(pedersen.py, clsag.py, elgamal_threshold.py, dkg.py, vrf.py, mixnet.py,
rangeproof.py, r_cons.py), until that notebook cell is green. ***

Every function here maps 1:1 onto a documented libsodium primitive so
there is nothing novel or home-grown about the group arithmetic itself
(only the protocols built ON TOP, in the other modules, are novel).
Reference: libsodium's ristretto255 API, exposed by PyNaCl >= 1.4 as
`nacl.bindings.crypto_core_ristretto255_*` / `crypto_scalarmult_ristretto255*`.
"""
from __future__ import annotations

from dataclasses import dataclass

try:
    import nacl.bindings as _b
    _HAVE_NACL = True
except ImportError:
    _HAVE_NACL = False


def _require_nacl():
    if not _HAVE_NACL:
        raise RuntimeError(
            "pynacl is not installed. This module requires network access "
            "(`pip install pynacl`), which this sandbox does not have. "
            "Run on Colab — see notebooks/00_Setup_and_Crypto_Tests.ipynb."
        )


_SCALAR_ZERO = b"\x00" * 32   # canonical mod-L encoding of 0 (Scalar.from_int(0) produces exactly this)
_POINT_IDENTITY = b"\x00" * 32  # ristretto255's identity element always encodes as 32 zero bytes (spec property)


@dataclass(frozen=True)
class Scalar:
    """An element of Z_L, as raw libsodium scalar bytes."""
    b: bytes

    @staticmethod
    def random() -> "Scalar":
        _require_nacl()
        return Scalar(_b.crypto_core_ristretto255_scalar_random())

    @staticmethod
    def from_int(x: int) -> "Scalar":
        """Reduce a big Python int mod L into libsodium's scalar encoding."""
        _require_nacl()
        raw = x.to_bytes(64, "little") if x.bit_length() <= 512 else (x % (2**512)).to_bytes(64, "little")
        return Scalar(_b.crypto_core_ristretto255_scalar_reduce(raw))

    def __add__(self, other: "Scalar") -> "Scalar":
        return Scalar(_b.crypto_core_ristretto255_scalar_add(self.b, other.b))

    def __sub__(self, other: "Scalar") -> "Scalar":
        return Scalar(_b.crypto_core_ristretto255_scalar_sub(self.b, other.b))

    def __mul__(self, other: "Scalar") -> "Scalar":
        return Scalar(_b.crypto_core_ristretto255_scalar_mul(self.b, other.b))

    def invert(self) -> "Scalar":
        return Scalar(_b.crypto_core_ristretto255_scalar_invert(self.b))

    def negate(self) -> "Scalar":
        return Scalar(_b.crypto_core_ristretto255_scalar_negate(self.b))


@dataclass(frozen=True)
class Point:
    """A group element (point on ristretto255), as raw libsodium point bytes."""
    b: bytes

    @staticmethod
    def base() -> "Point":
        """The standard base point G, via 1*G."""
        _require_nacl()
        one = Scalar.from_int(1)
        return Point(_b.crypto_scalarmult_ristretto255_base(one.b))

    @staticmethod
    def random() -> "Point":
        _require_nacl()
        return Point(_b.crypto_core_ristretto255_random())

    @staticmethod
    def hash_to_point(data: bytes, domain: bytes = b"janus-fl") -> "Point":
        """Deterministic, uniform-looking point from arbitrary bytes.
        Used for: the Pedersen second generator H (pedersen.py), the VRF's
        H_p(rho_t) and H_p(rho_e) (vrf.py, clsag.py's dual-scope tags).
        """
        _require_nacl()
        import hashlib
        wide = hashlib.sha512(domain + b":" + data).digest()  # 64 bytes, uniform
        return Point(_b.crypto_core_ristretto255_from_hash(wide))

    def __add__(self, other: "Point") -> "Point":
        return Point(_b.crypto_core_ristretto255_add(self.b, other.b))

    def __sub__(self, other: "Point") -> "Point":
        return Point(_b.crypto_core_ristretto255_sub(self.b, other.b))

    def scalar_mul(self, s: Scalar) -> "Point":
        if s.b == _SCALAR_ZERO:
            # 0*P = identity for any valid P, trivially. libsodium's
            # crypto_scalarmult_ristretto255 explicitly REJECTS a zero
            # scalar (rc=-1, "invalid input to a real curve operation")
            # since the result is the group identity and the API guards
            # against silently producing it -- no real curve operation
            # is needed for this degenerate case, so short-circuit it.
            return Point(_POINT_IDENTITY)
        return Point(_b.crypto_scalarmult_ristretto255(s.b, self.b))

    def is_valid(self) -> bool:
        return bool(_b.crypto_core_ristretto255_is_valid_point(self.b))

    def __eq__(self, other) -> bool:
        return isinstance(other, Point) and self.b == other.b

    def __hash__(self) -> int:
        return hash(self.b)


def scalar_mul_base(s: Scalar) -> Point:
    """s*G — the common case of a public key from a secret scalar.

    s=0 is special-cased to the identity point directly: libsodium's
    crypto_scalarmult_ristretto255_base rejects a zero scalar outright
    (rc=-1) rather than returning the identity, since silently producing
    the identity is usually a sign something upstream went wrong in a
    DH-style protocol. Here it's a legitimate input -- e.g.
    pedersen.py's commit(0, r) for a bit=0 range-proof branch, or any
    Pedersen commitment to the value 0 -- so we compute it directly
    instead of letting libsodium reject it.
    """
    _require_nacl()
    if s.b == _SCALAR_ZERO:
        return Point(_POINT_IDENTITY)
    return Point(_b.crypto_scalarmult_ristretto255_base(s.b))


def fiat_shamir_challenge(*parts: bytes) -> Scalar:
    """Hash a transcript into a Z_L challenge scalar — the standard way
    every Sigma-protocol / Schnorr-style NIZK in this repo (schnorr_nizk.py,
    clsag.py, vrf.py) turns an interactive proof into a non-interactive one.
    """
    _require_nacl()
    import hashlib
    h = hashlib.sha512()
    for p in parts:
        h.update(len(p).to_bytes(8, "little"))
        h.update(p)
    return Scalar(_b.crypto_core_ristretto255_scalar_reduce(h.digest()))
