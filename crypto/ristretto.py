"""ristretto255 group operations — thin, explicit wrapper over `rbcl`'s
low-level libsodium bindings.

*** EC-DEPENDENT MODULE — NOT RUNNABLE IN THIS SANDBOX (no network to
install rbcl). Validate on Colab: notebooks/00_Setup_and_Crypto_Tests.ipynb,
cell 1 does `pip install rbcl` then runs tests/colab/test_ristretto.py.
Do not trust any result from this module, or anything built on it
(pedersen.py, clsag.py, elgamal_threshold.py, dkg.py, vrf.py, mixnet.py,
rangeproof.py, r_cons.py), until that notebook cell is green. ***

Every function here maps 1:1 onto a documented libsodium primitive so
there is nothing novel or home-grown about the group arithmetic itself
(only the protocols built ON TOP, in the other modules, are novel).
Reference: libsodium's ristretto255 API (doc.libsodium.org/advanced/
point-arithmetic/ristretto), exposed here via the `rbcl` package.

*** CORRECTED 2026-11-13 — see FIX.md F8. This module previously called
`nacl.bindings.crypto_core_ristretto255_*`, on the assumption (stated in
this docstring and in notebooks/_build_notebooks.py's install cell) that
PyNaCl >= 1.5.0 exposes ristretto255. It does not, under ANY released
version: adding ristretto255 to mainline PyNaCl is a still-open, unmerged
pull request (github.com/pyca/pynacl/pull/646). Every `AttributeError`
chased on Colab against "stale pynacl" was this bug, not a stale kernel.
`rbcl` (pypi.org/project/rbcl, maintained by Nth Party) is a dedicated,
actively-released package that wraps exactly these libsodium ristretto255
functions and ships manylinux2014 x86_64 wheels, so it installs on Colab
with a plain `pip install rbcl` -- no source build. ***
"""
from __future__ import annotations

from dataclasses import dataclass

try:
    import rbcl as _b
    if not hasattr(_b, "crypto_core_ristretto255_from_hash"):
        # Some rbcl releases expose the functions on the `rbcl.rbcl`
        # submodule rather than re-exporting them at package level;
        # fall back to that before giving up.
        import rbcl.rbcl as _b  # type: ignore[import]
    _HAVE_RBCL = hasattr(_b, "crypto_core_ristretto255_from_hash")
except ImportError:
    _HAVE_RBCL = False


def _require_rbcl():
    if not _HAVE_RBCL:
        raise RuntimeError(
            "rbcl is not installed (or does not expose ristretto255 "
            "bindings under either `rbcl` or `rbcl.rbcl`). This module "
            "requires network access (`pip install rbcl`), which this "
            "sandbox does not have. Run on Colab -- see "
            "notebooks/00_Setup_and_Crypto_Tests.ipynb. Do NOT install "
            "`pynacl` for this: mainline PyNaCl has never shipped "
            "ristretto255 bindings, under any version (see this module's "
            "docstring, FIX.md F8)."
        )


_GROUP_ORDER_L = 2**252 + 27742317777372353535851937790883648493  # ristretto255/ed25519 scalar group order
_SCALAR_ZERO = b"\x00" * 32   # canonical mod-L encoding of 0 (Scalar.from_int(0) produces exactly this)
_POINT_IDENTITY = b"\x00" * 32  # ristretto255's identity element always encodes as 32 zero bytes (spec property)


def _wide_scalar_reduce(wide: bytes) -> bytes:
    """Reduce an arbitrary-length (here always 64-byte) little-endian
    integer mod the group order L, returning a canonical 32-byte
    little-endian scalar encoding.

    libsodium's C `crypto_core_ristretto255_scalar_reduce` takes exactly
    this 64-byte ("non-reduced scalar") input and does exactly this
    reduction -- but `rbcl`'s Python wrapper only accepts a 32-byte
    buffer for this call (checked directly against rbcl 0.4.1's source:
    it validates `len(p) == crypto_core_ristretto255_SCALARBYTES` (32),
    not `..._NONREDUCEDSCALARBYTES` (64), even though its own docstring
    describes the 64-byte case -- a real mismatch in that wrapper, not
    an assumption). Both of this module's callers (`Scalar.from_int`,
    `fiat_shamir_challenge`) need the 64-byte reduction, so it is done
    directly in Python instead of routing through rbcl: `x mod L` with
    the standard, public ristretto255/ed25519 group order is exactly
    the libsodium operation's definition, not an approximation of it.
    """
    x = int.from_bytes(wide, "little") % _GROUP_ORDER_L
    return x.to_bytes(32, "little")


@dataclass(frozen=True)
class Scalar:
    """An element of Z_L, as raw libsodium scalar bytes."""
    b: bytes

    @staticmethod
    def random() -> "Scalar":
        _require_rbcl()
        return Scalar(_b.crypto_core_ristretto255_scalar_random())

    @staticmethod
    def from_int(x: int) -> "Scalar":
        """Reduce a big Python int mod L into libsodium's scalar encoding."""
        raw = x.to_bytes(64, "little") if x.bit_length() <= 512 else (x % (2**512)).to_bytes(64, "little")
        return Scalar(_wide_scalar_reduce(raw))

    def __add__(self, other: "Scalar") -> "Scalar":
        _require_rbcl()
        return Scalar(_b.crypto_core_ristretto255_scalar_add(self.b, other.b))

    def __sub__(self, other: "Scalar") -> "Scalar":
        _require_rbcl()
        return Scalar(_b.crypto_core_ristretto255_scalar_sub(self.b, other.b))

    def __mul__(self, other: "Scalar") -> "Scalar":
        _require_rbcl()
        return Scalar(_b.crypto_core_ristretto255_scalar_mul(self.b, other.b))

    def invert(self) -> "Scalar":
        _require_rbcl()
        return Scalar(_b.crypto_core_ristretto255_scalar_invert(self.b))

    def negate(self) -> "Scalar":
        _require_rbcl()
        return Scalar(_b.crypto_core_ristretto255_scalar_negate(self.b))


@dataclass(frozen=True)
class Point:
    """A group element (point on ristretto255), as raw libsodium point bytes."""
    b: bytes

    @staticmethod
    def base() -> "Point":
        """The standard base point G, via 1*G."""
        _require_rbcl()
        one = Scalar.from_int(1)
        return Point(_b.crypto_scalarmult_ristretto255_base(one.b))

    @staticmethod
    def random() -> "Point":
        _require_rbcl()
        return Point(_b.crypto_core_ristretto255_random())

    @staticmethod
    def hash_to_point(data: bytes, domain: bytes = b"janus-fl") -> "Point":
        """Deterministic, uniform-looking point from arbitrary bytes.
        Used for: the Pedersen second generator H (pedersen.py), the VRF's
        H_p(rho_t) and H_p(rho_e) (vrf.py, clsag.py's dual-scope tags).
        """
        _require_rbcl()
        import hashlib
        wide = hashlib.sha512(domain + b":" + data).digest()  # 64 bytes, uniform -- matches rbcl's HASHBYTES=64 exactly
        return Point(_b.crypto_core_ristretto255_from_hash(wide))

    def __add__(self, other: "Point") -> "Point":
        _require_rbcl()
        return Point(_b.crypto_core_ristretto255_add(self.b, other.b))

    def __sub__(self, other: "Point") -> "Point":
        _require_rbcl()
        return Point(_b.crypto_core_ristretto255_sub(self.b, other.b))

    def scalar_mul(self, s: Scalar) -> "Point":
        _require_rbcl()
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
        _require_rbcl()
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
    _require_rbcl()
    if s.b == _SCALAR_ZERO:
        return Point(_POINT_IDENTITY)
    return Point(_b.crypto_scalarmult_ristretto255_base(s.b))


def fiat_shamir_challenge(*parts: bytes) -> Scalar:
    """Hash a transcript into a Z_L challenge scalar — the standard way
    every Sigma-protocol / Schnorr-style NIZK in this repo (schnorr_nizk.py,
    clsag.py, vrf.py) turns an interactive proof into a non-interactive one.
    """
    import hashlib
    h = hashlib.sha512()
    for p in parts:
        h.update(len(p).to_bytes(8, "little"))
        h.update(p)
    return Scalar(_wide_scalar_reduce(h.digest()))
