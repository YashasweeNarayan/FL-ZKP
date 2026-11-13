"""Group-law checks for whichever ristretto255 backend is live --
real `rbcl`/libsodium on Colab, or the test-only pure-Python backend
(crypto/_ristretto_py.py) in a network-less sandbox.

Deliberately backend-agnostic: these are exactly the properties BOTH
implementations must satisfy, so the same file validates the sandbox
backend's algebra and the real bindings' behaviour. That second job
matters as much as the first -- FIX.md F8 was a case of bindings that
imported fine and did not exist.

This file is the thing crypto/_ristretto_py.py's own module docstring
already claimed exists ("RFC 9496 test vectors ... are verified") --
until now that verification had only been run ad hoc, not committed
as a reproducible test. Written FIX.md F8-adjacent, during F7's
implementation.

Skips cleanly (not an error) when the env var is unset, same pattern
as tests/test_ristretto_pure.py and tests/test_f7_norm_proof.py.
"""
from __future__ import annotations

import os

import pytest

# Gate on whether ANY group backend is live, not on which one. With rbcl
# installed (Colab) these run against real libsodium; in a network-less
# sandbox they run against the pure-Python backend when
# JANUS_ALLOW_PURE_PYTHON_GROUP=1 selects it; with neither they skip.
#
# Gating on the env var alone -- as the first version of these files did --
# meant the whole F7 suite SKIPPED on Colab, which is the one place it most
# needs to run: the pure-Python backend proves the algebra, only rbcl proves
# the bindings.
from crypto.ristretto import backend_name as _backend_name

if _backend_name() is None:
    pytest.skip(
        "no ristretto255 backend available: install rbcl, or set "
        "JANUS_ALLOW_PURE_PYTHON_GROUP=1 to use the test-only pure-Python "
        "backend (crypto/_ristretto_py.py)",
        allow_module_level=True,
    )

from crypto.ristretto import Point, Scalar, backend_name, group_order, identity, msm, scalar_mul_base

# NOTE: this file checks the backend's internal GROUP-LAW consistency
# (identity behaves like an identity, k*B+B==(k+1)*B, scalar and point
# arithmetic agree, the group order annihilates the base point, MSM
# matches sequential multiplication) rather than hardcoding RFC 9496's
# official encoded-point hex test vectors -- those were checked by
# hand against crypto/_ristretto_py.py during its construction (see
# FIX.md), but are not reproduced here verbatim to avoid shipping a
# hand-copied hex string that looks authoritative without actually
# being diffed against the RFC text in this file. The checks below are
# weaker in one sense (they can't catch an implementation that is
# internally consistent but wrong relative to the RFC's specific curve
# constants). A wrong root for one of the RFC's constants (this
# module's own SQRT_AD_MINUS_ONE, fixed during development by
# hardcoding the RFC's literal value instead of deriving it) would
# likely still fail some test below even without checking official hex
# vectors, since it would make encode/decode internally inconsistent
# too -- but that is a weaker guarantee than a real RFC vector check,
# which this file does not substitute for.


def test_a_group_backend_is_live():
    assert backend_name() in ("rbcl", "pure-python (TEST ONLY)")


def test_identity_point_is_all_zero_bytes():
    I = identity()
    assert I.b == b"\x00" * 32


def test_zero_times_base_is_identity():
    assert scalar_mul_base(Scalar.from_int(0)) == identity()


def test_base_point_matches_one_times_base():
    G = Point.base()
    assert G == scalar_mul_base(Scalar.from_int(1))


def test_small_multiples_are_distinct_and_consistent_with_addition():
    G = Point.base()
    multiples = [scalar_mul_base(Scalar.from_int(k)) for k in range(6)]
    # 0*B is the identity, 1*B..5*B are all distinct from each other and from identity
    assert multiples[0] == identity()
    assert len({m.b for m in multiples}) == 6
    # k*B + B == (k+1)*B via point addition, independent of scalar_mul_base's own internals
    for k in range(5):
        assert (multiples[k] + G) == multiples[k + 1]


def test_group_order_times_base_is_identity():
    """L*B == identity (standard group-order check): computed as
    (L-1)*B + B rather than L*B directly, since Scalar.from_int(L) % L
    reduces straight to 0 and would trivially (and uninterestingly)
    give the identity without the group-order arithmetic actually
    running through a nontrivial scalar multiplication first."""
    L = group_order()
    G = Point.base()
    almost = scalar_mul_base(Scalar.from_int(L - 1))
    assert (almost + G) == identity()


def test_scalar_arithmetic_matches_group_arithmetic():
    a, b = 7, 11
    Pa = scalar_mul_base(Scalar.from_int(a))
    Pb = scalar_mul_base(Scalar.from_int(b))
    Pab = scalar_mul_base(Scalar.from_int(a + b))
    assert (Pa + Pb) == Pab


def test_scalar_to_int_roundtrip():
    for x in (0, 1, 42, 2 ** 200, group_order() - 1):
        s = Scalar.from_int(x % group_order())
        assert s.to_int() == x % group_order()


def test_hash_to_point_is_deterministic_and_valid():
    p1 = Point.hash_to_point(b"janus-fl-test-vector")
    p2 = Point.hash_to_point(b"janus-fl-test-vector")
    p3 = Point.hash_to_point(b"a-different-string")
    assert p1 == p2
    assert p1 != p3
    assert p1.is_valid()


def test_msm_matches_sequential_scalar_mults():
    G = Point.base()
    H = Point.hash_to_point(b"msm-test-generator")
    scalars = [3, -1, 0, 5]
    points = [G, H, G, H]
    expected = identity()
    for s, p in zip(scalars, points):
        if s == 0:
            continue
        term = p.scalar_mul(Scalar.from_int(s % group_order()))
        expected = expected + term
    assert msm(scalars, points) == expected


def test_msm_empty_and_all_zero_is_identity():
    G = Point.base()
    assert msm([], []) == identity()
    assert msm([0, 0], [G, G]) == identity()


def test_existing_ec_modules_import_under_test_backend():
    """FIX.md F7's design note: every EC-dependent module in the repo
    must still import and run its own basic self-consistency under
    this backend before any new F7 code is trusted on top of it."""
    import crypto.pedersen  # noqa: F401
    import crypto.clsag  # noqa: F401
    import crypto.dkg  # noqa: F401
    import crypto.vrf  # noqa: F401
    import crypto.r_cons  # noqa: F401
    import crypto.rangeproof  # noqa: F401
    import crypto.elgamal_threshold  # noqa: F401
    import crypto.mixnet  # noqa: F401
    import crypto.norm_proof  # noqa: F401
    import crypto.schnorr_nizk  # noqa: F401
