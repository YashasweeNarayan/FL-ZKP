"""Opening-protocol tests for fl/trace.py (§6.2, ARCHITECTURE.md).
Promoted from notebooks/traceability_demo/TRACEABILITY_DEMO.ipynb
(2026-11-18), which first validated this logic by hand against the
real crypto before these tests existed -- see FIX.md's "Traceability
demo" entry for that notebook's own writeup.

Split out from tests/test_trace.py (which covers `residual_anonymity_bits`,
pure math, no backend needed) because this file needs the real DKG +
threshold-ElGamal primitives (crypto/dkg.py, crypto/elgamal_threshold.py)
and is gated on a live ristretto255 backend. A module-level
`pytest.skip(allow_module_level=True)` skips this ENTIRE file when no
backend is available -- putting backend-independent tests in the same
file would have silently skipped them too, which is exactly why this
is a separate file rather than a later section of test_trace.py.

Runs against real libsodium (rbcl) when installed, or the test-only
pure-Python backend when JANUS_ALLOW_PURE_PYTHON_GROUP=1 selects it
(same gate as tests/test_round_end_to_end.py); skips cleanly if
neither is available.
"""
from __future__ import annotations

import math

import pytest

from fl.trace import residual_anonymity_bits

from crypto.ristretto import backend_name as _backend_name

if _backend_name() is None:
    pytest.skip(
        "no ristretto255 backend available: install rbcl, or set "
        "JANUS_ALLOW_PURE_PYTHON_GROUP=1 to use the test-only pure-Python "
        "backend (crypto/_ristretto_py.py)",
        allow_module_level=True,
    )

from dataclasses import dataclass

from crypto.dkg import generate_contribution, verify_sub_share, finalize, pk_share_for
from crypto.elgamal_threshold import compute_partial, encrypt
from crypto.ristretto import Scalar, scalar_mul_base
from fl.trace import OpeningRequest, RevocationRecord, revoke, run_opening_protocol

N_C, T = 10, 7


@dataclass
class _StubCalibrationCertificate:
    theta: float


def _build_committee():
    """Real Pedersen DKG over N_C=10 holders, threshold T=7 -- the
    project's actual committee parameters (configs/mvp_3day.yaml)."""
    holder_ids = list(range(1, N_C + 1))
    contributions = [generate_contribution(h, T, holder_ids) for h in holder_ids]
    for c in contributions:
        for j in holder_ids:
            assert verify_sub_share(c.sub_shares[j], c.feldman_commitments)
    pk_trace, final_shares = finalize(contributions, holder_ids)
    pk_shares = {i: pk_share_for(i, contributions) for i in holder_ids}
    return holder_ids, pk_trace, final_shares, pk_shares


def _make_request(pk_trace, escrowed_pk):
    ct = encrypt(escrowed_pk, pk_trace)
    request = OpeningRequest(
        epoch=1,
        accused_pseudonym=b"test-epoch-pseudonym",
        evidence_total=999.0,
        calibration_certificate=_StubCalibrationCertificate(theta=10.0),
        escrow_ciphertext=ct,
    )
    transcript = f"opening:{request.epoch}:{request.accused_pseudonym.hex()}".encode()
    return request, transcript


def _partials_from(holder_ids, final_shares, pk_shares, request, transcript):
    return {
        i: compute_partial(request.escrow_ciphertext, final_shares[i], pk_shares[i], transcript)
        for i in holder_ids
    }


def test_opening_succeeds_with_all_holders_and_recovers_exact_identity():
    holder_ids, pk_trace, final_shares, pk_shares = _build_committee()
    accused_pk = scalar_mul_base(Scalar.from_int(424242))
    request, transcript = _make_request(pk_trace, accused_pk)

    verdict = run_opening_protocol(
        request,
        _partials_from(holder_ids, final_shares, pk_shares, request, transcript),
        pk_shares,
        T,
        admin_authorized=True,
    )

    assert verdict.opened
    assert verdict.opened_pk == accused_pk


def test_opening_succeeds_at_exactly_threshold_holders():
    """t=7 of 10 present should still open -- the boundary case itself
    succeeds (only fewer than t fails)."""
    holder_ids, pk_trace, final_shares, pk_shares = _build_committee()
    accused_pk = scalar_mul_base(Scalar.from_int(13))
    request, transcript = _make_request(pk_trace, accused_pk)

    present = holder_ids[:T]  # exactly 7 of 10
    verdict = run_opening_protocol(
        request,
        _partials_from(present, final_shares, pk_shares, request, transcript),
        pk_shares,
        T,
        admin_authorized=True,
    )

    assert verdict.opened
    assert verdict.opened_pk == accused_pk


def test_opening_fails_below_threshold_holders():
    """Dropping 4 of 10 (holders 1-4) leaves 6 < t=7 -- below
    threshold, must fail: no traceability."""
    holder_ids, pk_trace, final_shares, pk_shares = _build_committee()
    accused_pk = scalar_mul_base(Scalar.from_int(99))
    request, transcript = _make_request(pk_trace, accused_pk)

    missing = {1, 2, 3, 4}
    present = [i for i in holder_ids if i not in missing]
    assert len(present) == 6

    verdict = run_opening_protocol(
        request,
        _partials_from(present, final_shares, pk_shares, request, transcript),
        pk_shares,
        T,
        admin_authorized=True,
    )

    assert not verdict.opened
    assert "6" in verdict.reason and "7" in verdict.reason


def test_opening_fails_below_calibrated_theta_regardless_of_holders():
    """Evidence below theta must block opening even with every holder
    present -- the evidence gate (step 1) comes before the
    partial-decryption threshold (steps 2-5), not after."""
    holder_ids, pk_trace, final_shares, pk_shares = _build_committee()
    accused_pk = scalar_mul_base(Scalar.from_int(7))
    ct = encrypt(accused_pk, pk_trace)
    request = OpeningRequest(
        epoch=2,
        accused_pseudonym=b"low-evidence-pseudonym",
        evidence_total=1.0,  # below theta
        calibration_certificate=_StubCalibrationCertificate(theta=10.0),
        escrow_ciphertext=ct,
    )
    transcript = f"opening:{request.epoch}:{request.accused_pseudonym.hex()}".encode()

    verdict = run_opening_protocol(
        request,
        _partials_from(holder_ids, final_shares, pk_shares, request, transcript),
        pk_shares,
        T,
        admin_authorized=True,
    )

    assert not verdict.opened
    assert "theta" in verdict.reason


def test_opening_fails_without_admin_authorization():
    holder_ids, pk_trace, final_shares, pk_shares = _build_committee()
    accused_pk = scalar_mul_base(Scalar.from_int(55))
    request, transcript = _make_request(pk_trace, accused_pk)

    verdict = run_opening_protocol(
        request,
        _partials_from(holder_ids, final_shares, pk_shares, request, transcript),
        pk_shares,
        T,
        admin_authorized=False,
    )

    assert not verdict.opened
    assert "authoriz" in verdict.reason.lower()


def test_revoke_shrinks_ring_by_exactly_one_and_matches_residual_anonymity():
    holder_ids, pk_trace, final_shares, pk_shares = _build_committee()
    accused_pk = scalar_mul_base(Scalar.from_int(2026))
    request, transcript = _make_request(pk_trace, accused_pk)

    verdict = run_opening_protocol(
        request,
        _partials_from(holder_ids, final_shares, pk_shares, request, transcript),
        pk_shares,
        T,
        admin_authorized=True,
    )
    assert verdict.opened

    ring = [accused_pk] + [scalar_mul_base(Scalar.random()) for _ in range(49)]
    new_ring, record = revoke(verdict, ring, epoch=request.epoch)

    assert len(new_ring) == len(ring) - 1 == record.new_ring_size == 49
    assert accused_pk not in new_ring

    before = residual_anonymity_bits(len(ring), n_opened_so_far=0)
    after = residual_anonymity_bits(len(ring), n_opened_so_far=1)
    assert before == pytest.approx(math.log2(50))
    assert after == pytest.approx(math.log2(49))
    assert after < before


def test_revoke_raises_if_opening_did_not_succeed():
    failed_verdict_reason = "evidence below calibrated theta -- opening not authorized"
    from fl.trace import OpeningVerdict

    failed = OpeningVerdict(opened=False, reason=failed_verdict_reason)
    with pytest.raises(ValueError):
        revoke(failed, current_ring=[scalar_mul_base(Scalar.random())], epoch=1)
