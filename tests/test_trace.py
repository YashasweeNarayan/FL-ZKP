"""Tests for fl/trace.py (§6.2 opening protocol, §6.3 residual
anonymity). Promoted from notebooks/traceability_demo/TRACEABILITY_DEMO.ipynb
(2026-11-18), which first validated this logic by hand against the
real crypto before these tests existed -- see FIX.md's "Traceability
demo" entry for that notebook's own writeup.

`residual_anonymity_bits` is pure math (no EC needed) and is tested
unconditionally. The opening-protocol tests need the real DKG +
threshold-ElGamal primitives (crypto/dkg.py, crypto/elgamal_threshold.py)
and are gated on a live ristretto255 backend, same pattern as
tests/test_round_end_to_end.py: they run against real libsodium (rbcl)
when installed, or the test-only pure-Python backend when
JANUS_ALLOW_PURE_PYTHON_GROUP=1 selects it, and skip if neither is
available.
"""
from __future__ import annotations

import math

import pytest

from fl.trace import residual_anonymity_bits


# ---------------------------------------------------------------------------
# residual_anonymity_bits -- pure math, no backend needed.
# ---------------------------------------------------------------------------

def test_residual_anonymity_full_ring_no_openings():
    # H(T) = log2(n - 0) = log2(n)
    assert residual_anonymity_bits(50, 0) == pytest.approx(math.log2(50))


def test_residual_anonymity_drops_after_one_opening():
    before = residual_anonymity_bits(50, 0)
    after = residual_anonymity_bits(50, 1)
    assert after < before
    assert after == pytest.approx(math.log2(49))


def test_residual_anonymity_monotonically_decreasing_in_openings():
    values = [residual_anonymity_bits(50, o) for o in range(0, 40)]
    assert all(values[i] > values[i + 1] for i in range(len(values) - 1)), (
        "H(T) must strictly decrease as more members are opened -- each "
        "opening can only shrink the anonymity set, never grow it"
    )


def test_residual_anonymity_bottoms_out_at_minus_inf_when_fully_opened():
    assert residual_anonymity_bits(10, 10) == float("-inf")
    assert residual_anonymity_bits(10, 15) == float("-inf")  # over-opened is still degenerate, not an error


def test_residual_anonymity_rejects_nonpositive_ring_size():
    with pytest.raises(ValueError):
        residual_anonymity_bits(0, 0)
    with pytest.raises(ValueError):
        residual_anonymity_bits(-5, 0)


def test_residual_anonymity_rejects_negative_opened_count():
    with pytest.raises(ValueError):
        residual_anonymity_bits(50, -1)
