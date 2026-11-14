"""FIX.md F7f -- crypto/norm_proof.py's BisectionSession (the resumable
form of localize_fault's group-testing bisection). The bisection logic
itself is pure control flow with no group operations of its own, but
`crypto/norm_proof.py` pulls in `crypto/pedersen.py` at import time,
which constructs a generator point immediately -- so importing this
module at all needs a live backend, same as every other F7 test file.
Gate on whether ANY group backend is live, not on which one (see those
other files' comment on why -- gating on the env var alone would skip
this on Colab, the one place it most needs to run).
"""
from __future__ import annotations

import pytest

from crypto.ristretto import backend_name as _backend_name

if _backend_name() is None:
    pytest.skip(
        "no ristretto255 backend available: install rbcl, or set "
        "JANUS_ALLOW_PURE_PYTHON_GROUP=1 to use the test-only pure-Python "
        "backend (crypto/_ristretto_py.py)",
        allow_module_level=True,
    )

from crypto.norm_proof import BisectionSession, localize_fault, partition_group


def _guilty_predicate(guilty: set[int]):
    """A `check_group_opens_correctly` stand-in: a group opens
    correctly (returns True) iff it contains none of `guilty`."""
    def predicate(group):
        return not (set(group) & guilty)
    return predicate


def test_session_matches_localize_fault_on_a_clean_set():
    ids = list(range(12))
    predicate = _guilty_predicate(set())  # nobody guilty -> everything opens
    expected = localize_fault(ids, 0, b"seed", predicate)

    session = BisectionSession.start(ids, 0, b"seed")
    while not session.is_done():
        group = session.pending_group()
        session.submit_result(group, predicate(group))
    assert session.result.suspect_groups == expected.suspect_groups == []
    assert session.result.rounds_used == expected.rounds_used


def test_session_matches_localize_fault_with_a_cheater():
    ids = list(range(12))
    guilty = {7}
    predicate = _guilty_predicate(guilty)
    expected = localize_fault(ids, 3, b"seed-2", predicate, min_group_size=2)

    session = BisectionSession.start(ids, 3, b"seed-2", min_group_size=2)
    while not session.is_done():
        group = session.pending_group()
        session.submit_result(group, predicate(group))

    assert session.result.suspect_groups == expected.suspect_groups
    assert session.result.rounds_used == expected.rounds_used
    flat = {c for g in session.result.suspect_groups for c in g}
    assert guilty <= flat
    assert len(flat) < len(ids)


def test_pending_group_is_idempotent_before_submit_result():
    """A caller that crashes mid-round-trip and re-asks must get the
    SAME group back, not a new one popped off the frontier -- the
    actual property that makes this safe to suspend across a real
    network round trip."""
    ids = list(range(10))
    session = BisectionSession.start(ids, 0, b"seed", min_group_size=2)
    first = session.pending_group()
    second = session.pending_group()
    third = session.pending_group()
    assert first == second == third


def test_submit_result_rejects_a_different_group():
    ids = list(range(10))
    session = BisectionSession.start(ids, 0, b"seed", min_group_size=2)
    group = session.pending_group()
    wrong_group = [c for c in group[:-1]] if len(group) > 1 else [99]
    with pytest.raises(ValueError):
        session.submit_result(wrong_group, True)


def test_submit_result_without_a_pending_query_rejects():
    session = BisectionSession.start([0, 1, 2], 0, b"seed")
    with pytest.raises(ValueError):
        session.submit_result([0, 1, 2], True)  # never called pending_group()


def test_a_clean_group_is_never_split_or_suspected():
    ids = list(range(8))
    session = BisectionSession.start(ids, 0, b"seed", min_group_size=2)
    group = session.pending_group()
    session.submit_result(group, True)
    # cleared -- not queued again, not suspect
    assert group not in session.frontier
    assert group not in session.suspects


def test_result_reports_leftovers_mid_session_as_suspect():
    """Before is_done(), `.result` must still report whatever's queued
    or outstanding -- a server that needs a status mid-bisection
    (e.g. to log progress) gets a conservative, non-empty answer,
    never an empty "nothing wrong" result."""
    ids = list(range(12))
    guilty = {5}
    predicate = _guilty_predicate(guilty)
    session = BisectionSession.start(ids, 0, b"seed", min_group_size=2)
    group = session.pending_group()
    session.submit_result(group, predicate(group))
    # mid-session: not done yet (unless the very first split already
    # hit min_group_size, which it won't for 12 clients), but the
    # result property must not crash and must not silently drop the
    # still-queued groups.
    mid_result = session.result
    flat = {c for g in mid_result.suspect_groups for c in g}
    assert flat  # never empty mid-session when there's still a cheater unresolved


def test_max_rounds_limit_is_respected():
    ids = list(range(64))
    guilty = {1}
    predicate = _guilty_predicate(guilty)
    session = BisectionSession.start(ids, 0, b"seed", min_group_size=1, max_rounds=2)
    while not session.is_done():
        group = session.pending_group()
        session.submit_result(group, predicate(group))
    assert session.rounds_used <= 2
    expected = localize_fault(ids, 0, b"seed", predicate, min_group_size=1, max_rounds=2)
    assert session.result.suspect_groups == expected.suspect_groups


def test_partition_group_returns_none_at_the_floor():
    assert partition_group([0, 1, 2], b"seed", min_group_size=4) is None
    left, right = partition_group(list(range(10)), b"seed", min_group_size=2)
    assert sorted(left + right) == list(range(10))
    assert set(left).isdisjoint(right)
