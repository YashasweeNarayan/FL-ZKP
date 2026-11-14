"""FIX.md F7f -- fl/suspect_tracker.py. Pure set/bookkeeping logic, no
crypto, no backend to gate on: these run everywhere.
"""
from __future__ import annotations

from fl.suspect_tracker import SuspectTracker


def test_a_single_report_never_bans_the_whole_group():
    """One suspect group alone is not proof against any one member --
    it must quarantine, not ban."""
    t = SuspectTracker()
    newly_banned = t.record([[3, 4, 5]])
    assert newly_banned == set()
    assert t.banned == set()
    assert t.quarantined == {3: 3, 4: 3, 5: 3}


def test_repeated_disjoint_evidence_narrows_to_the_cheater():
    """The actual cheater (3) shows up in two structurally different
    suspect groups across two rounds; the innocent members differ
    between rounds, so the intersection narrows to just the cheater."""
    t = SuspectTracker(narrow_threshold=1)
    t.record([[3, 4, 5]])
    newly_banned = t.record([[3, 6, 7]])
    assert newly_banned == {3}
    assert t.banned == {3}
    # the innocent members of either group were never banned
    assert 4 not in t.banned and 5 not in t.banned
    assert 6 not in t.banned and 7 not in t.banned


def test_two_reports_that_do_not_narrow_do_not_ban():
    """If every report keeps implicating the SAME group (no new
    information), the intersection never shrinks below the group size,
    so nobody gets banned just from repetition."""
    t = SuspectTracker(narrow_threshold=1)
    t.record([[3, 4, 5]])
    newly_banned = t.record([[3, 4, 5]])
    assert newly_banned == set()
    assert t.banned == set()


def test_quarantine_counts_down_and_expires():
    t = SuspectTracker(quarantine_rounds=2)
    t.record([[3, 4]])
    assert t.quarantined == {3: 2, 4: 2}
    t.tick()
    assert t.quarantined == {3: 1, 4: 1}
    t.tick()
    assert t.quarantined == {}


def test_eligible_excludes_quarantined_and_banned():
    t = SuspectTracker(narrow_threshold=1)
    t.record([[3, 4, 5]])   # quarantines 3, 4, 5
    t.record([[3, 6, 7]])   # bans 3 (intersection narrows), quarantines 6, 7 too
    pool = list(range(8))
    elig = t.eligible(pool)
    assert 3 not in elig        # banned
    assert 4 not in elig and 5 not in elig and 6 not in elig and 7 not in elig  # quarantined
    assert elig == [0, 1, 2]


def test_ban_is_permanent_across_ticks():
    t = SuspectTracker(narrow_threshold=1, quarantine_rounds=1)
    t.record([[3, 4, 5]])
    t.record([[3, 6, 7]])
    assert 3 in t.banned
    for _ in range(5):
        t.tick()
    assert 3 in t.banned
    assert 3 not in t.eligible(range(8))


def test_empty_and_no_groups_are_safe_no_ops():
    t = SuspectTracker()
    assert t.record([]) == set()
    assert t.banned == set() and t.quarantined == {}
    t.tick()  # must not raise with nothing quarantined
    assert t.eligible(range(5)) == [0, 1, 2, 3, 4]


def test_a_larger_first_group_still_narrows_on_the_second_disjoint_report():
    """Same narrowing property with a bigger first group (4 members
    instead of 3): still takes exactly the second, disjoint-outside-the-
    cheater report to ban."""
    t = SuspectTracker(narrow_threshold=1)
    t.record([[9, 1, 2, 3]])
    assert t.banned == set()  # first report: intersection is still the whole group
    newly_banned = t.record([[9, 4, 5]])
    assert newly_banned == {9}
    assert t.banned == {9}
