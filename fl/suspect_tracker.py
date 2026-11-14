"""FIX.md F7f -- what HAPPENS to a suspect group. Pure bookkeeping, no
crypto: F7f's bisection (crypto/norm_proof.py::localize_fault) already
does the hard, privacy-budgeted work of narrowing a failed round down
to a small suspect group; this module decides what the server DOES
with that group across rounds, closing the "still open" item FIX.md
and STATUS.md had logged for it (recorded + aborted, nothing excluded
a repeat offender).

F7f never names an individual -- by design (see crypto/norm_proof.py's
accountability-anonymity comment: narrowing all the way to one client
would hand the server that client's update in the clear). This tracker
never violates that: it only ever acts on the GROUP SETS F7f itself
returned, never on anything derived from an update's content.

Two actions:

- QUARANTINE: every member of a newly-reported suspect group sits out
  the next `quarantine_rounds` rounds. Cheap, reversible, and the
  right default response to a single report -- a suspect group is not
  proof against any one member of it.

- ESCALATE TO BAN: if the ACTUAL cheater keeps misbehaving, they show
  up in suspect groups across multiple, structurally DIFFERENT rounds
  (`localize_fault`'s bisection partitions the accepted set differently
  each round -- see `partition_group`'s seed). The INTERSECTION of the
  groups a given client has appeared in only ever shrinks or stays the
  same as more reports arrive, and it always contains at least that
  client (every group they were reported in contains them, by
  construction). Once it shrinks to `narrow_threshold` members or
  fewer, those members are banned outright. This spends NO additional
  privacy budget: it is pure set arithmetic over disclosures F7f
  already made, not a new inference about anyone's data.

A single suspect group alone never bans anyone (its intersection with
itself is the whole group); only independent evidence across rounds
narrows it. That asymmetry is deliberate -- see `test_a_single_report_
never_bans_the_whole_group` in tests/test_suspect_tracker.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SuspectTracker:
    narrow_threshold: int = 1
    quarantine_rounds: int = 3
    history: dict = field(default_factory=dict)      # client_idx -> list[frozenset[int]]
    quarantined: dict = field(default_factory=dict)   # client_idx -> rounds_remaining
    banned: set = field(default_factory=set)

    def record(self, suspect_groups) -> set:
        """Call once per round that ended with F7f-attributed suspect
        groups (an empty list/absence of groups -- including an F7e
        failure with no blame_material_provider -- means nothing to
        record). Returns the set of client indices newly banned by
        THIS call, if any.
        """
        newly_banned = set()
        for group in suspect_groups:
            g = frozenset(group)
            if not g:
                continue
            for idx in g:
                self.history.setdefault(idx, []).append(g)
                self.quarantined[idx] = self.quarantine_rounds
            for idx in g:
                groups_for_idx = self.history[idx]
                common = set.intersection(*(set(x) for x in groups_for_idx))
                if len(common) <= self.narrow_threshold and not common <= self.banned:
                    newly_banned |= (common - self.banned)
                    self.banned |= common
        return newly_banned

    def tick(self) -> None:
        """Call once per round, whether or not it aborted, to count
        down active quarantines. A banned client has no quarantine
        counter left to count down (banned is permanent, not a longer
        quarantine)."""
        for idx in list(self.quarantined):
            self.quarantined[idx] -= 1
            if self.quarantined[idx] <= 0:
                del self.quarantined[idx]

    def eligible(self, all_indices) -> list:
        """The client pool for the NEXT round: banned clients are gone
        for good, quarantined ones sit out until their counter reaches
        zero. Preserves the input order/type (any iterable in, a list
        out)."""
        return [i for i in all_indices if i not in self.banned and i not in self.quarantined]
