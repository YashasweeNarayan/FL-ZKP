"""Committee roster state machine with the consent rule (ARCHITECTURE.md §1.3, §6.4).

EC-DEPENDENT (Schnorr signatures need ristretto) — see ristretto.py's
header for the Colab-validation note.

*** This is the single most important guardrail in the whole repo. ***
REVIEW.md's D3 finding: if the server could install a new roster
unilaterally, it could swap in sock puppets and de-anonymize every
client alone. The fix is purely procedural and cheap to enforce: a
roster transition is valid ONLY if `t` members of the OUTGOING
committee signed it. install_roster() below is written so that
skipping this check requires deliberately deleting the one `if`
statement that does it — tests/test_committee.py (Colab) asserts a
server-only proposal is rejected, specifically so nobody can
"simplify" this file later without a red test.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .ristretto import Point, Scalar
from .schnorr_nizk import SchnorrProof, schnorr_prove, schnorr_verify


@dataclass
class RosterProposal:
    epoch: int
    new_members: list[int]
    admin_signature: SchnorrProof
    outgoing_signatures: dict[int, SchnorrProof] = field(default_factory=dict)


def propose(epoch: int, new_members: list[int], admin_sk: Scalar, admin_pk: Point) -> RosterProposal:
    transcript = _proposal_transcript(epoch, new_members)
    sig = schnorr_prove(admin_sk, Point.base(), transcript)
    return RosterProposal(epoch, new_members, sig)


def sign_as_outgoing_member(proposal: RosterProposal, member_id: int, member_sk: Scalar) -> SchnorrProof:
    transcript = _proposal_transcript(proposal.epoch, proposal.new_members)
    return schnorr_prove(member_sk, Point.base(), transcript)


def install_roster(
    proposal: RosterProposal,
    signatures: dict[int, SchnorrProof],
    outgoing_member_pks: dict[int, Point],
    admin_pk: Point,
    t: int,
) -> bool:
    """Returns True iff the roster transition is authorized. THIS IS THE
    GUARDRAIL: fails unless (a) the admin's proposal signature verifies,
    AND (b) at least t DISTINCT outgoing members' signatures verify.
    A malicious server holding only its own "authorization" and no
    outgoing-committee signatures cannot pass this -- by construction,
    not by policy.
    """
    transcript = _proposal_transcript(proposal.epoch, proposal.new_members)

    if not schnorr_verify(proposal.admin_signature, admin_pk, Point.base(), transcript):
        return False

    valid_signers = set()
    for member_id, sig in signatures.items():
        if member_id not in outgoing_member_pks:
            continue  # not actually an outgoing member -- doesn't count
        if schnorr_verify(sig, outgoing_member_pks[member_id], Point.base(), transcript):
            valid_signers.add(member_id)

    return len(valid_signers) >= t  # <-- THE guardrail. Do not relax without a new test.


def _proposal_transcript(epoch: int, new_members: list[int]) -> bytes:
    payload = f"janus-fl-roster:{epoch}:{','.join(map(str, sorted(new_members)))}"
    return payload.encode()
