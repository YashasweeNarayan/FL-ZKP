"""Independent V5 + opening re-verification (ARCHITECTURE.md's repo
layout: "verifier/audit.py -- independent V5 + opening re-verification",
marked with a star as a key novelty point: "Any third party can
re-verify the entire accusation"). EC-DEPENDENT -- not run in this
sandbox.

Deliberately a SEPARATE module from fl/server.py, even though V5's
math is identical to what the server itself could compute: the whole
point of V5 (ARCHITECTURE.md §4, "On V5") is that it does NOT require
trusting the server -- any party with only the public bulletin-board
contents (committed values c_k, the published sum, the published
sum-of-blindings) can run this file's checks completely independently
of whatever fl/server.py computed internally. If this module imported
and reused fl/server.py's own aggregation code path, a bug or
dishonesty in that path could silently escape detection; keeping the
two implementations independent (even though the underlying Pedersen-
homomorphism math is the same formula) is the actual security property
being tested, not just code hygiene.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from crypto.pedersen import Commitment, commit_sum, verify_opening
from crypto.ristretto import Scalar
from crypto.clsag import verify_with_scopes
from crypto.r_cons import verify_identity_binding, bind_escrow_to_identity
from crypto.elgamal_threshold import verify_partial, combine_partials


@dataclass
class AuditReport:
    v5_passed: bool
    detail: str
    discrepancy: "object" = None


def independent_v5_check(
    published_commitments: list,       # list[Point] -- c_k for k in accepted set A, read from BB
    published_sum_value: int,          # Sigma(q_k + m_k), read from BB
    published_sum_blinding: Scalar,    # Sigma r_k, read from BB
) -> AuditReport:
    """ARCHITECTURE.md §4's V5, run by a THIRD PARTY (this module),
    not the server: recompute Pi c_k from the public commitments, and
    independently check it equals Com(published_sum_value;
    published_sum_blinding) -- the server's claimed opening of that
    same product. Any mismatch is direct, reproducible evidence the
    server lied about the aggregate, without needing to trust
    anything the server computed internally.
    """
    commitments = [Commitment(p) for p in published_commitments]
    product = commit_sum(commitments)

    claimed_value_scalar = Scalar.from_int(published_sum_value % (2**252))
    matches = verify_opening(product, claimed_value_scalar, published_sum_blinding)

    return AuditReport(
        v5_passed=matches,
        detail="published aggregate opening matches the product of public commitments" if matches
               else "MISMATCH: server's published (sum_value, sum_blinding) does not open to Pi(c_k) -- server dishonesty or an aggregation bug",
        discrepancy=None if matches else {"expected_commitment": product.point.b, "claimed_commitment": commit_sum([Commitment(p) for p in published_commitments]).point.b},
    )


@dataclass
class OpeningAuditReport:
    opening_valid: bool
    detail: str


def independent_opening_reverification(
    escrow_ciphertext,
    partials: list,
    pk_shares: dict,
    t: int,
    transcript: bytes,
    identity_binding_proof,
    rho_t: bytes,
    rho_e: bytes,
    vrf_alpha: bytes,
) -> OpeningAuditReport:
    """§6.2 step 5: 'Any third party can re-verify the entire
    accusation' -- re-runs every check the opening committee claimed
    to have performed, from only the public transcript the committee
    posted (partials, their Chaum-Pedersen proofs, the roster's
    published pk_shares, and the R_cons proof from the accused round).
    """
    valid_partials = [p for p in partials if verify_partial(p, escrow_ciphertext, pk_shares[p.holder_id], transcript)]
    if len(valid_partials) < t:
        return OpeningAuditReport(opening_valid=False, detail=f"only {len(valid_partials)} of {len(partials)} partials verify; need t={t}")

    opened_pk = combine_partials(escrow_ciphertext, valid_partials[:t])

    identity_matches = bind_escrow_to_identity(opened_pk, identity_binding_proof)
    if not identity_matches:
        return OpeningAuditReport(opening_valid=False, detail="opened plaintext does not match the accused round's R_cons identity-binding proof -- opening points at the wrong key")

    r_cons_valid = verify_identity_binding(identity_binding_proof, rho_t, rho_e, vrf_alpha, transcript)
    if not r_cons_valid:
        return OpeningAuditReport(opening_valid=False, detail="the accused round's own R_cons proof does not verify -- the accusation's evidentiary basis is itself invalid")

    return OpeningAuditReport(opening_valid=True, detail=f"re-verified {len(valid_partials[:t])} partials, identity binding, and R_cons independently -- opening stands")
