"""§6.2 opening protocol (ARCHITECTURE.md): committee-side orchestration
that maps an accused epoch pseudonym back to a real public key, only
once evidence (fl/evidence.py) crosses the calibrated theta. EC-
DEPENDENT -- not run in this sandbox. Pure orchestration over already-
tested primitives (crypto/committee.py's consent gate, crypto/dkg.py's
Feldman-verified shares, crypto/elgamal_threshold.py's threshold
decryption, crypto/r_cons.py's escrow-to-identity binding check) -- no
new cryptography is introduced here, only the control flow
ARCHITECTURE.md §6.2 specifies, in the same six numbered steps.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from crypto.elgamal_threshold import compute_partial, verify_partial, combine_partials
from crypto.r_cons import bind_escrow_to_identity


@dataclass
class OpeningRequest:
    epoch: int
    accused_pseudonym: bytes      # P_k^e.b, the epoch pseudonym under evidence
    evidence_total: float
    calibration_certificate: object  # fl.evidence.CalibrationCertificate
    escrow_ciphertext: object        # the accused round's tau_k (ElGamalCiphertext)


@dataclass
class OpeningVerdict:
    opened: bool
    opened_pk: object = None          # Point, if opened
    partials: list = field(default_factory=list)
    reason: str = ""


def collect_partial_from_holder(holder_id: int, share, pk_share_i, request: OpeningRequest, is_accused: bool):
    """§6.2 step 2's recusal rule: 'A key holder recuses itself if it
    is the accused' -- a holder cannot both be a suspect and sit on
    the committee deciding whether to open its own identity, which
    would be an obvious conflict of interest undermining the "any
    third party can re-verify" property step 5 requires.
    """
    if is_accused:
        return None
    transcript = f"opening:{request.epoch}:{request.accused_pseudonym.hex()}".encode()
    return compute_partial(request.escrow_ciphertext, share, pk_share_i, transcript)


def run_opening_protocol(
    request: OpeningRequest,
    roster_partials: dict,       # holder_id -> PartialDecryption, already collected
    roster_pk_shares: dict,      # holder_id -> Point (published Feldman commitments' pk_share_i)
    t: int,
    admin_authorized: bool,
) -> OpeningVerdict:
    """§6.2 steps 2-5. Step 1 (posting evidence+certificate to BB) is
    fl/evidence.py's job and is assumed already done by the time this
    is called; this function is the committee's half of the protocol.
    """
    if request.evidence_total <= request.calibration_certificate.theta:
        return OpeningVerdict(opened=False, reason="evidence below calibrated theta -- opening not authorized")

    if not admin_authorized:
        return OpeningVerdict(opened=False, reason="missing server/admin co-signature (step 2 authorization gate)")

    transcript = f"opening:{request.epoch}:{request.accused_pseudonym.hex()}".encode()
    verified_partials = []
    for holder_id, partial in roster_partials.items():
        pk_share_i = roster_pk_shares[holder_id]
        if verify_partial(partial, request.escrow_ciphertext, pk_share_i, transcript):
            verified_partials.append(partial)

    if len(verified_partials) < t:
        return OpeningVerdict(
            opened=False,
            partials=verified_partials,
            reason=f"only {len(verified_partials)} valid partials, need t={t}",
        )

    opened_pk = combine_partials(request.escrow_ciphertext, verified_partials[:t])

    return OpeningVerdict(opened=True, opened_pk=opened_pk, partials=verified_partials[:t])


def verify_opening_binds_to_identity_proof(verdict: OpeningVerdict, identity_proof) -> bool:
    """§6.2 step 5's re-verifiability: before revoking pk_k, confirm
    the opened plaintext actually matches the SAME pk that the
    accused round's R_cons identity-binding proof committed to (see
    crypto/r_cons.py's bind_escrow_to_identity docstring for why this
    check exists -- without it, an opening could point at a pk
    unrelated to the actual signer).
    """
    if not verdict.opened:
        return False
    return bind_escrow_to_identity(verdict.opened_pk, identity_proof)


@dataclass
class RevocationRecord:
    epoch: int
    revoked_pk: object
    new_ring_size: int


def revoke(verdict: OpeningVerdict, current_ring: list, epoch: int) -> tuple[list, RevocationRecord]:
    """§6.2 step 6: 'the named client's pk is revoked and the ring
    shrinks'. Returns the new ring and a record for §6.3's residual-
    anonymity accounting (H(T) = log2(n - o(T))) -- that accounting
    itself lives wherever the paper's figures are generated, not here;
    this function only performs the mechanical removal and reports
    what changed.
    """
    if not verdict.opened:
        raise ValueError("cannot revoke: opening did not succeed")
    new_ring = [pk for pk in current_ring if pk != verdict.opened_pk]
    if len(new_ring) != len(current_ring) - 1:
        raise AssertionError("revoked pk was not found exactly once in the current ring")
    return new_ring, RevocationRecord(epoch=epoch, revoked_pk=verdict.opened_pk, new_ring_size=len(new_ring))
