"""A5: identity-laundering / transcript splicing (ARCHITECTURE.md §10,
marked the paper's "money figure"). An attacker takes a valid
ring-signature/key-image/escrow bundle from one client's genuine
submission and splices in a DIFFERENT commitment/update (or a
different VRF sampling claim) from elsewhere, hoping the pieces
individually verify even though they were never meant to go together.
Defense under test: V4 -- crypto.r_cons's AND-composition binds ONE
secret key across pk, I_t, P_e, and the VRF point in a SINGLE Sigma-
protocol proof with a SHARED challenge; splicing in a different
component (e.g. a different commitment's VRF alpha, or another
client's key image) breaks that single joint proof even if each
spliced-in piece, verified on its own against ITS OWN original
context, would have passed. "Ablate with R_cons off -> innocent client
framed." EC-DEPENDENT.
"""
from __future__ import annotations

from . import AttackResult
from crypto.r_cons import prove_identity_binding, verify_identity_binding


def run_attack(sk_victim, sk_attacker, rho_t: bytes, rho_e: bytes, vrf_alpha_victim: bytes, vrf_alpha_attacker: bytes):
    """The attacker has its OWN valid R_cons proof (sk_attacker, its
    own vrf_alpha). It tries to splice: reuse the attacker's AND-proof
    structure but swap in the VICTIM's public round/epoch tags (as if
    claiming the attacker's contribution came from the victim's
    identity), without actually knowing sk_victim.

    Concretely: the spliced "proof" presented to the verifier pairs
    the ATTACKER's genuine proof object with the VICTIM's (rho_t,
    rho_e, vrf_alpha) verification inputs, hoping and_verify doesn't
    notice the bases changed. It does notice: and_verify recomputes
    the Fiat-Shamir challenge from the EXACT bases passed to it, and
    the attacker's proof's responses were computed against a different
    set of bases (its own, not the victim's) -- the challenge-response
    algebra only closes for the bases it was actually built with.
    """
    transcript = b"r_cons:splice-attempt"

    attacker_proof = prove_identity_binding(sk_attacker, rho_t, rho_e, vrf_alpha_attacker, transcript)

    # honest baseline: attacker's own proof verifies against its OWN inputs
    attacker_proof_valid_honestly = verify_identity_binding(attacker_proof, rho_t, rho_e, vrf_alpha_attacker, transcript)

    # the splice: present the attacker's proof object but claim it
    # corresponds to the victim's vrf_alpha (as if impersonating which
    # round-sampling outcome belongs to whom)
    spliced_verifies = verify_identity_binding(attacker_proof, rho_t, rho_e, vrf_alpha_victim, transcript)

    defense_triggered = attacker_proof_valid_honestly and (not spliced_verifies)
    return AttackResult(
        "A5_splice", defense_triggered=defense_triggered, defense_name="V4_r_cons_and_binding",
        detail=f"attacker_proof_valid_on_own_inputs={attacker_proof_valid_honestly}, "
               f"spliced_proof_valid_on_victim_inputs={spliced_verifies} (must be False for the defense to hold)",
    )


def run_attack_key_image_splice(sk_attacker, pk_index_attacker: int, ring: list, rho_t: bytes, rho_e: bytes, victim_commitment_bytes: bytes):
    """A second splice variant: attacker signs its OWN message with
    its OWN key, then tries to claim the resulting signature
    authenticates the VICTIM's commitment (swap the message after
    signing). CLSAG's Fiat-Shamir transcript binds the message into
    every challenge in the ring equation (crypto/clsag.py's `sign`:
    `transcript = message + I_t.b + I_e.b`), so changing the message
    post-hoc breaks the closing equation at verification.
    """
    from crypto.clsag import sign as clsag_sign, verify_with_scopes

    attacker_message = b"attacker-genuine-commitment"
    sig = clsag_sign(sk_attacker, pk_index_attacker, ring, attacker_message, rho_t, rho_e)

    valid_on_own_message = verify_with_scopes(sig, attacker_message, rho_t, rho_e)
    valid_on_victim_message = verify_with_scopes(sig, victim_commitment_bytes, rho_t, rho_e)

    defense_triggered = valid_on_own_message and (not valid_on_victim_message)
    return AttackResult(
        "A5_splice_key_image", defense_triggered=defense_triggered, defense_name="V1_message_binding",
        detail=f"valid_on_own_message={valid_on_own_message}, valid_on_victim_message={valid_on_victim_message} (must be False)",
    )
