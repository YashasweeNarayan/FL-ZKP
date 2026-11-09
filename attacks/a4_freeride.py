"""A4: Free-riding (ARCHITECTURE.md §10) -- submit zero, replay round
t-1's update, or falsely claim "not sampled" to skip real training
while still collecting the benefits of participation. Defense under
test: the lower norm bound (near-zero updates look suspicious against
the same B_t that catches A1, just from the other direction) plus
rho_t binding (a replayed round-(t-1) update carries round-(t-1)'s
nonce baked into its signature/tags, so replaying it verbatim against
round t's rho_t fails V1) plus the VRF disjunction in R_cons (a client
claiming "not sampled" must prove it via the VRF output tied to the
SAME secret key as everything else -- it cannot just assert it).
EC-DEPENDENT.
"""
from __future__ import annotations

from . import AttackResult
from crypto.vrf import evaluate as vrf_evaluate, is_sampled, verify as vrf_verify
from crypto.clsag import sign as clsag_sign, verify_with_scopes


def run_attack_replay(sk, pk_index: int, ring: list, rho_t_old: bytes, rho_e: bytes, rho_t_new: bytes, old_message: bytes):
    """Client signed a real update at round t-1 (rho_t_old), then tries
    to resubmit the SAME signature/message at round t without
    retraining. The signature was computed binding rho_t_old into its
    key image; re-verifying it against rho_t_new's tags fails because
    verify_with_scopes recomputes H_p(rho_t_new) and the key image
    I_t = sk*H_p(rho_t_old) does not satisfy the round-t closing
    equation under a different base point.
    """
    old_sig = clsag_sign(sk, pk_index, ring, old_message, rho_t_old, rho_e)
    replayed_valid_under_new_round = verify_with_scopes(old_sig, old_message, rho_t_new, rho_e)
    defense_triggered = not replayed_valid_under_new_round
    return AttackResult(
        "A4_freeride_replay", defense_triggered=defense_triggered, defense_name="rho_t_binding_V1",
        detail=f"replayed round-(t-1) signature verified against round t's rho_t: {replayed_valid_under_new_round} (should be False)",
    )


def run_attack_false_not_sampled_claim(sk, pk, vrf_alpha: bytes, sampling_rate: float):
    """Client was actually sampled (VRF says b_k=True) but wants to
    skip training and claim it wasn't sampled to submit a free cover-
    traffic (zero/noise) update instead. The R_cons relation ties the
    VRF output to the same sk as everything else (crypto/r_cons.py's
    prove_identity_binding), so the server/verifier can recompute the
    VRF output itself from the PUBLIC alpha and pk and check
    is_sampled -- the client cannot lie about its own sampling outcome
    without the VRF proof failing verification.
    """
    vrf_out = vrf_evaluate(sk, pk, vrf_alpha)
    true_sampling_decision = is_sampled(vrf_out.beta, sampling_rate)
    vrf_proof_valid = vrf_verify(vrf_out, pk, vrf_alpha)

    claimed_decision = False  # attacker LIES and claims "not sampled"
    lie_detected = vrf_proof_valid and (true_sampling_decision != claimed_decision)

    return AttackResult(
        "A4_freeride_false_claim", defense_triggered=lie_detected, defense_name="R_cons_VRF_disjunction",
        detail=f"true_sampling_decision={true_sampling_decision}, claimed={claimed_decision}, "
               f"VRF proof publicly recomputable -> lie detectable: {lie_detected}",
    )
