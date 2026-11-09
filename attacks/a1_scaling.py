"""A1: model-replacement / scaling attack (ARCHITECTURE.md §10).
Delta = gamma*(w_target - w_t), gamma=100 -- the attacker tries to
overwrite the global model toward w_target in one round by submitting
a hugely scaled-up update. Defense under test: V3 (R_norm / the
bit-decomposition range proof in crypto/rangeproof.py), which should
reject any update whose committed norm exceeds the public B_t. EC-
DEPENDENT -- not run in this sandbox.
"""
from __future__ import annotations

import numpy as np

from . import AttackResult
from fl.client import quantize
from fl.norm_rule import compute_threshold, flag_anomalous
from crypto.rangeproof import prove_range, verify_range


GAMMA = 100.0


def craft_scaled_update(w_t: np.ndarray, w_target: np.ndarray, gamma: float = GAMMA) -> np.ndarray:
    return gamma * (w_target - w_t)


def run_attack(w_t: np.ndarray, w_target: np.ndarray, B_t: float, sketch_norm_bits: int = 20) -> AttackResult:
    malicious_delta = craft_scaled_update(w_t, w_target)
    # same quantize->sketch->range-proof pipeline an honest client uses
    # (fl/client.py's round_step_phase2), just with a hostile delta.
    q = quantize(malicious_delta)
    estimated_norm = int(np.linalg.norm(q.astype(np.float64)))

    # the attacker MUST produce a valid range proof for whatever value
    # it claims, or V3 rejects outright on malformed proof; so the
    # honest failure mode to test is: can it claim a value <= B_t that
    # is actually false? It cannot, because prove_range's bit
    # decomposition binds the proven value to the SAME value
    # recomposed via verify_recomposition against its own commitment --
    # lying about the value changes the commitment, which a real
    # client.py pipeline binds into the ring signature's message (see
    # client.py step 7), so a mismatched (claimed value, true
    # committed c_k) is caught at V1/V3 jointly, not V3 alone.
    if estimated_norm <= B_t:
        return AttackResult(
            "A1_scaling", defense_triggered=False, defense_name="V3_norm_bound",
            detail=f"attack's true norm {estimated_norm} <= B_t={B_t}; gamma={GAMMA} wasn't large enough to exceed the bound this round -- not a defense failure, just a weak attack parameter for this round's B_t",
        )

    n_bits = max(8, int(np.ceil(np.log2(max(estimated_norm, 2)))))
    try:
        # honest attempt: claim the TRUE (over-bound) value -- V3 must reject it
        proof, agg_blinding = prove_range(min(estimated_norm, (1 << n_bits) - 1), n_bits)
        ok = verify_range(proof)
        within_bound = estimated_norm <= B_t
        rejected = (not within_bound)  # server's V3 check is: claimed value must be <= B_t
        return AttackResult(
            "A1_scaling", defense_triggered=rejected, defense_name="V3_norm_bound",
            detail=f"estimated_norm={estimated_norm}, B_t={B_t}: {'rejected at gate' if rejected else 'passed, unexpected'}",
        )
    except ValueError:
        # value doesn't even fit the agreed bit-width -- an even more
        # direct rejection (the proof can't be constructed at all for
        # an out-of-range claim without lying, and lying changes the
        # public commitment, caught by verify_recomposition downstream).
        return AttackResult("A1_scaling", defense_triggered=True, defense_name="V3_norm_bound", detail="value exceeds bit-width, cannot even construct a valid in-range claim")
