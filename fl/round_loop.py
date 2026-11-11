"""Shared round-loop orchestration for CASE1 (MNIST) and CASE2
(FEMNIST), extracted into one place specifically so the two notebooks
can't silently drift out of sync -- CASE2_NONIID_FEMNIST.ipynb's own
markdown cell flagged that risk before this function existed, rather
than duplicating the wiring in both notebook generators.

TORCH-DEPENDENT (calls fl.client's local training step) and
EC-DEPENDENT (every crypto primitive it wires together, transitively).
Not run in this sandbox -- validate on Colab, same as every other fl/*
module (see STATUS.md).

This module does NOT reimplement any crypto or training logic; it only
calls fl/client.py, fl/server.py, fl/norm_rule.py and
verifier/audit.py in the order ARCHITECTURE.md's round protocol
specifies (same discipline fl/client.py's own docstring states for
itself) -- the only new risk surface here is ORDERING/WIRING.
"""
from __future__ import annotations

import os

import numpy as np

from crypto.secagg import derive_pairwise_secret_via_x25519
from crypto.sketch import seed_commitment
from verifier.audit import independent_v5_check

from .client import ClientKeys, ClientRoundContext, round_step_phase1, round_step_phase2
from .server import run_round, apply_update
from .norm_rule import bootstrap_from_server_root, update_threshold_ema
from .models import flatten_state_dict, unflatten_to_state_dict


def build_x25519_keys(n_clients: int):
    """One long-term X25519 keypair per client, for secure-aggregation's
    pairwise masking -- a DIFFERENT curve from the ristretto255 signing
    keys (see crypto/secagg.py's own note on this). Long-term: computed
    once, reused across every round, unlike the per-round masking nonce
    rho_t.
    """
    import nacl.public

    sks = [nacl.public.PrivateKey.generate() for _ in range(n_clients)]
    pks = [sk.public_key for sk in sks]
    return sks, pks


def precompute_pairwise_secrets(x25519_sks: list, x25519_pks: list) -> dict:
    """{client_idx: {other_idx: shared_secret_bytes}} for every ordered
    pair of the n_clients -- computed ONCE up front (O(n^2) ECDH calls,
    cheap: n=50 is 2450 scalarmults, milliseconds total), since the
    secret itself doesn't change round to round -- only the per-round
    nonce rho_t does, inside secagg.py's pairwise_mask.
    """
    n = len(x25519_sks)
    out = {}
    for i in range(n):
        out[i] = {
            j: derive_pairwise_secret_via_x25519(x25519_sks[i], x25519_pks[j])
            for j in range(n) if j != i
        }
    return out


def run_training_loop(
    model,
    model_dim: int,
    client_datasets: list,       # indexed by client_idx, each with .images/.labels
    ring: list,                  # list[Point], published ring (len n_clients)
    client_sks: list,            # list[Scalar], one per client, matching `ring`
    pairwise_secrets: dict,      # from precompute_pairwise_secrets
    pk_trace,                    # Point, the committee's PK_trace
    server_root_norms: np.ndarray,
    CFG: dict,
    rounds: int = None,
    lr: float = 0.01,
    epochs: int = 1,
    verbose: bool = True,
):
    """Runs ARCHITECTURE.md §4's round protocol for `rounds` rounds over
    one model: VRF self-sampling -> local train -> DP noise -> quantize
    -> double-mask -> commit -> escrow -> ring-sign (phase 1); sketch ->
    range proof -> R_cons identity binding (phase 2, after the server
    "reveals" the sketch seed); V1-V5 verification + k_min gate +
    aggregation (fl/server.py); an INDEPENDENT V5 re-check
    (verifier/audit.py, deliberately not reusing fl/server.py's own
    computation -- see that module's docstring); apply_update; and a
    robust median/MAD norm threshold that adapts round to round via
    EMA (fl/norm_rule.py).

    Returns (final_w_t, history), history a list of one dict per round
    (round index, aborted/accepted counts, V5 pass/fail, the resulting
    B_t) -- enough to plot convergence and audit-pass-rate afterward
    without re-deriving it from scratch.

    *** PERFORMANCE NOTE -- READ BEFORE LAUNCHING A FULL RUN ***
    crypto/sketch.py's SketchProjector is a CPU/NumPy CORRECTNESS
    reference (see its own PERFORMANCE NOTE): measured at ~16
    min/client/round extrapolated to ResNet-50's real d~23.5M. A full
    45-round x 50-client run at that scale, with this reference
    implementation, would take on the order of DAYS, not hours. This
    function is for validating the PROTOCOL WIRING is correct (call
    it with a small `rounds`, a handful of clients, or a reduced
    model_dim first) -- not for launching the real MVP-scale run until
    sketch.py's GPU/torch reimplementation lands (STATUS.md TODO #3).

    *** ONE REMAINING HONEST PLACEHOLDER: `server_root_norms` ***
    The caller is expected to pass real norms computed from the
    server's own root-set data (ROADMAP.md §3.4: "server root set is
    used for B_t bootstrap at round 0"). Computing those properly means
    running several pseudo-client local-train + sketch passes over
    server-root subsamples BEFORE round 0 -- itself bounded by the same
    sketch performance wall above, so it is not done inside this
    function. If the caller passes synthetic/placeholder norms instead
    (as both notebooks currently do), that is a real, flagged
    simplification affecting only round 0's starting B_t (every
    subsequent round's B_t is the real EMA-updated value from real
    accepted submissions' real sketch norms) -- not a fabricated result
    for the round loop itself.
    """
    n_clients = len(client_datasets)
    rounds = rounds if rounds is not None else CFG["mvp_scope"]["rounds_per_run"]
    sampling_rate = CFG["protocol"]["sampling_rate"]
    sigma_dp = CFG["dp"]["sigma"]
    sketch_m = CFG["sketch"]["m"]
    epoch_rounds = CFG["protocol"]["epoch_rounds"]
    k = CFG["norm_rule"]["k"]
    ema_alpha = CFG["norm_rule"]["ema_alpha"]

    w_t, model_shapes = flatten_state_dict(model)
    threshold = bootstrap_from_server_root(server_root_norms, k=k)
    history = []

    for t_round in range(rounds):
        rho_t = f"round-{t_round}".encode()
        rho_e = f"epoch-{t_round // epoch_rounds}".encode()
        vrf_alpha = rho_t

        # Server commits to this round's sketch seed BEFORE any client
        # commitment is posted -- sketch.py's soundness lemma needs the
        # seed chosen independent of any client's q_k. h_sigma is what
        # would be posted to the bulletin board at this point in a real
        # deployment; this simulation has no separate BB store, so it's
        # just logged for anyone reconstructing the protocol transcript.
        sigma_t = os.urandom(32)
        h_sigma = seed_commitment(sigma_t)

        ctx_by_client = {}
        phase1_by_client = {}
        for client_idx in range(n_clients):
            ctx = ClientRoundContext(
                keys=ClientKeys(sk=client_sks[client_idx], pk=ring[client_idx], index=client_idx),
                ring=ring,
                pairwise_secrets=pairwise_secrets[client_idx],
                self_mask_secret_int=int.from_bytes(os.urandom(32), "little"),
                model_dim=model_dim,
            )
            ctx_by_client[client_idx] = ctx
            phase1_by_client[client_idx] = round_step_phase1(
                ctx, w_t, rho_t, rho_e, vrf_alpha, sampling_rate, sigma_dp,
                n_clients, client_datasets[client_idx], model, lr, epochs, pk_trace,
            )

        # sigma_t "revealed" now that every c_k is committed -- reuse
        # each client's OWN ctx from phase 1 (not a fresh one), since
        # round_step_phase2 needs the SAME ctx.keys.sk for R_cons.
        phase2_by_client = {}
        for client_idx in range(n_clients):
            phase2_by_client[client_idx] = round_step_phase2(
                ctx_by_client[client_idx], phase1_by_client[client_idx],
                sigma_t, sketch_m, threshold.B_t, rho_t, rho_e, vrf_alpha,
            )

        submissions = [(i, phase1_by_client[i]) for i in range(n_clients)]
        outcome = run_round(submissions, phase2_by_client, rho_t, rho_e, vrf_alpha, threshold.B_t, n_clients, model_dim)

        if outcome.aborted:
            if verbose:
                print(f"round {t_round}: ABORTED -- {outcome.abort_reason}")
            history.append({"round": t_round, "aborted": True, "reason": outcome.abort_reason})
            continue

        accepted_commitments = [phase1_by_client[i].commitment_point for i in outcome.accepted_client_indices]
        audit = independent_v5_check(accepted_commitments, outcome.published_sum_value, outcome.published_sum_blinding)

        w_t = apply_update(w_t, outcome.aggregated_delta)

        next_round_norms = np.array([
            np.linalg.norm(phase2_by_client[i].sketch) for i in outcome.accepted_client_indices
        ])
        threshold = update_threshold_ema(threshold, next_round_norms, k=k, alpha=ema_alpha)

        if verbose:
            v5_msg = "PASS" if audit.v5_passed else f"FAIL: {audit.detail}"
            print(f"round {t_round}: accepted={len(outcome.accepted_client_indices)}/{n_clients}, "
                  f"V5={v5_msg}, next B_t={threshold.B_t:.3f}")

        history.append({
            "round": t_round,
            "aborted": False,
            "accepted": len(outcome.accepted_client_indices),
            "v5_passed": audit.v5_passed,
            "B_t": threshold.B_t,
        })

    # IMPORTANT: every call to local_train_delta (inside round_step_phase1,
    # once per client per round) reloads `model` with THAT client's own
    # locally-trained weights -- so by the time this loop finishes,
    # `model`'s in-memory state is whichever client trained LAST that
    # round, not the real aggregated global update. Sync it back to the
    # actual final w_t this function is returning, so a caller that
    # evaluates `model` directly (both notebooks' evaluation cells do
    # exactly this) gets the real federated result, not a stray
    # last-client overfit.
    model.load_state_dict(unflatten_to_state_dict(w_t, model_shapes))

    return w_t, history
