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
import time

import numpy as np

from crypto.secagg import (
    derive_pairwise_secret_via_x25519, new_self_mask_secret, pairwise_round_seed,
    ShareReleasePolicy, aggregate_masked,
)
from crypto.sketch import seed_commitment
from verifier.audit import independent_v5_check

from crypto.sketch import TorchSketchProjector

from .client import (
    ClientKeys, ClientRoundContext, round_step_phase1, round_step_phase2, quantize, center_mod_p_mask,
    share_self_mask_secret, sketch_phase1_batch,
)
from .server import run_round, apply_update, UnmaskingMaterial
from .vector_store import (
    RoundVectorStore, DEFAULT_BUDGET_BYTES, DEFAULT_SKETCH_BATCH_BYTES, suggested_batch_size,
)
from .norm_rule import bootstrap_from_server_root, update_threshold_ema
from .models import flatten_state_dict, unflatten_to_state_dict


def release_gpu_memory() -> None:
    """Free cached-but-idle GPU memory and run a GC pass.

    Call this between models/rounds in a long-lived notebook kernel --
    PyTorch's caching allocator holds onto freed tensors' memory for
    reuse rather than returning it to the driver, so running several
    models back-to-back in the same process (e.g. mobilenet_v2 then
    resnet50 in one smoke-test loop, or CASE1 then CASE2 in the same
    session) can accumulate enough cached-but-unused memory to starve
    a later, larger allocation even though nothing from the earlier
    model is actually still needed. `torch.cuda.empty_cache()` doesn't
    reduce PyTorch's memory *usage*, only returns what's unused back to
    the driver for other allocations (including this process's own
    next ones) to claim -- it does not fix memory truly still held by
    live references (e.g. a model or optimizer object still in scope).

    Safe to call with no GPU / no torch installed (no-op then); this
    is a cleanup hygiene helper, not a leak detector -- it won't help
    if the memory is still referenced somewhere (del the model/results
    dict entry first if that's the actual intent).
    """
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


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


def bootstrap_norms_from_model_dim(
    model_dim: int,
    sketch_dim: int,
    sketch_seed: bytes,
    quant_bits: int = 16,
    n_samples: int = 10,
    delta_scale: float = 5.0,
    rng: np.random.Generator = None,
    verbose: bool = True,
    sketch_batch_budget_bytes: int = DEFAULT_SKETCH_BATCH_BYTES,
) -> np.ndarray:
    """Round-0 B_t bootstrap, STAND-IN for ARCHITECTURE.md ROADMAP
    §3.4's real design ("server computes norms on its OWN root-set
    data") -- still a placeholder (no real root-set data exists
    here), but one that is at least dimensionally and scale-
    consistent with what real client sketches will actually produce,
    unlike the previous `N(5.0, 0.5)` magic-constant placeholder.

    That old placeholder caused a real, confusing failure caught live
    on Colab: once the two actual V3 wiring bugs were fixed (server.py
    checking the wrong commitment; client.py sketching the
    mask/mod-wrapped vector instead of the centered signed delta),
    every client STILL got rejected as "V3_norm_bound_exceeded" --
    this time because the hardcoded root_norms (~5) were simply the
    wrong order of magnitude for a quantized-and-sketched delta
    (which, at quant_bits=16, lands in the thousands-to-hundreds-of-
    thousands range even for small models), not because of a bug.

    This function closes that gap by running the SAME quantize ->
    center -> sketch pipeline fl/client.py's round_step_phase2 uses,
    over `n_samples` synthetic deltas drawn at scale `delta_scale`,
    so the resulting norms are in the right ballpark for
    `bootstrap_from_server_root` to derive a usable B_t from -- still
    not real root-set data, so still flagged in STATUS.md as a
    placeholder, just no longer a mis-scaled one.

    `delta_scale`'s default was RAISED from an initial 0.1 (the
    "typical gradient magnitude" `quantize()`'s own docstring cites,
    ~1e-2 to 1) to 5.0 after live Colab testing against the tiny
    debug model found the REAL achieved norm (~14.5M) was ~24x
    larger than what 0.1 predicted (~601K) -- plausibly because
    CASE1's MNIST pixels aren't normalized to [0,1] (raw 0-255
    values feeding a single linear layer produce much larger
    gradients than the 0.1 guess assumed). 5.0 gives a comfortable
    ~2x safety margin above that one observed data point, but it is
    still just an empirically-nudged guess, not a principled bound --
    if a different model/dataset/lr combination produces a real norm
    bigger than this bootstrap's B_t, round 0 will abort again with
    the same "V3_norm_bound_exceeded" message, now showing the actual
    numbers (see fl/server.py's verify_submission diagnostics) so the
    gap is immediately visible rather than requiring another guess.

    *** `n_samples` COST SCALES WITH `model_dim`, SAME AS EVERY OTHER
    SKETCH CALL -- caught live on Colab the first time this was run
    against a REAL model (mobilenet_v2, model_dim~2.24M) ***. Each
    sample calls `sketch.project()` over the full `model_dim`, the
    exact same O(m*d) cost `sketch.py`'s own PERFORMANCE NOTE already
    flags (~20s at d=500K on CPU). At the tiny debug model's d=7850
    this was negligible regardless of n_samples; at mobilenet_v2's
    real d~2.24M, one sample alone extrapolates to ~90s, so the
    original default (n_samples=50) meant calling this function once
    cost ~75 MINUTES, with nothing printed while it ran, before the
    round loop's own (comparatively modest) per-round sketch cost
    even started -- indistinguishable from a hang. Default lowered to
    10 (~15 min at mobilenet_v2 scale, still real but no longer
    absurd) -- callers working with a real, large model should pass
    an even smaller n_samples explicitly (3-5 is plenty for a rough
    median/MAD estimate; this is a placeholder bootstrap, not a
    precision statistic, and update_threshold_ema corrects B_t with
    REAL observed norms from round 1 onward regardless). Rule of
    thumb: budget n_samples * 20*(model_dim/500_000) seconds before
    calling this against anything bigger than a few hundred thousand
    parameters.
    """
    rng = rng or np.random.default_rng()
    sketch = TorchSketchProjector(seed=sketch_seed, m=sketch_dim, d=model_dim)
    norms = np.empty(n_samples)
    if verbose:
        # Rough estimate only, now using TorchSketchProjector (STATUS.md
        # TODO #3) instead of the NumPy reference this estimate was
        # originally calibrated against -- GPU timing is expected to beat
        # this estimate substantially; it's printed as an upper bound, not
        # a tight prediction, since actual speedup depends on the
        # available device (see TorchSketchProjector.device).
        est_per_sample = 20.0 * (model_dim / 500_000)
        print(f"bootstrap_norms_from_model_dim: {n_samples} samples at model_dim={model_dim}, "
              f"device={sketch.device} (NumPy-CPU upper-bound estimate ~{est_per_sample:.1f}s/sample, "
              f"~{est_per_sample * n_samples / 60:.1f} min total -- should be much faster on GPU)")
    # FIX.md F4/F5: every sample uses the SAME sketch_seed, so the sign
    # chunks are identical across samples -- generating them once per
    # batch instead of once per sample is the same saving F4 found in
    # phase 2, and this loop was costing 33 s/sample at ResNet-50
    # (330 s before round 0 even started). Batched in groups sized from
    # a byte budget, not all n_samples at once, so peak stays bounded
    # the same way phase 2's does.
    batch = suggested_batch_size(model_dim, sketch_batch_budget_bytes)
    done = 0
    while done < n_samples:
        group = min(batch, n_samples - done)
        t0 = time.time() if verbose else None
        stack = np.empty((group, model_dim), dtype=np.float32)
        for row in range(group):
            delta = rng.normal(0.0, delta_scale, size=model_dim)
            stack[row] = center_mod_p_mask(quantize(delta, bits=quant_bits))
            del delta
        for row, y in enumerate(sketch.project_batch(stack)):
            norms[done + row] = np.linalg.norm(y)
        del stack
        if verbose:
            print(f"  samples {done+1}-{done+group}/{n_samples} done "
                  f"({time.time()-t0:.1f}s for {group} in one pass)")
        done += group
    return norms


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
    lr: float = None,
    epochs: int = None,
    batch_size: int = None,
    momentum: float = None,
    verbose: bool = True,
    secagg_share_threshold: int = None,
    vector_budget_bytes: int = DEFAULT_BUDGET_BYTES,
    sketch_batch_budget_bytes: int = DEFAULT_SKETCH_BATCH_BYTES,
    spill_dir: str = None,
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
    This function now calls `crypto/sketch.py`'s `TorchSketchProjector`
    (STATUS.md TODO #3's GPU/torch reimplementation, done but WRITTEN,
    NOT YET COLAB-RUN as of 2026-11-11), not the NumPy
    `SketchProjector` reference (~16 min/client/round extrapolated to
    ResNet-50's real d~23.5M, which would make a full 45-round x
    50-client run take DAYS). Validate `TorchSketchProjector` on Colab
    (`tests/test_sketch_gpu.py`) before trusting this performance note
    is actually resolved -- it is expected to be dramatically faster,
    but "expected" is not the same as "confirmed", and this function
    is still for validating PROTOCOL WIRING with a small `rounds` /
    handful of clients first, not yet confirmed safe for the real
    MVP-scale sweep until that Colab run happens.

    *** ONE REMAINING HONEST PLACEHOLDER: `server_root_norms` ***
    The caller is expected to pass real norms computed from the
    server's own root-set data (ROADMAP.md §3.4: "server root set is
    used for B_t bootstrap at round 0"). Computing those properly means
    running several pseudo-client local-train + sketch passes over
    server-root subsamples BEFORE round 0 -- itself bounded by the same
    sketch performance wall above, so it is not done inside this
    function. If the caller passes synthetic/placeholder norms instead,
    that is a real, flagged simplification affecting only round 0's
    starting B_t (every subsequent round's B_t is the real EMA-updated
    value from real accepted submissions' real sketch norms) -- not a
    fabricated result for the round loop itself. Use
    `bootstrap_norms_from_model_dim` (above) rather than an arbitrary
    constant for this: a placeholder that isn't dimensionally/scale-
    consistent with the real quantize->sketch pipeline will size B_t
    (and therefore `n_bits`) wrong for round 0 regardless of how
    correct the rest of the wiring is -- this is exactly what caused
    "V3_norm_bound_exceeded" for every client even after the two real
    V3 bugs (wrong commitment in server.py; mask/mod-wrapped sketch
    input in client.py) were fixed.
    """
    n_clients = len(client_datasets)
    rounds = rounds if rounds is not None else CFG["mvp_scope"]["rounds_per_run"]
    # FIX.md F6: local-training hyperparameters now come from CFG's
    # local_training section by default (gold-standard FedAvg: multiple
    # local epochs, each a real mini-batch pass, not one full-batch step
    # repeated `epochs` times) -- explicit args here are an override,
    # e.g. for a smoke test or ablation, not the normal path.
    lt_cfg = CFG.get("local_training", {})
    lr = lr if lr is not None else lt_cfg.get("lr", 0.01)
    epochs = epochs if epochs is not None else lt_cfg.get("local_epochs", 5)
    batch_size = batch_size if batch_size is not None else lt_cfg.get("batch_size", 20)
    momentum = momentum if momentum is not None else lt_cfg.get("momentum", 0.0)
    sampling_rate = CFG["protocol"]["sampling_rate"]
    sigma_dp = CFG["dp"]["sigma"]
    sketch_m = CFG["sketch"]["m"]
    epoch_rounds = CFG["protocol"]["epoch_rounds"]
    k = CFG["norm_rule"]["k"]
    ema_alpha = CFG["norm_rule"]["ema_alpha"]

    # Self-mask secrets are t-of-n Shamir-shared among the clients. Default
    # t = floor(2n/3)+1 (Bonawitz et al.'s honest-majority-of-survivors
    # setting): fewer than t accepted clients means the round cannot be
    # unmasked and aborts, independently of the k_min privacy floor.
    share_t = secagg_share_threshold if secagg_share_threshold is not None else (2 * n_clients) // 3 + 1

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

        # Timing instrumentation added 2026-11-12: a real-model
        # (resnet50) round was taking tens of minutes with no visibility
        # into WHERE the time went (local training vs. the per-client EC
        # crypto -- ring signature, Shamir shares, Pedersen commitments,
        # and a range proof with ~n_bits Sigma-protocol components, all
        # real elliptic-curve operations, not numpy). Printing per-round
        # wall time for phase 1 (dominated by local training + most of
        # the crypto) and phase 2 (sketch + range proof) separately turns
        # "it's slow" into an actual number to act on -- e.g. whether the
        # full MVP sweep (45 rounds x 50 clients x 2 models) is hours or
        # days at this rate, and which half of the pipeline to optimize
        # first if it needs to be faster.
        _t_phase1_start = time.time()
        ctx_by_client = {}
        phase1_by_client = {}
        self_mask_shares_held = {}   # {k: {holder_id: ShamirShare}} -- held by the OTHER clients, not the server
        holder_ids = list(range(1, n_clients + 1))
        # FIX.md F5: q_k and x_k go here instead of staying on the
        # submissions, so peak RAM stops scaling with n_clients. Closed
        # in the `finally` below -- an aborted round must not leave
        # spill files behind across a 45-round sweep.
        store = RoundVectorStore(budget_bytes=vector_budget_bytes, dirpath=spill_dir)
        try:
            # FIX.md F3: NOT enabled here. A naive per-round cache of every
            # pair's expanded vector was measured to cost more memory than
            # it saves in time -- see FIX.md F3 for the numbers (2.5 GB at
            # 8 clients, 39 GB at 30, 110 GB at 50 -- directly worsening the
            # F5 crash this project already hit). pairwise_mask_cached and
            # its tests stay in the repo for a future pair-ordered redesign,
            # but round_loop.py must not pass a cache here until one exists.
            pairwise_mask_cache_this_round = None
            for client_idx in range(n_clients):
                ctx = ClientRoundContext(
                    keys=ClientKeys(sk=client_sks[client_idx], pk=ring[client_idx], index=client_idx),
                    ring=ring,
                    pairwise_secrets=pairwise_secrets[client_idx],
                    self_mask_secret_int=new_self_mask_secret(),
                    model_dim=model_dim,
                )
                ctx_by_client[client_idx] = ctx
                self_mask_shares_held[client_idx] = share_self_mask_secret(ctx, holder_ids, share_t)

                def _sink(q_k, masked, _k=client_idx):
                    store.put(("q", _k), q_k)
                    store.put(("x", _k), masked)

                phase1_by_client[client_idx] = round_step_phase1(
                    ctx, w_t, rho_t, rho_e, vrf_alpha, sampling_rate, sigma_dp,
                    n_clients, client_datasets[client_idx], model, lr, epochs, pk_trace,
                    batch_size=batch_size, momentum=momentum,
                    _sim_pairwise_mask_cache=pairwise_mask_cache_this_round,
                    vector_sink=_sink,
                )
            _t_phase1 = time.time() - _t_phase1_start

            # sigma_t "revealed" now that every c_k is committed -- reuse
            # each client's OWN ctx from phase 1 (not a fresh one), since
            # round_step_phase2 needs the SAME ctx.keys.sk for R_cons.
            _t_phase2_start = time.time()
            # FIX.md F4: sigma_t is the same for every client this round, so
            # TorchSketchProjector's (m, d) sign chunks are identical across
            # clients too. sketch_phase1_batch generates each chunk once and
            # applies it to every client's vector in the same pass, instead
            # of rebuilding the chunks once per client -- again a
            # simulation-only saving (a real client has only its own vector
            # to project).
            sketches_by_client = sketch_phase1_batch(
                range(n_clients), lambda k: store.get(("q", k)),
                sigma_t, sketch_m, model_dim,
                batch_size=suggested_batch_size(model_dim, sketch_batch_budget_bytes),
            )
            phase2_by_client = {}
            for client_idx in range(n_clients):
                phase2_by_client[client_idx] = round_step_phase2(
                    ctx_by_client[client_idx], phase1_by_client[client_idx],
                    sigma_t, sketch_m, threshold.B_t, rho_t, rho_e, vrf_alpha,
                    _sim_precomputed_sketch=sketches_by_client[client_idx],
                )
            _t_phase2 = time.time() - _t_phase2_start

            _t_verify_start = time.time()
            submissions = [(i, phase1_by_client[i]) for i in range(n_clients)]

            def unmask_provider(accepted_idx, excluded_idx, _held=self_mask_shares_held, _rho=rho_t):
                # Simulates the accepted (alive) clients' replies in the
                # unmasking round. One ShareReleasePolicy for the round
                # stands in for every honest client applying the same rule:
                # about any client, release self-mask shares OR pairwise
                # seeds, never both.
                policy = ShareReleasePolicy()
                shares = {}
                for k in accepted_idx:
                    policy.release_self_share(k)
                    shares[k] = [_held[k][h + 1] for h in accepted_idx]
                seeds = {}
                for k in excluded_idx:
                    policy.release_pairwise_seed(k)
                    for j in accepted_idx:
                        seeds[(j, k)] = pairwise_round_seed(pairwise_secrets[j][k], _rho)
                return UnmaskingMaterial(shares, seeds, share_t)

            outcome = run_round(submissions, phase2_by_client, rho_t, rho_e, vrf_alpha, threshold.B_t, n_clients, model_dim,
                                unmask_provider=unmask_provider,
                                masked_vector_provider=lambda k: store.get(("x", k)))
            _t_verify = time.time() - _t_verify_start

            if verbose:
                print(f"  round {t_round} timing: phase1(train+most-crypto)={_t_phase1:.1f}s, "
                      f"phase2(sketch+rangeproof)={_t_phase2:.1f}s, verify={_t_verify:.1f}s, "
                      f"total={_t_phase1 + _t_phase2 + _t_verify:.1f}s "
                      f"(n_clients={n_clients}, ~{(_t_phase1 + _t_phase2 + _t_verify) / n_clients:.2f}s/client)")

            if outcome.aborted:
                if verbose:
                    print(f"round {t_round}: ABORTED -- {outcome.abort_reason}")
                    for client_idx, reasons in outcome.rejected_reasons.items():
                        print(f"  client {client_idx} rejected: {reasons}")
                history.append({
                    "round": t_round, "aborted": True, "reason": outcome.abort_reason,
                    "rejected_reasons": outcome.rejected_reasons,
                })
                continue

            accepted_commitments = [phase1_by_client[i].commitment_point for i in outcome.accepted_client_indices]
            audit = independent_v5_check(accepted_commitments, outcome.published_sum_value, outcome.published_sum_blinding)

            # SIMULATION-ONLY self-check: this process happens to hold every
            # client's plaintext q_k, so it can confirm the unmasked aggregate
            # is exactly sum_{accepted} q_k. No party in the real protocol can
            # run this (nobody holds the q_k). It exists because the previous
            # version of this loop applied a never-unmasked aggregate for every
            # round while still printing V5=PASS.
            expected_aggregate = aggregate_masked(
                store.get(("q", i)) for i in outcome.accepted_client_indices
            )
            aggregate_exact = bool(np.array_equal(outcome.aggregate_quantized, expected_aggregate))
            del expected_aggregate
            if not aggregate_exact:
                if verbose:
                    print(f"round {t_round}: AGGREGATE MISMATCH -- unmasked aggregate != sum of accepted q_k; update NOT applied")
                history.append({"round": t_round, "aborted": True, "reason": "aggregate mismatch after unmasking"})
                continue

            w_t = apply_update(w_t, outcome.aggregated_delta)

            next_round_norms = np.array([
                np.linalg.norm(phase2_by_client[i].sketch) for i in outcome.accepted_client_indices
            ])
            threshold = update_threshold_ema(threshold, next_round_norms, k=k, alpha=ema_alpha)

            if verbose:
                v5_msg = "PASS" if audit.v5_passed else f"FAIL: {audit.detail}"
                print(f"round {t_round}: accepted={len(outcome.accepted_client_indices)}/{n_clients}, "
                      f"V5={v5_msg}, aggregate=EXACT, next B_t={threshold.B_t:.3f}")

            history.append({
                "round": t_round,
                "aborted": False,
                "accepted": len(outcome.accepted_client_indices),
                "v5_passed": audit.v5_passed,
                "aggregate_exact": aggregate_exact,
                "B_t": threshold.B_t,
            })
        finally:
            # Releases this round's resident arrays and deletes any
            # spill files -- on the normal path and on every `continue`
            # above (aborted round, aggregate mismatch) alike.
            _st = store.stats()
            if verbose and _st["spills"]:
                print(f"  round {t_round} vectors: {_st['resident_arrays']} held in RAM "
                      f"(peak {_st['peak_resident_bytes']/2**30:.2f} GiB), "
                      f"{_st['spilled_arrays']} spilled to disk "
                      f"({_st['bytes_written']/2**30:.2f} GiB written, "
                      f"{_st['bytes_read']/2**30:.2f} GiB read back)")
            store.close()

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
