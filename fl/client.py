"""Client-side round protocol (ARCHITECTURE.md §4, steps 0-10).
TORCH-DEPENDENT for LocalTrain (steps 1); every crypto step (2-10)
only needs the primitives already built and tested in crypto/, wired
together here. Not run in this sandbox (needs torch for training AND
pynacl for the EC-dependent crypto calls) -- validate end-to-end on
Colab per STATUS.md.

This module intentionally does NOT re-implement any crypto primitive;
it only calls crypto/* in the order ARCHITECTURE.md's round protocol
specifies, so correctness of the primitives (already unit-tested)
carries over, and the only new risk surface is ORDERING/WIRING, not
cryptography.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from crypto.ristretto import Scalar, Point, scalar_mul_base
from crypto.vrf import evaluate as vrf_evaluate, is_sampled as vrf_is_sampled
from crypto.dp import split_noise
from crypto.secagg import pairwise_mask_cached, self_mask, self_secret_bytes, shamir_share_self_secret, P_MASK
from crypto.pedersen import commit as pedersen_commit
from crypto.elgamal_threshold import encrypt as elgamal_encrypt
from crypto.clsag import sign as clsag_sign, h_p as clsag_h_p
from crypto.sketch import TorchSketchProjector  # GPU/torch (STATUS.md TODO #3); SketchProjector (NumPy) is the correctness reference used by tests, not the real training path
from crypto.rangeproof import prove_range, recompose_commitment
from crypto.r_cons import prove_identity_binding
from crypto.vector_commit import weighted_digest
from crypto.norm_proof import (
    commit_sketch, subset_sum_challenge, wraparound_window, prove_wraparound_binding,
    mask_blinding_vector,
)
from crypto.secagg import pairwise_round_seed


@dataclass
class ClientKeys:
    sk: Scalar
    pk: Point
    index: int   # position in the published ring R = {pk_1..pk_50}


@dataclass
class ClientRoundContext:
    keys: ClientKeys
    ring: list   # list[Point], the full published ring (len 50)
    pairwise_secrets: dict   # {other_client_index: bytes}, from X25519 ECDH (see secagg.py)
    self_mask_secret_int: int   # in Z_L (crypto.secagg.new_self_mask_secret), so its Shamir shares reconstruct it exactly
    blinding_self_mask_secret_int: int  # FIX.md F7e dropout resilience: a SEPARATE Z_L secret, same reason -- never reuse self_mask_secret_int here
    model_dim: int
    quant_bits: int = 16


@dataclass
class ClientSubmissionPhase1:
    """What the client sends BEFORE the sketch seed sigma_t is
    revealed (steps 0-7) -- committed values only, no sketch/range
    proof/R_cons yet, matching the protocol's explicit ordering
    constraint: "S publishes sigma_t ... AFTER all c_k are on BB".
    """
    commitment_point: Point          # c_k
    escrow_ciphertext: object        # tau_k (ElGamalCiphertext)
    key_image_round: Point           # I_k^t
    key_image_epoch: Point           # P_k^e
    ring_signature: object           # sigma_k (RingSignature)
    masked_vector_hash: bytes = b""  # FIX.md F7a: sha256 of `masked`, bound into the signed message
    was_sampled: bool = False        # b_k (kept client-side; not revealed directly)
    _secret_q: np.ndarray = field(repr=False, default=None)   # q_k, kept until phase 2
    _secret_blinding: Scalar = field(repr=False, default=None)  # r_k, revealed only after acceptance
    _secret_masked_vector: np.ndarray = field(repr=False, default=None)  # q_k + m_k


@dataclass
class ClientSubmissionPhase2:
    """FIX.md F7 (V3 redesign): the sketch `y_k` itself is NEVER on this
    submission in the clear any more -- `sketch_commitment` is a
    `crypto.norm_proof.SketchCommitment` (per-coordinate Pedersen
    commitments + per-coordinate square proofs + the one PUBLIC scalar
    v = <y_k,y_k>). `wraparound_bindings`/`wraparound_range_proofs` are
    F7d's cheap probabilistic wraparound-catching test, also hidden
    (crypto.norm_proof.prove_wraparound_binding + rangeproof.py reused
    on the small, FEW shifted test values -- NOT on the m=8192-dim
    sketch, which is what keeps this cheap).
    """
    sketch_commitment: object        # crypto.norm_proof.SketchCommitment
    wraparound_bindings: list        # list[SchnorrProof], one per subset-sum test; ties
                                      # recompose_commitment(wraparound_range_proofs[j]) to
                                      # sketch_commitment.y_commitments under that test's weights
    wraparound_range_proofs: list    # list[RangeProof] -- NO aggregate_blinding on the submission,
                                      # see round_step_phase2's comment on why revealing it would leak
    identity_binding_proof: object   # R_cons AND-proof
    masked_y_blindings: list = field(default=None)   # FIX.md F7e: rho_{k,i} = r_{k,i} + pairwise masks (mod L).
                                                     # PUBLIC and safe to publish; the raw r_{k,i} never leave the
                                                     # client. See crypto/norm_proof.py's F7e section for why the
                                                     # masking is necessary rather than revealing r_{k,i} directly.


def quantize(delta: np.ndarray, bits: int = 16) -> np.ndarray:
    """Fixed-point quantization into Z_{P_MASK} (ARCHITECTURE.md step
    3: "Quantize(Delta_k + nu_k) in Z_p^d, fixed-point 16-bit
    mantissa"). Scale chosen so typical gradient magnitudes (~1e-2 to
    1) map to a usable range of the mantissa before reducing mod
    P_MASK -- this scale is a training hyperparameter tuned once real
    models exist (fl/client.py's caller passes it in), not re-derived
    per round.
    """
    scale = 2 ** bits
    scaled = delta * scale

    # Caught live on Colab (resnet50, CASE2/FEMNIST real-model smoke
    # test, round 1, 2026-11-12) -- round 0 was clean, so this is NOT
    # the same failure as local_train_delta's own NaN/Inf check (which
    # didn't fire here): a value can be entirely FINITE yet still
    # overflow int64's representable range (~+-9.22e18) once multiplied
    # by `scale` -- a diverging-but-not-yet-infinite float32 gradient
    # step routinely produces finite-but-huge numbers (e.g. ~1e14+)
    # before/instead of ever reaching actual inf. `np.round(...)
    # .astype(np.int64)` on such a value is UNDEFINED, not an error --
    # only a RuntimeWarning -- so this is the second, distinct way
    # garbage silently reached the crypto layer that local_train_delta's
    # isfinite-only check could not catch (it checked `delta` alone,
    # before scaling, and finite-but-large passes that check easily).
    # Checking HERE, at the actual cast, covers every caller
    # (local_train_delta's output + DP noise, bootstrap synthetic
    # deltas, any future caller) rather than trying to anticipate every
    # upstream source of an out-of-range value.
    if not np.all(np.isfinite(scaled)):
        n_bad = int(np.sum(~np.isfinite(scaled)))
        raise FloatingPointError(
            f"quantize: {n_bad}/{scaled.size} values are non-finite "
            f"(NaN/Inf) after scaling by 2**{bits} -- cannot safely "
            f"cast to int64."
        )
    # int64's true range is roughly +-9.223e18; stay a bit under that
    # so np.round()'s own rounding can't itself tip something right at
    # the edge over the line.
    int64_safe_max = 9.0e18
    max_abs = float(np.max(np.abs(scaled))) if scaled.size else 0.0
    if max_abs > int64_safe_max:
        raise FloatingPointError(
            f"quantize: scaled value magnitude {max_abs:.3e} exceeds "
            f"the safe int64 cast range ({int64_safe_max:.1e}) -- "
            f"np.round(...).astype(np.int64) would silently overflow "
            f"rather than error. This means the pre-scale delta had an "
            f"entry around {max_abs / scale:.3e} or larger -- almost "
            f"certainly a diverging training step (finite but huge, "
            f"not NaN/Inf, which is why an isfinite-only check upstream "
            f"would miss it), not a crypto bug. Lower lr, add gradient "
            f"clipping, or mini-batch instead of one full-batch step."
        )
    quantized = np.round(scaled).astype(np.int64)
    return np.mod(quantized, P_MASK)


def center_mod_p_mask(q: np.ndarray) -> np.ndarray:
    """Un-wrap a Z_{P_MASK}-reduced integer array back to the signed
    representation it had before `np.mod(_, P_MASK)` was applied --
    e.g. `quantize()`'s own output, or `_secret_q`. Needed anywhere a
    quantized-but-not-yet-masked value's TRUE magnitude matters (norm
    bounding, dequantization): `np.mod` wraps every negative entry to
    a value near P_MASK, so a naive norm over the wrapped array is
    dominated by those near-P_MASK entries regardless of how small
    the real values are -- this undoes exactly that wrap.
    """
    half = P_MASK // 2
    return np.where(q > half, q - P_MASK, q)


def dequantize(q: np.ndarray, bits: int = 16) -> np.ndarray:
    scale = 2 ** bits
    signed = center_mod_p_mask(q)
    return signed.astype(np.float64) / scale


def local_train_delta(model, w_t, local_data, lr: float, epochs: int,
                       batch_size: int, momentum: float = 0.0, seed: int = None):
    """Step 1: Delta_k = LocalTrain(w_t, D_k) - w_t. TORCH-DEPENDENT --
    the actual SGD loop.

    FIX.md F6: this is now real FedAvg-style local training (McMahan et
    al. 2017, "Communication-Efficient Learning..." §3): `epochs` full
    passes over the client's shard, each pass freshly shuffled and cut
    into mini-batches of `batch_size`, one SGD step per mini-batch.

    Before this fix, `epochs` full-batch gradient steps were taken over
    the WHOLE client shard every "epoch" -- identical full gradient each
    time (no stochasticity: SGD's "S" was never real), and no notion of
    an epoch as a pass over shuffled batches. That was flagged live (see
    the comment this replaced) as the likely cause of the training-
    divergence FloatingPointError below, and is a distinct, ML-validity
    bug from the crypto-correctness bugs F1-F5/F8 already fixed --
    nothing here changes the protocol's cryptographic guarantees, only
    whether local training does what the paper's FedAvg baseline claims
    it does.

    The last mini-batch of a pass is smaller than `batch_size` whenever
    the shard size doesn't divide evenly (standard, not a bug: dropping
    it would silently throw away data, especially costly for FEMNIST's
    already-small per-writer shards). `batch_size >= len(local_data)`
    degenerates to exactly one full-batch step per epoch -- not special-
    cased, since that is simply what mini-batch SGD does when the batch
    covers the whole shard.
    """
    from .models import flatten_state_dict, unflatten_to_state_dict
    import torch

    flat_w_t, shapes = flatten_state_dict(model)
    state_dict = unflatten_to_state_dict(w_t, shapes)
    model.load_state_dict(state_dict)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    criterion = torch.nn.CrossEntropyLoss()

    images, labels = local_data.images, local_data.labels
    n = images.shape[0]
    if n == 0:
        raise ValueError("local_train_delta: client has zero local samples -- cannot train")
    if batch_size <= 0:
        raise ValueError(f"local_train_delta: batch_size must be positive, got {batch_size}")

    last_loss = None
    n_batches_total = 0
    for epoch_idx in range(epochs):
        # Reshuffled every epoch -- the other half of "real epochs": a
        # pass over the SAME batches every time is not meaningfully
        # different from one big batch, it is just split into pieces in
        # a fixed order. `seed` is optional and only for reproducible
        # ablations; real rounds leave it unset and use torch's global RNG.
        if seed is not None:
            g = torch.Generator().manual_seed(seed * 1_000_003 + epoch_idx)
            perm = torch.randperm(n, generator=g)
        else:
            perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            batch_idx = perm[start:start + batch_size]
            batch_images = images[batch_idx]
            batch_labels = labels[batch_idx]
            optimizer.zero_grad()
            outputs = model(batch_images)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu())
            n_batches_total += 1

    new_flat, _ = flatten_state_dict(model)
    delta = (new_flat - w_t).detach().cpu().numpy().astype(np.float64)

    # Caught live on Colab (mobilenet_v2, real-model smoke test,
    # 2026-11-12): without this check, a non-finite delta silently
    # reaches fl/client.py's quantize() (`np.round(delta * scale)
    # .astype(np.int64)`), which NumPy casts to an UNDEFINED int64
    # value for NaN/Inf (just a RuntimeWarning, not an error) -- the
    # crypto layer then happily commits to, masks, sketches, and
    # verifies that garbage, so a round can print accepted=8/8,
    # V5=PASS while actually aggregating a corrupted update for that
    # client. This is a training-stability failure (lr/momentum too
    # high for this model/data/batch-size combination), not a crypto
    # bug, but it must not be allowed to pass through silently -- raise
    # loudly instead.
    n_bad = int(np.sum(~np.isfinite(delta)))
    if n_bad:
        raise FloatingPointError(
            f"local_train_delta: {n_bad}/{delta.size} elements of the "
            f"trained delta are non-finite (NaN/Inf) after {epochs} "
            f"epoch(s) x {n_batches_total} total mini-batches "
            f"(batch_size={batch_size}) at lr={lr}, momentum={momentum} "
            f"(final loss={last_loss}) -- training diverged. Lower lr or "
            f"momentum, or reduce batch_size further, then retry. (Caught "
            f"here deliberately: letting this reach quantize() would "
            f"silently cast NaN/Inf to an undefined int64 value and "
            f"corrupt this client's contribution without any visible error.)"
        )
    return delta


def share_self_mask_secret(ctx: ClientRoundContext, holder_ids: list, t: int) -> dict:
    """Before masking: Shamir-share this round's self-mask secret among
    the other clients ({holder_id: ShamirShare}, holder ids 1-indexed).
    The holders release these to the server only if this client's
    submission ends up in the accepted set (ShareReleasePolicy) -- that
    is how the server removes the self-mask from the aggregate, and why
    it cannot remove it from a vector it merely claims was dropped.
    """
    return shamir_share_self_secret(ctx.self_mask_secret_int, t, holder_ids)


def share_blinding_self_mask_secret(ctx: ClientRoundContext, holder_ids: list, t: int) -> dict:
    """Same as `share_self_mask_secret`, for the SEPARATE blinding
    self-mask secret (FIX.md F7e dropout resilience). Shared at the
    same point in the round (before any masking happens), so the
    holders can release shares of it later regardless of whether this
    client is still reachable."""
    return shamir_share_self_secret(ctx.blinding_self_mask_secret_int, t, holder_ids)


def round_step_phase1(
    ctx: ClientRoundContext,
    w_t,
    rho_t: bytes,
    rho_e: bytes,
    vrf_alpha: bytes,
    sampling_rate: float,
    sigma_dp: float,
    n_clients: int,
    local_data,
    model,
    lr: float,
    epochs: int,
    pk_trace: Point,
    batch_size: int = 20,
    momentum: float = 0.0,
    _sim_pairwise_mask_cache: dict | None = None,
    vector_sink=None,
) -> ClientSubmissionPhase1:
    """Steps 0-7 of ARCHITECTURE.md §4's round protocol.

    `batch_size`, `momentum`: FIX.md F6, passed straight through to
    `local_train_delta`'s real mini-batch/multi-epoch SGD. Defaults here
    only cover a client called in isolation (e.g. a test); real rounds
    get these from `fl/round_loop.py::run_training_loop`'s `CFG["local_training"]`.

    `_sim_pairwise_mask_cache`: simulation-only (FIX.md F3), see
    `crypto.secagg.pairwise_mask_cached`. Leave as `None` for the real
    per-client cost; `fl/round_loop.py` passes a dict it owns for one
    round to avoid expanding each pair's mask twice in this
    single-process simulation.

    `vector_sink`: FIX.md F5. When given, it is called as
    `vector_sink(q_k, masked)` and the returned submission carries
    `_secret_q = _secret_masked_vector = None`, so n submissions do not
    pin 2*n*model_dim*8 bytes of RAM for the whole round. When `None`,
    both arrays stay on the submission exactly as before -- that is
    still the right shape for a single client in isolation, and for
    tests.
    """
    # Step 0: VRF self-sampling.
    vrf_out = vrf_evaluate(ctx.keys.sk, ctx.keys.pk, vrf_alpha)
    b_k = vrf_is_sampled(vrf_out.beta, sampling_rate)

    # Step 1: train or contribute zero.
    if b_k:
        try:
            delta = local_train_delta(model, w_t, local_data, lr, epochs, batch_size, momentum=momentum)
        except FloatingPointError as e:
            raise FloatingPointError(f"client {ctx.keys.index}: {e}") from e
    else:
        delta = np.zeros(ctx.model_dim, dtype=np.float64)

    # Step 2: DP noise share.
    noise_shares = split_noise(sigma_dp, n_shares=1, dim=ctx.model_dim, rng=np.random.default_rng())
    nu_k = noise_shares[0]

    # Step 3: quantize.
    try:
        q_k = quantize(delta + nu_k, bits=ctx.quant_bits)
    except FloatingPointError as e:
        raise FloatingPointError(f"client {ctx.keys.index}: {e}") from e

    # Step 4: double-masking.
    masked = q_k.copy()
    for other_idx, secret in ctx.pairwise_secrets.items():
        sign = 1 if ctx.keys.index < other_idx else -1
        pair_key = (min(ctx.keys.index, other_idx), max(ctx.keys.index, other_idx))
        masked = (masked + pairwise_mask_cached(
            secret, rho_t, ctx.model_dim, sign, _sim_pairwise_mask_cache, pair_key,
        )) % P_MASK
    masked = (masked + self_mask(self_secret_bytes(ctx.self_mask_secret_int), rho_t, ctx.model_dim)) % P_MASK

    # Step 5: Pedersen-commit the masked vector, via a single scalar
    # DIGEST rather than one commitment per coordinate (d can be tens
    # of millions of int64 entries) -- see crypto/vector_commit.py for
    # why a plain coordinate-sum digest is unsound and how the
    # randomized-weight fix restores a real binding property (sound
    # under the Schwartz-Zippel argument, and still homomorphic the
    # same way a plain sum would be, which is what fl/server.py's V5
    # check needs). Weights are derived from rho_t, which is already
    # public to every client at the START of this round -- before the
    # server has aggregated anything -- so this is sound against the
    # actual V5 threat model (a server misreporting the aggregate),
    # not merely a notation change from the old placeholder.
    masked_digest = Scalar.from_int(weighted_digest(masked, seed=rho_t))
    blinding = Scalar.random()
    commitment = pedersen_commit(masked_digest, blinding)

    # Step 6: identity escrow.
    plaintext_point = ctx.keys.pk  # escrow the client's own identity point
    escrow_ct = elgamal_encrypt(plaintext_point, pk_trace)

    # Step 7: dual-scope ring signature.
    #
    # FIX.md F7a: the signed message now also binds a hash of the
    # MASKED vector itself (`masked`), not just c_k and the escrow
    # ciphertext. Before this, nothing in the signature tied c_k to
    # any particular masked vector -- a client (or a server replaying
    # a captured submission) could pair a valid c_k/signature with a
    # DIFFERENT masked vector than the one c_k's digest was actually
    # computed over, since c_k is a commitment to a scalar digest, not
    # to the vector itself. Hashing `masked` into the signed message
    # closes that: V1 (clsag verification, unchanged) now also attests
    # "this specific masked vector belongs with this c_k and this
    # signature", which is the premise F7e's aggregate check relies on
    # (it only makes sense to compare "the vector this client actually
    # submitted" against a sketch of that same vector).
    masked_hash = hashlib.sha256(b"janus-f7a-masked:" + masked.tobytes()).digest()
    base_t = clsag_h_p(rho_t)
    base_e = clsag_h_p(rho_e)
    message = commitment.point.b + escrow_ct.c1.b + escrow_ct.c2.b + masked_hash
    sig = clsag_sign(ctx.keys.sk, ctx.keys.index, ctx.ring, message, rho_t, rho_e)

    # FIX.md F5: hand the two full-length arrays to the round's store
    # and keep them off the submission, so holding n submissions costs
    # O(n) small objects instead of O(n * model_dim) int64 entries.
    if vector_sink is not None:
        vector_sink(q_k, masked)
        q_k = None
        masked = None

    return ClientSubmissionPhase1(
        commitment_point=commitment.point,
        escrow_ciphertext=escrow_ct,
        key_image_round=sig.key_image_round,
        key_image_epoch=sig.key_image_epoch,
        ring_signature=sig,
        masked_vector_hash=masked_hash,
        was_sampled=b_k,
        _secret_q=q_k,
        _secret_blinding=blinding,
        _secret_masked_vector=masked,
    )


def sketch_phase1_batch(
    client_indices, q_provider, sketch_seed: bytes, sketch_dim: int, model_dim: int,
    batch_size: int = 8,
) -> dict:
    """SUPERSEDED by FIX.md F7: this batched the FLOAT sketch
    (`SketchProjector.project_batch`) across clients for the old
    cleartext-norm V3 check. `round_step_phase2` no longer calls
    `.project()` at all -- it calls `.project_int()` and commits the
    result coordinate by coordinate (`crypto.norm_proof.commit_sketch`),
    and takes no precomputed-sketch argument any more, so this
    function's output has nowhere to plug in. Left in place (not
    deleted) only because `crypto.sketch.SketchProjector.project_batch`
    /`TorchSketchProjector.project_batch` and the batching PATTERN
    (regenerate each (m,d) sign chunk once, apply across a small group
    of clients) are still real and still correct for the FLOAT
    estimator path those methods serve -- an analogous batched
    `project_int_batch` already exists in crypto/sketch.py for a future
    pass that wires batching into the committed-sketch path (that
    wiring is NOT done here: see FIX.md F7, "wiring"). Do not call this
    from `fl/round_loop.py`'s phase-2 step any more.

    Original docstring, for the float path this still serves:
    sigma_t (`sketch_seed`) is the same for every client this round, so
    `TorchSketchProjector`'s (m, d) sign chunks are identical across
    clients -- only the vector being projected differs. Batched in
    groups of `batch_size` (FIX.md F5), not across all n clients at
    once, to bound memory; `fl/vector_store.py::suggested_batch_size`
    picks it from a byte budget so it does not grow with client count.
    """
    sketch = TorchSketchProjector(seed=sketch_seed, m=sketch_dim, d=model_dim)
    indices = list(client_indices)
    batch_size = max(1, int(batch_size))
    out = {}
    for start in range(0, len(indices), batch_size):
        group = indices[start : start + batch_size]
        stack = np.empty((len(group), model_dim), dtype=np.float32)
        for row, client_index in enumerate(group):
            q = q_provider(client_index)
            # center_mod_p_mask returns a new array; assigning into the
            # preallocated row casts to float32 without building a
            # second full-length float64 copy that outlives the loop.
            stack[row] = center_mod_p_mask(q)
            del q
        for client_index, y_k in zip(group, sketch.project_batch(stack)):
            out[client_index] = y_k
        del stack
    return out


def round_step_phase2(
    ctx: ClientRoundContext,
    phase1: ClientSubmissionPhase1,
    sketch_seed: bytes,
    sketch_dim: int,
    B_t: float,
    rho_t: bytes,
    rho_e: bytes,
    vrf_alpha: bytes,
    n_tests: int = 8,
    _secret_q_override: np.ndarray | None = None,
) -> ClientSubmissionPhase2:
    """Steps 8-10, run AFTER the server reveals sigma_t.

    `_secret_q_override`: FIX.md F5 handed `q_k` to a `vector_sink` in
    phase 1 (`phase1._secret_q is None` on this submission) so n
    submissions do not pin O(n*model_dim) RAM; the caller (e.g.
    `fl/round_loop.py`, which stashed it in its `VectorStore`) passes
    it back in here instead of this function reading `phase1._secret_q`
    directly. Leave as `None` when phase 1 was NOT called with a
    `vector_sink` (a single client in isolation, or a test).

    Sketches the SIGNED, centered form of `phase1._secret_q` (the
    quantized, PRE-masking delta -- ARCHITECTURE.md line 289's
    "correctly quantized vector" that `R_norm` bounds), not
    `_secret_masked_vector`. Two separate reasons the masked vector
    is wrong here:

    1. `_secret_masked_vector` is `q_k + m_k mod P_MASK`: the mask
       `m_k` is drawn to look uniform over Z_{P_MASK} (so individual
       masked values reveal nothing -- only the SUM across all
       clients cancels it). Its norm is dominated by the mask term
       (~sqrt(d)*P_MASK/2) regardless of how small or large the real
       update is, which would make V3 incapable of distinguishing an
       honest update from a model-replacement attack (A1, line 499)
       -- both get swamped by the same mask-scale noise.
    2. `_secret_q` ALONE isn't enough either: `quantize()` already
       reduces into Z_{P_MASK} via `np.mod`, so any coordinate with a
       negative delta wraps to a value near P_MASK -- same
       representation problem as (1), just without the extra mask on
       top. `center_mod_p_mask` undoes exactly that wrap (the same
       operation `dequantize` already needed), giving back the true
       signed, small-magnitude quantized value the norm bound is
       actually meant to see.

    Together these explain a real symptom hit live on Colab: every
    client, every round, rejected as "V3_norm_bound_exceeded"
    regardless of model, because whatever got sketched was always
    far outside any B_t sized for real gradient magnitudes. This also
    makes `update_threshold_ema`'s round-to-round B_t adaptation
    (which assumes norms shrink as training converges) track a
    quantity that actually does shrink, instead of P_MASK-scale
    wrap-noise that wouldn't.

    `_sim_precomputed_sketch`: simulation-only (FIX.md F4), see
    `sketch_phase1_batch`. Leave as `None` for the real per-client
    cost -- this builds and uses its own `TorchSketchProjector` exactly
    as before.
    """
    secret_q = _secret_q_override if _secret_q_override is not None else phase1._secret_q
    if secret_q is None:
        raise ValueError(
            "phase 1 handed q_k to a vector_sink (FIX.md F5), so this "
            "submission carries no q_k on it -- pass it back in as "
            "_secret_q_override (e.g. from the same VectorStore the sink "
            "wrote to), or call round_step_phase1 without a vector_sink"
        )
    signed_q = center_mod_p_mask(secret_q).astype(np.int64)
    sketch = TorchSketchProjector(seed=sketch_seed, m=sketch_dim, d=ctx.model_dim)
    y_k = sketch.project_int(signed_q)  # exact integers (FIX.md F7b), not the float estimate

    transcript = b"norm_proof:" + phase1.commitment_point.b
    sketch_commitment, y_blindings, sq_blindings = commit_sketch(list(int(v) for v in y_k), transcript)

    density = sketch.density

    # `n_tests` must equal what the server passes to check_V3_norm_bound
    # (fl/server.py threads it through run_round/verify_submission for
    # exactly this reason): the two sides derive the SAME public weight
    # vectors from it, so a mismatch fails every honest client.
    weights_list = subset_sum_challenge(seed=sketch_seed, m=sketch_dim, n_tests=n_tests)
    # PUBLIC window -- derived only from (sketch_dim, model_dim, density,
    # quant_bits), never from this client's own data, so the server can
    # recompute the identical value. See wraparound_window's docstring
    # for why it must be quant_bits-derived, not P_MASK-derived.
    window = wraparound_window(sketch_dim, ctx.model_dim, density, ctx.quant_bits)
    n_bits = max(8, int(np.ceil(np.log2(max(2 * window + 2, 2)))))
    wraparound_bindings = []
    wraparound_range_proofs = []
    y_list = list(int(v) for v in y_k)
    for j, w in enumerate(weights_list):
        tx = transcript + b":wrap:" + j.to_bytes(4, "little")
        s = sum(wi * yi for wi, yi in zip(w, y_list))
        shifted = s + window
        if shifted < 0 or shifted >= (1 << n_bits):
            # Outside what prove_range can attest to -- clamp so phase2
            # still builds; check_V3 on the server re-derives n_bits/window
            # itself and rejects a submission whose real value needed clamping,
            # mirroring the old norm-estimate clamp's rejection rule.
            shifted = min(max(shifted, 0), (1 << n_bits) - 1)
        range_proof, agg_blinding = prove_range(shifted, n_bits)
        # agg_blinding is NEVER put on the submission: shifted is small
        # enough to be brute-force-recoverable from a revealed blinding
        # (the same oracle-attack shape F7e's reveal has to avoid -- see
        # that section's comment), so it stays client-side, used only to
        # build the Schnorr binding proof below. `recompose_commitment`
        # is a public function of `range_proof.bit_proofs` alone --
        # exactly the commitment agg_blinding opens -- so the server
        # recomputes the SAME point from the published range proof
        # without ever being told agg_blinding.
        bind_proof = prove_wraparound_binding(
            y_list, y_blindings, w, window, tx,
            recompose_commitment(range_proof), agg_blinding,
        )
        wraparound_bindings.append(bind_proof)
        wraparound_range_proofs.append(range_proof)

    identity_proof = prove_identity_binding(ctx.keys.sk, rho_t, rho_e, vrf_alpha, b"r_cons:" + phase1.commitment_point.b)

    # FIX.md F7e: publish the PAIRWISE-MASKED blinding vector, not the
    # blindings themselves. Derived from the same per-round pair seeds
    # secagg's vector path uses (`pairwise_round_seed(secret, rho_t)`),
    # so the excluded-pair seeds accepted clients already release for
    # `unmask_aggregate` strip these masks too. Published in phase 2,
    # i.e. before the accepted set is known -- same ordering as the
    # masked update vector, and the reason the server cannot choose a
    # convenient subset after the fact.
    pairwise_seeds = {
        other_idx: pairwise_round_seed(secret, rho_t)
        for other_idx, secret in ctx.pairwise_secrets.items()
    }
    masked_y_blindings = mask_blinding_vector(
        y_blindings, ctx.keys.index, pairwise_seeds,
        self_secret_bytes(ctx.blinding_self_mask_secret_int), rho_t,
    )

    return ClientSubmissionPhase2(
        sketch_commitment=sketch_commitment,
        wraparound_bindings=wraparound_bindings,
        wraparound_range_proofs=wraparound_range_proofs,
        identity_binding_proof=identity_proof,
        masked_y_blindings=masked_y_blindings,
    )


def release_straddling_pair_seeds(own_index: int, group: list, pairwise_secrets: dict, rho_t: bytes) -> dict:
    """F7f's per-client half. When the server bisects the accepted set
    to localize a failed F7e check, the sum over a subgroup G only
    cancels for pairs with BOTH parties inside G -- so each client in G
    releases its per-round pair seeds about the accepted clients
    OUTSIDE G, exactly as it would release seeds about an excluded
    client for `unmask_aggregate`.

    This is the accountability-anonymity budget being SPENT, not a free
    check: every bisection step hands the server one more subgroup sum,
    which is why `norm_proof.localize_fault` stops at `min_group_size`
    instead of narrowing to a single client. A client is entitled to
    refuse (the round then fails the F7e check without attribution,
    the pre-F7f behaviour) -- the policy question of what refusal costs
    is deliberately not decided here.

    Returns {(own_index, other): round_seed} for every accepted client
    `other` not in `group`.
    """
    return {
        (own_index, other): pairwise_round_seed(secret, rho_t)
        for other, secret in pairwise_secrets.items()
        if other not in set(group)
    }


def reveal_blinding_after_acceptance(phase1: ClientSubmissionPhase1) -> Scalar:
    """Step after V1-V4 acceptance: client sends r_k so the server can
    publish the aggregate opening for V5. Only called once the server
    has posted an accept verdict for this client's I_k^t -- revealing
    r_k before that would let a server retroactively decide whether to
    "accept" based on seeing the opening (ARCHITECTURE.md's explicit
    note: "blinding factor, after acceptance only").
    """
    return phase1._secret_blinding
