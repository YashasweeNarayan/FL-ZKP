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

from dataclasses import dataclass, field

import numpy as np

from crypto.ristretto import Scalar, Point, scalar_mul_base
from crypto.vrf import evaluate as vrf_evaluate, is_sampled as vrf_is_sampled
from crypto.dp import split_noise
from crypto.secagg import pairwise_mask, self_mask, P_MASK
from crypto.pedersen import commit as pedersen_commit
from crypto.elgamal_threshold import encrypt as elgamal_encrypt
from crypto.clsag import sign as clsag_sign, h_p as clsag_h_p
from crypto.sketch import TorchSketchProjector  # GPU/torch (STATUS.md TODO #3); SketchProjector (NumPy) is the correctness reference used by tests, not the real training path
from crypto.rangeproof import prove_range
from crypto.r_cons import prove_identity_binding
from crypto.vector_commit import weighted_digest


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
    self_mask_secret_int: int
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
    was_sampled: bool                # b_k (kept client-side; not revealed directly)
    _secret_q: np.ndarray = field(repr=False, default=None)   # q_k, kept until phase 2
    _secret_blinding: Scalar = field(repr=False, default=None)  # r_k, revealed only after acceptance
    _secret_masked_vector: np.ndarray = field(repr=False, default=None)  # q_k + m_k


@dataclass
class ClientSubmissionPhase2:
    sketch: np.ndarray               # y_k
    range_proof: object
    range_proof_aggregate_blinding: Scalar
    identity_binding_proof: object   # R_cons AND-proof


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
    quantized = np.round(delta * scale).astype(np.int64)
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


def local_train_delta(model, w_t, local_data, lr: float, epochs: int):
    """Step 1: Delta_k = LocalTrain(w_t, D_k) - w_t. TORCH-DEPENDENT --
    the actual SGD loop. Kept intentionally thin: training-loop detail
    (optimizer choice, batch size) is a model/config concern, not a
    crypto-wiring concern, so it's parameterized rather than hardcoded.
    """
    from .models import flatten_state_dict, unflatten_to_state_dict
    import torch

    flat_w_t, shapes = flatten_state_dict(model)
    state_dict = unflatten_to_state_dict(w_t, shapes)
    model.load_state_dict(state_dict)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    for _ in range(epochs):
        images, labels = local_data.images, local_data.labels
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    new_flat, _ = flatten_state_dict(model)
    delta = (new_flat - w_t).detach().cpu().numpy().astype(np.float64)
    return delta


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
) -> ClientSubmissionPhase1:
    """Steps 0-7 of ARCHITECTURE.md §4's round protocol."""
    # Step 0: VRF self-sampling.
    vrf_out = vrf_evaluate(ctx.keys.sk, ctx.keys.pk, vrf_alpha)
    b_k = vrf_is_sampled(vrf_out.beta, sampling_rate)

    # Step 1: train or contribute zero.
    if b_k:
        delta = local_train_delta(model, w_t, local_data, lr, epochs)
    else:
        delta = np.zeros(ctx.model_dim, dtype=np.float64)

    # Step 2: DP noise share.
    noise_shares = split_noise(sigma_dp, n_shares=1, dim=ctx.model_dim, rng=np.random.default_rng())
    nu_k = noise_shares[0]

    # Step 3: quantize.
    q_k = quantize(delta + nu_k, bits=ctx.quant_bits)

    # Step 4: double-masking.
    masked = q_k.copy()
    for other_idx, secret in ctx.pairwise_secrets.items():
        sign = 1 if ctx.keys.index < other_idx else -1
        masked = (masked + pairwise_mask(secret, rho_t, ctx.model_dim, sign)) % P_MASK
    masked = (masked + self_mask(ctx.self_mask_secret_int.to_bytes(32, "little"), rho_t, ctx.model_dim)) % P_MASK

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
    base_t = clsag_h_p(rho_t)
    base_e = clsag_h_p(rho_e)
    message = commitment.point.b + escrow_ct.c1.b + escrow_ct.c2.b
    sig = clsag_sign(ctx.keys.sk, ctx.keys.index, ctx.ring, message, rho_t, rho_e)

    return ClientSubmissionPhase1(
        commitment_point=commitment.point,
        escrow_ciphertext=escrow_ct,
        key_image_round=sig.key_image_round,
        key_image_epoch=sig.key_image_epoch,
        ring_signature=sig,
        was_sampled=b_k,
        _secret_q=q_k,
        _secret_blinding=blinding,
        _secret_masked_vector=masked,
    )


def round_step_phase2(
    ctx: ClientRoundContext,
    phase1: ClientSubmissionPhase1,
    sketch_seed: bytes,
    sketch_dim: int,
    B_t: float,
    rho_t: bytes,
    rho_e: bytes,
    vrf_alpha: bytes,
) -> ClientSubmissionPhase2:
    """Steps 8-10, run AFTER the server reveals sigma_t.

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
    """
    sketch = TorchSketchProjector(seed=sketch_seed, m=sketch_dim, d=ctx.model_dim)
    signed_q = center_mod_p_mask(phase1._secret_q)
    y_k = sketch.project(signed_q.astype(np.float64))

    norm_estimate = int(np.linalg.norm(y_k))
    n_bits = max(8, int(np.ceil(np.log2(max(B_t * 4, 2)))))
    range_proof, agg_blinding = prove_range(min(norm_estimate, (1 << n_bits) - 1), n_bits)

    transcript = b"r_cons:" + phase1.commitment_point.b
    identity_proof = prove_identity_binding(ctx.keys.sk, rho_t, rho_e, vrf_alpha, transcript)

    return ClientSubmissionPhase2(
        sketch=y_k,
        range_proof=range_proof,
        range_proof_aggregate_blinding=agg_blinding,
        identity_binding_proof=identity_proof,
    )


def reveal_blinding_after_acceptance(phase1: ClientSubmissionPhase1) -> Scalar:
    """Step after V1-V4 acceptance: client sends r_k so the server can
    publish the aggregate opening for V5. Only called once the server
    has posted an accept verdict for this client's I_k^t -- revealing
    r_k before that would let a server retroactively decide whether to
    "accept" based on seeing the opening (ARCHITECTURE.md's explicit
    note: "blinding factor, after acceptance only").
    """
    return phase1._secret_blinding
