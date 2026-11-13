# FIX.md — repair log for JANUS-FL

Opened 2026-11-13 after a full review of the round loop. One entry per
issue: what was wrong, how it was found, what was changed, how the change
was checked, and what the paper needs to say about it. Updated every time
a fix lands. `STATUS.md` remains the ledger of what is tested where; this
file is the record of what was broken and how it was repaired.

Status words used below:

- **FIXED, CONFIRMED** — changed, and confirmed on Colab with real EC + torch.
- **FIXED, SANDBOX-TESTED** — changed and tested in the authoring sandbox
  (NumPy only: no torch, no pynacl). The Colab run is still owed.
- **OPEN** — not started.

## Summary

| # | Issue | Severity | Status |
|---|---|---|---|
| F1 | Aggregate was never unmasked: the model received random values every round | Correctness, blocks everything | **FIXED, CONFIRMED** |
| F2 | Mask PRG is a Python SHA-256 loop: ~5.6 s per mask, n² masks per round | Performance | OPEN |
| F3 | Each pairwise mask is computed twice in the simulation | Performance (simulation only) | OPEN |
| F4 | Sketch sign matrix is regenerated once per client though it is the same for all | Performance (simulation only) | OPEN |
| F5 | Two full-dimension arrays are kept per client for the whole round: 17.6 GB at 50 clients | Memory, blocks 50 clients | OPEN |
| F6 | Local training is one full-batch SGD step; unlikely to learn | ML validity | OPEN |
| F7 | V3 norm check: sketch sent in clear, not bound to the update, commitment not binding for the client | Security and privacy | OPEN (design chosen) |

Order of work: F1 → F2–F5 → F6 → F7.

---

## F1 — The aggregate was never unmasked

**Status: FIXED, CONFIRMED (2026-11-13).** Confirmed on real Colab
infrastructure (CASE2, resnet50, model_dim=23,628,798, 8 clients):
round 0 printed `accepted=8/8, V5=PASS, aggregate=EXACT`. This is the
first round in the project's history whose aggregate was ever
mathematically correct.

### What was wrong

Each client adds two kinds of mask to its quantized update `q_k`
(`fl/client.py`, step 4):

- pairwise masks, one per other client, with opposite signs on the two sides;
- one private self-mask.

The server summed the accepted clients' masked vectors and dequantized the
sum directly (`fl/server.py::run_round`). That sum is

```
sum x_k = sum q_k  +  sum selfmask_k  +  (pairwise masks shared with excluded clients)
```

Pairwise masks cancel only between two clients that are both in the sum.
The self-masks never cancel, and nothing removed them: `recover_self_mask`
had no call site anywhere in `fl/`. A second, independent gap: when a
submission was rejected, the pairwise masks its partners had applied were
left in the sum as well.

### Evidence

Reproduced with the repository's own `quantize`, `pairwise_mask`,
`self_mask`, `aggregate_masked` and `dequantize` (8 clients, d = 1000):

| Quantity | Largest absolute value |
|---|---|
| True mean update | 0.012 |
| What `run_round` returned to `apply_update` | 2.2 × 10¹² |
| Same sum after subtracting the self-masks | equal to the true mean within 4.7 × 10⁻⁶ |

Consequences for everything run before this fix:

- Every round reported `V5=PASS` while applying values of order 10¹² to the
  weights. V5 checks that the summed vectors are the committed ones; it says
  nothing about what the sum decodes to.
- No accuracy figure from the old loop is meaningful.
- Probable cause of the intermittent `invalid value encountered in cast`
  warnings (STATUS.md, ninth and tenth issues): they only ever appeared from
  round 1 onward, the first round trained from destroyed weights. This link
  is an inference and has not been confirmed on Colab.

### What was changed

The missing step is the unmasking round of Bonawitz et al. (CCS 2017).

`crypto/secagg.py`
- `new_self_mask_secret()` draws the self-mask secret from Z_L. The loop
  used a raw 256-bit value, which can exceed L (≈2²⁵²); Shamir sharing
  reduces it mod L, so the reconstructed secret would have expanded to a
  different mask than the one applied.
- `pairwise_round_seed(secret, ρ_t)` derives a one-way, round-specific seed
  from a pair's long-term secret; `pairwise_mask` now expands from that
  seed. A surviving client can release the seed to cancel one pair's mask
  for one round without exposing the long-term secret or any other round.
- `unmask_aggregate(...)` subtracts (a) each accepted client's self-mask,
  reconstructed from at least `t` Shamir shares, and (b) each mask an
  accepted client shares with an excluded one. It raises if fewer than `t`
  distinct shares are supplied, because too few shares reconstruct a wrong
  secret without any error.
- `ShareReleasePolicy` encodes the rule that makes double-masking
  meaningful: about any one client, an honest holder releases self-mask
  shares or pairwise seeds, never both.

`fl/client.py`
- `share_self_mask_secret(ctx, holder_ids, t)`: the client Shamir-shares its
  self-mask secret among the other clients before masking.

`fl/server.py`
- `run_round` takes `unmask_provider(accepted, excluded)`, called after the
  accepted set is fixed, and aborts the round cleanly if unmasking fails.
- `RoundOutcome.aggregate_quantized` exposes the unmasked integer aggregate.

`fl/round_loop.py`
- Shares each client's self-mask secret each round; default threshold
  `t = ⌊2n/3⌋ + 1` (override: `secagg_share_threshold`).
- Supplies the unmasking material from the accepted clients under one
  `ShareReleasePolicy`.
- Simulation-only self-check: compares the unmasked aggregate with the sum
  of the accepted clients' plaintext `q_k`. The round line now prints
  `aggregate=EXACT`; on a mismatch the update is not applied. No party in
  the real protocol could run this check; it exists so this class of bug
  cannot pass silently again.

### How it was checked

Sandbox (NumPy, no EC):

- `tests/test_unmasking.py`, 7 tests, all pass: sum without unmasking is not
  the aggregate; all-accepted recovers the exact sum; rejected clients are
  removed exactly; self-masks alone are insufficient once a client is
  excluded; too few shares raises; the release policy refuses both kinds
  about one client in either order; round seeds differ per round.
- `tests/test_secagg.py`: still passes after the mask derivation change.
- `fl/server.py::run_round` driven end to end with the EC verification
  stubbed (10 clients, 2 rejected, t = 7): aggregate equals the sum of the
  accepted `q_k` exactly; applied delta equals the true mean; too few shares
  gives a clean abort.

Not yet checked: the `fl/round_loop.py` wiring itself, which needs torch
and pynacl. Expected on Colab: every round line shows `aggregate=EXACT`.

### Known limits of this fix

- Excluded-pair seeds are released directly by the surviving partner. A
  malicious survivor can release a wrong seed and corrupt the aggregate; it
  gains no information by doing so. Bonawitz et al. avoid this by
  Shamir-sharing a per-round key-agreement key. Not done here because the
  pairwise keys are long-term.
- The share threshold `t` is a second liveness condition next to `k_min`:
  fewer than `t` accepted clients aborts the round.
- Mask values changed (round-seed derivation), so runs before and after
  this fix are not bit-comparable. All earlier runs are void anyway.
- Observed, not changed: the mean divides by every accepted submission,
  including the unsampled clients' zero-update cover submissions, so the
  effective step is scaled by about the sampling rate.

### For the paper

- Protocol description: state the unmasking round explicitly (who releases
  what, and when), the share threshold, and the release rule. This is the
  mechanism behind claim C7.
- Security argument: C7 holds because no honest client releases both kinds
  of material about the same client; cite the release rule.
- Threat model / limitations: the wrong-seed availability attack above.
- Experiments: all results must come from runs after this fix.

---

## F2 — Mask PRG speed

**Status: OPEN.**
`crypto/secagg.py::_prg` calls SHA-256 from a Python loop, 32 bytes per
call. Measured in the sandbox at d = 23,628,798: 5.56 s per mask vector,
against 0.32 s for an AES-128-CTR keystream with the same reduction. A round
needs n² mask vectors, which is about 83 minutes at 30 clients.
Plan: expand masks with AES-CTR (the choice in Bonawitz et al.).

## F3 — Pairwise masks computed twice

**Status: OPEN.**
Both members of a pair expand the same vector. In a single-process
simulation it can be expanded once and applied with both signs.
For the paper: this is a simulation saving only; per-client cost must be
reported without it.

## F4 — Sketch sign matrix regenerated per client

**Status: OPEN.**
`round_step_phase2` builds a new `TorchSketchProjector` per client from the
same seed σ_t, so the 8192 × d sign matrix is generated n times per round.
Plan: generate each chunk once and apply it to all clients.
For the paper: simulation saving only; a real client still pays for one sketch.

## F5 — Memory: O(n·d) arrays held for the whole round

**Status: OPEN.**
`ClientSubmissionPhase1` keeps `_secret_q` and `_secret_masked_vector`
(2 × d × 8 bytes) per client until the round ends: 10.6 GB at 30 clients
(the 2026-11-12 crash), 17.6 GB at 50.
Plan: keep only small artifacts in RAM; spill the two arrays to disk and
stream them.

## F6 — Local training

**Status: OPEN.**
`local_train_delta` performs one full-batch SGD step per round.
Plan: mini-batches and several local steps; consider a pretrained-backbone
configuration alongside the full-dimension one.

## F7 — V3 norm check

**Status: OPEN. Design chosen 2026-11-13, algebra checked in NumPy and a toy group; no EC code yet.**

Three gaps between `ARCHITECTURE.md` and the implementation:

1. The sketch `y_k` is sent in the clear and the server computes its norm.
2. Nothing binds `y_k` to the update inside the masked vector: a client can
   send the sketch of a zero vector with an arbitrary update.
3. `c_k` commits to a linear digest with weights public before the round,
   so it does not bind the client to its vector before σ_t is revealed.

Chosen design, in implementation order:

| Part | Change | Closes |
|---|---|---|
| F7a | Hash of the masked vector inside the signed phase-1 message | gap 3 |
| F7b | Exact integer sketch; vector Pedersen commitment to it | gap 1 |
| F7c | Sigma protocol for ⟨y, y⟩ = v on the committed sketch, v public | gap 1 |
| F7d | Random subset-sum range tests against field wraparound (reuses `rangeproof.py`) | collusion via wraparound |
| F7e | Aggregate check: product of sketch commitments opens to the sketch of the aggregate | gap 2 |
| F7f | Blame step when F7e fails | abort without attribution |

Depends on F1 (F7e needs the true aggregate).
