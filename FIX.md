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
| F2 | Mask PRG and digest weights expanded by a Python SHA-256 loop; digest multiplied in Python big integers | Performance | **FIXED, SANDBOX-TESTED** |
| F3 | Each pairwise mask is computed twice in the simulation | Performance (simulation only) | **INVESTIGATED, NOT ADOPTED** -- see below |
| F4 | Sketch sign matrix is regenerated once per client though it is the same for all | Performance (simulation only) | **FIXED, SANDBOX-TESTED** |
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

## F2 — Mask PRG and weighted-digest speed

**Status: FIXED, SANDBOX-TESTED (2026-11-13). Colab timing owed.**

### What was wrong

Two O(model_dim) steps ran in slow reference code.

1. `crypto/secagg.py::_prg` expanded every mask by calling SHA-256 from a
   Python loop, 32 bytes per call. A round needs n² mask vectors
   (each of n clients expands n−1 pairwise masks and one self-mask).
2. `crypto/vector_commit.py::weighted_digest` (found while fixing 1) used
   the same SHA-256 loop for its weights, then multiplied weight × value
   in Python big integers, one object per coordinate. It runs once per
   client in phase 1 and once per accepted client in the server's V5 sum.

### What changed

- New `crypto/prg.py`: one expander for both uses.
  `key = SHA-256(tag ‖ length-prefixed parts)`, then an AES-256-CTR
  keystream, 8 bytes per element, shifted right by 3 to 61 bits, with the
  single value 2^61 − 1 mapped to 0. AES in counter mode is the mask PRG
  of Bonawitz et al. (CCS 2017). Distance from uniform: 2^-61 per element.
  The parts are length-prefixed, which the old concatenation was not.
- `_prg` and `_weight_stream` call it, under different domain tags.
- `weighted_digest` multiplies in uint64 with an exact reduction for the
  Mersenne prime 2^61 − 1 (`_dot_mod_p`), and reads the weights chunk by
  chunk, so it no longer allocates a model_dim-sized weight array.
- No fallback to the old expander: two parties on different expanders
  would derive different masks and the aggregate would not unmask.
  `cryptography` is therefore a hard dependency (preinstalled on Colab).

### Measured (sandbox CPU, best of 3)

| Step | model_dim | Before | After | Speed-up |
|---|---|---|---|---|
| One mask vector | 11,208,318 | 2.88 s | 0.091 s | 32× |
| One mask vector | 23,628,798 | 5.68 s | 0.164 s | 35× |
| One weighted digest | 11,208,318 | 4.63 s | 0.285 s | 16× |
| One weighted digest | 23,628,798 | 9.15 s | 0.575 s | 16× |

Per round these two steps cost n² masks + 2n digests. At 8 clients and
model_dim 23.6M that is about 510 s before and 20 s after on this CPU.
The Colab round measured 1617 s in total, so other costs remain
(training, sketch, range proofs, unmasking); the next Colab timing line
will show how much.

### How it was checked

`tests/test_prg.py`, 7 tests: determinism, range, prefix property,
independence from chunk size, domain and part-boundary separation, a
coarse uniformity check, `_dot_mod_p` against big-integer arithmetic
including the largest inputs, `weighted_digest` against the big-integer
sum across chunk boundaries, and linearity mod P. All 67 repository tests
pass, and the server-path dry run from F1 still gives an exact aggregate.

### Known limits

- Mask values differ from the old expander's. Nothing was stored, so
  nothing breaks, but old and new code cannot be mixed in one run.
- The 16× digest figure is NumPy on CPU; a GPU version is possible but
  not needed at this size.

### For the paper

- State the PRG: AES-256-CTR keyed by SHA-256 of the pairwise or self
  seed and the round nonce, output in Z_p with p = 2^61 − 1.
- T1 (update confidentiality) already assumes PRG security; this names
  the PRG.
- Per-client mask cost is (n−1) + 1 expansions of model_dim elements;
  report it from the measured per-vector time.

## F3 — Pairwise masks computed twice

**Status: INVESTIGATED, NOT ADOPTED (2026-11-13).** Built, measured, and
disabled by default -- the numbers below are why.

### The idea

Two parties in a pair share the same shared_secret (X25519 ECDH), so
they derive the same round seed and the same magnitude vector; only the
sign differs. In this single-process simulation, both parties' masking
calls run in the same process, so the vector could in principle be
expanded once per pair and reused with the opposite sign for the
second party, instead of expanding it twice.

### What was built

`crypto/secagg.py::pairwise_mask_cached(shared_secret, round_nonce, dim,
sign, cache, pair_key)`: with `cache=None` it is exactly `pairwise_mask`
(the real per-party cost); with a dict, the first call for a `pair_key`
expands and stores the magnitude, the second call reuses it with the
other sign.

### What the numbers said

Measured in the sandbox at model_dim = 11,208,318 (ResNet-18 estimate),
8 clients, best of 3:

| | Uncached | Cached |
|---|---|---|
| Wall time, all 56 expansions | 11.4 s | 12.7 s (slower) |

The cache is not free: the dict in `fl/round_loop.py`'s loop structure
does not release a pair's entry until the SECOND party is processed,
which can be much later in the per-client loop -- so it holds up to
`n(n-1)/2` full `model_dim`-length arrays at once in the worst case,
not `n`. At this project's own scale:

| Clients | Cache, worst case |
|---|---|
| 8 | 2.5 GB |
| 30 | 39.0 GB |
| 50 | 109.8 GB |

30 clients is exactly the count that caused the RAM crash FIX.md F5
exists to fix. Shipping F3 as designed would make that crash worse, not
better, and the sandbox timing shows it is not even a time win at 8
clients -- the memory traffic of holding many large arrays costs more
than the AES calls it avoids.

### Decision

`fl/round_loop.py` passes `pairwise_mask_cache_this_round = None`,
which makes every call go through the uncached path -- behavior is
unchanged from before F3 was investigated. `pairwise_mask_cached` and
`tests/test_mask_cache.py` stay in the repo: a pair-ORDERED redesign
(process pairs, not clients, as the outer loop, so each magnitude is
consumed by both parties immediately and freed) could recover the
saving without the memory blowup, but that is a real restructure of
`round_step_phase1`'s calling convention, not a drop-in cache, and
is not worth it unless F2 and F4 turn out to be insufficient.

### For the paper

Nothing changes: this never touched per-client real-world cost, and it
was never adopted in the simulation either.

---

## F4 — Sketch sign matrix regenerated per client

**Status: FIXED, SANDBOX-TESTED (2026-11-13). Colab timing owed.**

### What was wrong

`round_step_phase2` built a fresh `TorchSketchProjector(seed=sigma_t, ...)`
and called `.project()` once per client, even though sigma_t is the same
for every client in a round -- the projector's (m, d) sign chunks (an
internal detail of `.project()`'s chunked loop) are therefore identical
across clients, and were being regenerated from scratch n times.

### What changed

- `SketchProjector.project_batch(qs)` (NumPy) and
  `TorchSketchProjector.project_batch(qs)` (torch): generate each
  chunk's signs once and apply them to every vector in `qs` in the
  same pass, instead of once per vector. Same chunking and seeding as
  `project()`.
- `fl/client.py::sketch_phase1_batch(phase1_by_client, sketch_seed,
  sketch_dim, model_dim)`: calls `project_batch` across every client's
  `_secret_q` for the round, returns `{client_idx: y_k}`.
- `round_step_phase2` takes an optional `_sim_precomputed_sketch`; when
  given, it skips building its own projector. `fl/round_loop.py` calls
  `sketch_phase1_batch` once per round and passes each client's result
  in. Unlike F3, this holds no more memory than the round loop already
  needs: `project_batch` stacks the `n` per-client vectors it is given
  exactly once, which is the same `n * model_dim` the loop already
  holds in `phase1_by_client` (F5's problem, not a new one) -- it does
  not multiply by the number of pairs the way F3's cache did.

### Measured (sandbox CPU, NumPy reference, best of 3; TorchSketchProjector
itself is untested here, same caveat as the rest of that class)

m = 8192, 8 clients, d = 200,000 (smaller d so the per-client loop
finishes in sandbox time; cost is O(m*d), so this is representative):
per-client loop 79.1 s, batched 10.7 s -- **7.4x**.

### How it was checked

`tests/test_sketch.py`: `project_batch` output matches calling
`project()` once per vector (within float32 reordering tolerance),
handles 0/1 vectors, rejects wrong dimension. A separate dry run
(outside the test suite, torch stubbed) confirmed
`sketch_phase1_batch`'s own plumbing -- extracting `_secret_q`,
centering, keying the output dict -- matches the direct per-client
path exactly, with the NumPy `SketchProjector` standing in for
`TorchSketchProjector` since torch is not installed in this sandbox.
75 repository tests pass.

### Known limits

- `TorchSketchProjector.project_batch` itself has not run anywhere
  torch is actually installed. Written carefully against `project()`'s
  already-reviewed-but-unrun implementation; validate on Colab before
  trusting the GPU path, same as `project()`.
- This is a simulation-only saving, same as F3's idea was: a real
  client still pays for exactly one `project()` call, since it has no
  other client's vector to batch with. Must not be used when reporting
  per-client cost in the paper.

### For the paper

- State the real per-client sketch cost as one `project()` call at
  m=8192, d=model_dim -- `sketch_phase1_batch` is simulation plumbing,
  not a protocol change.

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
