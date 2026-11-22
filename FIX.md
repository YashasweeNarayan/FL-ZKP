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
| F2 | Mask PRG and digest weights expanded by a Python SHA-256 loop; digest multiplied in Python big integers | Performance | **FIXED, CONFIRMED** |
| F3 | Each pairwise mask is computed twice in the simulation | Performance (simulation only) | **INVESTIGATED, NOT ADOPTED** -- see below |
| F4 | Sketch sign matrix is regenerated once per client though it is the same for all | Performance (simulation only) | **FIXED, CONFIRMED** |
| F5 | Two full-dimension arrays are kept per client for the whole round: 17.6 GB at 50 clients | Memory, blocks 50 clients | **FIXED, CONFIRMED** |
| F6 | Local training is one full-batch SGD step; unlikely to learn | ML validity | **FIXED, SANDBOX-TESTED** (indexing scheme only) |
| F7 | V3 norm check: sketch sent in clear, not bound to the update, commitment not binding for the client | Security and privacy | OPEN (design chosen) |
| F8 | `crypto/ristretto.py` called ristretto255 bindings PyNaCl has never shipped, under any version | Correctness, blocks every EC-dependent module | **FIXED, CONFIRMED** |

Order of work: F1 → F2–F5 → F6 → F7. F8 was found and fixed out of this
order (2026-11-13), discovered while chasing a Colab install error that
blocked re-confirming F1-F5 at all.

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

**Status: FIXED, CONFIRMED ON COLAB (2026-11-13).** resnet18, 8 clients,
2 rounds, real GPU: `round 0 timing: phase1=45.8s, phase2=31.9s, verify=8.7s,
total=86.4s (~10.81s/client)`, round 1 nearly identical (`43.9s/31.8s/8.2s`).
Against the pre-fix resnet50 baseline at the same client count
(`phase1=1102.9s, phase2=257.7s, verify=256.9s, total=1617.5s`) -- not a
clean apples-to-apples isolation (model_dim also dropped ~53% with the
ResNet-18 swap), but the direction and magnitude are exactly what F2 (mask
PRG + exact-arithmetic digest) plus F4 (batched sketch) predict, and
`verify`'s ~30x drop in particular tracks F2 directly since `verify_submission`
calls the same `weighted_digest` the mask-PRG fix sped up. `aggregate=EXACT`
held on BOTH rounds, which also newly confirms F1 survives a real round
boundary (round 1 trains from a correctly-unmasked round-0 model), not
just round 0 in isolation.

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

**Status: FIXED, CONFIRMED ON COLAB (2026-11-13).** Bootstrap norm sampling
(this fix's other call site, `bootstrap_norms_from_model_dim`): 10 samples
at resnet18 (model_dim=11,202,046) in 39.7s total, vs. the pre-fix ~330s at
resnet50 one-sample-at-a-time -- confirms the batching change, not just the
smaller model (the printed per-call breakdown, `7 in one pass` then
`3 in one pass`, shows the batching itself firing). Phase 2 numbers are
under F2's entry above (both fixes land in the same round-loop run).

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

**Status: FIXED, CONFIRMED ON COLAB (2026-11-13).** This is the bug behind
the 2026-11-12 runtime crash. Confirmed at the exact crash config (resnet18,
30 clients, 1 round): `accepted=30/30, V5=PASS, aggregate=EXACT`, `23 held
in RAM (peak 1.92 GiB), 37 spilled to disk`, process RSS flat at 2.19 ->
2.28 GiB against the `5.01 GiB` the notebook itself computed the old code
would have pinned simultaneously for this config -- see STATUS.md.

### What was wrong

Each client produces two full-length int64 arrays in phase 1:

- `q_k`, the quantized update, needed later by phase 2's sketch;
- `x_k = q_k + m_k`, the masked vector, needed later by the server.

Both were carried on `ClientSubmissionPhase1`, and the round loop held
every client's submission until the round ended. That is
`2 · n · model_dim · 8` bytes resident at once: 2.8 GB at 8 clients and
ResNet-50, **10.6 GB at 30** — the crash — and 17.6 GB at 50. A ceiling
on client count, not a tuning problem.

### Why the obvious fix was the wrong one

The tempting restructure is to fuse phase 1 and phase 2 per client, so
each `q_k` dies immediately. That breaks the protocol:
ARCHITECTURE.md line 194 requires σ_t to be published *after every*
`c_k` is on the bulletin board, because §9's soundness argument needs
the projection seed chosen independently of any client's `q_k`. Fusing
the phases would reveal σ_t to client 1 before client 2 had committed.

So the ordering stays exactly as it was. What changed is only *where
the arrays live between the phases* — which is not a protocol fact.

### What changed

- **New `fl/vector_store.py`.** `RoundVectorStore` holds one round's
  arrays, keeping them resident up to a byte budget and spilling the
  rest to `.npy` files. Closed in a `finally`, so an aborted round
  leaves nothing behind across a 45-round sweep. Arrays come back
  read-only: the resident path returns the stored array while the
  spilled path returns a fresh load, so a caller that mutated one
  would corrupt data below the budget and not above it — read-only
  turns that into an immediate failure instead of a scale-dependent
  heisenbug.
- **`round_step_phase1` takes a `vector_sink`.** Given one, it hands
  over `q_k` and `x_k` and returns a submission carrying neither.
  Without one, behaviour is exactly as before.
- **`fl/server.py` streams.** `run_round` takes a
  `masked_vector_provider(client_index)` and computes the masked sum
  **and** each client's digest in a single pass, fetching one vector
  at a time and dropping it before the next. Peak is the accumulator
  plus one vector — O(d), independent of how many clients were
  accepted. The default provider reads off the submission, so the old
  calling convention still works.
- **`aggregate_masked` accepts any iterable**, so callers can stream a
  generator instead of materializing a list of n vectors.
- **Phase 2 batches in bounded groups.** F4's batching stacked all n
  clients, which would have reintroduced exactly this O(n·d)
  residency. `sketch_phase1_batch` now takes a `q_provider` and a
  `batch_size` chosen from a byte budget by
  `suggested_batch_size`, so it does not grow with n. Batch 7 at
  ResNet-18 still does the sign-chunk work once per 7 clients instead
  of once per client, keeping most of F4's win.
- **Two separate budgets**, not one knob shared: `vector_budget_bytes`
  (default 2 GiB, resident across the round) and
  `sketch_batch_budget_bytes` (default 1 GiB, transient inside phase
  2). They can be live together, so the honest peak is about their
  sum; sharing one number would have understated it by the batch.

### Measured — peak process RSS, not an estimate

The crash was a real-memory event, so this is a real-memory
measurement: `run_round` driven end to end at d = 2,000,000, one
process per configuration, `ru_maxrss` high-water mark. "old" pins each
masked vector on its submission; "new" streams from the store.

| Clients | old | new |
|---|---|---|
| 8 | 265 MiB | 150 MiB |
| 16 | 387 MiB | 150 MiB |
| 32 | 632 MiB | **151 MiB** |

The aggregate matched the true sum of the accepted `q_k` in all six
runs. "old" climbs by ~16 MiB per client — exactly one `model_dim`
int64 array — while "new" is flat. This harness pins one array per
client; the round loop pinned two, so the real slope was double.

Projected at the models in play, with the default budgets:

| Model | Clients | Before | After | Disk |
|---|---|---|---|---|
| ResNet-18 | 30 | 5.0 GiB | 2.5 GiB | 3.0 GiB |
| ResNet-18 | 50 | 8.4 GiB | 2.5 GiB | 6.4 GiB |
| ResNet-50 | 30 | 10.6 GiB | 2.6 GiB | 8.6 GiB |
| ResNet-50 | 50 | 17.6 GiB | 2.6 GiB | 15.6 GiB |

### How it was checked

- `tests/test_vector_store.py`, 13 tests: exact round-trip on both the
  resident and the spilled path, mixed residency staying under budget,
  read-only returns on both paths, cleanup and idempotent close,
  use-after-close, duplicate put, missing key, early drop, and the F5
  claim itself — peak residency identical at 8, 30, 50 and 200
  clients. Plus: `aggregate_masked` over a generator equals the list
  form, streaming from the store equals the list form, and the fused
  one-pass sum-and-digest equals the two separate list passes.
- A dry run (torch stubbed, NumPy projector substituted) confirmed
  `sketch_phase1_batch` returns results identical to the per-client
  path at batch sizes 1, 2, 3, 7 and 16, on both the all-resident and
  all-spilled store, and deterministic across repeats.
- The F1 server dry run still gives an exact aggregate through the
  default provider, and still aborts cleanly on too few shares.
- 88 repository tests pass.

### Known limits

- **Not yet run on Colab.** The round-loop wiring needs torch and
  pynacl. The pieces are tested here; the integration is not.
- **Disk traffic.** At 50 clients and ResNet-50 a round writes and
  re-reads ~15.6 GB. Colab's local disk has room, but this is real I/O
  time that the next Colab timing line will show. Raising
  `vector_budget_bytes` on a larger-RAM runtime removes it.
- **Slightly more RAM at small n.** At 8 clients and ResNet-18
  everything fits resident anyway, and the phase-2 batch adds ~0.4 GiB
  on top — about 1.8 GiB where the old code used 1.3. The win starts
  where it matters (n ≥ 30) and the ceiling is flat after that.
- **A full disk now fails the round loudly** (the `np.save` error
  propagates) rather than silently dropping a vector — deliberate,
  given F1 was exactly a silent-corruption bug.

### For the paper

- Peak server memory is **O(model_dim), independent of client count**:
  the server holds one submission's vector and a running accumulator.
  This is also what a real deployment does — process each submission
  as it arrives, accumulate, discard. Holding all n was an artifact of
  running every party in one process, so the simulation is now *more*
  faithful, not less.
- Per-client memory is unchanged and was always O(model_dim): a real
  client only ever holds its own vectors.
- The digest is recomputed by the server from the vector it received,
  never taken from a client-supplied value. That recomputation is what
  binds `x_k` to `c_k` in V5; a client-supplied digest would match its
  own `c_k` by construction and V5 would pass for any `x_k` at all.
  Worth stating explicitly, since it is the kind of shortcut that
  looks like a harmless optimization.
- σ_t is still published only after every `c_k` is posted. The fix
  changed storage, not message order.

## F6 — Local training

**Status: FIXED, SANDBOX-TESTED (the indexing scheme only — see below), 2026-11-13.**

### What was wrong

`local_train_delta` took `epochs` full-batch gradient steps over the
*entire* client shard per round -- not real FedAvg local training
(McMahan et al. 2017, §3): every "epoch" computed the exact same full
gradient (no stochasticity -- SGD's "S" was never actually stochastic),
and there was no notion of an epoch as a pass over shuffled mini-batches.
This was flagged live on 2026-11-12 in the training-divergence error
message itself ("note local_data is trained as ONE full-batch step per
`epochs`, not mini-batched") as a likely contributor to that crash, but
left unfixed at the time since F1-F5 took priority.

This is an ML-validity bug, independent of the crypto-correctness bugs
F1-F5/F8: it does not change any cryptographic guarantee, only whether
local training does what the paper's FedAvg baseline is supposed to do.

### What changed

`fl/client.py::local_train_delta` now runs real mini-batch, multi-epoch
SGD: for each of `epochs` local epochs, the client's shard is freshly
shuffled and walked in `batch_size`-sized slices, one optimizer step per
slice (the last slice of a pass is whatever's left over -- not dropped,
since FEMNIST per-writer shards are already small). Added `batch_size`
(required) and `momentum` (optional, default 0 — plain SGD, matching the
FedAvg paper's own default) parameters; also an optional `seed` for
reproducible ablations (unset in real rounds, which use torch's global
RNG as before).

Plumbed through `round_step_phase1` (new `batch_size`, `momentum` params)
and `run_training_loop`, which now reads `lr`, `local_epochs`,
`batch_size`, `momentum` from a new `configs/mvp_3day.yaml` `local_training`
section by default (explicit function args remain an override for tests/
ablations). Defaults chosen: `lr=0.01` (unchanged), `local_epochs=5`,
`batch_size=20`, `momentum=0.0` -- E and B both within the FedAvg paper's
own tested range for a non-IID, small-per-client-shard setting (this
repo's CASE1 gives each client 950 samples, CASE2 as few as
4 x 120 = 480), rather than its largest-E/smallest-B extreme.

### Checks

`fl/client.py` cannot be imported at all in this sandbox (it pulls in
`crypto/pedersen.py`, which computes its Pedersen generator at *module
import time* via `crypto/ristretto.py`, which needs `rbcl` -- not
installed here, see F8). So the real function is untested here, same as
before this fix. New `tests/test_f6_batching_logic.py` (6 tests) instead
mirrors the exact indexing scheme (`for start in range(0, n, batch_size):
perm[start:start+batch_size]`, reshuffled every epoch) with NumPy instead
of torch, and checks: full coverage with no duplicates within an epoch,
batch count matches `ceil(n/batch_size)`, only the last batch is
short, `batch_size >= n` degenerates to one batch per epoch, two
independent shuffles don't coincide (guards against a "reshuffle"
bug that silently reuses one permutation for every epoch), and total
batches across multiple epochs is `epochs * ceil(n/batch_size)`. This
verifies the scheme's arithmetic, not torch's gradients or loss --
**the actual training loop is still owed its first Colab run.** 96 tests
now pass across the repo (16 files).

### What the paper needs to say

- Local training is `local_epochs=5` passes of mini-batch SGD,
  `batch_size=20`, plain SGD (no momentum), `lr=0.01` -- state this
  explicitly as the FedAvg baseline configuration (ROADMAP.md's "FedAvg,
  50 clients, VRF self-sampling, no crypto" accuracy-ceiling run should
  use the SAME local-training hyperparameters, or the ceiling comparison
  is apples-to-oranges).
- Before this fix, every reported round number (F1's `aggregate=EXACT`
  confirmations, F2/F4's timing numbers) was trained with `epochs=1`
  full-batch steps -- correct aggregation and correct crypto, but not
  yet real FedAvg local training. Nothing about those confirmations is
  invalidated (they tested the crypto layer, which is agnostic to what
  `local_train_delta` computes), but any *accuracy* number collected
  before this fix should be re-run, not reused.

## F7 — V3 norm check

**Status: FIXED AND WIRED (F7a-f), CONFIRMED ON COLAB WITH REAL RBCL
(2026-11-14).** F7a-d and F7e/F7f are all implemented and wired into
`fl/client.py` + `fl/server.py::run_round`. 159/159 tests pass against real
`rbcl` bindings (not just the pure-Python stand-in), a real 8-client
resnet18 round loop runs `V5=PASS, F7e=PASS, aggregate=EXACT`, and the
lying-sketch/wraparound attack demo passes against real crypto -- see
"Gates B/C/D" and "Gate A" below for the actual numbers. Open items are
the timing anomalies and the handful of protocol-policy questions listed
under "Still open on F7, honestly" -- none of them are wiring or
correctness gaps.

Cause (unchanged from the original diagnosis): three gaps between
`ARCHITECTURE.md` and the implementation --

1. The sketch `y_k` is sent in the clear and the server computes its norm.
2. Nothing binds `y_k` to the update inside the masked vector: a client can
   send the sketch of a zero vector with an arbitrary update.
3. `c_k` commits to a linear digest with weights public before the round,
   so it does not bind the client to its vector before σ_t is revealed.

**Design correction made during implementation (2026-11-13/14), replacing the
plan below as originally written.** The original F7b/F7c pairing ("vector
Pedersen commitment to the sketch" + "Sigma protocol for ⟨y,y⟩=v on that
commitment") doesn't fit together: proving an inner product of a vector held
in ONE multi-generator commitment needs an inner-product argument
(Bulletproofs-class), which `schnorr_nizk.py`'s own docstring already ruled
out under the 3-day deadline. Fix: commit each sketch coordinate SEPARATELY
(m ordinary Pedersen commitments, not one aggregated point), each paired
with a commitment to its square and a real Sigma-protocol "square proof"
(the standard Pedersen multiplicative-relation construction, derived and
algebra-checked in `crypto/norm_proof.py`'s module docstring). Cost: roughly
8x the scalar multiplications of today's `rangeproof.py` per client (m=8192
square proofs vs. one ~16-32 bit range proof) -- not yet measured on Colab.

A second, independent design bug was caught and fixed before shipping (not
after): F7e's "reveal the aggregate blinding" and F7d's original "reveal the
flooded test value in the clear" both have the SAME oracle-attack shape --
a sketch coordinate's (or a flooded linear combination's) value range is
small enough to be recovered by brute-force discrete log once its blinding
is known, UNLIKE V5's existing `r_k` reveal (safe there only because
`weighted_digest`'s range is the full ~2^61 P_MASK space). Fix for F7d:
`prove_wraparound_binding`/`verify_wraparound_binding` -- a Schnorr proof
(on base H) that a FRESH range-proof commitment and the homomorphic sum of
already-published per-coordinate commitments open to the same value, so
neither the flooded sum nor any blinding is ever revealed. F7e's equivalent
fix is noted below (not yet wired).

A third bug, in the first draft of `wraparound_window`: it took its bound
on a sketch coordinate's magnitude from the client's OWN array (not
independently recomputable by the server) and from P_MASK//2 (a bound ~2^61,
vastly looser than an honest coordinate's real magnitude, ~2^(quant_bits-1)
times the number of nonzero sketch-row entries -- for model_dim in the
millions, still tens of orders of magnitude below P_MASK). Both were wrong:
the first breaks client/server agreement entirely; the second would make a
P_MASK-sized wraparound attack invisible against an "honest" window already
that loose. Fixed: `wraparound_window(m, model_dim, density, quant_bits)` is
a pure function of PUBLIC config only, tight enough that a P_MASK-sized shift
in even one coordinate dwarfs the honest window (verified in
`tests/test_f7_norm_proof.py::test_wraparound_window_is_tight_against_pmask`
and `test_wraparound_catches_pmask_sized_shift`).

Final design, in implementation order:

| Part | Change | Closes | Status |
|---|---|---|---|
| F7a | Hash of the masked vector inside the signed phase-1 CLSAG message (`fl/client.py::round_step_phase1`, `ClientSubmissionPhase1.masked_vector_hash`) | gap 3 | DONE, wired into `check_V1_ring_signature` |
| F7b | Exact integer sketch (`crypto/sketch.py::project_int`/`project_int_batch`); m PER-COORDINATE Pedersen commitments, not one vector commitment | gap 1 | DONE (`crypto/norm_proof.py::commit_sketch`) |
| F7c | Real Sigma-protocol "square proof" per coordinate; only Σ(square commitments) is opened, to the public v=⟨y,y⟩ | gap 1 | DONE (`SquareProof`/`prove_square`/`verify_square`) |
| F7d | Cheap random subset-sum wraparound test, HIDDEN (reworked from the original bit-decomposition-reuse plan into a Schnorr binding + `rangeproof.py` reuse on the few flooded values) | wraparound collusion | DONE (`prove_wraparound_binding`/`verify_wraparound_binding`) |
| F7e | Aggregate check: per-coordinate sum of accepted clients' sketches == sketch of the true aggregate, blinding sums only (never individual) | gap 2 | DONE and wired into `run_round` |
| F7f | Group-testing blame (bisect the accepted set on a public, round-seeded partition) when F7e fails, instead of abort-only | abort-without-attribution DoS | DONE and wired into `run_round` |

**Wiring done:**
- `fl/client.py::round_step_phase1`: computes `masked_hash = sha256(masked)`,
  includes it in the CLSAG-signed message, stores it as
  `ClientSubmissionPhase1.masked_vector_hash`.
- `fl/client.py::round_step_phase2`: rewritten. Uses `sketch.project_int`
  (exact integers, not the float estimate), builds a `SketchCommitment` via
  `commit_sketch`, runs F7d's 8 wraparound tests, returns a
  `ClientSubmissionPhase2` carrying the commitment + proofs and NO cleartext
  sketch. `sketch_phase1_batch`/`_sim_precomputed_sketch` (F4's cross-client
  float-sketch batching) are retired from this path -- they batched the
  wrong (float, cleartext) computation; `fl/round_loop.py` now calls
  `round_step_phase2` per client directly, at real per-client cost.
- `fl/server.py::check_V1_ring_signature`: includes `masked_vector_hash` in
  the reconstructed message.
- `fl/server.py::check_V3_norm_bound`: fully rewritten -- verifies the
  `SketchCommitment` (never sees y), re-derives the SAME public weights and
  window the client used (needs `sigma_t`, now threaded through
  `verify_submission`/`run_round` as an explicit required parameter -- a
  caller passing the wrong seed, or omitting it, fails loudly rather than
  silently mis-verifying), and checks each wraparound binding + range proof.
- `fl/round_loop.py`: next-round `B_t` norms now come from
  `int_sketch_norm(phase2.sketch_commitment.v, ...)` (the public scalar),
  not `np.linalg.norm(phase2.sketch)` (no longer exists on the submission).

### F7e/F7f wiring (done 2026-11-14)

**The mechanism, and the shortcut that is wrong.** The server needs
Σ_k r_{k,i} to open the product of the accepted clients' per-coordinate
commitments. It must not learn any single r_{k,i}: a sketch coordinate is
bounded by the norm check itself, so C_{y_i} plus that client's own r_{k,i}
gives y_i·G with y_i small enough to brute-force. V5's existing per-client
`r_k` reveal is safe ONLY because `weighted_digest`'s range is the full ~2^61
P_MASK space -- that argument does not transfer, and assuming it did was the
trap the first draft of this module fell into twice (once for F7d, once here).

So the blinding vector gets the same Bonawitz pairwise masking the update
already gets, one field up (Z_L instead of Z_{P_MASK}):
`rho_{k,i} = r_{k,i} + Σ_{j≠k} sign(k,j)·PRF(pair_seed_kj, i) mod L`, published
in phase 2 BEFORE the accepted set is known. New
`crypto/prg.py::expand_scalar_stream` (64 keystream bytes per scalar, reduced
mod L, own domain tag so it is a different keystream from the vector mask off
the same pair seed); new `crypto/norm_proof.py::mask_blinding_vector` /
`reconstruct_blinding_sum`, the latter mirroring `secagg.unmask_aggregate`'s
pairwise loop exactly -- same `pairwise_round_seed` values accepted clients
already release about excluded ones, same sign rule, same validation.

**F7f's subgroup test needed a real mechanism too.** A first attempt could not
work: for a strict subgroup the server has no independent "true" partial
aggregate to compare against, because `aggregate_quantized` is only the whole
accepted set. It does have another route -- once the accepted set's self-mask
shares are released (they already are, that is how `unmask_aggregate` works),
the only thing still hiding an individual q_k is its pairwise masks, so a
subgroup's partial aggregate becomes computable as soon as the clients IN that
subgroup release their pair seeds about the accepted clients OUTSIDE it. That
is `unmask_aggregate` run on the subgroup, supplied through `run_round`'s new
`blame_material_provider`. **That is the privacy cost, and it is the real
reason `min_group_size` exists:** each bisection step hands the server one more
partial aggregate, so bisecting to a single client would hand over one client's
update in the clear. `localize_fault` stops at `min_group_size`, so the
smallest thing the server ever learns is the summed update of that many
clients -- ARCHITECTURE.md §6.3's accountability-anonymity budget, spent
explicitly and only on a round that already failed an integrity check.

**Also wired:** `fl/client.py::release_straddling_pair_seeds` (the per-client
half, replacing the earlier `reveal_y_blindings_after_acceptance`, which was a
simulation-only stand-in and is gone); `RoundOutcome.aggregate_sketch_check_passed`
/ `_bad_coordinates` / `_suspect_groups`; an F7e failure ABORTS the round
(applying an aggregate that is not the sum of what was committed is the same
class of failure V5 aborts on) carrying the suspect groups; `fl/round_loop.py`
builds the provider and prints `F7e=PASS/FAIL/NOT-RUN` per round.

**`None` is not `False`.** A missing masked blinding vector, or a blinding-sum
reconstruction that cannot complete, returns `None` ("could not be run") and
the round proceeds as it did before F7e existed. Only a reconstruction that
SUCCEEDS and then disagrees returns `False`. Reporting a liveness gap as a
failed integrity check would be a false accusation, and with F7f attached it
would be a false accusation against a named group.

**One more wiring bug caught by the new end-to-end test:** `n_tests` (F7d's
number of wraparound tests) was hardcoded 8 in `round_step_phase2` and
defaulted to 8 in `check_V3_norm_bound` -- agreeing only by coincidence, with
nothing tying them. Any change to one would have failed every honest client
with a V3 rejection. Now threaded explicitly through
`run_round`/`verify_submission`/`round_step_phase2`, and
`tests/test_round_end_to_end.py` deliberately runs `n_tests=4` so the threading
is actually exercised rather than passing because both sides happen to say 8.

**Tests (all on real primitives, pure-Python group backend in the sandbox):**
- `tests/test_f7e_aggregate.py` (14): the masking algebra (full set, strict
  subset, missing seed, bad seed provenance), `_true_aggregate_sketch`'s
  centering assumption, F7e passing honestly and catching BOTH a wraparound
  cheater and a lying-sketch cheater, `None`-vs-`False`, and F7f localizing
  each cheat to a group containing it and smaller than the accepted set.
- `tests/test_round_end_to_end.py` (8): `run_round` itself -- the first test in
  the repo to execute it at all. Honest 8-client round passing V1/V2/V3/V5/F7e,
  the aggregate matching Σq_k, V5 via `verifier/audit.py`, V1 rejecting a
  tampered masked-vector hash (F7a), V3 rejecting an over-bound norm, and F7e
  aborting + localizing the lying-sketch cheat. Also the gap worth knowing:
  that cheat passes every other check, which is what makes F7e load-bearing
  rather than defense-in-depth.
- 169 tests pass with a group backend live; see below for the breakdown.

**Two of the three gaps above closed (2026-11-14):**

- **Blinding-vector dropout resilience, fixed.** The blinding vector now
  gets a SECOND self-mask layer, over Z_L, exactly mirroring
  `secagg.self_mask`/`secagg.unmask_aggregate`'s pattern for the update
  vector: each client draws a separate `blinding_self_mask_secret_int`
  (`crypto.secagg.new_self_mask_secret()`, never reused from the update
  vector's own secret), Shamir-shares it among the other clients at the
  start of the round, and adds `norm_proof.blinding_self_mask` on top of
  its own r_{k,i} values before publishing. The server recovers it from
  `share_t` pre-distributed shares via `reconstruct_blinding_sum`'s new
  `self_mask_shares`/`share_threshold`/`round_nonce` arguments -- so a
  client that drops out after acceptance no longer makes that round's
  F7e check un-runnable; reconstruction never depended on that client
  being reachable, same as the update path never did. Wired through
  `fl/client.py` (`ClientRoundContext.blinding_self_mask_secret_int`,
  `share_blinding_self_mask_secret`), `fl/server.py`
  (`UnmaskingMaterial.blinding_self_mask_shares`, threaded into
  `_check_F7e_aggregate_sketch` and its F7f subgroup path -- the
  `blame_material_provider` contract is now a 3-tuple,
  `(partial_q, straddling_seeds, group_blinding_self_mask_shares)`), and
  `fl/round_loop.py` (a second `blinding_self_mask_shares_held` dict,
  populated and released the same way as the existing one). New test:
  `test_f7e_survives_a_post_acceptance_dropout` in
  `tests/test_f7e_aggregate.py` -- asserts the check still passes using
  ONLY the pre-distributed shares, with the per-client secret never put
  anywhere the reconstruction call could read it.
- **What happens to a suspect group, decided and implemented.**
  `fl/suspect_tracker.py::SuspectTracker` turns F7f's suspect-group
  reports into an actual policy without ever violating F7f's own
  design (it still never names an individual): a FIRST report
  quarantines every member of the reported group for
  `quarantine_rounds` rounds -- a suspect group alone is not proof
  against any one member of it, so nothing is banned yet. If the SAME
  client keeps showing up across independent, structurally different
  bisections (`localize_fault`'s own partition seed differs per
  round), the INTERSECTION of the groups they've appeared in only ever
  shrinks, and once it narrows to `narrow_threshold` (default 1) or
  fewer members, those members are banned outright -- pure set
  arithmetic over disclosures F7f already made, spending no additional
  privacy budget. Wired into `fl/round_loop.py::run_training_loop` (new
  optional `suspect_tracker` parameter, `None` -> fresh tracker): every
  round calls `tracker.record(outcome.aggregate_sketch_suspect_groups)`
  then `tracker.tick()`, and each round's participant set is
  `tracker.eligible(range(n_clients))` rather than the unconditional
  full ring. Fully unit-tested (8 tests, pure logic, no backend to gate
  on): `tests/test_suspect_tracker.py`.

  **A real bug this wiring surfaced, caught before it shipped:**
  restricting the participant set broke `fl/server.py::run_round`'s
  `excluded_indices` computation, which derived it from `submissions`
  alone (`[idx for idx, _ in submissions if idx not in accepted_set]`).
  A client absent from `submissions` entirely (quarantined, never
  submitted) is NOT the same as a client present but rejected -- but
  every OTHER client's masked vector still carries a pairwise mask term
  against that absent client (masking iterates over every configured
  partner in `ctx.pairwise_secrets`, computed once up front for the
  full ring, regardless of per-round eligibility). Deriving
  `excluded_indices` from `submissions` meant that absent client's
  pairwise seed was never requested, so its uncancelled mask term would
  have silently corrupted the aggregate. Fixed by deriving it from the
  `n_clients` parameter instead (`[idx for idx in range(n_clients) if
  idx not in accepted_set]`) -- previously an unused parameter, now
  load-bearing. Regression test:
  `test_aggregate_is_exact_when_a_configured_client_never_submits_at_all`
  in `tests/test_round_end_to_end.py` -- a 9-configured-client round
  where the 9th never appears in `submissions` at all, asserting (a)
  its index is still in `excluded_idx` and (b) the unmasked aggregate
  is exact.

**Still open on F7, honestly:**
- **F7f attributes to a group, never an individual, by design** -- unchanged,
  and not something to fix; a single report can't implicate a member of a
  group without more independent evidence, which is the whole point of
  `SuspectTracker`'s intersection-narrowing design above.
- **F7f's bisection primitive now supports real protocol rounds; the
  simulation still doesn't use it that way (2026-11-14).**
  `crypto/norm_proof.py::BisectionSession` decomposes `localize_fault`'s
  loop into explicit `pending_group()` / `submit_result()` steps -- query a
  group, get an answer, decide the next group, repeat -- so a real
  deployment can suspend between rounds (each step = one real network round
  trip) instead of blocking in one Python call. `localize_fault` is now a
  thin synchronous wrapper around it, byte-for-byte unchanged behavior for
  every existing caller (9 new tests in `tests/test_norm_proof_bisection.py`
  confirm agreement with the old implementation on both clean and cheating
  sets, plus `pending_group` idempotency, `submit_result`'s group-match
  validation, and the `max_rounds`/`min_group_size` floor). `fl/server.py`'s
  `_check_F7e_aggregate_sketch` deliberately keeps calling the synchronous
  wrapper, not `BisectionSession` directly -- this simulation has no real
  transport layer to suspend across, so using the new primitive here would
  be decorative, not more correct. A real deployment is the one that needs
  `BisectionSession` directly, and now can.
- **`SuspectTracker`'s quarantine/ban state is not yet exercised on Colab,
  though a gate now exists to do it (2026-11-14).** The module itself was
  already fully unit-tested in isolation (`tests/test_suspect_tracker.py`).
  New this round: `fl/client.py::round_step_phase2` gained
  `_cheat_zero_sketch` (attack-simulation only -- zeroes `y_k` right after
  the real sketch is computed, so the real training/masking pipeline runs
  untouched and only the committed sketch is fake) and
  `fl/round_loop.py::run_training_loop` gained `cheater_client_idx`, which
  threads that flag to one designated client each round. The notebook's new
  Gate E (`notebooks/_build_notebooks.py`, 9 clients, 5 rounds, client 5
  cheating, `quarantine_rounds=1`) runs `SuspectTracker` through
  `run_training_loop` against this real cheater for the first time, rather
  than through `_check_F7e_aggregate_sketch` directly (what Gate D does).
  This is implemented and passing locally (178 tests total, pure-Python
  group backend) but genuinely unconfirmed until it runs on Colab with real
  `rbcl` -- same limitation as every other `run_training_loop`-dependent
  gate in this file.
- **All four gates run on real Colab GPU with real rbcl bindings
  (2026-11-14).** Gate A: full `pytest tests/ -q` -- 159 passed, 0 skipped,
  110.74s, confirming the entire suite (not just the F7-specific files)
  exercises real libsodium, not the pure-Python stand-in.
  `notebooks/VERIFY_FIXES_F6_F7.ipynb`'s real 8-client,
  2-round loop (resnet18) came back `accepted=8/8, V5=PASS, F7e=PASS,
  aggregate=EXACT` on both rounds, and Gate D's attack-demo subset
  (`-k "lying_sketch or wraparound_cheater"`) passed 4/4 on real rbcl. Measured
  per-round cost against the documented pre-F7 baseline
  (`phase1=45.8s, phase2=31.9s, verify=8.7s`):

  | phase | pre-F7 | round 0 | round 1 | factor |
  |---|---|---|---|---|
  | phase1 (train+most-crypto) | 45.8s | 62.9s | 84.8s | 1.4-1.85x |
  | phase2 (sketch+rangeproof) | 31.9s | 237.6s | 237.9s | ~7.45x |
  | verify | 8.7s | 121.0s | 122.9s | ~14x |

  phase2's ~7.45x lands right where FIX.md's own estimate said it would
  (~8x the old single range proof) -- the m=8192 sketch dimension does not
  need revisiting on that number. Two things weren't predicted and are worth
  tracking rather than explaining away: verify grew faster than phase2 did
  (~14x vs ~7.45x) even though both scale with roughly the same per-coordinate
  proof count, and phase1 -- which should be "roughly unchanged" since F7 adds
  no phase-1 work -- grew 1.4-1.85x on its own, with round 1 noticeably slower
  than round 0 within the same process (84.8s vs 62.9s). Neither has a root
  cause identified yet; both could be ordinary Colab/GPU run-to-run variance
  (shared tenancy, allocator warm-up) rather than anything F7-shaped, but
  that's a guess, not a finding. RSS grew modestly (1.68 -> 2.08 GiB) over the
  two rounds. **F7e/F7f are now validated end-to-end on real rbcl, not just
  the pure-Python backend** -- all four gates passed; the timing anomalies
  above are the only unresolved items from this pass.
- Depends on F1 (F7e needs the true aggregate) -- satisfied, F1 is fixed.
- **Gate E's first real run found a genuine correctness bug in F7f/SuspectTracker
  -- not a flaky run (2026-11-14 run, fixed 2026-11-15).** The 9-client, 5-round,
  real-cheater Colab run came back wrong in exactly the way accountability
  must never be wrong: across 4 rounds it banned clients **0, 1, 2, 6, 8** --
  all innocent -- and never isolated client 5, the actual cheater.

  **Root cause, traced through two independent bugs:**

  1. `fl/round_loop.py`'s `blame_material_provider` reconstructed a bisected
     subgroup's real partial aggregate (`partial_q`, needed to recompute what
     its sketch *should* be) using the MAIN round's self-mask shares, gated on
     `len(group) < share_t` where `share_t = (2*n_clients)//3+1` -- a
     super-majority chosen for the WHOLE round's dropout/collusion tolerance,
     not an accountability-specific choice. For n=9, `share_t=7`. Bisection's
     own floor (`min_group_size`, default 4) is smaller than that, so EVERY
     leaf group the bisection ever reached was below 7 -- `blame_material_provider`
     returned `None` for all of them, and `_group_opens_correctly`'s own stated
     policy ("never clear a group on missing evidence") then reported every
     one of them suspect, **regardless of whether the cheater was actually in
     it**. `SuspectTracker`'s cross-round intersection-narrowing assumed only
     the true cheater would keep showing up in narrowing reports; instead,
     EVERY client showed up in some always-suspect bucket every round, and
     whichever client's two random bisection partitions happened to overlap
     to a singleton first got banned -- pure chance, unrelated to guilt.
  2. Independently, `crypto/norm_proof.py::partition_group` only checked
     whether the CURRENT group was already at `min_group_size` before
     splitting -- not whether the resulting HALVES would land below it. A
     9-client group split into 4/5, both already at/below the floor, so this
     bug alone guaranteed every leaf was undersized even before bug #1's
     threshold mismatch is considered.

  Bug #1 means lowering `min_group_size` alone cannot fix this: the
  cryptographic floor for reconstructing a subgroup's real `partial_q` is set
  by whatever Shamir threshold its self-mask shares actually use, not by a
  bisection policy constant.

  **Fix, in two parts:**

  1. `crypto/norm_proof.py::partition_group` now refuses to split if EITHER
     resulting half would land below `min_group_size` (checks `len//2 <
     min_group_size` before splitting, not `len <= min_group_size` on the
     group about to be split) -- every group `localize_fault`/`BisectionSession`
     ever tests is now independently `>= min_group_size`, by construction.
  2. F7f's subgroup checks now use a SECOND, cryptographically independent,
     LOW-threshold Shamir sharing of each client's two secrets
     (`self_mask_secret_int` and `blinding_self_mask_secret_int`) --
     `fl/round_loop.py`'s new `f7f_share_threshold` parameter (default 4),
     producing `accountability_self_mask_shares_held` /
     `accountability_blinding_shares_held` via a SECOND call to the existing
     `share_self_mask_secret`/`share_blinding_self_mask_secret` functions
     (each draws a fresh random polynomial per call -- `crypto/field.py::
     ShamirPolynomial` -- so this is a genuinely separate sharing of the same
     secret, not a derivative of the main one). `blame_material_provider` now
     reconstructs using THESE shares and THIS threshold, never the main
     round's `share_t` -- the main round's dropout/collusion-resistance
     threshold is completely untouched by this change. `blame_material_provider`'s
     return contract widened from a 3-tuple to a 4-tuple -- `(partial_q,
     straddling_seeds, group_blinding_self_mask_shares, group_share_threshold)`
     -- so `_check_F7e_aggregate_sketch` always reconstructs the subgroup's
     blinding sum at the threshold the provider ACTUALLY used, rather than
     needing to independently guess/match it. `fl/server.py::run_round` and
     `_check_F7e_aggregate_sketch` both gained the matching `f7f_share_threshold`
     parameter (default 4), used as `localize_fault`'s `min_group_size`.

  **Checks.** Two new tests in `tests/test_norm_proof_bisection.py`:
  `test_partition_group_never_produces_a_child_below_the_floor` (exhaustive
  check over n=1..39, floor=1..9: every split `partition_group` returns has
  both halves `>= floor`) and `test_f7f_with_matched_threshold_never_bans_an_
  innocent_client` -- reproduces the Gate E failure's exact shape (9 clients,
  a persistent cheater, several independent rounds with fresh bisection
  seeds, a synthetic oracle that fails both "too small to verify" and
  "cheater present" the same way the real code conflates them) and asserts
  the core invariant Gate E violated: no round may ever ban anyone but the
  cheater. Pure logic, no real crypto, runs in milliseconds -- this is now a
  permanent regression test, not something that only gets caught on a Colab
  GPU run. `tests/test_f7e_aggregate.py` and `tests/test_round_end_to_end.py`'s
  own hand-built blame providers updated to the 4-tuple contract (both pass
  their own fixed threshold as the 4th element, unchanged behavior for
  them -- their N_CLIENTS/SHARE_T combinations never actually hit this bug,
  which is exactly why it took a realistic Colab-scale run, not a unit test,
  to surface it). Two localization tests (`test_f7f_localizes_the_cheater_to_
  a_small_group`, `test_f7f_localizes_the_lying_sketch_cheater`) now pass
  `f7f_share_threshold` explicitly, matching their own fixture's threshold,
  so the floor and the verifiable threshold stay in lockstep the way the fix
  requires. Full suite: 180 passed (178 + 2 new), pure-Python group backend.
  **CONFIRMED ON REAL COLAB (2026-11-15), real cheater, real rbcl crypto.**
  Gate E rerun, 9 clients / 5 rounds / client 5 cheating: client 5 appears in
  EVERY reported suspect group across all 5 rounds (`[1,7,5,0,2]`,
  `[5,4,8,6,1]`, `[5,3,4,1]`, `[4,5,1,3]`, `[2,0,7,5]`) -- no other client
  does -- and the tracker converges to `banned=[5], quarantined={}` by round
  4 (`PASS: cheater 5 was banned by round 4`). Localization also narrowed
  further than the earlier buggy run: single suspect groups of 4-5 members in
  3 bisection steps, versus the old run's 2-3 always-suspect groups covering
  all 9 clients. Gate B/C's real 8-client/2-round resnet18 loop and Gate D's
  4/4 attack-demo subset both still pass unaffected (`aggregate=EXACT` both
  rounds; `4 passed, 11 deselected`), confirming the fix didn't regress
  anything else F7e/F7f already had working. **Gate A rerun on real rbcl
  (2026-11-15): 180 passed, 0 skipped, 112.59s** -- the 2 new tests from this
  fix confirmed against real crypto, not just the local pure-Python backend.
  Nothing left open on this fix; F7f's group-not-individual attribution
  remains open by design, unchanged, not something to fix.

## Checkpoint/resume for multi-hour free-tier Colab runs (2026-11-15)

- **Context, not a correctness bug**: moving from Gate-E-scale smoke tests
  (8-9 clients, a handful of rounds) to a real first MVP-scoped result
  (`n_clients=50`, `sampling_rate=0.25` -> ~13 active clients/round,
  measured ~53s/client/round on resnet18) means a single run is hours
  long (~3.8h for 20 rounds, the scope-cut chosen to fit one free-tier
  Colab session -- see `notebooks/CASE1_IID_MNIST.ipynb`'s new markdown
  cell for the full time-budget math). `fl/round_loop.py::run_training_loop`
  had no way to save or resume progress -- a Colab disconnect at any
  point during a multi-hour run lost 100% of the already-paid-for GPU
  and crypto work, with no way to recover anything already computed.
- **Fix**: two new functions in `fl/round_loop.py`, `save_checkpoint(path,
  round_idx, w_t, threshold, history, suspect_tracker)` and
  `load_checkpoint(path)`, using `torch.save`/`torch.load` (picklable
  dataclasses alongside the `w_t` tensor, one file, no second
  serialization format needed). `run_training_loop` gained two new
  keyword-only params, both `None` by default so every existing call
  site (all four notebooks) is unaffected: `checkpoint_path` (if given,
  saves after EVERY round, inside the existing `finally` block so it
  fires on the normal path AND on every abort path -- aggregate
  mismatch, F7/F7e failure -- since `history` already has that round's
  entry by the time `finally` runs) and `resume_from` (the dict
  `load_checkpoint` returns; if given, restores `w_t`/`threshold`/
  `history`/`suspect_tracker` and resumes the loop at `round+1` instead
  of 0). The save itself writes to a `.tmp` path and `os.replace`s it
  over the real path, so a crash mid-save can never leave a corrupt,
  unreadable checkpoint -- the previous round's checkpoint (or none)
  survives until the new one fully lands.
- **Checks**: `python3 -m py_compile fl/round_loop.py` clean; full local
  pure-Python suite rerun after the change, unaffected (113 passed, 5
  skipped -- same as before this change, since no existing test calls
  `run_training_loop` directly; it's torch/EC-dependent like the rest of
  this module, exercised only via the notebooks on Colab per
  `STATUS.md`). Not yet exercised on a real multi-hour Colab run --
  flagging this as untested-on-real-infra rather than implying it's
  been proven to survive an actual disconnect.
- **Notes for the paper**: none -- this is scaffolding for producing a
  result, not part of the protocol itself; doesn't belong in the paper.

## Model choice corrected: tiny_cnn replaces resnet18 for the first real result (2026-11-15)

- **What was wrong**: the first real (non-smoke) Colab run of
  `notebooks/RUN_MVP_RESNET18_MNIST.ipynb` (resnet18, n_clients=50,
  20 rounds planned) showed `n_clients=50` accepted every round at
  ~85-87s/client, ~71-73 min/round -- not the ~13 active clients/round
  and ~11.5 min/round the scope-cut plan assumed. Root cause: this
  implementation's `sampling_rate` (`fl/client.py::round_step_phase1`,
  step 0/1, VRF self-sampling `b_k`) only decides whether a client
  trains on its real local data or contributes a zero delta -- it does
  NOT exempt non-selected clients from masking, sketching, range-proofs,
  or verification. Every one of n_clients pays the full crypto cost
  every round regardless of sampling_rate. (This is a real design
  property of the protocol as implemented, not a bug -- ARCHITECTURE.md
  never claimed otherwise; the plan's time-budget math simply assumed
  something the code doesn't do, and nobody checked `round_step_phase1`
  against that assumption before estimating. Flagging the assumption
  error here rather than silently revising the estimate.) Real cost:
  ~24h for 20 rounds at resnet18's 11.2M params, not ~3.8h.
- **Second finding, same investigation**: a literature check across
  recent privacy-preserving/anonymous FL papers (not plain-FedAvg
  scalability papers) -- specifically WhiteCloak (NDSS 2026, solving
  the same anonymous-accountability-in-secure-aggregation problem as
  this repo's F7f/SuspectTracker) and PQS-BFL (2025, post-quantum
  blockchain FL) -- found every one of them deliberately uses a SMALL
  model (19K-818K params across both papers' four+ configs), precisely
  so the measured per-round cost is the crypto's, not the model's.
  resnet18 (11.2M params) is ~500x bigger than WhiteCloak's MNIST CNN
  (19K params) and ~500x bigger than PQS-BFL's MNIST CNN (~21.8K
  params) -- two independent papers converging on the same ~20K-param
  scale for MNIST specifically. Since mask generation, sketching, and
  range proofs are all O(model_dim) (same reason `build_resnet18`'s own
  docstring gives for preferring it over resnet50), resnet18's size is
  the dominant driver of the ~72min/round cost, not an implementation
  inefficiency.
- **Fix**: `fl/models.py` gained `build_tiny_cnn(num_classes,
  in_channels)` -- conv(1->16,3x3)-GroupNorm-ReLU-maxpool ->
  conv(16->32,3x3)-GroupNorm-ReLU-maxpool -> flatten -> linear, ~20.6K
  params for MNIST (num_classes=10), landing directly between
  WhiteCloak's 19K and PQS-BFL's 21.8K -- sized to match the cited
  precedent, not arbitrary. Added to `MVP_MODEL_BUILDERS` as
  `"tiny_cnn"` (distinct from the existing `build_tiny_debug_model`,
  which is explicitly NOT a real backbone and stays excluded from
  `MVP_MODEL_BUILDERS` -- this is a real model meant to appear in the
  paper). `notebooks/RUN_MVP_RESNET18_MNIST.ipynb` renamed to
  `RUN_MVP_TINY_CNN_MNIST.ipynb` and its model-build cell switched from
  `REAL_MODEL = 'resnet18'` to `REAL_MODEL = 'tiny_cnn'`; every other
  cell (data load, ring/DKG, checkpointing, round count) is unchanged,
  since `REAL_MODEL` is read generically throughout.
- **Checks**: `python3 -m py_compile fl/models.py` clean; full local
  pure-Python suite rerun, unaffected (113 passed, 5 skipped -- same as
  before, `build_tiny_cnn` is TORCH-DEPENDENT like every other model
  builder in this file and not exercised in this sandbox). Not yet run
  on real Colab -- flagging as untested-on-real-infra, not proven.
- **Notes for the paper**: the model-scale choice (and its citation
  anchor: WhiteCloak NDSS'26, PQS-BFL 2025, both landing at ~19-22K
  params on MNIST independently) belongs in the Evaluation Setup
  section as the justification for not using a larger backbone.

## Second backbone: small_resnet replaces mobilenet_v2 (2026-11-15)

- **Why**: same reasoning as the resnet18 -> tiny_cnn swap above.
  Standard `mobilenet_v2` with a 10-class head is ~2-2.5M params --
  8-35x bigger than any model this paper's actual peer group reports,
  and no comparable paper (WhiteCloak, PQS-BFL, AFLBCRS, AIFL) uses
  mobilenet_v2 at all, so there was no literature precedent to defend
  it with regardless of size.
- **Fix**: `fl/models.py` gains `build_small_resnet` (~263K params for
  MNIST/10-class) -- a hand-built residual network (3 residual blocks,
  GroupNorm, 1x1 projection on the skip path where channels change,
  global-average-pool head; torchvision has no "tiny ResNet" preset at
  this scale, so this is built directly with `nn.Module`, not adapted
  from a torchvision model like every other builder in this file).
  Anchored specifically to WhiteCloak's OWN second reported model
  (CIFAR-10-L: a ResNet at 273K parameters) -- landing within ~4% of
  that number, not an unrelated size. Combined with `build_tiny_cnn`
  (~20.6K, anchored to WhiteCloak's MNIST CNN at 19K), this repo's two
  MVP backbones now mirror WhiteCloak's own two reported model sizes
  end to end, rather than defending two separately-chosen sizes.
  `MVP_MODEL_BUILDERS` updated: `"mobilenet_v2"` removed,
  `"small_resnet"` added (`build_mobilenet_v2` itself is untouched,
  just no longer registered -- same treatment `build_resnet50` already
  gets as a kept-but-unused later ablation). `configs/mvp_3day.yaml`'s
  `mvp_scope.models` updated to `[tiny_cnn, small_resnet]`.
- **Checks**: `python3 -m py_compile fl/models.py` clean; `yaml.safe_load`
  on the config confirms it parses; `fl.models.MVP_MODEL_BUILDERS` keys
  confirmed as `['resnet18', 'resnet50', 'tiny_cnn', 'small_resnet']`
  (import-only check -- `_require_torch()` isn't triggered by importing
  the dict, only by calling a builder, so this doesn't itself prove the
  forward pass runs). Full local pure-Python suite unaffected: 113
  passed, 5 skipped. Not yet run on real Colab -- `build_small_resnet`
  is TORCH-DEPENDENT like every other builder in this file.
- **Notes for the paper**: same note as the tiny_cnn entry above --
  this model-scale choice and its citation anchor belong in the
  Evaluation Setup section.

## F8 — ristretto255 bindings: wrong library entirely, not a version problem

**Status: FIXED, CONFIRMED ON COLAB (2026-11-13).** `rbcl` imports and its
group operations run correctly against real libsodium: a full round (ring
keys, 10-holder DKG, PK_trace, CLSAG signing, Pedersen commitments, the
sketch, range proof) completed with `V5=PASS, aggregate=EXACT` at 30
clients (see F5, STATUS.md) -- that round could not have started at all if
`rbcl`'s bindings were broken. Originally landed as sandbox-tested-only
(the pure-math half, below); the rest is no longer untested.

**Cause.** `crypto/ristretto.py` called `nacl.bindings.crypto_core_ristretto255_*`,
on the assumption (stated in this module's own old docstring, and in
`notebooks/_build_notebooks.py`'s install cell) that PyNaCl >= 1.5.0 exposes
ristretto255. **It does not, under any released version.** Mainline PyNaCl
has never merged ristretto255 support — the PR adding it
(`github.com/pyca/pynacl/pull/646`) is still open and unmerged upstream, as
of this writing. The earlier entries in this log and in STATUS.md that
diagnosed the resulting `AttributeError` as "stale pynacl, needs
`--upgrade --force-reinstall`, then a kernel restart" were wrong — upgrading
pynacl to any version, including the 1.6.2 actually installed on the Colab
run that surfaced this, can never produce
`nacl.bindings.crypto_core_ristretto255_from_hash`, because that function
has never existed in that package.

**Evidence.** Confirmed three ways: (1) PyNaCl's own published changelog
(`pynacl.readthedocs.io/en/latest/changelog/`) lists no ristretto255 entry
at any version, 1.0 through 1.6.2; (2) the only PyNaCl fork that adds it
(`joernheissler/pynacl`, branch `ristretto`) is a separate, unmerged fork,
not what `pip install pynacl` gives you; (3) the user's own Colab run
reported `pynacl version: 1.6.2` — genuinely new, passing this module's old
`>= 1.5.0` check — immediately followed by the exact
`crypto_core_ristretto255_from_hash` `AttributeError` this fix resolves,
which is only possible if the function is absent from the library itself,
not from a stale copy of it.

**Change.** Swapped the backend from `nacl.bindings` to `rbcl`
(`pypi.org/project/rbcl`, maintained by Nth Party) — a package built
specifically to wrap libsodium's ristretto255 group operations, with
published manylinux2014 x86_64 wheels (confirmed on PyPI), so it installs
on Colab with a plain `pip install rbcl`, no source build. Every function
`crypto/ristretto.py` calls maps onto an rbcl function with the same
argument order (`crypto_scalarmult_ristretto255(n, p)` matches this
module's existing `(scalar, point)` call convention; checked against
rbcl's actual source, not assumed from its README) — except one:

**The one real incompatibility, and how it's handled.** rbcl's Python
wrapper for `crypto_core_ristretto255_scalar_reduce` validates its input
is exactly 32 bytes (`SCALARBYTES`), even though its own docstring (and
libsodium's actual C API) describe a 64-byte ("non-reduced scalar")
input for this exact operation — confirmed by reading rbcl's source
directly, not its docs, since the two disagree. `Scalar.from_int` and
`fiat_shamir_challenge` both need the 64-byte reduction (that's the whole
point of hashing into a scalar). Fixed by doing that one reduction in
plain Python instead of through rbcl: `crypto/ristretto.py::_wide_scalar_reduce`
computes `int.from_bytes(data, "little") % L` with `L` the standard,
published ristretto255/ed25519 group order
(`2**252 + 27742317777372353535851937790883648493`, RFC 8032 §5.1) and
encodes the result as a canonical 32-byte little-endian scalar — this is
not a placeholder or an approximation, it is libsodium's own definition
of that operation, computed directly instead of through a wrapper that
can't take the right input size for it.

**Checks.** New `tests/test_ristretto_pure.py` (11 tests) covers
`_wide_scalar_reduce` and `Scalar.from_int` — the only two pieces of this
module that don't touch rbcl and are therefore actually runnable in this
network-less sandbox: group-order value against RFC 8032, reduction of
0/L/L+1, 200 random 64-byte values checked against a direct `% L`,
and `Scalar.from_int(0)` landing on the exact zero-sentinel byte string
`scalar_mul`/`scalar_mul_base` short-circuit on. All 15 repo test files
pass (89 tests total). **Honestly incomplete**: every rbcl-backed group
operation itself (`crypto_core_ristretto255_add`, `_random`, `_from_hash`,
`crypto_scalarmult_ristretto255[_base]`, `_is_valid_point`, and the other
four scalar ops) is still untested here — this sandbox has no network to
install rbcl, so none of it has actually executed yet on real libsodium.
Everything built on `crypto/ristretto.py` (`pedersen.py`, `clsag.py`,
`elgamal_threshold.py`, `dkg.py`, `vrf.py`, `mixnet.py`, `rangeproof.py`,
`r_cons.py`) remains exactly as unverified as before this fix — this fix
makes the import actually succeed on Colab, it does not newly verify the
math those modules do with the resulting group elements.

**Also changed**: `notebooks/_build_notebooks.py`'s `SETUP_CELLS` (and the
regenerated `notebooks/00_Setup_and_Crypto_Tests.ipynb`) now install and
check `rbcl` instead of upgrading `pynacl` for this; `pynacl` itself stays
installed, since `secagg.py`'s ECDH helper genuinely needs its (unrelated,
correctly-supported) X25519 bindings. `notebooks/VERIFY_FIXES_F1_F2_F4_F5.ipynb`
updated to match. **Not yet run on Colab** — the actual import of `rbcl`
and every group operation built on it needs that run before any of F1/F2/
F4/F5's Colab confirmation can proceed again, since all of them sit on top
of this module.

## F9 — DP noise scaled per-aggregate, applied per-client (ML-validity, not crypto-correctness)

**Symptom (real Colab run, 2026-11-15).** `tiny_cnn` on MNIST, `n_clients=50`,
`sampling_rate=0.25`, 12 real rounds completed. Every single round reported
clean crypto: `accepted=50/50, V5=PASS, F7e=PASS, aggregate=EXACT`. Despite
that, evaluating the resulting checkpoint gave:

```
tiny_cnn @ round 11: official MNIST test accuracy = 0.1538 (10000 images)
tiny_cnn @ round 11: held-out global_val accuracy = 0.1669 (2499 images)
```

~15% on a 10-class problem is barely above the 10% random-guess floor —
after 12 rounds of real FedAvg-style training this should already be well
above 90%. Since every crypto check passed every round, the bug had to be
in what the checks were validating, not in whether they ran.

**Root cause.** `fl/client.py::round_step_phase1`, Step 2 (DP noise share):

```python
noise_shares = split_noise(sigma_dp, n_shares=1, dim=ctx.model_dim, rng=np.random.default_rng())
nu_k = noise_shares[0]
```

`ARCHITECTURE.md` §4 step 2 specifies the per-client share as
`ν_k ← σ_DP / √(p·n)` (p = sampling_rate, n = n_clients) — i.e. each
client's own noise draw is supposed to already be shrunk by `√(p·n)`, so
that once the ~p·n sampled clients' shares are summed during aggregation,
the *aggregate* injected noise lands back at the configured `σ_DP`.

The code instead called `split_noise(sigma_dp, n_shares=1, ...)`.
`crypto/dp.py::split_noise`'s own contract (`per_share_sigma = total_sigma
/ √n_shares`) means `n_shares=1` applies no scaling at all — every client
added a *full* `σ_DP=2.5`-std draw per coordinate, independently, every
round, regardless of whether it was VRF-sampled to train on real data in
Step 1. Summing that over `n_clients=50` accepted submissions before the
server's FedAvg mean-division inflates the aggregate noise to roughly
`√50 ≈ 7.07x` the intended `σ_DP` — a per-coordinate noise std around the
2.5 raw / ~0.35 post-mean-division range, which is comparable to or larger
than real per-coordinate gradient magnitudes (`quantize()`'s own docstring
puts typical deltas in the ~1e-2 to 1 range). That is consistent with the
observed near-random accuracy: the signal was being drowned by noise, not
corrupted or dropped by the crypto pipeline.

This is an **ML-validity bug**, not a crypto-correctness bug — same
category as F6 (local-training realism). `tests/test_dp.py` only exercises
`split_noise` and `RDPAccountant` directly and confirms `split_noise`
itself behaves exactly as documented; `round_step_phase1` (where the bug
actually lived) is torch-dependent and has never been exercised by the
local pure-Python test suite, which is why 113 passing local tests never
caught this — it required a real multi-round accuracy evaluation on Colab
to surface.

**Fix.** Compute the correctly-scaled per-client sigma from
`ARCHITECTURE.md`'s own formula before calling `split_noise`:

```python
per_client_sigma = sigma_dp / np.sqrt(max(sampling_rate * n_clients, 1e-9))
noise_shares = split_noise(per_client_sigma, n_shares=1, dim=ctx.model_dim, rng=np.random.default_rng())
nu_k = noise_shares[0]
```

`sampling_rate` and `n_clients` were already in-scope parameters of
`round_step_phase1` — no new plumbing needed. The `max(..., 1e-9)` guards
against division by zero if either is misconfigured to 0.

**Checks.** Full local suite re-run after the fix: 113 passed, 5 skipped —
unchanged, because (as above) this code path isn't reachable from the
local, torch-free test suite. No regression possible to detect locally;
the real confirmation has to come from a fresh Colab run.

**Honestly incomplete / notes for the paper.** This fix has **not yet been
confirmed on Colab** — that is the immediate next step. The first 12
rounds of the run that surfaced this bug trained under the *wrong*,
over-noised DP mechanism; that checkpoint is likely not meaningfully
better than random initialization and should not be treated as a valid
partial result or cited in any figure. The paper's privacy-utility
tradeoff numbers (accuracy vs. ε) must be collected only from a run
started fresh with this fix in place, not resumed from the buggy
checkpoint. The fix does not change `crypto/dp.py::RDPAccountant`'s
ε accounting math — only how `sigma_dp` is divided between clients before
the noise is drawn — so privacy budget numbers already reported are
unaffected; only the realized model utility at a given budget was wrong.

## F10 — attacks/a1_scaling.py never applied the mod-wrap fix the rest of the pipeline already had

**Found while wiring the first-ever real run of `attacks/`'s 5 attack modules** (all were "written, not run anywhere" per STATUS.md -- no test, no notebook had ever imported them). A dry run against the pure-Python EC test backend (`JANUS_ALLOW_PURE_PYTHON_GROUP=1`) immediately produced a nonsensical result: both an honest-sized toy delta and a deliberately `gamma=100`-scaled malicious delta came back with `estimated_norm` in the same ~1e19-1e20 range, indistinguishable from each other regardless of true magnitude.

**Root cause.** `attacks/a1_scaling.py::run_attack` computed the norm directly from `quantize()`'s own output: `estimated_norm = int(np.linalg.norm(q.astype(np.float64)))`. `fl/client.py::quantize` reduces every entry `mod P_MASK` (`P_MASK ≈ 2.3e18`, a 61-bit prime), so any negative-valued coordinate wraps to a value near `P_MASK` — a naive norm over the wrapped array is dominated by that wraparound noise regardless of how small the real delta is. This is the exact same mod-wrap bug class STATUS.md already documents being found and fixed in `server.py`'s V3 check and the B_t-EMA sketch path, where `center_mod_p_mask` was added specifically to undo it before any norm is taken — `a1_scaling.py` was simply never updated to match, because nothing had ever run it to notice.

**Fix.** `center_mod_p_mask(q)` before taking the norm, matching what the real client/server pipeline already does:
```python
q = quantize(malicious_delta)
estimated_norm = int(np.linalg.norm(center_mod_p_mask(q).astype(np.float64)))
```
Confirmed fixed by the same dry run: with centering, a `dim=2000` toy honest-delta bootstrap gives `B_t≈147,423`, and the `gamma=100` attack's centered norm comes back at `≈410,119,348` — now correctly, enormously over `B_t`, and `V3_norm_bound`'s gate rejects it as intended.

**Checks.** Full local suite re-run: 113 passed, 5 skipped, unchanged. A full dry run of all 5 attack modules' 8 `run_attack*` variants (A1, A3, A4x2, A5x2, A7x2) against the pure-Python EC backend: all 8 `defense_triggered=True`. This is NOT a substitute for the real `rbcl`/libsodium run on Colab (the pure-Python backend is explicitly test-only, same caveat as every other EC-dependent module in this repo), but it does catch exactly this kind of wiring/logic bug before burning Colab time on it.

**New deliverable**: `notebooks/RUN_ATTACKS.ipynb` (generated by `notebooks/_build_attacks_notebook.py`, same reproducible-from-plain-Python-strings pattern as `_build_notebooks.py`) runs all 5 attacks' 8 variants against real `rbcl` crypto in one pass -- standalone from the MNIST training run (each attack exercises one crypto primitive/defense pair directly via a small ring of keys + a DKG committee, not the multi-round training loop), so it runs in minutes, not hours. Ends with a pandas summary table of attack x defense x caught/not-caught -- ARCHITECTURE.md §10 calls this table "the single most convincing figure in the paper" -- and an optional cell to save it as a CSV to Drive.

**CONFIRMED on real Colab with real `rbcl`, 2026-11-15**: all 8/8 variants caught by their target defense, output identical in shape to the pure-Python dry run above:

```
attack_name              defense_name            status
A1_scaling               V3_norm_bound           CAUGHT
A3_sybil                 V2_key_image_dedup      CAUGHT
A4_freeride_replay       rho_t_binding_V1        CAUGHT
A4_freeride_false_claim  R_cons_VRF_disjunction  CAUGHT
A5_splice                V4_r_cons_and_binding   CAUGHT
A5_splice_key_image      V1_message_binding      CAUGHT
A7_collusion             t_of_nc_threshold       CAUGHT
A7_collusion_mixnet      3hop_mix_cascade        CAUGHT
```

**Important caveat for the paper, not yet addressed**: this confirms each defense primitive works correctly in isolation (a ring of keys + a DKG committee, hand-crafted adversarial inputs, no honest co-participants, no real training round around it) -- it is NOT the same claim as "JANUS-FL rejects this attack during real federated training," which is what a reader would reasonably take "the single most convincing figure in the paper" (ARCHITECTURE.md §10) to mean. A1/A3/A4/A5 are per-round attacks and should additionally be re-run as a real attacker injected into an actual `run_training_loop` call alongside honest clients, the same pattern already proven once for the F7e cheater (Gate D/E, `VERIFY_FIXES_F6_F7.ipynb`: `cheater_client_idx` through several real rounds, `SuspectTracker` converging on a ban). A7 is the one exception -- it is not a per-round attack at all (it targets `fl/trace.py`'s out-of-band identity-opening protocol, which runs on accumulated evidence, not inside a training round), so standalone testing is actually the correct shape for it. This integrated re-run of A1/A3/A4/A5 is the next step, not yet started.

## F11 — A1 wired into a real round, not just tested in isolation

**Why**: after F10 confirmed all 8 standalone attack variants on real Colab/rbcl, a fair question came back: those tests prove each defense primitive correct in isolation (a ring of keys, a DKG committee, hand-crafted inputs, no honest co-participants, no real round around it) -- not the stronger claim "JANUS-FL rejects this attack during real federated training," which is what ARCHITECTURE.md §10's "single most convincing figure" is actually meant to show. This closes that gap for A1 (model-replacement scaling), the most natural one to re-run this way.

**What changed.** Two small, explicitly-scoped ATTACK-SIMULATION-ONLY additions, same pattern as the existing `cheater_client_idx`/`_cheat_zero_sketch` hooks (FIX.md F7e/F7f):
- `fl/client.py::round_step_phase1` gained `_force_delta: np.ndarray | None = None` -- when given, overrides whatever Step 1 computed (real trained delta, or the zero vector) with this array, AFTER the real VRF-sampling decision, so everything downstream (DP noise, quantize, masking, sketching, the real range proof) still runs exactly as it would for an honest client's delta.
- `fl/round_loop.py::run_training_loop` gained `cheater_force_delta: np.ndarray | None = None`, which only takes effect when `cheater_client_idx` is also set -- that client's `round_step_phase1` call gets `_force_delta=cheater_force_delta`.
- The non-aborted branch's `history.append(...)` dict also gained `accepted_client_indices` and `rejected_reasons` (previously only logged on the ABORTED branch) -- needed to see a per-round V3 rejection without a round-wide abort, since V3 excluding one client from the accepted set does NOT abort the round for everyone else.

**New Gate, in `notebooks/RUN_ATTACKS.ipynb`** (appended after F10's standalone results): a real `tiny_cnn`/MNIST `run_training_loop` call, 9 clients (k_min=8 + 1, same reasoning as the F7e Gate E), 3 rounds, one client's delta forced to `attacks.a1_scaling.craft_scaled_update`'s `gamma*(w_target - w_t)` output every round. Expected result: that client's index appears in `rejected_reasons` every round it's eligible, is absent from `accepted_client_indices`, and the round still completes normally (`aggregate_exact=True`) for the honest majority -- i.e. V3 catches the attacker without taking the round down with it.

**Honestly incomplete.** Only A1 is wired this way so far. A3 (Sybil) and A4-replay are genuinely per-round attacks too and belong in a future pass the same way. A4-false-claim and A5 (proof-splicing) are forgery tests the honest pipeline has no code path to produce on its own -- there's no "lie about my own VRF output" branch to inject into round_step_phase1/2, so the F10 isolated result is already the right shape for those, not a placeholder for something stronger. A7 is correctly out-of-band (`fl/trace.py`'s opening protocol runs on accumulated evidence between rounds, never inside one), so isolated testing is also already correct for it, not a gap.

**Checks.** Full local suite re-run: 113 passed, 5 skipped, unchanged (`round_step_phase1`/`run_training_loop` are torch-dependent, not exercised locally, same limitation as every prior change to these files). **Not yet run on Colab** -- this gate needs a real GPU runtime like every other `run_training_loop`-dependent check in this repo.

## F11 follow-up — the real result, after fixing a wiring bug that masked it

**The anomaly**: the first real-Colab run of the F11 gate (above) showed client 5 (the designated cheater) with `sketch_commitment.v = 0` exactly, and the round aborted via F7e's aggregate-sketch mismatch (all 8192 coordinates), not via V3. That looked like V3 failing to catch a ~20x-oversized update -- confirmed by direct instrumentation (temporary `JANUS_DEBUG_V3`-gated prints in `crypto/norm_proof.py`, `fl/client.py`, `fl/server.py`, added and removed in this same session) that `_force_delta` DID apply (delta norm ≈14,310, matching `craft_scaled_update`'s output) but `sketch_commitment.v` came out exactly 0 regardless.

**Root cause**: `fl/round_loop.py`'s phase-2 call site had `_cheat_zero_sketch=(client_idx == cheater_client_idx)` -- unconditional on `cheater_client_idx`, left over from the original F7e/F7f "lying zero sketch" demo (Gate D/E) this parameter was built for. Adding `cheater_force_delta` for this gate without updating that line meant client 5 got BOTH cheats stacked: phase 1's genuinely oversized masked delta (from `_force_delta`), AND phase 2's hard override to sketch the zero vector (from the old, unconditional `_cheat_zero_sketch`). The sketch commitment's `v` is computed from whatever gets sketched, so it was always going to be 0, independent of `cheater_force_delta`'s magnitude -- V3 was never actually exercised; the round only ever failed via F7e's "submitted sketch doesn't match the real masked vector" cross-check. A real finding, just not the one this gate was built to produce.

**Fix**: `_cheat_zero_sketch=(client_idx == cheater_client_idx and cheater_force_delta is None)` -- the two cheater mechanisms are independent now; a caller gets EITHER the F7e zero-sketch lie (Gate D/E, unchanged) OR an honestly-sketched A1 attack (this gate), never both from one `cheater_client_idx` assignment.

**Re-run after the fix, real Colab, 2026-11-16**: client 5's `sketch_commitment.v = 2,440,248,766,618,814,583,986` against `norm_bound_sq = 6,460,287,050,031,617,024` -- ratio 377.7, correctly outside the bound (`in_bound=False`). Round 0 outcome: `accepted=8/9, V5=PASS, F7e=PASS, aggregate=EXACT, next B_t=38,868,291.8`. This is the real result: **V3 alone rejects a model-replacement-scale attack when it is honestly (not additionally lyingly-)sketched, and the round completes normally for the other 8 honest clients** -- the claim this gate was built to test in the first place, now actually demonstrated rather than short-circuited by an unrelated wiring bug.

**Lesson for future cheater-simulation params**: `cheater_client_idx` is now host to two independent, mutually-exclusive attack mechanisms (`_cheat_zero_sketch` for F7e/F7f, `cheater_force_delta` for A1/V3). Any THIRD mechanism added later under the same `cheater_client_idx` must be checked against both existing ones for this same silent-composition failure mode before trusting a result.

**Checks**: 113 passed, 5 skipped locally throughout (none of this is torch-exercisable off Colab). All temporary debug prints added mid-investigation have been removed from `crypto/norm_proof.py`, `fl/client.py`, `fl/server.py`.

**Final clean 3-round confirmation, real Colab, 2026-11-16** (fresh runtime, no debug instrumentation):

```
round 0: accepted=8, cheater rejected=True (V3_norm_bound_exceeded, v=2437103467890242433137, norm_bound_sq=6460287050031617024, B_t=48639775.669), aggregate_exact=True
round 1: accepted=8, cheater rejected=True (V3_norm_bound_exceeded, v=2377761683253871082067, norm_bound_sq=4137176608505554944, B_t=38924019.890), aggregate_exact=True
round 2: accepted=8, cheater rejected=True (V3_norm_bound_exceeded, v=2338202999870878286652, norm_bound_sq=2792888172045597696, B_t=31981028.686), aggregate_exact=True

PASS: cheater 5's scaled update was rejected by V3 every round it was eligible, while the round still completed normally for honest clients.
```

All three rounds: V3 alone (no F7e involvement) rejects the model-replacement attack; `accepted=8/9`, `V5=PASS`, `F7e=PASS`, `aggregate=EXACT` every round; `B_t` tightens round-over-round (48.6M -> 38.9M -> 32.0M) as the honest clients' real training converges, exactly as `update_threshold_ema` is meant to do. **F11 closed**: A1 is now confirmed caught by V3 inside a real multi-round `tiny_cnn`/MNIST training loop, not just in isolation.

## F12 — multi-cheater generalization + trace.py's first real run (2026-11-16)

**Context**: after locking in the experimental design (LEFT.md #1.5/#1.6
-- 7 malicious clients for the fused run, matching ARCHITECTURE.md's own
N7 target, 10%/20%/30% grid for the attack-type sweep), `cheater_client_idx`
needed to support more than one cheater at once; it was a bare int.

**Fix**: `fl/round_loop.py::run_training_loop` normalizes `cheater_client_idx`
to a set once, at the top of the function (`None` -> empty set, a bare int
-> one-element set, a list/set -> itself). Every downstream check
(`_force_delta`'s and `_cheat_zero_sketch`'s conditionals, the per-round
verbose eligibility print) became a set-membership test instead of an
equality check. A bare int still works exactly as before -- every existing
call site (F11's gate, VERIFY_FIXES_F6_F7's Gate D/E) is unaffected.
113 passed, 5 skipped, unchanged. Used immediately by the new
`RUN_MVP_TINY_CNN_MNIST_WITH_ATTACK.ipynb` (7 of 50 clients malicious,
14%, same crafted delta on all 7 -- noted as a simplifying choice, not
independently-drawn attacks per client).

**Separately, same session**: `fl/trace.py` -- the actual "merge t-of-n_c
keys to open a real identity" orchestration -- had never been run (STATUS.md
had carried it as "WRITTEN, NOT RUN" since it was first written). Added a
gate to `RUN_ATTACKS.ipynb`, right after the A1 real-round gate, that runs
the real `OpeningRequest`/`collect_partial_from_holder`/`run_opening_protocol`
flow against the SAME client A1 just caught cheating: 7 of the 10 named
key holders correctly recover its real public key (checked against the
real ring key, not just "didn't crash"); 6 of the 10, same real function,
correctly fail with "need t=7". Step 5 (R_cons re-binding, needs a live
round's proof object) and `evidence_total`/`theta` (fl/evidence.py's own
calibration) are explicitly scoped out in the gate's own markdown, not
silently skipped.

Before committing, ran this gate's exact logic in this sandbox against
`crypto/_ristretto_py.py`'s test-only pure-Python ristretto255 backend
(`JANUS_ALLOW_PURE_PYTHON_GROUP=1`) -- confirms the control-flow and
object-wiring is correct (7-of-10 opens and names the right client,
6-of-10 correctly refuses) independent of the real `rbcl` backend this
sandbox can't install. The real Colab run with `rbcl` is still what
closes this per STATUS.md's standard for every other EC-dependent module
-- not done yet, flagged in STATUS.md as "control-flow verified in
sandbox, not yet real-Colab-confirmed."

Also added, user-requested: an explicit `MALICIOUS CLIENTS IN THIS RUN:
[...]` banner printed immediately before `run_training_loop` is called,
in both the A1 gate and the new fused 7-attacker notebook -- so a
reviewer reading the notebook's own output sees which clients are
malicious stated plainly, not left implicit in a variable assigned
earlier in the file.

## F13 — Drive dependency removed from real-training notebooks, git-backed checkpointing (2026-11-16)

**Context**: user can't mount Google Drive this session (personal
reasons), which both real-training notebooks (`RUN_MVP_TINY_CNN_MNIST.ipynb`,
its `_WITH_ATTACK.ipynb` companion) depended on for checkpoint
durability across a Colab disconnect.

**Fix**: `fl/round_loop.py::run_training_loop` gained an
`on_checkpoint_saved(checkpoint_path, round_idx)` callback, fired right
after every per-round `save_checkpoint()` call -- the library stays
git-agnostic (no subprocess/git import here), the notebooks supply their
own persistence. Both notebooks now clone a SEPARATE checkout
(`/content/FL-ZKP-checkpoints`, never the code checkout the notebook
imports from) onto a dedicated `colab-checkpoints` branch (never `main`
-- checkpoints are run artifacts, not paper code, and don't belong in
the history a reviewer clones), and commit+push the checkpoint file
there after every round. A GitHub PAT is entered via `getpass` (hidden,
never written to disk or committed). A push failure only costs a retry
next round -- it's wrapped so it never raises and never stops training.

Checkpoint size was checked first, not assumed: `tiny_cnn` is ~20.6K
params, so a checkpoint is well under 1MB -- the usual argument against
committing checkpoints into git (binary-blob repo bloat) doesn't bite
at this model's scale, which is why this is git and not, say, git-lfs.

**Verified the resume mechanism itself before handing it to the user**,
against a real throwaway local git remote (not the actual FL-ZKP repo):
session 1 pushes 3 round-checkpoints to a fresh `colab-checkpoints`
branch; the local checkout is then deleted (simulating exactly what a
real Colab disconnect does to `/content`); session 2 re-clones, detects
the branch already exists on the remote, fetches it, and recovers the
exact checkpoint content session 1 last pushed -- then pushes 3 more
rounds on top without error. This caught one real bug before it shipped:
a plain `git fetch origin <branch>` only updates `FETCH_HEAD`, not
`refs/remotes/origin/<branch>`, so the resume-detection `checkout -B`
step failed even though the fetch itself reported success. Fixed with
an explicit destination refspec (`<branch>:refs/remotes/origin/<branch>`).

113 passed, 5 skipped, unchanged.

## F14 — DP noise still added unconditionally to dummy clients, residual 2x aggregate overshoot (2026-11-16)

**Context**: the first real 20-round CASE1 baseline run (git-based
checkpointing from F13, author identity Yashaswee Narayan) completed
cleanly on every security check -- accepted=50/50 every round, V5=PASS,
F7e=PASS, aggregate=EXACT -- but came back with MNIST test accuracy of
only 33.34% (round 19), vs. the 90%+ expected for `tiny_cnn` on MNIST at
this FedAvg config. Since every crypto/consistency check passed, the bug
had to be in what gets aggregated, not in the aggregation machinery --
same category as F6 and F9.

**Root cause**: `fl/client.py` Step 2's noise-share draw (the F9 fix
site: `per_client_sigma = sigma_dp / sqrt(p*n)`, then
`split_noise(per_client_sigma, n_shares=1, ...)`) ran *unconditionally*,
for every one of the `n_clients=50` clients each round, regardless of
`b_k` (the Step 0/Step 1 VRF-sampling flag that already gates whether a
client trains or contributes an all-zero delta). `ARCHITECTURE.md` line
201 is explicit that this is wrong: "all 50 clients submit; ~38 are
dummies with q_k = 0" -- a dummy (unsampled) client's final quantized
value must be *exactly* zero, noise included, not zero-delta-plus-a-
full-noise-draw. `fl/server.py::aggregate_masked_vector` sums over every
accepted submission (confirmed 50/50 accepted in the real run), so all
~38 dummy clients' independent noise draws were entering the aggregate
sum too.

F9 (2026-11-15) had already fixed the *scale* of each individual
client's noise share (sigma_dp/sqrt(p*n) instead of raw sigma_dp),
correctly sized for summing over the ~p*n=12.5 *sampled* clients the
spec intends -- but with noise added by all 50 instead, the realized
aggregate variance is `n * (sigma_dp^2/(p*n)) = (n/(p*n)) * sigma_dp^2
= (1/p) * sigma_dp^2`, i.e. aggregate std = `sigma_dp / sqrt(p)` =
`sigma_dp * sqrt(1/0.25)` = `2 * sigma_dp` = 5.0 instead of the intended
2.5 -- a residual 2x overshoot left over after F9 only partially closed
the original ~7.07x bug.

**Fix**: gated Step 2 on the same `b_k` Step 1 already computes --
sampled clients (`b_k=True`) draw their scaled noise share exactly as
before; dummy clients (`b_k=False`) get `nu_k = np.zeros(ctx.model_dim)`
exactly, so `q_k = quantize(0 + 0) = 0` as the spec requires, and only
the ~p*n genuinely-sampled clients' noise enters the aggregate sum.

**Verified with a standalone numpy-only Monte Carlo sanity check**
(p=0.25, n=50, sigma_dp=2.5, dim=10000, 200 trials, independent of the
actual pipeline code) before treating this as resolved: old
(unconditional) behavior measured aggregate noise std ~4.9967 (predicted
5.0 = 2x target); new (gated) behavior measured ~2.4286 (predicted 2.5 =
target). Confirms the fix, doesn't just assert it.

**Consequence for the paper**: the completed 20-round baseline run's
*accuracy* numbers (33.34% test / 33.61% held-out) are invalidated by
this bug and must be re-run with the fix in place to get a valid number
for the paper. The same run's *security/crypto* results (accepted=50/50,
V5/F7e/aggregate all passing every round) are unaffected by this bug and
remain valid -- this bug lives entirely in the ML-validity layer (same
category as F6/F9), not the cryptographic-correctness layer.

113 passed, 5 skipped, unchanged.

## F15 — model training ran entirely on CPU even on a GPU Colab runtime (2026-11-16)

**Context**: asked directly whether the pipeline is GPU-optimized "at
its peak." Checked by direct code inspection rather than assuming --
the answer was no, and not just "not fully optimized": the actual
model training loop was never placed on GPU at all.

**Root cause**: `fl/models.py::build_model` returns a plain `nn.Module`
on CPU; no call site anywhere in `fl/` or any notebook ever moved it,
or the per-batch training/eval tensors, onto CUDA. `fl/client.py::
local_train_delta`'s forward/backward pass (`model(batch_images)`) is
the ONLY `model(...)` call in the entire codebase (confirmed via
`grep -rn "model(" fl/*.py`), and it ran on whatever device the model
happened to be on -- CPU, always, since nothing ever changed it. The
only part of the stack that WAS already GPU-aware is `crypto/
sketch.py`'s JL-projection module (`self.device = "cuda" if torch.
cuda.is_available() else "cpu"`), set up independently of the model.
This fully explains the ~1.5GB GPU-memory footprint observed on real
runs: that's CUDA context + the sketch matrix, not the model's own
compute, which was running on CPU the entire time.

(The ZKP/crypto layer itself -- Bulletproofs, Sigma-NIZKs, CLSAG,
Shamir/threshold-ElGamal -- is scalar/group-element arithmetic over
ristretto255, not matrix-heavy, so it genuinely would not benefit from
GPU acceleration; that part of the "why isn't everything GPU-optimized"
question, discussed earlier, was already correctly answered as "by
design, not a gap." This entry is about the part that WAS a real gap:
the model training/eval compute, which is exactly the kind of workload
GPU acceleration is for.)

**Fix, four files**:
- `fl/client.py::local_train_delta`: detect `model_device = next(model
  .parameters()).device` once; move each mini-batch (`batch_images`,
  `batch_labels`) onto it before the forward pass. Dataset tensors stay
  on CPU by design (so holding 50 clients' shards doesn't require GPU
  RAM for all of them at once) -- only the active mini-batch moves.
- `fl/server.py::apply_update`: `torch.from_numpy(aggregated_delta)`
  always lands on CPU (the crypto layer is numpy end to end); added
  `.to(w_t.device)` so the add doesn't raise a device-mismatch
  RuntimeError the instant `w_t` is ever on CUDA -- this would have
  been the very next crash after fixing model placement alone.
- `fl/round_loop.py::load_checkpoint`: added `map_location="cpu"` --
  without it, loading a checkpoint saved from CUDA fails outright on a
  machine with no GPU, or can target a CUDA device index that no
  longer exists after a disconnect/reconnect gives a fresh instance.
- `fl/round_loop.py::run_training_loop`'s resume block: moves the
  now-CPU-loaded `w_t` onto the live model's actual device right after
  restoring it, so a resumed run keeps training on GPU exactly like a
  fresh run does.
- All five run notebooks (`RUN_MVP_TINY_CNN_MNIST[_WITH_ATTACK]
  .ipynb` hand-patched; `RUN_MVP_TINY_CNN_FEMNIST`/`RUN_MVP_
  SMALL_RESNET_{MNIST,FEMNIST}.ipynb` regenerated from the updated
  `notebooks/_build_mvp_run_notebooks.py`): model-build cell now picks
  `device = torch.device('cuda' if torch.cuda.is_available() else
  'cpu')` and does `model = model.to(device)`; eval cells move each
  eval batch to `next(model.parameters()).device` the same way.

**Verified**: 113 passed, 5 skipped locally, unchanged (this sandbox
has no GPU, so the CUDA path itself can't be exercised here -- same
limitation as every other torch-dependent change in this repo; needs
Colab confirmation like the rest). All five notebooks' JSON validity
and Python syntax (`ast.parse`, magics stripped) re-checked after the
patch. Every other `model(...)`/`w_t`-touching call site in `fl/` was
grepped for and checked by hand -- these were the only ones.

**Consequence for the paper, and for tonight's runs**: this is a pure
speed fix, NOT a correctness change -- every numeric result that
leaves torch already crosses to numpy/CPU before touching the DP-noise,
quantization, masking, sketching, or verification layers (confirmed:
`local_train_delta`'s `delta = (new_flat - w_t).detach().cpu().numpy()
...` is unconditional), so nothing about F14's fix, the crypto
guarantees, or any already-reported security-check result changes.
What changes is wall-clock time: real GPU compute for the forward/
backward pass instead of CPU, which should meaningfully shorten all
four queued MVP runs (especially `small_resnet`, ~13x more params
than `tiny_cnn`). Worth having landed before tonight's runs, not after.

## F16 — concurrent MVP run notebooks racing on the shared checkpoint branch (2026-11-16)

**Context**: live on the real tiny_cnn+MNIST rerun (first GPU-confirmed
run, device=cuda), round 0's checkpoint push failed with git's
standard non-fast-forward rejection ("the remote contains work that
you do not have locally").

**Root cause**: all four MVP run notebooks (F13/F15/the matrix added
2026-11-16) deliberately share ONE `colab-checkpoints` branch, keyed
apart only by checkpoint filename (case+model). That's fine for
non-overlapping file writes, but `push_checkpoint_to_git` never
re-synced with the remote before pushing -- so if a second notebook
(one of the other 3 in the matrix) pushes to the same branch between
this notebook's last fetch and its own push, the push is rejected
outright, with no retry logic beyond "try again next round" (which
hits the exact same problem if the other notebook is still running).

**Fix**: `push_checkpoint_to_git` (all 5 run notebooks, plus
`notebooks/_build_mvp_run_notebooks.py`'s template) now retries once
on a push rejection: `git fetch` + `git rebase origin/<branch>`, then
push again. Since every notebook only ever commits its own filename,
the rebase has nothing to conflict on and always auto-merges cleanly.
If the rebase itself somehow fails, it's aborted cleanly and the
original warning is printed -- no change in the no-retry-logic
fallback behavior from F13.

**Applied live, hot-patch first**: the exact new function body was
handed to the user as a paste-into-a-new-cell snippet for the
already-running session (Python functions are called by reference
each round, so redefining it mid-run takes effect from the next round
with no restart needed) before being committed to source for every
future/restarted run.

113 passed, 5 skipped, unchanged (notebook-only change, no fl/ or
crypto/ code touched).

## F17 — F16's retry wasn't enough under real 4-way concurrent load; dedicated branch per notebook (2026-11-17)

**Context**: user uploaded a live Colab log from `RUN_MVP_TINY_CNN_FEMNIST.ipynb`
(identified from the log's `model_dim=102174`, which only matches
`build_tiny_cnn(num_classes=62)` — confirmed by hand against the
model's exact param-count formula, `1569*num_classes + 4832`) showing
12 consecutive round pushes (rounds 0-11), every single one failing
with git's non-fast-forward rejection. The warning text matched F16's
*pre-fix* message format exactly — no "lost a race, rebased, retried"
success message ever appeared, and no "even after rebase retry"
phrasing either — proving this particular running session was started
from a stale, un-hot-patched copy of the notebook (the F16 hot-patch
only reached sessions that were paused to receive it; this one wasn't).
Combined with the user confirming all 4 MVP run notebooks are
genuinely running concurrently right now with the same GitHub PAT,
this is real, sustained 4-way contention on the one shared
`colab-checkpoints` branch F13/F16 used.

**Root cause, restated**: F16 fixed the *symptom* (no retry logic) but
not the underlying *design*: one mutable branch ref with N≥2
concurrent writers will keep colliding under real load no matter how
good the retry is, because a retry can itself lose the same race again
(two notebooks can both rebase onto the same stale tip and both race
to push the rebased result). F16 reduces how often this is visible; it
doesn't eliminate it. The actual fix is to remove the shared resource,
not to retry harder against it.

**Fix**: every MVP run notebook — the clean baseline, the 7-attacker
variant, and all 3 matrix notebooks — now pushes checkpoints to its
OWN dedicated git branch instead of one shared `colab-checkpoints`
branch:

| notebook | branch |
|---|---|
| `RUN_MVP_TINY_CNN_MNIST.ipynb` | `colab-checkpoints-case1-tiny_cnn` |
| `RUN_MVP_TINY_CNN_MNIST_WITH_ATTACK.ipynb` | `colab-checkpoints-case1-tiny_cnn-attack7` |
| `RUN_MVP_TINY_CNN_FEMNIST.ipynb` | `colab-checkpoints-case2-tiny_cnn` |
| `RUN_MVP_SMALL_RESNET_MNIST.ipynb` | `colab-checkpoints-case1-small_resnet` |
| `RUN_MVP_SMALL_RESNET_FEMNIST.ipynb` | `colab-checkpoints-case2-small_resnet` |

With no shared ref, a collision between two MVP run notebooks is no
longer possible by construction — not "less likely," impossible,
since each notebook's `git push` only ever contends with itself across
re-runs/disconnects of that same notebook, which is exactly the single-
writer case F13's original design already handled correctly. F16's
fetch+rebase retry logic is kept in every notebook as defense-in-depth
(e.g. a notebook accidentally run twice at once), not removed.

**Applied**:
- `notebooks/_build_mvp_run_notebooks.py`: `_GIT_CHECKPOINT_CELL_TEMPLATE`
  parametrized on a `ckpt_branch` placeholder via a new
  `git_checkpoint_cell(ckpt_branch)` wrapper; each `RUN_CONFIGS` entry
  carries its own `"ckpt_branch"`; `title_cell`/`CHECKPOINT_MD`/`EVAL_MD`
  updated to describe the per-notebook design instead of one shared
  branch. The 3 generator-built notebooks (`*_FEMNIST.ipynb`,
  `*_SMALL_RESNET_MNIST.ipynb`) were regenerated from this fixed script.
- `RUN_MVP_TINY_CNN_MNIST.ipynb` and `RUN_MVP_TINY_CNN_MNIST_WITH_ATTACK.ipynb`
  (hand-built, not generator-produced) hand-patched the same way: each
  notebook's `CKPT_BRANCH` line, docstrings, and markdown updated to
  its own assigned branch from the table above.
- Live hot-patch snippets (one per currently-running notebook, each
  creating and pushing its own new branch off the session's existing
  local commits, then pointing `CKPT_BRANCH` at it) were handed to the
  user so the 4 already-running sessions can migrate without losing
  any in-progress rounds or restarting.

**Verification**: all 5 notebooks' JSON validated (`json.load`); every
code cell's Python re-checked with `ast.parse` (shell magics `!`/`%`
stripped to `pass`, indentation preserved); each notebook's `CKPT_BRANCH`
literal confirmed to match the table above. 113 passed, 5 skipped,
unchanged (notebook-only change, no `fl/` or `crypto/` code touched).

## F17 follow-up — completed tiny_cnn+MNIST baseline run was pre-F14, accuracy invalid (2026-11-17)

**Discovery**: checked the branch's commit history to confirm whether
`RUN_MVP_TINY_CNN_MNIST.ipynb` still needed to run. Its checkpoint
(`case1_tiny_cnn_r20_f9fix.pt`) had in fact already reached round 19
(all 20 rounds) — but the push timestamps (round 0 at 2026-11-16
05:57 UTC, round 19 at 10:56 UTC) put the *entire run* before F14 was
committed (15:51 UTC that same day). So this completed run has the
exact same bug the earlier 33.34%/33.61% run had: DP noise added to
every one of the ~50 clients instead of just the ~12 real (non-dummy)
ones. A second, independent invalidation, same root cause as F14's
original discovery — not a new bug.

**Fix**: bumped the checkpoint filename in
`notebooks/RUN_MVP_TINY_CNN_MNIST.ipynb` from `case1_tiny_cnn_r20_f9fix.pt`
to `case1_tiny_cnn_r20_f14fix.pt` (same pattern F9's own fix used to
force a fresh round-0 start rather than silently resuming invalid
weights), with a comment explaining both invalidations (F9, F14) for
future reference. Removed the stale `..._f9fix.pt` checkpoint file
from both `colab-checkpoints-case1-tiny_cnn` (its own dedicated branch)
and the old shared `colab-checkpoints` branch (where a copy also still
lived from before the F17 migration), so there's nothing left for a
rerun to accidentally resume from. The old shared branch is still live
(the attack7 notebook hasn't migrated off it yet — see the open item in
STATUS.md), so this was fetched/rebased against its current tip before
pushing, to avoid colliding with that notebook's own pushes.

113 passed, 5 skipped, unchanged (notebook-only change, no `fl/` or
`crypto/` code touched). `RUN_MVP_TINY_CNN_MNIST.ipynb` needs a fresh
20-round rerun under current code before its accuracy can be reported.

## F18 — new fl/attacks/ config module + the 3 missing WITH_ATTACK notebook variants (2026-11-17)

**Context**: only 1 of the 4 (model, dataset) cases in the MVP matrix
had a WITH_ATTACK variant (`RUN_MVP_TINY_CNN_MNIST_WITH_ATTACK.ipynb`,
hand-built for F11). The other 3 (tiny_cnn+FEMNIST, small_resnet+MNIST,
small_resnet+FEMNIST) had no attack run at all, so the paper's
utility+security claim ("honest clients converge while V3 rejects
attackers every round") only had evidence for one case out of four.

**Added**:
- `fl/attacks/` (new package): `fl/attacks/mvp_matrix.py` is the single
  source of truth for the attack-variant matrix — `N_CLIENTS=50`,
  `N_CHEATERS=7` (`CHEATER_FRACTION=0.14`, matching `ARCHITECTURE.md`'s
  own N7 contributions-table target, not A2's `f<n/2=25` theoretical
  ceiling), `CHEATER_INDICES=(0..6)`, and an `ATTACK_CASES` list of 4
  `AttackCaseConfig` entries (one per model×dataset combination) each
  carrying its own dedicated checkpoint git branch
  (`colab-checkpoints-<case>-<model>-attack7`, distinct from both its
  companion clean-run branch and every other attack variant's — same
  per-notebook-branch discipline as F17, extended to the attack
  matrix). Attack *construction* (crafting the malicious delta) stays
  in the pre-existing top-level `attacks/` package
  (`attacks.a1_scaling.craft_scaled_update`) — not duplicated here;
  this module only holds *which clients cheat, in which case, on which
  branch*.
- `tests/test_attack_mvp_matrix.py` (7 tests): cheater count/fraction
  match ARCHITECTURE.md's N7 target; `CHEATER_INDICES` are distinct and
  in range; all 4 (model, dataset) combinations present exactly once;
  notebook filenames and checkpoint branches are all unique; FEMNIST
  cases use `num_classes=62`/MNIST cases use `10`; `case_prefix`
  matches the dataset (case1=MNIST, case2=FEMNIST); and — the actual
  F16/F17 bug class, machine-checked — every attack branch is disjoint
  from every clean-run branch (imports `RUN_CONFIGS` from
  `notebooks/_build_mvp_run_notebooks.py` to check this directly rather
  than trusting two hand-maintained lists to agree).
- `notebooks/_build_mvp_run_notebooks.py` extended with
  `attack_title_cell()`, `attack_run_cell()`, and
  `build_attack_notebook()`, reusing every other cell
  (clone/install/sanity/config/dataset/ring/model/checkpoint-setup/eval)
  **verbatim** from the existing clean-run cell functions — only the
  title markdown and the run cell actually differ for an attack
  variant. Generates `RUN_MVP_TINY_CNN_FEMNIST_WITH_ATTACK.ipynb`,
  `RUN_MVP_SMALL_RESNET_MNIST_WITH_ATTACK.ipynb`, and
  `RUN_MVP_SMALL_RESNET_FEMNIST_WITH_ATTACK.ipynb`. The existing
  hand-built `RUN_MVP_TINY_CNN_MNIST_WITH_ATTACK.ipynb` is deliberately
  **not** regenerated (same reasoning as the clean tiny_cnn+MNIST
  notebook: it may have an in-progress Colab session running against
  its current content) — its `CHEATER_INDICES = list(range(7))` is
  currently hand-written rather than imported from
  `fl/attacks/mvp_matrix.py`, functionally identical but not yet
  import-linked; noted in the module's own docstring as a known,
  intentional gap.

**Verification**: confirmed the 3 generator-built clean-run notebooks
(`RUN_MVP_TINY_CNN_FEMNIST.ipynb`, `RUN_MVP_SMALL_RESNET_MNIST.ipynb`,
`RUN_MVP_SMALL_RESNET_FEMNIST.ipynb` — all 3 have active Colab sessions
running against them) came out **byte-identical** to before this change
(`git diff --stat` empty) — the generator extension touches only new
functions, nothing in the existing clean-run code path. All 3 new
attack notebooks' JSON validated, every code cell's Python re-checked
with `ast.parse` (shell magics stripped, indentation preserved), each
notebook's `CKPT_BRANCH` literal confirmed against the table in
`fl/attacks/mvp_matrix.py`. 120 passed (113 + 7 new), 5 skipped.

## F18 follow-up — attack-crafting line never updated for F15's GPU fix, crashed on real GPU (2026-11-17)

**Context**: user connected their Kaggle-imported copy of
`RUN_MVP_TINY_CNN_MNIST_WITH_ATTACK.ipynb` to a GPU runtime and hit
`TypeError: can't convert cuda:0 device type tensor to numpy` on the
line that crafts the malicious delta, before any training round ran.

**Root cause**: F15 (2026-11-16) moved the model onto CUDA
(`model = model.to(device)`) in every run notebook's model-build cell,
but the attack run cell's `flat_w.numpy().astype(np.float64)` call
(present since before F15 -- it crafts the A1 scaled-update attack
once, from the model's own initial weights) was never updated to
match. `flat_w` comes from `fl/models.py::flatten_state_dict`, which
detaches but does NOT move tensors off their current device --
correct on CPU (F15's audit of the training/eval paths), wrong the one
place a `.numpy()` call survived outside those paths. Confirmed via
grep this line is the ONLY unguarded `.numpy()` call across all 5
WITH_ATTACK-adjacent notebooks (the other hit, in `RUN_ATTACKS.ipynb`'s
F11 gate, is NOT a bug -- that notebook's gate model is never moved to
a device at all, so it stays CPU-only by construction and the identical
pattern there is safe).

**Fix**: `flat_w.numpy()` -> `flat_w.cpu().numpy()` in
`notebooks/_build_mvp_run_notebooks.py`'s `attack_run_cell()` template
(regenerating the 3 generator-built attack notebooks) and hand-patched
into `RUN_MVP_TINY_CNN_MNIST_WITH_ATTACK.ipynb` the same way. User was
also given the one-line live fix to paste into their already-crashed
Kaggle cell and re-run (nothing was lost -- the crash happens before
any training round, so there was no state to resume).

**Verification**: confirmed the 3 clean-run notebooks came out
byte-identical again (only the attack template changed). All 4
WITH_ATTACK notebooks' JSON + per-cell Python syntax re-validated.
120 passed, 5 skipped, unchanged (notebook-only change).

## F18 follow-up — WITH_ATTACK notebooks moved to Kaggle, checkpoint dir updated (2026-11-17)

**Context**: all 4 attack-variant notebooks are now run on Kaggle
instead of Colab (clean-run notebooks stay on Colab). The user had
already validated this manually on their Kaggle copy of
`RUN_MVP_TINY_CNN_FEMNIST_WITH_ATTACK.ipynb`: `/content/FL-ZKP-checkpoints`
still works on Kaggle (clone/install/sanity/dataset cells never needed
changing -- Kaggle accepts writes there too), but only
`/kaggle/working/` is actually surfaced as notebook output when a
Kaggle session ends, so the checkpoint clone needs to live there
specifically to be retrievable.

**Fix**: `notebooks/_build_mvp_run_notebooks.py`'s
`_GIT_CHECKPOINT_CELL_TEMPLATE` parametrized on a second placeholder
(`__CKPT_REPO_DIR_PLACEHOLDER__`, same pattern F17 used for the branch
name); `git_checkpoint_cell()` takes a new `ckpt_repo_dir` parameter
defaulting to `/content/FL-ZKP-checkpoints` (so every CLEAN run
notebook, still on Colab, is unaffected), and `build_attack_notebook()`
passes `/kaggle/working/FL-ZKP-checkpoints` explicitly. The 3
generator-built attack notebooks were regenerated; the hand-built
`RUN_MVP_TINY_CNN_MNIST_WITH_ATTACK.ipynb` was hand-patched the same
way. **The GitHub PAT entry was NOT changed to a hardcoded token** --
despite the user's own already-edited Kaggle cell hardcoding one as an
example of the directory fix, every committed notebook keeps prompting
for the PAT via `getpass()` (hidden input, never written to disk or
committed); a real token committed to a GitHub-hosted notebook is a
credential leak regardless of repo visibility.

**Verification**: confirmed the 3 clean-run notebooks are byte-identical
again (only the attack-notebook code path changed). All 4 WITH_ATTACK
notebooks' JSON + per-cell syntax re-validated; `CKPT_REPO_DIR`
confirmed as `/kaggle/working/FL-ZKP-checkpoints` and `CKPT_BRANCH`
confirmed unchanged (still the per-notebook dedicated branch from F17)
in all 4. 120 passed, 5 skipped, unchanged (notebook-only change).

## F19 — DP layer disabled; privacy claim moved onto the crypto layer (2026-11-17)

**Evidence that forced this.** Two completed 20-round runs came back at
chance-or-near-chance accuracy:

| run | accuracy | chance level |
|---|---|---|
| `small_resnet` + FEMNIST (round 19) | 1.34% unseen-writer / 1.49% global_val | 1/62 ≈ 1.6% |
| `tiny_cnn` + MNIST (round 19) | 42.35% official test / 40.42% global_val | 1/10 = 10% |

Both ran entirely AFTER F14 and F15 landed (checkpoint push timestamps
checked against commit times), so neither is explained by those bugs.

**Diagnosis.** `fl/client.py` added `nu_k` with a per-coordinate std of
`sigma_dp / sqrt(p*n) = 2.5/sqrt(12.5) ≈ 0.707`, fixed regardless of
model size. The noise VECTOR norm therefore scales as `sqrt(model_dim)`:
≈101 for `tiny_cnn` (20,586 params) and ≈367 for `small_resnet`
(269,166 params) — a 3.6× penalty for the larger model purely from
dimension, buying no extra privacy. That ordering matches the observed
severity exactly (small model degraded, large model wiped out).

**Secondary finding, recorded for any future re-enablement**: there is
a unit mismatch between the accountant and the mechanism.
`crypto/dp.py::RDPAccountant` documents `sigma` as
`noise_std / clip_norm` — a RATIO, per the standard DP-SGD definition —
but `fl/client.py` used it as an ABSOLUTE per-coordinate std, and
nothing ever clipped a delta to a fixed sensitivity `C`.
`ARCHITECTURE.md` line 411 claims "clipping already enforced by R_norm:
‖q_k‖₂ ≤ B_t", but `B_t` is an adaptive median+MAD threshold that
*rejects* rather than rescales, and it drifts with the data. So the ε
one would report and the noise actually injected were not connected by
the standard relationship — a referee checking one against the other
would find it.

**Why disabling rather than retuning.** Running the accountant across
the σ range shows no setting that gives both a reportable ε and usable
accuracy at this configuration:

| sigma_dp | ‖nu‖ tiny_cnn | ‖nu‖ small_resnet | eps (20 rounds) |
|---|---|---|---|
| 2.500 | 101.5 | 366.9 | 2.81 |
| 1.000 | 40.6 | 146.7 | 8.09 |
| 0.500 | 20.3 | 73.4 | 19.80 |
| 0.100 | 4.1 | 14.7 | 208.62 |
| 0.025 | 1.0 | 3.7 | 3021.12 |

Getting the noise into a range the signal can survive costs ε in the
hundreds to thousands. Note this was NOT verified against a measured
`‖Δ_k‖` — torch is unavailable in the sandbox, so the signal side of
the ratio is inferred, not measured. Recorded as a known gap: a short
instrumented run printing `‖Δ_k‖` next to `‖ν_k‖` would settle it, and
should be done before any re-enablement.

**Change.**
- `crypto/dp.py`: new `dp_noise_share(sigma_dp, sampling_rate,
  n_clients, model_dim, b_k, rng=None)`. Returns the exact zero vector
  when `b_k` is False (the F14 dummy-client gate) OR when
  `sigma_dp <= 0` (the new DP-disabled state), and otherwise the same
  `N(0, (sigma_dp/sqrt(p*n))^2 I)` draw as before. Placed in
  `crypto/dp.py` rather than `fl/client.py` deliberately: it is pure
  numpy, and `fl.client` transitively imports `crypto.pedersen`, which
  does a hash-to-point at module import time and needs a group backend —
  so housing it here lets it be unit-tested in a network-less sandbox
  without switching the whole pytest process onto the slow pure-Python
  backend (an earlier attempt that set `JANUS_ALLOW_PURE_PYTHON_GROUP=1`
  at test-module scope did exactly that and took the suite from 15s to
  over 2 minutes).
- `fl/client.py`: the inline noise block in `round_step_phase1` now
  calls `dp_noise_share`; behaviour identical when DP is on.
- `configs/mvp_3day.yaml`: `dp.sigma: 0.0`, with the reasoning and the
  re-enable instructions in the config itself.
- `ARCHITECTURE.md` §6.5: status banner stating the layer is disabled,
  what the threat model now claims and explicitly does NOT claim
  (inference from the released aggregate is out of scope), the
  peer-group precedent, and the clipping/accountant mismatch to fix
  first if it is ever turned back on.
- All 8 run notebooks: checkpoint filenames gain a `_nodp` suffix
  (`case1_tiny_cnn_r20_nodp.pt`,
  `case2_small_resnet_r20_withattack7_nodp.pt`, …) so no DP-era
  checkpoint can be resumed by accident — same mechanism F9 and F14
  used. Checkpoint BRANCHES are unchanged (the F17 per-notebook
  branches still apply).
- `tests/test_dp_disabled.py` (6 new tests): exact zeros for a sampled
  client with DP off; exact zeros for a dummy client; negative sigma
  also treated as disabled (rather than reaching `np.random.normal`
  with a negative scale, which raises); F14 regression — dummy still
  gets zeros when DP is ON; the DP-enabled path still produces noise at
  the correct `sigma_dp/sqrt(p*n)` scale; and the shipped config
  actually has DP off, so re-enabling it can't happen silently.

**Verification**: 126 passed, 5 skipped (120 + 6 new), suite still
~15s. Separately ran the 24 normally-skipped crypto tests
(`test_round_end_to_end.py`, `test_f7e_aggregate.py`) against the real
pure-Python group backend — all 24 pass, confirming the
`dp_noise_share` extraction did not disturb the round pipeline. All 9
notebooks' JSON + per-cell Python syntax re-validated; all 8 checkpoint
filenames confirmed unique and `_nodp`-suffixed.

## F20 — FedAvg aggregate divided by len(accepted), which counts dummy submissions (2026-11-17)

**Found while auditing the training path before committing to a long
run** (the question asked was "are the models actually tuned, or will
this waste 10 hours" — the hyperparameters turned out fine; this did
not).

**Bug.** `fl/server.py` ended each round with

```python
dequantized_mean_delta = dequantize(aggregate_quantized, bits=16) / max(len(accepted), 1)
```

`accepted` is every submission passing `verify_submission`
(`server.py:387-396`). Under VRF self-sampling (`sampling_rate: 0.25`)
only ~12.5 of 50 clients actually train; the other ~38 submit
`q_k = 0` as privacy padding so the server cannot tell who trained, and
a zero vector trivially satisfies the norm bound, so **every dummy is
accepted**. Confirmed in the live logs: `accepted=43/50`, `35/50`,
`40/50` — far above the ~12.5 real cohort.

So the numerator summed ~12.5 real deltas while the denominator counted
~43. **Every model update was ~1/p ≈ 4x smaller than FedAvg intends** —
a silent, systematic shrink of the effective server learning rate for
the entire history of the project. `apply_update`'s own docstring
already stated the intended semantic as `w_t + Dequantize(sum q_k)/|A_real|`;
the code and the documented intent disagreed.

This compounds with F19: the 42.4% MNIST result was produced by a model
being simultaneously noise-swamped AND stepped at a quarter speed.

**Fix.** New `fl/aggregation.py::real_contributor_count(n_accepted, p)`
returning `max(p * n_accepted, 1.0)`, threaded through
`run_round(..., sampling_rate=...)` from `fl/round_loop.py`.

Why `p * n_accepted` and not the obvious `p * n_clients - n_rejected`:

| case | p·n_accepted | p·n_clients − n_rejected | correct |
|---|---|---|---|
| p=1.0, 43 accepted (7 rejected) | 43 | 43 | 43 — pre-F20 behaviour, preserved |
| p=0.25, 50 accepted | 12.5 | 12.5 | 12.5 |
| p=0.25, 43 accepted (7 attackers rejected) | 10.75 | 5.5 | 10.75 |
| p=0.25, 35 accepted (15 rejected) | 8.75 | **−2.5** | — |

The attack row is the one that matters: `_force_delta` is applied in
`fl/client.py` AFTER the `b_k` branch (line 378), so attack submissions
bypass the sampling gate entirely and only ~p of the rejected attackers
displaced a genuinely-sampled honest contributor. Subtracting all 7
over-corrects to 5.5 and would roughly double the applied step. The
last row is not hypothetical — a 15-rejection round was observed live.

`p=1.0` reduces exactly to the old denominator, which is why that is the
default and why the three existing `run_round` test call sites are
unaffected.

**Privacy**: uses only the public `p` and the public accepted count, so
no individual `b_k` is revealed and the anonymity the dummies exist to
protect is intact.

**Known limitation, recorded**: this is an expectation, not the realized
count. `|S| ~ Binomial(50, 0.25)` has std ≈ 3.06, so the effective step
inherits that jitter round to round. The principled upgrade is for each
client to commit to `b_k` and open only the homomorphic SUM of those
commitments — exact count, no individual bit revealed, and the Pedersen
commitments already in the protocol are additively homomorphic, so it is
an extension rather than new machinery. Deferred: protocol change, and
the expectation is unbiased.

**Checkpoint filenames** bumped `_nodp` -> `_nodp_f20`, since any
checkpoint written in the ~20 minutes between F19 and this fix carries
the 4x-too-small dynamics and must not be resumed into a corrected run.

**Verification**: 134 passed (126 + 8 new in
`tests/test_aggregate_scaling.py`), 5 skipped. Separately re-ran the 24
normally-skipped crypto/e2e tests against the real pure-Python group
backend after the `server.py` change — all 24 pass. All 9 notebooks'
JSON + per-cell syntax re-validated. Confirmed `sampling_rate` is
defined before the `run_round` call that consumes it.

**Not verified**: the 4x larger step has never run against a real model
here (no torch in the sandbox). If `lr=0.01` is now too hot, `quantize()`
raises `FloatingPointError` with an explicit "lower lr" message rather
than failing silently — so a divergence shows up in round 0-1, not after
10 hours.

## F21 (2026-11-17): no per-round accuracy -- only one end-of-run number existed

**Cause.** `fl/round_loop.py::run_training_loop` tracked only protocol
state per round (`accepted`, `aborted`, `aggregate_exact`, `B_t`) in
`history`. Accuracy was computed exactly once, in each notebook's final
`eval_cell`, against whichever checkpoint happened to be on disk when
that cell ran. There was no accuracy-vs-round curve at all -- the
standard convergence-curve result every FL paper reports, FedAvg's own
included.

**Evidence.** Grepped `fl/round_loop.py` for `evaluate`/`accuracy` --
zero hits inside the training loop itself; the only call sites are in
the notebooks' own `eval_cell`, which runs after all rounds finish.

**Fix.** Added an optional `eval_fn: (round_idx, w_t) -> dict|float`
parameter to `run_training_loop`. Called once per successfully-completed
round, right after that round's `w_t` is applied (the real aggregated
global update, not a stray last-trained-client's local weights -- see
the function's own docstring on why `model` can't be trusted mid-loop).
Result stored at `history[-1]["eval"]`. `None` (default) -> no eval, no
behavior change for any existing caller (tests, attack notebooks that
don't pass it).

The call-and-swallow logic is factored into its own module,
`fl/eval_hook.py` (`run_eval_fn_safely`, sentinel `EVAL_FAILED`), NOT
defined inside `fl/round_loop.py` -- for the exact reason FIX.md F19
pulled `dp_noise_share` out of `fl/client.py`: `fl.round_loop` imports
`verifier.audit` -> `crypto.pedersen`, which hash-to-points at *module
import time* and requires a real ristretto255 backend. A test file doing
`from fl.round_loop import <anything>` fails at collection time in this
sandbox (no rbcl, no network) before a single test runs. This was
discovered live while writing `tests/test_eval_fn_hook.py` for this very
fix -- `fl.eval_hook` has no crypto/torch import chain, so it imports and
unit-tests cleanly on its own.

**Wired into all 8 run notebooks** (both hand-built and generated): each
defines `_eval_fn(round_idx, w_t)` closing over `model`/`model_shapes`/
the headline test split (MNIST `test` for case1, FEMNIST `unseen_test`
for case2), doing a forward-only pass (no backward, no optimizer step --
the real per-round cost is the sampled clients' local SGD, not one extra
inference loop), and passes it as `eval_fn=`. Each notebook now prints
a per-round accuracy line after the run, in addition to the existing
end-of-run `eval_cell` number.

**Verification**: new `tests/test_eval_fn_hook.py` (7 tests) exercises
`run_eval_fn_safely` directly -- success path returns the result
unchanged, a raising `eval_fn` returns the `EVAL_FAILED` sentinel (never
propagates), a legitimate `None` return from `eval_fn` is NOT confused
with that sentinel, and the exact `is not None and is not EVAL_FAILED`
condition `run_training_loop` uses to decide whether to store is tested
directly. Full suite: 141 passed (134 + 7 new), 5 skipped. Re-ran the 24
normally-skipped crypto/e2e tests against the real pure-Python backend
after touching `fl/round_loop.py`'s imports and loop body -- all 24 still
pass (181s). All 12 notebook files (8 main + 4 new ablation, see below)
re-validated as well-formed JSON.

**Not verified**: `run_training_loop` itself (and therefore the
`eval_fn` hook as actually exercised inside a real round) cannot run in
this sandbox -- no torch installed. The hook's call-and-swallow logic is
unit-tested in isolation (above); its wiring inside a real training loop
is verified by code reading only, same limitation noted throughout this
log for every `fl/client.py`/`fl/round_loop.py` change.

## Ablation study (2026-11-17): full-participation (p=1.0) scalability check

Added `notebooks/ablation_study/` -- 4 notebooks (tiny_cnn/small_resnet x
MNIST/FEMNIST), 2 rounds each, generated by
`notebooks/ablation_study/_build_ablation_notebooks.py`, which imports
every shared cell VERBATIM from `notebooks/_build_mvp_run_notebooks.py`
(no reimplementation, no library change) and overrides only
`CFG['protocol']['sampling_rate'] = 1.0` and `REAL_ROUNDS = 2` in its own
run cell.

**Purpose**: answer the anticipated reviewer question "is the protocol
actually capable of training with all 50 clients, or does it only work
at `p=0.25`?" directly, with a live run, not just an argument. At
`p=1.0` the VRF self-sampling gate (ARCHITECTURE.md SS5.5) passes every
one of the 50 clients -- zero dummy submissions -- so a clean pass
through all 4 combinations is evidence the masking/Shamir/ring-signature/
aggregation layers are not structurally limited to the subsampled case.
2 rounds, not 20: this is a correctness/scalability check, not a
convergence experiment, and running all 50 real every round costs ~4x
the GPU time of the main matrix for a result not being claimed.

**Industrial framing added to ARCHITECTURE.md SS5.5.1** (new
subsection): production cross-device FL (Google, Apple) never runs
100% participation either -- partial participation by construction, for
the same device-availability/bandwidth/compute-budget reasons this
paper uses `p=0.25` at `n=50`. This is the wording to reuse in the
paper's industrial-applicability section.

**Verification**: all 4 generated notebooks validated as well-formed
JSON. `fl/aggregation.py::real_contributor_count` already unit-tested
at `p=1.0` (FIX.md F20) to reduce exactly to the unsampled FedAvg
denominator, so the ablation's correctness claim rests on both a live
run and a proof-by-formula, not the live run alone.

**Not verified**: the ablation notebooks themselves have not been run
on Colab/Kaggle yet (no torch/GPU in this sandbox) -- this entry
describes what was built and why, not a confirmed pass/fail result.

## F21 follow-up (2026-11-18): FEMNIST eval_fn crashed on round 0 -- unseen_test isn't a tuple

**User-reported**, live Colab run of `RUN_MVP_TINY_CNN_FEMNIST.ipynb`:

```
TypeError: cannot unpack non-iterable FemnistClientDataset object
---> 45 _eval_images, _eval_labels = case2_data['unseen_test']
```

**Cause.** F21's `eval_fn` wiring assumed every case's headline split was
a plain `(images, labels)` tuple, copying the shape MNIST's `test` split
actually has (`fl/data_iid.py`: `datasets["test"] = (test_images,
test_labels)`). FEMNIST's `unseen_test` is NOT that shape -- it's a
`FemnistClientDataset` dataclass (`fl/data_noniid.py`) with `.images`/
`.labels` attributes. The existing `eval_cell` (F-pre-dating this
session) already knew this and read `unseen_ds.images`/`unseen_ds.labels`
correctly; the new per-round `eval_fn` code, written fresh instead of
reusing that pattern, did not.

Only hit the 2 of 4 (case_prefix == case2) notebooks and their attack
variants: `RUN_MVP_TINY_CNN_FEMNIST(_WITH_ATTACK).ipynb`,
`RUN_MVP_SMALL_RESNET_FEMNIST(_WITH_ATTACK).ipynb`. The MNIST (case1)
notebooks and the hand-built `RUN_MVP_TINY_CNN_MNIST(_WITH_ATTACK).ipynb`
were never affected -- `test` really is a tuple there.

**Fix.** `run_cell`/`attack_run_cell` in
`notebooks/_build_mvp_run_notebooks.py` now branch on `case_prefix`:
case1 keeps the tuple-unpack line unchanged; case2 does
`_eval_ds = case2_data['unseen_test']; _eval_images, _eval_labels =
_eval_ds.images, _eval_ds.labels` -- the exact same attribute access
`eval_cell` already uses successfully, not a new pattern. Regenerated
all 6 generated notebooks; the 2 hand-built MNIST notebooks needed no
change (case1 only).

**Verification**: all 18 notebook files (12 main + 4 ablation + 2
pre-existing setup/verify notebooks) re-validated as well-formed JSON;
confirmed by direct read that `_eval_images, _eval_labels = _eval_ds.images,
_eval_ds.labels` is now present in all 4 case2 notebooks and the
tuple-unpack form is unchanged in the 4 case1 notebooks. 141 passed, 5
skipped (unchanged -- this was a notebook-generator-only fix, no `fl/`
code touched, so the real-crypto-backend e2e re-check was not needed).

**Not verified**: not yet re-run on Colab/Kaggle (no torch/GPU here) --
this fixes the exact crash reported, by reusing code already proven to
work in `eval_cell`, but the next real run is what actually confirms it.

## Milestone (2026-11-18): first complete, trustworthy 20-round result matrix

All 4 (model, dataset) combinations finished round 19 on the final
`_nodp_f20` pipeline (F19 DP-disable + F20 aggregate-scaling fix), with
checkpoints confirmed pushed to all 4 dedicated branches
(`colab-checkpoints-case{1,2}-{tiny_cnn,small_resnet}`,
`case{1,2}_{model}_r20_nodp_f20.pt`, verified via `git ls-tree`/`git log`
on each branch -- all last-committed 2026-11-18 02:42-03:27 UTC).

| Model | Dataset | Test acc | Held-out global_val acc | Gap |
|---|---|---|---|---|
| tiny_cnn | MNIST | 0.9852 (10000 images) | 0.9788 (2499 images) | 0.0064 |
| small_resnet | MNIST | 0.9891 (10000 images) | 0.9852 (2499 images) | 0.0039 |
| tiny_cnn | FEMNIST | 0.7818 (69425 images) | 0.7743 (9327 images) | 0.0075 |
| small_resnet | FEMNIST | 0.7013 (69425 images) | 0.6986 (9327 images) | 0.0027 |

**Compare to the pre-fix (DP-enabled, aggregate-bugged) tiny_cnn+MNIST
run**: 0.4235 test / 0.4042 global_val -- the jump to 0.9852/0.9788 is
the direct, now-measured confirmation that F19+F20 were the real bugs,
not a hypothesis. No DP-noise-scaling argument or units-mismatch
analysis was ever run against a real model before this (no torch in
this sandbox); this is the first number that actually tests it.

**Pattern worth flagging for the paper, not smoothing over**: tiny_cnn
(20.6K params) BEATS small_resnet (263K params) on FEMNIST (0.7818 vs
0.7013) but loses narrowly on MNIST (0.9852 vs 0.9891). A ~13x larger
model does not automatically win under this protocol's non-IID,
writer-split, ~12-real-clients-per-round regime -- plausibly
undertrained at 20 rounds relative to its parameter count, or more
sensitive to per-client data skew than the smaller model. Worth a
sentence of discussion in the results section rather than treating
small_resnet's MNIST win as the whole story.

**test vs. global_val agree within ~1pt in all 4 cells** -- no
overfitting to the held-out global validation set, consistent
generalization signal across both datasets.

**Not yet checked**: these numbers have not been compared against any
specific benchmark figure from the peer-group papers anchored in
ARCHITECTURE.md (PQS-BFL, WhiteCloak, ZK-FL) -- ARCHITECTURE.md names
those papers for threat-model/architecture precedent only and does not
itself report their accuracy numbers, so there is nothing numeric in
this repo to check against yet. If the paper wants a head-to-head
accuracy comparison, that requires pulling each paper's own reported
numbers separately.

## Merged ablation notebook (2026-11-18): Ablation_model.ipynb, pushed to its own `ablation` branch

**User request**: merge the 4 standalone ablation notebooks
(`ABLATION_TINY_CNN_MNIST`, `ABLATION_TINY_CNN_FEMNIST`,
`ABLATION_SMALL_RESNET_MNIST`, `ABLATION_SMALL_RESNET_FEMNIST`) into
one notebook, `Ablation_model.ipynb`, and push it to a branch named
`ablation`.

**Not a naive concatenation.** The 4 standalone notebooks each
clone/install/sanity-check/load-config/build-the-crypto-ring from
scratch -- identical across all 4 (same repo, same `CFG`, `RING_CELL`
only reads `CFG['protocol']`, never model/dataset). Concatenating all
4 notebooks verbatim would re-clone, re-pip-install (~minutes each),
and rebuild an identical 50-client ring three times over for nothing.
`notebooks/ablation_study/_build_ablation_merged_notebook.py` dedupes
all of that to run once, and loads each dataset (MNIST, FEMNIST) once,
reused by both models that use it, instead of once per combo.

**The one real bug this merge would otherwise have shipped**: the
per-notebook checkpoint-setup cell only clones + picks a branch ONCE
per session (`if not os.path.exists(CKPT_REPO_DIR): <clone + pick
branch> else: print("reusing it")`), because every other notebook in
this repo is one-notebook-one-branch. Reused verbatim in a merged
notebook, combo 2's checkpoint cell would see the directory already
exists (from combo 1) and silently skip picking combo 2's branch --
quietly pushing combos 2-4's checkpoints into combo 1's branch.
Caught before running, not after: new `merged_checkpoint_cell` in the
build script runs the fetch-or-create-branch step on EVERY call, and
when a combo's branch doesn't exist yet, branches explicitly from
`origin/main` rather than "current HEAD" (which on combo 2+ would be
the previous combo's branch tip, carrying its checkpoint history into
a branch that should start clean).

**Verification**: `Ablation_model.ipynb` (31 cells) validated as
well-formed JSON; every code cell's source compiled for syntax errors
(one false positive on the reused `CLONE_CELL`'s `!git clone` magic
line, a Jupyter-only construct `compile()` can't parse -- not a bug,
identical to the cell already proven working in all 8 other run
notebooks). Full suite re-run: 141 passed, 5 skipped (no `fl/` code
touched, generator-only change).

**Not verified**: not run on Colab yet. The 4 individually-generated
`ABLATION_*.ipynb` notebooks remain in `notebooks/ablation_study/`
unchanged (not deleted) -- `Ablation_model.ipynb` is an additional,
merged alternative, not a replacement, in case the per-combo notebooks
are easier to run independently.

**Traceability demo (2026-11-18): §6.2 opening protocol tested standalone against real crypto, before pipeline integration -- two gaps found and one partially closed.**

User asked to review the traceability/audit subsystem (§6 of ARCHITECTURE.md) while the attack/ablation runs were in progress on Colab. Audit findings (no code run yet at this point, pure investigation):

- **Wired into the real pipeline and tested (not gaps)**: F7f suspect tracking (`fl/suspect_tracker.py`, quarantine/ban bookkeeping) is called every round from `fl/round_loop.py` and has 15 passing tests. `verifier/audit.py::independent_v5_check` is also called every round from `round_loop.py`.
- **Gap 1 -- §6.1 evidence accumulation (`fl/evidence.py`)**: solid, well-tested in isolation (`tests/test_evidence.py`: cosine/norm/loss-delta scoring, calibration bound, honest-vs-malicious separation) but never imported by `fl/server.py`/`fl/round_loop.py`/`fl/client.py` -- only referenced in `RUN_ATTACKS.ipynb`, whose own comment calls that usage "illustrative stand-ins."
- **Gap 2 -- §6.2 opening protocol (`fl/trace.py`) and §6.4 committee reconfiguration (`crypto/reshare.py`)**: both ★ NOVEL in ARCHITECTURE.md, both correct-looking orchestration over already-tested primitives (own docstrings say "EC-DEPENDENT -- not run in this sandbox"), both had **zero test files anywhere** and **zero pipeline callers** -- only referenced from setup/demo notebooks.
- **Gap 3 (new, found while reading `verifier/audit.py` in full)**: `independent_opening_reverification` -- itself a starred novelty point ("any third party can re-verify the entire accusation") -- has **zero callers anywhere in the repo**: not the pipeline, not a notebook, not a test. Completely orphaned.
- **Gap 4**: §6.3's residual-anonymity formula `H(T) = log2(n - o(T))` has no implementation anywhere. `fl/trace.py::revoke()`'s own docstring defers it "to wherever the paper's figures are generated" -- no such file exists in the repo.

Per the user's explicit plan ("test the opening protocol standalone first, then integrate into the full pipeline"), built `notebooks/traceability_demo/TRACEABILITY_DEMO.ipynb` (generator: `_build_traceability_demo.py`) closing part of Gap 2 and Gap 4 at the standalone level:

- Runs the REAL Pedersen DKG (`crypto/dkg.py`) + threshold ElGamal (`crypto/elgamal_threshold.py`) + `fl/trace.py::run_opening_protocol`, with the project's real committee params read from `configs/mvp_3day.yaml` (`n_key_holders=10`, `threshold=7`), not re-typed by hand.
- **Case A**: all 10 key holders cooperate -> opens, recovers the exact escrowed identity (`verdict.opened_pk == accused_client_pk`, asserted).
- **Case B**: holders 1, 2, 3, and 4 missing (6 of 10 present, below t=7) -> fails with `reason="only 6 valid partials, need t=7"`. Note: the original ask was "1,2,3 missing" -- with the real t=7-of-10 committee that leaves exactly 7, which still succeeds. Confirmed with the user to add holder 4 to the missing set so Case B actually demonstrates a failed opening.
- Bonus sweep cell: holders-present from 10 down to 0, confirms the opened/failed flip happens exactly at 7, not gradually -- same plot shape (success vs. count) the full pipeline's eventual per-round traceability figure will want.
- Added a demo-only `residual_anonymity_bits()` helper (NOT promoted into `fl/trace.py` yet) to report Gap 4's `H(T)` for Case A's `revoke()` result: a 50-member ring drops from `log2(50)=5.644` to `log2(49)=5.615` bits after the one opening.

Deliberately NOT exercised yet (scoped out, left for the pipeline-integration step): `fl/evidence.py`'s real evidence accumulation (Case A/B use a stubbed, already-above-threshold `CalibrationCertificate` instead, since no training rounds ran to accrue real evidence), and §6.2 step 5's R_cons identity-binding re-check (needs a genuine round's `rho_t`/`rho_e`/`vrf_alpha` and proof, which only exist once this is wired into `fl/round_loop.py`).

**Verification**: executed every code cell for real (not just syntax-checked) against the test-only pure-Python ristretto255 backend (`JANUS_ALLOW_PURE_PYTHON_GROUP=1`, same gate `tests/test_round_end_to_end.py` uses) -- Case A opens and recovers the exact identity, Case B fails with the expected reason, the sweep confirms the 7/6 boundary, revocation shrinks the ring by exactly 1. Notebook JSON validated well-formed, 14 cells (one expected false positive on the reused `CLONE_CELL`'s `!git clone` line, same known non-bug as every other notebook). No `fl/`/`crypto/` library files touched -- generator + notebook only. 141 passed, 5 skipped, unchanged.

**Not yet done**: not run on Colab with `rbcl` (should be identical -- same algebra, only the group-operation bindings differ, per the sanity cell's fallback design). Gap 1 (`fl/evidence.py` unwired), Gap 3 (`independent_opening_reverification` orphaned), and the full pipeline-integration step (wiring §6 into `fl/round_loop.py` for all 4 model/dataset combinations, with the per-round "traceability happened at round N" result and H(T) curve for the paper) are still open, per the user's own stated next step.

**Traceability demo follow-up (2026-11-18, same day): `residual_anonymity_bits` promoted from a demo-only draft into a real, tested `fl/trace.py` function; the opening protocol itself now has test coverage too.** User asked directly why the §6.3 helper was left as an inline draft in the notebook instead of being implemented properly -- fair question, there was no real reason to hold back on a function this small and self-contained. Moved `residual_anonymity_bits(ring_size_before, n_opened_so_far) -> float` into `fl/trace.py` itself, right after `revoke()` (which it's meant to pair with), with a proper docstring tied to ARCHITECTURE.md §6.3 and explicit semantics for how `n_opened_so_far` accumulates across repeated `revoke()` calls.

Also closed part of the other §6.2 gap found during the review (`fl/trace.py` had zero test files): new `tests/test_trace.py` (6 tests, pure math, run unconditionally -- full ring, drop after one opening, strict monotonic decrease, bottoms out at `-inf` once every member is opened rather than erroring, rejects a non-positive ring size or a negative opened-count) and `tests/test_trace_opening_protocol.py` (7 tests, gated on a live ristretto255 backend exactly like `tests/test_round_end_to_end.py` -- real DKG over the real n_c=10/t=7 committee, opening succeeds with all 10 holders and recovers the exact escrowed identity, succeeds at exactly t=7 present (the boundary itself still works), fails below threshold (6 of 10, same Case B as the demo notebook), fails when evidence is below the calibrated theta even with every holder present (the evidence gate runs before the partial-decryption threshold, not after), fails without admin authorization, and `revoke()` shrinks the ring by exactly one member with `residual_anonymity_bits` matching). Both files promoted directly from the demo notebook's own by-hand validation.

**One bug caught while writing these**: the first version of `tests/test_trace.py` put the EC-gated tests in the SAME file as the pure-math ones, with the module-level `pytest.skip(allow_module_level=True)` partway down the file -- that skip call aborts collection of the ENTIRE file regardless of where it appears, so without a backend the pure-math tests were silently skipping too (caught immediately: ran once with no backend set and got "collected 0 items / 1 skipped" instead of 6 passes). Fixed by splitting into two files, matching this repo's own existing convention (`tests/test_evidence.py` pure-math vs. `tests/test_round_end_to_end.py` backend-gated, already separate for the same reason) -- never mix an `allow_module_level` skip into a file that also has backend-independent tests.

Updated `notebooks/traceability_demo/TRACEABILITY_DEMO.ipynb`'s revocation cell to import `residual_anonymity_bits` from `fl.trace` instead of redefining it inline; title/closing markdown updated to stop calling it a draft. Re-executed the full notebook end to end against the real pure-Python backend -- identical output to before (Case A opens and matches, Case B fails with the same reason, sweep boundary unchanged, `H(T)`: 5.644 -> 5.615 bits for the one Case-A revocation).

**Verification**: 147 passed (141+6), 6 skipped (5+1, the opening-protocol file skips without a backend) with no backend set; 13/13 of the two new files' tests pass against the pure-Python backend (`JANUS_ALLOW_PURE_PYTHON_GROUP=1`); full suite re-confirmed against the real backend: 221 passed, 0 failed, in ~4m35s (no `fl/round_loop.py`/`fl/server.py`/`fl/client.py` touched, so this is purely the new tests plus the pre-existing crypto/e2e suite, all green).

## Milestone (2026-11-19/20): attack-matrix (attack7) results -- all 4 final

All 4 (model, dataset) combinations ran 20 rounds with 7 of 50 clients
(14%, matching ARCHITECTURE.md's N7 target) forced to a model-
replacement-scale malicious delta every round, on the same `_nodp_f20`
pipeline as the clean matrix. Checkpoints confirmed pushed to all 4
dedicated `colab-checkpoints-case{1,2}-{tiny_cnn,small_resnet}-attack7`
branches.

| Model | Dataset | Test acc | Held-out global_val acc | vs. clean matrix |
|---|---|---|---|---|
| tiny_cnn | MNIST | 0.9830 (10000 images) | 0.9804 (2499 images) | 0.9852 -> 0.9830 (-0.0022) |
| small_resnet | MNIST | 0.9878 (10000 images) | 0.9844 (2499 images) | 0.9891 -> 0.9878 (-0.0013) |
| tiny_cnn | FEMNIST | 0.7787 (69425 images) | 0.7758 (9327 images) | 0.7818 -> 0.7787 (-0.0031) |
| small_resnet | FEMNIST | 0.6626 (69425 images) | 0.6573 (9327 images) | 0.7013 -> 0.6626 (-0.0387) |

**Security claim, confirmed for all 4 combos**: "PASS: every attacker
that was ever eligible ([0, 1, 2, 3, 4, 5, 6]) was rejected by V3
every time, across all 20 rounds, while the honest majority kept
training normally" -- identical verdict line in every one of the 4
logs, 7/7 attackers rejected every single round, `accepted=43/50`
every round, `aggregate_exact=True` every round.

**Utility-under-attack pattern worth a sentence in the results
section**: accuracy loss from the attack is small and roughly tracks
task difficulty -- small_resnet+MNIST and tiny_cnn+FEMNIST both lose
under half a point (consistent with losing 7/50=14% of potential
training contributions but V3 successfully isolating 100% of the
damage they'd otherwise cause), while small_resnet+FEMNIST loses
~3.9 points -- the harder (non-IID, writer-split) task combined with
the larger model is more sensitive to the reduced effective
participant count than the easier MNIST runs are. Not a new
phenomenon, same pattern F19/F20's milestone already flagged for
small_resnet's relative underperformance on FEMNIST generally.

**tiny_cnn+MNIST+attack7 re-run completed clean (2026-11-20).** The
prior run's unexplained round-10 history gap (flagged to the user
2026-11-19) did not recur: this run's log shows continuous `round 0`
through `round 19` in both the accuracy curve and the attacker-verdict
list with no gaps, `done: 20 rounds in history, 20 accepted (not
aborted)`, `accepted=43/50`/`aggregate_exact=True` every round, and the
same `PASS: every attacker that was ever eligible ([0, 1, 2, 3, 4, 5,
6]) was rejected by V3 every time, across all 20 rounds` verdict the
other 3 combos already had. Slotted into the table above -- all 4
attack7 combos are now final.

**Traceability result for this combo (first real §6.2 confirmation on
an attack run)**: `round 0: §6.2 TRACEABILITY -- 7 pseudonym(s) crossed
theta=9.680 this epoch (epoch 0)` -- exactly the 7 injected attackers
([0, 1, 2, 3, 4, 5, 6]), flagged in the very first epoch. This matches
the prediction recorded in the TWENTY-NINTH item before this run: clean
runs should show zero traceability events, attack7 runs should show all
7 attackers opened very early. First empirical confirmation of that
prediction on a real run (this notebook -- `RUN_MVP_TINY_CNN_MNIST_WITH_ATTACK.ipynb`,
run standalone by the user, not the combo notebook -- already had the
§6.2 opening-protocol wiring from the TWENTY-NINTH item).

**Verification**: all 4 combos' logs show continuous `round 0` through
`round 19` in both the accuracy curve and the attacker-verdict list
with no gaps, `done: 20 rounds in history, 20 accepted (not aborted)`.

## Traceability demo confirmed on real libsodium backend (2026-11-19)

User ran `notebooks/traceability_demo/TRACEABILITY_DEMO.ipynb` on
Colab. `ristretto255 backend: rbcl` -- the real backend, not the
local sandbox's pure-Python fallback this was first validated
against. Every result matches exactly what was checked locally:

- Case A (all 10 of 10 key holders): `opened: True`, recovered
  identity matches the accused client's real pk: `True`.
- Case B (holders 1-4 missing, 6 of 10 present): `opened: False`,
  `reason: only 6 valid partials, need t=7`.
- Sweep: boundary flips exactly at 7 present holders (10 through 7 ->
  opened=True; 6 through 0 -> opened=False).
- Revocation: ring 50 -> 49; `residual_anonymity_bits`: `H(T) =
  log2(50-0) = 5.644` bits before, `log2(50-1) = 5.615` bits after.

Closes the one open item noted in this notebook's own docstring ("not
yet done: not run on Colab with rbcl"). Same algebra, same outcome,
real backend -- the opening protocol, the DKG, and the H(T) helper
are now validated against both the test-only pure-Python group and
the real libsodium-backed one.

## §6 traceability wired into the real per-round pipeline (2026-11-19)

Continuation of the "next phase" the user asked for after the attack-
matrix + traceability-demo results landed. Investigated what's
actually available inside `fl/round_loop.py` before writing any code
(same discipline as the original §6 review): the real epoch pseudonym
(`key_image_epoch`, `P_k^e`) and escrow ciphertext are already attached
to every submission (`fl/client.py`), ready to use as-is -- but the
sketch commitment only ever exposes a committed SCALAR (`v = ||Sq||^2`,
via `crypto.sketch.int_sketch_norm`), never the raw vector `q`, by
design (the whole point of the norm-bound ZK proof is that the server
verifies without learning the update). The one place a plaintext `q`
exists is a documented "simulation-only" self-check copy, explicitly
destroyed before round end and explicitly noted as something no real
party could access.

That kills §6.1's `w1*cos_dist(q, median)` term outright for the real
pipeline -- not a bug, a direct structural consequence. Checked whether
`loss_delta_on_root_set` (the formula's third term) could stand in:
it can't either, for the identical reason -- computing it needs
`w_t + q_k` evaluated on the root set, i.e. the same raw vector. The
only way to get a loss-delta signal at all would be the CLIENT
self-reporting it, unverified (nothing ZK-proves it matches the
client's actual committed update) -- a real adversary could lie about
it freely, unlike `norm_q` which the sketch+rangeproof already
enforces cryptographically.

Put both findings to the user before writing any code (two
AskUserQuestion checkpoints, since this affects what the paper can
honestly claim about the wired pipeline): **evidence_t for the real
pipeline is `norm_q/B_t` only** (w1=w3=0) -- fully ZK-verified, zero
added trust assumptions, at the cost of not catching a "small,
persistently off-distribution but under-the-norm-bound" attacker (a
different attack shape than anything in the current attack matrix,
all of which scale the norm up, not stay under it).

**Built, tests-first, library layer only (notebook wiring is next)**:
- `fl/evidence.py`: added `norm_only_evidence(norm_q, B_t)` (the
  formula reduced to its one live term) and
  `calibrate_theta_from_norms(root_set_norms, B_t, rounds_per_epoch,
  sampling_rate, ...)` (calibrate_theta's norm-only counterpart,
  reusing the SAME root-set-norms array `fl/norm_rule.py`'s
  `bootstrap_from_server_root` already consumes for B_t -- no new data
  plumbing). The original 3-term `per_round_evidence`/`calibrate_theta`
  are untouched, still exactly as specified, for the standalone demo
  and the paper's theoretical description of the full formula.
- `fl/trace_wiring.py` (new): `EpochEvidenceState` -- owns one
  `EvidenceAccumulator` + its calibrated theta for the current epoch,
  rolls over automatically when `t_round // epoch_rounds` changes
  (the same expression `round_loop.py` already uses for `rho_e`),
  and reports newly-flagged pseudonyms per round. Deliberately does
  NOT call `fl/trace.py`'s `run_opening_protocol` itself -- the
  committee's secret key-holder shares live only in notebook scope
  (`crypto/dkg.py`'s `finalize()` output), never passed into
  `round_loop.py`; this module's job is server-side detection only
  (who crossed theta, and when), cheap enough to run every round for
  every client, leaving the slower EC-heavy actual decryption to a
  post-hoc notebook step for just the handful of flagged pseudonyms.
- `fl/round_loop.py`: wired in. Every ELIGIBLE client each round
  (accepted AND rejected -- a client V3 rejects this round still
  submitted a real committed norm, and theta is calibrated against
  the full honest-population distribution, so excluding rejected
  clients would stop tracking exactly who evidence accumulation
  exists to catch) feeds `evidence_state.record_round(...)`, using
  `threshold.B_t` as it stood for THIS round (captured before the EMA
  update reassigns it). New optional params: `evidence_alpha` (default
  0.05), `evidence_n_permutations`, `evidence_rng`, and
  `on_pseudonym_flagged(pseudonym_bytes, t_round, evidence_total)` --
  all default to not breaking any existing call site. Non-aborted
  rounds' `history` entries gained `evidence_newly_flagged` (hex
  strings), `evidence_theta`, `evidence_epoch` for post-hoc plotting.
  **Known, documented gap**: `evidence_state` is NOT checkpointed --
  a resumed run starts a fresh epoch's accumulator. Harmless for every
  run so far (all <= epoch_rounds=25 long, so "this epoch" and "this
  whole run" coincide); would need fixing before a run spans multiple
  epochs AND gets interrupted mid-epoch.

Confirmed real dummy-padding behavior directly from the code (not
assumed) before relying on it for calibration: `eligible_this_round`
is EVERY non-quarantined client every round (VRF self-sampling happens
INSIDE `round_step_phase1`, deciding real-delta vs. honest-zero-delta
internally), matching `calibrate_theta_from_norms`'s
Binomial(rounds_per_epoch, sampling_rate) participation model exactly
-- a non-sampled client's dummy zero delta naturally yields `norm_q~0`
that round, so it contributes ~nothing to its own evidence total
without needing a separate "is this a dummy round" flag anywhere.

**Tests**: `tests/test_trace_wiring.py` (new, 6 pure-math tests --
epoch rollover resets totals, theta doesn't recalibrate mid-epoch, a
realistically-dummy-padded honest pseudonym clears theta, a
persistently-scaled pseudonym gets flagged exactly once, rejected
clients still accumulate evidence) + 5 new tests in
`tests/test_evidence.py` (norm_only_evidence matches the full
formula's w2-only term exactly, B_t<=0 edge case, the same
false-accusation-rate and scaled-attacker guardrail tests
`calibrate_theta` already had, now for the norm-only path). One test
bug caught and fixed before it was a code bug: an early draft of
`test_honest_pseudonym_stays_unflagged_across_a_full_epoch` had
"honest" submit a real (nonzero) norm EVERY round -- 100% participation,
not the ~sampling_rate=0.25 the calibration assumes -- which of course
exceeded theta; fixed by modeling honest participation the same
dummy-padded way the real pipeline actually works (mostly zero-norm
rounds, a real norm only on the VRF-sampled ones), which is also
exactly how the current wiring will behave for real.

**Verification**: 158 passed, 6 skipped locally (unchanged EC-gated
skips); separately re-ran the full suite against the real pure-Python
EC backend (`JANUS_ALLOW_PURE_PYTHON_GROUP=1`) after touching
`fl/round_loop.py` -- 232 passed, 0 failed, in ~4m12s.

**Not yet done** (next step, notebook layer): wire
`on_pseudonym_flagged` in the actual run notebooks, which already
build a real DKG committee (`PK_trace, key_holder_shares`) for escrow
-- the callback can call the already-validated (both locally and on
real Colab rbcl) `run_opening_protocol`/`revoke()` for real, for each
flagged pseudonym, giving the paper's per-round "traceability happened
at round N" result and the H(T) curve. Not yet run on Colab (no
torch/EC in this sandbox for the full `round_loop.py` path -- same
limitation as every torch-dependent module here).

## §6.2 opening protocol wired into all 8 run notebooks, real committee (2026-11-19, same day)

Completes the traceability integration started above: `on_pseudonym_flagged`
(fl/round_loop.py) is now wired, for real, in every MVP run notebook --
the main training matrix (4 clean + 4 attack7), not just the standalone
demo. User authorized proceeding through the full pipeline
autonomously ("do what needs to be done... while updating the .md
files... giving the perfect systematic way of working").

**What changed, concretely**:
- `notebooks/_build_mvp_run_notebooks.py`'s shared `RING_CELL` now also
  Feldman-verifies every DKG sub-share (same check
  `notebooks/traceability_demo` already ran standalone -- held this
  notebook's committee to the same standard now that opening runs for
  real from it) and computes `pk_shares` (each holder's PUBLIC Feldman
  commitment, via the already-tested `pk_share_for`) -- needed by
  `run_opening_protocol`'s `verify_partial` step, which the ring cell
  never computed before since nothing used it.
- New shared `OPENING_HANDLER_CELL`: defines `on_pseudonym_flagged`,
  wired straight to `fl/trace.py`'s already-validated
  `run_opening_protocol` + `revoke` + `residual_anonymity_bits` --
  the SAME functions the standalone demo proved correct against both
  the pure-Python backend and real `rbcl` on Colab, nothing new
  introduced. Collects a real Chaum-Pedersen partial from every key
  holder (none of the 10 holders is ever one of the 50 clients in this
  simulation, so nobody ever recuses as the accused) and appends one
  entry per flagged pseudonym to `traceability_log` (round, opened,
  theta, evidence total, and -- when opened -- the new ring size and
  H(T)). Tracks a REPORTING-ONLY shrinking ring (`_opening_ring`) for
  the H(T) curve -- deliberately does NOT feed back into this run's
  own ring-signature verification ring, since actually reconfiguring
  the live ring mid-run is §6.4 (`crypto/reshare.py`), still untested
  and out of scope.
- `run_cell`/`attack_run_cell`: `on_pseudonym_flagged=on_pseudonym_flagged`
  added to the `run_training_loop(...)` call; both print a "§6
  traceability summary" after the run, built from `traceability_log`
  -- this IS the paper's "traceability happened at round N" result,
  for real per-round evidence, not a synthetic demo case.
- Regenerated the 6 auto-generated notebooks (3 clean + 3 attack7).
  The 2 hand-built/frozen notebooks this generator never overwrites
  (`RUN_MVP_TINY_CNN_MNIST.ipynb`, `..._WITH_ATTACK.ipynb` -- the
  baseline and the attack7 variant about to be re-run from scratch)
  were hand-patched with the byte-identical cell content instead of
  regenerated, so neither loses its hand-built history; verified the
  patch produced the exact same ring-cell/handler-cell/run-cell text
  the generator itself emits for the other 6.

**Verification possible without Colab (no torch/GPU here, same
limitation as every training-path change this whole project)**: all 8
notebooks re-checked as valid JSON; every non-magic code cell
(`!pip`/`!git` cells excluded, as always) parses as valid Python via
`ast.parse` -- including the 2 newly-inserted opening-handler cells
and both patched run cells. 159 passed, 6 skipped locally (one more
than the last entry -- a new `EpochEvidenceState.certificate` property
test); separately re-ran the full suite against the real pure-Python
EC backend after touching `fl/round_loop.py` again -- 233 passed, 0
failed, in ~4m14s. **Not yet run on Colab.**

**Expected result once run** (not yet observed -- a prediction worth
recording before it's confirmed, so a surprising outcome is visible as
surprising): a clean (all-honest) run should show ZERO traceability
events across all 20 rounds (every pseudonym's evidence stays under
the calibrated theta). An attack7 run should show the 7 attackers'
pseudonyms crossing theta and getting opened very early (their forced
delta is a model-replacement-scale outlier every single round they're
eligible -- evidence accumulation is independent of, and in addition
to, V3's existing hard per-round rejection), giving the paper a real
"traceability happened at round N" number for all 4 (model, dataset)
combinations once their attack7 variants are (re-)run under this
wiring.

## Combined attack7-traceability notebook for the 3 remaining combos (2026-11-19, same day)

User is re-running `tiny_cnn+MNIST+attack7` themselves (already needed
a fresh run, per the round-10-gap item above). Asked whether the other
3 attack7 combos (`small_resnet+MNIST`, `tiny_cnn+FEMNIST`,
`small_resnet+FEMNIST`) should also get a fresh traceability run, and
if so how to run them efficiently rather than needing 3 separate Colab
tabs. Flagged first: each of those 3 already has a FINAL, good
accuracy/security checkpoint from before traceability wiring existed
-- **resuming any of them produces zero traceability output**, since
`run_training_loop`'s round loop (`range(start_round, rounds)`) is
empty when `start_round` already equals `rounds`. Getting a real
result needs a fresh run from round 0.

User chose: run `tiny_cnn+MNIST+attack7` themselves now, and asked for
ONE combined notebook covering the other 3 combos sequentially, with
checkpoints saved "has to be different" (distinct from each combo's
existing final branch).

Built `notebooks/_build_attack_traceability_combo_notebook.py` ->
`RUN_MVP_ATTACK7_TRACEABILITY_COMBO.ipynb`: one notebook, 40 cells,
running `tiny_cnn+FEMNIST`, `small_resnet+MNIST`, `small_resnet+FEMNIST`
back-to-back in a single Colab session (~15 hours total, unattended).
Reuses every cell-generating function from
`notebooks/_build_mvp_run_notebooks.py` directly (dataset_cell,
RING_CELL, OPENING_HANDLER_CELL, model_cell, git_checkpoint_cell,
attack_run_cell, eval_cell) and `fl/attacks/mvp_matrix.py`'s
`ATTACK_CASES` for the attack parameters -- nothing here reimplements
a cell that generator already owns, so this notebook can't silently
drift from the other 4 attack7 notebooks' own content.

**Each combo gets its own checkpoint branch AND its own clone
directory**: `<combo>-attack7-traceability` (distinct from
`<combo>-attack7`, which keeps its existing final result completely
untouched) and `/content/FL-ZKP-checkpoints-trace-{0,1,2}`. The
dedicated-clone-dir part matters mechanically, not just for
tidiness: `git_checkpoint_cell`'s generated code only clones + checks
out a branch inside an `if not os.path.exists(CKPT_REPO_DIR):` guard
-- reusing ONE clone dir across 3 combos in the same notebook would
mean combo 2 and 3's cells see the dir already exists and skip
straight to "reuse it", never checking out their OWN different
branch. Caught this by reading the existing template closely before
writing the generator, not by trial and error.

Between combos, a small cell copies `traceability_log`/
`_n_opened_so_far` into combo-specific names
(`traceability_log_case2_tiny_cnn`, etc.) before the next combo's
`OPENING_HANDLER_CELL` cell resets both to empty for its own run; a
final combined cell prints all 3 combos' results together.

**Verification possible without Colab**: notebook valid JSON, 40
cells in the expected order (title/setup once, then 3 full combo
blocks, then one combined summary); every non-magic code cell parses
(`ast.parse`); each combo's `git_checkpoint_cell` call confirmed to
carry a distinct `CKPT_REPO_DIR`/`CKPT_BRANCH` (`trace-0`/`trace-1`/
`trace-2`, `...-tiny_cnn-attack7-traceability` /
`...-small_resnet-attack7-traceability` x2). 159 passed, 6 skipped
(unchanged -- no library code touched this entry, generator-only).
**Not yet run on Colab.**

## Combo notebook widened to all 4 attack7 combos (2026-11-20)

User clarified: the "cnn mnist old one with attack" they're running
right now is the standalone `RUN_MVP_TINY_CNN_MNIST_WITH_ATTACK.ipynb`
(already traceability-wired, per the earlier same-day entry) -- but
they want tiny_cnn+MNIST included in the combo notebook too, not left
out of it.

`notebooks/_build_attack_traceability_combo_notebook.py`:
`COMBO_CASES` widened from 3 to all 4 `fl/attacks/mvp_matrix.ATTACK_CASES`
entries. tiny_cnn+MNIST's copy here checkpoints to its OWN branch
(`colab-checkpoints-case1-tiny_cnn-attack7-traceability`) -- distinct
from both its existing final branch AND the standalone notebook's own
branch (which the user's separate, concurrent run is currently
writing to) -- so running both at once cannot collide; it's redundant
compute, not a correctness risk. Regenerated
`RUN_MVP_ATTACK7_TRACEABILITY_COMBO.ipynb`: now 51 cells, 4 combo
blocks (`tiny_cnn+MNIST`, `tiny_cnn+FEMNIST`, `small_resnet+MNIST`,
`small_resnet+FEMNIST`), one combined summary at the end. Verified all
4 `CKPT_BRANCH`/`CKPT_REPO_DIR` pairs distinct from each other (confirmed
by printing every generated cell's values, not just asserted) and from
the standalone notebook's branch. 159 passed, 6 skipped (generator-only
change). **Not yet run on Colab.**

## A1 malicious-fraction grid + clean-run traceability confirmation notebooks (2026-11-20)

Two new notebooks, both decided after a direct investigation into what
LEFT.md §1.6's "attack-type x fraction grid" and the attack7-combo
notebook's "traceability confirmed" claim actually cover, surfaced to
the user before building anything.

**1. `RUN_MVP_A1_FRACTION_GRID.ipynb`** (via new
`notebooks/_build_attack_fraction_grid_notebook.py`). LEFT.md §1.6
originally scoped {A1, A3, A4-replay, A5} x {10%, 20%, 30%} malicious
clients. Checked what the real per-round pipeline actually supports
before building: only A1 (model-replacement scaling) has real
per-round integration (`fl/client.py`'s `_force_delta`, exercised by
every attack7 notebook). A3 (sybil), A4-replay, and A5 (splice) attack
the ring-signature/VRF/key-image layer at the submission-construction
step, not the trained delta -- they are deliberately tested in
isolation instead (`RUN_ATTACKS.ipynb`, 8/8 caught, F10/F11's own
writeup already argues this is correct scope, not a gap). Wiring them
into real per-round submissions would be new, materially riskier
engineering in `fl/round_loop.py`/`fl/client.py`, not a notebook
addition. Put to the user directly; decided: **A1-only** grid. Built:
3 cells, 10%/20%/30% malicious of 50 (5/10/15 clients), 5 rounds each
(short on purpose -- the per-round V3-rejection claim doesn't need a
long run), same tiny_cnn+MNIST base case as LEFT.md item #1, each cell
on its own dedicated checkpoint branch
(`colab-checkpoints-case1-tiny_cnn-fracgrid-{10,20,30}pct`) and clone
directory, no §6.2 traceability wiring (out of scope for this grid --
pure V3 defense-trigger-rate table). Final cell prints a combined
fraction-vs-(completed/aborted/rejected/accepted) summary table.
Reuses every shared cell from `_build_mvp_run_notebooks.py` verbatim
except the new parametrized `fraction_grid_run_cell`.

**2. `RUN_MVP_CLEAN_TRACEABILITY_CONFIRM.ipynb`** (via new
`notebooks/_build_clean_traceability_confirm_notebook.py`). User asked
directly whether the attack7-combo notebook covers the "zero false
positives" half of the traceability claim -- it does not: it only
ever runs all-attacker combos. Every existing CLEAN checkpoint in this
repo predates §6 traceability wiring (2026-11-19), so there is no log
anywhere showing an honest run producing zero flags; this is a real,
previously-unflagged gap, not something already covered under another
name. Built the same way the attack7-combo notebook was: all 4 clean
model/dataset combos, each run FRESH from round 0 (resuming a finished
clean checkpoint would produce zero rounds, same reasoning as the
attack-combo notebook), each on its own new `-clean-traceability`
checkpoint branch and clone directory so no existing final clean
result is ever touched. Reuses `_build_mvp_run_notebooks.py`'s own
`run_cell` (already threads `on_pseudonym_flagged` into
`run_training_loop`) verbatim -- no new attack/eval logic, only the
combo-assembly structure (copied from the attack7-combo notebook's own
pattern) and a `tiny_cnn+MNIST` case1 config block (that combo's clean
notebook predates `RUN_CONFIGS` and was never folded into it). Final
cell reports, per combo, how many pseudonyms were flagged (expected:
0) and prints a plain-language verdict either way -- a nonzero result
would be a real limitations-section finding, not something to hide.

**Verification (both notebooks)**: valid JSON; every real code cell's
Python syntax parses cleanly (`ast.parse`, Jupyter magic/shell lines
replaced with `pass` rather than blanked, which an earlier pass of
this same check got wrong and had to be corrected before trusting the
result); all checkpoint branches/clone directories confirmed pairwise
distinct from each other and from every other notebook's branches in
this repo (grid: 3, clean-traceability: 4, both new); 159 passed/6
skipped, unchanged (notebook-only change, no `fl/`/`crypto/` touched).
**Not yet run on Colab** -- same limitation as every notebook in this
repo built in this sandbox.

## Benchmarked baseline attacks (sign-flip, Gaussian noise, scale-to-threshold, low-and-slow) + `cheater_delta_transform` hook (2026-11-20)

User asked, after a rigorous web search (3 `WebSearch` calls, ~30
sources) for recent Byzantine-FL benchmark attacks, to build the ones
that are both standard-enough that a reviewer would expect them and
safe to run without touching the paper's own threat model -- then
merge them into a combined grid notebook alongside the already-built
A1 fraction grid, with §6.2 traceability wired in.

**Why A1's existing `_force_delta` couldn't carry these.** `_force_delta`
*replaces* a client's real trained delta outright with a hand-picked
vector -- right for A1's model-replacement attack, wrong for sign-flip
(`g_m = -g_b`), additive noise, or any norm-adaptive attack (scale to
just under `B_t`, or well under it) -- all of these need the client's
*own real* Step-1 output (or, for the norm-adaptive pair, this round's
active `B_t`) as an input, not a value chosen in advance. Negating or
scaling a hand-picked replacement would not be the attack the
literature defines.

**New engineering (purely additive, `_force_delta` untouched):**
- `fl/client.py::round_step_phase1` gained `_delta_transform:
  Callable[[np.ndarray], np.ndarray] | None` -- applied to Step 1's
  real `delta` (trained, or zero if not VRF-sampled) only when
  `_force_delta is None` (checked first; `_force_delta` wins if both
  are somehow given, though that combination is asserted against one
  level up).
- `fl/round_loop.py::run_training_loop` gained `cheater_delta_transform:
  Callable[[np.ndarray, float], np.ndarray] | None`, called as
  `cheater_delta_transform(real_delta, threshold.B_t)` for every index
  in `cheater_client_idx`, plumbed down via a closure over
  `threshold.B_t` so `fl/client.py` itself stays `B_t`-agnostic. A new
  assertion (`cheater_force_delta is None or cheater_delta_transform is
  None`) fires immediately after normalizing `cheater_client_idx`,
  before `suspect_tracker` is created -- the two are mutually
  exclusive by construction, not just by convention.
- The existing `_cheat_zero_sketch` condition (round_step_phase2 call
  site) was extended to also require `cheater_delta_transform is None`
  -- a `cheater_delta_transform` client submits a genuinely,
  honestly-sketched crafted delta and must not get the
  lying-zero-sketch treatment A1's clients get.

**New module, `attacks/baseline_norm_attacks.py`** (4 craft functions,
full citations in its own docstring):
- `craft_sign_flip`: `g_m = -g_b` (Shi et al. 2021).
- `craft_gaussian_noise`: `g_m = g_b + N(0, sigma^2 I)` (Shi et al. 2021's
  "Noise Attack", distinct from their delta-replacing "Random Attack").
- `craft_scale_to_threshold`: rescale the real delta to
  `safety_margin * B_t` (default 0.95), just under this round's R_norm
  bound rather than wildly past it like A1's `gamma=100` -- the
  realistic version of a scaling attack per Lycklama et al. 2025
  ("Hammer and Anvil"), which shows a single adaptive client scaled to
  exactly a norm-bounding defense's threshold breaks norm-bounding
  outright, for any functional threshold.
- `craft_low_and_slow`: this project's own extension (not from any
  paper) -- rescale to `target_ratio * B_t` (default 0.3), deliberately
  smaller, testing whether staying well under B_t also keeps the §6.1
  evidence term small enough to evade the epoch-accumulated theta
  check too, i.e. whether patience beats both defenses at once.
- Shared `_scale_to_norm` helper; both scale-based attacks fall back to
  a random unit direction (then scaled to the target norm) when the
  real delta is exactly zero (not VRF-sampled), rather than silently
  submitting zero -- a strictly stronger adversarial assumption.

**Deliberately excluded** (and why, so this isn't re-litigated later):
true LIE/Min-Max/Min-Sum (Baruch et al.; Shejwalkar & Houmansadr) all
require the attacker to observe *other* clients' real raw gradients
each round -- directly contradicts this project's own secure-
aggregation/masking/sketching threat model, which exists specifically
to deny that visibility. Building them would mean breaking the
paper's own privacy guarantee to attack it. Backdoor-objective attacks
(BackFed, RING) are a genuinely different attack goal (targeted
misclassification via trigger) and out of scope for this round.

**Tests** (new, this session): `tests/test_baseline_norm_attacks.py`
-- 15 pure-NumPy unit tests (shape preservation, sign-flip negation,
exact-norm correctness post-scaling via `_scale_to_norm`, the
zero-real-delta fallback path, and that `craft_scale_to_threshold` and
`craft_low_and_slow` produce different norms at their different
defaults) -- run unconditionally, no EC backend needed.
`tests/test_cheater_delta_transform.py` -- 7 integration tests of the
real `_delta_transform`/`cheater_delta_transform` wiring, gated like
`test_round_end_to_end.py` (skips cleanly with no group backend;
`JANUS_ALLOW_PURE_PYTHON_GROUP=1` enables the pure-Python backend used
to run them here). Confirms: `_delta_transform` is called with Step
1's real (zero, since `sampling_rate=0.0` forces `b_k=False`) delta
and its result reaches a real commitment/signature; the honest
no-transform default is unaffected; `_force_delta` wins when both are
given directly to `round_step_phase1` (the one caller-level escape
hatch the docstring documents, unreachable via `run_training_loop`);
a transform producing a genuinely non-zero vector survives the full
real crypto pipeline; and `run_training_loop`'s mutual-exclusivity
assertion fires for the forbidden combination but not for either
argument alone (confirmed by getting past it to `fl/models.py`'s
`_require_torch()` RuntimeError instead, since torch genuinely isn't
installed in this sandbox -- that RuntimeError, not an AssertionError,
is itself the proof the assert didn't fire). Discovered along the way:
`fl.client`/`fl.round_loop` import successfully under
`JANUS_ALLOW_PURE_PYTHON_GROUP=1` in this sandbox, contrary to
`test_eval_fn_hook.py`'s own docstring precedent (written before this
env var's test-only pure-Python backend existed) -- real integration
tests of this new wiring are therefore possible here, not just code
review.

**Verification**: `pytest tests/ -q` -- 174 passed/7 skipped without
the env var (up from 159/6: +15 unconditional tests, +1 newly-skipped
module), 255 passed with `JANUS_ALLOW_PURE_PYTHON_GROUP=1` set (every
previously-skipped test, including the new integration file, runs and
passes). Zero regressions either way.

**Not yet done**: the combined grid notebook itself (sign-flip/noise/
scale-to-threshold/low-and-slow x 10%/20%/30% malicious, tiny_cnn+MNIST,
§6.2 traceability wired in) -- this entry covers the engineering and
tests it depends on, not the notebook. Also pending: the user's prior
ICIST Byzantine-robust conference paper's own attack list, asked for
but not yet provided (checked memory: only submission-status metadata
on file, no attack definitions) -- to be folded in once supplied, if
it passes the same buildability/safety filter applied above.

## `RUN_MVP_BASELINE_ATTACK_GRID.ipynb`: combined baseline-attack grid, §6.2 traceability wired in (2026-11-20)

The notebook the previous entry's engineering (`cheater_delta_transform`,
`attacks/baseline_norm_attacks.py`) was built for. Via new
`notebooks/_build_baseline_attack_grid_notebook.py`: 4 attacks
(sign-flip, Gaussian noise, scale-to-threshold, low-and-slow) x
{10%, 20%, 30%} malicious of 50 clients = 12 cells, `tiny_cnn`+MNIST
(same base case as the A1 fraction grid and every attack7 notebook),
10 rounds each, each cell on its own dedicated checkpoint branch
(`colab-checkpoints-case1-tiny_cnn-baselinegrid-<attack>-<fraction>`)
and clone directory.

**Does NOT replace `RUN_MVP_A1_FRACTION_GRID.ipynb`** -- per the
"never delete, always add" convention and the user's own "keep all
existing results" decision, this is an ADDITIONAL notebook covering
the 4 new attacks the A1 grid was never scoped for (that one is
A1-only, no traceability, already committed/final).

**§6.2 traceability wired in** (the A1 grid deliberately left this
out as out of scope for its own pure V3 defense-trigger-rate table) --
each of the 12 cells builds its own real DKG committee (`RING_CELL`)
and real opening-protocol handler (`OPENING_HANDLER_CELL`), exactly
like the attack7-traceability combo notebook, so a real evidence-theta
crossing triggers a real identity opening against real escrow
ciphertexts, not a simulated one.

**Attack-to-transform mapping** (each via the new `cheater_delta_transform`
hook, not `cheater_force_delta`):
- sign-flip: `craft_sign_flip(d)`.
- Gaussian noise: `craft_gaussian_noise(d, sigma=0.5*B_t)` -- sigma
  scaled to half the round's active norm bound rather than a fixed
  constant, so the perturbation is comparably sized across rounds/
  cells regardless of the real delta's own magnitude.
- scale-to-threshold / low-and-slow: `craft_scale_to_threshold(d, B_t)`
  / `craft_low_and_slow(d, B_t)` at their module defaults (0.95 and
  0.3 respectively).
- A per-cell `_attack_rng = np.random.default_rng(CFG['seed'] + 1000 + i)`
  seeds each attack's own randomness (noise draws; the zero-real-delta
  fallback direction) deterministically, on a separate stream from
  `root_norms`' own bootstrap RNG.

**Honest limitation, called out in the notebook's own title markdown
and its final summary cell, not just here**: 10 rounds settles
sign-flip/noise/scale-to-threshold's claim (does V3 catch it, does it
get traced) the same way the A1 grid's 5 rounds did, but low-and-slow
is a DIFFERENT kind of claim -- it is specifically a patience attack
testing whether staying under both V3's bound and the accumulating
§6.1 evidence threshold `theta` survives a WHOLE epoch
(`configs/mvp_3day.yaml`'s `epoch_rounds: 25`). 10 rounds shows the
evidence accumulating but is not a definitive epoch-length answer for
that one attack specifically -- a dedicated >=25-round run of just
that cell would be needed before citing a "survives" or "caught"
verdict for low-and-slow in the paper. The other 3 attacks' claims do
not have this limitation.

**Verification**: valid JSON (92 cells); every real code cell's Python
syntax parses cleanly (`ast.parse`, magic/shell lines replaced with
`pass`); all 12 new checkpoint branches and all 12 new clone
directories confirmed pairwise distinct from each other and from
every other notebook's branches/dirs in the repo (31 distinct branches
total across the whole repo, zero collisions; clone-dir collisions
found are pre-existing and shared only among the 4 clean/4 attack
notebooks respectively per the existing convention, untouched by this
notebook). One bug caught and fixed before this check passed: the
final summary cell's low-and-slow caveat line had a nested-quote
f-string/plain-string concatenation error (an unescaped `"` broke the
generated cell's own syntax) -- fixed by making the whole line one
proper f-string in the generator. `pytest tests/ -q`: 174 passed/7
skipped, unchanged (notebook-only change, no `fl/`/`crypto/`/`attacks/`
touched since the previous entry). **Not yet run on Colab.**

**Notebooks the user needs to run now** (per the original "give me the
list of notebooks i have to run" ask): this new
`RUN_MVP_BASELINE_ATTACK_GRID.ipynb`; the already-committed
`RUN_MVP_A1_FRACTION_GRID.ipynb` and
`RUN_MVP_CLEAN_TRACEABILITY_CONFIRM.ipynb`; the already-committed
`RUN_MVP_ATTACK7_TRACEABILITY_COMBO.ipynb` (covers the remaining 3
attack7 combos' traceability, plus a redundant-but-harmless 4th);
and whichever of the user's own already-running standalone notebooks
(the tiny_cnn+MNIST+attack7 traceability run) they still want results
from.

**Follow-up, same day: rounds raised from 10 to 20.** User asked to
match the 20-round length every attack7 notebook already uses,
instead of the shorter length this grid started with. `REAL_ROUNDS`
changed in `notebooks/_build_baseline_attack_grid_notebook.py`
(generator) only -- no change to `fl/`/`crypto/`/`attacks/`, the
attack-to-transform mapping, the checkpoint branch/clone-dir names, or
the cell structure. Low-and-slow's epoch-length caveat is UPDATED, not
removed: 20 rounds is closer to `epoch_rounds=25` than 10 was, but
still one short of a complete epoch, so the notebook's own markdown
and this entry still say a reviewer-safe low-and-slow verdict needs a
dedicated run of at least 25 rounds for that cell specifically.
Re-validated: 92 cells, valid JSON, every code cell's syntax parses,
`REAL_ROUNDS = 20` confirmed present in the generated notebook; branch/
clone-dir names unchanged from the earlier distinctness check (round
count doesn't affect naming); `pytest tests/ -q`: 174 passed/7 skipped,
unchanged. **Not yet run on Colab.**

## Split `RUN_MVP_BASELINE_ATTACK_GRID.ipynb` into 5 parallel-runnable parts (2026-11-20)

User wanted the ~60-hour total runtime spread across 5 separate
Colab/Kaggle sessions rather than one long sequential one, and asked
to preserve the git-checkpoint cells exactly as they already had them
set up (a real GitHub PAT filled in, plus `/content` repo-dir paths
changed to `/kaggle/working` to run on Kaggle) rather than regenerate
them with a placeholder.

**New generator**: `notebooks/_build_baseline_attack_grid_split_notebooks.py`
-> `RUN_MVP_BASELINE_ATTACK_GRID_PART{1..5}.ipynb`. Split by original
grid index (0-11, unchanged from the single notebook's own `GRID_CELLS`
order), 3/3/2/2/2 cells per part:
- Part 1: sign-flip, all 3 fractions.
- Part 2: Gaussian noise, all 3 fractions.
- Part 3: scale-to-threshold @ 10%/20%.
- Part 4: scale-to-threshold @ 30% + low-and-slow @ 10% -- the one
  part that straddles an attack-type boundary, unavoidable splitting
  12 cells into 5 roughly-even parts; flagged in that part's own
  title markdown, not silently done.
- Part 5: low-and-slow @ 20%/30%.

Each part is fully standalone (its own clone/install/sanity/config/
dataset cells) and keeps every cell's ORIGINAL checkpoint branch and
clone-dir name unchanged -- a checkpoint started in a split part is
the exact same branch the single notebook would have used for that
cell, so the single notebook and the 5 parts can never collide with
each other, and nothing about a cell's identity changes by which file
it ended up in. Each part ends with its own summary cell scoped to
only the slugs it actually contains (not the original's all-12-slug
summary, which would `NameError` on any part missing a slug) --
the low-and-slow epoch-length caveat note is included only in the
parts that actually contain a low-and-slow cell (parts 4 and 5).

**Live-credential handling, flagged plainly rather than acted on
silently**: the user's uploaded working copy had a real GitHub PAT
(`ghp_...`) hardcoded in all 12 git-checkpoint cells (replacing this
project's own placeholder comment), plus `CKPT_REPO_DIR` changed from
`/content/...` to `/kaggle/working/...`. Diffed the upload against the
already-committed single notebook first: confirmed the ONLY cells
that differ are exactly those 12 checkpoint cells -- everything else
(dataset/model/RING/opening-handler/run cells) is byte-identical to
what's already on GitHub, so nothing else needed reconciling.

Two separate outputs, deliberately not mixed:
1. **Committed to this repo**: the 5 split notebooks built with this
   project's normal PLACEHOLDER checkpoint cells (via
   `gen.git_checkpoint_cell`, same template every other grid notebook
   uses) -- no real token anywhere in git history.
2. **Delivered to the user directly (NOT committed)**: 5 "personalized"
   copies with the user's own real token + `/kaggle/working` paths
   patched back in (by matching each cell's `CKPT_BRANCH` string, not
   by positional guessing), byte-for-byte identical to their own
   working checkpoint cells, split across the same 5 files. A live
   PAT embedded in a notebook file is a real exposure if that file is
   ever made public, forked, or uploaded somewhere else -- it was not
   committed to git for that reason, and the user was told so plainly
   in this session (not silently decided).

**Verification**: all 5 placeholder parts and all 5 personalized
copies -- valid JSON, every code cell's syntax parses cleanly
(magic/shell lines blanked to `pass` first); total cells 29+29+22+22+22
= 124 across the 5 parts (expected, since each part repeats the 7
shared setup cells, unlike the single 92-cell notebook which has them
once); all 12 checkpoint branches and all 12 clone-dirs across the 5
parts match the single notebook's own 12 EXACTLY (set equality
confirmed programmatically), zero collisions among the 5 parts or
with the single notebook or any other notebook in this repo.
`pytest tests/ -q`: 174 passed/7 skipped, unchanged (notebook-only
change). **Not yet run on Colab/Kaggle** -- these are an alternate
way to run the same 12 cells, not a new result.

## `RUN_MVP_FINAL_GRID.ipynb` built -- the "final run" covering all 4 combos x clean/attack7, §6.2 traceability wired in (2026-11-20)

**User's ask**: "create the final run with training attack and tracebility! as final grid notebook to be run which will have all the values and acuracy and evaluation values of all the 4 combination and attacks, after checking there must not be anything that should be left."

**Checking phase, done first (before writing any new code)**: re-read STATUS.md's full "Top TODO list" and "What is NOT cut" sections top to bottom. Found that at the library level nothing is missing for this ask -- the clean 20-round accuracy matrix (STATUS.md's TWENTY-FOURTH item: tiny_cnn+MNIST 98.52%/97.88%, small_resnet+MNIST 98.91%/98.52%, tiny_cnn+FEMNIST 78.18%/77.43%, small_resnet+FEMNIST 70.13%/69.86%) and the attack7 20-round accuracy+security matrix (TWENTY-SEVENTH item, all 4 combos confirmed, V3 rejecting all 7 attackers every round) are both already complete and Colab-confirmed. §6.2 traceability is wired into the library and into all 8 per-combo run notebooks (TWENTY-EIGHTH/NINTH items) and validated standalone (TWENTY-FIFTH/SIXTH items), but had not yet been **run** alongside training+eval for any combo, and the two notebooks built specifically for that (`RUN_MVP_CLEAN_TRACEABILITY_CONFIRM.ipynb`, `RUN_MVP_ATTACK7_TRACEABILITY_COMBO.ipynb`, THIRTY-SECOND/THIRTIETH items) had not been run on Colab yet either, and each only covers half (clean-only or attack7-only) of the 8 combinations.

**Decision taken, given the significant runtime cost of guessing wrong**: rather than a full from-scratch retrain of everything (which would re-spend the already-confirmed 4 clean + 4 attack7 accuracy numbers for no new information), built ONE merged notebook, `notebooks/_build_final_grid_notebook.py` -> `RUN_MVP_FINAL_GRID.ipynb`, that runs all 8 combos (4 clean + 4 attack7) fresh from round 0, each with §6.2 traceability wired in -- reusing the exact cell-generation sequences of the two existing traceability notebooks verbatim (no new library code, no new wiring), merged into one file, plus a new final summary cell reporting accuracy + V3/security outcome + §6.2 traceability outcome for all 8 runs side by side -- the literal "all the values and accuracy and evaluation values of all the 4 combination and attacks" table asked for.

**Checkpoints**: 8 brand-new dedicated branches (`<branch>-final-clean` / `<branch>-final-attack7`, one per combo, own clone dir too) -- distinct from every branch already in the repo (the plain clean/attack7 branches, and the two standalone traceability notebooks' own `-clean-traceability`/`-traceability` branches), confirmed via the repo's usual branch/dir uniqueness grep across every notebook. Zero new collisions introduced (the only matches found are the single baseline-attack-grid notebook vs. its own pre-existing 5-way split, unrelated to this file).

**Runtime, stated plainly in the notebook's own title cell before any cell runs**: ~30+ hours sequential (4 clean combos at a few hours each + 4 attack7 combos at ~5h each) in one Colab session -- the most expensive single artifact built this session. The title cell also names the two cheaper alternatives (run the two standalone notebooks separately, or ask for this one to be split the same way the baseline-attack grid was split into 5 parts) in case 30+ hours in one sitting isn't what's wanted.

**Verification**: valid JSON (103 cells), every code cell's syntax parses (same `ast.parse`-after-blanking-magic-lines convention used throughout this repo), all 8 of this notebook's own checkpoint branches/dirs confirmed unique across every notebook in the repo. `pytest tests/ -q`: 174 passed/7 skipped, unchanged (notebook-only change, no library touched). **Not yet run on Colab.**

## A3 (Sybil) and A4-replay confirmed inside a REAL round, not just isolated primitives (2026-11-21)

**Context**: `RUN_ATTACKS.ipynb`'s own A1-gate markdown (written 2026-11-15) already named this as a known TODO: "A3 (Sybil) and A4-replay are also genuinely per-round attacks and belong here too in a future pass." This is that pass, done after the user explicitly asked for every attack to get equal, real-round treatment (not just A1).

**What was actually checked first, before writing any code**: whether A3/A4-replay/A4-false-claim/A5/A7 could all be wired into `run_training_loop` the same way A1 was (via `cheater_client_idx`/`cheater_force_delta`). Found that only A1 fits that hook cleanly (it only needs to override one client's delta content, a single substitution point `run_training_loop` already supports). A3 needs a genuine SECOND submission from the same client in the same round -- `run_training_loop`'s `phase1_by_client`/`phase2_by_client` dicts are keyed one entry per client index per round, with no slot for a duplicate, and restructuring that 900-line orchestration file for this was judged not worth the regression risk against a file with a long history of subtle numerics bugs (36 prior real bugs found by running this project). A4-false-claim and A5 have no round-content dimension at all -- they attack the VRF-proof/signature algebra directly (a "lie" the honest pipeline has no code path to even construct), and A7 targets `fl/trace.py`'s identity-opening protocol, which runs on accumulated evidence BETWEEN rounds, not inside one. Forcing any of these three into an accuracy-curve/round-loop shape would fabricate a connection to training dynamics that doesn't exist -- explicitly avoided, per this project's standing "never fabricate, never silently paper over gaps" instruction.

**What WAS genuinely buildable**: A3 and A4-replay both fit a different, equally rigorous real-round harness -- built directly against `fl/server.py::run_round` (the exact same real-crypto, torch-free entry point `tests/test_round_end_to_end.py` already exercises for F7e/F7f), rather than through `run_training_loop`. New `tests/test_a3_sybil_real_round.py` (2 tests): 8 genuinely honest clients submit once each; a 9th real client (same sk, same ring position, same rho_t) submits TWICE with different crafted content -- confirms exactly one of the two is accepted, the other rejected specifically for `V2_double_submission`, round not aborted, all 8 honest clients unaffected. New `tests/test_a4_replay_real_round.py` (3 tests): 8 fresh honest clients submit against round t's `rho_t_new`; a 9th client's entry is the EXACT submission that same client genuinely produced for round t-1 (signed against `rho_t_old`), resubmitted verbatim -- confirms it's rejected specifically for `V1_ring_signature_failed`, with a control test confirming the same client's FRESH round-t submission would have been accepted (proving the rejection is about the replay, not a harness bug). Both also cross-check against their primitive-level isolated test (`attacks/a3_sybil.py`, `attacks/a4_freeride.py::run_attack_replay`) to confirm agreement.

**Run and confirmed in this sandbox** (not "pending Colab" -- these are 100% numpy/EC-primitive level, no torch needed): all 5 new tests pass against the real pure-Python ristretto255 backend (`JANUS_ALLOW_PURE_PYTHON_GROUP=1`). Full suite: 174 passed/9 skipped without the backend (unchanged except the 2 new skip entries), 260 passed with it (was 255).

**`RUN_ATTACKS.ipynb` updated** (via `notebooks/_build_attacks_notebook.py`): the A1-gate markdown's "future pass" note is now resolved with an honest account of exactly what was and wasn't extended, and a new "Gate -- A3 (Sybil) and A4-replay injected into a REAL round" section that literally runs both new pytest files as part of the notebook's own execution (subprocess, not a reimplementation) and raises loudly if either fails. The notebook generator had drifted from the live `.ipynb` (a hand-patched trace-opening-protocol gate, added 2026-11-18/19, was never folded back into the `.py` generator) -- regenerated carefully by capturing those 2 live cells first, regenerating from the updated generator, then splicing the captured cells back in at the same relative position, rather than naively overwriting and losing them. Final notebook: 31 cells (was 29), valid JSON, zero syntax errors, cell order confirmed correct.

**What this does NOT claim, stated plainly**: A4-false-claim, A5, and A7 are not "upgraded" to a round-loop treatment, and will not be -- the honest answer, given to the user directly, is that doing so would misrepresent what those attacks threaten (none of them affect model content or round acceptance in a way a round-loop wrapper would meaningfully exercise). They remain confirmed via direct, real (non-mocked) cryptographic verification, which is the correct and complete standard of rigor for that category of attack, not a lesser one.

## Gaps #1 and #3 closed: baseline robust-aggregation comparison and quantified overhead vs. plain FedAvg (2026-11-21)

**Context**: the brutal-reviewer self-assessment flagged 5 gaps that
would survive even after every pending Colab notebook finishes.
Gaps #1 and #3 are addressed in this entry (per the user's explicit
"do 3 AND 1" instruction, followed by an explicit safety-scoped green
light: "go for it, if you feel it's safe and won't hinder the system
or make our framework weak").

**Engineering constraint honored throughout**: everything below lives
in a brand-new, standalone `baselines/` package. Nothing in `fl/`,
`crypto/`, or `attacks/` was modified -- every import from those
modules is read-only. This was a deliberate choice to satisfy the
user's stated safety concern: new work that cannot regress or weaken
the already-tested pipeline, by construction, not by promise.

**`baselines/aggregation_rules.py`** -- `mean_aggregate`,
`coordinate_median`, `trimmed_mean`, `krum`, `multi_krum`
(Blanchard et al. 2017 / Yin et al. 2018), plus an `AGGREGATION_RULES`
registry dict. 14 new tests in `tests/test_aggregation_rules.py`, pure
NumPy, run unconditionally (no torch/EC backend required). **Bug
found and fixed during testing**: `multi_krum`'s bound check (`m >
len(deltas) - n_byzantine`) didn't account for each successive
internal `krum()` call needing `remaining >= 2*n_byzantine + 3`
candidates -- failed at n=10/n_byz=1 by the 6th-7th iteration of the
selection loop. Fixed to `max_m = n - 2*n_byzantine - 2`, matching the
registry's own default `m`. All 14 tests pass after the fix.

**`baselines/plain_fedavg_loop.py`** -- `run_plain_fedavg_loop`: a
standalone plaintext FedAvg loop that reuses `fl/client.py`'s real,
unmodified `local_train_delta` for every client's local step (so the
ONLY thing that ever differs between this loop and the real pipeline
is the aggregation/security layer, never the training procedure).
Syntax-checked via `py_compile` only -- torch isn't installed in this
sandbox, same limitation as every other torch-dependent module here;
not yet run end-to-end outside Colab.

**Gap #3 -- `notebooks/_build_overhead_comparison_notebook.py` ->
`RUN_MVP_OVERHEAD_COMPARISON.ipynb`** (19 cells): at tiny_cnn+MNIST,
50 clients, 3 rounds, MEASURES (never estimates) two numbers: (1)
wall-clock via `time.perf_counter()` around `run_plain_fedavg_loop`'s
plain-mean loop vs. `fl/round_loop.py::run_training_loop`'s real
masked pipeline, both completely unmodified; (2) one real client's
actual pickled `ClientSubmissionPhase1`+`ClientSubmissionPhase2`
objects (built via `fl/client.py`'s real `round_step_phase1`/
`round_step_phase2`, same ring/committee the run already built) vs.
the same client's raw delta, pickled the same way. 3 rounds is stated
explicitly, in the notebook's own markdown, as enough for a stable
timing average and NOT a substitute for the already-confirmed
20-round accuracy numbers (STATUS.md's TWENTY-FOURTH item) -- no
accuracy number from either loop in this notebook is a result in its
own right.

**Gap #1 -- `notebooks/_build_baseline_defenses_comparison_notebook.py`
-> `RUN_MVP_BASELINE_DEFENSES_COMPARISON.ipynb`** (16 cells): a grid of
5 aggregation rules (mean/coordinate_median/trimmed_mean/krum/multi_krum)
x 3 attack conditions (none, sign-flip, Gaussian-noise) at
tiny_cnn+MNIST, 50 clients, 20% Byzantine fraction (10 cheaters), 15
rounds per cell, final-round test accuracy recorded for every cell and
saved to a CSV. **Deliberately scoped to sign-flip and Gaussian-noise
only, not scale-to-threshold or low-and-slow**: those latter two
attacks are specifically built to evade JANUS-FL's OWN R_norm
adaptive threshold (`fl/trace_wiring.py`), which a plaintext comparison
loop has no equivalent of at all -- forcing them in would fabricate a
connection to a defense mechanism this grid doesn't have. Sign-flip
and Gaussian-noise are the two classic unbounded-magnitude attacks the
Krum/Multi-Krum/trimmed-mean/median literature itself benchmarks
against, so the grid still answers the reviewer's actual question with
real, standard attacks.

**What this does NOT claim, stated plainly**: every rule in the gap-#1
grid runs over PLAINTEXT deltas, never through JANUS-FL's own secure-
aggregation layer -- coordinate-median and Krum both need plaintext
visibility to compute order statistics/pairwise distances, which
secure aggregation specifically hides from the server. That tension is
real and is stated in `baselines/aggregation_rules.py`'s own module
docstring, not papered over. This work answers "how does JANUS-FL's
rejection-based defense compare to the field's standard toolkit, under
the same attacks and scale" -- it is not a claim that any of these
rules could be substituted into the real secure pipeline unmodified,
nor is the gap-#3 timing number a claim about any model/scale other
than tiny_cnn+MNIST at 50 clients.

**Status**: both notebooks are valid JSON, `ast.parse`-clean (after
blanking magic/shell lines), and cell order checked. Neither has been
run on Colab yet -- that run is still owed, same as every other
pending MVP notebook.

## Brutal-review gap #4 extended to tiny_cnn+FEMNIST: new baseline-attack grid + 5-way split (2026-11-22)

**Request**: "create #4 for tiny_CNN+FMINST, and split it in 5, while
keeping the same token value as you kept in tinyCNN+MNIST" -- extends
gap #4 ("the baseline-attack grid only covers ONE combo") to a second
combo, same 4 attacks x 3 fractions grid, now on `tiny_cnn`+FEMNIST
instead of MNIST.

**Scope, stated plainly**: this adds ONE more combo, not the full
remaining 2x2 matrix. `small_resnet+MNIST` and `small_resnet+FEMNIST`
remain unaddressed for this gap -- a scoping choice matching the
original brainstorm's own recommendation (one additional combo is
enough to show the grid isn't cherry-picked to the easiest case), not
an oversight.

**What's identical to the already-committed MNIST grid**: the 4
attacks (sign-flip, Gaussian noise, scale-to-threshold, low-and-slow),
the 3 fractions (10%/20%/30% of 50 clients), 20 rounds/cell, §6.2
traceability wiring, the `cheater_delta_transform` hook. New
`notebooks/_build_baseline_attack_grid_femnist_notebook.py` imports
`ATTACKS`/`FRACTIONS`/`GRID_CELLS`/`_TRANSFORM_EXPR` directly from the
MNIST grid's own generator module rather than redefining them, so the
two grids' attack/fraction definitions can never drift apart.

**What's different**: a new `CASE2_CFG` (`case_prefix="case2"`,
`num_classes=62`, `ckpt_branch="colab-checkpoints-case2-tiny_cnn-
baselinegrid"`); `case2_data` in place of `case1_data`; the headline
eval split is `case2_data['unseen_test']`, a `FemnistClientDataset`
**dataclass** (`.images`/`.labels` attributes), not a plain tuple --
same two-line unpack fix (`_eval_ds = case2_data['unseen_test'];
_eval_images, _eval_labels = _eval_ds.images, _eval_ds.labels`)
`_build_mvp_run_notebooks.py::run_cell` already uses for every other
FEMNIST notebook in this repo, replacing the MNIST grid's plain
`case1_data['test']` tuple unpack. New checkpoint-branch suffix
(`-baselinegrid`) distinguishes these 12 branches both from the
already-completed plain `colab-checkpoints-case2-tiny_cnn` branch and
from every `case1-...-baselinegrid-*` branch the MNIST grid uses; new
clone-dir prefix (`/content/FL-ZKP-checkpoints-baselinegrid-femnist-
{i}`) likewise distinct from the MNIST grid's own `-baselinegrid-{i}`
dirs. Confirmed via a full-repo regex scan of every `.ipynb`/`.py`
file: zero collisions anywhere.

**5-way split**: new `notebooks/_build_baseline_attack_grid_femnist_
split_notebooks.py`, same `PART_SPLIT = [[0,1,2],[3,4,5],[6,7],[8,9],
[10,11]]` grouping as the MNIST split (sign-flip/noise/scale-to-
threshold-10-20/scale-to-threshold-30+low-and-slow-10/low-and-slow-
20-30), each part fully standalone with its own scoped summary cell,
each cell keeping its ORIGINAL grid index in its branch/dir name so
split vs. single-notebook checkpoints never diverge.

**Validation**: generated `RUN_MVP_BASELINE_ATTACK_GRID_FEMNIST.ipynb`
(92 cells) and `..._PART{1..5}.ipynb` (29/29/22/22/22 cells). All 6
valid JSON; every code cell's syntax confirmed clean via `ast.parse`
after blanking magic/shell lines to `pass` (the naive all-blank
approach used earlier this session falsely flags the shared
`CLONE_CELL`'s `if/else` body on EVERY notebook in this repo,
including the already-working MNIST grid -- confirmed by running the
same check against it before trusting the result; the `pass`-based
check is the correct one and found zero real syntax errors across all
6 new files). Full-repo branch/dir collision scan: zero collisions.
`pytest -q`: 188 passed/9 skipped, unchanged (notebook-generator-only
change, no library code touched).

**Live-credential handling, same rule as every prior personalized
copy**: the single notebook and 5 split parts committed to this repo
use the normal `getpass()`-based placeholder checkpoint cell -- no
real token anywhere in git history. Five separate "personalized"
copies (real GitHub PAT, `/kaggle/working` paths, same transform
diffed from the cached MNIST personalized files: add `import os`,
drop the `getpass` import, hardcode `_token = "ghp_...``" in place of
the `getpass(...)` call) are delivered directly to the user via file
transfer, never committed.

**Status**: both notebooks are valid JSON, `ast.parse`-clean, cell
order checked, zero branch/dir collisions. Not yet run on Colab/
Kaggle -- that run is still owed, same as every other pending MVP
notebook.

## Two real bugs caught on Kaggle: both gap-#1/#3 notebooks fixed (2026-11-22)

**User ran both notebooks for real on Kaggle and both failed on their
first real code cell** -- caught exactly the kind of bug that can only
surface on a real run, not from `ast.parse`-only validation.

**Bug 1 -- `RUN_MVP_OVERHEAD_COMPARISON.ipynb`: `NameError: name 'np'
is not defined`.** `CRYPTO_LOOP_CELL` calls `np.random.default_rng(...)`
and `COMM_SIZE_CELL` calls `np.float64`/`np.random.default_rng(...)`,
but neither cell ever imports `numpy` -- every other cell in this
notebook gets `np` from an EARLIER cell's import in the real Colab/
Kaggle kernel, but `ast.parse`-per-cell validation checks each cell's
syntax in isolation and has no way to catch a missing import that only
breaks at actual execution time. Fixed: added `import numpy as np` to
both `CRYPTO_LOOP_CELL` and `COMM_SIZE_CELL` in
`notebooks/_build_overhead_comparison_notebook.py`.

**Bug 2 -- both notebooks: `KeyError: 'epochs'`.** Both
`_build_overhead_comparison_notebook.py` (COMM_SIZE_CELL,
PLAIN_LOOP_CELL) and `_build_baseline_defenses_comparison_notebook.py`
(RUN_GRID_CELL) read `CFG['local_training']['epochs']` -- but
`configs/mvp_3day.yaml`'s actual key (confirmed from the user's own
pasted log output, which printed the loaded `CFG` dict in full) is
`local_epochs`, not `epochs`. `run_plain_fedavg_loop`'s own PARAMETER
is named `epochs` (matching `fl/client.py::local_train_delta`'s own
parameter name) -- the bug was purely in the CFG dict LOOKUP key, not
the function signature. Every other MVP run notebook
(`_build_mvp_run_notebooks.py::run_cell`) already reads
`CFG['local_training']['local_epochs']` correctly; these two new
generators were the only place this typo was introduced. Fixed: all
3 occurrences across both generator files now read
`CFG['local_training']['local_epochs']`.

**Why `ast.parse`-only validation missed both**: `ast.parse` only
confirms a cell's Python is syntactically well-formed, not that every
name is defined or every dict key exists -- that is a real limitation
of the validation this project has been using for every notebook, not
specific to these two. Neither bug could have been caught without
either a real run or a line-by-line cross-check against the actual
`CFG` dict and every cell's own import list -- done here for both
fixes by reading the generator source directly and checking every
`np.`/`CFG[...]` use against what the cell actually imports/against
`configs/mvp_3day.yaml`'s real keys, not by guessing.

**Verification**: regenerated both notebooks, confirmed (a) zero
`ast.parse` syntax errors (same magic-line-blanking method as every
other notebook check this project does), (b) no cell uses `np.`
without importing it, (c) no cell reads the old `['epochs']` key
anywhere, (d) `pytest -q`: 188 passed/9 skipped, unchanged (notebook-
generator-only fix, no library code touched).

**Not yet re-run on Kaggle/Colab** -- both notebooks need re-running
end to end to confirm these were the ONLY two bugs; flagging plainly
that a `KeyError`/`NameError` this early (both failed on roughly the
9th real code cell, before any of the grid's own 12-15 result cells
ran) means nothing past that point in either notebook has been
exercised yet even once.

## Third real bug from the same Kaggle run: w_t reached local_train_delta as numpy, not torch.Tensor (2026-11-22)

**Error, from the grid's very first cell x attack combination**:
```
RuntimeError: Error(s) in loading state_dict for Sequential:
    While copying the parameter named "0.weight", expected
    torch.Tensor or Tensor-like object from checkpoint but received
    <class 'numpy.ndarray'>
```
(and the same for every other parameter: `0.bias`, `1.weight`,
`1.bias`, `4.weight`, `4.bias`, `5.weight`, `5.bias`, `9.weight`,
`9.bias` -- i.e. every single layer of `tiny_cnn`, on every single
client, for every rule x attack cell: this is not an edge case, it is
the main path failing on its very first call.)

**Root cause**: `baselines/plain_fedavg_loop.py::run_plain_fedavg_loop`
keeps `w_t` as a raw `numpy.ndarray` across rounds (deliberately --
every aggregation rule in `baselines/aggregation_rules.py` operates on
plain numpy deltas, by design, since that's what a plaintext
robust-aggregation comparison needs). But it then passed that SAME
numpy array straight into `fl/client.py::local_train_delta`, which
does `unflatten_to_state_dict(w_t, shapes)` then
`model.load_state_dict(state_dict)` -- `unflatten_to_state_dict` is a
pure reshape/slice operation that preserves whatever type `w_t`
already is, so passing it a numpy array produces a state_dict of numpy
array views, and `torch.nn.Module.load_state_dict` rejects those
outright. The `eval_fn` branch a few lines below this exact call
already converted `w_t` to a `torch.tensor(...)` correctly before its
own `unflatten_to_state_dict`/`load_state_dict` call -- the bug was
that the per-client training call, a few lines above, never got the
same treatment.

**Why pytest's 188/190 passing tests never caught this**: there was no
test anywhere in this repo that exercised
`run_plain_fedavg_loop`'s actual integration with
`fl.client.local_train_delta` end-to-end -- `tests/test_aggregation_rules.py`
tests the aggregation rules themselves in isolation (plain numpy in,
plain numpy out, no model/training involved at all), and this sandbox
cannot import `fl.client` or `torch` at all (no `rbcl`, no `torch`, no
network to install either -- same limitation
`tests/test_f6_batching_logic.py`'s own docstring already documents),
so a real end-to-end test literally cannot run here. This is a real
gap in test coverage, not a typo this time -- a numpy/tensor type
mismatch that only a real run, or a test that specifically asserts on
the TYPE crossing a module boundary, would ever catch.

**Fix**: `run_plain_fedavg_loop` now re-converts `w_t` to a
`torch.tensor(w_t, dtype=model_dtype)` at the top of every round,
immediately before the per-client training loop -- `w_t` itself stays
a numpy array for the aggregator call right after (unchanged), only
the COPY handed to `local_train_delta` is a tensor. Added `import
torch` to the function (previously only pulled in ad hoc inside the
`eval_fn` branch via `__import__("torch")`, replaced with a normal
import used consistently in both places).

**New test coverage added, closing the actual gap**: new
`tests/test_plain_fedavg_loop.py`, two tests. Since this sandbox can't
import the real `torch`/`fl.client`/`fl.models` (documented above),
both tests install minimal fakes via `sys.modules` injection -- a fake
`torch.tensor(...)` that produces a tagged wrapper type, and a fake
`local_train_delta` whose own `model.load_state_dict` mirrors the
REAL one's exact contract (raises on a raw `numpy.ndarray`, same error
shape as production) -- then run `run_plain_fedavg_loop`'s real,
unmodified control flow end-to-end and assert the type crossing into
`local_train_delta` is never a bare `numpy.ndarray`. **Verified this
test actually catches the regression**: temporarily reverted the fix,
confirmed both new tests fail with the identical error signature
(`RuntimeError: load_state_dict expected torch.Tensor or Tensor-like
object but received <class 'numpy.ndarray'>`) the real Kaggle run
produced, then restored the fix and confirmed they pass again. This
cannot prove the real torch accepts the real tensor end to end (that
still needs a live Colab/Kaggle run, like every other torch-dependent
path in this repo) -- but it does pin the exact type confusion down
permanently in a test this sandbox CAN run, so a future refactor that
reintroduces "pass w_t through as whatever type it already is" fails
immediately instead of waiting for the next live run to find it.

**Verification**: `pytest -q`: 190 passed (188 + 2 new)/9 skipped.
Only `baselines/plain_fedavg_loop.py` changed -- `fl/client.py` and
every other already-tested module untouched, consistent with both
notebooks' own "deliberately NOT touching fl/, crypto/, or attacks/"
scoping.

**Third bug from the same batch of live runs** -- after the missing
`numpy` import and the `epochs`/`local_epochs` CFG-key typo fixed
earlier today, this is now the third distinct real bug this one
Kaggle session surfaced, all in the same two new notebooks/module. Not
yet re-run past this point on Kaggle -- flagging again, as with the
first two fixes, that this is the FIRST real cell of the grid to
execute even once; anything further into the 15-cell grid (or the
overhead-comparison notebook's own cells, which share this same
`run_plain_fedavg_loop` call) is still unexercised.

## Fourth bug, caught by a full line-by-line audit before another live run: aggregator arity mismatch (2026-11-23)

**User asked to "check all the cells & verify the fix"** -- rather than
wait for a fourth Kaggle failure, re-read every cell of both
`RUN_MVP_OVERHEAD_COMPARISON.ipynb` and
`RUN_MVP_BASELINE_DEFENSES_COMPARISON.ipynb` line by line against the
real function signatures they call (`run_training_loop`,
`bootstrap_norms_from_model_dim`, `round_step_phase1`/`phase2`,
`ClientRoundContext`/`ClientKeys`, `run_plain_fedavg_loop`,
`AGGREGATION_RULES`, `craft_sign_flip`/`craft_gaussian_noise`), not
just re-running `ast.parse`.

**Found**: `notebooks/_build_overhead_comparison_notebook.py`'s
`PLAIN_LOOP_CELL` imported `mean_aggregate` directly
(`from baselines.aggregation_rules import mean_aggregate`) and passed
it as `aggregator=mean_aggregate`. But `run_plain_fedavg_loop` always
calls `aggregator(deltas, n_byzantine)` -- a 2-argument contract that
every `AGGREGATION_RULES` dict entry wraps around the raw rule
functions (`"mean_no_defense": lambda deltas, n_byz:
mean_aggregate(deltas)`), precisely because `mean_aggregate(deltas)`
itself only takes ONE argument. This would have raised `TypeError:
mean_aggregate() takes 1 positional argument but 2 were given` the
instant round 0 tried to aggregate -- never actually hit yet, because
this cell runs AFTER `COMM_SIZE_CELL`, which is where the overhead
notebook's run failed earlier today (the missing-`numpy`-import bug).
The baseline-defenses notebook's own grid already used the correct
`AGGREGATION_RULES[rule_name]` wrapper (via `AGGREGATION_RULES.items()`)
-- this one cell in the OTHER notebook was the only place the raw
function slipped through.

**Fix**: `PLAIN_LOOP_CELL` now imports `AGGREGATION_RULES` instead of
the raw `mean_aggregate`, and passes
`aggregator=AGGREGATION_RULES["mean_no_defense"]`.

**Everything else audited and confirmed correct** (no further bugs
found): `CRYPTO_LOOP_CELL`'s `bootstrap_norms_from_model_dim(...)` and
`run_training_loop(...)` calls match `fl/round_loop.py`'s real
signatures argument-for-argument, including `on_pseudonym_flagged`
(supplied by the shared `OPENING_HANDLER_CELL`, already included).
`COMM_SIZE_CELL`'s `ClientRoundContext`/`ClientKeys` construction and
`round_step_phase1`/`round_step_phase2` calls match `fl/client.py`'s
real dataclass fields and function signatures exactly. The baseline-
defenses notebook's `EVAL_FN_CELL`'s `_eval_fn(model) -> (acc, None)`
matches `run_plain_fedavg_loop`'s `eval_fn(model)` two-tuple-unpack
contract; `craft_sign_flip(delta)`/`craft_gaussian_noise(delta,
sigma=..., rng=...)` match `attacks/baseline_norm_attacks.py`'s real
signatures; `result.history[-1].test_acc` matches
`PlainRoundRecord`'s real field. Brace-escaping (the earlier-session
bug class) re-checked in both files: no `{x}` vs `{{x}}` mismatches
remain.

**New regression test**: `tests/test_plain_fedavg_loop.py` gained a
third test, `test_run_plain_fedavg_loop_rejects_raw_aggregation_rule_arity`,
pinning down the exact contract both ways -- the raw `mean_aggregate(deltas,
0)` call raises `TypeError` (confirmed), the `AGGREGATION_RULES["mean_no_defense"]`
wrapper does not, and `run_plain_fedavg_loop(..., aggregator=mean_aggregate)`
itself raises `TypeError` end-to-end through the same fake-torch/fl.client
harness the prior fix's tests use.

**Verification**: regenerated `RUN_MVP_OVERHEAD_COMPARISON.ipynb`;
confirmed no remaining reference to the raw `mean_aggregate` import
anywhere in the generated notebook; `ast.parse`-clean, valid JSON,
19 cells (unchanged count). `pytest -q`: 191 passed (188+3)/9 skipped.

**Fourth distinct bug found in this one module/notebook pair in 24
hours** (missing `numpy` import, `epochs`/`local_epochs` CFG-key typo,
`w_t` numpy-vs-tensor mismatch, now this aggregator-arity mismatch) --
all four are now fixed and test-covered to the extent this sandbox
allows (no torch/rbcl/network here, same documented limitation).
Genuinely not yet re-run end-to-end on Kaggle/Colab for either
notebook -- this audit found the fourth bug by static cross-checking
alone, before a live run reached it; a live run is still the only way
to confirm there isn't a fifth.

## Fifth bug, same numpy-vs-tensor class, DIFFERENT call site missed by the prior audit (2026-11-23)

**The prior "full audit" (this same day, a few hours earlier) was
incomplete.** It checked `CRYPTO_LOOP_CELL`'s and `COMM_SIZE_CELL`'s
calls against the real signatures of `run_training_loop`,
`bootstrap_norms_from_model_dim`, `round_step_phase1`/`phase2`,
`ClientRoundContext`/`ClientKeys` -- correctly, as far as argument
names and dataclass fields go -- but did NOT re-check `comm_w_t`'s
TYPE against what `round_step_phase1` actually does with it
internally (calls `local_train_delta`, which needs a real
`torch.Tensor` for `model.load_state_dict`). That is exactly the same
root confusion the `plain_fedavg_loop.py` fix diagnosed the day
before, at a DIFFERENT call site this cell independently reintroduced,
and the earlier audit's signature-matching pass missed it because
`round_step_phase1`'s own parameter is typed as plain `w_t` (no type
hint), so a signature match alone cannot catch a type mismatch -- only
checking against the ACTUAL CALLING CONVENTION every real call site
(`run_training_loop`'s own `w_t, model_shapes = flatten_state_dict(model)`,
never converted) uses would catch it.

**Error, live on Kaggle**, identical signature to the already-fixed
`plain_fedavg_loop.py` bug:
```
RuntimeError: Error(s) in loading state_dict for Sequential:
    expected torch.Tensor or Tensor-like object from checkpoint but
    received <class 'numpy.ndarray'>
```

**Root cause**: `COMM_SIZE_CELL` built `comm_w_t, _ =
flatten_state_dict(model)` (a tensor, correct) and then immediately
reassigned it: `comm_w_t = comm_w_t.detach().cpu().numpy().astype(np.float64)`
-- a leftover habit from writing `baselines/plain_fedavg_loop.py`
(which DOES need `w_t` as numpy, for its own aggregation rules) in the
SAME session, carried over into a cell that calls the REAL protocol's
`round_step_phase1` instead, which does not.

**Fix**: removed the numpy conversion entirely. `comm_w_t` now stays
exactly the tensor `flatten_state_dict` returns, matching
`run_training_loop`'s own untouched `w_t` all the way through.

**New regression test**: `tests/test_overhead_notebook_generator_static.py`
-- a static source check (this sandbox still has no torch/rbcl to
actually call `round_step_phase1`, same documented limitation as
every other torch-dependent test here) asserting the generated
`COMM_SIZE_CELL` text never reassigns `comm_w_t` from itself and that
`round_step_phase1` is called with the bare `comm_w_t` name, not a
converted copy. **Verified it catches the regression**: reintroduced
the exact buggy line, confirmed the test fails with the right
assertion message, restored the fix, confirmed it passes again. Also
added a matching static check for the aggregator-arity fix from
earlier today, in the same file, for the same reason (nothing
previously pinned that one down outside the live-call-based
`test_plain_fedavg_loop.py` tests).

**Verification**: regenerated the notebook; `ast.parse`-clean, valid
JSON, 19 cells (unchanged). `pytest -q`: 193 passed (191+2 new)/9
skipped.

**Being direct about what this means**: this is the FIFTH distinct bug
from this one notebook pair in roughly 24 hours, and the second time a
"full audit" claimed completeness and then missed something a live
run caught minutes later. The honest lesson, stated plainly rather
than glossed over: signature-matching (argument names, dataclass
fields, parameter counts) is necessary but not sufficient -- it cannot
catch a TYPE mismatch when the callee has no type hints, which is the
exact shape of bug #3 (plain_fedavg_loop.py's w_t) and bug #5
(COMM_SIZE_CELL's comm_w_t) both. Both are now fixed and
regression-tested to the extent a torch-less sandbox allows. This
notebook has still never been run past this exact cell on a live GPU
runtime -- everything past `COMM_SIZE_CELL` (the `PLAIN_LOOP_CELL` and
`SUMMARY_CELL`) remains completely unexercised by any real run to
date.

## Sixth bug, same notebook pair, a DIFFERENT class this time: device mismatch, not type (2026-11-23)

Caught live on Kaggle, running `RUN_MVP_BASELINE_DEFENSES_COMPARISON.ipynb`'s
15-round grid cell (5 aggregation rules x 3 attacks):

```
RuntimeError: Expected all tensors to be on the same device, but found
at least two devices, cuda:0 and cpu!
```

Traceback bottomed out in `fl/client.py::local_train_delta`'s final line,
`delta = (new_flat - w_t).detach().cpu().numpy().astype(np.float64)`.

**Root cause**: `baselines/plain_fedavg_loop.py::run_plain_fedavg_loop` built
`w_t_tensor = torch.tensor(w_t, dtype=model_dtype)` fresh every round (the
item-41 fix, two bugs ago) -- but with no `device=` kwarg, `torch.tensor(...)`
from a plain numpy array always lands on CPU. The grid cell's model was moved
to `cuda:0` (`grid_model = build_model(...).to(device)`), so `new_flat`
(`flatten_state_dict(model)`, inside `local_train_delta`) lives on cuda:0.
`model.load_state_dict` tolerates the mismatch (it's a data copy), but the
final `new_flat - w_t` subtraction does not. Same root class of bug as item
41 (w_t reaching `local_train_delta` in the wrong representation) but a
different axis of "wrong" -- device, not type -- so item 41's own fix (and
its test) didn't cover it.

**Fixed**: added `model_device = next(model.parameters()).device` once, and
pass `device=model_device` to both `torch.tensor(w_t, ...)` call sites
(the per-round `w_t_tensor`, and the `eval_fn` branch's own conversion).

**Caught by this same audit process missing it twice is itself the lesson
repeated**: item 42/43 established that signature-matching misses type
mismatches with no type hints; this is the same point one level further --
even checking *type* (`torch.Tensor` vs `numpy.ndarray`) isn't the full
contract when the callee also has an implicit device requirement. Neither
of the sandbox's existing fakes modeled device at all, so this bug could
not have been caught by any test written before today even in principle --
upgraded `tests/test_plain_fedavg_loop.py`'s fakes to track a fake `device`
string and raise on mismatch (mirroring real torch's own error) before
writing the new regression test, rather than writing a test that could
only ever pass.

**Verification**: reverted the fix, confirmed the new test (and, as a
side effect of the now device-aware fakes, the two pre-existing tests
in the same file) fail with the exact real error message, restored the
fix, confirmed all pass. `pytest -q`: 194 passed (193+1 new)/9 skipped.

Sixth distinct bug in this notebook pair in ~24 hours. Still never run
past the grid's per-round training step on a live GPU runtime.
