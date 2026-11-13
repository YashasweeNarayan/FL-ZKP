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
