# STATUS — honest ledger of what's tested, what's written-but-unverified, and what's stubbed

This file exists because the authoring sandbox for this repo had **no
network access** (pip/npm blocked, no pypi, no HuggingFace, no GitHub
except via the push path) and **no GPU/torch**. Only `numpy`,
`cryptography`, and the standard library were available. That
constraint is the reason this repo has a sharp split between code that
is genuinely verified right now and code that is written carefully,
cross-checked by hand against the modules it calls, and syntax-checked,
but has never actually executed. Treat the two categories differently
when deciding how much to trust a result from this repo.

**Read this before running anything on Colab, and update it after
`notebooks/00_Setup_and_Crypto_Tests.ipynb`'s run** — that notebook is
Gate 1, and its actual pass/fail outcome belongs here, not just in the
notebook's own output cells (which can be lost if the Colab session is
reset).

## Legend
- ✅ **TESTED** — unit tests exist, were actually run in this sandbox, and passed. Command given.
- 🟡 **WRITTEN, NOT RUN** — syntax-checked (`python3 -m py_compile`) and manually cross-referenced against every function it calls (names, signatures, return types all verified to match), but never executed, because it needs `pynacl` and/or `torch`, neither installed here.
- 🔴 **STUB / SPECIFICATION ONLY** — exists to document an intended design or upgrade path; not meant to run as-is.

## crypto/ — cryptographic primitives

| Module | Status | Notes |
|---|---|---|
| `field.py` | ✅ TESTED | `python3 tests/test_field.py` |
| `shamir.py` | ✅ TESTED | `python3 tests/test_shamir.py` — includes the critical `test_reshare_preserves_the_secret_pk_trace_invariant` |
| `sketch.py` | ✅ TESTED (correctness only, NOT performance) | `python3 tests/test_sketch.py`. **See its own PERFORMANCE NOTE**: this CPU/NumPy reference extrapolates to ~16 min/client/round at ResNet-50's real d=23.5M. MUST be reimplemented in torch/GPU before the real CASE1/CASE2 runs — do not port this class verbatim into the training loop. |
| `dp.py` | ✅ TESTED | `python3 tests/test_dp.py` |
| `secagg.py` | ✅ TESTED (masking algebra + X25519 ECDH helper) | `python3 tests/test_secagg.py` for the masking algebra, including `test_false_dropout_does_not_unmask` (the R5 guardrail). `derive_pairwise_secret_via_x25519` passed on Colab Gate 1 (2026-11-10/11): both sides derive the same pairwise secret. |
| `ristretto.py` | ✅ TESTED on Colab Gate 1 (2026-11-10/11) | Basic group-law sanity passed. **One real bug found and fixed here**: `scalar_mul_base`/`Point.scalar_mul` didn't special-case a zero scalar, and libsodium's underlying C functions reject a zero scalar outright (rc=-1) instead of returning the identity — this broke `rangeproof.py`'s `prove_range` on any value with a zero bit (i.e. almost every value). Fixed by short-circuiting `0*P = identity` directly before calling libsodium (commit `ebdf2a9`). |
| `pedersen.py` | ✅ TESTED on Colab Gate 1 | Homomorphism check (1000 random pairs) passed, after the `ristretto.py` zero-scalar fix above (pedersen commitments to 0 go through this same code path). |
| `schnorr_nizk.py` | ✅ TESTED on Colab Gate 1 | AND/OR composition toolkit, including the added `or_prove_branch1_true`, exercised transitively by `rangeproof.py`'s and `r_cons.py`'s passing tests. |
| `elgamal_threshold.py` | ✅ TESTED on Colab Gate 1 | 7-of-10 recovery / 6-of-10 failure both passed, after the `dkg.py` fix below. |
| `dkg.py` | ✅ TESTED on Colab Gate 1 | **One real bug found and fixed here**: `finalize()` returned `dict[int, int]` (plain share values), but `elgamal_threshold.py`'s `compute_partial()` needs a full `ShamirShare` (both `.value` and `.index` — the latter flows into `PartialDecryption.holder_id`, used by `combine_partials`' Lagrange step). `reshare.py` already returned proper `ShamirShare`s for the same consumer; `dkg.py` was the inconsistent one. Fixed to wrap final shares as `ShamirShare(index=j, value=...)` (commit `cc9a59b`). |
| `committee.py` | ✅ TESTED on Colab Gate 1 | **The single most important guardrail** (per its own docstring) — the consent-rule test (admin alone cannot install a roster; t-of-n_c consent can) passed. |
| `reshare.py` | ✅ TESTED on Colab Gate 1 | PK_trace invariant across resharing confirmed (EC wrapper on top of `shamir.py`'s already-tested field math), after the `dkg.py` fix above (old/new shares are now the same `ShamirShare` type everywhere). |
| `vrf.py` | ✅ TESTED on Colab Gate 1 | Empirical sampling-rate check (`±0.02` of `p=0.25` over 10^4 trials) passed. |
| `clsag.py` | ✅ TESTED on Colab Gate 1 | All-signers + tamper + key-image consistency tests passed. |
| `mixnet.py` | ✅ TESTED on Colab Gate 1 | **One real bug found and fixed here**: `build_onion` padded the client payload to the full `CELL_SIZE` before wrapping it through 3 more layers of address+encryption overhead, so the very first wrap always overflowed the cell budget. Fixed by reserving `n_hops * WRAP_OVERHEAD` bytes up front and using fixed-width hop-address fields (commit `7f111af`). 3-hop round-trip now passes. |
| `rangeproof.py` | ✅ TESTED on Colab Gate 1 | In-range accept / out-of-range reject passed, after the `ristretto.py` zero-scalar fix above (every zero bit in the decomposition hit that bug directly). |
| `r_cons.py` | ✅ TESTED on Colab Gate 1 | Identity-binding proof verifies. **The sampling-disjunction OR statement is still left generic** (see the module's own docstring) — the exact noise-cover-commitment encoding is an `fl/`-layer decision not yet finalized; this is a scope gap, not a bug. |

## circuits/ — camera-ready PLONK upgrade path

| Item | Status |
|---|---|
| `consistency.circom`, `share_wellformed.circom`, `build.sh` | 🔴 STUB / SPECIFICATION ONLY. Not compiled, not run. See `circuits/README.md`'s upgrade checklist before trusting these as more than a design document. The curve-arithmetic gadgets inside them (`EdwardsScalarMul`, `EdwardsAdd`) are explicitly unimplemented placeholders. |

## fl/ — the FL pipeline

| Module | Status | Notes |
|---|---|---|
| `splits.py` | ✅ TESTED | `python3 tests/test_splits.py` — disjointness and coverage for both MNIST (stratified-IID) and FEMNIST (writer-disjoint) splits, with synthetic data standing in for real images/writers. |
| `norm_rule.py` | ✅ TESTED | `python3 tests/test_norm_rule.py` — includes robustness to a 48%-coalition attempting to drag the median. |
| `evidence.py` | ✅ TESTED | `python3 tests/test_evidence.py` — **this is the test that actually validates ARCHITECTURE.md's "bounded false accusation" Theorem**: calibrating theta on honest root-set data keeps the empirical false-accusation rate on FRESH simulated honest epochs within ~0.02 of the target alpha=0.05, while a persistent scaling attacker is reliably flagged within one epoch. |
| `models.py` | 🟡 WRITTEN, NOT RUN | Needs torch/torchvision. GroupNorm-replacement logic and the 1-channel-stem surgery were checked by reading torchvision's `resnet50`/`mobilenet_v2` source structure, not by running a forward pass. **First real test should be a forward pass on one batch before anything else in CASE1/CASE2.** |
| `data_iid.py` | 🟡 WRITTEN, NOT RUN | Needs torch/torchvision; CSV path needs an actual MNIST CSV uploaded to Drive. |
| `data_noniid.py` | 🟡 WRITTEN, NOT RUN | Needs torch + `datasets` + `flwr_datasets`; first real download of `flwrlabs/femnist` happens on Colab. |
| `client.py` | 🟡 WRITTEN, NOT RUN (torch-dependent steps) / ✅ step 5's digest scheme TESTED | Steps 0, 2, 3, 4, 6, 7 call already-tested-elsewhere logic. **Step 5's old placeholder (a plain coordinate-sum digest) has been replaced** with `crypto/vector_commit.py`'s `weighted_digest` — a randomized-linear-fingerprint digest, sound under a Schwartz-Zippel argument (the old sum-based digest was NOT sound: an attacker could move value between coordinates and the digest wouldn't change — see `tests/test_vector_commit.py::test_digest_catches_a_coordinate_sum_preserving_tamper`, which demonstrates exactly that attack being caught now). Pure-math, no EC dependency: `python3 tests/test_vector_commit.py` — 6/6 passed, including linearity across a simulated multi-client aggregate. **Second real bug found and fixed here, caught live on Colab after the server.py V3 commitment fix still left every client rejected**: `round_step_phase2` sketched `_secret_masked_vector` (`q_k + m_k mod P_MASK`) for the norm-bound proof. Two compounding problems made that unusable: (1) the mask `m_k` is drawn to look uniform over Z_{P_MASK}, so its norm is dominated by mask noise regardless of the real update's size, and (2) even `_secret_q` alone is `np.mod`-wrapped into Z_{P_MASK} by `quantize()`, so any negative-delta coordinate wraps to a value near P_MASK — a standalone dry-run confirmed this wrapped representation gives a norm of `~3.3e18` for a five-element toy delta, vs. `~2e5` for the correctly-centered signed version (matching the true scaled delta norm within quantization rounding). Fixed by adding `center_mod_p_mask` (the un-wrap `dequantize` already needed, now shared) and sketching `center_mod_p_mask(phase1._secret_q)` instead. Confirmed isolated to V3/B_t-EMA: V5's aggregation and R_cons both use `_secret_masked_vector`/`commitment_point` directly and never touch the sketch. |
| `server.py` | 🟡 WRITTEN, NOT RUN (torch-dependent steps) / ✅ V5 digest computation TESTED | V1-V4 checks wire directly to tested-elsewhere crypto calls. `run_round` now computes `published_sum_value` via the same `weighted_digest(aggregate_masked_vector, seed=rho_t)` client.py uses, and `published_sum_blinding` from the accepted set's revealed `r_k`'s — both exposed on `RoundOutcome` for `verifier/audit.py`'s independent V5 check to consume. **One real bug found and fixed here, caught live on Colab**: `check_V3_norm_bound` checked the range proof's recomposition against `submission.commitment_point` (`c_k`, a Pedersen commitment to `weighted_digest(masked, rho_t)` with blinding `r_k`) — a completely different value AND blinding than `prove_range`'s own `(norm_estimate, agg_blinding)`. This made V3 fail for every client, every round, regardless of model (observed: "0 accepted, k_min=8" on both rounds of the tiny-debug-model smoke test). Fixed by having the server independently recompute `norm_estimate = ‖phase2.sketch‖` (sent in cleartext in this reference implementation) and re-derive the commitment via `phase2.range_proof_aggregate_blinding`, plus an explicit reject when the real norm is at/above `2**n_bits` (closing a clamp-to-cap loophole that would otherwise silently accept an out-of-range norm). Verified via a standalone non-EC dry-run (`/tmp/dryrun_v3fix.py`'s approach — integers mod a prime standing in for the ristretto255 group) confirming the old check spuriously rejects honest clients, the new check accepts them, and still catches a client lying about its norm or exceeding the provable range. **Known, separate, pre-existing approximation**: `n_bits` is sized client-side as `ceil(log2(B_t*4))`, so this still only enforces roughly `norm < 4×B_t`, not a tight `norm < B_t` as ARCHITECTURE.md §5.2 specifies — a real Bulletproofs-style tight bound was out of scope for this fix. |
| `trace.py` | 🟡 WRITTEN, NOT RUN | Pure orchestration over `committee.py`/`elgamal_threshold.py`/`r_cons.py`; no new crypto. |

## attacks/ and verifier/

All 🟡 WRITTEN, NOT RUN — each is a thin, direct exercise of already-listed EC-dependent crypto calls (no new primitives), written to assert the SPECIFIC defense ARCHITECTURE.md's Table in §10 claims for that attack. Run these in a dedicated notebook cell after Gate 1 passes, before trusting the attack-rejection numbers in any figure.

## Gate 1 — PASSED (2026-11-11, Colab)

`00_Setup_and_Crypto_Tests.ipynb` ran end to end with no errors. Three real bugs were found and fixed live against the actual Colab run (not environment/install issues — genuine code defects), all pushed to `main`:
- `ebdf2a9` — `ristretto.py`: zero-scalar case in `scalar_mul`/`scalar_mul_base` (broke `pedersen.py` commitments-to-0 and `rangeproof.py`'s bit=0 branches).
- `cc9a59b` — `dkg.py`: `finalize()` returning plain ints instead of `ShamirShare` objects (broke `elgamal_threshold.py`/`reshare.py`'s shared consumer `compute_partial`).
- `7f111af` — `mixnet.py`: `build_onion` not reserving wrapping overhead before padding the payload (broke the 3-hop round trip).

Every EC-dependent crypto/ module above is now ✅ TESTED, not 🟡. The only remaining 🟡/🔴 items are the torch-dependent `fl/` modules (untested until CASE1/CASE2 actually run) and the `circuits/` specification stubs.

## notebooks/ and configs/

- `00_Setup_and_Crypto_Tests.ipynb` — the Gate 1 notebook. **Passed, 2026-11-11.** Generated via `notebooks/_build_notebooks.py` (kept so the cell content is reviewable as plain Python rather than raw notebook JSON).
- `CASE1_IID_MNIST.ipynb` / `CASE2_NONIID_FEMNIST.ipynb` — **round-loop wiring is now complete**, via a new shared `fl/round_loop.py::run_training_loop` (extracted once, not duplicated between the two notebooks, so they can't drift apart): real per-client X25519 pairwise secrets, real local data, the sketch-seed commit/reveal ordering, phase1/phase2 calls, V1-V5, an INDEPENDENT V5 re-check (`verifier/audit.py`), `apply_update`, and `B_t`'s round-to-round EMA update. Both notebooks run a small SMOKE TEST (2 rounds, 5 clients) first, not the full MVP-scale sweep — see the performance note below. One bug caught and fixed while wiring this (not yet Colab-exercised, but a real static-review catch): `fl/client.py`'s `local_train_delta` reloads the model with each client's own locally-trained weights every call, so without an explicit fix the shared `model` object would end a round holding whichever client trained *last*, not the real federated aggregate — `run_training_loop` now re-syncs `model` to the actual returned `final_w_t` before returning, so the notebooks' evaluation cells (which evaluate `models[name]` directly) score the real result. **Not yet Colab-run — needs the same "first real execution" caveat as every other 🟡 EC/torch-dependent module.**
  - **Sketch performance wall, unchanged from before**: `crypto/sketch.py`'s `SketchProjector` is correctness-tested but ~16 min/client/round extrapolated at ResNet-50's real d~23.5M — do not launch the full 45-round × 50-client × 2-model sweep before STATUS.md TODO #3's GPU/torch reimplementation lands. The smoke test avoids this by using a handful of rounds/clients.
  - **Still-placeholder `server_root_norms`** for round-0's `B_t` bootstrap (every later round's `B_t` is real, EMA-updated from real submissions).
  - **`fl/evidence.py` (false-accusation calibration) is NOT wired into the round loop** — needs per-client loss deltas `local_train_delta` doesn't currently return, plus a per-round median-reference vector. Tracked as a separate follow-up, not assumed done.
- `configs/mvp_3day.yaml` — single source of truth for every MVP parameter; both notebooks and `fl/*` should read from this rather than hardcoding numbers.

## Top TODO list, in priority order, for the Colab session

1. ~~Run `00_Setup_and_Crypto_Tests.ipynb` end to end first.~~ **DONE — Gate 1 passed 2026-11-11.**
2. ~~Finish `fl/client.py` step 5's vector commitment scheme.~~ **DONE** — replaced the unsound coordinate-sum digest with `crypto/vector_commit.py`'s randomized-weight digest (pure math, tested in-sandbox: `python3 tests/test_vector_commit.py`); `fl/server.py`'s `run_round` and `verifier/audit.py`'s `independent_v5_check` updated to match. Still needs a real end-to-end Colab exercise once CASE1/CASE2's round loop (#4 below) actually calls it against a real aggregate — the pure-math linearity/soundness properties are verified, but the full commit→escrow→sign→verify wiring through real EC calls is not yet.
3. **Reimplement `sketch.py`'s `SketchProjector.project()` on GPU/torch** before running real CASE1/CASE2 rounds — the NumPy reference is correctness-only and far too slow at real model scale (see its own PERFORMANCE NOTE).
4. ~~Finish the CASE1/CASE2 round-loop wiring.~~ **DONE (not yet Colab-exercised)** — see the `notebooks/` section above for exactly what `fl/round_loop.py` now covers and what's still genuinely open (sketch performance wall, `server_root_norms` placeholder, evidence/calibration not wired).
5. Once 2-4 pass on a small scale (few rounds, few clients), scale up to the full MVP parameters in `configs/mvp_3day.yaml` and run the 5-attack grid in `attacks/`.
6. Update this file's status table as each remaining 🟡 item actually gets exercised and passes (or fails and gets fixed) on Colab.

## What is NOT cut, scope-wise (for context when reading the above)

Per `ROADMAP.md` §0.1: all 13 crypto primitives, both circuit specifications, the full `R_cons` relation, the committee consent rule, double-masking, the mix cascade, and the V5 aggregation check are all present in this repo (as 🟡 or ✅ above) — none of the crypto/security surface was cut for the 3-day MVP, only the ML-scale parameters (2 backbones instead of 3, ~45 rounds instead of 120-150, 1 seed instead of 3, 5 attacks instead of 7).
