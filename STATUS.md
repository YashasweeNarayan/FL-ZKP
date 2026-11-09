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
| `secagg.py` | ✅ TESTED (masking algebra) / 🟡 (the one X25519 ECDH helper function at the bottom) | `python3 tests/test_secagg.py` — includes `test_false_dropout_does_not_unmask`, the actual guardrail test for the R5 finding. The `derive_pairwise_secret_via_x25519` function needs `pynacl`, untested here. |
| `ristretto.py` | 🟡 WRITTEN, NOT RUN | Needs `pynacl`. First real test: `00_Setup_and_Crypto_Tests.ipynb`'s basic sanity cell. |
| `pedersen.py` | 🟡 WRITTEN, NOT RUN | Homomorphism claim untested until Colab (`00_Setup...ipynb` has a 1000-random-pair check). |
| `schnorr_nizk.py` | 🟡 WRITTEN, NOT RUN | The AND/OR composition toolkit. Closing-equation algebra for AND was traced by hand; OR composition (both `or_prove_branch0_true` and the added `or_prove_branch1_true`) traced by hand but not executed. |
| `elgamal_threshold.py` | 🟡 WRITTEN, NOT RUN | 7-of-10 recovery / 6-of-10 failure is in `00_Setup...ipynb`. |
| `dkg.py` | 🟡 WRITTEN, NOT RUN | Pedersen DKG, Feldman verification. |
| `committee.py` | 🟡 WRITTEN, NOT RUN | **The single most important guardrail** (per its own docstring) — the consent-rule test (`admin alone cannot install a roster`) is in `00_Setup...ipynb` and should be the FIRST thing you check if anything here fails. |
| `reshare.py` | 🟡 WRITTEN, NOT RUN (EC wrapper) | The underlying field-math `reshare()` in `shamir.py` IS fully tested; this file only adds the Feldman-commitment publishing on top, untested. |
| `vrf.py` | 🟡 WRITTEN, NOT RUN | Empirical sampling-rate check (`±0.02` of `p=0.25` over 10^4 trials) is in `00_Setup...ipynb`. |
| `clsag.py` | 🟡 WRITTEN, NOT RUN | **Rewritten once already** after catching a broken `verify()`/dead-code first draft during authoring — see the file's own history in conversation. Closing-equation algebra traced by hand for the signer index. All-signers + tamper + key-image tests are in `00_Setup...ipynb`. |
| `mixnet.py` | 🟡 WRITTEN, NOT RUN | 3-hop round-trip test is in `00_Setup...ipynb`. |
| `rangeproof.py` | 🟡 WRITTEN, NOT RUN | Bit-decomposition OR-composition, built on `schnorr_nizk.py`. In-range accept / out-of-range reject tests are in `00_Setup...ipynb`. |
| `r_cons.py` | 🟡 WRITTEN, NOT RUN | The compound AND-binding (pk, round tag, epoch tag, VRF point) is the core identity-laundering fix. **The sampling-disjunction OR statement is left generic** (see the module's own docstring) — the exact noise-cover-commitment encoding is an `fl/`-layer decision not yet finalized. |

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
| `client.py` | 🟡 WRITTEN, NOT RUN — **and partially a placeholder, see below** | Steps 0, 2, 3, 4, 6, 7 call already-tested-elsewhere logic correctly as far as static review can confirm. **Step 5 (Pedersen-committing the masked update vector) uses an explicitly-flagged placeholder**: it commits a scalar digest (`sum(masked) mod L`) instead of a real per-coordinate or chunked vector commitment, because the real vector-commitment scheme for a d=23.5M-dimensional masked update is a design decision that belongs with `fl/server.py`'s V5 check, and committing to one here before `server.py`'s V5 is exercised against real data risked a silent mismatch. **This is the single biggest "not actually finished" item in the repo — see the TODO list below.** |
| `server.py` | 🟡 WRITTEN, NOT RUN | V1-V4 checks wire directly to tested-elsewhere crypto calls. V5 (`commit_sum`/`verify_opening`) inherits `client.py`'s digest-placeholder limitation above. |
| `trace.py` | 🟡 WRITTEN, NOT RUN | Pure orchestration over `committee.py`/`elgamal_threshold.py`/`r_cons.py`; no new crypto. |

## attacks/ and verifier/

All 🟡 WRITTEN, NOT RUN — each is a thin, direct exercise of already-listed EC-dependent crypto calls (no new primitives), written to assert the SPECIFIC defense ARCHITECTURE.md's Table in §10 claims for that attack. Run these in a dedicated notebook cell after Gate 1 passes, before trusting the attack-rejection numbers in any figure.

## notebooks/ and configs/

- `00_Setup_and_Crypto_Tests.ipynb` — the Gate 1 notebook. **Run this first.** Generated via `notebooks/_build_notebooks.py` (kept so the cell content is reviewable as plain Python rather than raw notebook JSON).
- `CASE1_IID_MNIST.ipynb` — **contains a known, explicitly-flagged gap**: the round loop is a scaffold that sets up keys/models/committee correctly but stops before wiring in real per-client pairwise ECDH secrets and the full phase1/phase2 client call sequence. The notebook's own markdown cell spells out the exact 6 steps needed to finish it. Do this on Colab, interactively, rather than guessing at it blind.
- `CASE2_NONIID_FEMNIST.ipynb` — same round-loop gap as CASE1 (intentionally not duplicated — see its markdown cell), plus a working seen-writer vs. unseen-writer evaluation cell once models are trained.
- `configs/mvp_3day.yaml` — single source of truth for every MVP parameter; both notebooks and `fl/*` should read from this rather than hardcoding numbers.

## Top TODO list, in priority order, for the Colab session

1. **Run `00_Setup_and_Crypto_Tests.ipynb` end to end first.** If anything fails, fix it before touching CASE1/CASE2 — every ML number downstream depends on these primitives being correct.
2. **Finish `fl/client.py` step 5's vector commitment scheme** (currently a scalar-digest placeholder) and update `fl/server.py`'s V5 check and `verifier/audit.py` to match. This is the one piece of real cryptographic design work still open, not just wiring.
3. **Reimplement `sketch.py`'s `SketchProjector.project()` on GPU/torch** before running real CASE1/CASE2 rounds — the NumPy reference is correctness-only and far too slow at real model scale (see its own PERFORMANCE NOTE).
4. **Finish the CASE1/CASE2 round-loop wiring** per each notebook's "Known gap" markdown cell (pairwise ECDH secrets, phase1/phase2 calls, V1-V5, apply_update).
5. Once 1-4 pass on a small scale (few rounds, few clients), scale up to the full MVP parameters in `configs/mvp_3day.yaml` and run the 5-attack grid in `attacks/`.
6. Update this file's status table as each 🟡 item actually gets exercised and passes (or fails and gets fixed) on Colab.

## What is NOT cut, scope-wise (for context when reading the above)

Per `ROADMAP.md` §0.1: all 13 crypto primitives, both circuit specifications, the full `R_cons` relation, the committee consent rule, double-masking, the mix cascade, and the V5 aggregation check are all present in this repo (as 🟡 or ✅ above) — none of the crypto/security surface was cut for the 3-day MVP, only the ML-scale parameters (2 backbones instead of 3, ~45 rounds instead of 120-150, 1 seed instead of 3, 5 attacks instead of 7).
