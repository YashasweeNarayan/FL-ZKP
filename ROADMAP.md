# JANUS-FL — Build Roadmap

Companion to `ARCHITECTURE.md` (rev. 3) and `REVIEW.md`. Ten stages, each with a **gate**: a checkable condition that must hold before the next stage starts, so a failure is caught in the stage that caused it rather than three stages later inside a long training run.

`n = 50` clients · `n_c = 10` key holders · `t = 7` · participation `p = 0.25` · epoch `R = 25`.

---

## 0.1 — 3-day MVP scope (current build target)

The full matrix below (~132 GPU-hr, §9) is the camera-ready plan — 3.5 months, full seed counts, three backbones. **This repo's first build targets a 3-day MVP** that proves every mechanism in `ARCHITECTURE.md` works end-to-end, with a GPU budget that actually fits 3 days on one Colab instance.

**What's cut, and why it doesn't weaken the crypto/security claims:**

| Full plan | MVP | Effect |
|---|---|---|
| 3 backbones | **2 backbones: ResNet-50 + MobileNetV2** | spans the same `d` range (23.5M vs 2.3M) that proves the sketched-proof claim (N4) is `d`-independent; EfficientNet-B0 sits between them and adds no new evidence |
| 120–150 rounds | **~45 rounds** | enough to show clear convergence trend, not full plateau. Every crypto check (V1–V5) and every attack-rejection result is valid at round 5 just as at round 120 — those don't depend on convergence |
| 3 seeds | **1 seed** | no mean±std in the MVP tables; add seeds in the camera-ready pass |
| 4 malicious fractions × 7 attacks | **2 fractions × the money-shot attacks (A1, A3, A4, A5, A7)** | A5 (identity-laundering, with/without `R_cons`) and A1 (scaling) are the figures that matter; A2/A6 (poisoning, inversion) need many rounds to show signal and are deferred |

**Revised compute budget** (2 models × 2 datasets × 2 arms [baseline, JANUS-FL] × 45 rounds × 1 seed, plus the 5-attack reduced grid, plus ablations):

| Block | GPU-hr |
|---|---|
| Baseline + crypto-in-loop, 4 configs × 1 seed | ~1.8 |
| Attacks (5 × 2 fractions × 2 models, short runs) | ~4 |
| Ablations (defense-off checks, 3 key ones) | ~2 |
| Reruns / debugging buffer (+50%, generous for an MVP) | ~4 |
| **Total** | **≈ 12 GPU-hr** |

12 GPU-hr fits comfortably in 3 days on a single Colab Pro instance (even 1 day, in principle) — the real constraint is debugging time and session management, not raw compute. The camera-ready 132-hr matrix (3 backbones, 3 seeds, full attack grid, `R`/`m` sweeps) is Stage 8 proper, run later once the MVP has validated every mechanism.

**What does NOT get cut, because it's where the actual novelty and risk live:** all 13 crypto primitives, both circuits, the full `R_cons` relation (VRF + ring + escrow + consistency), the committee consent rule, double-masking, the mix cascade, and the V5 aggregation check. The MVP proves the *framework*; the full matrix later proves the *paper's numbers*.

---

---

## Stage 0 — Decisions + skeleton — ✅ **CLEARED**

All twelve decisions locked in `ARCHITECTURE.md` §12, including the five from `REVIEW.md` §7: PLONK over Groth16, GroupNorm over BatchNorm, double-masking in scope, public `B_t` rule, and the anonymous channel constructed rather than assumed.

---

## Stage 1 — Crypto core, no ML  (5–6 days)

Build and unit-test every primitive in isolation. **No PyTorch in this stage.** If the crypto is wrong, every ML number afterwards is wasted compute.

| Module | Test |
|---|---|
| `ristretto.py` | RFC 9496 test vectors |
| `pedersen.py` | homomorphism `Com(a;r)·Com(b;s) = Com(a+b;r+s)` over 10⁴ random pairs |
| `clsag.py` | verify for all 50 signers; fails on 1-bit tamper of message, ring or tag; **same round ⇒ equal `I_k^t`; different rounds ⇒ unequal; same epoch ⇒ equal `P_k^e`; different epochs ⇒ unequal** |
| `dkg.py` | 10-member Pedersen DKG; every Feldman check passes; any 7 shares reconstruct a *test* secret (then discard — never on `sk_trace`) |
| `elgamal_threshold.py` | 7-of-10 partial decryption recovers `pk`; 6-of-10 fails; **a corrupted partial is rejected by its Chaum–Pedersen proof** |
| `committee.py` | roster epochs advance only with 7 outgoing signatures **plus** admin signature; **a server-only roster change is rejected** — write this test first, it is the guardrail the whole anonymity claim rests on |
| `reshare.py` | resharing changes every share while `PK_trace` is bit-identical; a retired member's old share no longer combines |
| `secagg.py` | 50 masks sum to zero; dropout of 8 recovered; **double-masking: a falsely-declared dropout does NOT unmask an honest client** |
| `mixnet.py` | 3-hop onion round-trips 50 payloads unmodified; `M1` output is unlinkable to its input without `M3`'s key; a dropped message is detected by its owner; all payloads byte-identical in length |
| `vrf.py` | ECVRF test vectors; empirical participation rate within `±0.02` of `p = 0.25` over 10⁴ trials; output uniqueness per `(sk, ρ_t)` |
| `dp.py` | noise shares sum to `N(0, σ²I)`; RDP accountant matches a reference implementation on known `(ε, δ)` points |
| `sketch.py` | JL distortion within `(1±ε)` for `ε=0.05, m=8192` on real ResNet-50 gradient vectors |
| `bulletproofs/` | range proof accepts in-range, rejects out-of-range; **measure throughput (constraints/s) — this number sets the whole compute budget** |

**Gate 1:** all tests green, and measured Bulletproofs throughput confirms §9's feasibility claim. If throughput is >5× worse than assumed, revisit `m` before going further.

> ⚠️ I cannot run these here — package installs are blocked in my environment, so the crypto stack must be validated on your Colab. Stage 1 is the first place you run a notebook and send me output. Expect one or two iterations on the Rust/maturin build cell; it is the schedule's critical path.

---

## Stage 2 — Circuits  (3 days)

`R_cons` (§5.4), `R_share` (§5.3) in Circom 2 with Poseidon, compiled to **PLONK**.

- Setup uses the **public perpetual Powers-of-Tau** universal SRS. No per-circuit ceremony, no toxic waste. Commit the SRS hash and `verification_key.json` so verification is reproducible.
- Constraint budget: `R_cons` ≤ 2×10⁵ (it now carries the VRF check and the sampling disjunction), `R_share` ≤ 3×10⁴. If `R_cons` overruns, move ring membership to a Merkle path over `root(R)` instead of a 50-way comparison — at `n = 50` this is likely necessary.
- **Negative tests first, before the circuit is written:**
  - a witness where `sk` in the round tag ≠ `sk` encrypted in `τ` must be **unsatisfiable** (this is A5)
  - a witness with `b = 0` but `q ≠ ν` must be **unsatisfiable** (this is A4's dummy-impersonation route)
  - a witness with a VRF output above threshold but `b = 1` must be **unsatisfiable**

**Gate 2:** all three negative tests unsatisfiable; proof ≤ 1 KB; verification ≤ 20 ms.

---

## Stage 3 — Data pipelines and splits  (3 days)

### 3.1 CASE-1 — MNIST, IID, from your Google Drive

Download once from Kaggle (`datasets/hojjatk/mnist-dataset`) or via torchvision, then put it in Drive so Colab never re-downloads:

```
MyDrive/FL-ZKP/data/MNIST/raw/
    train-images-idx3-ubyte
    train-labels-idx1-ubyte
    t10k-images-idx3-ubyte
    t10k-labels-idx1-ubyte
```

```python
from google.colab import drive
drive.mount('/content/drive')

DATA_ROOT = '/content/drive/MyDrive/FL-ZKP/data'   # MNIST/raw/... lives under here
import torchvision
ds = torchvision.datasets.MNIST(DATA_ROOT, train=True, download=False, transform=tf)
```

The `raw/` subfolder matters — `download=False` fails without it. If you have the Kaggle CSVs rather than the idx files, say so and I'll write a CSV loader.

### 3.2 CASE-2 — FEMNIST, natural non-IID, from Hugging Face

`flwrlabs/femnist` — 814,277 samples, 62 classes, 28×28 grayscale, with a `writer_id` field. Partitioning by writer gives **genuine** non-IID (writer-level label skew *and* handwriting-style skew) rather than a synthetic Dirichlet split. Claim that as a strength: natural heterogeneity, not simulated.

```python
!pip install -q "flwr-datasets[vision]"

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import NaturalIdPartitioner

fds = FederatedDataset(
    dataset="flwrlabs/femnist",
    partitioners={"train": NaturalIdPartitioner(partition_by="writer_id")},
)
part_0 = fds.load_partition(partition_id=0)      # one writer
```

Cache to `MyDrive/FL-ZKP/data/femnist_cache/` after the first load (~200 MB download). The whole dataset sits in the `train` split; §3.4 carves every role out of it by writer.

### 3.3 Model adapters

| Change | Why |
|---|---|
| Resize 28→32, pad-reflect | ResNet-50's stride-2 stem + 4 stages reduce 28×28 below 1×1 |
| First conv → 1 input channel | grayscale |
| Classifier head → 10 (MNIST) / 62 (FEMNIST) | task |
| **BatchNorm → GroupNorm** (`groups=32`) | BN is a known non-IID failure mode in FL **and** its running statistics leak the local data distribution — a privacy hole the rest of the framework closes. Must be decided here; retrofitting invalidates every accuracy number |
| **No pretrained weights — train from scratch** | pretrained ResNet-50 reaches ~99% on MNIST in two rounds: flat curves, nothing to poison, and a reviewer calls the evaluation vacuous |

Dimensions (hand-computed; **re-verify with `sum(p.numel())` in this stage's notebook and correct the paper's table from that output** — I could not run torch here):

| Model | `d`, 10-class | `d`, 62-class |
|---|---|---|
| ResNet-50 | ≈ 23.5 M | ≈ 23.6 M |
| EfficientNet-B0 | ≈ 4.02 M | ≈ 4.09 M |
| MobileNetV2 | ≈ 2.24 M | ≈ 2.30 M |

GroupNorm changes these slightly (no running buffers, same weight/bias count) — take the notebook's numbers as authoritative.

### 3.4 ★ Splits — the full specification

Every role is disjoint, every index file is seeded and committed to `configs/splits/`. The governing rules:

- **No client sample ever appears in any test set.**
- **The server root set is carved from train, never from test**, so the test sets stay pristine.
- **For FEMNIST, roles are disjoint at the writer level, not the sample level** — otherwise the same hand appears in train and test and the generalization number is inflated.

**MNIST (IID)** — from the official 60,000 train / 10,000 test:

| Role | Size | Source | Use |
|---|---|---|---|
| Client train | 50 × 950 = 47,500 | official train, stratified | local training |
| Client local val | 50 × 100 = 5,000 | official train, stratified | local sanity only; never a global decision |
| Server root set | 5,000 | official train, stratified | `B_t` bootstrap at round 0, evidence calibration (§6.1), A2's loss-delta term |
| Global validation | 2,500 | official train, stratified | learning rate, rounds, any hyperparameter choice |
| **Global test** | **10,000** | **official test, untouched** | **every reported accuracy number** |

Stratified means each client sees all 10 classes in near-equal proportion; plot the per-client class histogram as a sanity figure.

**FEMNIST (natural non-IID)** — partitioned by writer, 4 writers per client so local datasets are usable (mean ≈ 226 samples/writer):

| Role | Writers | ≈ Samples | Use |
|---|---|---|---|
| Client train | 200 (4 per client × 50) | ≈ 36,000 | local training |
| Client local val | same writers, 20% holdout | ≈ 9,000 | **seen-writer test** (see below) |
| Server root set | 60 **disjoint** writers | ≈ 13,600 | calibration, `B_t`, loss-delta |
| Global validation | 40 **disjoint** writers | ≈ 9,000 | hyperparameters |
| **Unseen-writer test** | **300 disjoint writers** | **≈ 68,000** | **generalization to new users** |

**Report both FEMNIST metrics, always.** They measure different things and quoting only one is how FL papers overclaim:

- **Seen-writer accuracy** — held-out samples from the 200 training writers. Measures how well the global model serves the clients who contributed. Will be the higher number.
- **Unseen-writer accuracy** — the 300 disjoint writers. Measures whether the federation produced a model that generalizes to a new participant. This is the harder and more meaningful metric, and it is what the headline table should lead with.

Writers are filtered to ≥120 samples before assignment, shuffled with a fixed seed, then dealt to roles in the order above so no writer holds two roles. Remaining writers are unused and available if you later scale `n` past 50.

**Gate 3:** `splits.py` asserts pairwise-disjointness of every role (sample-level for MNIST, writer-level for FEMNIST) and the assertions run in CI; per-client class histograms plotted for both cases; all 3 models forward a batch and reach >97% centralized MNIST accuracy when trained centrally from scratch.

---

## Stage 4 — Plaintext FL baseline  (3 days)

FedAvg, 50 clients, VRF self-sampling at `p = 0.25`, no crypto. This is the accuracy ceiling every later number is measured against.

Settings: 120 rounds (MNIST) / 150 (FEMNIST), 1 local epoch, batch 32, SGD momentum 0.9, lr 0.05 cosine-annealed, AMP on, GroupNorm, from scratch.

**Gate 4:** MNIST IID ≥ 98% on the global test set for all three models. FEMNIST: seen-writer ≥ 78%, unseen-writer ≥ 70% (62-class, writer-partitioned, from scratch — do not expect MNIST-like numbers, and do not "fix" it by collapsing classes or leaking writers across roles).

---

## Stage 5 — Integrate crypto into the loop  (4–5 days)

Wire Stages 1–2 into Stage 4. All 50 clients honest.

**Gate 5 (the critical correctness gate):**
- masked-aggregation accuracy matches the plaintext baseline to within **±0.3%** at every round, with DP off
- **check V5 passes every round**, recomputed by `verifier/audit.py` independently from the bulletin board
- round-abort rate from `k_min = 8` lands near the predicted 4.5%

If accuracy drifts, the quantization or mask arithmetic is wrong. Check the `Z_p` modulus and the fixed-point overflow bound first — 50 clients summing 16-bit-mantissa values is the regime where silent wraparound appears, and the Stage 1 unit test should already cover it.

---

## Stage 6 — Overhead characterization  (3 days, `CRYPTO_BENCH.ipynb`)

No ML. **Measure, don't estimate** — and label clearly in the paper which numbers are measured and which extrapolated. Reviewers forgive flagged extrapolation; they do not forgive extrapolation presented as measurement.

1. Prover / verifier time and proof size per component (CLSAG, Bulletproofs, PLONK, escrow, VRF), per `d ∈ {2.24M, 4.02M, 23.5M}`.
2. Sketch ablation: `m ∈ {1024, 4096, 8192, 16384}` × (prover time, empirical soundness, JL distortion).
3. **Mixnet overhead**: added latency per hop, bandwidth with dummy traffic on vs. off, batch size 50.
4. Per-round wall clock and bytes-on-wire vs. plaintext FedAvg → the headline **overhead factor**.
5. DKG cost (once), resharing cost (per epoch), opening cost (per tracing event).
6. DP: `(ε, δ)` curve over 120 rounds with subsampling amplification at `p = 0.25`.

**Gate 6:** overhead factor measured and sane; `δ_JL` tabulated per `m`.

---

## Stage 7 — Attacks + ablations  (5–6 days)

Seven adversaries (`ARCHITECTURE.md` §10). Run order matters: A1, A3, A4, A5, A7 are cheap because they are rejected at the gate and need few rounds; A2 and A6 are expensive.

Malicious fraction `f/n ∈ {0.04, 0.12, 0.24, 0.40}` (2, 6, 12, 20 of 50).

Also sweep **epoch length `R ∈ {1, 5, 10, 25, 50}`** against detection rate and residual anonymity — figure F7, and nearly free since it reuses the A2 runs.

**A5 is the one to get exactly right** — it is the novelty. With `R_cons` on: rejected. With `R_cons` off: an innocent client is named by the committee. Capture the full opening transcript in both runs.

**Gate 7:** every cell of the defense × attack heatmap populated; A5-without-`R_cons` demonstrably frames an innocent client; A4's dummy-impersonation route demonstrably closed by the VRF disjunction.

---

## Stage 8 — Full matrix

```
CASE-1 (MNIST IID)        × 3 models × {baseline, JANUS-FL, JANUS-FL+DP} × 120 rounds
CASE-2 (FEMNIST non-IID)  × 3 models × {baseline, JANUS-FL, JANUS-FL+DP} × 150 rounds
Attacks   × 7 adversaries × 4 fractions × 2 models (MobileNetV2 + ResNet-50)
Ablations × 9 components  × 3 representative attacks
Sweeps    × R ∈ {1,5,10,25,50} , m ∈ {1024,4096,8192,16384}
Seeds     × 3 for headline tables, 1 for ablations and sweeps (state this)
```

**Gate 8:** every result in `results/*.csv` with a config hash, so every number in the paper traces to a config and a seed.

---

## 9. Compute budget

Measured per-run estimates assume a Colab T4/L4, 32×32 inputs, AMP, and a 4× FL-simulation overhead factor (sequential client loop, state-dict copies for a 23.5M-parameter model, small-batch GPU underutilization).

| Block | GPU-hr |
|---|---|
| Baselines + crypto-in-loop, 6 configs × 3 seeds | 46 |
| Attacks, 7 × 4 fractions × 2 models, 1 seed | 43 |
| Ablations, 9 components × 3 attacks | 12 |
| `R` and `m` sweeps | 5 |
| Reruns and smoke tests (+25%) | 26 |
| **Total** | **≈ 132 GPU-hr** |

**Why this is lower than the 160 GPU-hr of the `n = 20` plan**, despite 2.5× the clients and training from scratch: **VRF self-sampling.** Per-round GPU cost is driven by total samples trained, not client count, and at `p = 0.25` only ~13,000 of 52,500 samples are touched per round. Sampling cuts compute ~4× while *strengthening* anonymity — the rare case where the cheap option is also the more secure one.

**Cost controls, in the order I would apply them:**

1. Attacks on 3 malicious fractions instead of 4 → **−11 hr**
2. FEMNIST 120 rounds instead of 150, if the convergence plot justifies it → **−8 hr**
3. Attacks on MobileNetV2 only for the cheap adversaries (A1, A3, A4, A5, A7 are gate-rejections and need ~15 rounds, not 150) → **−15 hr**
4. 1 seed for the DP arm, 3 only for baseline and JANUS-FL → **−8 hr**

All four together bring it to ≈ **90 GPU-hr**, about 15 days at 6 h/day. I would apply 1 and 3 immediately and hold 2 and 4 in reserve.

**Non-negotiable:** checkpoint every round to Drive. Colab will disconnect mid-run, and a run you cannot resume costs double.

**CPU side, parallel to GPU:** 50 clients × one proof set per round. At 1–2.5 s per proof set that is 1.7–4.2 CPU-hr per run, which overlaps GPU time and is not additive — but it *is* what the overhead table reports, so measure it rather than hiding it.

---

## 10. Paper

### Section mapping

| § | Content | Source |
|---|---|---|
| I | Intro, the anonymity-vs-accountability tension, contributions N1–N11 | — |
| II | Related work + gap table | `ARCHITECTURE.md` §7 |
| III | System model, privilege separation, **anonymous channel construction**, threat model `A1`–`A5` | §1.1, §2 |
| IV | Preliminaries: Pedersen, Bulletproofs, CLSAG, threshold ElGamal, Pedersen DKG, PSS, VRF, mixnets | §3 |
| V | **JANUS-FL protocol**, the five relations, V1–V5 | §4, §5 |
| VI | **Identity-laundering attack and `R_cons`** ★ | §5.4 |
| VII | **Anonymous participation via VRF self-sampling** ★ | §5.5 |
| VIII | **Calibrated tracing + false-accusation bound**; committee rotation | §6.1, §6.2, §6.4 |
| IX | **Accountability–anonymity budget** ★ | §6.3 |
| X | Distributed DP and the `(ε, δ)` guarantee | §6.5 |
| XI | Security analysis T1–T12 | §8 |
| XII | Sketched norm proofs + complexity | §9 |
| XIII | Evaluation | Stages 4–8 |
| XIV | Limitations + future work | **`REVIEW.md` §6, verbatim** |

### Figures and tables

| # | What | From |
|---|---|---|
| F1 | System architecture, privilege separation | §4.1 |
| F2 | Round protocol sequence incl. mix cascade | §4 |
| F3 | Accuracy vs. round, 3 models × 2 cases × 3 arms; FEMNIST split into seen- and unseen-writer | Stage 5 |
| F4 | **Defense × attack heatmap** (9 × 7) | Stage 7 |
| F5 | **A5 with/without `R_cons`** — innocent client framed | Stage 7 |
| F6 | Sketch ablation: `m` vs. prover time vs. soundness | Stage 6 |
| F7 | **Residual anonymity `H(T)` vs. detection rate — Pareto frontier**, swept over `R` | §6.3 |
| F8 | Empirical FP rate vs. nominal `α`, IID and non-IID | §6.1 |
| F9 | Privacy–utility: accuracy vs. `ε` | §6.5 |
| T1 | Gap table | §7 |
| T2 | Per-component overhead vs. `d`, incl. mixnet | Stage 6 |
| T3 | Accuracy, mean ± std over 3 seeds, both FEMNIST metrics | Stage 8 |

F5 and F7 are the two a reviewer will remember. Budget time for them.

### Candidate venues (SCIE-indexed)

| Venue | Fit | Note |
|---|---|---|
| IEEE TIFS | ★★★★★ | best fit — expects formal theorems *and* a real implementation, which is this paper's shape |
| IEEE TDSC | ★★★★★ | strong on formal models + systems |
| IEEE IoT Journal | ★★★★ | faster, wants an IoT framing |
| Computers & Security | ★★★★ | reliable turnaround |
| IEEE TNSM / FGCS | ★★★ | fallback |

**TIFS first.** It is the venue where §XI being non-negotiable pays off.

---

## 11. Timeline

| Week | Work |
|---|---|
| 1–2 | Stage 1 (crypto core, + your Colab validation of the Bulletproofs build) |
| 3 | Stage 2 (circuits, PLONK setup) |
| 4 | Stage 3 (data + splits) |
| 5 | Stages 4–5 — **Gate 5 is make-or-break** |
| 6 | Stage 6 (overhead) |
| 7–8 | Stage 7 (attacks, ablations) |
| 9–10 | Stage 8 (full matrix, runs in background while you write) |
| 11–13 | Paper, figures |
| 14 | Internal review, submission |

~3.5 months at a sustainable pace. The two schedule risks are Stage 1 (crypto build on Colab) and Stage 8 (GPU hours); both are front-loadable, and Stage 8 can start the moment Gate 5 passes.

---

## 12. Division of labour

- **Me:** all code, circuits, notebooks, configs, figure scripts; drafts of §III–§XII.
- **You:** run notebooks on Colab, paste back logs and errors, own the final paper voice.
- **Loop:** I commit to `FL-ZKP` → you run → you paste output → I fix. One notebook per case plus the crypto bench, so a failure in one does not block the others.
