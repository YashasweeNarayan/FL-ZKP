# What's left — in the order to do it

Last updated 2026-11-16. Goal: experimental work done today (Nov 16),
drafting starts tomorrow (Nov 17). This list is ordered — do items in
order, not in parallel, unless a step says otherwise.

## 1. [BLOCKING, START NOW] Real tiny_cnn/MNIST run with the F9 DP fix

**STATUS UPDATE, 2026-11-16: this run completed (20/20 rounds, all
accepted, git-checkpointed per F13, authored as Yashaswee Narayan) --
but its result is now INVALIDATED, and item #1 is NOT actually done.**
Final numbers were `test_acc=0.3334` / `val_acc=0.3361` -- far below
the 90%+ expected. Investigated and root-caused (not guessed): F9's
fix only partially closed the DP-noise bug -- it scaled each client's
noise share correctly but still applied it to EVERY client
unconditionally, including the ~38 dummy (unsampled) clients each
round, which `ARCHITECTURE.md` line 201 explicitly requires to submit
`q_k = 0`. This left a residual 2x aggregate-noise overshoot (full
writeup: FIX.md F14, STATUS.md's F9 entry). **Fixed in `fl/client.py`
just now** -- the noise draw is gated on the same `b_k` sampling flag
Step 1 already computes, confirmed by a standalone numpy Monte Carlo
check (old std=4.9967 ≈ predicted 2x overshoot, new std=2.4286 ≈
predicted target) and the full local suite (113 passed/5 skipped,
unchanged). **This run needs to be redone from round 0** with the
fix in place (same notebook, git-checkpoint mechanism from #1 below
still applies) before its accuracy number can go in the paper. The
security/crypto results from the completed run (`accepted=50/50,
V5=PASS, F7e=PASS, aggregate=EXACT` every round) are NOT affected by
this bug and remain valid -- only the accuracy number needs a re-run.

**Why first**: this is the one number the paper's utility claim
currently rests on, and it's the longest-running thing on this list.
The last real-Colab run of this exact setup, before F9's fix, hit
15.4% (near-random) after 12 rounds — purely from a DP-noise scaling
bug, now fixed (`fl/client.py`, commit `39ef154`). This run is what
confirms the fix actually works at the real training scale, not just
in the unit tests (113 passed/5 skipped locally, but the fix path is
torch-only — never exercised off Colab).

**Notebook**: `notebooks/RUN_MVP_TINY_CNN_MNIST.ipynb` — ready to run
as-is, no changes needed from you. I already fixed one thing in it
just now: the checkpoint filename now has an `_f9fix` suffix, so this
run starts fresh from round 0 instead of silently resuming the old,
pre-fix checkpoint (which would have corrupted the result — the old
checkpoint's weights reflect ~7x too much DP noise baked in).

**Run it top to bottom.** 20 rounds, `tiny_cnn` (~20.6K params), MNIST,
50 clients — scope already justified in the notebook's own first
markdown cell (matches WhiteCloak/PQS-BFL's model scale). If Colab
disconnects, just re-run from cell 1 — it resumes from the Drive
checkpoint automatically, from this run's checkpoint, not the old one.

**When it's done**, come back with:
- The final `test_acc`/`val_acc` printed by the last cell
- Whether any round aborted (`n_accepted` vs `len(real_history)` from
  the training cell's own print)
- If accuracy still looks wrong (near-random, or NaN, or stuck),
  paste the full training-cell output, not just the final numbers —
  there may be a second issue to chase, and we want to know that
  TODAY, not during drafting.

## 1.5. [after #1 confirms good, before drafting] Fused end-to-end run: attack INSIDE the full training run

**Why**: #1 proves utility (converges under real overhead), F11
proves security (catches A1 inside a real loop, but only at toy
scale -- 9 clients, 3 rounds). Neither alone is "attackers sitting
inside the FULL 50-client/20-round run, caught every round, while the
honest majority still converges normally." That combined claim is the
strongest single result this paper can show, and it's cheap to get:
`cheater_client_idx`/`cheater_force_delta` are already proven correct
(F11).

**Malicious-client count, decided 2026-11-16**: **7 of 50 (14%)**, not
1. This matches `ARCHITECTURE.md`'s own stated experimental target
(contributions table, N7: "n=50 ... 7 adversaries") -- using any other
number would make the paper inconsistent with what it already claims
to validate. The system's theoretical tolerance is `f < n/2 = 25`
(§2.2, A2) -- that ceiling is for a stress-test/sensitivity figure
only (see #1.6 below), not the main result.

**Plan**: a second run, same model/rounds/client-count as #1, with 7
clients forced malicious via `craft_scaled_update` the whole time
(same construction F11 used, same crafted delta reused across all 7 --
noted as a simplifying choice, not independently-drawn attacks).
Produces a second accuracy-vs-round curve to put next to #1's clean
one -- "convergence with vs. without 14% active attackers" is a
strong, easy-to-read figure. Round-availability math (below) also
matters here: with 7 malicious clients excluded from `k_min=8`
eligibility, abort probability rises from ~4.5% to ~9.4% -- expect,
don't be alarmed by, more ABORTED rounds in this run's log than in
#1's.

## 1.6. [after #1 confirms good, before drafting] Attack-type x malicious-fraction grid + one concurrent-mixed run

**Decided 2026-11-16, after discussing what a reviewer will ask for.**
Two separate things, both needed, for different reasons -- do not
substitute one for the other:

1. **Grid**: {A1, A3, A4-replay, A5} x {10%, 20%, 30%} malicious
   clients, ONE attack type per run (not mixed) -- gives a clean,
   attributable defense-trigger-rate table per attack/fraction cell.
   This is what makes the result comparable to prior Byzantine-FL work
   (Krum/Trimmed-mean/FLTrust/RoFL all report a fraction sweep, never
   a single fixed point). Run at **5 rounds**, not 20 -- the security
   claim ("defense fires, round still completes for honest clients")
   doesn't need a long run to be valid, only the *accuracy* curves in
   #1/#1.5 do. State this scope explicitly in the paper's experimental
   setup section.
2. **One concurrent-mixed run**: 30% malicious, but split across
   attack TYPES on different clients simultaneously (not all A1) --
   this is the composition stress test. Your paper's actual novelty
   (`R_cons` closing the identity-laundering hole between an escrow
   ciphertext and a commitment) is a claim about composition
   specifically; a reviewer's natural question is "does this still
   hold when different adversaries act in different roles at once,"
   and the per-attack grid above never answers that. One run at 5
   rounds answers it.

Computed abort-probability table (exact, from this config's
`p=0.25, n=50, k_min=8`; `f`=malicious count, excluded from
`k_min`-eligibility):

| malicious | honest | E[sampled] | P(round aborts) |
|---|---|---|---|
| 0% (f=0) | 50 | 12.50 | 4.53% |
| 10% (f=5) | 45 | 11.25 | 9.41% |
| 20% (f=10) | 40 | 10.00 | 18.20% |
| 30% (f=15) | 35 | 8.75 | 32.23% |

Reproduces ARCHITECTURE.md line 339's stated ~4.5% at f=0, confirming
the model -- this table itself is worth a line in the experimental
setup section, since it predicts (and explains) why the 30% cells
will show more aborted rounds than the 10% cells, independent of
whether any defense is working correctly.

**Engineering needed before this can run**: `cheater_client_idx` is
currently a single int in `fl/round_loop.py::run_training_loop` --
needs generalizing to accept multiple indices (for both the 7-client
fused run in #1.5 and the 5/10/15-client grid cells here) before any
of this can be built. This is the next thing being built now.

## 1.7. [can run any time, independent of #1/#1.5/#1.6] Remaining 3 of the 2-model x 2-dataset matrix

**Added 2026-11-16.** `configs/mvp_3day.yaml`'s `mvp_scope.models` is
`[tiny_cnn, small_resnet]`, over both CASE1 (MNIST, IID) and CASE2
(FEMNIST, non-IID) — a 2x2 matrix. Item #1 above is 1 of the 4 cells
(tiny_cnn+MNIST); these are the other 3, newly built just now so all
four stay structurally in sync (same clone/install/git-checkpoint
cells, same F14 DP fix already in place — no stale-checkpoint risk,
since none of these three have ever been run):

- `notebooks/RUN_MVP_TINY_CNN_FEMNIST.ipynb`
- `notebooks/RUN_MVP_SMALL_RESNET_MNIST.ipynb`
- `notebooks/RUN_MVP_SMALL_RESNET_FEMNIST.ipynb`

Generated by `notebooks/_build_mvp_run_notebooks.py` (new) — run that
script, not hand-edits, if the shared cells (clone/install/sanity/
git-checkpoint/evaluate) ever need to change, so the four notebooks
can't drift apart the way hand-maintained copies would. Each pushes
checkpoints to the SAME shared `colab-checkpoints` branch as #1, under
a filename keyed by case+model (`case1_small_resnet_r20.pt`, etc.), so
all four can be run in the same or different sessions without
colliding.

**Not yet run — Colab-confirmation still needed for all three**,
same limitation as everything torch-dependent in this repo (this
sandbox has no torch, no GPU, no network for installs). Verified here
to the extent this sandbox allows: JSON validity, cell-by-cell
structural parity with the already-run tiny_cnn+MNIST notebook, every
code cell's Python syntax (checked with `ast.parse`, magics stripped),
and the dataset/model parameter wiring by direct inspection against
`fl/models.py` (`build_model`'s `num_classes`/`in_channels` — 10/1 for
MNIST, 62/1 for FEMNIST) and `fl/data_noniid.py` (`unseen_test` is
FEMNIST's headline metric, writer-disjoint, not `global_val`).

**Suggested order, lowest-risk first**: `RUN_MVP_SMALL_RESNET_MNIST`
(same dataset as #1, already proven to load without Drive; only the
model changes) before either FEMNIST run (new dataset path, never
exercised end-to-end on Colab yet, higher chance of a first-run
surprise worth catching early rather than on the last run before
drafting).

## 2. [after #1 confirms good] Pull final numbers together

No new experiments — this is just collecting what already exists:
- Accuracy-vs-round curve (new, from #1's `real_history`)
- `B_t` adaptation curve across #1's rounds (same `real_history`)
- Attack × defense table: 8/8 standalone (`RUN_ATTACKS.ipynb`) + A1's
  real-round confirmation (FIX.md F11's final entry — already have
  this, done)
- Per-round timing breakdown (phase1/phase2/verify) — already have
  real numbers from every Colab run this session, including #1
- Grid + mixed-run results from #1.6

Tell me when #1's numbers are in and I'll build the tables/figures
from them directly.

## 2.5. Traceability status — answered 2026-11-16, one real gap flagged

Three separate mechanisms, not one — status differs across them:

1. **Core identity-opening** (`shamir.py`/`dkg.py`/`elgamal_threshold.py`/
   `reshare.py`): **confirmed working**, real Colab/`rbcl`. 7-of-10
   holders recover, 6-of-10 correctly fail; the secret survives a
   committee reshare. One real bug found+fixed along the way (`dkg.py`
   wasn't wrapping shares as proper `ShamirShare` objects, commit
   `cc9a59b`).
2. **Accountability/blame path** (F7f's bisection, `SuspectTracker`,
   its own independent `f7f_share_threshold=4` Shamir sharing):
   **confirmed working, after a real bug and a real fix** — first
   Colab run banned 5 innocent clients and missed the real cheater
   (reused the main round's threshold=7, so every bisection leaf group
   was unverifiable by construction). Fixed by giving F7f its own
   threshold, matched to the bisection floor (`min_group_size=4`).
   Re-run: converges correctly on the real cheater, no false bans.
3. **`trace.py` itself — the actual end-to-end "open this identity"
   orchestration — has NEVER been run.** No local test file, no Colab
   run. All three things it calls are individually confirmed; the
   function tying them together isn't. This is a real, not-yet-closed
   gap: **if the paper asserts the full end-to-end traceability claim
   (not just "the pieces work"), `trace.py` needs at least one real
   run before that claim is made.** Candidate for a cheap addition if
   time allows after #1/#1.5/#1.6 — flagging here so it isn't
   forgotten, not scheduling it yet given the timeline.

Also noted, not a bug: `r_cons.py`'s sampling-disjunction OR-proof has
its exact encoding left generic (module's own docstring) — a design
decision not yet finalized, should be stated as a scope limitation if
the paper's writing doesn't already resolve it.

## 3. [SKIP for this submission, unless #1 finishes with hours to spare]

- `small_resnet` run, FEMNIST/non-IID run (`CASE2_NONIID_FEMNIST.ipynb`
  exists but hasn't been run with current fixes) — scope these out,
  state as future work in Limitations. A second model/dataset is not
  required for a defensible submission, and there isn't runway for
  the bug-fix-rerun cycle this session has shown is likely.
- Extending A3/A4-replay into real-round gates (like A1's F11) — one
  fully-verified integrated exemplar (A1) plus 8/8 isolated results is
  a legitimate, honestly-scoped evaluation. Document in the paper
  *why* A1 was chosen as the exemplar and why the rest are adequately
  isolated (reasoning already written in FIX.md F11).

## 4. [tomorrow, Nov 17] Start drafting

Once #1/#2 are done, drafting can lean heavily on material that
already exists rather than being written from scratch:
- **System Design / Threat Model**: adapt `ARCHITECTURE.md` directly
- **Security Analysis**: adapt the V1–V5/F7e rationale already written
  as docstrings in `fl/server.py`/`fl/client.py` and the root-cause
  writeups in `FIX.md`
- **Experimental Setup / Results**: from #2's tables/figures
- **Limitations**: explicitly own the scope cuts from #3 as deliberate,
  reasoned decisions — this reads as rigor, not as a gap, if stated
  plainly with the reasoning already on record in FIX.md

Tell me which section to start on first once you're back from #1, and
I'll begin drafting in parallel with whatever you're doing.
