# What's left — in the order to do it

Last updated 2026-11-16. Goal: experimental work done today (Nov 16),
drafting starts tomorrow (Nov 17). This list is ordered — do items in
order, not in parallel, unless a step says otherwise.

## 1. [BLOCKING, START NOW] Real tiny_cnn/MNIST run with the F9 DP fix

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

## 2. [after #1 confirms good] Pull final numbers together

No new experiments — this is just collecting what already exists:
- Accuracy-vs-round curve (new, from #1's `real_history`)
- `B_t` adaptation curve across #1's rounds (same `real_history`)
- Attack × defense table: 8/8 standalone (`RUN_ATTACKS.ipynb`) + A1's
  real-round confirmation (FIX.md F11's final entry — already have
  this, done)
- Per-round timing breakdown (phase1/phase2/verify) — already have
  real numbers from every Colab run this session, including #1

Tell me when #1's numbers are in and I'll build the tables/figures
from them directly.

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
