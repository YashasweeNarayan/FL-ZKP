"""Generates TRACEABILITY_DEMO.ipynb: an isolated test of the §6.2
opening protocol (fl/trace.py) against the REAL crypto primitives
(crypto/dkg.py, crypto/elgamal_threshold.py), using the project's real
committee parameters (configs/mvp_3day.yaml: n_key_holders=10,
threshold=7) -- deliberately BEFORE wiring this into the actual
training pipeline, per the user's explicit "test it standalone first,
then integrate" plan (2026-11-18).

Two cases, as requested:
  CASE A: all n_c=10 key holders cooperate -> opens successfully.
  CASE B: holders {1,2,3,4} are missing (6 of 10 remain, below t=7)
          -> opening fails -- "no traceability".

Note on Case B's exact holder set: the user's original ask was "1,2,3
missing". With the REAL committee (t=7 of 10), dropping only 3 leaves
7 -- still exactly at threshold, so opening would still SUCCEED, not
fail. Confirmed with the user (2026-11-18) to keep the real n=10/t=7
params and drop a 4th holder (so {1,2,3,4} are missing, 6 remain) so
Case B actually demonstrates a failed opening. See the title cell's
markdown for this same note, surfaced to whoever runs the notebook.

Unlike every other notebook in this repo, this one needs NO GPU, NO
torch, and NO dataset -- fl/trace.py and crypto/elgamal_threshold.py
are pure EC orchestration. It runs identically on Colab (with rbcl,
the real libsodium-backed ristretto255) and in a plain CPU sandbox
with no network (JANUS_ALLOW_PURE_PYTHON_GROUP=1 selects the
test-only pure-Python backend, crypto/_ristretto_py.py -- same gate
`tests/test_round_end_to_end.py` already uses). The sanity cell below
tries rbcl first and only falls back to the pure-Python backend if
rbcl isn't installed, exactly like that test file.

Deliberately scoped OUT of this demo (left for the full-pipeline
integration step that comes after this is validated):
  - fl/evidence.py's real evidence accumulation + calibrate_theta
    (`OpeningRequest.evidence_total`/`.calibration_certificate` are
    stubbed here with a fake-but-honest certificate, since this demo
    isn't running any training rounds to accrue real evidence from).
  - `verify_opening_binds_to_identity_proof` (§6.2 step 5's R_cons
    re-check) -- that needs a genuine round's rho_t/rho_e/vrf_alpha
    and identity-binding proof, which only exist once this is wired
    into fl/round_loop.py.
  - §6.3's residual-anonymity accounting (H(T) = log2(n - o(T))) has
    NO implementation anywhere in the repo yet (fl/trace.py's own
    `revoke()` docstring explicitly defers it "to wherever the paper's
    figures are generated"). This notebook adds a minimal, real,
    demo-only `residual_anonymity_bits()` helper INLINE (not in
    fl/trace.py) so Case A's result can be reported the way the paper
    will want it -- flagged in its own cell as a draft to promote into
    a real module once the full pipeline needs it.
"""
import json
import os
import sys

DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
NB_DIR = os.path.dirname(DEMO_DIR)              # notebooks/
REPO_ROOT = os.path.dirname(NB_DIR)              # repo root
sys.path.insert(0, NB_DIR)                       # for `import _build_mvp_run_notebooks`
sys.path.insert(0, REPO_ROOT)                    # for `import fl`/`import crypto` (unused
                                                  # directly here, but kept for parity)

import _build_mvp_run_notebooks as base          # noqa: E402  (reused: nb() helper only --
                                                  # none of base's other cells fit here, since
                                                  # this notebook needs no torch/dataset/model)

nb = base.nb


TITLE_MD = """\
# Traceability demo: §6.2 opening protocol, standalone

Tests `fl/trace.py`'s opening protocol against the REAL EC primitives
(`crypto/dkg.py`, `crypto/elgamal_threshold.py`) with the project's
real committee size (`configs/mvp_3day.yaml`: `n_key_holders=10`,
`threshold=7`) -- isolated from the training pipeline, as a
correctness check before wiring §6 into `fl/round_loop.py` for the
full 4-combination run.

**Two cases:**
- **Case A** -- all 10 key holders cooperate -> opening succeeds, the
  escrowed identity is recovered exactly.
- **Case B** -- key holders 1, 2, 3, **and 4** are missing (6 of 10
  remain) -> opening fails: *no traceability*.

  Note: the original ask was "1, 2, 3 missing". With the real t=7-of-10
  committee, dropping only 3 holders leaves 7 -- still exactly at
  threshold, so opening would still succeed, not fail. Holder 4 is
  added to the missing set here (confirmed with the user) so Case B
  actually demonstrates a failed opening at 6 < 7.

**Not yet exercised here** (comes after this is validated, when §6
is wired into the real pipeline): `fl/evidence.py`'s real evidence
accumulation (Case A/B use a stubbed, already-above-threshold
evidence certificate instead), the §6.2 step 5 R_cons identity-binding
re-check (needs a genuine round's proof), and §6.3's residual-anonymity
accounting (no implementation exists anywhere in the repo yet -- this
notebook adds a minimal demo-only version in its own cell, clearly
marked as a draft).

**No GPU/torch/dataset needed.** Runs identically on Colab (real
`rbcl`-backed ristretto255) or on a plain CPU machine with no network
(falls back to the test-only pure-Python backend, same gate
`tests/test_round_end_to_end.py` uses).
"""

CLONE_CELL = base.CLONE_CELL  # identical pattern to every other notebook

INSTALL_CELL = """\
# 2. Install just the EC toolchain -- no torch, no dataset libraries:
# this notebook never trains a model.
!pip install -q rbcl pynacl pyyaml
"""

SANITY_CELL = """\
# 3. Sanity-check the backend. Tries the real libsodium-backed
# ristretto255 (rbcl) first; if it's not installed (e.g. a sandboxed
# CPU box with no network), falls back to the test-only pure-Python
# backend via JANUS_ALLOW_PURE_PYTHON_GROUP=1 -- same gate
# tests/test_round_end_to_end.py uses, and the SAME algebra either
# way (only the group-operation bindings differ).
import sys
sys.path.insert(0, '/content/FL-ZKP')
import os

from crypto.ristretto import backend_name
if backend_name() is None:
    os.environ['JANUS_ALLOW_PURE_PYTHON_GROUP'] = '1'
    import importlib
    import crypto.ristretto as _ristretto
    importlib.reload(_ristretto)

from crypto.ristretto import backend_name
print("ristretto255 backend:", backend_name())
assert backend_name() is not None, "no backend available -- this should not happen"
"""

CONFIG_CELL = """\
# 4. Load the REAL committee parameters from the project's own config
# -- not re-typed by hand, so this demo can never silently drift from
# what the actual pipeline uses.
import yaml
with open('configs/mvp_3day.yaml') as f:
    CFG = yaml.safe_load(f)

N_C = CFG['protocol']['n_key_holders']   # 10
T = CFG['protocol']['threshold']          # 7
print(f"key-holder committee: n_c={N_C}, threshold t={T}")
"""

COMMITTEE_CELL = """\
# 5. Run the real Pedersen DKG (crypto/dkg.py) to stand up the trace
# committee: PK_trace plus each holder's own final Shamir share and
# its public Feldman commitment (pk_share_i) -- IDENTICAL to what
# fl/round_loop.py's RING_CELL does for the real training pipeline
# (same function calls, same n_c/t), just without a training run
# around it.
from crypto.dkg import generate_contribution, verify_sub_share, finalize, pk_share_for

key_holder_ids = list(range(1, N_C + 1))
dkg_contributions = [generate_contribution(h, T, key_holder_ids) for h in key_holder_ids]

# Feldman-verify every sub-share before finalizing -- a real committee
# would discard any contribution that fails this and re-run DKG
# without the cheating holder; all contributions here are honest, so
# every check is expected to pass.
for c in dkg_contributions:
    for j in key_holder_ids:
        assert verify_sub_share(c.sub_shares[j], c.feldman_commitments), (
            f"holder {c.holder_id}'s sub-share to {j} failed Feldman verification"
        )

PK_trace, key_holder_shares = finalize(dkg_contributions, key_holder_ids)
pk_shares = {i: pk_share_for(i, dkg_contributions) for i in key_holder_ids}
print(f"PK_trace established among {N_C} key holders (t={T}); all Feldman checks passed.")
"""

ESCROW_CELL = """\
# 6. Stand in for one accused round's identity escrow. In the real
# pipeline this would be the Point encoding of an actual client's
# long-term pk_k, encrypted under PK_trace as part of that client's
# own round submission (crypto/elgamal_threshold.py::encrypt, called
# from fl/client.py). Here it's a synthetic point -- the point of
# this demo is the OPENING protocol's correctness, not re-deriving a
# full round submission.
from dataclasses import dataclass

from crypto.elgamal_threshold import encrypt, compute_partial, verify_partial, combine_partials
from crypto.ristretto import Scalar, scalar_mul_base
from fl.trace import OpeningRequest, run_opening_protocol, RevocationRecord, revoke

accused_client_sk = Scalar.from_int(424242)          # stand-in for a real client's long-term sk_k
accused_client_pk = scalar_mul_base(accused_client_sk)  # the escrowed plaintext -- what opening should recover

escrow_ciphertext = encrypt(accused_client_pk, PK_trace)
print("accused client's identity encrypted under PK_trace.")

# fl/evidence.py's real accumulation isn't run in this demo (no
# training rounds happened) -- stub a certificate that's already
# satisfied, so run_opening_protocol's first gate (evidence_total >
# theta) passes and the two cases below exercise the PARTIAL-DECRYPTION
# threshold logic, which is what's actually under test here.
@dataclass
class _StubCalibrationCertificate:
    theta: float

request = OpeningRequest(
    epoch=1,
    accused_pseudonym=b"demo-epoch-pseudonym",
    evidence_total=999.0,                       # comfortably above the stub theta
    calibration_certificate=_StubCalibrationCertificate(theta=10.0),
    escrow_ciphertext=escrow_ciphertext,
)
transcript = f"opening:{request.epoch}:{request.accused_pseudonym.hex()}".encode()


def partials_from(holder_ids):
    \"\"\"Collect real Chaum-Pedersen-proven partial decryptions from
    exactly the given holders -- mirrors fl/trace.py::collect_partial_from_holder
    for every non-accused holder (none of these holders is the accused
    party in this demo, so none recuses).\"\"\"
    return {
        i: compute_partial(escrow_ciphertext, key_holder_shares[i], pk_shares[i], transcript)
        for i in holder_ids
    }
"""

CASE_A_CELL = """\
# 7. CASE A -- all 10 key holders cooperate.
verdict_a = run_opening_protocol(
    request, partials_from(key_holder_ids), pk_shares, T, admin_authorized=True,
)

print("CASE A -- all", N_C, "holders cooperate")
print("  opened:", verdict_a.opened)
if verdict_a.opened:
    print("  recovered identity matches the accused client's real pk:",
          verdict_a.opened_pk == accused_client_pk)
else:
    print("  reason:", verdict_a.reason)

assert verdict_a.opened and verdict_a.opened_pk == accused_client_pk, (
    "Case A should open and recover the exact escrowed identity"
)
"""

CASE_B_CELL = """\
# 8. CASE B -- key holders 1, 2, 3, and 4 are missing (6 of 10 remain,
# below t=7) -> opening must fail.
missing_holders = [1, 2, 3, 4]
present_holders = [i for i in key_holder_ids if i not in missing_holders]

verdict_b = run_opening_protocol(
    request, partials_from(present_holders), pk_shares, T, admin_authorized=True,
)

print("CASE B -- holders", missing_holders, "missing;", len(present_holders), "of", N_C, "present")
print("  opened:", verdict_b.opened)
print("  reason:", verdict_b.reason)

assert not verdict_b.opened, "Case B should fail -- 6 of 10 is below t=7"
print("\\n-> no traceability: the accused identity stays anonymous when the committee can't reach threshold.")
"""

SWEEP_MD = """\
## Bonus: where exactly does the threshold flip?

A quick sweep -- drop an increasing number of holders (always from
the low end of the id range) and re-run the opening protocol at each
count -- shows the flip from "opens" to "no traceability" happens
exactly at `t=7` present holders, not gradually. This is the same
shape of plot the full pipeline's traceability section will want,
just over holder-count here instead of over training round.
"""

SWEEP_CELL = """\
# 9. Sweep holders-present from N_C down to 0 and record whether
# opening succeeds at each count.
sweep_results = []
for n_present in range(N_C, -1, -1):
    present = key_holder_ids[:n_present]
    v = run_opening_protocol(request, partials_from(present), pk_shares, T, admin_authorized=True)
    sweep_results.append((n_present, v.opened))
    print(f"  holders present: {n_present:2d}  ->  opened: {v.opened}")

import matplotlib.pyplot as plt

counts = [c for c, _ in sweep_results]
opened_flags = [1 if o else 0 for _, o in sweep_results]

fig, ax = plt.subplots(figsize=(7, 3.5))
colors = ["#2e7d32" if o else "#c62828" for o in opened_flags]
ax.bar(counts, opened_flags, color=colors, width=0.8)
ax.axvline(T - 0.5, color="black", linestyle="--", linewidth=1, label=f"threshold t={T}")
ax.set_xlabel("key holders present (of n_c=10)")
ax.set_ylabel("opening succeeded")
ax.set_yticks([0, 1])
ax.set_yticklabels(["no (0/1)", "yes (1/1)"])
ax.set_xticks(counts)
ax.set_title("Opening success vs. number of cooperating key holders")
ax.legend(loc="lower left")
plt.tight_layout()
_out_dir = '/content' if os.path.isdir('/content') else '.'
_out_path = os.path.join(_out_dir, 'traceability_threshold_sweep.png')
plt.savefig(_out_path, dpi=150)
plt.show()
print("saved:", _out_path)
"""

REVOKE_MD = """\
## Revocation + residual anonymity (Case A only, since only Case A opened)

`fl/trace.py::revoke()` removes the opened identity from the ring and
reports the new ring size. §6.3's accountability-anonymity budget,
`H(T) = log2(n - o(T))` (bits of residual anonymity after `o(T)`
openings out of `n` ring members), has **no implementation anywhere
in the repo yet** -- `revoke()`'s own docstring defers it "to wherever
the paper's figures are generated". The helper below is a minimal,
real, demo-only version so this notebook can report the number the
paper will actually want; it is NOT yet promoted into `fl/trace.py`
(flagging that as a candidate for the full-pipeline integration step).
"""

REVOKE_CELL = """\
# 10. Revoke the opened identity (Case A) and report residual anonymity.
import math

def residual_anonymity_bits(ring_size_before: int, n_opened_so_far: int) -> float:
    \"\"\"DEMO-ONLY draft of ARCHITECTURE.md SS6.3's H(T) = log2(n - o(T)).
    Not yet a real module -- promote into fl/trace.py (with a test)
    once SS6 is wired into the actual per-round pipeline, where o(T)
    will be tracked across real rounds rather than computed once here.
    \"\"\"
    remaining = ring_size_before - n_opened_so_far
    if remaining <= 0:
        return float("-inf")
    return math.log2(remaining)

original_ring = [accused_client_pk] + [
    scalar_mul_base(Scalar.random()) for _ in range(49)
]  # a 50-member ring, same size as the real pipeline's n_clients, with the accused client in it

new_ring, revocation_record = revoke(verdict_a, original_ring, epoch=request.epoch)

bits_before = residual_anonymity_bits(len(original_ring), n_opened_so_far=0)
bits_after = residual_anonymity_bits(len(original_ring), n_opened_so_far=1)

print(f"ring size before revocation: {len(original_ring)}")
print(f"ring size after revocation:  {revocation_record.new_ring_size}")
print(f"residual anonymity before:  H(T) = log2({len(original_ring)} - 0) = {bits_before:.3f} bits")
print(f"residual anonymity after:   H(T) = log2({len(original_ring)} - 1) = {bits_after:.3f} bits")
"""

CLOSING_MD = """\
## Summary

- **Case A** (all 10 holders): opening succeeds, recovers the exact
  escrowed identity.
- **Case B** (holders 1-4 missing, 6 of 10 present): opening fails --
  no traceability, exactly as the certified-false-accusation design
  requires (below threshold, nothing is revealed).
- The success/failure boundary sits exactly at `t=7` present holders
  (the sweep above), not gradually.
- Revocation shrinks the ring by exactly one member; residual
  anonymity dropped from `log2(50)=5.64` to `log2(49)=5.61` bits for
  this single opening (the demo-only `residual_anonymity_bits` helper
  above).

**Left for the full-pipeline integration step** (next, per plan):
wiring this into `fl/round_loop.py` so `fl/evidence.py` triggers real
openings from real per-round evidence instead of a stubbed
certificate, exercising the step-5 R_cons identity-binding re-check
against a genuine round's proof, running all 4 (model, dataset)
combinations end to end, and promoting `residual_anonymity_bits` into
a real, tested module (likely alongside `fl/trace.py`) so the paper's
traceability section can report, per round, both WHEN an opening
happened and the resulting H(T) curve.
"""


def build_traceability_demo_notebook():
    return nb([
        ("md", TITLE_MD),
        ("code", CLONE_CELL),
        ("code", INSTALL_CELL),
        ("code", SANITY_CELL),
        ("code", CONFIG_CELL),
        ("code", COMMITTEE_CELL),
        ("code", ESCROW_CELL),
        ("code", CASE_A_CELL),
        ("code", CASE_B_CELL),
        ("md", SWEEP_MD),
        ("code", SWEEP_CELL),
        ("md", REVOKE_MD),
        ("code", REVOKE_CELL),
        ("md", CLOSING_MD),
    ])


if __name__ == "__main__":
    out_path = os.path.join(DEMO_DIR, "TRACEABILITY_DEMO.ipynb")
    with open(out_path, "w") as f:
        json.dump(build_traceability_demo_notebook(), f, indent=1)
    print(f"wrote {out_path}")
