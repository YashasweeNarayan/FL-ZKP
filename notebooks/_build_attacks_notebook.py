"""Generates notebooks/RUN_ATTACKS.ipynb from plain Python strings, same
pattern as _build_notebooks.py (reproducible, reviewable as text). Run
this to (re)build the notebook after editing the ATTACK_CELLS list below.

Why a separate script instead of adding to _build_notebooks.py directly:
that file's CELLS lists are each ~150-900 lines; appending here keeps the
diff small and this notebook's cells reviewable on their own.
"""
import json
import os
import sys

NB_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, NB_DIR)
from _build_notebooks import nb  # noqa: E402 -- reuse the exact same cell-dict shape


ATTACK_CELLS = [

("md", """# RUN_ATTACKS — A1/A3/A4/A5/A7 vs. their defenses

Per `STATUS.md`: every attack in `attacks/` is written but **not run
anywhere** before this notebook — no test, no prior notebook imports
any of them. This is that first run, against real `rbcl`/libsodium
crypto (not the pure-Python test backend).

This is standalone from the MNIST training run: each attack exercises
a single crypto primitive/defense pair directly (a ring of keys, a
DKG committee, a range proof), not the full multi-round training loop
— so this notebook runs in **minutes, not hours**.

| Attack | What it tries | Defense under test |
|---|---|---|
| A1 | model-replacement: submit a hugely scaled-up update to overwrite the global model in one round | V3 — the range proof bounding the committed update norm by `B_t` |
| A3 | Sybil: one client submits 5 updates in one round, hoping ring-signature anonymity hides the reuse | V2 — the per-round key image, deterministic given `(sk, rho_t)`, is identical across all 5 and gets deduped |
| A4 | free-riding: (a) replay a stale signed update from a past round, (b) falsely claim "not sampled" to skip training | (a) `rho_t` binding in V1, (b) the VRF disjunction in R_cons — a client can't lie about its own public, recomputable sampling outcome |
| A5 | identity-laundering: (a) splice a genuine R_cons proof onto someone else's round tags, (b) splice a genuine ring signature onto someone else's commitment | V4 — R_cons's AND-composition binds everything to ONE joint Fiat-Shamir challenge; V1 — CLSAG binds the signed message into the same challenge |
| A7 | collusive de-anonymization: (a) 6-of-10 key holders (one short of `t=7`) try to open an identity anyway, (b) 2-of-3 colluding mix nodes try to see both ends of a route | (a) Lagrange interpolation is underdetermined below `t` points — colluders get garbage, not the real secret; (b) no single hop (or pair, short of all 3) sits on both the client-facing and server-facing end of the cascade |

A2 (label-flipping) and A6 (gradient inversion) are out of scope for
this notebook — both need many real training rounds or a real DLG
optimization to produce evidence, see `attacks/__init__.py` and
`ROADMAP.md` §0.1.

**If anything in this notebook fails unexpectedly** (an attack's
`defense_triggered` comes back `False` where the table above says it
shouldn't), stop and send me the full cell output and traceback —
that would be a real defense failure, worth knowing before the paper
cites these as tested, not something to retry past.
"""),

("md", "## Clone (idempotent — same pattern as the other notebooks, no `rmtree`)"),

("code", """import os

if not os.path.exists('/content/FL-ZKP'):
    !git clone -q https://github.com/YashasweeNarayan/FL-ZKP.git /content/FL-ZKP
else:
    !cd /content/FL-ZKP && git fetch -q && git reset -q --hard origin/main

%cd /content/FL-ZKP
!git log --oneline -1
"""),

("md", "## Install dependencies\n\n`rbcl` gives `crypto/ristretto.py` its real ristretto255 bindings (mainline PyNaCl never shipped these — FIX.md F8)."),

("code", """!pip install -q rbcl
!pip install -q pynacl pyyaml numpy
"""),

("md", "**If this is the first time this cell has run in this session, restart now: Runtime -> Restart session.** Then continue from the next cell (do not re-run the install cell)."),

("code", """import sys
sys.path.insert(0, '/content/FL-ZKP')

import rbcl
assert hasattr(rbcl, "crypto_core_ristretto255_from_hash"), (
    "rbcl imported but is missing crypto_core_ristretto255_from_hash -- "
    "see crypto/ristretto.py's import fallback and FIX.md F8."
)

from crypto.ristretto import backend_name
print("group backend:", backend_name())
assert backend_name() != "pure_python", "this run is meant to test against real rbcl/libsodium, not the pure-Python fallback"
"""),

("md", "## Setup: ring of client keys, round nonces, DKG committee\n\nSame conventions `fl/round_loop.py::run_training_loop` itself uses, so this test exercises the real protocol's exact objects, not stand-ins."),

("code", """import yaml
import numpy as np

with open('configs/mvp_3day.yaml') as f:
    CFG = yaml.safe_load(f)

t = CFG['protocol']['threshold']        # 7
n_c = CFG['protocol']['n_key_holders']  # 10
print(f"committee: t={t}, n_c={n_c}")

from crypto.ristretto import Scalar, scalar_mul_base

N_RING = 12  # small, standalone ring for these primitive-level tests -- independent of the n_clients=50 used in the real MVP run
client_sks = [Scalar.random() for _ in range(N_RING)]
ring = [scalar_mul_base(sk) for sk in client_sks]

# same convention as fl/round_loop.py::run_training_loop -- rho_t/rho_e/vrf_alpha
# are derived from the round/epoch index, not chosen freely.
ROUND_T = 3
EPOCH_ROUNDS = CFG.get('protocol', {}).get('epoch_rounds', 5)
rho_t = f"round-{ROUND_T}".encode()
rho_e = f"epoch-{ROUND_T // EPOCH_ROUNDS}".encode()
vrf_alpha = rho_t
rho_t_prev = f"round-{ROUND_T - 1}".encode()

print(f"ring size: {len(ring)}, rho_t={rho_t}, rho_e={rho_e}")
"""),

("code", """from crypto.dkg import generate_contribution, verify_sub_share, finalize, pk_share_for
from crypto.elgamal_threshold import encrypt, compute_partial, verify_partial, combine_partials

holder_ids = list(range(1, n_c + 1))
contributions = [generate_contribution(hid, t, holder_ids) for hid in holder_ids]

# Feldman-verify every sub-share before trusting any contribution (an
# honest DKG run -- all should pass; this is the same check a real
# deployment runs to exclude a cheating holder before finalizing).
for contrib in contributions:
    for hid in holder_ids:
        assert verify_sub_share(contrib.sub_shares[hid], contrib.feldman_commitments), (
            f"holder {contrib.holder_id}'s sub-share to {hid} failed Feldman verification"
        )

pk_trace, final_shares = finalize(contributions, holder_ids)
pk_shares = {hid: pk_share_for(hid, contributions) for hid in holder_ids}
print("DKG committee finalized, PK_trace derived, all sub-shares Feldman-verified.")
"""),

("md", "## A1 — model-replacement / scaling attack vs. V3 (norm bound)\n\nUses a `B_t` computed by the same median/MAD rule (`fl/norm_rule.py::compute_threshold`) the real protocol uses, from a handful of honest-sized quantized deltas -- not an arbitrary number."),

("code", """from attacks import a1_scaling
from fl.client import quantize, center_mod_p_mask
from fl.norm_rule import compute_threshold

TOY_DIM = 2000
rng = np.random.default_rng(0)

# a handful of honest-sized deltas (small, like real local-training
# updates), quantized the same way a real client would -- gives B_t in
# the actual deployed units. Centered the same way the real pipeline
# centers before bounding a norm (fl/client.py::center_mod_p_mask) --
# the raw quantize() output wraps negative entries to near P_MASK, which
# would swamp the norm regardless of the real delta's size (same
# mod-wrap bug class already fixed in server.py's V3/B_t-EMA path;
# a1_scaling.py itself was fixed to do this too, see FIX.md).
honest_norms = []
for _ in range(10):
    honest_delta = rng.normal(0, 0.05, size=TOY_DIM)
    honest_norms.append(np.linalg.norm(center_mod_p_mask(quantize(honest_delta)).astype(np.float64)))
threshold = compute_threshold(np.array(honest_norms))
print(f"bootstrapped B_t = {threshold.B_t:,.1f} (median={threshold.median:,.1f}, mad={threshold.mad:,.1f})")

w_t = rng.normal(0, 1.0, size=TOY_DIM)
w_target = rng.normal(0, 1.0, size=TOY_DIM)  # attacker's desired model -- far from w_t

result_a1 = a1_scaling.run_attack(w_t, w_target, threshold.B_t)
print(result_a1)
"""),

("md", "## A3 — Sybil / multi-submission vs. V2 (key-image dedup)"),

("code", """from attacks import a3_sybil

ATTACKER_IDX = 2
result_a3 = a3_sybil.run_attack(client_sks[ATTACKER_IDX], ATTACKER_IDX, ring, rho_t, rho_e, n_submissions=5)
print(result_a3)
"""),

("md", "## A4 — free-riding: replay vs. `rho_t` binding, and false not-sampled claim vs. the VRF disjunction"),

("code", """from attacks import a4_freeride

old_message = b"round-t-minus-1-genuine-update-commitment"
result_a4_replay = a4_freeride.run_attack_replay(
    client_sks[ATTACKER_IDX], ATTACKER_IDX, ring, rho_t_prev, rho_e, rho_t, old_message,
)
print(result_a4_replay)

SAMPLING_RATE = CFG.get('protocol', {}).get('sampling_rate', 0.25)

# The "lie" this attack models only exists if the client was genuinely
# sampled (b_k=True) -- otherwise "claiming not sampled" isn't a lie at
# all. VRF output is deterministic in (sk, pk, alpha), so search a few
# alpha candidates for one where this client's real draw is True, same
# as a real deployment would see across different rounds.
from crypto.vrf import evaluate as _vrf_evaluate, is_sampled as _vrf_is_sampled
for _attempt in range(100):
    _alpha_candidate = f"round-{ROUND_T}-sampling-search-{_attempt}".encode()
    _beta = _vrf_evaluate(client_sks[ATTACKER_IDX], ring[ATTACKER_IDX], _alpha_candidate).beta
    if _vrf_is_sampled(_beta, SAMPLING_RATE):
        vrf_alpha_sampled = _alpha_candidate
        break
else:
    raise RuntimeError(f"no sampled round found for this client in 100 tries at p={SAMPLING_RATE} -- try more attempts")
print(f"found a round where this client is genuinely sampled after {_attempt + 1} attempt(s)")

result_a4_false_claim = a4_freeride.run_attack_false_not_sampled_claim(
    client_sks[ATTACKER_IDX], ring[ATTACKER_IDX], vrf_alpha_sampled, SAMPLING_RATE,
)
print(result_a4_false_claim)
"""),

("md", "## A5 — identity-laundering: splice an R_cons proof, and splice a ring signature's message"),

("code", """from attacks import a5_splice
from crypto.ristretto import Scalar as _Scalar

VICTIM_IDX = 5
vrf_alpha_victim = f"round-{ROUND_T}-victim".encode()
vrf_alpha_attacker = f"round-{ROUND_T}-attacker".encode()

result_a5_splice = a5_splice.run_attack(
    client_sks[VICTIM_IDX], client_sks[ATTACKER_IDX], rho_t, rho_e, vrf_alpha_victim, vrf_alpha_attacker,
)
print(result_a5_splice)

victim_commitment_bytes = b"victim-genuine-commitment-bytes"
result_a5_key_image = a5_splice.run_attack_key_image_splice(
    client_sks[ATTACKER_IDX], ATTACKER_IDX, ring, rho_t, rho_e, victim_commitment_bytes,
)
print(result_a5_key_image)
"""),

("md", "## A7 — collusive de-anonymization: 6-of-10 key holders short of `t=7`, and 2-of-3 colluding mix nodes"),

("code", """from attacks import a7_collusion

pk_victim = ring[VICTIM_IDX]
ct_victim = encrypt(pk_victim, pk_trace)
transcript = b"a7-collusion-demo"

all_partials = {}
for hid in holder_ids:
    pd = compute_partial(ct_victim, final_shares[hid], pk_shares[hid], transcript)
    assert verify_partial(pd, ct_victim, pk_shares[hid], transcript), f"holder {hid}'s own partial failed self-verification"
    all_partials[hid] = pd

COLLUDING_IDS = holder_ids[:6]   # 6 < t=7, deliberately one short
HONEST_IDS = [h for h in holder_ids if h not in COLLUDING_IDS]
colluding_partials = [all_partials[h] for h in COLLUDING_IDS]
honest_partials = [all_partials[h] for h in HONEST_IDS]

result_a7_threshold = a7_collusion.run_attack(ct_victim, colluding_partials, honest_partials, pk_victim, t=t, n_c=n_c)
print(result_a7_threshold)

result_a7_mixnet = a7_collusion.run_attack_mixnet_partial_view(
    client_addr="client-victim", server_addr="server", mix_node_ids_colluding=[0, 1],  # hop 0 and hop 1, NOT hop 2 (the server-facing hop)
)
print(result_a7_mixnet)
"""),

("md", "## Summary — attack x defense table\n\nARCHITECTURE.md §10 calls this \"the single most convincing figure in the paper\". `defense_triggered=True` means the attack was caught."),

("code", """import pandas as pd

all_results = [
    result_a1, result_a3,
    result_a4_replay, result_a4_false_claim,
    result_a5_splice, result_a5_key_image,
    result_a7_threshold, result_a7_mixnet,
]

df = pd.DataFrame([r.__dict__ for r in all_results])
df['status'] = df['defense_triggered'].map({True: 'CAUGHT', False: 'NOT CAUGHT'})
print(df[['attack_name', 'defense_name', 'status']].to_string(index=False))

n_caught = df['defense_triggered'].sum()
print(f"\\n{n_caught}/{len(df)} attacks caught by their target defense.")
if n_caught < len(df):
    print("At least one attack was NOT caught -- see the `detail` column below before trusting this table in the paper.")
    print(df[~df['defense_triggered']][['attack_name', 'detail']].to_string(index=False))
"""),

("md", "## Save the results table (optional, for the paper)"),

("code", """from google.colab import drive
if not os.path.isdir('/content/drive/MyDrive'):
    drive.mount('/content/drive')

out_path = '/content/drive/MyDrive/FL-ZKP/results/attack_results.csv'
os.makedirs(os.path.dirname(out_path), exist_ok=True)
df.to_csv(out_path, index=False)
print(f"saved to {out_path}")
"""),

]


if __name__ == "__main__":
    with open(os.path.join(NB_DIR, "RUN_ATTACKS.ipynb"), "w") as f:
        json.dump(nb(ATTACK_CELLS), f, indent=1)
    print("wrote notebooks/RUN_ATTACKS.ipynb")
