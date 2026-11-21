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

("md", """## Gate — A1 injected into a REAL round, not tested in isolation

Fair question raised after the first run of this notebook: every result
above tests a defense primitive directly (a ring of keys, a DKG
committee, hand-crafted inputs) -- it proves V3/V2/V1/V4/R_cons/the
threshold/the mix cascade are each correct in isolation, but it is NOT
the same claim as "JANUS-FL rejects this attack during real federated
training." This gate closes that gap for A1: one malicious client, sitting
among genuinely honest clients, inside a real multi-round `tiny_cnn`/MNIST
`run_training_loop` call -- same pattern already proven once for the F7e
cheater (`VERIFY_FIXES_F6_F7.ipynb`'s Gate E), extended here with a new
`cheater_force_delta` parameter on `run_training_loop` (and `_force_delta`
on `round_step_phase1`) that substitutes a crafted delta for one client's
real trained delta while every other step of its submission -- ring
signature, VRF, masking, sketching, the real range proof -- runs exactly
as it would for an honest client.

A3 (Sybil) and A4-replay are genuinely per-round attacks too -- closed
below via `tests/test_a3_sybil_real_round.py` and
`tests/test_a4_replay_real_round.py` (2026-11-21), run directly against
`fl/server.py::run_round` (same real-crypto, torch-free entry point
`tests/test_round_end_to_end.py` already exercises) rather than through
`run_training_loop`: a genuine SECOND submission from the SAME client in
the SAME round (A3) or a genuine submission REUSED across two different
rounds (A4-replay) has no slot in `run_training_loop`'s one-entry-per-
client-index dict structure without restructuring that 900-line
orchestration file, which this deliberately does not do. Unlike A1, A3
and A4-replay have no accuracy dimension to show either way: a
correctly-caught duplicate or replay never reaches aggregation, so it
cannot move the model one way or the other -- their real-round claim is
"the round completes normally for honest clients while the violation is
caught," which the pytest files below confirm with real other
participants and real (non-mocked) crypto, same rigor as the A1 gate.

A4-false-claim and A5 are proof-forgery tests that the honest pipeline
has no code path to produce on its own (there is no "lie about my VRF
output" branch to inject into, and no delta/signature content to vary
either -- these attack the signature/proof algebra directly, independent
of any round's training content), so the isolated result above is
already the right, and complete, shape for those -- not a lesser
confirmation, a different one for a genuinely different threat. A7 is
correctly out-of-band (it targets `fl/trace.py`'s identity-opening
protocol, which runs on accumulated evidence between rounds, not inside
one) -- isolated testing against the real crypto is also already the
right shape for it.
"""),

("code", """from fl.models import build_model, flatten_state_dict
from fl.data_iid import prepare_case1
from fl.round_loop import (
    run_training_loop, bootstrap_norms_from_model_dim, release_gpu_memory,
    build_x25519_keys, precompute_pairwise_secrets,
)
from attacks.a1_scaling import craft_scaled_update

GATE_A1_CLIENTS = 9   # k_min(8) + 1, same reasoning as Gate E: quarantining/rejecting the attacker must still leave k_min eligible
GATE_A1_ROUNDS = 3
CHEATER_IDX = 5

gate_a1_model = build_model('tiny_cnn', num_classes=10, in_channels=1)
gate_a1_flat, _ = flatten_state_dict(gate_a1_model)
gate_a1_dim = gate_a1_flat.numel()
print("gate A1 model_dim:", gate_a1_dim)

gate_a1_data = prepare_case1(seed=CFG['seed'], source="torchvision")

gate_a1_sks = [Scalar.random() for _ in range(GATE_A1_CLIENTS)]
gate_a1_ring = [scalar_mul_base(sk) for sk in gate_a1_sks]
gate_a1_x25519_sks, gate_a1_x25519_pks = build_x25519_keys(GATE_A1_CLIENTS)
gate_a1_pairwise_secrets = precompute_pairwise_secrets(gate_a1_x25519_sks, gate_a1_x25519_pks)

# reuse the SAME committee (pk_trace) already established above -- it
# doesn't depend on which clients participate, only on the key holders.
gate_a1_root_norms = bootstrap_norms_from_model_dim(
    model_dim=gate_a1_dim,
    sketch_dim=CFG['sketch']['m'],
    sketch_seed=b"bootstrap-gate-a1",
    quant_bits=16,
    rng=np.random.default_rng(CFG['seed'] + 2),
)

# craft the malicious delta ONCE, the same way the standalone A1 test
# did: gamma*(w_target - w_t) against the model's actual initial weights,
# far enough from a random target that it should never pass V3 regardless
# of which round it's submitted in.
w_target_gate = np.random.default_rng(CFG['seed']).normal(0, 1.0, size=gate_a1_dim)
gate_a1_malicious_delta = craft_scaled_update(gate_a1_flat.numpy().astype(np.float64), w_target_gate)
print("crafted malicious delta, raw norm:", np.linalg.norm(gate_a1_malicious_delta))
"""),

("code", """final_w_t_a1, history_a1 = run_training_loop(
    model=gate_a1_model,
    model_dim=gate_a1_dim,
    client_datasets=gate_a1_data['client_train'][:GATE_A1_CLIENTS],
    ring=gate_a1_ring,
    client_sks=gate_a1_sks,
    pairwise_secrets=gate_a1_pairwise_secrets,
    pk_trace=pk_trace,
    server_root_norms=gate_a1_root_norms,
    CFG=CFG,
    rounds=GATE_A1_ROUNDS,
    cheater_client_idx=CHEATER_IDX,
    cheater_force_delta=gate_a1_malicious_delta,
)
release_gpu_memory()

print("\\nper-round summary:")
cheater_ever_rejected = False
cheater_ever_accepted = False
for h in history_a1:
    if h.get("aborted"):
        print(f"  round {h.get('round')}: ABORTED -- {h.get('reason')}")
        continue
    reasons = h.get("rejected_reasons", {})
    accepted_idxs = h.get("accepted_client_indices", [])
    cheater_rejected_this_round = CHEATER_IDX in reasons
    cheater_accepted_this_round = CHEATER_IDX in accepted_idxs
    cheater_ever_rejected |= cheater_rejected_this_round
    cheater_ever_accepted |= cheater_accepted_this_round
    print(f"  round {h.get('round')}: accepted={h.get('accepted')}, "
          f"cheater rejected={cheater_rejected_this_round} (reasons={reasons.get(CHEATER_IDX)}), "
          f"aggregate_exact={h.get('aggregate_exact')}")

if cheater_ever_accepted:
    print(f"\\nFAIL: cheater {CHEATER_IDX}'s scaled update was accepted at least once -- a real defense failure, report this.")
elif cheater_ever_rejected:
    print(f"\\nPASS: cheater {CHEATER_IDX}'s scaled update was rejected by V3 every round it was eligible, while the round still completed normally for honest clients.")
else:
    print(f"\\nINCONCLUSIVE: cheater {CHEATER_IDX} was never sampled/eligible in any of these {GATE_A1_ROUNDS} rounds -- re-run with more rounds or check sampling_rate.")
"""),

("md", """## Gate — A3 (Sybil) and A4-replay injected into a REAL round

Both run directly here (not just imported for inspection) so this
notebook's own execution IS the confirmation, same as the standalone
pytest files -- `tests/test_a3_sybil_real_round.py` and
`tests/test_a4_replay_real_round.py` -- which this cell mirrors exactly
(8-9 genuinely distinct honest clients, real CLSAG/Pedersen/sketch/range-
proof crypto, torch-free, built directly against `fl/server.py::run_round`
rather than `run_training_loop` -- see the markdown above for why).

**A3 (Sybil)**: a 9th real client (same signing key, same ring position,
same round nonce) submits TWICE in the same round with different crafted
content. The key image `I_t = sk*H_p(rho_t)` is identical across both --
V2 (`check_V2_no_double_submission`) must accept exactly one.

**A4-replay**: a 9th real client's submission is NOT fresh -- it is the
exact object that client genuinely produced for the PREVIOUS round,
resubmitted verbatim at this round's (different) `rho_t`. V1's ring-
signature verification, which recomputes `H_p(rho_t)` from whatever
`rho_t` is passed to it, must reject it.
"""),

("code", """import subprocess, sys as _sys

_result = subprocess.run(
    [_sys.executable, "-m", "pytest", "-v",
     "tests/test_a3_sybil_real_round.py", "tests/test_a4_replay_real_round.py"],
    cwd="/content/FL-ZKP", capture_output=True, text=True,
)
print(_result.stdout[-4000:])
if _result.returncode != 0:
    print(_result.stderr[-2000:])
    raise AssertionError(
        "A3/A4-replay real-round gate FAILED -- see pytest output above. "
        "This is a real defense failure or a harness bug, not a formatting issue."
    )
print("\\nPASS: A3's Sybil duplicate and A4's replayed submission were both "
      "rejected by the real protocol (V2 and V1 respectively), inside a real "
      "multi-client round, while every genuinely honest/fresh client was "
      "accepted normally -- same rigor as the A1 gate above.")
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
