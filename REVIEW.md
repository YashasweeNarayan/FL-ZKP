# JANUS-FL — Design Review and Limitations

Adversarial review against `ARCHITECTURE.md` rev. 3. Rev. 1 of this document raised eighteen issues; this revision records which are closed, which remain, and what the paper must concede.

**Status: all ten original limitations addressed. Four residual limitations remain and are stated in §4 for the paper's limitations section.**

---

## 1. Resolution status

| ID | Issue | Status | Resolution |
|---|---|---|---|
| D1 | Evidence accumulator contradicted the anonymity design — per-round scopes make rounds unlinkable, so there was nothing to accumulate and the tracing trigger was vacuous | ✅ **closed** | dual-scope tags (§5.1): round key image + epoch pseudonym, `R` as the explicit dial |
| D2 | Server's aggregation was never verified, contradicting the malicious-server threat model | ✅ **closed** | check V5 (§4) — one group operation, exploits Pedersen homomorphism |
| D3 | A server-controlled key-holder roster is a total break of anonymity | ✅ **closed** | consent rule (§6.4): roster changes need `t = 7` outgoing-committee signatures; admin proposes, incumbents execute |
| R1 | **Network-layer de-anonymization defeats the ring signature entirely** | ✅ **closed** | anonymous channel **constructed** (§2.3): 3-hop onion cascade, fixed windows, padded payloads, dummy traffic. Residual: `{M1, M3}` collusion |
| R2 | Groth16's trusted setup holder can forge `R_cons` — i.e. mount exactly the framing attack the paper claims to prevent, making T5 false | ✅ **closed** | **PLONK** with the public perpetual Powers-of-Tau universal SRS. No per-circuit toxic waste |
| R3 | Server chooses `B_t`, giving it a censorship and de-anonymization oracle over non-IID clients | ✅ **closed** | `B_t` publicly derived from `BB` by the median rule, recomputable by anyone |
| R4 | "No privacy leak in the whole pipeline" not achievable — the released sum leaks over 120 rounds | ✅ **closed** | distributed DP (§6.5) with RDP accounting and subsampling amplification. Residual: bounded- not provably-Gaussian noise (L2) |
| R5 | Dropout recovery is an unmasking oracle — a malicious server falsely declares a client dropped | ✅ **closed** | double-masking in `secagg.py`, with the false-dropout test in Gate 1 |
| R6 | Anonymity set of only 20 (`4.32` bits) | ✅ **improved** | `n = 50` → `5.64` bits. Residual: still erodes with openings (L3) |
| R7 | Sybil resistance at enrollment assumed, not proved | ⚠️ **assumption, stated** | PKI enrollment assumption in §III; anonymity degrades gracefully in `n − |corrupt|` |
| R8 | Timing side channel — training time ∝ dataset size ⇒ identity | ✅ **closed** | fixed submission window, local training padded to a common deadline (§2.3) |
| R9 | No revocation after tracing — a named client keeps submitting | ✅ **closed** | revocation in §6.2 step 6; the resulting anonymity-set shrink is accounted for in §6.3 rather than ignored |
| R10 | Committee liveness — attackers on the committee simply refuse to produce partials | ✅ **mitigated** | `t = 7` of 10 tolerates 3 refusals; per-epoch resharing makes capture temporary. Residual: genesis committee (L4) |
| R11 | Fixed-point overflow in `Z_p` with many clients | ✅ **closed** | overflow bound as a Stage 1 unit test, re-checked at Gate 5 |
| R12 | BatchNorm is a non-IID failure mode **and** leaks local data statistics | ✅ **closed** | GroupNorm throughout; BN excluded entirely |
| R13 | ImageNet-pretrained backbones make MNIST vacuous | ✅ **closed** | trained from scratch; MNIST explicitly positioned as the sanity/overhead baseline |
| R14 | MNIST IID shows nothing interesting | ✅ **scoped** | kept for overhead and correctness; the scientific weight is on FEMNIST, with both seen- and unseen-writer metrics |
| — | **Split hygiene** (raised by you) | ✅ **closed** | `ROADMAP.md` §3.4: every role disjoint, **writer-level** disjointness for FEMNIST, server root carved from train so test stays pristine, disjointness asserted in CI |

---

## 2. Does it hold together end to end?

Audited adjective by adjective, since that is how a reviewer reads the title.

| Claim | Verdict | Basis |
|---|---|---|
| **Anonymously** | ✅ **holds** | CLSAG over a 50-ring (T2) + constructed anonymous channel (§2.3) + VRF self-sampling so participation itself is hidden (T11) + fixed windows and padding. The three channels that leaked in rev. 1 — network address, timing, participation — are each closed by a named mechanism. |
| **Authenticated** | ✅ holds | ring membership, scoped linkability, `R_cons` binding (T3, T4) |
| **Verified** | ✅ holds | client inputs (V1–V4) and server aggregation (V5) both independently checkable; transparent setup so T5 is sound |
| **Privacy-preserving** | ✅ holds, with a stated noise assumption | per-update confidentiality (T1) + `(ε, δ)`-DP on the release (T12), amplified by subsampling |
| **Traceable** | ✅ holds | threshold opening, non-frameable, publicly auditable, rotatable roster, revocation (T7–T10) |
| **Key sharding + ZKP between client and server** | ✅ holds | Shamir `(7, 10)` over the selected key holders, never reconstructed; five ZK relations on the client→server path |

**Answer to your question: yes, it is now end to end.** Each of the five properties is delivered by a named mechanism, carries a theorem, and has an ablation that shows what breaks without it. That was not true of rev. 1 — anonymity leaked outside the cryptography and the privacy claim exceeded what was proved.

The honest qualifier: "end to end" means *every property has a mechanism and a proof*, not *every property is unconditional*. Four conditions remain, in §4. All four are of the kind that reviewers accept when stated plainly and reject when discovered.

---

## 3. Remaining execution risks

These are about shipping, not design.

**E1 — Thirteen cryptographic subsystems.** P1–P13 plus two circuits plus seven adversaries. The realistic failure mode for one person in 3.5 months is that eleven work, two are stubs, and the paper claims thirteen. Tier the work now:

| Tier | Components | Supports |
|---|---|---|
| **Core — must ship** | Pedersen, CLSAG dual-scope, threshold ElGamal, DKG, secagg + double-masking, Bulletproofs norm, `R_cons`, `R_vrf`, V5, public `B_t`, committee consent rule, attacks A1/A3/A4/A5 | N1, N6, N8, N10, N11 — a complete paper on its own |
| **Strong — should ship** | mixnet cascade, evidence calibration, sketched proofs, DP, A2/A6/A7 | N2, N4, N9, and the DP claim |
| **Stretch — cut first** | proactive resharing, mobile-adversary refresh, `H(T)` composition theorem | N3, N5 |

Note the asymmetry inside §6.4: the **consent rule** is Core (it is a signature check, and the anonymity claim depends on it), while the **resharing protocol** is Stretch (it is the expensive part, and without it the roster is simply static). Ship the rule; defer the rotation and say the roster is fixed per deployment.

**E2 — The Bulletproofs Rust build on Colab is the critical path.** It is the one dependency that can simply refuse to build. Gate 1 exists to find out in week 1 rather than week 6. Fallback: PLONK range proofs at reduced `m`.

**E3 — `R_cons` now carries five relations including a VRF.** Constraint count may overrun the 2×10⁵ budget. Mitigation already identified: Merkle path for ring membership instead of a 50-way comparison, which at `n = 50` is probably required anyway.

**E4 — Colab session limits against ~132 GPU-hr.** Checkpoint every round; apply cost controls 1 and 3 from `ROADMAP.md` §9 immediately to reach ~90 GPU-hr.

---

## 4. Limitations section for the paper

Four residual limitations plus five standard scope statements. Write these yourself rather than letting a reviewer write them for you — a paper that names its limits precisely reads as stronger.

**L1 — Mix-cascade collusion.** Network-layer anonymity holds unless the first and last hop of the 3-node cascade collude. This is the standard assumption of every deployed mixnet and is far weaker than assuming an anonymous channel outright, but it is an assumption. Path rotation by public randomness means the adversary must corrupt a path it cannot predict; it does not make corruption impossible.

**L2 — Bounded, not provably-Gaussian, DP noise.** The framework proves in zero knowledge that each client's noise share is *bounded* (`‖ν_k‖ ≤ B_ν`), not that it was *sampled from the correct distribution*. A client could contribute less noise than specified while remaining inside the bound, weakening the effective `ε`. The `(ε, δ)` guarantee therefore assumes honest noise contribution by a majority of the sampled cohort. Proving distributional correctness in ZK is open; it is the clearest future-work item.

**L3 — Anonymity set of 50, eroding.** `H(0) ≈ 5.64` bits, decreasing with every opening and every revocation. §6.3 quantifies the decay and makes it a tunable frontier; it does not prevent it. A deployment that traces often ends up with weak anonymity, and that tension is intrinsic rather than an artifact of this construction.

**L4 — The genesis committee is trusted.** If `t = 7` of the initial 10 key holders are adversarial from the start, tracing is compromised from genesis and no later reconfiguration recovers it — the compromised set simply refuses consent to any roster change. Bootstrapping a trustworthy initial committee is out of scope.

**Scope statements (not defects, but state them):**

**L5 — Norm bounds are not a complete Byzantine defense.** RoFL's analysis shows bounds are of limited effectiveness against attacks exploiting model memorization. JANUS-FL eliminates the scaling class at the gate and raises the cost of the rest via calibrated tracing; it does not claim general Byzantine-robustness.

**L6 — JL soundness gap.** Sketched `L_2` proofs admit failure probability `δ_JL`, tabulated per `m`. `L_∞` is not JL-preserved and is reported only on a top-`k` sparsified update.

**L7 — Static corruption, random oracle model.** No adaptive-corruption security. Consistent with the accountable-ring-signature literature, weaker than a standard-model result.

**L8 — Cross-silo scale.** `n = 50` is a consortium setting. Nothing here is evaluated at cross-device scale (`n = 10⁴+`), where pairwise mask exchange, ring size and mixnet batching all change character.

**L9 — Simulation, not deployment.** Single-machine simulation of 50 clients and a 3-node cascade. No real network, no heterogeneous hardware, no stragglers beyond simulated dropout. Latency figures are therefore lower bounds on a real deployment.

**L10 — Tracing liveness depends on the server.** The co-signature gate that protects against a key-holder cabal also means a server refusing to co-sign can stall accountability. `BB` records the refusal, which is a deterrent rather than a fix.

---

## 5. What I'd do next

1. **Stage 1**, in the order the table lists it — `committee.py`'s server-only-rejection test and `mixnet.py`'s unlinkability test first, since those two underwrite the anonymity claim.
2. **Validate the Bulletproofs build on Colab** before anything depends on it.
3. **Write `splits.py` with the disjointness assertions before any training code exists**, so no run can silently leak a writer across roles.
4. Then Stages 2–3 in parallel with your first Colab validation round.
