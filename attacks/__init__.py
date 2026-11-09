"""Attack suite -- MVP scope cut to 5 of ARCHITECTURE.md's 7 attacks
(ROADMAP.md §0.1: "reduced attack grid: 5 attacks x 2 fractions"):

    A1  a1_scaling.py     -- model-replacement/scaling, caught by V3 (R_norm)
    A3  a3_sybil.py        -- multi-submission under anonymity, caught by V2 (key image)
    A4  a4_freeride.py     -- free-riding / false-dropout claim, caught by R_cons's VRF disjunction
    A5  a5_splice.py       -- identity-laundering / transcript splicing, caught by V4 (R_cons)
    A7  a7_collusion.py    -- collusive de-anonymization, defeated by (t,n_c) threshold + mix cascade

A2 (label-flipping poisoning, needs many real training rounds for
evidence to accumulate) and A6 (gradient inversion, needs a real DLG/
iDLG optimization against actual gradients) are deferred past the
3-day MVP -- both are compute-heavy and orthogonal to the crypto-
correctness claims the other 5 attacks exercise directly. See
ROADMAP.md §0.1 and STATUS.md.

Each attack module runs EC-DEPENDENT crypto (ristretto/clsag/vrf/
elgamal) and is therefore NOT executed in this sandbox -- same
constraint as crypto/*.py's EC-dependent tier. Validate on Colab.
Each module exposes `run_attack(...)` returning an AttackResult with
an explicit `defense_triggered` field so the Colab notebook can
tabulate attack x defense x (fraction of attackers) into the
"single most convincing figure in the paper" (ARCHITECTURE.md §10).
"""
from dataclasses import dataclass


@dataclass
class AttackResult:
    attack_name: str
    defense_triggered: bool
    defense_name: str
    detail: str
