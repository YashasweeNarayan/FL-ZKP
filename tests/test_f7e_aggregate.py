"""FIX.md F7e/F7f -- the cross-client aggregate sketch check and its
group-testing blame step, end to end through fl/server.py's helpers.

Builds a small multi-client round by hand (no torch: uses the NumPy
`SketchProjector` on BOTH sides, which is legitimate here precisely
because `_true_aggregate_sketch` takes the projector as an argument --
a real round uses the torch one on both sides instead, see that
function's docstring on why the two must not be mixed).

Runs under the test-only pure-Python ristretto255 backend; skips
cleanly without it.
"""
from __future__ import annotations

import hashlib
import itertools
import os

import numpy as np
import pytest

# Gate on whether ANY group backend is live, not on which one. With rbcl
# installed (Colab) these run against real libsodium; in a network-less
# sandbox they run against the pure-Python backend when
# JANUS_ALLOW_PURE_PYTHON_GROUP=1 selects it; with neither they skip.
#
# Gating on the env var alone -- as the first version of these files did --
# meant the whole F7 suite SKIPPED on Colab, which is the one place it most
# needs to run: the pure-Python backend proves the algebra, only rbcl proves
# the bindings.
from crypto.ristretto import backend_name as _backend_name

if _backend_name() is None:
    pytest.skip(
        "no ristretto255 backend available: install rbcl, or set "
        "JANUS_ALLOW_PURE_PYTHON_GROUP=1 to use the test-only pure-Python "
        "backend (crypto/_ristretto_py.py)",
        allow_module_level=True,
    )

from types import SimpleNamespace

from crypto import norm_proof as NP
from crypto.secagg import (
    P_MASK, aggregate_masked, pairwise_mask, pairwise_round_seed, self_mask,
    self_secret_bytes, new_self_mask_secret, shamir_share_self_secret, unmask_aggregate,
)
from crypto.sketch import SketchProjector
from fl.client import center_mod_p_mask
from fl.server import _check_F7e_aggregate_sketch, _true_aggregate_sketch

N_CLIENTS = 6
MODEL_DIM = 64
SKETCH_M = 8
SHARE_T = 3
RHO_T = b"rho_t_round0"
SIGMA_T = b"sigma_t_round0"


def _exact_sketch_by_hand(projector, q) -> list[int]:
    """S*q with unbounded Python ints, using the projector's OWN sign
    blocks so the matrix is identical to `project_int`'s.

    Needed because `project_int_batch` REFUSES inputs this large (its
    float64-exactness guard raises OverflowError at
    INT_SKETCH_BLOCK * max|q| >= 2^53). That guard is a real piece of
    defense-in-depth -- an honest client literally cannot compute a
    P_MASK-wrapped sketch through this repo's own API -- but it does
    not constrain an ATTACKER, who writes their own arithmetic. So the
    cheating client below is simulated the way a real one would behave,
    not the way our helper would permit.
    """
    from crypto.sketch import INT_SKETCH_BLOCK
    q = np.asarray(q, dtype=object)
    out = [0] * projector.m
    for start in range(0, projector.d, INT_SKETCH_BLOCK):
        end = min(start + INT_SKETCH_BLOCK, projector.d)
        signs = projector._int_signs(start, end - start).astype(np.int64)
        block = q[start:end]
        for i in range(projector.m):
            out[i] += int(sum(int(s) * int(v) for s, v in zip(signs[i], block) if s))
    return out


def _build_round(cheater: int | None = None, cheat_scale: int = 1, cheat: str = "wraparound"):
    """One synthetic round. Every client picks a small signed update,
    masks it the way `round_step_phase1` does, and commits the exact
    integer sketch of its own centered update the way `round_step_phase2`
    does.

    `cheater` commits the sketch of a DIFFERENT vector than the one it
    masked. Two flavours, because they are caught by different checks:

    - "wraparound": the real update with `cheat_scale * P_MASK` added to
      one coordinate. Same value mod P_MASK, so unmasking and V5 are
      unaffected. Its sketch values are astronomically large, so V3's
      own norm bound and F7d's window test ALSO reject it -- F7e is
      defense-in-depth here, not the only line.
    - "lying_sketch": a large malicious update is masked, but the sketch
      committed is of a small, innocuous vector (here: the zero vector).
      This is the cheat F7e UNIQUELY catches -- precisely FIX.md F7's
      original gap (b), "a client can send the sketch of a zero vector
      with an arbitrary update". V3 sees a small norm and is satisfied;
      F7d's flooded sums are all inside the window; V5 passes because
      the masked vector and its commitment are mutually consistent.
      Only comparing sum_k y_k against S(sum_k q_k) exposes it.
    """
    rng = np.random.default_rng(7)
    projector = SketchProjector(seed=SIGMA_T, m=SKETCH_M, d=MODEL_DIM)

    secrets = {}
    for a, b in itertools.combinations(range(N_CLIENTS), 2):
        s = hashlib.sha256(f"pair-{a}-{b}".encode()).digest()
        secrets.setdefault(a, {})[b] = s
        secrets.setdefault(b, {})[a] = s

    q_true = {k: rng.integers(-500, 500, size=MODEL_DIM).astype(np.int64) for k in range(N_CLIENTS)}
    if cheater is not None and cheat == "lying_sketch":
        # A model-replacement-scale update (A1): orders of magnitude
        # larger than any honest client's, and the whole point is that
        # it goes into the MASKED vector while the committed sketch
        # claims something innocuous.
        q_true[cheater] = np.full(MODEL_DIM, 10 ** 6, dtype=np.int64)
    self_secrets = {k: new_self_mask_secret() for k in range(N_CLIENTS)}
    shares_held = {
        k: shamir_share_self_secret(self_secrets[k], SHARE_T, list(range(1, N_CLIENTS + 1)))
        for k in range(N_CLIENTS)
    }

    masked_vectors, phase2s = {}, {}
    for k in range(N_CLIENTS):
        x = q_true[k] % P_MASK
        for j in range(N_CLIENTS):
            if j == k:
                continue
            sign = 1 if k < j else -1
            x = (x + pairwise_mask(secrets[k][j], RHO_T, MODEL_DIM, sign)) % P_MASK
        x = (x + self_mask(self_secret_bytes(self_secrets[k]), RHO_T, MODEL_DIM)) % P_MASK
        masked_vectors[k] = x

        if cheater is not None and k == cheater and cheat == "wraparound":
            sketched_from = q_true[k].astype(object).copy()
            sketched_from[0] += cheat_scale * P_MASK  # same value mod P_MASK, different integer
            y = _exact_sketch_by_hand(projector, sketched_from)
        elif cheater is not None and k == cheater and cheat == "lying_sketch":
            # commits the sketch of the ZERO vector while masking a huge update
            y = [int(v) for v in projector.project_int(np.zeros(MODEL_DIM, dtype=np.int64))]
        else:
            y = [int(v) for v in projector.project_int(q_true[k])]
        sc, y_bl, _ = NP.commit_sketch(y, f"client{k}".encode())
        seeds = {j: pairwise_round_seed(secrets[k][j], RHO_T) for j in range(N_CLIENTS) if j != k}
        phase2s[k] = SimpleNamespace(
            sketch_commitment=sc,
            masked_y_blindings=NP.mask_blinding_vector(y_bl, k, seeds),
        )

    return SimpleNamespace(
        projector=projector, secrets=secrets, q_true=q_true, masked_vectors=masked_vectors,
        shares_held=shares_held, phase2s=phase2s,
    )


def _unmask_all(env, accepted):
    masked_sum = aggregate_masked(env.masked_vectors[k] for k in accepted)
    shares = {k: [env.shares_held[k][h + 1] for h in accepted] for k in accepted}
    excluded = [k for k in range(N_CLIENTS) if k not in set(accepted)]
    seeds = {}
    for j in accepted:
        for k in excluded:
            seeds[(j, k)] = pairwise_round_seed(env.secrets[j][k], RHO_T)
    return unmask_aggregate(masked_sum, accepted, shares, SHARE_T, seeds, RHO_T, MODEL_DIM), seeds


def _blame_provider(env, accepted):
    def provider(group):
        group = list(group)
        if len(group) < SHARE_T:
            return None
        outside = [k for k in range(N_CLIENTS) if k not in set(group)]
        straddling = {}
        for j in group:
            for k in outside:
                straddling[(j, k)] = pairwise_round_seed(env.secrets[j][k], RHO_T)
        partial_masked = aggregate_masked(env.masked_vectors[k] for k in group)
        shares = {k: [env.shares_held[k][h + 1] for h in group] for k in group}
        try:
            partial_q = unmask_aggregate(partial_masked, group, shares, SHARE_T, straddling, RHO_T, MODEL_DIM)
        except ValueError:
            return None
        return partial_q, straddling
    return provider


# --------------------------------------------------------------------
# the mechanism itself
# --------------------------------------------------------------------

def test_unmasked_aggregate_centers_to_the_sum_of_centered_updates():
    """The assumption `_true_aggregate_sketch` rests on: for honest,
    small updates, center(sum q_k mod P_MASK) == sum center(q_k)."""
    env = _build_round()
    accepted = list(range(N_CLIENTS))
    agg, _ = _unmask_all(env, accepted)
    centered = center_mod_p_mask(agg)
    expected = sum(env.q_true[k] for k in accepted)
    assert np.array_equal(centered, expected)


def test_true_aggregate_sketch_equals_sum_of_client_sketches():
    env = _build_round()
    accepted = list(range(N_CLIENTS))
    agg, _ = _unmask_all(env, accepted)
    server_side = _true_aggregate_sketch(agg, SKETCH_M, MODEL_DIM, SIGMA_T, env.projector)
    client_side = [0] * SKETCH_M
    for k in accepted:
        y = env.projector.project_int(env.q_true[k])
        client_side = [a + int(b) for a, b in zip(client_side, y)]
    assert server_side == client_side


# --------------------------------------------------------------------
# F7e
# --------------------------------------------------------------------

def test_f7e_passes_for_an_honest_round():
    env = _build_round()
    accepted = list(range(N_CLIENTS))
    agg, excluded_seeds = _unmask_all(env, accepted)
    ok, bad, groups, detail = _check_F7e_aggregate_sketch(
        env.phase2s, accepted, agg, excluded_seeds, SKETCH_M, MODEL_DIM, SIGMA_T, env.projector,
    )
    assert ok is True, detail
    assert bad == [] and groups == []


def test_f7e_passes_with_an_excluded_client():
    """Pairwise masks straddling the accepted/excluded boundary must be
    stripped by the same released seeds the vector path uses."""
    env = _build_round()
    accepted = [0, 1, 2, 4]
    agg, excluded_seeds = _unmask_all(env, accepted)
    ok, bad, _, detail = _check_F7e_aggregate_sketch(
        env.phase2s, accepted, agg, excluded_seeds, SKETCH_M, MODEL_DIM, SIGMA_T, env.projector,
    )
    assert ok is True, detail
    assert bad == []


def test_f7e_catches_a_wraparound_cheater():
    """The cheat V3 alone cannot see: commit to the sketch of a
    P_MASK-shifted representative, mask the real one. Same value mod
    P_MASK, so unmasking is unaffected and V5 still passes."""
    env = _build_round(cheater=3)
    accepted = list(range(N_CLIENTS))
    agg, excluded_seeds = _unmask_all(env, accepted)
    ok, bad, _, detail = _check_F7e_aggregate_sketch(
        env.phase2s, accepted, agg, excluded_seeds, SKETCH_M, MODEL_DIM, SIGMA_T, env.projector,
    )
    assert ok is False, detail
    assert len(bad) >= 1


def test_f7e_reports_none_when_it_cannot_run():
    """A submission with no masked blinding vector is a liveness gap,
    not evidence of cheating -- must be None, never False."""
    env = _build_round()
    accepted = list(range(N_CLIENTS))
    agg, excluded_seeds = _unmask_all(env, accepted)
    env.phase2s[2].masked_y_blindings = None
    ok, bad, groups, detail = _check_F7e_aggregate_sketch(
        env.phase2s, accepted, agg, excluded_seeds, SKETCH_M, MODEL_DIM, SIGMA_T, env.projector,
    )
    assert ok is None
    assert "no masked blinding vector" in detail


def test_f7e_reports_none_on_incomplete_released_seeds():
    env = _build_round()
    accepted = [0, 1, 2, 4]
    agg, excluded_seeds = _unmask_all(env, accepted)
    broken = {k: v for k, v in excluded_seeds.items() if k != (0, 3)}
    ok, _, _, detail = _check_F7e_aggregate_sketch(
        env.phase2s, accepted, agg, broken, SKETCH_M, MODEL_DIM, SIGMA_T, env.projector,
    )
    # A missing seed leaves an uncancelled mask: the sum is wrong, which
    # surfaces as a mismatch rather than an exception. Either way it must
    # not silently pass.
    assert ok is not True


# --------------------------------------------------------------------
# F7f
# --------------------------------------------------------------------

def test_f7f_not_localized_without_a_provider():
    env = _build_round(cheater=3)
    accepted = list(range(N_CLIENTS))
    agg, excluded_seeds = _unmask_all(env, accepted)
    ok, bad, groups, detail = _check_F7e_aggregate_sketch(
        env.phase2s, accepted, agg, excluded_seeds, SKETCH_M, MODEL_DIM, SIGMA_T, env.projector,
        blame_material_provider=None,
    )
    assert ok is False
    assert groups == []
    assert "not localized" in detail


def test_f7f_localizes_the_cheater_to_a_small_group():
    env = _build_round(cheater=3)
    accepted = list(range(N_CLIENTS))
    agg, excluded_seeds = _unmask_all(env, accepted)
    ok, bad, groups, detail = _check_F7e_aggregate_sketch(
        env.phase2s, accepted, agg, excluded_seeds, SKETCH_M, MODEL_DIM, SIGMA_T, env.projector,
        blame_material_provider=_blame_provider(env, accepted),
    )
    assert ok is False, detail
    assert groups, f"expected at least one suspect group, got none: {detail}"
    flat = set()
    for g in groups:
        flat |= set(g)
    assert 3 in flat, f"cheater 3 must be inside a suspect group, got {groups}"
    assert len(flat) < N_CLIENTS, f"localization must narrow the set, got {groups}"


def test_f7f_subgroup_check_clears_an_honest_subgroup():
    """The bisection's own predicate: a subgroup with no cheater in it
    must verify, or the search would implicate everyone."""
    env = _build_round(cheater=3)
    accepted = list(range(N_CLIENTS))
    provider = _blame_provider(env, accepted)
    honest = [0, 1, 2]
    partial_q, straddling = provider(honest)
    sub_blindings = NP.reconstruct_blinding_sum(
        {k: env.phase2s[k].masked_y_blindings for k in honest}, honest, straddling, SKETCH_M,
    )
    sub_true = _true_aggregate_sketch(partial_q, SKETCH_M, MODEL_DIM, SIGMA_T, env.projector)
    ok, bad = NP.verify_aggregate_sketch(
        [env.phase2s[k].sketch_commitment.y_commitments for k in honest], sub_blindings, sub_true,
    )
    assert ok, f"honest subgroup should verify, mismatched at {bad}"


def test_f7f_subgroup_check_flags_a_subgroup_holding_the_cheater():
    env = _build_round(cheater=3)
    accepted = list(range(N_CLIENTS))
    provider = _blame_provider(env, accepted)
    guilty = [3, 4, 5]
    partial_q, straddling = provider(guilty)
    sub_blindings = NP.reconstruct_blinding_sum(
        {k: env.phase2s[k].masked_y_blindings for k in guilty}, guilty, straddling, SKETCH_M,
    )
    sub_true = _true_aggregate_sketch(partial_q, SKETCH_M, MODEL_DIM, SIGMA_T, env.projector)
    ok, _ = NP.verify_aggregate_sketch(
        [env.phase2s[k].sketch_commitment.y_commitments for k in guilty], sub_blindings, sub_true,
    )
    assert not ok, "subgroup containing the cheater must fail"


# --------------------------------------------------------------------
# The cheat F7e uniquely catches: an honest-LOOKING sketch committed
# alongside a model-replacement-scale masked update (FIX.md F7 gap (b)).
# --------------------------------------------------------------------

def test_lying_sketch_passes_the_per_client_norm_check():
    """Establishes that F7e is not redundant: the cheater's own V3
    inputs look BETTER than an honest client's, because it committed the
    sketch of the zero vector."""
    from crypto.sketch import int_sketch_norm_sq_bound
    env = _build_round(cheater=3, cheat="lying_sketch")
    honest_v = env.phase2s[0].sketch_commitment.v
    cheat_v = env.phase2s[3].sketch_commitment.v
    assert cheat_v == 0, f"zero-vector sketch should have v=0, got {cheat_v}"
    assert cheat_v <= honest_v
    # and it is comfortably inside any bound an honest client satisfies
    bound = int_sketch_norm_sq_bound(1e6, SKETCH_M, 1.0 / 3.0)
    assert cheat_v <= bound and honest_v <= bound


def test_f7e_catches_the_lying_sketch():
    env = _build_round(cheater=3, cheat="lying_sketch")
    accepted = list(range(N_CLIENTS))
    agg, excluded_seeds = _unmask_all(env, accepted)
    ok, bad, _, detail = _check_F7e_aggregate_sketch(
        env.phase2s, accepted, agg, excluded_seeds, SKETCH_M, MODEL_DIM, SIGMA_T, env.projector,
    )
    assert ok is False, f"F7e must catch a sketch that does not match the masked update: {detail}"
    assert len(bad) >= 1


def test_f7f_localizes_the_lying_sketch_cheater():
    env = _build_round(cheater=3, cheat="lying_sketch")
    accepted = list(range(N_CLIENTS))
    agg, excluded_seeds = _unmask_all(env, accepted)
    ok, bad, groups, detail = _check_F7e_aggregate_sketch(
        env.phase2s, accepted, agg, excluded_seeds, SKETCH_M, MODEL_DIM, SIGMA_T, env.projector,
        blame_material_provider=_blame_provider(env, accepted),
    )
    assert ok is False, detail
    flat = set()
    for g in groups:
        flat |= set(g)
    assert 3 in flat, f"cheater 3 must land in a suspect group, got {groups} ({detail})"
    assert len(flat) < N_CLIENTS, f"must narrow the set, got {groups}"
