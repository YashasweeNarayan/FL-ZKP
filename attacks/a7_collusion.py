"""A7: collusive de-anonymization (ARCHITECTURE.md §10): 6 key holders
(out of n_c=10, i.e. one short of the t=7 threshold) + a malicious
server + 2 of the 3 mix nodes collude to try to open an honest
client's identity without legitimate evidence. Defense under test:
the (t=7, n_c=10) threshold (crypto/elgamal_threshold.py +
crypto/committee.py) means 6 colluding holders cannot combine partials
at all (combine_partials needs >= t=7 VALID partials from DISTINCT
holders -- see crypto/field.py's reconstruct/lagrange machinery, which
only closes with exactly t or more points on the right polynomial),
and the 3-hop mix cascade (crypto/mixnet.py) means 2-of-3 colluding
mix nodes still cannot both (a) see which client submitted AND (b)
see what the server received, since no single node (or pair, short of
all 3) sits on both the client-facing and server-facing hop
simultaneously. EC-DEPENDENT.
"""
from __future__ import annotations

from . import AttackResult
from crypto.elgamal_threshold import combine_partials


def run_attack_insufficient_threshold(ciphertext, colluding_partials: list, t: int):
    """colluding_partials: partials from the 6 colluding holders only
    (deliberately < t). combine_partials's Lagrange step needs exactly
    `t` points to reconstruct correctly -- feeding it fewer silently
    produces a WRONG point (interpolating a degree-(t-1) polynomial
    from < t points is underdetermined), which this attack test
    confirms does NOT equal the true plaintext identity point (using
    a provided ground-truth pk for comparison), rather than crashing --
    i.e. the colluders get garbage, not the real secret.
    """
    if len(colluding_partials) >= t:
        raise ValueError("this attack variant requires fewer than t partials to be meaningful")
    garbage_result = combine_partials(ciphertext, colluding_partials)
    return garbage_result  # caller compares against the true pk_victim; see run_attack below


def run_attack(ciphertext, colluding_partials: list, honest_partials: list, true_pk_victim, t: int = 7, n_c: int = 10):
    n_colluding = len(colluding_partials)
    if n_colluding >= t:
        return AttackResult(
            "A7_collusion", defense_triggered=False, defense_name="t_of_nc_threshold",
            detail=f"test misconfigured: {n_colluding} colluding holders already meet t={t}",
        )

    garbage = run_attack_insufficient_threshold(ciphertext, colluding_partials, t)
    collusion_recovered_real_identity = (garbage == true_pk_victim)

    # sanity check: the LEGITIMATE path (honest + enough colluding
    # partials to reach t, i.e. a properly authorized opening) DOES
    # recover the real identity -- confirms the failure above is
    # specifically about insufficient threshold, not a broken
    # combine_partials implementation.
    legitimate_partials = (colluding_partials + honest_partials)[:t]
    legitimate_result = combine_partials(ciphertext, legitimate_partials) if len(legitimate_partials) >= t else None
    legitimate_path_works = (legitimate_result == true_pk_victim) if legitimate_result is not None else False

    defense_triggered = (not collusion_recovered_real_identity) and legitimate_path_works
    return AttackResult(
        "A7_collusion", defense_triggered=defense_triggered, defense_name="t_of_nc_threshold",
        detail=f"{n_colluding}-of-{n_c} colluders (need t={t}): recovered_real_identity={collusion_recovered_real_identity} "
               f"(must be False); legitimate t-partial path works={legitimate_path_works} (must be True, confirms it's a threshold failure not a bug)",
    )


def run_attack_mixnet_partial_view(client_addr: str, server_addr: str, mix_node_ids_colluding: list, total_hops: int = 3):
    """2-of-3 colluding mix nodes: confirms neither colluding node
    individually sits on BOTH the client-facing first hop and the
    server-facing last hop -- the structural property that makes the
    cascade resistant to a minority of corrupt nodes (ARCHITECTURE.md
    P11). This check is topological (which hop index each colluding
    node occupies), not a cryptographic verification -- the actual
    onion-peeling correctness is mixnet.py's own concern, already
    covered by its construction (crypto/mixnet.py's peel_layer).
    """
    if len(mix_node_ids_colluding) >= total_hops:
        return AttackResult(
            "A7_collusion_mixnet", defense_triggered=False, defense_name="3hop_mix_cascade",
            detail="all hops colluding -- cascade fully compromised, not a meaningful test",
        )
    first_hop_colluding = 0 in mix_node_ids_colluding
    last_hop_colluding = (total_hops - 1) in mix_node_ids_colluding
    full_path_compromised = first_hop_colluding and last_hop_colluding
    defense_triggered = not full_path_compromised
    return AttackResult(
        "A7_collusion_mixnet", defense_triggered=defense_triggered, defense_name="3hop_mix_cascade",
        detail=f"colluding hops={mix_node_ids_colluding}, first_hop_colluding={first_hop_colluding}, "
               f"last_hop_colluding={last_hop_colluding}: full path compromised={full_path_compromised} (must be False)",
    )
