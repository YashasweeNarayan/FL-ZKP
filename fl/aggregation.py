"""Aggregate scaling for FedAvg under VRF self-sampling (FIX.md F20).

Pure arithmetic, no crypto imports, so the scale factor that every
round's model update depends on can be unit-tested directly rather than
only through a full end-to-end round against a group backend.
"""

from __future__ import annotations


def real_contributor_count(n_accepted: int, sampling_rate: float) -> float:
    """Expected number of REAL (non-dummy) contributors in the accepted set.

    Why this is not simply `n_accepted`
    -----------------------------------
    Under VRF self-sampling (ARCHITECTURE.md §5.5) only ~p*n clients
    actually train each round. The other ~(1-p)*n submit q_k = 0 as
    privacy padding, so that the server cannot tell who trained --
    participation is itself a linkability signal, and hiding it is the
    point of the dummies. A zero vector trivially satisfies the norm
    bound, so every dummy lands in `accepted`.

    The aggregate sum is therefore over real contributors only (dummies
    add exactly zero), while `len(accepted)` counts real + dummy. Using
    it as the FedAvg denominator under-scales every update by roughly
    1/p -- at p=0.25 with ~43 accepted, a factor of ~3.4. That is a
    silent, systematic shrink of the effective server learning rate,
    and `fl/server.py::apply_update`'s own docstring already specified
    the intended semantic as `/|A_real|`.

    Why p * n_accepted rather than p * n_clients
    --------------------------------------------
    Rejected clients are excluded from the sum, so the denominator has
    to track them. Scaling the ACCEPTED count by p does this correctly
    in every regime we care about:

      * p=1.0, no dummies      -> p*n_accepted = n_accepted
        (exactly the pre-F20 behaviour, which was correct in that case;
        this is why the default below is 1.0 and why existing callers
        that omit the argument are unaffected)
      * p=0.25, 50 accepted    -> 12.5, the expected sampled cohort
      * p=0.25, 43 accepted
        (7 attackers rejected)  -> 10.75

    That last case is the one a naive `p*n_clients - n_rejected` gets
    wrong. Attack submissions bypass the sampling gate entirely
    (`fl/client.py`'s `_force_delta` is applied AFTER the b_k branch),
    so only ~p of the rejected attackers displaced a genuinely-sampled
    honest contributor. Subtracting all 7 would over-correct to 5.5 and
    roughly double the applied step; scaling the accepted count instead
    lands on 10.75. It also cannot go negative, which
    `p*n_clients - n_rejected` does as soon as rejections exceed the
    expected cohort (observed live: a round with 15 rejections).

    Privacy
    -------
    Uses only public quantities -- the sampling rate p and the number of
    accepted submissions. It reveals nothing about any individual b_k,
    so the anonymity property the dummies exist to protect is preserved.

    Known limitation
    ----------------
    This is an expectation, not the realized count: |S| ~ Binomial(n, p)
    has std sqrt(n*p*(1-p)) ≈ 3.06 at n=50, p=0.25, so the realized
    cohort varies around 12.5 and the effective step inherits that
    jitter. The principled fix is to have each client commit to b_k and
    open only the homomorphic SUM of those commitments, which yields the
    exact count while revealing no individual bit -- the Pedersen
    commitments already in the protocol are additively homomorphic, so
    this is a natural extension rather than new machinery. Not done
    here: it is a protocol change, and the expectation is unbiased.
    """
    if not 0.0 < sampling_rate <= 1.0:
        raise ValueError(f"sampling_rate must be in (0, 1], got {sampling_rate!r}")
    return max(sampling_rate * max(n_accepted, 0), 1.0)
