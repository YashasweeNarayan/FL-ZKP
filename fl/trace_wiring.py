"""§6.1 evidence accumulation, wired for the REAL per-round pipeline
(fl/round_loop.py), as opposed to the standalone/demo context
fl/evidence.py's own docstring describes.

PURE-LOGIC / TESTABLE HERE, same discipline as fl/evidence.py: this
module only accumulates scalars (norm_only_evidence, already reduced
to its one cryptographically-verified term -- see that function's
docstring for why cos_dist and loss_delta are not real-pipeline-
computable) and tracks which pseudonyms have crossed a calibrated
threshold. It does NOT touch torch or the EC backend, and deliberately
does NOT call fl/trace.py's run_opening_protocol itself.

That split is intentional, not a missing piece: the committee's secret
key-holder shares (crypto/dkg.py's `finalize()` output) live only in
notebook scope (see notebooks/_build_mvp_run_notebooks.py's "trace
committee" cell) -- round_loop.py never receives them, and should not
need to, since actually decrypting an escrowed identity is a slower,
EC-heavy operation that only needs to happen for the handful of
pseudonyms this module flags, not every round for every client. This
module's job is narrower and cheaper: detect WHO crossed theta and
WHEN (which round), every round, from data round_loop.py already has
in hand. The notebook layer is expected to call fl/trace.py's
run_opening_protocol (and then revoke()) post-hoc for each pseudonym
reported via `on_pseudonym_flagged`, using the committee it already
built for escrow -- exactly the mechanism already validated standalone
in notebooks/traceability_demo (both against the local pure-Python
backend and, per FIX.md, the real rbcl/libsodium backend on Colab).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .evidence import EvidenceAccumulator, CalibrationCertificate, calibrate_theta_from_norms, norm_only_evidence


@dataclass
class EpochEvidenceState:
    """Owns one EvidenceAccumulator + its calibrated theta for the
    CURRENT epoch, and rolls over to a fresh one when the round loop
    crosses an epoch boundary (`t_round // epoch_rounds` changes --
    the same expression fl/round_loop.py already uses for rho_e).
    §6.4's committee reconfiguration is a separate timescale (roster,
    not evidence) and is intentionally untouched by this rollover.
    """
    epoch_rounds: int
    sampling_rate: float
    server_root_norms: "np.ndarray"
    alpha: float = 0.05
    n_permutations: int = 2000
    rng: "np.random.Generator | None" = None

    _current_epoch: int = field(default=-1, repr=False)
    _accumulator: "EvidenceAccumulator | None" = field(default=None, repr=False)
    _certificate: "CalibrationCertificate | None" = field(default=None, repr=False)
    _opened_pseudonyms: set = field(default_factory=set, repr=False)

    def _ensure_epoch(self, epoch: int, B_t: float) -> None:
        if epoch == self._current_epoch:
            return
        self._current_epoch = epoch
        self._accumulator = EvidenceAccumulator(epoch=epoch)
        self._certificate = calibrate_theta_from_norms(
            self.server_root_norms, B_t, self.epoch_rounds, self.sampling_rate,
            alpha=self.alpha, n_permutations=self.n_permutations, rng=self.rng,
        )
        self._opened_pseudonyms = set()

    def record_round(self, t_round: int, pseudonym_norm_pairs, B_t: float) -> list:
        """Call once per round with `pseudonym_norm_pairs`, an iterable
        of (pseudonym, norm_q) pairs -- one for EVERY eligible client
        this round, accepted or rejected alike. Theta is calibrated
        against the full honest population's distribution, so a
        client rejected this round (e.g. by V3) still needs to
        contribute its ev^(t) to the SAME running total; excluding
        rejected clients would silently stop tracking exactly the
        clients evidence accumulation exists to catch.

        Returns the pseudonyms newly flagged THIS round (crossed theta
        for the first time this epoch) -- an empty list is the normal,
        expected result for an uneventful round, not a sign nothing
        was recorded.
        """
        epoch = t_round // self.epoch_rounds
        self._ensure_epoch(epoch, B_t)
        for pseudonym, norm_q in pseudonym_norm_pairs:
            self._accumulator.record(pseudonym, norm_only_evidence(norm_q, B_t))
        newly_flagged = [
            p for p in self._accumulator.flagged(self._certificate.theta)
            if p not in self._opened_pseudonyms
        ]
        self._opened_pseudonyms.update(newly_flagged)
        return newly_flagged

    @property
    def theta(self) -> float:
        return self._certificate.theta if self._certificate is not None else float("inf")

    @property
    def certificate(self) -> "CalibrationCertificate | None":
        """The current epoch's CalibrationCertificate -- a caller
        acting on a flagged pseudonym (e.g. building fl/trace.py's
        OpeningRequest) needs this object itself, not just its theta.
        """
        return self._certificate

    @property
    def epoch(self) -> "int | None":
        return self._current_epoch if self._current_epoch >= 0 else None

    def total_for(self, pseudonym) -> float:
        return self._accumulator.total_for(pseudonym) if self._accumulator is not None else 0.0
