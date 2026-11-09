# circuits/ — camera-ready PLONK upgrade path (NOT built for the 3-day MVP)

## Status: documented, NOT compiled, NOT tested, NOT used by the MVP pipeline

This directory is the camera-ready upgrade path referenced throughout
`crypto/schnorr_nizk.py`, `crypto/rangeproof.py`, and `crypto/r_cons.py`.
Those modules implement the **same relations** (R_cons, R_norm /
"share well-formedness") as compound Sigma-protocol NIZKs, which are
real zero-knowledge proofs but scale as O(n) in the number of witness
bits/scopes rather than O(log n). The `.circom` sources below describe
the PLONK-equivalent circuit encoding of those same relations, which
ARCHITECTURE.md §3.1/§8 specifies as the eventual target for
publication-quality proof size and verifier speed.

**Why these are not built now:** compiling and proving with circom
requires a Node.js toolchain (`circom` compiler + `snarkjs`), a
universal structured-reference-string ceremony download, and a Rust
toolchain for witness generation performance — all flagged in
`REVIEW.md` (E2/R17) as the single largest schedule risk under a 3-day
deadline, and the sandbox this repo was built in has no network access
to fetch any of them. See `crypto/schnorr_nizk.py`'s module docstring
for the full scope-decision rationale, made 2026-11-09.

**What's here instead:** circuit source written to the same
constraints the Sigma-protocol code already enforces, so that porting
later is a mechanical translation exercise, not a re-derivation of the
relations from scratch. Treat every file below as a **specification**,
not a tested artifact — do not claim in the paper that these circuits
were compiled or benchmarked unless `build.sh` has actually been run
on a machine with the toolchain installed and the resulting proving/
verification times recorded in `STATUS.md`.

## Files

- `consistency.circom` — the R_cons relation: binds one secret key
  across an identity commitment, a round-scope tag, an epoch-scope
  tag, and a VRF output, matching `crypto/r_cons.py`'s
  `prove_identity_binding` / `verify_identity_binding` exactly
  (same four public bases, same one-witness AND structure).
- `share_wellformed.circom` — the R_share relation: proves a
  Shamir share `(i, y_i)` lies on the same degree-(t-1) polynomial as
  a set of published Feldman/Pedersen commitments, without revealing
  the polynomial's other coefficients — the circuit-form analogue of
  the Feldman-VSS check already implemented directly (no ZK needed
  there since Feldman commitments are already publicly checkable) in
  `crypto/dkg.py::verify_sub_share`. Kept here because a future
  PLONK-based R_cons extension that also wants to prove share
  validity *inside the same proof* (rather than as a separate public
  check) would compose this circuit with `consistency.circom`.
- `build.sh` — the intended compile/setup/prove/verify pipeline once
  circom + snarkjs are available. **Not run in this sandbox.** Running
  it is the Colab-side (or local-toolchain) action that would move
  these circuits from "specification" to "verified camera-ready
  artifact."

## Upgrade checklist (for whoever runs this later)

1. Install circom (Rust-based compiler) and snarkjs (Node package).
2. Run `build.sh circuits/consistency.circom` — compiles, runs the
   Powers-of-Tau + Plonk setup, and produces a verification key.
3. Generate a witness from real `crypto/r_cons.py` inputs (sk, rho_t,
   rho_e, vrf_alpha) and confirm the circuit's public outputs
   (pk, I_t, P_e, gamma) match what `prove_identity_binding` computes
   bit-for-bit — this is the correctness gate before trusting the
   circuit at all.
4. Benchmark proving/verification time vs. the Sigma-protocol version
   at the same model scale; record both in `STATUS.md`.
5. Only after step 3 passes should the paper describe this as a
   validated alternative proof system, not before.
