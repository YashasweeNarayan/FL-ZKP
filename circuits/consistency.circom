pragma circom 2.1.0;

// consistency.circom -- camera-ready circuit encoding of R_cons.
// SPECIFICATION ONLY -- not compiled/tested in this repo. See
// circuits/README.md for why, and the exact mapping to
// crypto/r_cons.py's prove_identity_binding/verify_identity_binding,
// which is the TESTED, WORKING version of this same relation for the
// MVP build.
//
// Relation proved: given a secret scalar `sk` (the witness), and
// public bases G, BaseT, BaseE, BaseVrf (each a fixed elliptic-curve
// point, encoded here as circuit constants derived from
// crypto/clsag.py's h_p(rho_t)/h_p(rho_e) and crypto/vrf.py's
// h_p(alpha) for the specific round/epoch/VRF-input in play):
//
//     pk    == sk * G
//     I_t   == sk * BaseT
//     P_e   == sk * BaseE
//     gamma == sk * BaseVrf
//
// i.e. exactly crypto/r_cons.py's AND-composition, but checked as
// four scalar-multiplication constraints inside one PLONK circuit
// instead of one 4-base Sigma-protocol AND proof. The *security
// property* (one secret key, four simultaneous public commitments,
// nothing else learned about sk) is identical; only the proof
// system differs.
//
// NOTE: circom has no native ristretto255 gadget in its standard
// library. A real implementation needs a twisted-Edwards or Montgomery
// curve gadget library (e.g. circomlib's baby-jubjub, or a custom
// ristretto255-in-circom gadget) -- this is itself nontrivial
// engineering, which is part of why this was deferred past the 3-day
// MVP. The template below is written against a generic
// `EdwardsScalarMul` gadget as a placeholder for whichever curve
// gadget is actually selected during the camera-ready port.

template EdwardsScalarMul() {
    // PLACEHOLDER GADGET -- not a real implementation. A camera-ready
    // port must replace this with a vetted scalar-multiplication
    // circuit (e.g. from circomlib) over whichever curve is chosen.
    signal input scalar;
    signal input basePointX;
    signal input basePointY;
    signal output outX;
    signal output outY;
    // Intentionally left unconstrained -- see note above.
}

template RConsIdentityBinding() {
    // Private witness: the signer's secret scalar.
    signal input sk;

    // Public inputs: the four bases (derived off-circuit from
    // rho_t, rho_e, vrf_alpha exactly as crypto/r_cons.py does) and
    // the four claimed output points.
    signal input baseG_x, baseG_y;
    signal input baseT_x, baseT_y;
    signal input baseE_x, baseE_y;
    signal input baseVrf_x, baseVrf_y;

    signal input pk_x, pk_y;
    signal input It_x, It_y;
    signal input Pe_x, Pe_y;
    signal input gamma_x, gamma_y;

    component mulG = EdwardsScalarMul();
    mulG.scalar <== sk;
    mulG.basePointX <== baseG_x;
    mulG.basePointY <== baseG_y;
    pk_x === mulG.outX;
    pk_y === mulG.outY;

    component mulT = EdwardsScalarMul();
    mulT.scalar <== sk;
    mulT.basePointX <== baseT_x;
    mulT.basePointY <== baseT_y;
    It_x === mulT.outX;
    It_y === mulT.outY;

    component mulE = EdwardsScalarMul();
    mulE.scalar <== sk;
    mulE.basePointX <== baseE_x;
    mulE.basePointY <== baseE_y;
    Pe_x === mulE.outX;
    Pe_y === mulE.outY;

    component mulVrf = EdwardsScalarMul();
    mulVrf.scalar <== sk;
    mulVrf.basePointX <== baseVrf_x;
    mulVrf.basePointY <== baseVrf_y;
    gamma_x === mulVrf.outX;
    gamma_y === mulVrf.outY;
}

component main {public [baseG_x, baseG_y, baseT_x, baseT_y, baseE_x, baseE_y, baseVrf_x, baseVrf_y, pk_x, pk_y, It_x, It_y, Pe_x, Pe_y, gamma_x, gamma_y]} = RConsIdentityBinding();
