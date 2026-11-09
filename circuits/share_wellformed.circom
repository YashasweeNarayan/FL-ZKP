pragma circom 2.1.0;

// share_wellformed.circom -- SPECIFICATION ONLY, not compiled/tested.
// See circuits/README.md.
//
// Relation proved: a claimed Shamir share (index i, value y_i) lies
// on the SAME degree-(t-1) polynomial whose coefficients are
// Feldman-committed as [C_0, C_1, ..., C_{t-1}] (C_j = coeff_j * G),
// i.e.:
//
//     y_i * G == sum_{j=0}^{t-1} (i^j) * C_j
//
// This is the circuit-form analogue of crypto/dkg.py's
// verify_sub_share, which already checks exactly this equation
// directly (no ZK transform needed, since Feldman commitments are
// public anyway and the check itself reveals nothing beyond what's
// already public). This circuit exists only as a composability
// building block for a future PLONK R_cons variant that wants to
// prove "I hold a valid share" INSIDE a larger proof (e.g. combined
// with consistency.circom) rather than as a separate public check --
// see circuits/README.md's upgrade checklist, step 2's parenthetical.

template PowMod(T) {
    // Computes i^0, i^1, ..., i^(T-1) as circuit signals.
    signal input i;
    signal output powers[T];
    powers[0] <== 1;
    for (var j = 1; j < T; j++) {
        powers[j] <== powers[j - 1] * i;
    }
}

template EdwardsScalarMul() {
    // Same placeholder gadget as consistency.circom -- see that
    // file's note on why this isn't a real implementation here.
    signal input scalar;
    signal input basePointX;
    signal input basePointY;
    signal output outX;
    signal output outY;
}

template EdwardsAdd() {
    signal input aX, aY, bX, bY;
    signal output outX, outY;
    // PLACEHOLDER -- real twisted-Edwards point addition needed.
}

template ShareWellFormed(T) {
    signal input index;           // i (public)
    signal input share_value;     // y_i (PRIVATE witness)
    signal input commitments_x[T]; // C_0..C_{t-1} (public, Feldman commitments)
    signal input commitments_y[T];
    signal input baseG_x, baseG_y;

    component pow = PowMod(T);
    pow.i <== index;

    component mulY = EdwardsScalarMul();
    mulY.scalar <== share_value;
    mulY.basePointX <== baseG_x;
    mulY.basePointY <== baseG_y;
    // mulY.outX/outY == y_i * G -- the LHS of the check.

    component muls[T];
    component adds[T - 1];
    for (var j = 0; j < T; j++) {
        muls[j] = EdwardsScalarMul();
        muls[j].scalar <== pow.powers[j];
        muls[j].basePointX <== commitments_x[j];
        muls[j].basePointY <== commitments_y[j];
    }
    for (var j = 0; j < T - 1; j++) {
        adds[j] = EdwardsAdd();
    }
    // Chain the adds to sum muls[0..T-1] -- wiring omitted, see
    // circuits/README.md: this file is a specification, not a
    // compiled circuit.

    // Final constraint (schematic): mulY.outX/outY ==
    // running sum of muls[j].outX/outY for j in [0, T).
}

component main {public [index, commitments_x, commitments_y, baseG_x, baseG_y]} = ShareWellFormed(10);
