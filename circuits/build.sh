#!/usr/bin/env bash
# build.sh -- intended compile/setup/prove/verify pipeline for the
# circuits/ camera-ready upgrade path. NOT RUN in this sandbox (no
# network access to fetch circom/snarkjs/the Powers-of-Tau ceremony
# file -- see circuits/README.md and REVIEW.md's E2/R17 risk entry).
#
# This script is a specification of the intended workflow, written so
# that whoever has toolchain access (e.g. on Colab, or a local machine
# with Node+Rust) can run it directly rather than re-deriving the
# circom/snarkjs invocation sequence from scratch. Treat any claim that
# this was "run successfully" as false unless STATUS.md records actual
# timing/output from an actual run.
set -euo pipefail

CIRCUIT="${1:-consistency.circom}"
BUILD_DIR="build/$(basename "$CIRCUIT" .circom)"
PTAU_FILE="pot15_final.ptau"  # Powers-of-Tau, degree must exceed circuit's constraint count

mkdir -p "$BUILD_DIR"

echo "[1/6] Checking toolchain..."
command -v circom >/dev/null 2>&1 || { echo "circom not found -- install from https://docs.circom.io"; exit 1; }
command -v snarkjs >/dev/null 2>&1 || { echo "snarkjs not found -- npm install -g snarkjs"; exit 1; }

echo "[2/6] Compiling $CIRCUIT ..."
circom "$CIRCUIT" --r1cs --wasm --sym -o "$BUILD_DIR"

echo "[3/6] Fetching/locating Powers-of-Tau ceremony file ($PTAU_FILE) ..."
if [ ! -f "$PTAU_FILE" ]; then
    echo "  -- not found locally. Download from the Hermez/iden3 ceremony,"
    echo "     e.g.: https://github.com/iden3/snarkjs#7-prepare-phase-2"
    echo "     (requires network access this sandbox does not have)."
    exit 1
fi

echo "[4/6] PLONK setup..."
snarkjs plonk setup "$BUILD_DIR"/*.r1cs "$PTAU_FILE" "$BUILD_DIR/circuit_final.zkey"
snarkjs zkey export verificationkey "$BUILD_DIR/circuit_final.zkey" "$BUILD_DIR/verification_key.json"

echo "[5/6] Generating witness from inputs.json (must be supplied, matching"
echo "       the public/private signal names in the .circom source) ..."
if [ ! -f "$BUILD_DIR/inputs.json" ]; then
    echo "  -- $BUILD_DIR/inputs.json not found. Generate it from real"
    echo "     crypto/r_cons.py inputs before running this step -- see"
    echo "     circuits/README.md's upgrade checklist, step 3."
    exit 1
fi
node "$BUILD_DIR"/*_js/generate_witness.js "$BUILD_DIR"/*_js/*.wasm "$BUILD_DIR/inputs.json" "$BUILD_DIR/witness.wtns"

echo "[6/6] Proving + verifying..."
snarkjs plonk prove "$BUILD_DIR/circuit_final.zkey" "$BUILD_DIR/witness.wtns" "$BUILD_DIR/proof.json" "$BUILD_DIR/public.json"
snarkjs plonk verify "$BUILD_DIR/verification_key.json" "$BUILD_DIR/public.json" "$BUILD_DIR/proof.json"

echo "Done. Record proving/verification timings in STATUS.md before citing these as validated."
