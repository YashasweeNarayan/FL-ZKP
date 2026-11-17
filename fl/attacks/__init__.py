"""Per-case attack configuration for the MVP WITH_ATTACK notebook matrix.

See fl/attacks/mvp_matrix.py -- the single source of truth for which
clients are malicious, and which dedicated checkpoint git branch each
WITH_ATTACK notebook variant pushes to (FIX.md F17).

This package holds attack CONFIGURATION only (which clients cheat, in
which of the 4 model x dataset cases, on which branch). The attack
CONSTRUCTION itself (how the malicious delta is actually crafted) stays
in the top-level `attacks/` package (e.g. `attacks.a1_scaling`), which
predates this one and is also used standalone by RUN_ATTACKS.ipynb and
the F11 verification gate -- not duplicated or moved here.
"""
