"""fl/ -- the federated-learning pipeline: models, data splits, the
client/server training loop, evidence accumulation, and trace opening.

Same tested-vs-untested split discipline as crypto/__init__.py, for the
same reason: this sandbox has numpy but no torch and no network (no
torchvision/datasets/huggingface download), so anything that needs a
real model forward pass or a real dataset download cannot be verified
here. The honest split:

PURE-MATH / TESTABLE HERE (no torch, no network; see tests/):
    - splits.py      -- stratified-IID and writer-disjoint split logic,
                         operates on plain index arrays / writer-id
                         lists, not actual images. Disjointness and
                         coverage are asserted and tested with
                         synthetic indices (tests/test_splits.py).
    - norm_rule.py    -- the public update-norm anomaly threshold B_t
                         (median/MAD-based), pure numpy over arbitrary
                         norm arrays (tests/test_norm_rule.py).
    - evidence.py     -- the §6.1 evidence accumulator: bookkeeping
                         over ring-signature key images and epoch
                         pseudonyms (plain Python objects / byte
                         strings), no crypto computation itself, no
                         torch (tests/test_evidence.py).

TORCH-DEPENDENT (not installed in this sandbox -- written carefully,
NOT run here; validate in notebooks/00_Setup_and_Crypto_Tests.ipynb
and the CASE notebooks on Colab, where torch/torchvision/datasets
install normally):
    - models.py       -- ResNet-50 / MobileNetV2 adapters (GroupNorm,
                         from-scratch init per ARCHITECTURE.md's
                         non-IID BatchNorm-leak finding).
    - data_iid.py      -- MNIST loading + the stratified-IID split
                         applied to real tensors.
    - data_noniid.py   -- FEMNIST loading (HuggingFace) + the
                         writer-disjoint split applied to real tensors.
    - client.py        -- local training step + the full crypto
                         integration (ring sign, VRF self-sample,
                         secure-agg mask, Pedersen commit, range proof).
    - server.py        -- round orchestration, aggregation, the V1-V5
                         verification checks, dropout/evidence
                         updates.
    - trace.py         -- the §6.2 opening protocol: committee-side
                         orchestration calling crypto.elgamal_threshold
                         and crypto.r_cons against real round data.

attacks/ and verifier/ follow the same split and are documented at
their own file headers.
"""
