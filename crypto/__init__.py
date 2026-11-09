"""JANUS-FL cryptographic core.

Two tiers, by what this sandbox could actually verify while writing them
(see repo-root STATUS.md for the authoritative, up-to-date ledger):

  PURE-MATH  (verified here, in a network-less sandbox, with real unit
              tests that ran and passed before this file was committed):
    field.py, shamir.py, sketch.py, dp.py

  EC-DEPENDENT (written carefully against the spec in ARCHITECTURE.md,
              but require `pynacl` for ristretto255 group operations,
              which this sandbox could not install — no network. These
              are validated on first run of notebooks/00_Setup_and_Crypto_Tests.ipynb
              on Colab, which DOES have network. Do not trust a result
              produced by these modules until that notebook's test cells
              have gone green for your environment):
    ristretto.py, pedersen.py, elgamal_threshold.py, clsag.py, dkg.py,
    committee.py, reshare.py, vrf.py, mixnet.py, rangeproof.py
"""
