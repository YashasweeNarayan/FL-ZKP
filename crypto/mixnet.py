"""3-hop onion mix cascade (ARCHITECTURE.md §2.3) -- EC-DEPENDENT, Colab-only
validation (needs pynacl; see ristretto.py's header for the general note).

Ring signatures and the R_cons relation give cryptographic
unlinkability of WHO authored an update, but say nothing about the
network layer: if the server (or a network observer) can see which
client IP submitted at which timestamp, round-level participation
metadata leaks even when the signed content is anonymous. This module
closes that gap with a fixed-size, padded, layered-encryption mix
cascade across 3 independent mix nodes (assumed non-colluding -- see
ARCHITECTURE.md threat model A4), Chaumian-style:

    client -> mix1 -> mix2 -> mix3 -> server

Each hop decrypts one layer (NaCl sealed box, X25519+XSalsa20-Poly1305)
to recover (next_hop_address, inner_ciphertext), re-pads to the fixed
cell size, and forwards. No single mix node, even if malicious, learns
both "who the client was" AND "what the server received" -- only the
two endpoints it directly touches. Fixed cell size + padding prevents
size-correlation attacks; batching + random forwarding delay per hop
(not implemented at the crypto layer, left to the network simulator in
fl/ -- see mixnet.simulate_batch_delay for the hook) prevents simple
timing correlation across the full path.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

CELL_SIZE = 4096  # fixed onion cell size in bytes; all payloads padded/split to this


def _require_nacl():
    try:
        import nacl.public  # noqa: F401
        import nacl.secret  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "crypto/mixnet.py needs pynacl (not installed in this sandbox). "
            "This module is EC-DEPENDENT -- validate it in "
            "notebooks/00_Setup_and_Crypto_Tests.ipynb on Colab, where pynacl "
            "installs normally (network available there)."
        ) from e


@dataclass(frozen=True)
class MixKeypair:
    """A mix node's long-lived X25519 keypair (distinct from the
    ristretto255 signing keys used elsewhere -- this is pure
    Diffie-Hellman key-encapsulation key material, via nacl.public).
    """
    private_key: object  # nacl.public.PrivateKey
    public_key: object  # nacl.public.PublicKey

    @staticmethod
    def generate() -> "MixKeypair":
        _require_nacl()
        import nacl.public
        sk = nacl.public.PrivateKey.generate()
        return MixKeypair(sk, sk.public_key)


def _pad(data: bytes, size: int = CELL_SIZE) -> bytes:
    if len(data) > size - 4:
        raise ValueError(f"payload {len(data)} bytes exceeds max {size - 4} after length prefix")
    length_prefix = len(data).to_bytes(4, "big")
    body = length_prefix + data
    padding = os.urandom(size - len(body))
    return body + padding


def _unpad(padded: bytes) -> bytes:
    length = int.from_bytes(padded[:4], "big")
    return padded[4 : 4 + length]


def build_onion(payload: bytes, hop_public_keys: list, hop_addresses: list[str]) -> bytes:
    """Layer encryption from the INNERMOST hop outward, so the client
    produces a single blob that peels off one layer per hop.

    hop_public_keys[-1] is the final mix node (adjacent to the server);
    hop_public_keys[0] is the first hop (adjacent to the client).
    len(hop_public_keys) == len(hop_addresses) == 3 for the committed
    3-hop design, but the function is written generally.
    """
    _require_nacl()
    import nacl.public

    layer = _pad(payload)
    # work from last hop to first: each outer layer wraps
    # (next_hop_address, inner_ciphertext_so_far)
    for i in range(len(hop_public_keys) - 1, -1, -1):
        next_addr = hop_addresses[i + 1] if i + 1 < len(hop_addresses) else "__SERVER__"
        plaintext_for_this_hop = next_addr.encode() + b"||" + layer
        box = nacl.public.SealedBox(hop_public_keys[i])
        layer = box.encrypt(_pad(plaintext_for_this_hop))
    return layer


def peel_layer(onion: bytes, my_keypair: MixKeypair) -> tuple[str, bytes]:
    """One mix node's operation: decrypt the outer layer meant for it,
    return (next_hop_address_or___SERVER__, remaining_onion_bytes).
    """
    _require_nacl()
    import nacl.public

    box = nacl.public.SealedBox(my_keypair.private_key)
    decrypted_padded = box.decrypt(onion)
    decrypted = _unpad(decrypted_padded)
    next_addr_bytes, remaining = decrypted.split(b"||", 1)
    return next_addr_bytes.decode(), remaining


def final_payload(onion: bytes) -> bytes:
    """After the last hop has been peeled (next_addr == '__SERVER__'),
    `remaining` IS the original client payload wrapped in one more pad
    layer from build_onion's innermost _pad call -- unpad it here.
    """
    return _unpad(onion)


def cell_fingerprint(onion: bytes) -> str:
    """For the network simulator / audit log: a content-blind
    fixed-size fingerprint so mix traffic can be counted/batched without
    revealing cell contents."""
    return hashlib.sha256(onion).hexdigest()[:16]


def simulate_batch_delay(rng, min_ms: int = 50, max_ms: int = 500) -> int:
    """Hook for the network simulator (fl/ layer, not this module) to
    draw a random per-hop forwarding delay, defeating naive end-to-end
    timing correlation. Pure utility, no crypto -- kept here because
    it's mix-cascade-specific policy, not a general fl/ concern.
    """
    return int(rng.uniform(min_ms, max_ms))
