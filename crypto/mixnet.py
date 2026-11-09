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
SEAL_OVERHEAD = 48  # nacl.public.SealedBox ciphertext expansion: crypto_box_SEALBYTES
                     # (32-byte ephemeral pubkey + 16-byte Poly1305 MAC)
LEN_PREFIX = 4       # _pad's own length-prefix, added once per _pad call
ADDR_FIELD = 32      # fixed-width hop-address field (hostnames padded/truncated to
                     # this) -- a variable-length address would itself be a
                     # size side-channel, defeating the point of fixed cells
WRAP_OVERHEAD = SEAL_OVERHEAD + LEN_PREFIX + ADDR_FIELD  # bytes added per hop of wrapping


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


def _encode_addr(addr: str) -> bytes:
    """Fixed-width ADDR_FIELD encoding so every hop's routing-info field
    is the same size regardless of the actual hostname -- otherwise the
    length of `addr` itself would leak through the layer size, which is
    exactly what the fixed-CELL_SIZE design is trying to prevent.
    """
    raw = addr.encode()
    if len(raw) > ADDR_FIELD:
        raise ValueError(f"hop address {addr!r} ({len(raw)} bytes) exceeds ADDR_FIELD={ADDR_FIELD}")
    return raw + b"\x00" * (ADDR_FIELD - len(raw))


def _decode_addr(field: bytes) -> str:
    return field.rstrip(b"\x00").decode()


def build_onion(payload: bytes, hop_public_keys: list, hop_addresses: list[str]) -> bytes:
    """Layer encryption from the INNERMOST hop outward, so the client
    produces a single blob that peels off one layer per hop.

    hop_public_keys[-1] is the final mix node (adjacent to the server);
    hop_public_keys[0] is the first hop (adjacent to the client).
    len(hop_public_keys) == len(hop_addresses) == 3 for the committed
    3-hop design, but the function is written generally.

    Each wrap adds exactly WRAP_OVERHEAD bytes (a fixed-width address
    field + SealedBox's ciphertext expansion + this _pad call's own
    length-prefix), so the budget for the CLIENT's actual payload has
    to be CELL_SIZE minus n_hops * WRAP_OVERHEAD, reserved up front --
    not CELL_SIZE itself. Padding the payload to the full CELL_SIZE
    before wrapping (the earlier, broken version of this function)
    leaves no room for any of the hop wrapping and overflows on the
    very first wrap. Sizes are chosen so the FINAL (outermost,
    client-to-hop1) ciphertext lands at exactly CELL_SIZE; inner,
    already-peeled layers are smaller by design -- real mix nodes
    re-pad after peeling before forwarding (see peel_layer's docstring)
    so the shrinkage never appears on the wire.
    """
    _require_nacl()
    import nacl.public

    n_hops = len(hop_public_keys)
    capacity = CELL_SIZE - n_hops * WRAP_OVERHEAD
    if len(payload) > capacity:
        raise ValueError(
            f"payload {len(payload)} bytes exceeds max {capacity} bytes "
            f"for a {n_hops}-hop onion with CELL_SIZE={CELL_SIZE}"
        )

    layer = _pad(payload, size=capacity)
    # work from last hop to first: each outer layer wraps
    # (next_hop_address, inner_ciphertext_so_far). layer's size grows by
    # exactly WRAP_OVERHEAD on each iteration, by construction, so it
    # reaches exactly CELL_SIZE after the final (i=0) wrap.
    for i in range(len(hop_public_keys) - 1, -1, -1):
        next_addr = hop_addresses[i + 1] if i + 1 < len(hop_addresses) else "__SERVER__"
        plaintext_for_this_hop = _encode_addr(next_addr) + layer
        pad_target = len(plaintext_for_this_hop) + LEN_PREFIX  # minimum -- no slack needed, size is already exact
        box = nacl.public.SealedBox(hop_public_keys[i])
        layer = box.encrypt(_pad(plaintext_for_this_hop, size=pad_target))
    assert len(layer) == CELL_SIZE, f"internal sizing bug: outermost cell is {len(layer)}, expected {CELL_SIZE}"
    return layer


def peel_layer(onion: bytes, my_keypair: MixKeypair) -> tuple[str, bytes]:
    """One mix node's operation: decrypt the outer layer meant for it,
    return (next_hop_address_or___SERVER__, remaining_onion_bytes).

    NOTE: `remaining` is smaller than CELL_SIZE by this hop's own
    WRAP_OVERHEAD -- in a real deployment the mix node re-pads it back
    up to CELL_SIZE with fresh random bytes before forwarding, so an
    observer on the next hop's link can't tell how many layers remain
    from the cell size alone. That re-padding is a network-layer
    concern (left to fl/'s network simulator, same as
    simulate_batch_delay below); this function returns the real
    (unpadded-back-up) bytes so the next peel_layer/final_payload call
    in a test or simulation can consume them directly.
    """
    _require_nacl()
    import nacl.public

    box = nacl.public.SealedBox(my_keypair.private_key)
    decrypted_padded = box.decrypt(onion)
    decrypted = _unpad(decrypted_padded)
    addr_field, remaining = decrypted[:ADDR_FIELD], decrypted[ADDR_FIELD:]
    return _decode_addr(addr_field), remaining


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
