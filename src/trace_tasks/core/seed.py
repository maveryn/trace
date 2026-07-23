"""Deterministic namespace-based seed derivation for Trace."""

from __future__ import annotations

from random import Random

from blake3 import blake3


SEED_DERIVATION_VERSION = "v0"
_JSON_SAFE_INT_MAX = (1 << 53) - 1


def hash64(instance_seed: int, namespace: str, index: int = 0, version: str = SEED_DERIVATION_VERSION) -> int:
    """Derive a deterministic 64-bit sub-seed from instance seed and namespace."""
    payload = f"{int(instance_seed)}|{version}|{namespace}|{int(index)}".encode("utf-8")
    digest = blake3(payload).digest(length=8)
    # Keep seeds inside the RFC8785 / JSON-safe integer domain.
    return int.from_bytes(digest, byteorder="big", signed=False) % _JSON_SAFE_INT_MAX


def spawn_rng(instance_seed: int, namespace: str, index: int = 0) -> Random:
    """Create a `random.Random` seeded via namespace-based derivation."""
    return Random(hash64(instance_seed=instance_seed, namespace=namespace, index=index))
