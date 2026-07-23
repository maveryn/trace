"""Hash helpers used by Trace identity and integrity checks."""

from __future__ import annotations

from pathlib import Path

from blake3 import blake3


def blake3_hex(data: bytes) -> str:
    """Return a prefixed blake3 digest for raw bytes."""
    return f"blake3:{blake3(data).hexdigest()}"


def blake3_file(path: str | Path) -> str:
    """Return a prefixed blake3 digest for a file's bytes."""
    digest = blake3()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"blake3:{digest.hexdigest()}"
