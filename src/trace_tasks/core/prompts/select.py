"""Deterministic prompt variant selection utilities."""

from __future__ import annotations

from typing import Sequence, Tuple

from ..seed import spawn_rng


def choose_variant(templates: Sequence[str], *, instance_seed: int, namespace: str) -> Tuple[str, int, int]:
    """Select one template deterministically from a template list."""
    if not templates:
        raise ValueError("templates must be non-empty")
    rng = spawn_rng(instance_seed, namespace)
    index = int(rng.randrange(len(templates)))
    return str(templates[index]), index, int(len(templates))

