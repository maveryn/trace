"""Sampling helpers for tangent-packing construction cases."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from trace_tasks.core.sampling import uniform_choice
from trace_tasks.core.seed import spawn_rng

from .state import TangentPackingCase

RADIUS_SUPPORT: tuple[int, ...] = tuple(range(3, 81))


def uniform_probability_map(values: Sequence[Any], *, key_fn: Callable[[Any], str] = str) -> dict[str, float]:
    """Return a uniform probability map over unique formatted values."""

    keys = tuple(dict.fromkeys(str(key_fn(value)) for value in values))
    if not keys:
        return {}
    probability = 1.0 / float(len(keys))
    return {key: probability for key in keys}


def choose_radius(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
    support: Sequence[int] = RADIUS_SUPPORT,
) -> tuple[TangentPackingCase, dict[str, float]]:
    """Choose one radius case, with optional explicit radius override."""

    radius_support = tuple(int(value) for value in support)
    if not radius_support:
        raise ValueError("empty tangent-packing radius support")
    explicit = params.get("radius")
    if explicit is None:
        explicit = params.get("target_radius")
    if explicit is not None:
        radius = int(explicit)
        if radius not in set(radius_support):
            raise ValueError(f"unsupported tangent-packing radius: {radius}")
        return TangentPackingCase(radius=int(radius)), {str(radius): 1.0}
    rng = spawn_rng(int(instance_seed), str(namespace))
    radius = int(uniform_choice(rng, radius_support))
    return TangentPackingCase(radius=int(radius)), uniform_probability_map(radius_support)


def answer_support_probabilities(
    *,
    support: Sequence[int] = RADIUS_SUPPORT,
    answer_fn: Callable[[int], float],
    key_fn: Callable[[float], str],
) -> dict[str, float]:
    """Return answer-support probabilities induced by uniform radius sampling."""

    return uniform_probability_map((answer_fn(int(radius)) for radius in support), key_fn=key_fn)


__all__ = [
    "RADIUS_SUPPORT",
    "answer_support_probabilities",
    "choose_radius",
    "uniform_probability_map",
]
