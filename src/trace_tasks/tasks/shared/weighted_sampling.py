"""Small weighted-support sampling helpers for task samplers."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple, TypeVar


T = TypeVar("T")


def weighted_probability_map(
    support: Sequence[T],
    raw_weights: Mapping[Any, Any] | Sequence[Any] | None,
) -> Dict[str, float]:
    """Normalize optional weights over a concrete support sequence."""

    values: Tuple[T, ...] = tuple(support)
    if not values:
        raise ValueError("weighted support must contain at least one value")

    weights: Dict[T, float] = {}
    if isinstance(raw_weights, Mapping):
        for value in values:
            raw = raw_weights.get(value, raw_weights.get(str(value), 0.0))
            weights[value] = max(0.0, float(raw))
    elif isinstance(raw_weights, Sequence) and not isinstance(raw_weights, (str, bytes)) and len(raw_weights) == len(values):
        weights = {
            value: max(0.0, float(raw_weights[index]))
            for index, value in enumerate(values)
        }
    else:
        weights = {value: 1.0 for value in values}

    total = sum(float(value) for value in weights.values())
    if total <= 0.0:
        probability = 1.0 / float(len(values))
        return {str(value): probability for value in values}
    return {str(value): float(weights[value]) / float(total) for value in values}


def sample_weighted_value(rng, support: Sequence[T], probabilities: Mapping[T, float]) -> T:
    """Sample one value from a probability map over support."""

    values: Tuple[T, ...] = tuple(support)
    if not values:
        raise ValueError("weighted support must contain at least one value")
    threshold = float(rng.random())
    cumulative = 0.0
    for value in values:
        cumulative += float(probabilities.get(value, probabilities.get(str(value), 0.0)))  # type: ignore[arg-type]
        if threshold <= cumulative:
            return value
    return values[-1]


__all__ = ["sample_weighted_value", "weighted_probability_map"]
