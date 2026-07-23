"""Shared semantic sampling helpers."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence, TypeVar

T = TypeVar("T")


def normalize_positive_weights(
    weights: Mapping[str, float],
    *,
    default_keys: Iterable[str] | None = None,
) -> dict[str, float]:
    """Normalize positive weights to probabilities.

    If no positive weights are present:
    - use uniform probabilities over `default_keys` when provided;
    - otherwise raise `ValueError`.
    """
    positive = {str(key): float(value) for key, value in weights.items() if float(value) > 0.0}
    if positive:
        total = sum(positive.values())
        if total <= 0.0:
            raise ValueError("at least one positive weight is required")
        return {key: value / total for key, value in sorted(positive.items())}

    if default_keys is None:
        raise ValueError("at least one positive weight is required")

    keys = [str(key) for key in default_keys]
    if not keys:
        raise ValueError("cannot normalize empty weights without default keys")
    probability = 1.0 / float(len(keys))
    return {key: probability for key in keys}


def weighted_choice(rng, probabilities: Mapping[str, float], *, sort_keys: bool = False) -> str:
    """Sample one key from a probability map using cumulative weights."""
    items = sorted(probabilities.items()) if bool(sort_keys) else list(probabilities.items())
    if not items:
        raise ValueError("cannot sample from an empty probability map")

    roll = float(rng.random())
    cumulative = 0.0
    last_key = str(items[-1][0])
    for key, probability in items:
        cumulative += float(probability)
        if roll <= cumulative:
            return str(key)
    return last_key


def support_probability_map(
    values: Sequence[T],
    *,
    selected: T | None = None,
    sort_keys: bool = False,
) -> dict[str, float]:
    """Return a probability map over one explicit support."""

    items = _uniform_items(values, sort_keys=bool(sort_keys))
    if selected is not None:
        selected_key = str(selected)
        keys = {key for key, _value in items}
        if selected_key not in keys:
            raise ValueError(f"selected value is not in support: {selected!r}")
        return {key: (1.0 if key == selected_key else 0.0) for key, _value in items}
    probability = 1.0 / float(len(items))
    return {key: float(probability) for key, _value in items}


def weighted_support_choice(
    rng,
    values: Sequence[T],
    *,
    weights: Mapping[str, float] | None = None,
    sort_keys: bool = False,
) -> tuple[T, dict[str, float]]:
    """Sample one value from explicit support using optional weights."""

    items = _uniform_items(values, sort_keys=bool(sort_keys))
    value_by_key = {key: value for key, value in items}
    if weights is None:
        probabilities = support_probability_map(values, sort_keys=bool(sort_keys))
    else:
        support_keys = [key for key, _value in items]
        filtered_weights = {
            str(key): float(value)
            for key, value in weights.items()
            if str(key) in set(support_keys)
        }
        probabilities = normalize_positive_weights(
            filtered_weights,
            default_keys=support_keys,
        )
    selected_key = weighted_choice(rng, probabilities, sort_keys=False)
    return value_by_key[str(selected_key)], dict(probabilities)


def _uniform_items(values: Sequence[T], *, sort_keys: bool) -> list[tuple[str, T]]:
    """Return string-keyed support items for uniform semantic sampling."""

    items = [(str(value), value) for value in values]
    if not items:
        raise ValueError("cannot sample from an empty support")
    keys = [key for key, _value in items]
    if len(set(keys)) != len(keys):
        raise ValueError("uniform support values must have unique string keys")
    if bool(sort_keys):
        return sorted(items, key=lambda item: item[0])
    return items


def uniform_choice_with_probabilities(
    rng,
    values: Sequence[T],
    *,
    sort_keys: bool = False,
) -> tuple[T, dict[str, float]]:
    """Sample one support value with equal RNG weight and return probabilities.

    This is the canonical helper for semantic uniform/equal-weight axes inside
    tasks. It intentionally uses an RNG draw over an explicit probability map,
    not seed modulo or deterministic cycling.
    """

    return weighted_support_choice(rng, values, sort_keys=bool(sort_keys))


def uniform_choice(rng, values: Sequence[T], *, sort_keys: bool = False) -> T:
    """Sample one support value with equal RNG weight."""

    selected, _probabilities = uniform_choice_with_probabilities(
        rng,
        values,
        sort_keys=bool(sort_keys),
    )
    return selected


def integer_range_choice(
    rng,
    minimum: int,
    maximum: int,
    *,
    weights: Mapping[str, float] | None = None,
) -> tuple[int, dict[str, float]]:
    """Sample one integer from a contiguous inclusive range."""

    lower = int(minimum)
    upper = int(maximum)
    if lower > upper:
        raise ValueError("minimum must be <= maximum")
    return weighted_support_choice(
        rng,
        tuple(range(lower, upper + 1)),
        weights=weights,
        sort_keys=True,
    )


def sample_without_replacement(rng, values: Sequence[T], count: int) -> tuple[T, ...]:
    """Sample a fixed-size candidate set without replacement."""

    items = list(values)
    sample_count = int(count)
    if sample_count < 0:
        raise ValueError("sample count must be non-negative")
    if sample_count > len(items):
        raise ValueError("sample count cannot exceed support size")
    return tuple(rng.sample(items, sample_count))


def shuffled_support(rng, values: Sequence[T]) -> tuple[T, ...]:
    """Return all support values in seeded random order."""

    items = list(values)
    rng.shuffle(items)
    return tuple(items)


__all__ = [
    "normalize_positive_weights",
    "integer_range_choice",
    "sample_without_replacement",
    "shuffled_support",
    "support_probability_map",
    "uniform_choice",
    "uniform_choice_with_probabilities",
    "weighted_support_choice",
    "weighted_choice",
]
