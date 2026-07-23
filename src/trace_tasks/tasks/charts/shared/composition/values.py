"""Pure value helpers for composition-style chart objectives."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class UniqueExtremum(Generic[T]):
    item: T
    value: int
    nearest_value: int
    margin: int


@dataclass(frozen=True)
class UniqueNearest(Generic[T]):
    item: T
    value: int
    distance: int
    margin: int


def int_sum(values: Sequence[int]) -> int:
    """Return a stable integer sum from integer-like values."""

    return int(sum(int(value) for value in values))


def count_from_percent_share(total: int, share: int) -> int:
    """Convert one integer percentage share into an integer count."""

    return int(int(total) * int(share) // 100)


def counts_from_percent_shares(total: int, shares: Mapping[str, int]) -> dict[str, int]:
    """Convert segment percentage shares into per-segment integer counts."""

    return {
        str(segment): count_from_percent_share(int(total), int(share))
        for segment, share in shares.items()
    }


def select_unique_extremum(
    values: Sequence[tuple[T, int]],
    *,
    select_largest: bool,
    min_margin: int = 1,
    error_label: str,
    item_label: str = "items",
) -> UniqueExtremum[T]:
    """Select the unique extremal item from precomputed item/value pairs."""

    ranked = tuple(sorted(values, key=lambda item: int(item[1]), reverse=bool(select_largest)))
    if len(ranked) < 2:
        raise ValueError(f"{str(error_label)} requires at least two {str(item_label)}")
    answer_item, answer_value = ranked[0]
    closest_value = int(ranked[1][1])
    if int(answer_value) == int(closest_value):
        raise ValueError(f"{str(error_label)} extremum is tied")
    margin = abs(int(answer_value) - int(closest_value))
    if int(margin) < int(min_margin):
        raise ValueError(f"{str(error_label)} extremum margin is too small")
    return UniqueExtremum(
        item=answer_item,
        value=int(answer_value),
        nearest_value=int(closest_value),
        margin=int(margin),
    )


def select_unique_nearest(
    items: Sequence[T],
    *,
    value_fn: Callable[[T], int],
    target_value: int,
    min_margin: int,
    error_label: str,
    item_label: str = "items",
) -> UniqueNearest[T]:
    """Select the item whose value is uniquely nearest to a target."""

    ranked = tuple(sorted(items, key=lambda item: abs(int(value_fn(item)) - int(target_value))))
    if len(ranked) < 2:
        raise ValueError(f"{str(error_label)} requires at least two {str(item_label)}")
    answer_item = ranked[0]
    answer_value = int(value_fn(answer_item))
    answer_distance = abs(int(answer_value) - int(target_value))
    closest_distance = abs(int(value_fn(ranked[1])) - int(target_value))
    if int(answer_distance) == int(closest_distance):
        raise ValueError(f"{str(error_label)} nearest target is tied")
    margin = int(closest_distance) - int(answer_distance)
    if int(margin) < int(min_margin):
        raise ValueError(f"{str(error_label)} nearest-target margin is too small")
    return UniqueNearest(
        item=answer_item,
        value=int(answer_value),
        distance=int(answer_distance),
        margin=int(margin),
    )


__all__ = [
    "UniqueExtremum",
    "UniqueNearest",
    "count_from_percent_share",
    "counts_from_percent_shares",
    "int_sum",
    "select_unique_extremum",
    "select_unique_nearest",
]
