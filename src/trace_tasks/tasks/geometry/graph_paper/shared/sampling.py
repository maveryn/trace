"""Deterministic sampling helpers for graph-paper scene primitives."""

from __future__ import annotations

from math import gcd, sqrt
from random import Random
from typing import Mapping, Sequence

from trace_tasks.core.sampling import uniform_choice
from trace_tasks.core.seed import spawn_rng

from .defaults import int_default
from .state import LABEL_POOL, Point


def rng_for(instance_seed: int, salt: str) -> Random:
    """Create one deterministic random stream for a semantic sampling role."""

    return spawn_rng(int(instance_seed), f"geometry.graph_paper.{salt}")


def choose_from_seed(values: Sequence[str], *, instance_seed: int, salt: str) -> str:
    """Choose one value uniformly using a stable hash stream."""

    if not values:
        raise ValueError("cannot sample from an empty value list")
    rng = rng_for(int(instance_seed), str(salt))
    return str(uniform_choice(rng, tuple(values)))


def resolve_count(
    params: Mapping[str, object], defaults: Mapping[str, object], *, fallback: int
) -> int:
    """Resolve an object count while keeping the result in the supported label pool."""

    requested = int_default(params, defaults, "object_count", fallback)
    return max(1, min(len(LABEL_POOL), int(requested)))


def label_subset(count: int) -> tuple[str, ...]:
    """Return the first visible option labels for one scene."""

    return tuple(
        str(label) for label in LABEL_POOL[: max(1, min(len(LABEL_POOL), int(count)))]
    )


def unique_metric_values(
    rng: Random,
    *,
    count: int,
    low: int,
    high: int,
    avoid_zero: bool = False,
) -> list[int]:
    """Sample unique integer metrics used for unambiguous extrema."""

    support = [
        value
        for value in range(int(low), int(high) + 1)
        if (not avoid_zero or value != 0)
    ]
    rng.shuffle(support)
    if len(support) < int(count):
        raise ValueError("metric support is too small for requested object count")
    return [int(value) for value in support[: int(count)]]


def count_target(
    params: Mapping[str, object],
    defaults: Mapping[str, object],
    *,
    object_count: int,
    instance_seed: int,
    salt: str,
) -> int:
    """Resolve a target count for classification-count objectives."""

    if "target_count" in params:
        return max(0, min(int(object_count), int(params["target_count"])))
    low = int_default(params, defaults, "target_count_min", 1)
    high = int_default(
        params, defaults, "target_count_max", max(1, int(object_count) - 1)
    )
    low = max(0, min(int(object_count), int(low)))
    high = max(low, min(int(object_count), int(high)))
    support = list(range(low, high + 1))
    if not support:
        return 0
    rng = rng_for(int(instance_seed), f"{salt}.target_count")
    return int(uniform_choice(rng, tuple(support)))


def make_class_sequence(
    *,
    target_class: str,
    distractor_classes: Sequence[str],
    object_count: int,
    target_count: int,
    rng: Random,
) -> list[str]:
    """Build a shuffled class sequence with exactly the requested matches."""

    if int(target_count) > int(object_count):
        raise ValueError("target_count cannot exceed object_count")
    if not distractor_classes and int(target_count) < int(object_count):
        raise ValueError("distractor_classes required when not all objects match")
    classes = [str(target_class)] * int(target_count)
    for _index in range(int(object_count) - int(target_count)):
        classes.append(str(rng.choice(tuple(distractor_classes))))
    rng.shuffle(classes)
    return classes


def reduced_slope(dy: int, dx: int) -> tuple[int, int]:
    """Return a reduced dy/dx pair for slope metadata."""

    if int(dx) == 0:
        raise ValueError("dx cannot be zero")
    divisor = gcd(abs(int(dy)), abs(int(dx))) or 1
    return int(dy) // divisor, int(dx) // divisor


def distance_units(start: Point, end: Point) -> float:
    """Euclidean distance in graph units."""

    return sqrt(
        (float(end[0]) - float(start[0])) ** 2 + (float(end[1]) - float(start[1])) ** 2
    )
