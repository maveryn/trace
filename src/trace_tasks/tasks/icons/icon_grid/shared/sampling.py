"""Frequency-structure sampling helpers for icon-grid scenes."""

from __future__ import annotations

from typing import Any, List, Mapping, Tuple

from .....core.sampling import uniform_choice
from .....core.seed import spawn_rng
from ....shared.config_defaults import group_default
from ....shared.deterministic_sampling import uniform_probability_map

from .defaults import IconGridDefaults
from .state import IconGridFrequencySpec


def bounded_compositions(total: int, parts: int, *, min_part: int, max_part: int) -> List[Tuple[int, ...]]:
    """Enumerate ordered integer compositions under inclusive part bounds."""

    if int(parts) <= 0:
        return [()] if int(total) == 0 else []
    if int(parts) == 1:
        if int(min_part) <= int(total) <= int(max_part):
            return [(int(total),)]
        return []
    compositions: List[Tuple[int, ...]] = []
    remaining_parts = int(parts) - 1
    for head in range(int(min_part), int(max_part) + 1):
        min_rest = int(remaining_parts) * int(min_part)
        max_rest = int(remaining_parts) * int(max_part)
        rest_total = int(total) - int(head)
        if rest_total < min_rest or rest_total > max_rest:
            continue
        for tail in bounded_compositions(
            int(rest_total),
            int(remaining_parts),
            min_part=int(min_part),
            max_part=int(max_part),
        ):
            compositions.append((int(head), *tuple(int(value) for value in tail)))
    return compositions


def _frequency_spec_from_distinct_count(
    *,
    rng,
    object_count: int,
    distinct_type_count: int,
    object_count_probabilities: Mapping[str, float],
    target_count_probabilities: Mapping[str, float],
    distinct_color_count: int | None = None,
) -> IconGridFrequencySpec:
    """Build a frequency spec with the requested distinct type count."""

    distinct_count = int(distinct_type_count)
    if distinct_count <= 0:
        raise ValueError("distinct_type_count must be positive")
    if int(object_count) < distinct_count:
        raise ValueError("object_count must be at least distinct_type_count")
    multiplicity_support = bounded_compositions(
        int(object_count),
        distinct_count,
        min_part=1,
        max_part=int(object_count),
    )
    if not multiplicity_support:
        raise ValueError("no feasible multiplicities exist for icon-grid counting")
    multiplicities = tuple(int(value) for value in uniform_choice(rng, tuple(multiplicity_support)))
    singleton_count = sum(1 for value in multiplicities if int(value) == 1)
    repeated_type_multiplicities = tuple(int(value) for value in multiplicities if int(value) > 1)
    return IconGridFrequencySpec(
        object_count=int(object_count),
        singleton_count=int(singleton_count),
        repeated_type_multiplicities=tuple(repeated_type_multiplicities),
        object_count_probabilities=dict(object_count_probabilities),
        target_count_probabilities=dict(target_count_probabilities),
        distinct_color_count=int(distinct_color_count) if distinct_color_count is not None else None,
    )


def _resolve_count_bounds(params: Mapping[str, Any], gen_defaults: Mapping[str, Any], defaults: IconGridDefaults) -> tuple[int, int, int, int]:
    """Resolve common object-count and answer-count bounds."""

    object_count_min = int(params.get("object_count_min", group_default(gen_defaults, "object_count_min", defaults.object_count_min)))
    object_count_max = int(params.get("object_count_max", group_default(gen_defaults, "object_count_max", defaults.object_count_max)))
    target_count_min = int(params.get("target_count_min", group_default(gen_defaults, "target_count_min", defaults.target_count_min)))
    target_count_max = int(params.get("target_count_max", group_default(gen_defaults, "target_count_max", defaults.target_count_max)))
    if object_count_min <= 0 or object_count_max < object_count_min:
        raise ValueError("object_count range is invalid for icon-grid counting")
    if target_count_min <= 0 or target_count_max < target_count_min:
        raise ValueError("target_count range is invalid for icon-grid counting")
    return int(object_count_min), int(object_count_max), int(target_count_min), int(target_count_max)


def resolve_distinct_type_frequency_spec(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    defaults: IconGridDefaults,
    selection_namespace: str,
) -> IconGridFrequencySpec:
    """Resolve a grid with a task-bound count of distinct icon types."""

    object_count_min, object_count_max, target_count_min, target_count_max = _resolve_count_bounds(
        params,
        gen_defaults,
        defaults,
    )
    answer_support = tuple(range(int(target_count_min), int(target_count_max) + 1))
    explicit_target = params.get("target_count")
    if explicit_target is None:
        distinct_type_count = int(
            uniform_choice(
                spawn_rng(int(instance_seed), f"{str(selection_namespace)}:distinct_type_count"),
                answer_support,
            )
        )
    else:
        distinct_type_count = int(explicit_target)
    if distinct_type_count not in answer_support:
        raise ValueError("target_count is outside configured distinct-type support")

    object_support = tuple(range(max(int(object_count_min), int(distinct_type_count)), int(object_count_max) + 1))
    explicit_object_count = params.get("object_count")
    object_count = (
        int(explicit_object_count)
        if explicit_object_count is not None
        else int(
            uniform_choice(
                spawn_rng(int(instance_seed), f"{str(selection_namespace)}:object_count"),
                object_support,
            )
        )
    )
    if object_count not in object_support:
        raise ValueError("object_count is outside configured distinct-type support")

    return _frequency_spec_from_distinct_count(
        rng=spawn_rng(int(instance_seed), f"{str(selection_namespace)}:multiplicity_partition"),
        object_count=int(object_count),
        distinct_type_count=int(distinct_type_count),
        object_count_probabilities=uniform_probability_map(
            object_support,
            selected=int(object_count) if explicit_object_count is not None else None,
        ),
        target_count_probabilities=uniform_probability_map(
            answer_support,
            selected=int(distinct_type_count) if explicit_target is not None else None,
        ),
        distinct_color_count=1,
    )


def resolve_distinct_color_frequency_spec(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    defaults: IconGridDefaults,
    selection_namespace: str,
) -> IconGridFrequencySpec:
    """Resolve a grid with a task-bound count of distinct rendered colors."""

    object_count_min, object_count_max, target_count_min, target_count_max = _resolve_count_bounds(
        params,
        gen_defaults,
        defaults,
    )
    answer_support = tuple(range(int(target_count_min), int(target_count_max) + 1))
    explicit_target = params.get("target_count")
    distinct_color_count = (
        int(explicit_target)
        if explicit_target is not None
        else int(
            uniform_choice(
                spawn_rng(int(instance_seed), f"{str(selection_namespace)}:distinct_color_count"),
                answer_support,
            )
        )
    )
    if distinct_color_count not in answer_support:
        raise ValueError("target_count is outside configured distinct-color support")

    object_support = tuple(range(max(int(object_count_min), int(distinct_color_count)), int(object_count_max) + 1))
    explicit_object_count = params.get("object_count")
    object_count = (
        int(explicit_object_count)
        if explicit_object_count is not None
        else int(
            uniform_choice(
                spawn_rng(int(instance_seed), f"{str(selection_namespace)}:object_count"),
                object_support,
            )
        )
    )
    if object_count not in object_support:
        raise ValueError("object_count is outside configured distinct-color support")

    return _frequency_spec_from_distinct_count(
        rng=spawn_rng(int(instance_seed), f"{str(selection_namespace)}:multiplicity_partition"),
        object_count=int(object_count),
        distinct_type_count=1,
        object_count_probabilities=uniform_probability_map(
            object_support,
            selected=int(object_count) if explicit_object_count is not None else None,
        ),
        target_count_probabilities=uniform_probability_map(
            answer_support,
            selected=int(distinct_color_count) if explicit_target is not None else None,
        ),
        distinct_color_count=int(distinct_color_count),
    )


__all__ = [
    "bounded_compositions",
    "resolve_distinct_color_frequency_spec",
    "resolve_distinct_type_frequency_spec",
]
