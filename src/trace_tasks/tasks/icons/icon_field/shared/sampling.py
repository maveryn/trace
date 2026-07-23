"""Frequency-structure sampling helpers for icon-field scenes."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Tuple

from .....core.sampling import uniform_choice
from .....core.seed import spawn_rng
from ....shared.config_defaults import group_default
from ....shared.deterministic_sampling import uniform_probability_map

from .defaults import IconFieldDefaults
from .state import TypeFrequencySpec


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


def resolve_singleton_frequency_spec(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    defaults: IconFieldDefaults,
    selection_namespace: str,
) -> TypeFrequencySpec:
    """Resolve singleton and repeated-type counts for one icon field."""

    object_count_min = int(params.get("object_count_min", group_default(gen_defaults, "object_count_min", defaults.object_count_min)))
    object_count_max = int(params.get("object_count_max", group_default(gen_defaults, "object_count_max", defaults.object_count_max)))
    target_count_min = int(params.get("target_count_min", group_default(gen_defaults, "target_count_min", defaults.target_count_min)))
    target_count_max = int(params.get("target_count_max", group_default(gen_defaults, "target_count_max", defaults.target_count_max)))
    repeated_type_count_min = int(
        params.get(
            "repeated_type_count_min",
            group_default(gen_defaults, "repeated_type_count_min", defaults.repeated_type_count_min),
        )
    )
    repeated_type_count_max = int(
        params.get(
            "repeated_type_count_max",
            group_default(gen_defaults, "repeated_type_count_max", defaults.repeated_type_count_max),
        )
    )
    repeated_type_multiplicity_min = int(
        params.get(
            "repeated_type_multiplicity_min",
            group_default(gen_defaults, "repeated_type_multiplicity_min", defaults.repeated_type_multiplicity_min),
        )
    )
    repeated_type_multiplicity_max = int(
        params.get(
            "repeated_type_multiplicity_max",
            group_default(gen_defaults, "repeated_type_multiplicity_max", defaults.repeated_type_multiplicity_max),
        )
    )
    if object_count_min <= 0 or object_count_max < object_count_min:
        raise ValueError("object_count range is invalid")
    if target_count_min < 0 or target_count_max < target_count_min:
        raise ValueError("target_count range is invalid")
    if repeated_type_count_min <= 0 or repeated_type_count_max < repeated_type_count_min:
        raise ValueError("repeated_type_count range is invalid")
    if repeated_type_multiplicity_min < 2 or repeated_type_multiplicity_max < repeated_type_multiplicity_min:
        raise ValueError("repeated_type_multiplicity range is invalid")

    singleton_support = tuple(range(int(target_count_min), int(target_count_max) + 1))
    if not singleton_support:
        raise ValueError("singleton target support is empty")
    explicit_target = params.get("target_count")
    explicit_object_count = params.get("object_count")

    if explicit_target is not None:
        singleton_count = int(explicit_target)
        answer_probability_selected = int(singleton_count)
    else:
        singleton_count = int(
            uniform_choice(
                spawn_rng(int(instance_seed), f"{str(selection_namespace)}:singleton_count"),
                singleton_support,
            )
        )
        answer_probability_selected = None
    if singleton_count not in singleton_support:
        raise ValueError("target_count is outside configured support")

    object_support = tuple(
        value
        for value in range(
            max(int(object_count_min), int(singleton_count) + int(repeated_type_multiplicity_min)),
            int(object_count_max) + 1,
        )
    )
    if not object_support:
        raise ValueError("no feasible object_count values exist for singleton-type counting")
    if explicit_object_count is not None:
        object_count = int(explicit_object_count)
    else:
        object_count = int(
            uniform_choice(
                spawn_rng(int(instance_seed), f"{str(selection_namespace)}:object_count"),
                object_support,
            )
        )
    if object_count not in object_support:
        raise ValueError("object_count is outside configured singleton-type support")
    repeated_icon_count = int(object_count) - int(singleton_count)

    if repeated_icon_count < int(repeated_type_multiplicity_min):
        raise ValueError("repeated icon count is too small to form one repeated type")

    def group_support_for(count: int) -> Tuple[int, ...]:
        min_group_count = max(
            int(repeated_type_count_min),
            (int(count) + int(repeated_type_multiplicity_max) - 1) // int(repeated_type_multiplicity_max),
        )
        max_group_count = min(
            int(repeated_type_count_max),
            int(count) // int(repeated_type_multiplicity_min),
        )
        if min_group_count > max_group_count:
            return ()
        return tuple(range(int(min_group_count), int(max_group_count) + 1))

    group_support = group_support_for(int(repeated_icon_count))
    if not group_support:
        raise ValueError("no feasible repeated_type_count values exist for singleton-type counting")
    repeated_type_count = int(
        uniform_choice(
            spawn_rng(int(instance_seed), f"{str(selection_namespace)}:repeated_type_count"),
            group_support,
        )
    )
    multiplicity_support = bounded_compositions(
        int(repeated_icon_count),
        int(repeated_type_count),
        min_part=int(repeated_type_multiplicity_min),
        max_part=int(repeated_type_multiplicity_max),
    )
    if not multiplicity_support:
        raise ValueError("no feasible repeated multiplicities exist for singleton-type counting")
    repeated_type_multiplicities = tuple(
        int(value)
        for value in uniform_choice(
            spawn_rng(int(instance_seed), f"{str(selection_namespace)}:multiplicity_partition"),
            tuple(multiplicity_support),
        )
    )

    return TypeFrequencySpec(
        object_count=int(object_count),
        singleton_count=int(singleton_count),
        repeated_type_multiplicities=tuple(repeated_type_multiplicities),
        object_count_probabilities=dict(
            uniform_probability_map(
                tuple(
                    value
                    for value in range(
                        max(int(object_count_min), int(singleton_count) + int(repeated_type_multiplicity_min)),
                        int(object_count_max) + 1,
                    )
                ),
                selected=int(object_count) if explicit_object_count is not None else None,
            )
        ),
        target_count_probabilities=dict(
            uniform_probability_map(
                singleton_support,
                selected=answer_probability_selected,
            )
        ),
        distinct_color_count=1,
    )


def candidate_most_frequent_partitions(
    *,
    object_count: int,
    winning_frequency: int,
    other_repeated_type_count_max: int,
) -> List[Tuple[int, Tuple[int, ...]]]:
    """Return feasible `(singleton_count, repeated multiplicities)` candidates.

    The first multiplicity is the unique most-frequent type. All other repeated
    type multiplicities are below it.
    """

    remaining = int(object_count) - int(winning_frequency)
    if remaining < 1:
        return []
    candidates: List[Tuple[int, Tuple[int, ...]]] = []
    max_other_groups = max(0, int(other_repeated_type_count_max))
    for singleton_count in range(1, int(remaining) + 1):
        repeated_remainder = int(remaining) - int(singleton_count)
        if repeated_remainder == 0:
            candidates.append((int(singleton_count), (int(winning_frequency),)))
            continue
        for other_group_count in range(1, int(max_other_groups) + 1):
            if int(repeated_remainder) < 2 * int(other_group_count):
                continue
            other_parts = bounded_compositions(
                int(repeated_remainder),
                int(other_group_count),
                min_part=2,
                max_part=max(2, int(winning_frequency) - 1),
            )
            for parts in other_parts:
                if any(int(value) >= int(winning_frequency) for value in parts):
                    continue
                candidates.append(
                    (
                        int(singleton_count),
                        (int(winning_frequency), *tuple(int(value) for value in parts)),
                    )
                )
    return candidates


def resolve_most_frequent_frequency_spec(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    defaults: IconFieldDefaults,
    selection_namespace: str,
) -> TypeFrequencySpec:
    """Resolve a unique-most-frequent type structure for one icon field."""

    object_count_min = int(params.get("object_count_min", group_default(gen_defaults, "object_count_min", defaults.object_count_min)))
    object_count_max = int(params.get("object_count_max", group_default(gen_defaults, "object_count_max", defaults.object_count_max)))
    target_count_min = int(params.get("target_count_min", group_default(gen_defaults, "target_count_min", defaults.target_count_min)))
    target_count_max = int(params.get("target_count_max", group_default(gen_defaults, "target_count_max", defaults.target_count_max)))
    other_repeated_type_count_max = int(
        params.get(
            "other_repeated_type_count_max",
            group_default(gen_defaults, "other_repeated_type_count_max", defaults.other_repeated_type_count_max),
        )
    )
    if object_count_min <= 1 or object_count_max < object_count_min:
        raise ValueError("object_count range is invalid for most-frequent-type counting")
    if target_count_min < 2 or target_count_max < target_count_min:
        raise ValueError("target_count range is invalid for most-frequent-type counting")
    if other_repeated_type_count_max < 0:
        raise ValueError("other_repeated_type_count_max must be non-negative")

    answer_support = tuple(range(int(target_count_min), int(target_count_max) + 1))
    explicit_target = params.get("target_count")
    explicit_object_count = params.get("object_count")

    if explicit_target is not None:
        winning_frequency = int(explicit_target)
    else:
        winning_frequency = int(
            uniform_choice(
                spawn_rng(int(instance_seed), f"{str(selection_namespace)}:winning_frequency"),
                answer_support,
            )
        )
    if winning_frequency not in answer_support:
        raise ValueError("target_count is outside configured most-frequent support")

    object_support = tuple(
        value
        for value in range(max(int(object_count_min), int(winning_frequency) + 1), int(object_count_max) + 1)
        if candidate_most_frequent_partitions(
            object_count=int(value),
            winning_frequency=int(winning_frequency),
            other_repeated_type_count_max=int(other_repeated_type_count_max),
        )
    )
    if not object_support:
        raise ValueError("no feasible object_count values exist for most-frequent-type counting")
    if explicit_object_count is not None:
        object_count = int(explicit_object_count)
    else:
        object_count = int(
            uniform_choice(
                spawn_rng(int(instance_seed), f"{str(selection_namespace)}:object_count"),
                object_support,
            )
        )
    if object_count not in object_support:
        raise ValueError("object_count is outside configured most-frequent support")

    candidates = candidate_most_frequent_partitions(
        object_count=int(object_count),
        winning_frequency=int(winning_frequency),
        other_repeated_type_count_max=int(other_repeated_type_count_max),
    )
    if not candidates:
        raise ValueError("no feasible frequency partition exists")
    singleton_count, repeated_type_multiplicities = uniform_choice(
        spawn_rng(int(instance_seed), f"{str(selection_namespace)}:frequency_partition"),
        tuple(candidates),
    )

    return TypeFrequencySpec(
        object_count=int(object_count),
        singleton_count=int(singleton_count),
        repeated_type_multiplicities=tuple(int(value) for value in repeated_type_multiplicities),
        object_count_probabilities=dict(
            uniform_probability_map(
                object_support,
                selected=int(object_count) if explicit_object_count is not None else None,
            )
        ),
        target_count_probabilities=dict(
            uniform_probability_map(
                answer_support,
                selected=int(winning_frequency) if explicit_target is not None else None,
            )
        ),
        distinct_color_count=1,
    )


__all__ = [
    "bounded_compositions",
    "candidate_most_frequent_partitions",
    "resolve_most_frequent_frequency_spec",
    "resolve_singleton_frequency_spec",
]
