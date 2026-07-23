"""Neutral dashboard metric and selection helpers."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from trace_tasks.core.sampling import normalize_positive_weights, uniform_choice, weighted_choice
from trace_tasks.core.seed import spawn_rng

from .defaults import generation_default
from .state import (
    Category,
    Panel,
    OPTION_LETTERS,
    SCENE_NAMESPACE,
    SUPPORTED_CONDITION_COMPARISONS,
    SUPPORTED_RANK_DIRECTIONS,
)


def weighted_choice_from_defaults(
    rng,
    *,
    params: Mapping[str, Any],
    key: str,
    supported: Sequence[str],
    fallback_weights_key: str,
) -> str:
    explicit = params.get(str(key))
    supported_values = [str(item) for item in supported]
    if explicit is not None:
        selected = str(explicit)
        if selected not in set(supported_values):
            raise ValueError(f"unsupported {key}: {selected}")
        return selected
    raw_weights = params.get(
        str(fallback_weights_key),
        generation_default(str(fallback_weights_key), {value: 1.0 for value in supported_values}),
    )
    if not isinstance(raw_weights, Mapping):
        raise ValueError(f"{fallback_weights_key} must be a mapping when provided")
    probabilities = normalize_positive_weights(
        {str(name): float(weight) for name, weight in raw_weights.items() if str(name) in set(supported_values)},
        default_keys=supported_values,
    )
    return str(weighted_choice(rng, probabilities, sort_keys=True))


def rank_support(params: Mapping[str, Any]) -> Tuple[int, ...]:
    raw = params.get("rank_n_support", generation_default("rank_n_support", [1, 2, 3]))
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("rank_n_support must be a sequence")
    values = sorted({int(value) for value in raw if int(value) >= 1})
    if not values:
        raise ValueError("rank_n_support must contain at least one positive integer")
    return tuple(values)


def condition_count_support(params: Mapping[str, Any], category_count: int) -> Tuple[int, ...]:
    raw = params.get("condition_count_support", generation_default("condition_count_support", [1, 2, 3, 4, 5]))
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("condition_count_support must be a sequence")
    values = sorted({int(value) for value in raw if 0 <= int(value) <= int(category_count)})
    if not values:
        raise ValueError("condition_count_support has no values feasible for category count")
    return tuple(values)


def panel_condition_count_support(params: Mapping[str, Any], panel_count: int) -> Tuple[int, ...]:
    raw = params.get("panel_condition_count_support", generation_default("panel_condition_count_support", [1, 2, 3, 4, 5]))
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("panel_condition_count_support must be a sequence")
    values = sorted({int(value) for value in raw if 0 <= int(value) <= int(panel_count)})
    if not values:
        raise ValueError("panel_condition_count_support has no values feasible for panel count")
    return tuple(values)


def panel_value_range_support(
    params: Mapping[str, Any],
    *,
    value_min: int,
    value_max: int,
    explicit_keys: Sequence[str] = ("target_answer", "range_value"),
) -> Tuple[int, ...]:
    explicit = None
    for key in explicit_keys:
        if str(key) in params:
            explicit = params[str(key)]
            break
    raw_support = params.get("panel_value_range_support", generation_default("panel_value_range_support", tuple(range(12, 71))))
    if isinstance(raw_support, Sequence) and not isinstance(raw_support, (str, bytes)):
        support = tuple(sorted({int(value) for value in raw_support if 2 <= int(value) <= int(value_max) - int(value_min)}))
    else:
        support = tuple(value for value in range(12, 71) if int(value) <= int(value_max) - int(value_min))
    if explicit is not None:
        value = int(explicit)
        if value not in set(support):
            raise ValueError("target range value is outside configured panel_value_range_support")
        return (int(value),)
    if not support:
        raise ValueError("panel_value_range_support has no feasible values for configured value range")
    return support


def balanced_support_choice(params: Mapping[str, Any], *, instance_seed: int, namespace: str, support: Sequence[int]) -> int:
    values = tuple(int(value) for value in support)
    if not values:
        raise ValueError("support must contain at least one value")
    return int(
        uniform_choice(
            spawn_rng(int(instance_seed), str(namespace)),
            values,
            sort_keys=True,
        )
    )


def bounded_integer_partition(
    rng,
    *,
    count: int,
    total: int,
    value_min: int,
    value_max: int,
) -> Tuple[int, ...]:
    """Return ``count`` shuffled integers in bounds whose sum is ``total``."""

    count = int(count)
    value_min = int(value_min)
    value_max = int(value_max)
    total = int(total)
    if count <= 0:
        raise ValueError("count must be positive")
    if value_min > value_max:
        raise ValueError("value_min must be <= value_max")
    min_total = int(count * value_min)
    max_total = int(count * value_max)
    if total < min_total or total > max_total:
        raise ValueError("total is outside feasible bounded partition range")
    remaining = int(total - min_total)
    capacity = int(value_max - value_min)
    values = [int(value_min) for _ in range(count)]
    remaining_capacities = [int(capacity) for _ in range(count)]
    for index in range(count - 1):
        future_capacity = int(sum(remaining_capacities[index + 1 :]))
        min_take = max(0, int(remaining) - int(future_capacity))
        max_take = min(int(remaining_capacities[index]), int(remaining))
        take = int(rng.randint(int(min_take), int(max_take)))
        values[index] += int(take)
        remaining -= int(take)
    values[-1] += int(remaining)
    rng.shuffle(values)
    return tuple(int(value) for value in values)


def assign_unique_totals(
    rng,
    *,
    item_ids: Sequence[str],
    total_min: int,
    total_max: int,
    answer_item_id: str,
    direction: str,
) -> Dict[str, int]:
    """Assign unique totals so ``answer_item_id`` is the requested extremum."""

    ids = tuple(str(item_id) for item_id in item_ids)
    if not ids:
        raise ValueError("item_ids must not be empty")
    if str(answer_item_id) not in set(ids):
        raise ValueError("answer_item_id must be one of item_ids")
    if str(direction) not in {"largest", "smallest"}:
        raise ValueError("direction must be largest or smallest")
    support = range(int(total_min), int(total_max) + 1)
    if len(support) < len(ids):
        raise ValueError("total range must contain enough unique values")
    sampled = sorted(int(value) for value in rng.sample(list(support), len(ids)))
    answer_total = int(sampled[-1] if str(direction) == "largest" else sampled[0])
    other_totals = list(sampled[:-1] if str(direction) == "largest" else sampled[1:])
    rng.shuffle(other_totals)
    assigned: Dict[str, int] = {str(answer_item_id): int(answer_total)}
    other_index = 0
    for item_id in ids:
        if str(item_id) == str(answer_item_id):
            continue
        assigned[str(item_id)] = int(other_totals[other_index])
        other_index += 1
    return dict(assigned)


def join_labels(values: Sequence[str]) -> str:
    labels = [str(value) for value in values]
    if not labels:
        return ""
    if len(labels) == 1:
        return labels[0]
    if len(labels) == 2:
        return f"{labels[0]} and {labels[1]}"
    return f"{', '.join(labels[:-1])}, and {labels[-1]}"


def join_quoted_labels(values: Sequence[str]) -> str:
    return join_labels([f'"{str(value)}"' for value in values])


def option_count_support(params: Mapping[str, Any]) -> Tuple[int, ...]:
    explicit = params.get("option_count")
    if explicit is not None:
        count = int(explicit)
        if count not in {4, 6}:
            raise ValueError("option_count must be either 4 or 6 for dashboard statement options")
        return (int(count),)
    raw_support = params.get("statement_option_count_support", generation_default("statement_option_count_support", (4, 6)))
    if isinstance(raw_support, Sequence) and not isinstance(raw_support, (str, bytes)):
        support = tuple(sorted({int(value) for value in raw_support if int(value) in {4, 6}}))
    else:
        support = ()
    if not support:
        raise ValueError("statement option count support is empty")
    return support


def rank_phrase(direction: str, rank_n: int) -> str:
    if int(rank_n) == 1:
        return "largest" if str(direction) == "largest" else "smallest"
    prefix = {2: "second", 3: "third", 4: "fourth"}.get(int(rank_n), f"{int(rank_n)}th")
    suffix = "largest" if str(direction) == "largest" else "smallest"
    return f"{prefix} {suffix}"


def condition_phrase(comparison: str, threshold: int) -> str:
    if str(comparison) == "greater_than":
        return f"greater than {int(threshold)}"
    if str(comparison) == "less_than":
        return f"less than {int(threshold)}"
    raise ValueError(f"unsupported comparison: {comparison}")


def panel_by_id(panels: Sequence[Panel], panel_id: str) -> Panel:
    for panel in panels:
        if str(panel.panel_id) == str(panel_id):
            return panel
    raise KeyError(panel_id)


def category_by_id(categories: Sequence[Category], category_id: str) -> Category:
    for category in categories:
        if str(category.category_id) == str(category_id):
            return category
    raise KeyError(category_id)


def ranked_category_id(*, categories: Sequence[Category], panel: Panel, direction: str, rank_n: int) -> str:
    reverse = str(direction) == "largest"
    ordered = sorted(categories, key=lambda category: int(panel.values_by_category_id[str(category.category_id)]), reverse=bool(reverse))
    if int(rank_n) < 1 or int(rank_n) > len(ordered):
        raise ValueError("rank_n is outside category support")
    return str(ordered[int(rank_n) - 1].category_id)


def rank_positions_by_category_id(*, categories: Sequence[Category], panel: Panel, direction: str) -> Dict[str, int]:
    reverse = str(direction) == "largest"
    ordered = sorted(categories, key=lambda category: int(panel.values_by_category_id[str(category.category_id)]), reverse=bool(reverse))
    return {str(category.category_id): int(index + 1) for index, category in enumerate(ordered)}


def compare_condition(value: int, comparison: str, threshold: int) -> bool:
    if str(comparison) == "greater_than":
        return int(value) > int(threshold)
    if str(comparison) == "less_than":
        return int(value) < int(threshold)
    raise ValueError(f"unsupported comparison: {comparison}")


def choose_threshold_pair_for_count(
    *,
    rng,
    categories: Sequence[Category],
    first_panel: Panel,
    second_panel: Panel,
    first_comparison: str,
    second_comparison: str,
    target_count: int,
    value_min: int,
    value_max: int,
) -> Tuple[int, int, Tuple[str, ...]]:
    candidates: List[Tuple[int, int, Tuple[str, ...]]] = []
    for first_threshold in range(int(value_min) + 2, int(value_max) - 1):
        for second_threshold in range(int(value_min) + 2, int(value_max) - 1):
            matches = tuple(
                str(category.category_id)
                for category in categories
                if compare_condition(int(first_panel.values_by_category_id[str(category.category_id)]), str(first_comparison), int(first_threshold))
                and compare_condition(int(second_panel.values_by_category_id[str(category.category_id)]), str(second_comparison), int(second_threshold))
            )
            if len(matches) == int(target_count):
                candidates.append((int(first_threshold), int(second_threshold), matches))
    if not candidates:
        raise ValueError("no threshold pair realizes the requested condition count")
    return candidates[int(rng.randrange(len(candidates)))]


def choose_rank_params(rng, *, params: Mapping[str, Any], category_count: int) -> Tuple[str, int]:
    direction = weighted_choice_from_defaults(
        rng,
        params=params,
        key="rank_direction",
        supported=SUPPORTED_RANK_DIRECTIONS,
        fallback_weights_key="rank_direction_weights",
    )
    support = tuple(value for value in rank_support(params) if int(value) <= int(category_count))
    if not support:
        raise ValueError("rank support has no feasible values for category count")
    rank_n = int(support[int(rng.randrange(len(support)))])
    return str(direction), int(rank_n)


def choose_distinct_rank_params(rng, *, params: Mapping[str, Any], category_count: int, avoid_phrase: str) -> Tuple[str, int]:
    """Choose rank parameters whose visible phrase differs from another phrase."""

    candidates: List[Tuple[str, int]] = []
    raw_weights = params.get("rank_direction_weights", generation_default("rank_direction_weights", {direction: 1.0 for direction in SUPPORTED_RANK_DIRECTIONS}))
    if not isinstance(raw_weights, Mapping):
        raise ValueError("rank_direction_weights must be a mapping when provided")
    feasible_ranks = tuple(value for value in rank_support(params) if int(value) <= int(category_count))
    for direction in SUPPORTED_RANK_DIRECTIONS:
        if float(raw_weights.get(str(direction), 0.0)) <= 0.0:
            continue
        for rank_n in feasible_ranks:
            if rank_phrase(str(direction), int(rank_n)) != str(avoid_phrase):
                candidates.append((str(direction), int(rank_n)))
    if not candidates:
        raise ValueError("no distinct rank phrase is feasible")
    return candidates[int(rng.randrange(len(candidates)))]


def statement_candidate(*, panel_a: Panel, category_a: Category, panel_b: Panel, category_b: Category, comparison: str, statement_kind: str) -> Dict[str, Any]:
    value_a = int(panel_a.values_by_category_id[str(category_a.category_id)])
    value_b = int(panel_b.values_by_category_id[str(category_b.category_id)])
    if str(comparison) == "greater_than":
        truth = bool(value_a > value_b)
        relation = "greater than"
    elif str(comparison) == "less_than":
        truth = bool(value_a < value_b)
        relation = "less than"
    else:
        raise ValueError(f"unsupported statement comparison: {comparison}")
    if str(statement_kind) == "same_category_cross_panel":
        text = f'"{category_a.label}" in "{panel_a.name}" is {relation} in "{panel_b.name}".'
    elif str(statement_kind) == "two_categories_one_panel":
        text = f'In "{panel_a.name}", "{category_a.label}" is {relation} "{category_b.label}".'
    else:
        raise ValueError(f"unsupported statement kind: {statement_kind}")
    return {
        "text": str(text),
        "truth_value": bool(truth),
        "comparison": str(comparison),
        "statement_kind": str(statement_kind),
        "first_panel_id": str(panel_a.panel_id),
        "first_panel_name": str(panel_a.name),
        "first_category_id": str(category_a.category_id),
        "first_category_label": str(category_a.label),
        "first_value": int(value_a),
        "second_panel_id": str(panel_b.panel_id),
        "second_panel_name": str(panel_b.name),
        "second_category_id": str(category_b.category_id),
        "second_category_label": str(category_b.label),
        "second_value": int(value_b),
    }


def statement_option_candidates(*, rng, categories: Sequence[Category], panels: Sequence[Panel]) -> Tuple[Dict[str, Any], ...]:
    candidates: List[Dict[str, Any]] = []
    seen: set[str] = set()
    comparisons = ("greater_than", "less_than")
    panel_pairs = [(panel_a, panel_b) for panel_a in panels for panel_b in panels if str(panel_a.panel_id) != str(panel_b.panel_id)]
    category_pairs = [(category_a, category_b) for category_a in categories for category_b in categories if str(category_a.category_id) != str(category_b.category_id)]
    rng.shuffle(panel_pairs)
    rng.shuffle(category_pairs)
    for panel_a, panel_b in panel_pairs:
        shuffled_categories = list(categories)
        rng.shuffle(shuffled_categories)
        for category in shuffled_categories:
            for comparison in comparisons:
                candidate = statement_candidate(panel_a=panel_a, category_a=category, panel_b=panel_b, category_b=category, comparison=str(comparison), statement_kind="same_category_cross_panel")
                if str(candidate["text"]) not in seen:
                    seen.add(str(candidate["text"]))
                    candidates.append(candidate)
    shuffled_panels = list(panels)
    rng.shuffle(shuffled_panels)
    for panel in shuffled_panels:
        for category_a, category_b in category_pairs:
            for comparison in comparisons:
                candidate = statement_candidate(panel_a=panel, category_a=category_a, panel_b=panel, category_b=category_b, comparison=str(comparison), statement_kind="two_categories_one_panel")
                if str(candidate["text"]) not in seen:
                    seen.add(str(candidate["text"]))
                    candidates.append(candidate)
    rng.shuffle(candidates)
    return tuple(candidates)


__all__ = [
    "assign_unique_totals",
    "balanced_support_choice",
    "bounded_integer_partition",
    "category_by_id",
    "choose_distinct_rank_params",
    "choose_rank_params",
    "choose_threshold_pair_for_count",
    "compare_condition",
    "condition_count_support",
    "condition_phrase",
    "join_labels",
    "join_quoted_labels",
    "option_count_support",
    "panel_value_range_support",
    "panel_by_id",
    "panel_condition_count_support",
    "rank_phrase",
    "rank_positions_by_category_id",
    "rank_support",
    "ranked_category_id",
    "statement_option_candidates",
    "weighted_choice_from_defaults",
]
