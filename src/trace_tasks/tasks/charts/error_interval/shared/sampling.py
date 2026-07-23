"""Scene-local sampling primitives for error-interval charts."""

from __future__ import annotations

from itertools import cycle, islice
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from trace_tasks.core.sampling import integer_range_choice, shuffled_support, uniform_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.charts.error_interval.shared.defaults import (
    SCENE_NAMESPACE,
    SUPPORTED_SCENE_VARIANTS,
    _GEN_DEFAULTS,
    _RENDER_DEFAULTS,
    support_probability_map,
)
from trace_tasks.tasks.charts.error_interval.shared.state import RGB, _IntervalItem
from trace_tasks.tasks.charts.shared.label_assets import resolve_chart_entity_labels
from trace_tasks.tasks.charts.shared.labeled_chart_variants import resolve_chart_axis_variant
from trace_tasks.tasks.shared.config_defaults import group_default, resolve_required_int_bounds


REFERENCE_PREDICATES: tuple[str, ...] = ("contains", "above", "below")
WIDTH_RANK_KEYS: tuple[str, ...] = ("widest", "narrowest", "second_widest", "second_narrowest")


def resolve_scene_variant(params: Mapping[str, Any], *, instance_seed: int) -> Tuple[str, Dict[str, float]]:
    """Sample the scene render variant without exposing it as a public query."""

    return resolve_chart_axis_variant(
        params=params,
        gen_defaults=_GEN_DEFAULTS,
        instance_seed=int(instance_seed),
        supported_variants=SUPPORTED_SCENE_VARIANTS,
        task_id=SCENE_NAMESPACE,
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        axis_namespace="scene_variant",
    )


def sample_int_range(
    params: Mapping[str, Any],
    *,
    min_key: str,
    max_key: str,
    fallback_min: int,
    fallback_max: int,
    instance_seed: int,
    namespace: str,
) -> Tuple[int, Dict[str, float]]:
    """Sample one integer from a configured inclusive support."""

    low, high = resolve_required_int_bounds(
        params,
        _GEN_DEFAULTS,
        min_key=str(min_key),
        max_key=str(max_key),
        fallback_min=int(fallback_min),
        fallback_max=int(fallback_max),
        context=f"generation defaults for {SCENE_NAMESPACE}",
    )
    support = list(range(int(low), int(high) + 1))
    selected, _probabilities = integer_range_choice(
        spawn_rng(int(instance_seed), str(namespace)),
        int(low),
        int(high),
    )
    return int(selected), support_probability_map(support)


def sample_category_count(params: Mapping[str, Any], *, instance_seed: int) -> tuple[int, dict[str, float]]:
    """Sample the number of categories displayed in the interval chart."""

    return sample_int_range(
        params,
        min_key="category_count_min",
        max_key="category_count_max",
        fallback_min=6,
        fallback_max=10,
        instance_seed=int(instance_seed),
        namespace="charts.error_interval.category_count",
    )


def choose_labels(*, count: int, instance_seed: int) -> List[str]:
    """Sample compact chart labels for the visible category rows or columns."""

    rng = spawn_rng(int(instance_seed), "charts.error_interval.labels")
    labels = resolve_chart_entity_labels(
        rng,
        count=int(count),
        min_chars=2,
        max_chars=7,
        allow_spaces=False,
    ).labels
    return [str(label) for label in labels]


def palette(params: Mapping[str, Any], *, count: int, instance_seed: int) -> List[RGB]:
    """Sample the interval mark palette used for the category marks."""

    raw_palette = params.get("interval_palette_rgb", group_default(_RENDER_DEFAULTS, "interval_palette_rgb", []))
    resolved: List[RGB] = []
    if isinstance(raw_palette, Sequence):
        for raw in raw_palette:
            if isinstance(raw, Sequence) and len(raw) == 3:
                resolved.append(tuple(int(channel) for channel in raw))
    if not resolved:
        resolved = [
            (37, 99, 235),
            (220, 84, 45),
            (16, 132, 96),
            (139, 92, 246),
            (202, 138, 4),
            (14, 116, 144),
            (190, 58, 90),
        ]
    shuffled = shuffled_support(
        spawn_rng(int(instance_seed), "charts.error_interval.palette"),
        tuple(resolved),
    )
    return list(islice(cycle(shuffled), int(count)))


def sample_title(params: Mapping[str, Any], *, instance_seed: int) -> str:
    """Sample a non-semantic chart title from the configured title pool."""

    title_options = [
        str(value)
        for value in params.get(
            "title_options",
            group_default(_RENDER_DEFAULTS, "title_options", ["Estimate Intervals"]),
        )
    ] or ["Estimate Intervals"]
    return str(
        uniform_choice(
            spawn_rng(int(instance_seed), "charts.error_interval.title"),
            tuple(title_options),
        )
    )


def sample_reference_value(params: Mapping[str, Any], *, instance_seed: int, namespace: str) -> int:
    """Sample the numeric reference value used by reference-count tasks."""

    ref_min, ref_max = resolve_required_int_bounds(
        params,
        _GEN_DEFAULTS,
        min_key="reference_value_min",
        max_key="reference_value_max",
        fallback_min=38,
        fallback_max=62,
        context=f"generation defaults for {SCENE_NAMESPACE}",
    )
    selected, _probabilities = integer_range_choice(
        spawn_rng(int(instance_seed), str(namespace)),
        int(ref_min),
        int(ref_max),
    )
    return int(selected)


def clamp_interval(lower: int, midpoint: int, upper: int) -> Tuple[int, int, int]:
    """Clamp one lower/midpoint/upper interval into the chart axis support."""

    lower = max(0, min(100, int(lower)))
    upper = max(0, min(100, int(upper)))
    if int(lower) > int(upper):
        lower, upper = upper, lower
    midpoint = max(int(lower), min(int(upper), int(midpoint)))
    return int(lower), int(midpoint), int(upper)


def interval_around(midpoint: int, width: int) -> Tuple[int, int, int]:
    """Return a clamped interval centered as closely as possible on midpoint."""

    width = max(2, int(width))
    half_low = int(width // 2)
    lower = int(midpoint) - int(half_low)
    upper = int(lower) + int(width)
    if lower < 0:
        upper -= lower
        lower = 0
    if upper > 100:
        lower -= int(upper) - 100
        upper = 100
    midpoint = int(round((int(lower) + int(upper)) / 2.0))
    return clamp_interval(int(lower), int(midpoint), int(upper))


def construct_reference_intervals(
    *,
    predicate: str,
    category_count: int,
    answer_count: int,
    labels: Sequence[str],
    colors: Sequence[RGB],
    params: Mapping[str, Any],
    instance_seed: int,
) -> Tuple[Tuple[_IntervalItem, ...], int, Tuple[str, ...], Dict[str, Any]]:
    """Construct intervals whose reference predicate matches exactly the target rows."""

    predicate_text = str(predicate)
    if predicate_text not in REFERENCE_PREDICATES:
        raise ValueError(f"unsupported reference predicate: {predicate_text}")
    rng = spawn_rng(int(instance_seed), f"charts.error_interval.reference.{predicate_text}")
    reference_value = sample_reference_value(
        params,
        instance_seed=int(instance_seed),
        namespace=f"charts.error_interval.reference_value.{predicate_text}",
    )
    target_indices = set(rng.sample(range(int(category_count)), k=int(answer_count)))

    items: List[_IntervalItem] = []
    annotation_ids: List[str] = []
    for index in range(int(category_count)):
        item_id = f"i{index}"
        is_target = int(index) in target_indices
        if predicate_text == "contains":
            if is_target:
                half_width = int(rng.randint(6, 15))
                mid = int(reference_value + rng.randint(-3, 3))
                lower = min(int(reference_value), int(mid - half_width))
                upper = max(int(reference_value), int(mid + half_width))
                annotation_ids.append(item_id)
            elif (index + int(reference_value)) % 2 == 0:
                upper = int(reference_value) - int(rng.randint(3, 13))
                width = int(rng.randint(8, 18))
                lower = int(upper) - int(width)
                mid = int(round((int(lower) + int(upper)) / 2.0))
            else:
                lower = int(reference_value) + int(rng.randint(3, 13))
                width = int(rng.randint(8, 18))
                upper = int(lower) + int(width)
                mid = int(round((int(lower) + int(upper)) / 2.0))
        elif predicate_text == "above":
            if is_target:
                lower = int(reference_value) + int(rng.randint(3, 13))
                width = int(rng.randint(8, 18))
                upper = int(lower) + int(width)
                mid = int(round((int(lower) + int(upper)) / 2.0))
                annotation_ids.append(item_id)
            elif (index + int(reference_value)) % 2 == 0:
                half_width = int(rng.randint(6, 14))
                mid = int(reference_value + rng.randint(-2, 2))
                lower = min(int(reference_value), int(mid - half_width))
                upper = max(int(reference_value), int(mid + half_width))
            else:
                upper = int(reference_value) - int(rng.randint(3, 11))
                width = int(rng.randint(8, 16))
                lower = int(upper) - int(width)
                mid = int(round((int(lower) + int(upper)) / 2.0))
        else:
            if is_target:
                upper = int(reference_value) - int(rng.randint(3, 13))
                width = int(rng.randint(8, 18))
                lower = int(upper) - int(width)
                mid = int(round((int(lower) + int(upper)) / 2.0))
                annotation_ids.append(item_id)
            elif (index + int(reference_value)) % 2 == 0:
                half_width = int(rng.randint(6, 14))
                mid = int(reference_value + rng.randint(-2, 2))
                lower = min(int(reference_value), int(mid - half_width))
                upper = max(int(reference_value), int(mid + half_width))
            else:
                lower = int(reference_value) + int(rng.randint(3, 11))
                width = int(rng.randint(8, 16))
                upper = int(lower) + int(width)
                mid = int(round((int(lower) + int(upper)) / 2.0))

        lower, mid, upper = clamp_interval(int(lower), int(mid), int(upper))
        items.append(
            _IntervalItem(
                item_id=item_id,
                label=str(labels[index]),
                lower=int(lower),
                midpoint=int(mid),
                upper=int(upper),
                color_rgb=tuple(int(v) for v in colors[index]),
            )
        )
    return tuple(items), int(reference_value), tuple(annotation_ids), {"answer_count": int(answer_count)}


def construct_width_rank_intervals(
    *,
    rank_key: str,
    category_count: int,
    labels: Sequence[str],
    colors: Sequence[RGB],
    instance_seed: int,
) -> Tuple[Tuple[_IntervalItem, ...], Tuple[str, ...], str, Dict[str, Any]]:
    """Construct intervals with one unique category at the requested width rank."""

    rank_text = str(rank_key)
    if rank_text not in WIDTH_RANK_KEYS:
        raise ValueError(f"unsupported width rank: {rank_text}")
    rng = spawn_rng(int(instance_seed), f"charts.error_interval.relation.{rank_text}")
    winner_index = int(
        uniform_choice(
            spawn_rng(
                int(instance_seed),
                f"charts.error_interval.winner.{rank_text}",
            ),
            tuple(range(int(category_count))),
            sort_keys=True,
        )
    )
    width_low = int(rng.randint(6, 13))
    width_high = min(34, int(width_low) + int(rng.randint(int(category_count) + 5, int(category_count) + 10)))
    width_pool = list(range(int(width_low), int(width_high) + 1))
    rng.shuffle(width_pool)
    widths_sorted = sorted(width_pool[: int(category_count)])
    if rank_text == "widest":
        winner_width = int(widths_sorted[-1])
    elif rank_text == "narrowest":
        winner_width = int(widths_sorted[0])
    elif rank_text == "second_widest":
        winner_width = int(widths_sorted[-2])
    else:
        winner_width = int(widths_sorted[1])
    remaining_widths = [int(width) for width in widths_sorted if int(width) != int(winner_width)]
    rng.shuffle(remaining_widths)
    items: List[_IntervalItem] = []

    for index in range(int(category_count)):
        width = int(winner_width) if int(index) == int(winner_index) else int(remaining_widths.pop())
        midpoint = int(rng.randint(25, 75))
        lower, mid, upper = interval_around(int(midpoint), int(width))
        items.append(
            _IntervalItem(
                item_id=f"i{index}",
                label=str(labels[index]),
                lower=int(lower),
                midpoint=int(mid),
                upper=int(upper),
                color_rgb=tuple(int(v) for v in colors[index]),
            )
        )
    winner_id = f"i{winner_index}"
    return tuple(items), (winner_id,), str(labels[int(winner_index)]), {
        "winner_index": int(winner_index),
        "winner_label": str(labels[int(winner_index)]),
        "winner_width": int(winner_width),
        "width_support": [int(width) for width in widths_sorted],
    }


__all__ = [
    "REFERENCE_PREDICATES",
    "WIDTH_RANK_KEYS",
    "choose_labels",
    "construct_reference_intervals",
    "construct_width_rank_intervals",
    "palette",
    "resolve_scene_variant",
    "sample_category_count",
    "sample_int_range",
    "sample_title",
]
