"""Sampling helpers for named-ring icon scenes."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from ....shared.config_defaults import group_default
from ....shared.deterministic_sampling import uniform_probability_map
from ...shared.procedural_named_icons import (
    PROCEDURAL_NAMED_ICON_SHAPES,
    procedural_named_icon_display_name,
    validate_procedural_named_icon_fill_style_support,
)

from .defaults import NamedRingDefaults
from .state import RingArcPlan


def string_probability_map(values: Sequence[str], *, selected: str | None = None) -> Dict[str, float]:
    """Return a JSON-stable probability map over string support."""

    support = tuple(str(value) for value in values)
    if not support:
        return {}
    if selected is not None:
        return {str(value): (1.0 if str(value) == str(selected) else 0.0) for value in support}
    probability = 1.0 / float(len(support))
    return {str(value): float(probability) for value in support}


def int_bounds(
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    *,
    low_key: str,
    high_key: str,
    fallback_low: int,
    fallback_high: int,
) -> Tuple[int, int]:
    """Resolve inclusive integer bounds from params/defaults/fallbacks."""

    low = int(params.get(str(low_key), group_default(gen_defaults, str(low_key), int(fallback_low))))
    high = int(params.get(str(high_key), group_default(gen_defaults, str(high_key), int(fallback_high))))
    if int(low) < 0 or int(high) < int(low):
        raise ValueError(f"invalid {low_key}/{high_key} bounds")
    return int(low), int(high)


def shape_support(params: Mapping[str, Any], gen_defaults: Mapping[str, Any]) -> Tuple[str, ...]:
    """Return supported procedural shape ids for this scene."""

    raw = params.get("shape_id_support", group_default(gen_defaults, "shape_id_support", PROCEDURAL_NAMED_ICON_SHAPES))
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("shape_id_support must be a sequence")
    values = tuple(dict.fromkeys(str(value).strip() for value in raw if str(value).strip()))
    unsupported = sorted(set(values) - set(PROCEDURAL_NAMED_ICON_SHAPES))
    if unsupported:
        raise ValueError(f"unsupported procedural named icon shapes: {unsupported}")
    if len(values) < 8:
        raise ValueError("named-ring arc task needs at least eight supported named shapes")
    return values


def fill_style_support(
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    fallback_defaults: NamedRingDefaults,
) -> Tuple[str, ...]:
    """Return supported named-icon fill styles."""

    raw = params.get(
        "named_icon_fill_style_support",
        group_default(gen_defaults, "named_icon_fill_style_support", fallback_defaults.named_icon_fill_style_support),
    )
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raw = fallback_defaults.named_icon_fill_style_support
    return validate_procedural_named_icon_fill_style_support(tuple(str(value) for value in raw))


def fill_style_probability_map(
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    support: Sequence[str],
) -> Dict[str, float]:
    """Return normalized fill-style probabilities over support."""

    raw = params.get(
        "named_icon_fill_style_weights",
        group_default(gen_defaults, "named_icon_fill_style_weights", None),
    )
    if not isinstance(raw, Mapping):
        probability = 1.0 / float(len(tuple(support)))
        return {str(value): float(probability) for value in support}
    weights = {str(value): max(0.0, float(raw.get(str(value), 0.0))) for value in support}
    total = sum(float(value) for value in weights.values())
    if total <= 0.0:
        probability = 1.0 / float(len(tuple(support)))
        return {str(value): float(probability) for value in support}
    return {str(value): float(weights[str(value)]) / float(total) for value in support}


def resolve_target_shape(rng, *, params: Mapping[str, Any], support: Sequence[str]) -> Tuple[str, Dict[str, float]]:
    """Choose the target shape id and record its sampling probabilities."""

    explicit_shape = params.get("shape_id", params.get("target_shape_id"))
    if explicit_shape is not None:
        target_shape_id = str(explicit_shape)
        if target_shape_id not in set(support):
            raise ValueError(f"target shape must be one of {support}")
        return str(target_shape_id), string_probability_map(tuple(str(value) for value in support), selected=str(target_shape_id))
    target_shape_id = str(rng.choice(tuple(str(value) for value in support)))
    return str(target_shape_id), string_probability_map(tuple(str(value) for value in support))


def choose_answer_count(
    rng,
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    fallback_defaults: NamedRingDefaults,
) -> Tuple[int, Dict[str, float]]:
    """Choose the answer count support value."""

    low, high = int_bounds(
        params,
        gen_defaults,
        low_key="answer_count_min",
        high_key="answer_count_max",
        fallback_low=fallback_defaults.answer_count_min,
        fallback_high=fallback_defaults.answer_count_max,
    )
    explicit = params.get("answer_count", params.get("target_count", params.get("answer")))
    support = tuple(range(int(low), int(high) + 1))
    if not support:
        raise ValueError("answer_count support is empty")
    if explicit is not None:
        value = int(explicit)
        if value not in set(support):
            raise ValueError("answer_count is outside configured support")
        return int(value), uniform_probability_map(support, selected=int(value))
    value = int(rng.choice(support))
    return int(value), uniform_probability_map(support)


def choose_ring_icon_count(
    rng,
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    fallback_defaults: NamedRingDefaults,
    answer_count: int,
) -> Tuple[int, Dict[str, float]]:
    """Choose a feasible number of icons around the ring."""

    low, high = int_bounds(
        params,
        gen_defaults,
        low_key="ring_icon_count_min",
        high_key="ring_icon_count_max",
        fallback_low=fallback_defaults.ring_icon_count_min,
        fallback_high=fallback_defaults.ring_icon_count_max,
    )
    low = max(int(low), int(answer_count) + 6)
    explicit = params.get("ring_icon_count", params.get("object_count"))
    support = tuple(range(int(low), int(high) + 1))
    if not support:
        raise ValueError("ring_icon_count support is empty")
    if explicit is not None:
        value = int(explicit)
        if value not in set(support):
            raise ValueError("ring_icon_count is outside configured support")
        return int(value), uniform_probability_map(support, selected=int(value))
    value = int(rng.choice(support))
    return int(value), uniform_probability_map(support)


def arc_indices_between(start_index: int, end_index: int, *, count: int, direction: str) -> Tuple[int, ...]:
    """Return indices strictly between two ring endpoints along one direction."""

    step = 1 if str(direction) == "clockwise" else -1
    values: List[int] = []
    cursor = (int(start_index) + int(step)) % int(count)
    while int(cursor) != int(end_index):
        values.append(int(cursor))
        cursor = (int(cursor) + int(step)) % int(count)
        if len(values) >= int(count):
            raise ValueError("invalid arc endpoints for ring traversal")
    return tuple(values)


def choose_arc(
    rng,
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    fallback_defaults: NamedRingDefaults,
    direction: str,
    ring_icon_count: int,
    answer_count: int,
) -> Tuple[int, int, int, Tuple[int, ...], Dict[str, float]]:
    """Choose a feasible directed arc between endpoint markers."""

    low, high = int_bounds(
        params,
        gen_defaults,
        low_key="arc_span_min",
        high_key="arc_span_max",
        fallback_low=fallback_defaults.arc_span_min,
        fallback_high=fallback_defaults.arc_span_max,
    )
    low = max(int(low), int(answer_count))
    high = min(int(high), int(ring_icon_count) - 4)
    if high < low:
        raise ValueError("arc_span support is empty")
    support = tuple(range(int(low), int(high) + 1))
    explicit_span = params.get("arc_span_count", params.get("arc_length"))
    if explicit_span is not None:
        arc_span_count = int(explicit_span)
        if arc_span_count not in set(support):
            raise ValueError("arc_span_count is outside configured support")
    else:
        arc_span_count = int(rng.choice(support))
    explicit_start = params.get("start_index")
    start_index = int(explicit_start) % int(ring_icon_count) if explicit_start is not None else int(rng.randrange(int(ring_icon_count)))
    if str(direction) == "clockwise":
        end_index = (int(start_index) + int(arc_span_count) + 1) % int(ring_icon_count)
    else:
        end_index = (int(start_index) - int(arc_span_count) - 1) % int(ring_icon_count)
    explicit_end = params.get("end_index")
    if explicit_end is not None:
        end_index = int(explicit_end) % int(ring_icon_count)
        arc_indices = arc_indices_between(int(start_index), int(end_index), count=int(ring_icon_count), direction=str(direction))
        if len(arc_indices) != int(arc_span_count):
            raise ValueError("explicit end_index does not match arc_span_count")
    else:
        arc_indices = arc_indices_between(int(start_index), int(end_index), count=int(ring_icon_count), direction=str(direction))
    return (
        int(start_index),
        int(end_index),
        int(arc_span_count),
        tuple(int(value) for value in arc_indices),
        uniform_probability_map(support, selected=int(arc_span_count) if explicit_span is not None else None),
    )


def choose_off_arc_target_count(
    rng,
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    fallback_defaults: NamedRingDefaults,
    feasible_count: int,
) -> Tuple[int, Dict[str, float]]:
    """Choose how many non-counted target-shape distractors to place."""

    low, high = int_bounds(
        params,
        gen_defaults,
        low_key="off_arc_target_count_min",
        high_key="off_arc_target_count_max",
        fallback_low=fallback_defaults.off_arc_target_count_min,
        fallback_high=fallback_defaults.off_arc_target_count_max,
    )
    high = min(int(high), int(feasible_count))
    low = min(int(low), int(high))
    explicit = params.get("off_arc_target_count")
    support = tuple(range(int(low), int(high) + 1)) if int(high) >= int(low) else (0,)
    if explicit is not None:
        value = int(explicit)
        if value < 0 or value > int(feasible_count):
            raise ValueError("off_arc_target_count is outside feasible support")
        return int(value), uniform_probability_map(support, selected=int(value) if value in set(support) else None)
    if not support:
        return 0, {"0": 1.0}
    value = int(rng.choice(support))
    return int(value), uniform_probability_map(support)


def sample_ring_arc_plan(
    *,
    rng,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    fallback_defaults: NamedRingDefaults,
    direction: str,
) -> RingArcPlan:
    """Sample a directed ring arc and target placements for one question."""

    answer_count, answer_probabilities = choose_answer_count(
        rng,
        params=params,
        gen_defaults=gen_defaults,
        fallback_defaults=fallback_defaults,
    )
    ring_icon_count, ring_icon_count_probabilities = choose_ring_icon_count(
        rng,
        params=params,
        gen_defaults=gen_defaults,
        fallback_defaults=fallback_defaults,
        answer_count=int(answer_count),
    )
    start_index, end_index, arc_span_count, arc_indices, arc_span_probabilities = choose_arc(
        rng,
        params=params,
        gen_defaults=gen_defaults,
        fallback_defaults=fallback_defaults,
        direction=str(direction),
        ring_icon_count=int(ring_icon_count),
        answer_count=int(answer_count),
    )
    shape_values = shape_support(params, gen_defaults)
    target_shape_id, shape_probabilities = resolve_target_shape(rng, params=params, support=shape_values)

    arc_pool = list(int(value) for value in arc_indices)
    rng.shuffle(arc_pool)
    counted_indices = tuple(sorted(arc_pool[: int(answer_count)]))
    blocked = set(arc_indices) | {int(start_index), int(end_index)}
    off_arc_candidates = [index for index in range(int(ring_icon_count)) if int(index) not in blocked]
    off_arc_count, off_arc_probabilities = choose_off_arc_target_count(
        rng,
        params=params,
        gen_defaults=gen_defaults,
        fallback_defaults=fallback_defaults,
        feasible_count=len(off_arc_candidates),
    )
    rng.shuffle(off_arc_candidates)
    off_arc_target_indices = tuple(sorted(int(value) for value in off_arc_candidates[: int(off_arc_count)]))

    target_indices = set(counted_indices) | set(off_arc_target_indices)
    distractor_support = tuple(str(value) for value in shape_values if str(value) != str(target_shape_id))
    shape_ids: List[str] = []
    for index in range(int(ring_icon_count)):
        if int(index) in target_indices:
            shape_ids.append(str(target_shape_id))
        else:
            shape_ids.append(str(rng.choice(distractor_support)))
    for index in (int(start_index), int(end_index)):
        if str(shape_ids[int(index)]) == str(target_shape_id):
            shape_ids[int(index)] = str(rng.choice(distractor_support))

    realized_count = sum(1 for index in arc_indices if str(shape_ids[int(index)]) == str(target_shape_id))
    if int(realized_count) != int(answer_count):
        raise RuntimeError("constructed named-ring arc does not realize requested answer")

    fill_values = fill_style_support(params, gen_defaults, fallback_defaults)
    fill_probabilities = fill_style_probability_map(params, gen_defaults, fill_values)
    return RingArcPlan(
        direction=str(direction),
        target_shape_id=str(target_shape_id),
        target_shape_name=procedural_named_icon_display_name(str(target_shape_id)),
        answer_count=int(answer_count),
        ring_icon_count=int(ring_icon_count),
        arc_span_count=int(arc_span_count),
        start_index=int(start_index),
        end_index=int(end_index),
        arc_indices=tuple(int(value) for value in arc_indices),
        counted_indices=tuple(int(value) for value in counted_indices),
        off_arc_target_indices=tuple(int(value) for value in off_arc_target_indices),
        shape_ids_by_index=tuple(str(value) for value in shape_ids),
        answer_probabilities=dict(answer_probabilities),
        ring_icon_count_probabilities=dict(ring_icon_count_probabilities),
        arc_span_probabilities=dict(arc_span_probabilities),
        off_arc_target_count_probabilities=dict(off_arc_probabilities),
        shape_probabilities=dict(shape_probabilities),
        fill_style_support=tuple(fill_values),
        fill_style_probabilities=dict(fill_probabilities),
    )
