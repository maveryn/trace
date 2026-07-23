"""Sampling helpers for straight 3D conveyor belt scenes."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.three_d.shared.camera_projection import (
    CameraSpec,
    build_projection_frame,
    project_screen,
    vec_cross,
    vec_norm,
    vec_sub,
)
from trace_tasks.tasks.three_d.shared.projected_object_geometry import object_reference_points
from trace_tasks.tasks.three_d.shared.object_confusions import compatible_distractor_pool
from trace_tasks.tasks.three_d.shared.semantic_colors import sample_readout_palette as sample_semantic_readout_palette
from trace_tasks.tasks.three_d.shared.task_support import (
    resolve_axis_variant_for_namespace,
    resolve_count_for_namespace,
    shuffled_repeated_support,
)

from .state import (
    CONVEYOR_COLOR_READOUT_SHAPE_TYPES,
    CONVEYOR_OBJECT_SHAPE_TYPES,
    HORIZONTAL_LANE_CENTER_BY_KEY,
    HORIZONTAL_LANE_KEYS,
    HORIZONTAL_LANE_LENGTH,
    HORIZONTAL_SLOT_LENGTH,
    LANE_SLOT_JITTER_ACROSS,
    LANE_SLOT_JITTER_ALONG,
    LANE_HALF_WIDTH,
    LANE_LABELS,
    SCENE_ID,
    SEMANTIC_COLOR_RGB,
    SEMANTIC_COLOR_SUPPORT,
    SUPPORTED_SCENE_VARIANTS,
    VERTICAL_LANE_CENTER_BY_KEY,
    VERTICAL_LANE_KEYS,
    VERTICAL_LANE_LENGTH,
    VERTICAL_SLOT_LENGTH,
    object_dimensions,
    public_object_name,
    public_object_plural,
    sample_visual_color_names,
    semantic_color_label,
)


PREDICATE_BELT_TOTAL = "belt_total"
PREDICATE_OBJECT_TYPE = "object_type"
PREDICATE_COLOR = "color"
PREDICATE_COLOR_TYPE = "color_type"
PREDICATE_OBJECT_TYPE_ARITHMETIC = "object_type_count_arithmetic"
PREDICATE_ORDERED_OBJECT_PAIR = "ordered_object_pair"
PREDICATE_ORDERED_COLOR_PAIR = "ordered_color_pair"
PREDICATE_OBJECT_TYPE_TRANSFER = "object_type_transfer"
PREDICATE_COLOR_TRANSFER = "color_transfer"
PREDICATE_BETWEEN_OBJECT_ANCHORS = "between_object_anchors"
ARITHMETIC_SUM = "sum"
ARITHMETIC_DIFFERENCE = "difference"
TRANSFER_OPERATION = "move_to_destination"
LAYOUT_HORIZONTAL = "horizontal_lanes"
LAYOUT_VERTICAL = "vertical_lanes"


@dataclass(frozen=True)
class ResolvedConveyorAxes:
    """Resolved straight conveyor scene axes for one generated instance."""

    scene_variant: str
    scene_variant_probabilities: Dict[str, float]


def _uniform_string_probability_map(values: Sequence[str], *, selected: str | None = None) -> Dict[str, float]:
    support = tuple(str(value) for value in values)
    if selected is not None:
        return {str(value): (1.0 if str(value) == str(selected) else 0.0) for value in support}
    probability = 1.0 / float(max(1, len(support)))
    return {str(value): float(probability) for value in support}


def resolve_conveyor_axes(
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
) -> ResolvedConveyorAxes:
    """Resolve the straight conveyor scene variant."""

    scene_variant, scene_probabilities = resolve_axis_variant_for_namespace(
        params,
        namespace=f"{namespace}.scene_variant",
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        supported_variants=SUPPORTED_SCENE_VARIANTS,
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
    )
    return ResolvedConveyorAxes(
        scene_variant=str(scene_variant),
        scene_variant_probabilities=dict(scene_probabilities),
    )


def _resolve_layout_orientation(
    *,
    params: Mapping[str, Any],
    rng: Any,
    render_params: Any,
) -> tuple[str, Dict[str, float]]:
    support = (LAYOUT_HORIZONTAL, LAYOUT_VERTICAL)
    width = int(render_params.canvas_width)
    height = int(render_params.canvas_height)
    if width > height:
        return LAYOUT_HORIZONTAL, {LAYOUT_HORIZONTAL: 1.0, LAYOUT_VERTICAL: 0.0}
    if height > width:
        return LAYOUT_VERTICAL, {LAYOUT_HORIZONTAL: 0.0, LAYOUT_VERTICAL: 1.0}
    explicit = params.get("layout_orientation")
    if explicit is not None:
        orientation = str(explicit)
        if orientation not in set(support):
            raise ValueError(f"unsupported layout_orientation: {orientation}")
        return orientation, _uniform_string_probability_map(support, selected=orientation)
    orientation = str(support[int(rng.randrange(len(support)))])
    return orientation, _uniform_string_probability_map(support)


def _lane_keys_for_orientation(layout_orientation: str) -> Tuple[str, ...]:
    return HORIZONTAL_LANE_KEYS if str(layout_orientation) == LAYOUT_HORIZONTAL else VERTICAL_LANE_KEYS


def _resolve_target_lane(
    *,
    params: Mapping[str, Any],
    rng: Any,
    lane_keys: Sequence[str],
) -> tuple[str, Dict[str, float]]:
    support = tuple(str(key) for key in lane_keys)
    explicit = params.get("target_lane_key", params.get("target_belt_key"))
    if explicit is not None:
        lane_key = str(explicit)
        if lane_key not in set(support):
            raise ValueError(f"unsupported target_lane_key for layout: {lane_key}")
        return lane_key, _uniform_string_probability_map(support, selected=lane_key)
    lane_key = str(support[int(rng.randrange(len(support)))])
    return lane_key, _uniform_string_probability_map(support)


def _resolve_shape(
    *,
    params: Mapping[str, Any],
    rng: Any,
    support: Sequence[str] = CONVEYOR_OBJECT_SHAPE_TYPES,
) -> tuple[str, Dict[str, float]]:
    support = tuple(str(shape) for shape in support)
    explicit = params.get("target_shape_type")
    if explicit is not None:
        shape = str(explicit)
        if shape not in set(support):
            raise ValueError(f"unsupported target_shape_type: {shape}")
        return shape, _uniform_string_probability_map(support, selected=shape)
    shape = str(support[int(rng.randrange(len(support)))])
    return shape, _uniform_string_probability_map(support)


def _resolve_target_color(
    *,
    params: Mapping[str, Any],
    rng: Any,
) -> tuple[str, Dict[str, float]]:
    support = tuple(str(color) for color in SEMANTIC_COLOR_SUPPORT)
    explicit = params.get("target_color_name")
    if explicit is not None:
        color = str(explicit)
        if color not in set(support):
            raise ValueError(f"unsupported target_color_name: {color}")
        return color, _uniform_string_probability_map(support, selected=color)
    color = str(support[int(rng.randrange(len(support)))])
    return color, _uniform_string_probability_map(support)


def _sample_readout_palette(rng: Any, *, target_color: str, size: int = 4) -> tuple[str, ...]:
    return sample_semantic_readout_palette(
        rng,
        target_color=str(target_color),
        support=SEMANTIC_COLOR_SUPPORT,
        size=int(size),
    )


def _resolve_lane_count(
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
    key: str,
) -> tuple[int, Dict[str, float]]:
    count, probabilities = resolve_count_for_namespace(
        params,
        namespace=str(namespace),
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        key=str(key),
        default_min=1,
        default_max=8,
        lower=1,
        upper=8,
    )
    return int(count), dict(probabilities)


def _resolve_scoped_target_count(
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
) -> tuple[int, Dict[str, float]]:
    count, probabilities = resolve_count_for_namespace(
        params,
        namespace=str(namespace),
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        key="target_count",
        default_min=0,
        default_max=5,
        lower=0,
        upper=5,
    )
    return int(count), dict(probabilities)


def _resolve_ordered_pair_count(
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
) -> tuple[int, Dict[str, float]]:
    count, probabilities = resolve_count_for_namespace(
        params,
        namespace=str(namespace),
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        key="target_count",
        default_min=int(params.get("target_count_min", group_default(gen_defaults, "target_count_min", 0))),
        default_max=int(params.get("target_count_max", group_default(gen_defaults, "target_count_max", 4))),
        lower=0,
        upper=4,
    )
    return int(count), dict(probabilities)


def _resolve_between_items_count(
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
) -> tuple[int, Dict[str, float]]:
    return resolve_count_for_namespace(
        params,
        namespace=str(namespace),
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        key="target_count",
        default_min=int(params.get("target_count_min", group_default(gen_defaults, "target_count_min", 1))),
        default_max=int(params.get("target_count_max", group_default(gen_defaults, "target_count_max", 5))),
        lower=1,
        upper=5,
    )


def _resolve_transfer_answer_and_counts(
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    rng: Any,
    instance_seed: int,
    namespace: str,
    destination_cap: int,
) -> tuple[int, int, int, Dict[str, float], Dict[str, float]]:
    """Resolve final count, moved count, and original destination count."""

    answer_value, answer_probabilities = resolve_count_for_namespace(
        params,
        namespace=f"{namespace}.answer_value",
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        key="answer_value",
        default_min=int(params.get("answer_value_min", group_default(gen_defaults, "answer_value_min", 2))),
        default_max=int(params.get("answer_value_max", group_default(gen_defaults, "answer_value_max", 12))),
        lower=2,
        upper=12,
    )
    moved_explicit = params.get("moved_count")
    destination_explicit = params.get("destination_existing_count")
    if moved_explicit is not None or destination_explicit is not None:
        if moved_explicit is None or destination_explicit is None:
            raise ValueError("moved_count and destination_existing_count must be provided together")
        moved_count = int(moved_explicit)
        destination_existing_count = int(destination_explicit)
        if not (1 <= moved_count <= 4):
            raise ValueError("moved_count must be in 1..4")
        if not (1 <= destination_existing_count <= int(destination_cap)):
            raise ValueError("destination_existing_count exceeds lane cap")
        if int(moved_count + destination_existing_count) != int(answer_value):
            raise ValueError("explicit transfer counts do not match answer_value")
        return (
            int(answer_value),
            int(moved_count),
            int(destination_existing_count),
            dict(answer_probabilities),
            {f"{moved_count},{destination_existing_count}": 1.0},
        )

    pairs = [
        (moved_count, int(answer_value) - moved_count)
        for moved_count in range(1, 5)
        if 1 <= int(answer_value) - moved_count <= int(destination_cap)
    ]
    if not pairs:
        raise ValueError(f"no conveyor transfer operands for answer {answer_value}")
    moved_count, destination_existing_count = pairs[int(rng.randrange(len(pairs)))]
    return (
        int(answer_value),
        int(moved_count),
        int(destination_existing_count),
        dict(answer_probabilities),
        {f"{moved_count},{destination_existing_count}": 1.0},
    )


def _resolve_arithmetic_operand_counts(
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    rng: Any,
    instance_seed: int,
    namespace: str,
    operation: str,
) -> tuple[int, int, int, Dict[str, float], Dict[str, float]]:
    """Resolve two lane operand counts and the arithmetic answer."""

    operation = str(operation)
    if operation == ARITHMETIC_SUM:
        default_min = int(params.get("sum_answer_min", group_default(gen_defaults, "sum_answer_min", 1)))
        default_max = int(params.get("sum_answer_max", group_default(gen_defaults, "sum_answer_max", 12)))
        lower, upper = 1, 12
    elif operation == ARITHMETIC_DIFFERENCE:
        default_min = int(params.get("difference_answer_min", group_default(gen_defaults, "difference_answer_min", 1)))
        default_max = int(params.get("difference_answer_max", group_default(gen_defaults, "difference_answer_max", 5)))
        lower, upper = 1, 5
    else:
        raise ValueError(f"unsupported conveyor count arithmetic operation: {operation}")
    answer_value, answer_probabilities = resolve_count_for_namespace(
        params,
        namespace=f"{namespace}.{operation}.answer_value",
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        key="answer_value",
        default_min=int(default_min),
        default_max=int(default_max),
        lower=int(lower),
        upper=int(upper),
    )
    explicit_first = params.get("first_scope_count")
    explicit_second = params.get("second_scope_count")
    if explicit_first is not None or explicit_second is not None:
        if explicit_first is None or explicit_second is None:
            raise ValueError("first_scope_count and second_scope_count must be provided together")
        first_count = int(explicit_first)
        second_count = int(explicit_second)
        if not (0 <= first_count <= 6 and 0 <= second_count <= 6):
            raise ValueError("conveyor arithmetic operand counts must be in 0..6")
        expected = first_count + second_count if operation == ARITHMETIC_SUM else abs(first_count - second_count)
        if int(expected) != int(answer_value):
            raise ValueError("explicit conveyor arithmetic operands do not match answer_value")
        return int(first_count), int(second_count), int(answer_value), dict(answer_probabilities), {"first_scope_count": 1.0, "second_scope_count": 1.0}

    if operation == ARITHMETIC_SUM:
        pairs = [(a, int(answer_value) - a) for a in range(0, 7) if 0 <= int(answer_value) - a <= 6]
    else:
        pairs = [(a, b) for a in range(0, 7) for b in range(0, 7) if abs(a - b) == int(answer_value)]
    if not pairs:
        raise ValueError(f"no conveyor arithmetic operands for answer {answer_value}")
    first_count, second_count = pairs[int(rng.randrange(len(pairs)))]
    return int(first_count), int(second_count), int(answer_value), dict(answer_probabilities), {f"{first_count},{second_count}": 1.0}


def _resolve_arithmetic_lanes(
    *,
    params: Mapping[str, Any],
    rng: Any,
    lane_keys: Sequence[str],
) -> tuple[tuple[str, str], Dict[str, float]]:
    support = tuple(str(key) for key in lane_keys)
    explicit_first = params.get("first_lane_key")
    explicit_second = params.get("second_lane_key")
    if explicit_first is not None or explicit_second is not None:
        if explicit_first is None or explicit_second is None:
            raise ValueError("first_lane_key and second_lane_key must be provided together")
        first = str(explicit_first)
        second = str(explicit_second)
        if first == second or first not in set(support) or second not in set(support):
            raise ValueError("conveyor arithmetic lanes must be two distinct visible lane keys")
        return (first, second), {str(key): (1.0 if str(key) in {first, second} else 0.0) for key in support}
    pairs = [(support[i], support[j]) for i in range(len(support)) for j in range(i + 1, len(support))]
    selected = pairs[int(rng.randrange(len(pairs)))]
    return (str(selected[0]), str(selected[1])), {str(key): (1.0 if str(key) in set(selected) else 0.0) for key in support}


def _resolve_transfer_lanes(
    *,
    params: Mapping[str, Any],
    rng: Any,
    lane_keys: Sequence[str],
) -> tuple[str, str, Dict[str, float], Dict[str, float]]:
    support = tuple(str(key) for key in lane_keys)
    source_explicit = params.get("source_lane_key", params.get("source_belt_key"))
    destination_explicit = params.get("destination_lane_key", params.get("destination_belt_key"))
    if source_explicit is not None or destination_explicit is not None:
        if source_explicit is None or destination_explicit is None:
            raise ValueError("source_lane_key and destination_lane_key must be provided together")
        source = str(source_explicit)
        destination = str(destination_explicit)
        if source == destination or source not in set(support) or destination not in set(support):
            raise ValueError("conveyor transfer source and destination must be distinct visible lane keys")
        return (
            source,
            destination,
            _uniform_string_probability_map(support, selected=source),
            _uniform_string_probability_map(support, selected=destination),
        )
    source, destination = rng.sample(list(support), 2)
    return (
        str(source),
        str(destination),
        _uniform_string_probability_map(support, selected=str(source)),
        _uniform_string_probability_map(support, selected=str(destination)),
    )


def _sample_scoped_lane_counts(
    *,
    rng: Any,
    lane_keys: Sequence[str],
    target_lane_key: str,
    target_count: int,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for lane_key in lane_keys:
        if str(lane_key) == str(target_lane_key):
            min_count = min(8, max(2, int(target_count) + 1))
        else:
            min_count = 2
        counts[str(lane_key)] = int(rng.randrange(int(min_count), 9))
    return counts


def _lane_center_value(layout_orientation: str, lane_key: str) -> float:
    if str(layout_orientation) == LAYOUT_HORIZONTAL:
        return float(HORIZONTAL_LANE_CENTER_BY_KEY[str(lane_key)])
    return float(VERTICAL_LANE_CENTER_BY_KEY[str(lane_key)])


def _lane_records(layout_orientation: str) -> list[dict[str, Any]]:
    lane_keys = _lane_keys_for_orientation(str(layout_orientation))
    records: list[dict[str, Any]] = []
    for lane_key in lane_keys:
        records.append(
            {
                "lane_key": str(lane_key),
                "lane_label": str(LANE_LABELS[str(lane_key)]),
                "layout_orientation": str(layout_orientation),
                "center_value": float(_lane_center_value(str(layout_orientation), str(lane_key))),
                "max_count": 8,
            }
        )
    return records


def _slot_positions_for_lane(
    *,
    rng: Any,
    layout_orientation: str,
    lane_key: str,
    count: int,
) -> list[tuple[float, float, float]]:
    count = int(count)
    if count <= 0:
        return []
    center = _lane_center_value(str(layout_orientation), str(lane_key))
    slots: list[tuple[float, float, float]] = []
    if str(layout_orientation) == LAYOUT_HORIZONTAL:
        length = float(HORIZONTAL_SLOT_LENGTH)
        spacing = length / float(max(1, count))
        start = -0.5 * length + 0.5 * spacing
        for index in range(count):
            x = start + spacing * float(index) + rng.uniform(-float(LANE_SLOT_JITTER_ALONG), float(LANE_SLOT_JITTER_ALONG))
            y = center + rng.uniform(-float(LANE_SLOT_JITTER_ACROSS), float(LANE_SLOT_JITTER_ACROSS))
            slots.append((round(float(x), 4), round(float(y), 4), 0.0))
    else:
        length = float(VERTICAL_SLOT_LENGTH)
        spacing = length / float(max(1, count))
        start = -0.5 * length + 0.5 * spacing
        for index in range(count):
            x = center + rng.uniform(-float(LANE_SLOT_JITTER_ACROSS), float(LANE_SLOT_JITTER_ACROSS))
            y = start + spacing * float(index) + rng.uniform(-float(LANE_SLOT_JITTER_ALONG), float(LANE_SLOT_JITTER_ALONG))
            slots.append((round(float(x), 4), round(float(y), 4), 90.0))
    rng.shuffle(slots)
    return slots


def _ordered_slot_positions_for_lane(
    *,
    rng: Any,
    layout_orientation: str,
    lane_key: str,
    count: int,
) -> list[tuple[float, float, float]]:
    """Return slot positions in belt-arrow order for ordered-neighbor tasks."""

    count = int(count)
    if count <= 0:
        return []
    center = _lane_center_value(str(layout_orientation), str(lane_key))
    slots: list[tuple[float, float, float]] = []
    if str(layout_orientation) == LAYOUT_HORIZONTAL:
        length = float(HORIZONTAL_SLOT_LENGTH)
        spacing = length / float(max(1, count))
        start = -0.5 * length + 0.5 * spacing
        for index in range(count):
            x = start + spacing * float(index) + rng.uniform(-0.45 * float(LANE_SLOT_JITTER_ALONG), 0.45 * float(LANE_SLOT_JITTER_ALONG))
            y = center + rng.uniform(-0.55 * float(LANE_SLOT_JITTER_ACROSS), 0.55 * float(LANE_SLOT_JITTER_ACROSS))
            slots.append((round(float(x), 4), round(float(y), 4), 0.0))
    else:
        length = float(VERTICAL_SLOT_LENGTH)
        spacing = length / float(max(1, count))
        start = -0.5 * length + 0.5 * spacing
        for index in range(count):
            x = center + rng.uniform(-0.55 * float(LANE_SLOT_JITTER_ACROSS), 0.55 * float(LANE_SLOT_JITTER_ACROSS))
            y = start + spacing * float(index) + rng.uniform(-0.45 * float(LANE_SLOT_JITTER_ALONG), 0.45 * float(LANE_SLOT_JITTER_ALONG))
            slots.append((round(float(x), 4), round(float(y), 4), 90.0))
    return slots


def _ordered_pair_symbol_sequence(
    *,
    rng: Any,
    first_symbol: str,
    second_symbol: str,
    filler_symbols: Sequence[str],
    pair_count: int,
    total_count: int,
    circular: bool = False,
) -> tuple[str, ...]:
    """Construct a sequence with exactly `pair_count` adjacent first->second pairs."""

    pair_count = int(pair_count)
    total_count = int(total_count)
    if total_count < 2 * pair_count:
        raise ValueError("ordered pair sequence needs at least two slots per target pair")
    fillers = [str(symbol) for symbol in filler_symbols if str(symbol) not in {str(first_symbol), str(second_symbol)}]
    if not fillers and total_count > 2 * pair_count:
        raise ValueError("ordered pair sequence needs filler symbols")
    units: list[tuple[str, ...]] = [(str(first_symbol), str(second_symbol)) for _ in range(pair_count)]
    for filler in shuffled_repeated_support(rng, fillers, total_count - 2 * pair_count):
        units.append((str(filler),))
    rng.shuffle(units)
    sequence = tuple(symbol for unit in units for symbol in unit)
    observed = sum(
        1
        for index, symbol in enumerate(sequence)
        if str(symbol) == str(first_symbol)
        and (bool(circular) or index + 1 < len(sequence))
        and str(sequence[0 if index + 1 >= len(sequence) else index + 1]) == str(second_symbol)
    )
    if int(observed) != int(pair_count):
        raise ValueError("ordered pair sequence construction produced the wrong count")
    return tuple(sequence)


def _make_object_spec(
    *,
    rng: Any,
    object_id: str,
    shape_type: str,
    color_name: str,
    lane_key: str,
    layout_orientation: str,
    slot: Sequence[float],
    matches_query: bool,
    count_role: str,
    dimension_scale: float,
) -> Dict[str, Any]:
    """Create one countable lane object bound to metadata and projection.

    The lane key, prompt-facing belt label, semantic color, and world position
    are all recorded here so answer filtering and annotation projection use the
    same finalized object record.
    """

    dimensions = object_dimensions(str(shape_type), scale=float(dimension_scale))
    height = float(dimensions[2])
    color_rgb = SEMANTIC_COLOR_RGB[str(color_name)]
    orientation_base = float(slot[2])
    return {
        "object_id": str(object_id),
        "object_type": str(shape_type),
        "shape_type": str(shape_type),
        "object_name": public_object_name(str(shape_type)),
        "prompt_name": public_object_name(str(shape_type)),
        "public_name": public_object_name(str(shape_type)),
        "nameable_for_prompt": True,
        "is_countable_object": True,
        "matches_query": bool(matches_query),
        "count_role": str(count_role),
        "lane_key": str(lane_key),
        "lane_label": str(LANE_LABELS[str(lane_key)]),
        "belt_key": str(lane_key),
        "belt_label": str(LANE_LABELS[str(lane_key)]),
        "layout_orientation": str(layout_orientation),
        "color_name": str(color_name),
        "prompt_color_name": str(color_name),
        "fill_rgb": [int(channel) for channel in color_rgb],
        "semantic_color": True,
        "dimensions_xyz": [float(value) for value in dimensions],
        "world_xyz": [round(float(slot[0]), 4), round(float(slot[1]), 4), round(0.08 + height * 0.5, 4)],
        "base_xyz": [round(float(slot[0]), 4), round(float(slot[1]), 4), 0.08],
        "orientation_deg": round(float(orientation_base + rng.uniform(-15.0, 15.0)), 3),
        "render_order_bias": round(float(rng.uniform(-0.015, 0.015)), 5),
        "renderer_id": "object_scene_shape",
        "object_role": "target" if bool(matches_query) else "distractor",
    }


def _sample_line_camera(rng: Any) -> CameraSpec:
    yaw_degrees = float(rng.uniform(-4.0, 4.0))
    pitch_degrees = float(rng.uniform(65.0, 71.0))
    distance = float(rng.uniform(8.2, 9.0))
    yaw = math.radians(float(yaw_degrees))
    pitch = math.radians(float(pitch_degrees))
    target = (0.0, 0.0, 0.42)
    camera_position = (
        float(distance * math.cos(pitch) * math.sin(yaw)),
        float(-distance * math.cos(pitch) * math.cos(yaw)),
        float(target[2] + distance * math.sin(pitch)),
    )
    forward = vec_norm(vec_sub(target, camera_position))
    right = vec_norm(vec_cross(forward, (0.0, 0.0, 1.0)))
    up = vec_norm(vec_cross(right, forward))
    return CameraSpec(
        camera_position=tuple(camera_position),
        target=tuple(target),
        right=tuple(right),
        up=tuple(up),
        forward=tuple(forward),
        yaw_degrees=float(yaw_degrees),
        pitch_degrees=float(pitch_degrees),
        distance=float(distance),
    )


def _lane_reference_points(layout_orientation: str) -> list[tuple[float, float, float]]:
    points: list[tuple[float, float, float]] = []
    half_width = float(LANE_HALF_WIDTH)
    if str(layout_orientation) == LAYOUT_HORIZONTAL:
        half_length = 0.5 * float(HORIZONTAL_LANE_LENGTH)
        x0, x1 = -half_length, half_length
        for lane_key in HORIZONTAL_LANE_KEYS:
            y = _lane_center_value(str(layout_orientation), str(lane_key))
            points.extend([(x0, y - half_width, 0.02), (x0, y + half_width, 0.02), (x1, y - half_width, 0.02), (x1, y + half_width, 0.02)])
    else:
        half_length = 0.5 * float(VERTICAL_LANE_LENGTH)
        y0, y1 = -half_length, half_length
        for lane_key in VERTICAL_LANE_KEYS:
            x = _lane_center_value(str(layout_orientation), str(lane_key))
            points.extend([(x - half_width, y0, 0.02), (x + half_width, y0, 0.02), (x - half_width, y1, 0.02), (x + half_width, y1, 0.02)])
    return points


def _finalize_camera_and_projection(
    *,
    rng: Any,
    render_params: Any,
    layout_orientation: str,
    object_specs: Sequence[Mapping[str, Any]],
) -> tuple[CameraSpec, Any, dict[str, Any], dict[str, Any]]:
    camera = _sample_line_camera(rng)
    reference_points = [point for spec in object_specs for point in object_reference_points(spec)]
    frame = build_projection_frame(
        camera=camera,
        render_params=render_params,
        point_worlds=[*_lane_reference_points(str(layout_orientation)), *reference_points],
    )
    camera_meta = {
        "camera_position": [round(float(value), 4) for value in camera.camera_position],
        "target": [round(float(value), 4) for value in camera.target],
        "yaw_degrees": round(float(camera.yaw_degrees), 4),
        "pitch_degrees": round(float(camera.pitch_degrees), 4),
        "distance": round(float(camera.distance), 4),
        "right": [round(float(value), 5) for value in camera.right],
        "up": [round(float(value), 5) for value in camera.up],
        "forward": [round(float(value), 5) for value in camera.forward],
    }
    frame_meta = {
        "scale": round(float(frame.scale), 5),
        "center_x": round(float(frame.center_x), 3),
        "center_y": round(float(frame.center_y), 3),
        "normalized_center_u": round(float(frame.normalized_center_u), 6),
        "normalized_center_v": round(float(frame.normalized_center_v), 6),
    }
    return camera, frame, camera_meta, frame_meta


def _screen_finalize_specs(
    *,
    object_specs: Sequence[Mapping[str, Any]],
    camera: CameraSpec,
    frame: Any,
) -> list[dict[str, Any]]:
    finalized: list[dict[str, Any]] = []
    for spec in object_specs:
        updated = dict(spec)
        screen = project_screen(updated["world_xyz"], camera, frame)
        updated["screen_xy"] = [round(float(screen[0]), 3), round(float(screen[1]), 3)]
        updated["camera_distance"] = round(float(screen[7]), 5)
        finalized.append(updated)
    return sorted(finalized, key=lambda item: str(item["object_id"]))


def _base_dataset_metadata(
    *,
    axes: ResolvedConveyorAxes,
    layout_orientation: str,
    layout_orientation_probabilities: Mapping[str, float],
    target_lane_key: str,
    target_lane_probabilities: Mapping[str, float],
    target_shape: str,
    target_shape_probabilities: Mapping[str, float],
    target_color_name: str,
    target_color_probabilities: Mapping[str, float],
    target_count: int,
    target_count_probabilities: Mapping[str, float],
    object_specs: Sequence[Mapping[str, Any]],
    target_object_ids: Sequence[str],
    predicate_kind: str,
    camera_meta: Mapping[str, Any],
    frame_meta: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble trace metadata after one finalized straight-conveyor sample.

    The key invariant is that target ids, lane totals, answer value, and solver
    trace are all derived from the same finalized object specs that rendering
    receives, so annotation boxes and answer counts cannot diverge.
    """

    finalized_specs = [dict(spec) for spec in object_specs]
    shape_counts = Counter(str(spec["shape_type"]) for spec in finalized_specs)
    color_counts = Counter(str(spec["color_name"]) for spec in finalized_specs)
    lane_counts_final = Counter(str(spec["lane_key"]) for spec in finalized_specs)
    target_lane_object_ids = [
        str(spec["object_id"])
        for spec in finalized_specs
        if str(spec["lane_key"]) == str(target_lane_key)
    ]
    return {
        "scene_id": SCENE_ID,
        "scene_variant": str(axes.scene_variant),
        "layout_family": "straight_parallel_conveyors",
        "layout_orientation": str(layout_orientation),
        "layout_orientation_probabilities": dict(layout_orientation_probabilities),
        "predicate_kind": str(predicate_kind),
        "lane_records": _lane_records(str(layout_orientation)),
        "target_lane_key": str(target_lane_key),
        "target_lane_label": str(LANE_LABELS[str(target_lane_key)]),
        "target_belt_key": str(target_lane_key),
        "target_belt_label": str(LANE_LABELS[str(target_lane_key)]),
        "target_shape_type": str(target_shape),
        "target_object_name": public_object_name(str(target_shape)),
        "target_object_plural": public_object_plural(str(target_shape)),
        "target_color_name": str(target_color_name),
        "target_color_label": semantic_color_label(str(target_color_name)) if str(target_color_name) else "",
        "answer_value": int(target_count),
        "target_count": int(target_count),
        "target_object_ids": [str(object_id) for object_id in target_object_ids],
        "target_lane_object_ids": list(target_lane_object_ids),
        "target_belt_object_ids": list(target_lane_object_ids),
        "object_count": int(len(finalized_specs)),
        "object_specs": [dict(spec) for spec in finalized_specs],
        "shape_counts": {str(key): int(value) for key, value in sorted(shape_counts.items())},
        "color_counts": {str(key): int(value) for key, value in sorted(color_counts.items())},
        "lane_counts": {str(key): int(value) for key, value in sorted(lane_counts_final.items())},
        "belt_counts": {str(key): int(value) for key, value in sorted(lane_counts_final.items())},
        "target_shape_type_probabilities": dict(target_shape_probabilities),
        "target_color_name_probabilities": dict(target_color_probabilities),
        "target_count_probabilities": dict(target_count_probabilities),
        "lane_count_probabilities": {},
        "target_lane_key_probabilities": dict(target_lane_probabilities),
        "target_belt_key_probabilities": dict(target_lane_probabilities),
        "target_belt_probabilities": dict(target_lane_probabilities),
        "semantic_color_palette": {str(key): list(value) for key, value in sorted(SEMANTIC_COLOR_RGB.items())},
        "camera": dict(camera_meta),
        "projection_frame": dict(frame_meta),
        "solver_trace": {
            "count_predicate": str(predicate_kind),
            "scope": {
                "lane_key": str(target_lane_key),
                "lane_label": str(LANE_LABELS[str(target_lane_key)]),
            },
            "target_shape_type": str(target_shape),
            "target_color_name": str(target_color_name),
            "target_count": int(target_count),
            "target_object_ids": [str(object_id) for object_id in target_object_ids],
            "target_lane_object_ids": list(target_lane_object_ids),
            "answer_value": int(target_count),
            "unique_integer_answer": True,
        },
    }


def build_belt_total_count_dataset(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    render_params: Any,
    axes: ResolvedConveyorAxes,
    namespace: str,
) -> dict[str, Any]:
    """Build a straight three-lane conveyor dataset for one lane-total count."""

    rng = spawn_rng(int(instance_seed), f"{namespace}.dataset")
    layout_orientation, layout_orientation_probabilities = _resolve_layout_orientation(
        params=params,
        rng=rng,
        render_params=render_params,
    )
    lane_keys = _lane_keys_for_orientation(str(layout_orientation))
    target_lane_key, target_lane_probabilities = _resolve_target_lane(params=params, rng=rng, lane_keys=lane_keys)
    target_lane_label = str(LANE_LABELS[str(target_lane_key)])
    target_count, target_count_probabilities = _resolve_lane_count(
        params=params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.{target_lane_key}.target_count",
        key="target_count",
    )
    target_shape, target_shape_probabilities = _resolve_shape(params=params, rng=rng)
    color_names = sample_visual_color_names(rng, palette_size=4)
    if not color_names:
        raise ValueError("empty conveyor visual color palette")
    dimension_scale = float(params.get("object_dimension_scale", group_default(gen_defaults, "object_dimension_scale", 0.66)))

    lane_counts: Dict[str, int] = {}
    lane_count_probabilities: Dict[str, Dict[str, float]] = {}
    for lane_key in lane_keys:
        if str(lane_key) == str(target_lane_key):
            lane_counts[str(lane_key)] = int(target_count)
            lane_count_probabilities[str(lane_key)] = dict(target_count_probabilities)
        else:
            count, probabilities = _resolve_lane_count(
                params=params,
                gen_defaults=gen_defaults,
                instance_seed=int(instance_seed),
                namespace=f"{namespace}.{lane_key}.distractor_count",
                key=f"{lane_key}_object_count",
            )
            lane_counts[str(lane_key)] = int(count)
            lane_count_probabilities[str(lane_key)] = dict(probabilities)

    object_specs: List[Dict[str, Any]] = []
    target_object_ids: list[str] = []
    for lane_key in lane_keys:
        slots = _slot_positions_for_lane(
            rng=rng,
            layout_orientation=str(layout_orientation),
            lane_key=str(lane_key),
            count=int(lane_counts[str(lane_key)]),
        )
        for index, slot in enumerate(slots):
            object_id = f"obj_{len(object_specs):03d}"
            matches_query = str(lane_key) == str(target_lane_key)
            if matches_query:
                target_object_ids.append(str(object_id))
            object_specs.append(
                _make_object_spec(
                    rng=rng,
                    object_id=str(object_id),
                    shape_type=str(target_shape),
                    color_name=str(rng.choice(color_names)),
                    lane_key=str(lane_key),
                    layout_orientation=str(layout_orientation),
                    slot=slot,
                    matches_query=bool(matches_query),
                    count_role="target" if matches_query else "lane_distractor",
                    dimension_scale=float(dimension_scale),
                )
            )

    camera, frame, camera_meta, frame_meta = _finalize_camera_and_projection(
        rng=rng,
        render_params=render_params,
        layout_orientation=str(layout_orientation),
        object_specs=object_specs,
    )
    finalized_specs = _screen_finalize_specs(object_specs=object_specs, camera=camera, frame=frame)
    dataset = _base_dataset_metadata(
        axes=axes,
        layout_orientation=str(layout_orientation),
        layout_orientation_probabilities=layout_orientation_probabilities,
        target_lane_key=str(target_lane_key),
        target_lane_probabilities=target_lane_probabilities,
        target_shape=str(target_shape),
        target_shape_probabilities=target_shape_probabilities,
        target_color_name="",
        target_color_probabilities={},
        target_count=int(target_count),
        target_count_probabilities=target_count_probabilities,
        object_specs=finalized_specs,
        target_object_ids=target_object_ids,
        predicate_kind=PREDICATE_BELT_TOTAL,
        camera_meta=camera_meta,
        frame_meta=frame_meta,
    )
    dataset["lane_count_probabilities"] = {str(key): dict(value) for key, value in lane_count_probabilities.items()}
    return dataset


def build_scoped_belt_count_dataset(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    render_params: Any,
    axes: ResolvedConveyorAxes,
    predicate_kind: str,
    namespace: str,
) -> dict[str, Any]:
    """Build a straight three-lane conveyor dataset for one scoped lane count."""

    rng = spawn_rng(int(instance_seed), f"{namespace}.dataset")
    layout_orientation, layout_orientation_probabilities = _resolve_layout_orientation(
        params=params,
        rng=rng,
        render_params=render_params,
    )
    lane_keys = _lane_keys_for_orientation(str(layout_orientation))
    target_lane_key, target_lane_probabilities = _resolve_target_lane(params=params, rng=rng, lane_keys=lane_keys)
    target_count, target_count_probabilities = _resolve_scoped_target_count(
        params=params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.{target_lane_key}.target_count",
    )
    lane_counts = _sample_scoped_lane_counts(
        rng=rng,
        lane_keys=lane_keys,
        target_lane_key=str(target_lane_key),
        target_count=int(target_count),
    )
    dimension_scale = float(params.get("object_dimension_scale", group_default(gen_defaults, "object_dimension_scale", 0.66)))
    object_specs: List[Dict[str, Any]] = []
    target_object_ids: list[str] = []

    if str(predicate_kind) == PREDICATE_OBJECT_TYPE:
        target_shape, target_shape_probabilities = _resolve_shape(params=params, rng=rng)
        distractor_shapes = list(
            compatible_distractor_pool(str(target_shape), support=CONVEYOR_OBJECT_SHAPE_TYPES)
        )
        color_names = sample_visual_color_names(rng, palette_size=4)
        target_color_name = ""
        target_color_probabilities: Dict[str, float] = {}
    elif str(predicate_kind) == PREDICATE_COLOR:
        target_shape, target_shape_probabilities = _resolve_shape(
            params=params,
            rng=rng,
            support=CONVEYOR_COLOR_READOUT_SHAPE_TYPES,
        )
        distractor_shapes = [str(target_shape)]
        target_color_name, target_color_probabilities = _resolve_target_color(params=params, rng=rng)
        color_names = _sample_readout_palette(rng, target_color=str(target_color_name), size=4)
    elif str(predicate_kind) == PREDICATE_COLOR_TYPE:
        lane_counts[str(target_lane_key)] = max(
            int(lane_counts[str(target_lane_key)]),
            min(8, max(4, int(target_count) + 3)),
        )
        target_shape, target_shape_probabilities = _resolve_shape(
            params=params,
            rng=rng,
            support=CONVEYOR_COLOR_READOUT_SHAPE_TYPES,
        )
        distractor_shapes = list(
            compatible_distractor_pool(str(target_shape), support=CONVEYOR_COLOR_READOUT_SHAPE_TYPES)
        )
        target_color_name, target_color_probabilities = _resolve_target_color(params=params, rng=rng)
        color_names = _sample_readout_palette(rng, target_color=str(target_color_name), size=4)
    else:
        raise ValueError(f"unsupported straight conveyor scoped predicate: {predicate_kind}")
    if not color_names:
        raise ValueError("empty conveyor visual color palette")
    wrong_colors = [str(color) for color in color_names if str(color) != str(target_color_name)]

    for lane_key in lane_keys:
        slots = _slot_positions_for_lane(
            rng=rng,
            layout_orientation=str(layout_orientation),
            lane_key=str(lane_key),
            count=int(lane_counts[str(lane_key)]),
        )
        for index, slot in enumerate(slots):
            matches_query = str(lane_key) == str(target_lane_key) and int(index) < int(target_count)
            if str(predicate_kind) == PREDICATE_OBJECT_TYPE:
                shape_type = str(target_shape) if bool(matches_query) or (str(lane_key) != str(target_lane_key) and rng.random() < 0.35) else str(
                    distractor_shapes[int(rng.randrange(len(distractor_shapes)))]
                )
                color_name = str(rng.choice(color_names))
                count_role = "target" if bool(matches_query) else ("same_belt_distractor" if str(lane_key) == str(target_lane_key) else "lane_distractor")
            else:
                if str(predicate_kind) == PREDICATE_COLOR:
                    shape_type = str(target_shape)
                    if bool(matches_query):
                        color_name = str(target_color_name)
                    elif str(lane_key) != str(target_lane_key) and rng.random() < 0.35:
                        color_name = str(target_color_name)
                    else:
                        color_name = str(rng.choice(wrong_colors))
                    count_role = "target" if bool(matches_query) else ("same_belt_distractor" if str(lane_key) == str(target_lane_key) else "lane_distractor")
                else:
                    if bool(matches_query):
                        shape_type = str(target_shape)
                        color_name = str(target_color_name)
                        count_role = "target"
                    elif str(lane_key) == str(target_lane_key):
                        pattern = (int(index) - int(target_count)) % 3
                        if pattern == 0:
                            shape_type = str(distractor_shapes[int(rng.randrange(len(distractor_shapes)))])
                            color_name = str(target_color_name)
                            count_role = "same_belt_same_color_wrong_type"
                        elif pattern == 1:
                            shape_type = str(target_shape)
                            color_name = str(rng.choice(wrong_colors))
                            count_role = "same_belt_same_type_wrong_color"
                        else:
                            shape_type = str(distractor_shapes[int(rng.randrange(len(distractor_shapes)))])
                            color_name = str(rng.choice(wrong_colors))
                            count_role = "same_belt_wrong_type_wrong_color"
                    elif int(index) == 0:
                        shape_type = str(target_shape)
                        color_name = str(target_color_name)
                        count_role = "other_belt_same_color_type"
                    elif int(index) % 3 == 1:
                        shape_type = str(distractor_shapes[int(rng.randrange(len(distractor_shapes)))])
                        color_name = str(target_color_name)
                        count_role = "other_belt_same_color_wrong_type"
                    elif int(index) % 3 == 2:
                        shape_type = str(target_shape)
                        color_name = str(rng.choice(wrong_colors))
                        count_role = "other_belt_same_type_wrong_color"
                    else:
                        shape_type = str(distractor_shapes[int(rng.randrange(len(distractor_shapes)))])
                        color_name = str(rng.choice(wrong_colors))
                        count_role = "other_belt_wrong_type_wrong_color"
            object_id = f"obj_{len(object_specs):03d}"
            if bool(matches_query):
                target_object_ids.append(str(object_id))
            object_specs.append(
                _make_object_spec(
                    rng=rng,
                    object_id=str(object_id),
                    shape_type=str(shape_type),
                    color_name=str(color_name),
                    lane_key=str(lane_key),
                    layout_orientation=str(layout_orientation),
                    slot=slot,
                    matches_query=bool(matches_query),
                    count_role=str(count_role),
                    dimension_scale=float(dimension_scale),
                )
            )

    camera, frame, camera_meta, frame_meta = _finalize_camera_and_projection(
        rng=rng,
        render_params=render_params,
        layout_orientation=str(layout_orientation),
        object_specs=object_specs,
    )
    finalized_specs = _screen_finalize_specs(object_specs=object_specs, camera=camera, frame=frame)
    return _base_dataset_metadata(
        axes=axes,
        layout_orientation=str(layout_orientation),
        layout_orientation_probabilities=layout_orientation_probabilities,
        target_lane_key=str(target_lane_key),
        target_lane_probabilities=target_lane_probabilities,
        target_shape=str(target_shape),
        target_shape_probabilities=target_shape_probabilities,
        target_color_name=str(target_color_name),
        target_color_probabilities=target_color_probabilities,
        target_count=int(target_count),
        target_count_probabilities=target_count_probabilities,
        object_specs=finalized_specs,
        target_object_ids=target_object_ids,
        predicate_kind=str(predicate_kind),
        camera_meta=camera_meta,
        frame_meta=frame_meta,
    )


def build_lane_count_arithmetic_dataset(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    render_params: Any,
    axes: ResolvedConveyorAxes,
    predicate_kind: str,
    operation: str,
    namespace: str,
) -> dict[str, Any]:
    """Build a three-lane conveyor dataset for two-lane count arithmetic."""

    rng = spawn_rng(int(instance_seed), f"{namespace}.dataset")
    layout_orientation, layout_orientation_probabilities = _resolve_layout_orientation(
        params=params,
        rng=rng,
        render_params=render_params,
    )
    lane_keys = _lane_keys_for_orientation(str(layout_orientation))
    selected_lanes, selected_lane_probabilities = _resolve_arithmetic_lanes(
        params=params,
        rng=rng,
        lane_keys=lane_keys,
    )
    first_count, second_count, answer_value, answer_probabilities, operand_probabilities = _resolve_arithmetic_operand_counts(
        params=params,
        gen_defaults=gen_defaults,
        rng=rng,
        instance_seed=int(instance_seed),
        namespace=str(namespace),
        operation=str(operation),
    )
    operand_counts_by_lane = {
        str(selected_lanes[0]): int(first_count),
        str(selected_lanes[1]): int(second_count),
    }
    lane_counts: Dict[str, int] = {}
    for lane_key in lane_keys:
        if str(lane_key) in operand_counts_by_lane:
            target_count = int(operand_counts_by_lane[str(lane_key)])
            min_count = min(8, max(2, target_count + (1 if target_count < 8 else 0)))
            lane_counts[str(lane_key)] = int(rng.randrange(int(min_count), 9))
        else:
            lane_counts[str(lane_key)] = int(rng.randrange(2, 9))

    dimension_scale = float(params.get("object_dimension_scale", group_default(gen_defaults, "object_dimension_scale", 0.74)))
    object_specs: List[Dict[str, Any]] = []
    target_object_ids_by_lane: dict[str, list[str]] = {str(lane_key): [] for lane_key in selected_lanes}

    if str(predicate_kind) == PREDICATE_OBJECT_TYPE_ARITHMETIC:
        target_shape, target_shape_probabilities = _resolve_shape(params=params, rng=rng)
        distractor_shapes = list(compatible_distractor_pool(str(target_shape), support=CONVEYOR_OBJECT_SHAPE_TYPES))
        color_names = sample_visual_color_names(rng, palette_size=4)
        target_color_name = ""
        target_color_probabilities: Dict[str, float] = {}
        wrong_colors = list(color_names)
    else:
        raise ValueError(f"unsupported straight conveyor arithmetic predicate: {predicate_kind}")
    if not color_names:
        raise ValueError("empty conveyor arithmetic color palette")

    for lane_key in lane_keys:
        slots = _slot_positions_for_lane(
            rng=rng,
            layout_orientation=str(layout_orientation),
            lane_key=str(lane_key),
            count=int(lane_counts[str(lane_key)]),
        )
        target_count = int(operand_counts_by_lane.get(str(lane_key), 0))
        for index, slot in enumerate(slots):
            in_selected_scope = str(lane_key) in set(str(value) for value in selected_lanes)
            matches_query = bool(in_selected_scope and int(index) < int(target_count))
            if bool(matches_query) or (not in_selected_scope and rng.random() < 0.35):
                shape_type = str(target_shape)
            else:
                shape_type = str(distractor_shapes[int(rng.randrange(len(distractor_shapes)))])
            color_name = str(rng.choice(wrong_colors))
            object_id = f"obj_{len(object_specs):03d}"
            if bool(matches_query):
                target_object_ids_by_lane[str(lane_key)].append(str(object_id))
            object_specs.append(
                _make_object_spec(
                    rng=rng,
                    object_id=str(object_id),
                    shape_type=str(shape_type),
                    color_name=str(color_name),
                    lane_key=str(lane_key),
                    layout_orientation=str(layout_orientation),
                    slot=slot,
                    matches_query=bool(matches_query),
                    count_role="operand_target" if bool(matches_query) else ("selected_lane_distractor" if bool(in_selected_scope) else "outside_scope_distractor"),
                    dimension_scale=float(dimension_scale),
                )
            )

    camera, frame, camera_meta, frame_meta = _finalize_camera_and_projection(
        rng=rng,
        render_params=render_params,
        layout_orientation=str(layout_orientation),
        object_specs=object_specs,
    )
    finalized_specs = _screen_finalize_specs(object_specs=object_specs, camera=camera, frame=frame)
    shape_counts = Counter(str(spec["shape_type"]) for spec in finalized_specs)
    color_counts = Counter(str(spec["color_name"]) for spec in finalized_specs)
    lane_counts_final = Counter(str(spec["lane_key"]) for spec in finalized_specs)
    target_object_ids = [
        str(object_id)
        for lane_key in selected_lanes
        for object_id in target_object_ids_by_lane[str(lane_key)]
    ]
    annotation_key_by_lane = {str(lane_key): f"{lane_key}_objects" for lane_key in selected_lanes}
    target_object_ids_by_annotation_key = {
        str(annotation_key_by_lane[str(lane_key)]): [str(object_id) for object_id in target_object_ids_by_lane[str(lane_key)]]
        for lane_key in selected_lanes
    }
    operand_counts_by_scope = {
        str(annotation_key_by_lane[str(lane_key)]): int(operand_counts_by_lane[str(lane_key)])
        for lane_key in selected_lanes
    }
    return {
        "scene_id": SCENE_ID,
        "scene_variant": str(axes.scene_variant),
        "layout_family": "straight_parallel_conveyors",
        "layout_orientation": str(layout_orientation),
        "layout_orientation_probabilities": dict(layout_orientation_probabilities),
        "predicate_kind": str(predicate_kind),
        "arithmetic_operation": str(operation),
        "lane_records": _lane_records(str(layout_orientation)),
        "scope_keys": [str(lane_key) for lane_key in selected_lanes],
        "scope_labels": {str(lane_key): str(LANE_LABELS[str(lane_key)]) for lane_key in selected_lanes},
        "annotation_key_by_scope": dict(annotation_key_by_lane),
        "target_object_ids_by_annotation_key": dict(target_object_ids_by_annotation_key),
        "operand_counts_by_scope": dict(operand_counts_by_scope),
        "operand_count_probabilities": dict(operand_probabilities),
        "target_lane_key": "_".join(str(lane_key) for lane_key in selected_lanes),
        "target_lane_label": " and ".join(str(LANE_LABELS[str(lane_key)]) for lane_key in selected_lanes),
        "target_belt_key": "_".join(str(lane_key) for lane_key in selected_lanes),
        "target_belt_label": " and ".join(str(LANE_LABELS[str(lane_key)]) for lane_key in selected_lanes),
        "target_shape_type": str(target_shape),
        "target_object_name": public_object_name(str(target_shape)),
        "target_object_plural": public_object_plural(str(target_shape)),
        "target_color_name": str(target_color_name),
        "target_color_label": semantic_color_label(str(target_color_name)) if str(target_color_name) else "",
        "answer_value": int(answer_value),
        "target_count": int(answer_value),
        "target_object_ids": list(target_object_ids),
        "target_lane_object_ids": list(target_object_ids),
        "target_belt_object_ids": list(target_object_ids),
        "object_count": int(len(finalized_specs)),
        "object_specs": [dict(spec) for spec in finalized_specs],
        "shape_counts": {str(key): int(value) for key, value in sorted(shape_counts.items())},
        "color_counts": {str(key): int(value) for key, value in sorted(color_counts.items())},
        "lane_counts": {str(key): int(value) for key, value in sorted(lane_counts_final.items())},
        "belt_counts": {str(key): int(value) for key, value in sorted(lane_counts_final.items())},
        "target_shape_type_probabilities": dict(target_shape_probabilities),
        "target_color_name_probabilities": dict(target_color_probabilities),
        "target_count_probabilities": dict(answer_probabilities),
        "lane_count_probabilities": {str(lane_key): {str(count): 1.0} for lane_key, count in lane_counts.items()},
        "target_lane_key_probabilities": {"_".join(str(lane_key) for lane_key in selected_lanes): 1.0},
        "target_belt_key_probabilities": {"_".join(str(lane_key) for lane_key in selected_lanes): 1.0},
        "target_belt_probabilities": {"_".join(str(lane_key) for lane_key in selected_lanes): 1.0},
        "semantic_color_palette": {str(key): list(value) for key, value in sorted(SEMANTIC_COLOR_RGB.items())},
        "camera": dict(camera_meta),
        "projection_frame": dict(frame_meta),
        "solver_trace": {
            "count_predicate": str(predicate_kind),
            "operation": str(operation),
            "scopes": {str(lane_key): str(LANE_LABELS[str(lane_key)]) for lane_key in selected_lanes},
            "target_shape_type": str(target_shape),
            "target_color_name": str(target_color_name),
            "operand_counts_by_scope": dict(operand_counts_by_scope),
            "target_object_ids_by_annotation_key": dict(target_object_ids_by_annotation_key),
            "target_object_ids": list(target_object_ids),
            "answer_value": int(answer_value),
            "unique_integer_answer": True,
        },
    }


def build_transfer_count_dataset(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    render_params: Any,
    axes: ResolvedConveyorAxes,
    predicate_kind: str,
    namespace: str,
) -> dict[str, Any]:
    """Build a three-lane conveyor dataset for a counterfactual transfer count."""

    rng = spawn_rng(int(instance_seed), f"{namespace}.dataset")
    layout_orientation, layout_orientation_probabilities = _resolve_layout_orientation(
        params=params,
        rng=rng,
        render_params=render_params,
    )
    lane_keys = _lane_keys_for_orientation(str(layout_orientation))
    source_lane_key, destination_lane_key, source_lane_probabilities, destination_lane_probabilities = _resolve_transfer_lanes(
        params=params,
        rng=rng,
        lane_keys=lane_keys,
    )
    answer_value, moved_count, destination_existing_count, answer_probabilities, operand_probabilities = _resolve_transfer_answer_and_counts(
        params=params,
        gen_defaults=gen_defaults,
        rng=rng,
        instance_seed=int(instance_seed),
        namespace=str(namespace),
        destination_cap=8,
    )
    lane_counts: Dict[str, int] = {}
    for lane_key in lane_keys:
        if str(lane_key) == str(source_lane_key):
            min_count = max(int(moved_count), min(8, int(moved_count) + 1))
            lane_counts[str(lane_key)] = int(rng.randrange(int(min_count), 9))
        elif str(lane_key) == str(destination_lane_key):
            lane_counts[str(lane_key)] = int(destination_existing_count)
        else:
            lane_counts[str(lane_key)] = int(rng.randrange(2, 9))

    dimension_scale = float(params.get("object_dimension_scale", group_default(gen_defaults, "object_dimension_scale", 0.74)))
    object_specs: List[Dict[str, Any]] = []
    source_moved_ids: list[str] = []
    destination_existing_ids: list[str] = []

    if str(predicate_kind) == PREDICATE_COLOR_TRANSFER:
        target_shape, target_shape_probabilities = _resolve_shape(
            params=params,
            rng=rng,
            support=CONVEYOR_COLOR_READOUT_SHAPE_TYPES,
        )
        target_color_name, target_color_probabilities = _resolve_target_color(params=params, rng=rng)
        color_names = _sample_readout_palette(rng, target_color=str(target_color_name), size=5)
        wrong_colors = [str(color) for color in color_names if str(color) != str(target_color_name)]
        distractor_shapes = list(compatible_distractor_pool(str(target_shape), support=CONVEYOR_COLOR_READOUT_SHAPE_TYPES))
    elif str(predicate_kind) == PREDICATE_OBJECT_TYPE_TRANSFER:
        target_shape, target_shape_probabilities = _resolve_shape(
            params=params,
            rng=rng,
            support=CONVEYOR_COLOR_READOUT_SHAPE_TYPES,
        )
        target_color_name = ""
        target_color_probabilities = {}
        color_names = tuple(sample_visual_color_names(rng, palette_size=4))
        wrong_colors = list(color_names)
        distractor_shapes = list(compatible_distractor_pool(str(target_shape), support=CONVEYOR_COLOR_READOUT_SHAPE_TYPES))
    else:
        raise ValueError(f"unsupported straight conveyor transfer predicate: {predicate_kind}")
    if not wrong_colors or not distractor_shapes:
        raise ValueError("conveyor transfer task needs non-target distractors")

    for lane_key in lane_keys:
        slots = _slot_positions_for_lane(
            rng=rng,
            layout_orientation=str(layout_orientation),
            lane_key=str(lane_key),
            count=int(lane_counts[str(lane_key)]),
        )
        for index, slot in enumerate(slots):
            object_id = f"obj_{len(object_specs):03d}"
            if str(lane_key) == str(source_lane_key) and int(index) < int(moved_count):
                if str(predicate_kind) == PREDICATE_COLOR_TRANSFER:
                    shape_type = str(target_shape if index % 2 == 0 else rng.choice(distractor_shapes))
                    color_name = str(target_color_name)
                else:
                    shape_type = str(target_shape)
                    color_name = str(rng.choice(color_names))
                matches_query = True
                count_role = "source_moved_object"
                source_moved_ids.append(str(object_id))
            elif str(lane_key) == str(source_lane_key):
                if str(predicate_kind) == PREDICATE_COLOR_TRANSFER:
                    shape_type = str(rng.choice(distractor_shapes))
                    color_name = str(rng.choice(wrong_colors))
                else:
                    shape_type = str(rng.choice(distractor_shapes))
                    color_name = str(rng.choice(color_names))
                matches_query = False
                count_role = "source_non_moved_distractor"
            elif str(lane_key) == str(destination_lane_key):
                if str(predicate_kind) == PREDICATE_COLOR_TRANSFER:
                    shape_type = str(target_shape if index % 3 == 0 else rng.choice(distractor_shapes))
                    color_name = str(rng.choice(color_names))
                else:
                    shape_type = str(target_shape if index % 3 == 0 else rng.choice(distractor_shapes))
                    color_name = str(rng.choice(color_names))
                matches_query = True
                count_role = "destination_existing_object"
                destination_existing_ids.append(str(object_id))
            else:
                if str(predicate_kind) == PREDICATE_COLOR_TRANSFER and index % 3 == 0:
                    shape_type = str(target_shape)
                    color_name = str(target_color_name)
                elif str(predicate_kind) == PREDICATE_OBJECT_TYPE_TRANSFER and index % 3 == 0:
                    shape_type = str(target_shape)
                    color_name = str(rng.choice(color_names))
                else:
                    shape_type = str(rng.choice(distractor_shapes))
                    color_name = str(rng.choice(color_names))
                matches_query = False
                count_role = "outside_scope_distractor"
            object_specs.append(
                _make_object_spec(
                    rng=rng,
                    object_id=str(object_id),
                    shape_type=str(shape_type),
                    color_name=str(color_name),
                    lane_key=str(lane_key),
                    layout_orientation=str(layout_orientation),
                    slot=slot,
                    matches_query=bool(matches_query),
                    count_role=str(count_role),
                    dimension_scale=float(dimension_scale),
                )
            )

    camera, frame, camera_meta, frame_meta = _finalize_camera_and_projection(
        rng=rng,
        render_params=render_params,
        layout_orientation=str(layout_orientation),
        object_specs=object_specs,
    )
    finalized_specs = _screen_finalize_specs(object_specs=object_specs, camera=camera, frame=frame)
    shape_counts = Counter(str(spec["shape_type"]) for spec in finalized_specs)
    color_counts = Counter(str(spec["color_name"]) for spec in finalized_specs)
    lane_counts_final = Counter(str(spec["lane_key"]) for spec in finalized_specs)
    annotation_key_by_scope = {
        "source_moved": "source_moved_objects",
        "destination_existing": "destination_existing_objects",
    }
    target_object_ids_by_annotation_key = {
        "source_moved_objects": list(source_moved_ids),
        "destination_existing_objects": list(destination_existing_ids),
    }
    target_object_ids = [*list(source_moved_ids), *list(destination_existing_ids)]
    return {
        "scene_id": SCENE_ID,
        "scene_variant": str(axes.scene_variant),
        "layout_family": "straight_parallel_conveyors",
        "layout_orientation": str(layout_orientation),
        "layout_orientation_probabilities": dict(layout_orientation_probabilities),
        "predicate_kind": str(predicate_kind),
        "transfer_operation": TRANSFER_OPERATION,
        "lane_records": _lane_records(str(layout_orientation)),
        "scope_keys": [str(source_lane_key), str(destination_lane_key)],
        "scope_labels": {
            str(source_lane_key): str(LANE_LABELS[str(source_lane_key)]),
            str(destination_lane_key): str(LANE_LABELS[str(destination_lane_key)]),
        },
        "source_lane_key": str(source_lane_key),
        "source_lane_label": str(LANE_LABELS[str(source_lane_key)]),
        "destination_lane_key": str(destination_lane_key),
        "destination_lane_label": str(LANE_LABELS[str(destination_lane_key)]),
        "source_belt_key": str(source_lane_key),
        "source_belt_label": str(LANE_LABELS[str(source_lane_key)]),
        "destination_belt_key": str(destination_lane_key),
        "destination_belt_label": str(LANE_LABELS[str(destination_lane_key)]),
        "annotation_key_by_scope": dict(annotation_key_by_scope),
        "target_object_ids_by_annotation_key": dict(target_object_ids_by_annotation_key),
        "operand_counts_by_scope": {
            "source_moved_objects": int(moved_count),
            "destination_existing_objects": int(destination_existing_count),
        },
        "operand_count_probabilities": dict(operand_probabilities),
        "target_lane_key": str(destination_lane_key),
        "target_lane_label": str(LANE_LABELS[str(destination_lane_key)]),
        "target_belt_key": str(destination_lane_key),
        "target_belt_label": str(LANE_LABELS[str(destination_lane_key)]),
        "target_shape_type": str(target_shape),
        "target_object_name": public_object_name(str(target_shape)),
        "target_object_plural": public_object_plural(str(target_shape)),
        "target_color_name": str(target_color_name),
        "target_color_label": semantic_color_label(str(target_color_name)) if str(target_color_name) else "",
        "answer_value": int(answer_value),
        "target_count": int(answer_value),
        "moved_count": int(moved_count),
        "destination_existing_count": int(destination_existing_count),
        "target_object_ids": [str(object_id) for object_id in target_object_ids],
        "source_moved_object_ids": list(source_moved_ids),
        "destination_existing_object_ids": list(destination_existing_ids),
        "target_lane_object_ids": [
            str(spec["object_id"])
            for spec in finalized_specs
            if str(spec["lane_key"]) == str(destination_lane_key)
        ],
        "target_belt_object_ids": [
            str(spec["object_id"])
            for spec in finalized_specs
            if str(spec["lane_key"]) == str(destination_lane_key)
        ],
        "object_count": int(len(finalized_specs)),
        "object_specs": [dict(spec) for spec in finalized_specs],
        "shape_counts": {str(key): int(value) for key, value in sorted(shape_counts.items())},
        "color_counts": {str(key): int(value) for key, value in sorted(color_counts.items())},
        "lane_counts": {str(key): int(value) for key, value in sorted(lane_counts_final.items())},
        "belt_counts": {str(key): int(value) for key, value in sorted(lane_counts_final.items())},
        "target_shape_type_probabilities": dict(target_shape_probabilities),
        "target_color_name_probabilities": dict(target_color_probabilities),
        "target_count_probabilities": dict(answer_probabilities),
        "lane_count_probabilities": {str(lane_key): {str(count): 1.0} for lane_key, count in lane_counts.items()},
        "target_lane_key_probabilities": dict(destination_lane_probabilities),
        "target_belt_key_probabilities": dict(destination_lane_probabilities),
        "target_belt_probabilities": dict(destination_lane_probabilities),
        "source_lane_key_probabilities": dict(source_lane_probabilities),
        "source_belt_key_probabilities": dict(source_lane_probabilities),
        "semantic_color_palette": {str(key): list(value) for key, value in sorted(SEMANTIC_COLOR_RGB.items())},
        "camera": dict(camera_meta),
        "projection_frame": dict(frame_meta),
        "solver_trace": {
            "count_predicate": str(predicate_kind),
            "operation": TRANSFER_OPERATION,
            "source": {
                "lane_key": str(source_lane_key),
                "lane_label": str(LANE_LABELS[str(source_lane_key)]),
            },
            "destination": {
                "lane_key": str(destination_lane_key),
                "lane_label": str(LANE_LABELS[str(destination_lane_key)]),
            },
            "target_shape_type": str(target_shape),
            "target_color_name": str(target_color_name),
            "moved_count": int(moved_count),
            "destination_existing_count": int(destination_existing_count),
            "answer_value": int(answer_value),
            "target_object_ids_by_annotation_key": dict(target_object_ids_by_annotation_key),
            "target_object_ids": [str(object_id) for object_id in target_object_ids],
            "unique_integer_answer": True,
        },
    }


def build_ordered_pair_count_dataset(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    render_params: Any,
    axes: ResolvedConveyorAxes,
    predicate_kind: str,
    namespace: str,
) -> dict[str, Any]:
    """Build a straight-conveyor dataset for ordered adjacent-pair counting."""

    rng = spawn_rng(int(instance_seed), f"{namespace}.dataset")
    layout_orientation, layout_orientation_probabilities = _resolve_layout_orientation(
        params=params,
        rng=rng,
        render_params=render_params,
    )
    lane_keys = _lane_keys_for_orientation(str(layout_orientation))
    target_lane_key, target_lane_probabilities = _resolve_target_lane(params=params, rng=rng, lane_keys=lane_keys)
    target_count, target_count_probabilities = _resolve_ordered_pair_count(
        params=params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.{target_lane_key}.target_count",
    )
    target_total_min = max(6, 2 * int(target_count))
    target_total = int(rng.randrange(int(target_total_min), 9))
    lane_counts: Dict[str, int] = {
        str(lane_key): (int(target_total) if str(lane_key) == str(target_lane_key) else int(rng.randrange(3, 9)))
        for lane_key in lane_keys
    }
    dimension_scale = float(params.get("object_dimension_scale", group_default(gen_defaults, "object_dimension_scale", 0.66)))

    if str(predicate_kind) == PREDICATE_ORDERED_COLOR_PAIR:
        target_shape, target_shape_probabilities = _resolve_shape(
            params=params,
            rng=rng,
            support=CONVEYOR_COLOR_READOUT_SHAPE_TYPES,
        )
        first_color, first_color_probabilities = _resolve_target_color(params=params, rng=rng)
        palette = list(_sample_readout_palette(rng, target_color=str(first_color), size=6))
        second_explicit = params.get("second_target_color_name")
        if second_explicit is not None:
            second_color = str(second_explicit)
            if second_color == str(first_color) or second_color not in set(palette):
                raise ValueError(f"unsupported second_target_color_name: {second_color}")
        else:
            candidates = [str(color) for color in palette if str(color) != str(first_color)]
            second_color = str(candidates[int(rng.randrange(len(candidates)))])
        filler_symbols = [str(color) for color in palette if str(color) not in {str(first_color), str(second_color)}]
        target_color_name = str(first_color)
        second_target_color_name = str(second_color)
        target_color_probabilities = dict(first_color_probabilities)
        target_shape_pair = ("", "")
        target_object_name_pair = ("", "")
        target_object_plural_pair = ("", "")
    elif str(predicate_kind) == PREDICATE_ORDERED_OBJECT_PAIR:
        first_shape, target_shape_probabilities = _resolve_shape(
            params=params,
            rng=rng,
            support=CONVEYOR_COLOR_READOUT_SHAPE_TYPES,
        )
        second_explicit = params.get("second_target_shape_type")
        second_candidates = list(compatible_distractor_pool(str(first_shape), support=CONVEYOR_COLOR_READOUT_SHAPE_TYPES))
        if second_explicit is not None:
            second_shape = str(second_explicit)
            if second_shape not in set(second_candidates):
                raise ValueError(f"unsupported second_target_shape_type: {second_shape}")
        else:
            second_shape = str(second_candidates[int(rng.randrange(len(second_candidates)))])
        filler_pool = list(compatible_distractor_pool(str(first_shape), support=CONVEYOR_COLOR_READOUT_SHAPE_TYPES))
        filler_pool = list(compatible_distractor_pool(str(second_shape), support=filler_pool))
        filler_symbols = [str(shape) for shape in filler_pool if str(shape) not in {str(first_shape), str(second_shape)}]
        target_color_name = ""
        second_target_color_name = ""
        target_color_probabilities: Dict[str, float] = {}
        target_shape = str(first_shape)
        target_shape_pair = (str(first_shape), str(second_shape))
        target_object_name_pair = (public_object_name(str(first_shape)), public_object_name(str(second_shape)))
        target_object_plural_pair = (public_object_plural(str(first_shape)), public_object_plural(str(second_shape)))
        palette = list(sample_visual_color_names(rng, palette_size=4))
    else:
        raise ValueError(f"unsupported straight conveyor ordered-pair predicate: {predicate_kind}")
    if not filler_symbols:
        raise ValueError("ordered pair task needs at least one filler symbol")

    target_sequence = _ordered_pair_symbol_sequence(
        rng=rng,
        first_symbol=str(target_color_name if predicate_kind == PREDICATE_ORDERED_COLOR_PAIR else target_shape_pair[0]),
        second_symbol=str(second_target_color_name if predicate_kind == PREDICATE_ORDERED_COLOR_PAIR else target_shape_pair[1]),
        filler_symbols=filler_symbols,
        pair_count=int(target_count),
        total_count=int(target_total),
        circular=False,
    )

    object_specs: List[Dict[str, Any]] = []
    object_sequences_by_lane: Dict[str, list[str]] = {}
    target_pair_object_id_pairs: list[list[str]] = []
    target_object_ids: list[str] = []

    for lane_key in lane_keys:
        slots = _ordered_slot_positions_for_lane(
            rng=rng,
            layout_orientation=str(layout_orientation),
            lane_key=str(lane_key),
            count=int(lane_counts[str(lane_key)]),
        )
        if str(lane_key) == str(target_lane_key):
            sequence = tuple(target_sequence)
        else:
            if str(predicate_kind) == PREDICATE_ORDERED_COLOR_PAIR:
                sequence = tuple(str(palette[int(rng.randrange(len(palette)))]) for _ in slots)
            else:
                distractor_shapes = [*list(target_shape_pair), *list(filler_symbols)]
                sequence = tuple(str(distractor_shapes[int(rng.randrange(len(distractor_shapes)))]) for _ in slots)
        lane_object_ids: list[str] = []
        for index, slot in enumerate(slots):
            object_id = f"obj_{len(object_specs):03d}"
            lane_object_ids.append(str(object_id))
            if str(predicate_kind) == PREDICATE_ORDERED_COLOR_PAIR:
                shape_type = str(target_shape)
                color_name = str(sequence[index])
            else:
                shape_type = str(sequence[index])
                color_name = str(rng.choice(palette))
            object_specs.append(
                _make_object_spec(
                    rng=rng,
                    object_id=str(object_id),
                    shape_type=str(shape_type),
                    color_name=str(color_name),
                    lane_key=str(lane_key),
                    layout_orientation=str(layout_orientation),
                    slot=slot,
                    matches_query=False,
                    count_role="ordered_pair_scope" if str(lane_key) == str(target_lane_key) else "lane_distractor",
                    dimension_scale=float(dimension_scale),
                )
            )
        object_sequences_by_lane[str(lane_key)] = list(lane_object_ids)
        if str(lane_key) == str(target_lane_key):
            first_symbol = str(target_color_name if predicate_kind == PREDICATE_ORDERED_COLOR_PAIR else target_shape_pair[0])
            second_symbol = str(second_target_color_name if predicate_kind == PREDICATE_ORDERED_COLOR_PAIR else target_shape_pair[1])
            for index in range(len(sequence) - 1):
                if str(sequence[index]) == first_symbol and str(sequence[index + 1]) == second_symbol:
                    pair = [str(lane_object_ids[index]), str(lane_object_ids[index + 1])]
                    target_pair_object_id_pairs.append(pair)
                    target_object_ids.extend(pair)

    target_object_ids = list(dict.fromkeys(target_object_ids))
    if len(target_pair_object_id_pairs) != int(target_count):
        raise ValueError("ordered pair dataset target pair count mismatch")
    for object_id in set(target_object_ids):
        for spec in object_specs:
            if str(spec["object_id"]) == str(object_id):
                spec["matches_query"] = True
                spec["object_role"] = "target"

    camera, frame, camera_meta, frame_meta = _finalize_camera_and_projection(
        rng=rng,
        render_params=render_params,
        layout_orientation=str(layout_orientation),
        object_specs=object_specs,
    )
    finalized_specs = _screen_finalize_specs(object_specs=object_specs, camera=camera, frame=frame)
    shape_counts = Counter(str(spec["shape_type"]) for spec in finalized_specs)
    color_counts = Counter(str(spec["color_name"]) for spec in finalized_specs)
    lane_counts_final = Counter(str(spec["lane_key"]) for spec in finalized_specs)
    target_lane_object_ids = [
        str(spec["object_id"])
        for spec in finalized_specs
        if str(spec["lane_key"]) == str(target_lane_key)
    ]
    return {
        "scene_id": SCENE_ID,
        "scene_variant": str(axes.scene_variant),
        "layout_family": "straight_parallel_conveyors",
        "layout_orientation": str(layout_orientation),
        "layout_orientation_probabilities": dict(layout_orientation_probabilities),
        "predicate_kind": str(predicate_kind),
        "lane_records": _lane_records(str(layout_orientation)),
        "target_lane_key": str(target_lane_key),
        "target_lane_label": str(LANE_LABELS[str(target_lane_key)]),
        "target_belt_key": str(target_lane_key),
        "target_belt_label": str(LANE_LABELS[str(target_lane_key)]),
        "target_shape_type": str(target_shape),
        "target_object_name": public_object_name(str(target_shape)),
        "target_object_plural": public_object_plural(str(target_shape)),
        "target_shape_pair": [str(shape) for shape in target_shape_pair],
        "target_object_name_pair": [str(name) for name in target_object_name_pair],
        "target_object_plural_pair": [str(name) for name in target_object_plural_pair],
        "target_color_name": str(target_color_name),
        "second_target_color_name": str(second_target_color_name),
        "target_color_label": semantic_color_label(str(target_color_name)) if str(target_color_name) else "",
        "second_target_color_label": semantic_color_label(str(second_target_color_name)) if str(second_target_color_name) else "",
        "answer_value": int(target_count),
        "target_count": int(target_count),
        "target_object_ids": [str(object_id) for object_id in target_object_ids],
        "target_pair_object_id_pairs": [list(pair) for pair in target_pair_object_id_pairs],
        "target_lane_object_ids": list(target_lane_object_ids),
        "target_belt_object_ids": list(target_lane_object_ids),
        "object_sequences_by_lane": {str(key): list(value) for key, value in object_sequences_by_lane.items()},
        "object_count": int(len(finalized_specs)),
        "object_specs": [dict(spec) for spec in finalized_specs],
        "shape_counts": {str(key): int(value) for key, value in sorted(shape_counts.items())},
        "color_counts": {str(key): int(value) for key, value in sorted(color_counts.items())},
        "lane_counts": {str(key): int(value) for key, value in sorted(lane_counts_final.items())},
        "belt_counts": {str(key): int(value) for key, value in sorted(lane_counts_final.items())},
        "target_shape_type_probabilities": dict(target_shape_probabilities),
        "target_color_name_probabilities": dict(target_color_probabilities),
        "target_count_probabilities": dict(target_count_probabilities),
        "lane_count_probabilities": {str(lane_key): {str(count): 1.0} for lane_key, count in lane_counts.items()},
        "target_lane_key_probabilities": dict(target_lane_probabilities),
        "target_belt_key_probabilities": dict(target_lane_probabilities),
        "target_belt_probabilities": dict(target_lane_probabilities),
        "semantic_color_palette": {str(key): list(value) for key, value in sorted(SEMANTIC_COLOR_RGB.items())},
        "camera": dict(camera_meta),
        "projection_frame": dict(frame_meta),
        "solver_trace": {
            "count_predicate": str(predicate_kind),
            "scope": {
                "lane_key": str(target_lane_key),
                "lane_label": str(LANE_LABELS[str(target_lane_key)]),
            },
            "ordered_pair": {
                "first_color_name": str(target_color_name),
                "second_color_name": str(second_target_color_name),
                "first_shape_type": str(target_shape_pair[0]),
                "second_shape_type": str(target_shape_pair[1]),
            },
            "target_count": int(target_count),
            "target_object_ids": [str(object_id) for object_id in target_object_ids],
            "target_pair_object_id_pairs": [list(pair) for pair in target_pair_object_id_pairs],
            "answer_value": int(target_count),
            "unique_integer_answer": True,
        },
    }


def build_between_marked_items_count_dataset(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    render_params: Any,
    axes: ResolvedConveyorAxes,
    predicate_kind: str,
    namespace: str,
) -> dict[str, Any]:
    """Build a straight-conveyor dataset for counting objects between marked anchors."""

    rng = spawn_rng(int(instance_seed), f"{namespace}.dataset")
    layout_orientation, layout_orientation_probabilities = _resolve_layout_orientation(
        params=params,
        rng=rng,
        render_params=render_params,
    )
    lane_keys = _lane_keys_for_orientation(str(layout_orientation))
    target_lane_key, target_lane_probabilities = _resolve_target_lane(params=params, rng=rng, lane_keys=lane_keys)
    target_count, target_count_probabilities = _resolve_between_items_count(
        params=params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.{target_lane_key}.target_count",
    )
    target_total_min = max(6, int(target_count) + 2)
    target_total = int(rng.randrange(int(target_total_min), 9))
    lane_counts: Dict[str, int] = {
        str(lane_key): (int(target_total) if str(lane_key) == str(target_lane_key) else int(rng.randrange(3, 9)))
        for lane_key in lane_keys
    }
    start_index = int(rng.randrange(0, int(target_total) - int(target_count) - 1))
    end_index = int(start_index + int(target_count) + 1)
    between_indices = set(range(int(start_index) + 1, int(end_index)))
    dimension_scale = float(params.get("object_dimension_scale", group_default(gen_defaults, "object_dimension_scale", 0.74)))

    if str(predicate_kind) == PREDICATE_BETWEEN_OBJECT_ANCHORS:
        first_shape, target_shape_probabilities = _resolve_shape(
            params=params,
            rng=rng,
            support=CONVEYOR_COLOR_READOUT_SHAPE_TYPES,
        )
        second_explicit = params.get("second_target_shape_type")
        second_candidates = list(compatible_distractor_pool(str(first_shape), support=CONVEYOR_COLOR_READOUT_SHAPE_TYPES))
        if second_explicit is not None:
            second_shape = str(second_explicit)
            if second_shape not in set(second_candidates):
                raise ValueError(f"unsupported second_target_shape_type: {second_shape}")
        else:
            second_shape = str(second_candidates[int(rng.randrange(len(second_candidates)))])
        filler_pool = list(compatible_distractor_pool(str(first_shape), support=CONVEYOR_COLOR_READOUT_SHAPE_TYPES))
        filler_pool = list(compatible_distractor_pool(str(second_shape), support=filler_pool))
        filler_symbols = [str(shape) for shape in filler_pool if str(shape) not in {str(first_shape), str(second_shape)}]
        target_shape = str(first_shape)
        target_shape_pair = (str(first_shape), str(second_shape))
        target_object_name_pair = (public_object_name(str(first_shape)), public_object_name(str(second_shape)))
        target_object_plural_pair = (public_object_plural(str(first_shape)), public_object_plural(str(second_shape)))
        target_color_name = ""
        second_target_color_name = ""
        target_color_probabilities = {}
        color_palette = list(sample_visual_color_names(rng, palette_size=4))
        shape_palette = [str(first_shape), str(second_shape), *list(filler_symbols)]
    else:
        raise ValueError(f"unsupported straight conveyor between-anchor predicate: {predicate_kind}")
    if not filler_symbols:
        raise ValueError("between-anchor task needs at least one filler symbol")
    if not color_palette:
        raise ValueError("between-anchor task needs a visual color palette")

    target_sequence: list[str] = []
    for index in range(int(target_total)):
        if int(index) == int(start_index):
            target_sequence.append(str(target_shape_pair[0]))
        elif int(index) == int(end_index):
            target_sequence.append(str(target_shape_pair[1]))
        else:
            target_sequence.append(str(rng.choice(filler_symbols)))

    object_specs: List[Dict[str, Any]] = []
    object_sequences_by_lane: Dict[str, list[str]] = {}
    target_object_ids: list[str] = []
    start_anchor_object_id = ""
    end_anchor_object_id = ""

    for lane_key in lane_keys:
        slots = _ordered_slot_positions_for_lane(
            rng=rng,
            layout_orientation=str(layout_orientation),
            lane_key=str(lane_key),
            count=int(lane_counts[str(lane_key)]),
        )
        if str(lane_key) == str(target_lane_key):
            sequence = tuple(target_sequence)
        else:
            sequence = tuple(str(shape_palette[int(rng.randrange(len(shape_palette)))]) for _ in slots)

        lane_object_ids: list[str] = []
        for index, slot in enumerate(slots):
            object_id = f"obj_{len(object_specs):03d}"
            lane_object_ids.append(str(object_id))
            shape_type = str(sequence[index])
            color_name = str(rng.choice(color_palette))

            is_start_anchor = str(lane_key) == str(target_lane_key) and int(index) == int(start_index)
            is_end_anchor = str(lane_key) == str(target_lane_key) and int(index) == int(end_index)
            matches_query = str(lane_key) == str(target_lane_key) and int(index) in between_indices
            if is_start_anchor:
                start_anchor_object_id = str(object_id)
                count_role = "start_anchor"
            elif is_end_anchor:
                end_anchor_object_id = str(object_id)
                count_role = "end_anchor"
            elif matches_query:
                count_role = "between_counted_object"
                target_object_ids.append(str(object_id))
            else:
                count_role = "same_lane_distractor" if str(lane_key) == str(target_lane_key) else "lane_distractor"
            object_specs.append(
                _make_object_spec(
                    rng=rng,
                    object_id=str(object_id),
                    shape_type=str(shape_type),
                    color_name=str(color_name),
                    lane_key=str(lane_key),
                    layout_orientation=str(layout_orientation),
                    slot=slot,
                    matches_query=bool(matches_query),
                    count_role=str(count_role),
                    dimension_scale=float(dimension_scale),
                )
            )
        object_sequences_by_lane[str(lane_key)] = list(lane_object_ids)

    if not start_anchor_object_id or not end_anchor_object_id:
        raise ValueError("between-anchor task failed to assign marked anchors")
    if len(target_object_ids) != int(target_count):
        raise ValueError("between-anchor dataset target count mismatch")

    camera, frame, camera_meta, frame_meta = _finalize_camera_and_projection(
        rng=rng,
        render_params=render_params,
        layout_orientation=str(layout_orientation),
        object_specs=object_specs,
    )
    finalized_specs = _screen_finalize_specs(object_specs=object_specs, camera=camera, frame=frame)
    shape_counts = Counter(str(spec["shape_type"]) for spec in finalized_specs)
    color_counts = Counter(str(spec["color_name"]) for spec in finalized_specs)
    lane_counts_final = Counter(str(spec["lane_key"]) for spec in finalized_specs)
    target_lane_object_ids = [
        str(spec["object_id"])
        for spec in finalized_specs
        if str(spec["lane_key"]) == str(target_lane_key)
    ]
    start_spec = next(spec for spec in finalized_specs if str(spec["object_id"]) == str(start_anchor_object_id))
    end_spec = next(spec for spec in finalized_specs if str(spec["object_id"]) == str(end_anchor_object_id))
    marked_anchor_records = [
        {
            "anchor_label": "A",
            "anchor_role": "start_anchor",
            "object_id": str(start_anchor_object_id),
            "shape_type": str(start_spec["shape_type"]),
            "object_name": public_object_name(str(start_spec["shape_type"])),
            "object_plural": public_object_plural(str(start_spec["shape_type"])),
            "color_name": str(start_spec["color_name"]),
            "color_label": semantic_color_label(str(start_spec["color_name"])),
        },
        {
            "anchor_label": "B",
            "anchor_role": "end_anchor",
            "object_id": str(end_anchor_object_id),
            "shape_type": str(end_spec["shape_type"]),
            "object_name": public_object_name(str(end_spec["shape_type"])),
            "object_plural": public_object_plural(str(end_spec["shape_type"])),
            "color_name": str(end_spec["color_name"]),
            "color_label": semantic_color_label(str(end_spec["color_name"])),
        },
    ]
    return {
        "scene_id": SCENE_ID,
        "scene_variant": str(axes.scene_variant),
        "layout_family": "straight_parallel_conveyors",
        "layout_orientation": str(layout_orientation),
        "layout_orientation_probabilities": dict(layout_orientation_probabilities),
        "predicate_kind": str(predicate_kind),
        "lane_records": _lane_records(str(layout_orientation)),
        "target_lane_key": str(target_lane_key),
        "target_lane_label": str(LANE_LABELS[str(target_lane_key)]),
        "target_belt_key": str(target_lane_key),
        "target_belt_label": str(LANE_LABELS[str(target_lane_key)]),
        "target_shape_type": str(target_shape),
        "target_object_name": public_object_name(str(target_shape)),
        "target_object_plural": public_object_plural(str(target_shape)),
        "target_shape_pair": [str(shape) for shape in target_shape_pair],
        "target_object_name_pair": [str(name) for name in target_object_name_pair],
        "target_object_plural_pair": [str(name) for name in target_object_plural_pair],
        "target_color_name": str(target_color_name),
        "second_target_color_name": str(second_target_color_name),
        "target_color_label": semantic_color_label(str(target_color_name)) if str(target_color_name) else "",
        "second_target_color_label": semantic_color_label(str(second_target_color_name)) if str(second_target_color_name) else "",
        "start_anchor_object_id": str(start_anchor_object_id),
        "end_anchor_object_id": str(end_anchor_object_id),
        "marked_anchor_object_ids": [str(start_anchor_object_id), str(end_anchor_object_id)],
        "marked_anchor_records": [dict(record) for record in marked_anchor_records],
        "between_object_ids": [str(object_id) for object_id in target_object_ids],
        "start_anchor_index": int(start_index),
        "end_anchor_index": int(end_index),
        "answer_value": int(target_count),
        "target_count": int(target_count),
        "target_object_ids": [str(object_id) for object_id in target_object_ids],
        "target_lane_object_ids": list(target_lane_object_ids),
        "target_belt_object_ids": list(target_lane_object_ids),
        "object_sequences_by_lane": {str(key): list(value) for key, value in object_sequences_by_lane.items()},
        "object_count": int(len(finalized_specs)),
        "object_specs": [dict(spec) for spec in finalized_specs],
        "shape_counts": {str(key): int(value) for key, value in sorted(shape_counts.items())},
        "color_counts": {str(key): int(value) for key, value in sorted(color_counts.items())},
        "lane_counts": {str(key): int(value) for key, value in sorted(lane_counts_final.items())},
        "belt_counts": {str(key): int(value) for key, value in sorted(lane_counts_final.items())},
        "target_shape_type_probabilities": dict(target_shape_probabilities),
        "target_color_name_probabilities": dict(target_color_probabilities),
        "target_count_probabilities": dict(target_count_probabilities),
        "lane_count_probabilities": {str(lane_key): {str(count): 1.0} for lane_key, count in lane_counts.items()},
        "target_lane_key_probabilities": dict(target_lane_probabilities),
        "target_belt_key_probabilities": dict(target_lane_probabilities),
        "target_belt_probabilities": dict(target_lane_probabilities),
        "semantic_color_palette": {str(key): list(value) for key, value in sorted(SEMANTIC_COLOR_RGB.items())},
        "camera": dict(camera_meta),
        "projection_frame": dict(frame_meta),
        "solver_trace": {
            "count_predicate": str(predicate_kind),
            "scope": {
                "lane_key": str(target_lane_key),
                "lane_label": str(LANE_LABELS[str(target_lane_key)]),
            },
            "anchors": [dict(record) for record in marked_anchor_records],
            "start_anchor_object_id": str(start_anchor_object_id),
            "end_anchor_object_id": str(end_anchor_object_id),
            "between_object_ids": [str(object_id) for object_id in target_object_ids],
            "target_count": int(target_count),
            "target_object_ids": [str(object_id) for object_id in target_object_ids],
            "answer_value": int(target_count),
            "unique_integer_answer": True,
        },
    }


__all__ = [
    "ARITHMETIC_DIFFERENCE",
    "ARITHMETIC_SUM",
    "LAYOUT_HORIZONTAL",
    "LAYOUT_VERTICAL",
    "PREDICATE_BELT_TOTAL",
    "PREDICATE_COLOR",
    "PREDICATE_COLOR_TYPE",
    "PREDICATE_COLOR_TRANSFER",
    "PREDICATE_BETWEEN_OBJECT_ANCHORS",
    "PREDICATE_ORDERED_COLOR_PAIR",
    "PREDICATE_ORDERED_OBJECT_PAIR",
    "PREDICATE_OBJECT_TYPE",
    "PREDICATE_OBJECT_TYPE_ARITHMETIC",
    "PREDICATE_OBJECT_TYPE_TRANSFER",
    "ResolvedConveyorAxes",
    "TRANSFER_OPERATION",
    "build_between_marked_items_count_dataset",
    "build_belt_total_count_dataset",
    "build_lane_count_arithmetic_dataset",
    "build_ordered_pair_count_dataset",
    "build_scoped_belt_count_dataset",
    "build_transfer_count_dataset",
    "resolve_conveyor_axes",
]
