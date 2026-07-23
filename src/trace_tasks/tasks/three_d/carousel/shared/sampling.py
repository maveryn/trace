"""Sampling helpers for conveyor carousel scopes and object layouts."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.named_colors import sample_named_color_palette
from trace_tasks.tasks.three_d.shared.camera_projection import (
    CameraSpec,
    build_projection_frame,
    project_screen,
    vec_cross,
    vec_norm,
    vec_sub,
)
from trace_tasks.tasks.three_d.shared.object_resources import OBJECT_CLUSTER_DIMENSIONS
from trace_tasks.tasks.three_d.shared.projected_object_geometry import object_reference_points
from trace_tasks.tasks.three_d.shared.object_confusions import compatible_distractor_pool
from trace_tasks.tasks.three_d.shared.semantic_colors import sample_readout_palette as sample_semantic_readout_palette
from trace_tasks.tasks.three_d.shared.task_support import (
    resolve_axis_variant_for_namespace,
    resolve_count_for_namespace,
    shuffled_repeated_support,
)

from .state import (
    BELT_GEOMETRY,
    BELT_KEYS,
    BELT_LABELS,
    CONVEYOR_COLOR_READOUT_SHAPE_TYPES,
    CONVEYOR_OBJECT_SHAPE_TYPES,
    SCENE_ID,
    SEMANTIC_COLOR_RGB,
    SEMANTIC_COLOR_SUPPORT,
    SUPPORTED_SCENE_VARIANTS,
    public_object_name,
    public_object_plural,
    semantic_color_label,
)


PREDICATE_OBJECT_TYPE = "object_type"
PREDICATE_COLOR = "color"
PREDICATE_COLOR_TYPE = "color_type"
PREDICATE_BELT_TOTAL = "belt_total"
PREDICATE_OBJECT_TYPE_ARITHMETIC = "object_type_count_arithmetic"
PREDICATE_ORDERED_OBJECT_PAIR = "ordered_object_pair"
PREDICATE_ORDERED_COLOR_PAIR = "ordered_color_pair"
PREDICATE_OBJECT_TYPE_TRANSFER = "object_type_transfer"
PREDICATE_COLOR_TRANSFER = "color_transfer"
PREDICATE_BETWEEN_OBJECT_ANCHORS = "between_object_anchors"
ARITHMETIC_SUM = "sum"
ARITHMETIC_DIFFERENCE = "difference"
TRANSFER_OPERATION = "move_to_destination"

CAMERA_YAW_BANDS_DEGREES: Tuple[Tuple[float, float], ...] = (
    (-66.0, -42.0),
    (42.0, 66.0),
    (-138.0, -114.0),
    (114.0, 138.0),
)


@dataclass(frozen=True)
class ResolvedConveyorAxes:
    """Resolved conveyor scene axes for one generated instance."""

    scene_variant: str
    scene_variant_probabilities: Dict[str, float]


def _uniform_string_probability_map(values: Sequence[str], *, selected: str | None = None) -> Dict[str, float]:
    support = tuple(str(value) for value in values)
    if selected is not None:
        return {str(value): (1.0 if str(value) == str(selected) else 0.0) for value in support}
    probability = 1.0 / float(max(1, len(support)))
    return {str(value): float(probability) for value in support}


def _configured_int(params: Mapping[str, Any], gen_defaults: Mapping[str, Any], key: str, default: int) -> int:
    return int(params.get(str(key), group_default(gen_defaults, str(key), int(default))))


def resolve_conveyor_axes(
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
) -> ResolvedConveyorAxes:
    """Resolve the conveyor carousel scene variant."""

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


def _sample_readout_palette(rng: Any, *, target_color: str, size: int) -> Tuple[str, ...]:
    return sample_semantic_readout_palette(
        rng,
        target_color=str(target_color),
        support=SEMANTIC_COLOR_SUPPORT,
        size=int(size),
    )


def _sample_shape(rng: Any, support: Sequence[str], *, exclude: Sequence[str] = ()) -> str:
    choices = [str(shape) for shape in support if str(shape) not in set(str(item) for item in exclude)]
    if not choices:
        raise ValueError("empty conveyor object shape support")
    return str(choices[int(rng.randrange(len(choices)))])


def _resolve_target_shape(
    *,
    params: Mapping[str, Any],
    rng: Any,
    support: Sequence[str] = CONVEYOR_OBJECT_SHAPE_TYPES,
) -> tuple[str, Dict[str, float]]:
    explicit = params.get("target_shape_type")
    support = tuple(str(shape) for shape in support)
    if explicit is not None:
        shape = str(explicit)
        if shape not in set(support):
            raise ValueError(f"unsupported target_shape_type: {shape}")
        return shape, _uniform_string_probability_map(support, selected=shape)
    shape = _sample_shape(rng, support)
    return str(shape), _uniform_string_probability_map(support)


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


def _resolve_target_belt(
    *,
    params: Mapping[str, Any],
    rng: Any,
) -> tuple[str, Dict[str, float]]:
    explicit = params.get("target_belt_key")
    support = tuple(str(key) for key in BELT_KEYS)
    if explicit is not None:
        belt_key = str(explicit)
        if belt_key not in set(support):
            raise ValueError(f"unsupported target_belt_key: {belt_key}")
        return belt_key, _uniform_string_probability_map(support, selected=belt_key)
    belt_key = str(support[int(rng.randrange(len(support)))])
    return belt_key, _uniform_string_probability_map(support)


def _resolve_transfer_belts(
    *,
    params: Mapping[str, Any],
    rng: Any,
) -> tuple[str, str, Dict[str, float], Dict[str, float]]:
    support = tuple(str(key) for key in BELT_KEYS)
    source_explicit = params.get("source_belt_key")
    destination_explicit = params.get("destination_belt_key")
    if source_explicit is not None or destination_explicit is not None:
        if source_explicit is None or destination_explicit is None:
            raise ValueError("source_belt_key and destination_belt_key must be provided together")
        source = str(source_explicit)
        destination = str(destination_explicit)
        if source == destination or source not in set(support) or destination not in set(support):
            raise ValueError("carousel transfer source and destination must be distinct visible belt keys")
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


def _resolve_belt_total_target_count(
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
    belt_key: str,
) -> tuple[int, Dict[str, float]]:
    """Resolve answer count with belt-specific support for total belt counts."""

    default_max = 8 if str(belt_key) == "inner" else 10
    count, probabilities = resolve_count_for_namespace(
        params,
        namespace=f"{namespace}.{belt_key}.target_count",
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        key="target_count",
        default_min=1,
        default_max=int(default_max),
        lower=1,
        upper=int(default_max),
    )
    return int(count), dict(probabilities)


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
        default_max=int(params.get("answer_value_max", group_default(gen_defaults, "answer_value_max", int(destination_cap)))),
        lower=2,
        upper=int(destination_cap),
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
            raise ValueError("destination_existing_count exceeds belt cap")
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
        raise ValueError(f"no carousel transfer operands for answer {answer_value}")
    moved_count, destination_existing_count = pairs[int(rng.randrange(len(pairs)))]
    return (
        int(answer_value),
        int(moved_count),
        int(destination_existing_count),
        dict(answer_probabilities),
        {f"{moved_count},{destination_existing_count}": 1.0},
    )


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


def _belt_max_object_count(belt_key: str) -> int:
    return 8 if str(belt_key) == "inner" else 12


def _sample_scoped_belt_totals(
    *,
    rng: Any,
    target_belt_key: str,
    target_count: int,
) -> dict[str, int]:
    totals: dict[str, int] = {}
    for belt_key in BELT_KEYS:
        max_count = int(_belt_max_object_count(str(belt_key)))
        if str(belt_key) == str(target_belt_key):
            min_count = max(2, int(target_count) + 1)
        else:
            min_count = 2
        min_count = min(int(max_count), int(min_count))
        totals[str(belt_key)] = int(rng.randrange(int(min_count), int(max_count) + 1))
    return totals


def _resolve_arithmetic_operand_counts(
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    rng: Any,
    instance_seed: int,
    namespace: str,
    operation: str,
) -> tuple[int, int, int, Dict[str, float], Dict[str, float]]:
    """Resolve two belt operand counts and the arithmetic answer."""

    operation = str(operation)
    if operation == ARITHMETIC_SUM:
        default_min = _configured_int(params, gen_defaults, "sum_answer_min", 1)
        default_max = _configured_int(params, gen_defaults, "sum_answer_max", 12)
        lower, upper = 1, 12
    elif operation == ARITHMETIC_DIFFERENCE:
        default_min = _configured_int(params, gen_defaults, "difference_answer_min", 1)
        default_max = _configured_int(params, gen_defaults, "difference_answer_max", 5)
        lower, upper = 1, 5
    else:
        raise ValueError(f"unsupported carousel count arithmetic operation: {operation}")

    answer_min = max(int(lower), min(int(upper), int(params.get("answer_value_min", group_default(gen_defaults, "answer_value_min", int(default_min))))))
    answer_max = max(answer_min, min(int(upper), int(params.get("answer_value_max", group_default(gen_defaults, "answer_value_max", int(default_max))))))
    answer_support = tuple(range(int(answer_min), int(answer_max) + 1))
    explicit_answer = params.get("answer_value")
    if explicit_answer is not None:
        answer_value = int(explicit_answer)
        if int(answer_value) not in set(answer_support):
            raise ValueError(f"unsupported carousel arithmetic answer_value: {answer_value}")
        answer_probabilities = {str(answer_value): 1.0}
    else:
        answer_value, answer_probabilities = resolve_count_for_namespace(
            params,
            namespace=f"{namespace}.{operation}.answer_value",
            gen_defaults=gen_defaults,
            instance_seed=int(instance_seed),
            key="answer_value",
            default_min=int(answer_min),
            default_max=int(answer_max),
            lower=int(lower),
            upper=int(upper),
        )
        answer_probabilities = {str(value): 1.0 / float(len(answer_support)) for value in answer_support}
    explicit_first = params.get("first_scope_count")
    explicit_second = params.get("second_scope_count")
    if explicit_first is not None or explicit_second is not None:
        if explicit_first is None or explicit_second is None:
            raise ValueError("first_scope_count and second_scope_count must be provided together")
        first_count = int(explicit_first)
        second_count = int(explicit_second)
        if not (0 <= first_count <= 6 and 0 <= second_count <= 6):
            raise ValueError("carousel arithmetic operand counts must be in 0..6")
        expected = first_count + second_count if operation == ARITHMETIC_SUM else abs(first_count - second_count)
        if int(expected) != int(answer_value):
            raise ValueError("explicit carousel arithmetic operands do not match answer_value")
        return int(first_count), int(second_count), int(answer_value), dict(answer_probabilities), {"first_scope_count": 1.0, "second_scope_count": 1.0}

    if operation == ARITHMETIC_SUM:
        pairs = [(a, int(answer_value) - a) for a in range(0, 7) if 0 <= int(answer_value) - a <= 6]
    else:
        pairs = [(a, b) for a in range(0, 7) for b in range(0, 7) if abs(a - b) == int(answer_value)]
    if not pairs:
        raise ValueError(f"no carousel arithmetic operands for answer {answer_value}")
    first_count, second_count = pairs[int(rng.randrange(len(pairs)))]
    operand_probabilities = {
        f"{first_count},{second_count}": 1.0,
    }
    return int(first_count), int(second_count), int(answer_value), dict(answer_probabilities), operand_probabilities


def _sample_arithmetic_belt_totals(
    *,
    rng: Any,
    operand_counts_by_belt: Mapping[str, int],
) -> dict[str, int]:
    totals: dict[str, int] = {}
    for belt_key in BELT_KEYS:
        target_count = int(operand_counts_by_belt[str(belt_key)])
        max_count = int(_belt_max_object_count(str(belt_key)))
        min_count = min(max_count, max(2, target_count + (1 if target_count < max_count else 0)))
        totals[str(belt_key)] = int(rng.randrange(int(min_count), int(max_count) + 1))
    return totals


def _belt_point(belt_key: str, theta: float, radial_offset: float = 0.0) -> tuple[float, float]:
    geometry = BELT_GEOMETRY[str(belt_key)]
    radius_x = float(geometry["radius_x"]) + float(radial_offset)
    radius_y = float(geometry["radius_y"]) + float(radial_offset) * 0.62
    return (round(radius_x * math.cos(float(theta)), 4), round(radius_y * math.sin(float(theta)), 4))


def _slot_positions_for_belt(
    *,
    rng: Any,
    belt_key: str,
    slots_per_belt: int,
) -> list[tuple[float, float, float]]:
    start = float(rng.uniform(0.0, 2.0 * math.pi))
    slots: list[tuple[float, float, float]] = []
    for index in range(int(slots_per_belt)):
        theta = start + (2.0 * math.pi * float(index) / float(slots_per_belt)) + rng.uniform(-0.035, 0.035)
        width = float(BELT_GEOMETRY[str(belt_key)]["band_width"])
        radial_offset = rng.uniform(-0.18 * width, 0.18 * width)
        x, y = _belt_point(str(belt_key), float(theta), radial_offset=float(radial_offset))
        slots.append((float(x), float(y), round(math.degrees(float(theta)) % 360.0, 3)))
    rng.shuffle(slots)
    return slots


def _ordered_slot_positions_for_belt(
    *,
    rng: Any,
    belt_key: str,
    count: int,
) -> list[tuple[float, float, float]]:
    """Return belt slots in carousel arrow order for ordered-neighbor tasks."""

    count = int(count)
    if count <= 0:
        return []
    start = float(rng.uniform(0.0, 2.0 * math.pi))
    width = float(BELT_GEOMETRY[str(belt_key)]["band_width"])
    jitter = min(0.03, (math.pi / float(max(2, count))) * 0.18)
    slots: list[tuple[float, float, float]] = []
    for index in range(count):
        theta = start + (2.0 * math.pi * float(index) / float(count)) + rng.uniform(-float(jitter), float(jitter))
        radial_offset = rng.uniform(-0.14 * width, 0.14 * width)
        x, y = _belt_point(str(belt_key), float(theta), radial_offset=float(radial_offset))
        slots.append((float(x), float(y), round(math.degrees(float(theta)) % 360.0, 3)))
    return slots


def _ordered_pair_symbol_sequence(
    *,
    rng: Any,
    first_symbol: str,
    second_symbol: str,
    filler_symbols: Sequence[str],
    pair_count: int,
    total_count: int,
    circular: bool,
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
    observed = 0
    for index, symbol in enumerate(sequence):
        if not circular and index == len(sequence) - 1:
            continue
        next_index = index + 1
        if next_index >= len(sequence):
            next_index = 0
        next_symbol = sequence[next_index]
        if str(symbol) == str(first_symbol) and str(next_symbol) == str(second_symbol):
            observed += 1
    if int(observed) != int(pair_count):
        return _ordered_pair_symbol_sequence(
            rng=rng,
            first_symbol=str(first_symbol),
            second_symbol=str(second_symbol),
            filler_symbols=fillers,
            pair_count=int(pair_count),
            total_count=int(total_count),
            circular=bool(circular),
        )
    return tuple(sequence)


def _angular_distance_degrees(a: float, b: float) -> float:
    distance = abs(float(a) - float(b)) % 360.0
    return min(float(distance), 360.0 - float(distance))


def _object_dimensions(shape_type: str, *, scale: float) -> tuple[float, float, float]:
    base = OBJECT_CLUSTER_DIMENSIONS.get(str(shape_type), (0.52, 0.52, 0.52))
    return tuple(round(float(value) * float(scale), 4) for value in base)


def _make_object_spec(
    *,
    rng: Any,
    object_id: str,
    shape_type: str,
    color_name: str,
    slot: Sequence[float],
    belt_key: str,
    matches_query: bool,
    count_role: str,
    dimension_scale: float,
) -> Dict[str, Any]:
    """Create one countable conveyor carousel object spec."""

    dimensions = _object_dimensions(str(shape_type), scale=float(dimension_scale))
    height = float(dimensions[2])
    color_rgb = SEMANTIC_COLOR_RGB[str(color_name)]
    theta_degrees = float(slot[2])
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
        "belt_key": str(belt_key),
        "belt_label": str(BELT_LABELS[str(belt_key)]),
        "angular_position_degrees": round(float(theta_degrees), 3),
        "color_name": str(color_name),
        "prompt_color_name": str(color_name),
        "fill_rgb": [int(channel) for channel in color_rgb],
        "semantic_color": True,
        "dimensions_xyz": [float(value) for value in dimensions],
        "world_xyz": [round(float(slot[0]), 4), round(float(slot[1]), 4), round(0.08 + height * 0.5, 4)],
        "base_xyz": [round(float(slot[0]), 4), round(float(slot[1]), 4), 0.08],
        "orientation_deg": round(float(theta_degrees + 90.0 + rng.uniform(-18.0, 18.0)), 3),
        "render_order_bias": round(float(rng.uniform(-0.015, 0.015)), 5),
        "renderer_id": "object_scene_shape",
        "object_role": "target" if bool(matches_query) else "distractor",
    }


def _sample_slot(
    slots_by_belt: Dict[str, list[tuple[float, float, float]]],
    used_angles_by_belt: Dict[str, list[float]],
    *,
    belt_key: str,
    min_angle_gap_degrees: float,
) -> tuple[float, float, float]:
    slots = slots_by_belt.get(str(belt_key), [])
    if not slots:
        raise ValueError(f"no free conveyor carousel slots for {belt_key}")
    used_angles = used_angles_by_belt.setdefault(str(belt_key), [])
    fallback_index = len(slots) - 1
    selected_index = fallback_index
    for index in range(len(slots) - 1, -1, -1):
        candidate_angle = float(slots[index][2])
        if all(_angular_distance_degrees(candidate_angle, used_angle) >= float(min_angle_gap_degrees) for used_angle in used_angles):
            selected_index = index
            break
    slot = slots.pop(selected_index)
    used_angles.append(float(slot[2]))
    return slot


def _sample_other_belt_slot(
    slots_by_belt: Dict[str, list[tuple[float, float, float]]],
    used_angles_by_belt: Dict[str, list[float]],
    *,
    rng: Any,
    target_belt_key: str,
    min_angle_gap_degrees: float,
) -> tuple[str, tuple[float, float, float]]:
    candidates = [
        str(belt_key)
        for belt_key in BELT_KEYS
        if str(belt_key) != str(target_belt_key) and bool(slots_by_belt.get(str(belt_key)))
    ]
    if not candidates:
        raise ValueError("no non-target conveyor carousel slots available")
    belt_key = str(candidates[int(rng.randrange(len(candidates)))])
    return belt_key, _sample_slot(
        slots_by_belt,
        used_angles_by_belt,
        belt_key=belt_key,
        min_angle_gap_degrees=float(min_angle_gap_degrees),
    )


def _sample_carousel_camera(rng: Any, yaw_band_degrees: Sequence[float]) -> CameraSpec:
    yaw_degrees = float(rng.uniform(float(yaw_band_degrees[0]), float(yaw_band_degrees[1])))
    pitch_degrees = float(rng.uniform(39.0, 50.0))
    distance = float(rng.uniform(7.8, 9.2))
    yaw = math.radians(float(yaw_degrees))
    pitch = math.radians(float(pitch_degrees))
    target = (0.0, 0.0, 0.58)
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


def _carousel_reference_points() -> list[tuple[float, float, float]]:
    points: list[tuple[float, float, float]] = []
    for belt_key in BELT_KEYS:
        width = float(BELT_GEOMETRY[str(belt_key)]["band_width"])
        for angle_index in range(32):
            theta = 2.0 * math.pi * float(angle_index) / 32.0
            for radial_offset in (-0.5 * width, 0.5 * width):
                x, y = _belt_point(str(belt_key), theta, radial_offset=float(radial_offset))
                points.append((float(x), float(y), 0.04))
    points.extend([(0.0, 0.0, 0.04), (0.0, 0.0, 0.62)])
    return points


def _finalize_camera_and_projection(
    *,
    rng: Any,
    render_params: Any,
    object_specs: Sequence[Mapping[str, Any]],
) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    """Sample camera and bind projection metadata for finalized objects."""

    band_index = int(rng.randrange(len(CAMERA_YAW_BANDS_DEGREES)))
    yaw_band = CAMERA_YAW_BANDS_DEGREES[int(band_index)]
    camera = _sample_carousel_camera(rng, yaw_band)
    reference_points = [point for spec in object_specs for point in object_reference_points(spec)]
    frame = build_projection_frame(
        camera=camera,
        render_params=render_params,
        point_worlds=[*_carousel_reference_points(), *reference_points],
    )
    camera_meta = {
        "camera_position": [round(float(value), 4) for value in camera.camera_position],
        "target": [round(float(value), 4) for value in camera.target],
        "yaw_degrees": round(float(camera.yaw_degrees), 4),
        "yaw_band_degrees": [round(float(value), 4) for value in yaw_band],
        "yaw_band_index": int(band_index),
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
    camera: Any,
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


def build_belt_count_dataset(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    render_params: Any,
    axes: ResolvedConveyorAxes,
    predicate_kind: str,
    namespace: str,
) -> dict[str, Any]:
    """Build an elliptical conveyor carousel dataset for one belt-scoped count."""

    rng = spawn_rng(int(instance_seed), f"{namespace}.dataset")
    target_belt_key, target_belt_probabilities = _resolve_target_belt(params=params, rng=rng)
    target_belt_label = str(BELT_LABELS[str(target_belt_key)])
    if str(predicate_kind) == PREDICATE_BELT_TOTAL:
        target_count, target_count_probabilities = _resolve_belt_total_target_count(
            params=params,
            gen_defaults=gen_defaults,
            instance_seed=int(instance_seed),
            namespace=str(namespace),
            belt_key=str(target_belt_key),
        )
    else:
        target_count, target_count_probabilities = resolve_count_for_namespace(
            params,
            namespace=f"{namespace}.target_count",
            gen_defaults=gen_defaults,
            instance_seed=int(instance_seed),
            key="target_count",
            default_min=0,
            default_max=5,
            lower=0,
            upper=5,
        )
    object_count_probabilities: Dict[str, float] = {}
    slots_per_belt = max(24, _configured_int(params, gen_defaults, "slots_per_belt", 30))
    min_angle_gap_degrees = float(params.get("min_same_belt_angle_gap_degrees", group_default(gen_defaults, "min_same_belt_angle_gap_degrees", 20.0)))
    dimension_scale = float(params.get("object_dimension_scale", group_default(gen_defaults, "object_dimension_scale", 0.64)))
    slots_by_belt = {
        str(belt_key): _slot_positions_for_belt(rng=rng, belt_key=str(belt_key), slots_per_belt=int(slots_per_belt))
        for belt_key in BELT_KEYS
    }
    used_angles_by_belt: Dict[str, list[float]] = {str(belt_key): [] for belt_key in BELT_KEYS}
    belt_records = [
        {
            "belt_key": str(belt_key),
            "belt_label": str(BELT_LABELS[str(belt_key)]),
            "geometry": dict(BELT_GEOMETRY[str(belt_key)]),
            "slot_count": int(slots_per_belt),
        }
        for belt_key in BELT_KEYS
    ]

    object_specs: List[Dict[str, Any]] = []
    target_object_ids: list[str] = []

    if str(predicate_kind) == PREDICATE_BELT_TOTAL:
        target_shape, target_shape_probabilities = _resolve_target_shape(
            params=params,
            rng=rng,
            support=CONVEYOR_OBJECT_SHAPE_TYPES,
        )
        active_colors = sample_named_color_palette(rng, palette_size=4)
        color_names = tuple(str(name) for name, _rgb in active_colors)
        if not color_names:
            raise ValueError("empty conveyor visual color palette")
        target_color_name = ""
        target_color_probabilities: Dict[str, float] = {}
        other_belt_key = str("outer" if str(target_belt_key) == "inner" else "inner")
        other_default_max = 8 if other_belt_key == "inner" else 10
        other_count, _other_count_probabilities = resolve_count_for_namespace(
            params,
            namespace=f"{namespace}.{other_belt_key}.distractor_count",
            gen_defaults=gen_defaults,
            instance_seed=int(instance_seed),
            key="other_belt_object_count",
            default_min=1,
            default_max=int(other_default_max),
            lower=1,
            upper=int(other_default_max),
        )
        for index in range(int(target_count)):
            slot = _sample_slot(
                slots_by_belt,
                used_angles_by_belt,
                belt_key=target_belt_key,
                min_angle_gap_degrees=float(min_angle_gap_degrees),
            )
            object_id = f"obj_{len(object_specs):03d}"
            target_object_ids.append(object_id)
            object_specs.append(
                _make_object_spec(
                    rng=rng,
                    object_id=object_id,
                    shape_type=str(target_shape),
                    color_name=str(rng.choice(color_names)),
                    slot=slot,
                    belt_key=target_belt_key,
                    matches_query=True,
                    count_role="target",
                    dimension_scale=float(dimension_scale),
                )
            )
        for index in range(int(other_count)):
            slot = _sample_slot(
                slots_by_belt,
                used_angles_by_belt,
                belt_key=other_belt_key,
                min_angle_gap_degrees=float(min_angle_gap_degrees),
            )
            object_specs.append(
                _make_object_spec(
                    rng=rng,
                    object_id=f"obj_{len(object_specs):03d}",
                    shape_type=str(target_shape),
                    color_name=str(rng.choice(color_names)),
                    slot=slot,
                    belt_key=other_belt_key,
                    matches_query=False,
                    count_role="other_belt_distractor",
                    dimension_scale=float(dimension_scale),
                )
            )
        object_count = len(object_specs)
        object_count_probabilities = {str(object_count): 1.0}
        target_prompt_phrase = "objects"
    elif str(predicate_kind) == PREDICATE_OBJECT_TYPE:
        scoped_belt_totals = _sample_scoped_belt_totals(
            rng=rng,
            target_belt_key=str(target_belt_key),
            target_count=int(target_count),
        )
        target_shape, target_shape_probabilities = _resolve_target_shape(
            params=params,
            rng=rng,
            support=CONVEYOR_OBJECT_SHAPE_TYPES,
        )
        distractor_shapes = list(
            compatible_distractor_pool(str(target_shape), support=CONVEYOR_OBJECT_SHAPE_TYPES)
        )
        active_colors = sample_named_color_palette(rng, palette_size=4)
        color_names = tuple(str(name) for name, _rgb in active_colors)
        if not color_names:
            raise ValueError("empty conveyor visual color palette")
        target_color_name = ""
        target_color_probabilities: Dict[str, float] = {}
        for index in range(int(target_count)):
            slot = _sample_slot(
                slots_by_belt,
                used_angles_by_belt,
                belt_key=target_belt_key,
                min_angle_gap_degrees=float(min_angle_gap_degrees),
            )
            color_name = str(rng.choice(color_names))
            object_id = f"obj_{len(object_specs):03d}"
            target_object_ids.append(object_id)
            object_specs.append(
                _make_object_spec(
                    rng=rng,
                    object_id=object_id,
                    shape_type=str(target_shape),
                    color_name=color_name,
                    slot=slot,
                    belt_key=target_belt_key,
                    matches_query=True,
                    count_role="target",
                    dimension_scale=float(dimension_scale),
                )
            )
        for index in range(max(0, int(scoped_belt_totals[str(target_belt_key)]) - int(target_count))):
            slot = _sample_slot(
                slots_by_belt,
                used_angles_by_belt,
                belt_key=target_belt_key,
                min_angle_gap_degrees=float(min_angle_gap_degrees),
            )
            object_specs.append(
                _make_object_spec(
                    rng=rng,
                    object_id=f"obj_{len(object_specs):03d}",
                    shape_type=str(distractor_shapes[int(rng.randrange(len(distractor_shapes)))]),
                    color_name=str(rng.choice(color_names)),
                    slot=slot,
                    belt_key=target_belt_key,
                    matches_query=False,
                    count_role="same_belt_distractor",
                    dimension_scale=float(dimension_scale),
                )
            )
        for belt_key in BELT_KEYS:
            if str(belt_key) == str(target_belt_key):
                continue
            for index in range(int(scoped_belt_totals[str(belt_key)])):
                slot = _sample_slot(
                    slots_by_belt,
                    used_angles_by_belt,
                    belt_key=str(belt_key),
                    min_angle_gap_degrees=float(min_angle_gap_degrees),
                )
                shape_type = str(target_shape) if rng.random() < 0.35 else str(distractor_shapes[int(rng.randrange(len(distractor_shapes)))])
                object_specs.append(
                    _make_object_spec(
                        rng=rng,
                        object_id=f"obj_{len(object_specs):03d}",
                        shape_type=str(shape_type),
                        color_name=str(rng.choice(color_names)),
                        slot=slot,
                        belt_key=str(belt_key),
                        matches_query=False,
                        count_role="other_belt_distractor",
                        dimension_scale=float(dimension_scale),
                    )
                )
        object_count = len(object_specs)
        object_count_probabilities = {str(object_count): 1.0}
        target_prompt_phrase = public_object_plural(str(target_shape))
    elif str(predicate_kind) == PREDICATE_COLOR:
        scoped_belt_totals = _sample_scoped_belt_totals(
            rng=rng,
            target_belt_key=str(target_belt_key),
            target_count=int(target_count),
        )
        target_color_name, target_color_probabilities = _resolve_target_color(params=params, rng=rng)
        target_shape, target_shape_probabilities = _resolve_target_shape(
            params=params,
            rng=rng,
            support=CONVEYOR_COLOR_READOUT_SHAPE_TYPES,
        )
        color_names = _sample_readout_palette(rng, target_color=str(target_color_name), size=4)
        wrong_colors = [str(color) for color in color_names if str(color) != str(target_color_name)]
        for _index in range(int(target_count)):
            slot = _sample_slot(
                slots_by_belt,
                used_angles_by_belt,
                belt_key=target_belt_key,
                min_angle_gap_degrees=float(min_angle_gap_degrees),
            )
            object_id = f"obj_{len(object_specs):03d}"
            target_object_ids.append(object_id)
            object_specs.append(
                _make_object_spec(
                    rng=rng,
                    object_id=object_id,
                    shape_type=str(target_shape),
                    color_name=str(target_color_name),
                    slot=slot,
                    belt_key=target_belt_key,
                    matches_query=True,
                    count_role="target",
                    dimension_scale=float(dimension_scale),
                )
            )
        for index in range(max(0, int(scoped_belt_totals[str(target_belt_key)]) - int(target_count))):
            slot = _sample_slot(
                slots_by_belt,
                used_angles_by_belt,
                belt_key=target_belt_key,
                min_angle_gap_degrees=float(min_angle_gap_degrees),
            )
            object_specs.append(
                _make_object_spec(
                    rng=rng,
                    object_id=f"obj_{len(object_specs):03d}",
                    shape_type=str(target_shape),
                    color_name=str(rng.choice(wrong_colors)),
                    slot=slot,
                    belt_key=target_belt_key,
                    matches_query=False,
                    count_role="same_belt_distractor",
                    dimension_scale=float(dimension_scale),
                )
            )
        for belt_key in BELT_KEYS:
            if str(belt_key) == str(target_belt_key):
                continue
            for index in range(int(scoped_belt_totals[str(belt_key)])):
                slot = _sample_slot(
                    slots_by_belt,
                    used_angles_by_belt,
                    belt_key=str(belt_key),
                    min_angle_gap_degrees=float(min_angle_gap_degrees),
                )
                color_name = str(target_color_name) if rng.random() < 0.35 else str(rng.choice(wrong_colors))
                object_specs.append(
                    _make_object_spec(
                        rng=rng,
                        object_id=f"obj_{len(object_specs):03d}",
                        shape_type=str(target_shape),
                        color_name=str(color_name),
                        slot=slot,
                        belt_key=str(belt_key),
                        matches_query=False,
                        count_role="other_belt_distractor",
                        dimension_scale=float(dimension_scale),
                    )
                )
        object_count = len(object_specs)
        object_count_probabilities = {str(object_count): 1.0}
        target_prompt_phrase = str(target_color_name)
    elif str(predicate_kind) == PREDICATE_COLOR_TYPE:
        scoped_belt_totals = _sample_scoped_belt_totals(
            rng=rng,
            target_belt_key=str(target_belt_key),
            target_count=int(target_count),
        )
        target_belt_max = int(_belt_max_object_count(str(target_belt_key)))
        scoped_belt_totals[str(target_belt_key)] = max(
            int(scoped_belt_totals[str(target_belt_key)]),
            min(int(target_belt_max), max(4, int(target_count) + 3)),
        )
        target_shape, target_shape_probabilities = _resolve_target_shape(
            params=params,
            rng=rng,
            support=CONVEYOR_COLOR_READOUT_SHAPE_TYPES,
        )
        distractor_shapes = list(
            compatible_distractor_pool(str(target_shape), support=CONVEYOR_COLOR_READOUT_SHAPE_TYPES)
        )
        target_color_name, target_color_probabilities = _resolve_target_color(params=params, rng=rng)
        color_names = _sample_readout_palette(rng, target_color=str(target_color_name), size=4)
        wrong_colors = [str(color) for color in color_names if str(color) != str(target_color_name)]
        for _index in range(int(target_count)):
            slot = _sample_slot(
                slots_by_belt,
                used_angles_by_belt,
                belt_key=target_belt_key,
                min_angle_gap_degrees=float(min_angle_gap_degrees),
            )
            object_id = f"obj_{len(object_specs):03d}"
            target_object_ids.append(object_id)
            object_specs.append(
                _make_object_spec(
                    rng=rng,
                    object_id=object_id,
                    shape_type=str(target_shape),
                    color_name=str(target_color_name),
                    slot=slot,
                    belt_key=target_belt_key,
                    matches_query=True,
                    count_role="target",
                    dimension_scale=float(dimension_scale),
                )
            )
        same_belt_distractor_count = max(0, int(scoped_belt_totals[str(target_belt_key)]) - int(target_count))
        for index in range(int(same_belt_distractor_count)):
            slot = _sample_slot(
                slots_by_belt,
                used_angles_by_belt,
                belt_key=target_belt_key,
                min_angle_gap_degrees=float(min_angle_gap_degrees),
            )
            pattern = int(index) % 3
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
            object_specs.append(
                _make_object_spec(
                    rng=rng,
                    object_id=f"obj_{len(object_specs):03d}",
                    shape_type=str(shape_type),
                    color_name=str(color_name),
                    slot=slot,
                    belt_key=target_belt_key,
                    matches_query=False,
                    count_role=str(count_role),
                    dimension_scale=float(dimension_scale),
                )
            )
        for belt_key in BELT_KEYS:
            if str(belt_key) == str(target_belt_key):
                continue
            for index in range(int(scoped_belt_totals[str(belt_key)])):
                slot = _sample_slot(
                    slots_by_belt,
                    used_angles_by_belt,
                    belt_key=str(belt_key),
                    min_angle_gap_degrees=float(min_angle_gap_degrees),
                )
                if int(index) == 0:
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
                object_specs.append(
                    _make_object_spec(
                        rng=rng,
                        object_id=f"obj_{len(object_specs):03d}",
                        shape_type=str(shape_type),
                        color_name=str(color_name),
                        slot=slot,
                        belt_key=str(belt_key),
                        matches_query=False,
                        count_role=str(count_role),
                        dimension_scale=float(dimension_scale),
                    )
                )
        object_count = len(object_specs)
        object_count_probabilities = {str(object_count): 1.0}
        target_prompt_phrase = f"{target_color_name} {public_object_plural(str(target_shape))}"
    else:
        raise ValueError(f"unsupported conveyor predicate kind: {predicate_kind}")

    camera, frame, camera_meta, frame_meta = _finalize_camera_and_projection(
        rng=rng,
        render_params=render_params,
        object_specs=object_specs,
    )
    finalized_specs = _screen_finalize_specs(object_specs=object_specs, camera=camera, frame=frame)
    shape_counts = Counter(str(spec["shape_type"]) for spec in finalized_specs)
    color_counts = Counter(str(spec["color_name"]) for spec in finalized_specs)
    belt_counts = Counter(str(spec["belt_key"]) for spec in finalized_specs)
    target_belt_object_ids = [
        str(spec["object_id"])
        for spec in finalized_specs
        if str(spec["belt_key"]) == str(target_belt_key)
    ]
    return {
        "scene_id": SCENE_ID,
        "scene_variant": str(axes.scene_variant),
        "layout_family": "elliptical_carousel",
        "predicate_kind": str(predicate_kind),
        "belt_records": [dict(record) for record in belt_records],
        "target_belt_key": str(target_belt_key),
        "target_belt_label": str(target_belt_label),
        "target_shape_type": str(target_shape),
        "target_object_name": public_object_name(str(target_shape)),
        "target_object_plural": public_object_plural(str(target_shape)),
        "target_color_name": str(target_color_name),
        "target_prompt_phrase": str(target_prompt_phrase),
        "answer_value": int(target_count),
        "target_count": int(target_count),
        "target_object_ids": list(target_object_ids),
        "target_belt_object_ids": list(target_belt_object_ids),
        "object_count": int(len(finalized_specs)),
        "object_specs": [dict(spec) for spec in finalized_specs],
        "shape_counts": {str(key): int(value) for key, value in sorted(shape_counts.items())},
        "color_counts": {str(key): int(value) for key, value in sorted(color_counts.items())},
        "belt_counts": {str(key): int(value) for key, value in sorted(belt_counts.items())},
        "target_shape_type_probabilities": dict(target_shape_probabilities),
        "target_color_name_probabilities": dict(target_color_probabilities),
        "target_count_probabilities": {} if str(predicate_kind) == PREDICATE_BELT_TOTAL else dict(target_count_probabilities),
        "object_count_probabilities": {} if str(predicate_kind) == PREDICATE_BELT_TOTAL else dict(object_count_probabilities),
        "target_belt_key_probabilities": dict(target_belt_probabilities),
        "target_belt_probabilities": dict(target_belt_probabilities),
        "slots_per_belt": int(slots_per_belt),
        "min_same_belt_angle_gap_degrees": round(float(min_angle_gap_degrees), 3),
        "semantic_color_palette": {str(key): list(value) for key, value in sorted(SEMANTIC_COLOR_RGB.items())},
        "camera": dict(camera_meta),
        "projection_frame": dict(frame_meta),
        "solver_trace": {
            "count_predicate": str(predicate_kind),
            "scope": {
                "belt_key": str(target_belt_key),
                "belt_label": str(target_belt_label),
            },
            "target_shape_type": str(target_shape),
            "target_color_name": str(target_color_name),
            "target_count": int(target_count),
            "target_object_ids": list(target_object_ids),
            "target_belt_object_ids": list(target_belt_object_ids),
            "answer_value": int(target_count),
            "unique_integer_answer": True,
        },
    }


def build_belt_count_arithmetic_dataset(
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
    """Build a two-belt carousel dataset for scoped count arithmetic."""

    rng = spawn_rng(int(instance_seed), f"{namespace}.dataset")
    first_count, second_count, answer_value, answer_probabilities, operand_probabilities = _resolve_arithmetic_operand_counts(
        params=params,
        gen_defaults=gen_defaults,
        rng=rng,
        instance_seed=int(instance_seed),
        namespace=str(namespace),
        operation=str(operation),
    )
    belt_keys = ("inner", "outer")
    operand_counts_by_belt = {
        "inner": int(first_count),
        "outer": int(second_count),
    }
    scoped_belt_totals = _sample_arithmetic_belt_totals(
        rng=rng,
        operand_counts_by_belt=operand_counts_by_belt,
    )
    slots_per_belt = max(24, _configured_int(params, gen_defaults, "slots_per_belt", 30))
    min_angle_gap_degrees = float(params.get("min_same_belt_angle_gap_degrees", group_default(gen_defaults, "min_same_belt_angle_gap_degrees", 20.0)))
    dimension_scale = float(params.get("object_dimension_scale", group_default(gen_defaults, "object_dimension_scale", 0.64)))
    slots_by_belt = {
        str(belt_key): _slot_positions_for_belt(rng=rng, belt_key=str(belt_key), slots_per_belt=int(slots_per_belt))
        for belt_key in BELT_KEYS
    }
    used_angles_by_belt: Dict[str, list[float]] = {str(belt_key): [] for belt_key in BELT_KEYS}
    belt_records = [
        {
            "belt_key": str(belt_key),
            "belt_label": str(BELT_LABELS[str(belt_key)]),
            "geometry": dict(BELT_GEOMETRY[str(belt_key)]),
            "slot_count": int(slots_per_belt),
        }
        for belt_key in BELT_KEYS
    ]
    object_specs: List[Dict[str, Any]] = []
    target_object_ids_by_belt: dict[str, list[str]] = {str(belt_key): [] for belt_key in BELT_KEYS}

    if str(predicate_kind) == PREDICATE_OBJECT_TYPE_ARITHMETIC:
        target_shape, target_shape_probabilities = _resolve_target_shape(
            params=params,
            rng=rng,
            support=CONVEYOR_OBJECT_SHAPE_TYPES,
        )
        distractor_shapes = list(compatible_distractor_pool(str(target_shape), support=CONVEYOR_OBJECT_SHAPE_TYPES))
        active_colors = sample_named_color_palette(rng, palette_size=4)
        color_names = tuple(str(name) for name, _rgb in active_colors)
        if not color_names:
            raise ValueError("empty carousel visual color palette")
        target_color_name = ""
        target_color_probabilities: Dict[str, float] = {}
        wrong_colors = list(color_names)
    else:
        raise ValueError(f"unsupported carousel arithmetic predicate: {predicate_kind}")

    for belt_key in belt_keys:
        target_count = int(operand_counts_by_belt[str(belt_key)])
        total_count = int(scoped_belt_totals[str(belt_key)])
        for _index in range(target_count):
            slot = _sample_slot(
                slots_by_belt,
                used_angles_by_belt,
                belt_key=str(belt_key),
                min_angle_gap_degrees=float(min_angle_gap_degrees),
            )
            object_id = f"obj_{len(object_specs):03d}"
            target_object_ids_by_belt[str(belt_key)].append(str(object_id))
            object_specs.append(
                _make_object_spec(
                    rng=rng,
                    object_id=str(object_id),
                    shape_type=str(target_shape),
                    color_name=str(rng.choice(color_names)),
                    slot=slot,
                    belt_key=str(belt_key),
                    matches_query=True,
                    count_role=f"{belt_key}_operand_target",
                    dimension_scale=float(dimension_scale),
                )
            )
        for index in range(max(0, total_count - target_count)):
            slot = _sample_slot(
                slots_by_belt,
                used_angles_by_belt,
                belt_key=str(belt_key),
                min_angle_gap_degrees=float(min_angle_gap_degrees),
            )
            shape_type = str(distractor_shapes[int(rng.randrange(len(distractor_shapes)))])
            color_name = str(rng.choice(color_names))
            object_specs.append(
                _make_object_spec(
                    rng=rng,
                    object_id=f"obj_{len(object_specs):03d}",
                    shape_type=str(shape_type),
                    color_name=str(color_name),
                    slot=slot,
                    belt_key=str(belt_key),
                    matches_query=False,
                    count_role=f"{belt_key}_operand_distractor",
                    dimension_scale=float(dimension_scale),
                )
            )

    camera, frame, camera_meta, frame_meta = _finalize_camera_and_projection(
        rng=rng,
        render_params=render_params,
        object_specs=object_specs,
    )
    finalized_specs = _screen_finalize_specs(object_specs=object_specs, camera=camera, frame=frame)
    shape_counts = Counter(str(spec["shape_type"]) for spec in finalized_specs)
    color_counts = Counter(str(spec["color_name"]) for spec in finalized_specs)
    belt_counts = Counter(str(spec["belt_key"]) for spec in finalized_specs)
    target_object_ids = [
        str(object_id)
        for belt_key in belt_keys
        for object_id in target_object_ids_by_belt[str(belt_key)]
    ]
    annotation_key_by_belt = {str(belt_key): f"{belt_key}_objects" for belt_key in belt_keys}
    target_object_ids_by_annotation_key = {
        str(annotation_key_by_belt[str(belt_key)]): [str(object_id) for object_id in target_object_ids_by_belt[str(belt_key)]]
        for belt_key in belt_keys
    }
    operand_counts_by_scope = {
        str(annotation_key_by_belt[str(belt_key)]): int(operand_counts_by_belt[str(belt_key)])
        for belt_key in belt_keys
    }
    return {
        "scene_id": SCENE_ID,
        "scene_variant": str(axes.scene_variant),
        "layout_family": "elliptical_carousel",
        "predicate_kind": str(predicate_kind),
        "arithmetic_operation": str(operation),
        "scope_keys": list(belt_keys),
        "scope_labels": {str(belt_key): str(BELT_LABELS[str(belt_key)]) for belt_key in belt_keys},
        "annotation_key_by_scope": dict(annotation_key_by_belt),
        "target_object_ids_by_annotation_key": dict(target_object_ids_by_annotation_key),
        "operand_counts_by_scope": dict(operand_counts_by_scope),
        "operand_count_probabilities": dict(operand_probabilities),
        "belt_records": [dict(record) for record in belt_records],
        "target_belt_key": "inner_outer",
        "target_belt_label": "INNER and OUTER",
        "target_shape_type": str(target_shape),
        "target_object_name": public_object_name(str(target_shape)),
        "target_object_plural": public_object_plural(str(target_shape)),
        "target_color_name": str(target_color_name),
        "target_color_label": semantic_color_label(str(target_color_name)) if str(target_color_name) else "",
        "answer_value": int(answer_value),
        "target_count": int(answer_value),
        "target_object_ids": list(target_object_ids),
        "target_belt_object_ids": list(target_object_ids),
        "object_count": int(len(finalized_specs)),
        "object_specs": [dict(spec) for spec in finalized_specs],
        "shape_counts": {str(key): int(value) for key, value in sorted(shape_counts.items())},
        "color_counts": {str(key): int(value) for key, value in sorted(color_counts.items())},
        "belt_counts": {str(key): int(value) for key, value in sorted(belt_counts.items())},
        "target_shape_type_probabilities": dict(target_shape_probabilities),
        "target_color_name_probabilities": dict(target_color_probabilities),
        "target_count_probabilities": dict(answer_probabilities),
        "object_count_probabilities": {str(len(finalized_specs)): 1.0},
        "target_belt_key_probabilities": {"inner_outer": 1.0},
        "target_belt_probabilities": {"inner_outer": 1.0},
        "slots_per_belt": int(slots_per_belt),
        "min_same_belt_angle_gap_degrees": round(float(min_angle_gap_degrees), 3),
        "semantic_color_palette": {str(key): list(value) for key, value in sorted(SEMANTIC_COLOR_RGB.items())},
        "camera": dict(camera_meta),
        "projection_frame": dict(frame_meta),
        "solver_trace": {
            "count_predicate": str(predicate_kind),
            "operation": str(operation),
            "scopes": {str(belt_key): str(BELT_LABELS[str(belt_key)]) for belt_key in belt_keys},
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
    """Build a two-belt carousel dataset for a counterfactual transfer count."""

    rng = spawn_rng(int(instance_seed), f"{namespace}.dataset")
    source_belt_key, destination_belt_key, source_belt_probabilities, destination_belt_probabilities = _resolve_transfer_belts(
        params=params,
        rng=rng,
    )
    destination_cap = int(_belt_max_object_count(str(destination_belt_key)))
    answer_value, moved_count, destination_existing_count, answer_probabilities, operand_probabilities = _resolve_transfer_answer_and_counts(
        params=params,
        gen_defaults=gen_defaults,
        rng=rng,
        instance_seed=int(instance_seed),
        namespace=str(namespace),
        destination_cap=int(destination_cap),
    )
    belt_counts: Dict[str, int] = {}
    for belt_key in BELT_KEYS:
        cap = int(_belt_max_object_count(str(belt_key)))
        if str(belt_key) == str(source_belt_key):
            min_count = max(int(moved_count), min(cap, int(moved_count) + 1))
            belt_counts[str(belt_key)] = int(rng.randrange(int(min_count), int(cap) + 1))
        elif str(belt_key) == str(destination_belt_key):
            belt_counts[str(belt_key)] = int(destination_existing_count)

    slots_per_belt = max(24, _configured_int(params, gen_defaults, "slots_per_belt", 30))
    min_angle_gap_degrees = float(params.get("min_same_belt_angle_gap_degrees", group_default(gen_defaults, "min_same_belt_angle_gap_degrees", 20.0)))
    dimension_scale = float(params.get("object_dimension_scale", group_default(gen_defaults, "object_dimension_scale", 0.64)))
    slots_by_belt = {
        str(belt_key): _slot_positions_for_belt(rng=rng, belt_key=str(belt_key), slots_per_belt=int(slots_per_belt))
        for belt_key in BELT_KEYS
    }
    used_angles_by_belt: Dict[str, list[float]] = {str(belt_key): [] for belt_key in BELT_KEYS}
    belt_records = [
        {
            "belt_key": str(belt_key),
            "belt_label": str(BELT_LABELS[str(belt_key)]),
            "geometry": dict(BELT_GEOMETRY[str(belt_key)]),
            "slot_count": int(slots_per_belt),
        }
        for belt_key in BELT_KEYS
    ]

    if str(predicate_kind) == PREDICATE_COLOR_TRANSFER:
        target_shape, target_shape_probabilities = _resolve_target_shape(
            params=params,
            rng=rng,
            support=CONVEYOR_COLOR_READOUT_SHAPE_TYPES,
        )
        target_color_name, target_color_probabilities = _resolve_target_color(params=params, rng=rng)
        color_names = _sample_readout_palette(rng, target_color=str(target_color_name), size=5)
        wrong_colors = [str(color) for color in color_names if str(color) != str(target_color_name)]
        distractor_shapes = list(compatible_distractor_pool(str(target_shape), support=CONVEYOR_COLOR_READOUT_SHAPE_TYPES))
    elif str(predicate_kind) == PREDICATE_OBJECT_TYPE_TRANSFER:
        target_shape, target_shape_probabilities = _resolve_target_shape(
            params=params,
            rng=rng,
            support=CONVEYOR_COLOR_READOUT_SHAPE_TYPES,
        )
        target_color_name = ""
        target_color_probabilities = {}
        active_colors = sample_named_color_palette(rng, palette_size=4)
        color_names = tuple(str(name) for name, _rgb in active_colors)
        wrong_colors = list(color_names)
        distractor_shapes = list(compatible_distractor_pool(str(target_shape), support=CONVEYOR_COLOR_READOUT_SHAPE_TYPES))
    else:
        raise ValueError(f"unsupported carousel transfer predicate: {predicate_kind}")
    if not wrong_colors or not distractor_shapes or not color_names:
        raise ValueError("carousel transfer task needs non-target distractors")

    object_specs: List[Dict[str, Any]] = []
    source_moved_ids: list[str] = []
    destination_existing_ids: list[str] = []
    for belt_key in BELT_KEYS:
        total_count = int(belt_counts[str(belt_key)])
        for index in range(total_count):
            slot = _sample_slot(
                slots_by_belt,
                used_angles_by_belt,
                belt_key=str(belt_key),
                min_angle_gap_degrees=float(min_angle_gap_degrees),
            )
            object_id = f"obj_{len(object_specs):03d}"
            if str(belt_key) == str(source_belt_key) and int(index) < int(moved_count):
                if str(predicate_kind) == PREDICATE_COLOR_TRANSFER:
                    shape_type = str(target_shape if index % 2 == 0 else rng.choice(distractor_shapes))
                    color_name = str(target_color_name)
                else:
                    shape_type = str(target_shape)
                    color_name = str(rng.choice(color_names))
                matches_query = True
                count_role = "source_moved_object"
                source_moved_ids.append(str(object_id))
            elif str(belt_key) == str(source_belt_key):
                if str(predicate_kind) == PREDICATE_COLOR_TRANSFER:
                    shape_type = str(rng.choice(distractor_shapes))
                    color_name = str(rng.choice(wrong_colors))
                else:
                    shape_type = str(rng.choice(distractor_shapes))
                    color_name = str(rng.choice(color_names))
                matches_query = False
                count_role = "source_non_moved_distractor"
            else:
                if str(predicate_kind) == PREDICATE_COLOR_TRANSFER:
                    shape_type = str(target_shape if index % 3 == 0 else rng.choice(distractor_shapes))
                    color_name = str(rng.choice(color_names))
                else:
                    shape_type = str(target_shape if index % 3 == 0 else rng.choice(distractor_shapes))
                    color_name = str(rng.choice(color_names))
                matches_query = True
                count_role = "destination_existing_object"
                destination_existing_ids.append(str(object_id))
            object_specs.append(
                _make_object_spec(
                    rng=rng,
                    object_id=str(object_id),
                    shape_type=str(shape_type),
                    color_name=str(color_name),
                    slot=slot,
                    belt_key=str(belt_key),
                    matches_query=bool(matches_query),
                    count_role=str(count_role),
                    dimension_scale=float(dimension_scale),
                )
            )

    camera, frame, camera_meta, frame_meta = _finalize_camera_and_projection(
        rng=rng,
        render_params=render_params,
        object_specs=object_specs,
    )
    finalized_specs = _screen_finalize_specs(object_specs=object_specs, camera=camera, frame=frame)
    shape_counts = Counter(str(spec["shape_type"]) for spec in finalized_specs)
    color_counts = Counter(str(spec["color_name"]) for spec in finalized_specs)
    belt_counts_final = Counter(str(spec["belt_key"]) for spec in finalized_specs)
    target_object_ids_by_annotation_key = {
        "source_moved_objects": list(source_moved_ids),
        "destination_existing_objects": list(destination_existing_ids),
    }
    target_object_ids = [*list(source_moved_ids), *list(destination_existing_ids)]
    return {
        "scene_id": SCENE_ID,
        "scene_variant": str(axes.scene_variant),
        "layout_family": "elliptical_carousel",
        "predicate_kind": str(predicate_kind),
        "transfer_operation": TRANSFER_OPERATION,
        "belt_records": [dict(record) for record in belt_records],
        "scope_keys": [str(source_belt_key), str(destination_belt_key)],
        "scope_labels": {
            str(source_belt_key): str(BELT_LABELS[str(source_belt_key)]),
            str(destination_belt_key): str(BELT_LABELS[str(destination_belt_key)]),
        },
        "source_belt_key": str(source_belt_key),
        "source_belt_label": str(BELT_LABELS[str(source_belt_key)]),
        "destination_belt_key": str(destination_belt_key),
        "destination_belt_label": str(BELT_LABELS[str(destination_belt_key)]),
        "annotation_key_by_scope": {
            "source_moved": "source_moved_objects",
            "destination_existing": "destination_existing_objects",
        },
        "target_object_ids_by_annotation_key": dict(target_object_ids_by_annotation_key),
        "operand_counts_by_scope": {
            "source_moved_objects": int(moved_count),
            "destination_existing_objects": int(destination_existing_count),
        },
        "operand_count_probabilities": dict(operand_probabilities),
        "target_belt_key": str(destination_belt_key),
        "target_belt_label": str(BELT_LABELS[str(destination_belt_key)]),
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
        "target_belt_object_ids": [
            str(spec["object_id"])
            for spec in finalized_specs
            if str(spec["belt_key"]) == str(destination_belt_key)
        ],
        "object_count": int(len(finalized_specs)),
        "object_specs": [dict(spec) for spec in finalized_specs],
        "shape_counts": {str(key): int(value) for key, value in sorted(shape_counts.items())},
        "color_counts": {str(key): int(value) for key, value in sorted(color_counts.items())},
        "belt_counts": {str(key): int(value) for key, value in sorted(belt_counts_final.items())},
        "target_shape_type_probabilities": dict(target_shape_probabilities),
        "target_color_name_probabilities": dict(target_color_probabilities),
        "target_count_probabilities": dict(answer_probabilities),
        "object_count_probabilities": {str(len(finalized_specs)): 1.0},
        "target_belt_key_probabilities": dict(destination_belt_probabilities),
        "target_belt_probabilities": dict(destination_belt_probabilities),
        "source_belt_key_probabilities": dict(source_belt_probabilities),
        "slots_per_belt": int(slots_per_belt),
        "min_same_belt_angle_gap_degrees": round(float(min_angle_gap_degrees), 3),
        "semantic_color_palette": {str(key): list(value) for key, value in sorted(SEMANTIC_COLOR_RGB.items())},
        "camera": dict(camera_meta),
        "projection_frame": dict(frame_meta),
        "solver_trace": {
            "count_predicate": str(predicate_kind),
            "operation": TRANSFER_OPERATION,
            "source": {
                "belt_key": str(source_belt_key),
                "belt_label": str(BELT_LABELS[str(source_belt_key)]),
            },
            "destination": {
                "belt_key": str(destination_belt_key),
                "belt_label": str(BELT_LABELS[str(destination_belt_key)]),
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
    """Build an elliptical-carousel dataset for ordered adjacent-pair counting."""

    rng = spawn_rng(int(instance_seed), f"{namespace}.dataset")
    target_belt_key, target_belt_probabilities = _resolve_target_belt(params=params, rng=rng)
    target_belt_label = str(BELT_LABELS[str(target_belt_key)])
    target_count, target_count_probabilities = _resolve_ordered_pair_count(
        params=params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.{target_belt_key}.target_count",
    )
    target_total_min = max(6, 2 * int(target_count))
    target_total_max = int(_belt_max_object_count(str(target_belt_key)))
    belt_counts: Dict[str, int] = {}
    for belt_key in BELT_KEYS:
        if str(belt_key) == str(target_belt_key):
            belt_counts[str(belt_key)] = int(rng.randrange(int(target_total_min), int(target_total_max) + 1))
        else:
            belt_counts[str(belt_key)] = int(rng.randrange(3, int(_belt_max_object_count(str(belt_key))) + 1))

    dimension_scale = float(params.get("object_dimension_scale", group_default(gen_defaults, "object_dimension_scale", 0.60)))
    if str(predicate_kind) == PREDICATE_ORDERED_COLOR_PAIR:
        target_shape, target_shape_probabilities = _resolve_target_shape(
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
        first_shape, target_shape_probabilities = _resolve_target_shape(
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
        active_colors = sample_named_color_palette(rng, palette_size=4)
        palette = [str(name) for name, _rgb in active_colors]
    else:
        raise ValueError(f"unsupported carousel ordered-pair predicate: {predicate_kind}")
    if not filler_symbols:
        raise ValueError("ordered pair task needs at least one filler symbol")
    if not palette:
        raise ValueError("ordered pair task needs a visual color palette")

    first_symbol = str(target_color_name if predicate_kind == PREDICATE_ORDERED_COLOR_PAIR else target_shape_pair[0])
    second_symbol = str(second_target_color_name if predicate_kind == PREDICATE_ORDERED_COLOR_PAIR else target_shape_pair[1])
    target_sequence = _ordered_pair_symbol_sequence(
        rng=rng,
        first_symbol=str(first_symbol),
        second_symbol=str(second_symbol),
        filler_symbols=filler_symbols,
        pair_count=int(target_count),
        total_count=int(belt_counts[str(target_belt_key)]),
        circular=True,
    )
    belt_records = [
        {
            "belt_key": str(belt_key),
            "belt_label": str(BELT_LABELS[str(belt_key)]),
            "geometry": dict(BELT_GEOMETRY[str(belt_key)]),
            "slot_count": int(belt_counts[str(belt_key)]),
        }
        for belt_key in BELT_KEYS
    ]
    object_specs: List[Dict[str, Any]] = []
    object_sequences_by_belt: Dict[str, list[str]] = {}
    target_pair_object_id_pairs: list[list[str]] = []
    target_object_ids: list[str] = []

    for belt_key in BELT_KEYS:
        slots = _ordered_slot_positions_for_belt(
            rng=rng,
            belt_key=str(belt_key),
            count=int(belt_counts[str(belt_key)]),
        )
        if str(belt_key) == str(target_belt_key):
            sequence = tuple(target_sequence)
        elif str(predicate_kind) == PREDICATE_ORDERED_COLOR_PAIR:
            sequence = tuple(str(palette[int(rng.randrange(len(palette)))]) for _ in slots)
        else:
            distractor_shapes = [*list(target_shape_pair), *list(filler_symbols)]
            sequence = tuple(str(distractor_shapes[int(rng.randrange(len(distractor_shapes)))]) for _ in slots)

        belt_object_ids: list[str] = []
        for index, slot in enumerate(slots):
            object_id = f"obj_{len(object_specs):03d}"
            belt_object_ids.append(str(object_id))
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
                    slot=slot,
                    belt_key=str(belt_key),
                    matches_query=False,
                    count_role="ordered_pair_scope" if str(belt_key) == str(target_belt_key) else "belt_distractor",
                    dimension_scale=float(dimension_scale),
                )
            )
        object_sequences_by_belt[str(belt_key)] = list(belt_object_ids)
        if str(belt_key) == str(target_belt_key):
            for index, symbol in enumerate(sequence):
                next_index = index + 1
                if next_index >= len(sequence):
                    next_index = 0
                if str(symbol) == str(first_symbol) and str(sequence[next_index]) == str(second_symbol):
                    pair = [str(belt_object_ids[index]), str(belt_object_ids[next_index])]
                    target_pair_object_id_pairs.append(pair)
                    target_object_ids.extend(pair)

    target_object_ids = list(dict.fromkeys(target_object_ids))
    if len(target_pair_object_id_pairs) != int(target_count):
        raise ValueError("carousel ordered pair dataset target pair count mismatch")
    for object_id in set(target_object_ids):
        for spec in object_specs:
            if str(spec["object_id"]) == str(object_id):
                spec["matches_query"] = True
                spec["object_role"] = "target"

    camera, frame, camera_meta, frame_meta = _finalize_camera_and_projection(
        rng=rng,
        render_params=render_params,
        object_specs=object_specs,
    )
    finalized_specs = _screen_finalize_specs(object_specs=object_specs, camera=camera, frame=frame)
    shape_counts = Counter(str(spec["shape_type"]) for spec in finalized_specs)
    color_counts = Counter(str(spec["color_name"]) for spec in finalized_specs)
    belt_counts_final = Counter(str(spec["belt_key"]) for spec in finalized_specs)
    target_belt_object_ids = [
        str(spec["object_id"])
        for spec in finalized_specs
        if str(spec["belt_key"]) == str(target_belt_key)
    ]
    return {
        "scene_id": SCENE_ID,
        "scene_variant": str(axes.scene_variant),
        "layout_family": "elliptical_carousel",
        "predicate_kind": str(predicate_kind),
        "belt_records": [dict(record) for record in belt_records],
        "target_belt_key": str(target_belt_key),
        "target_belt_label": str(target_belt_label),
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
        "target_belt_object_ids": list(target_belt_object_ids),
        "object_sequences_by_belt": {str(key): list(value) for key, value in object_sequences_by_belt.items()},
        "object_count": int(len(finalized_specs)),
        "object_specs": [dict(spec) for spec in finalized_specs],
        "shape_counts": {str(key): int(value) for key, value in sorted(shape_counts.items())},
        "color_counts": {str(key): int(value) for key, value in sorted(color_counts.items())},
        "belt_counts": {str(key): int(value) for key, value in sorted(belt_counts_final.items())},
        "target_shape_type_probabilities": dict(target_shape_probabilities),
        "target_color_name_probabilities": dict(target_color_probabilities),
        "target_count_probabilities": dict(target_count_probabilities),
        "object_count_probabilities": {str(len(finalized_specs)): 1.0},
        "target_belt_key_probabilities": dict(target_belt_probabilities),
        "target_belt_probabilities": dict(target_belt_probabilities),
        "slots_per_belt": int(max(belt_counts.values())),
        "min_same_belt_angle_gap_degrees": 0.0,
        "semantic_color_palette": {str(key): list(value) for key, value in sorted(SEMANTIC_COLOR_RGB.items())},
        "camera": dict(camera_meta),
        "projection_frame": dict(frame_meta),
        "solver_trace": {
            "count_predicate": str(predicate_kind),
            "scope": {
                "belt_key": str(target_belt_key),
                "belt_label": str(target_belt_label),
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
    """Build an elliptical-carousel dataset for counting objects between marked anchors."""

    rng = spawn_rng(int(instance_seed), f"{namespace}.dataset")
    target_belt_key, target_belt_probabilities = _resolve_target_belt(params=params, rng=rng)
    target_belt_label = str(BELT_LABELS[str(target_belt_key)])
    target_count, target_count_probabilities = _resolve_between_items_count(
        params=params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.{target_belt_key}.target_count",
    )
    target_total_min = max(6, int(target_count) + 2)
    target_total_max = int(_belt_max_object_count(str(target_belt_key)))
    belt_counts: Dict[str, int] = {}
    for belt_key in BELT_KEYS:
        if str(belt_key) == str(target_belt_key):
            belt_counts[str(belt_key)] = int(rng.randrange(int(target_total_min), int(target_total_max) + 1))
        else:
            belt_counts[str(belt_key)] = int(rng.randrange(3, int(_belt_max_object_count(str(belt_key))) + 1))

    target_total = int(belt_counts[str(target_belt_key)])
    start_index = int(rng.randrange(int(target_total)))
    end_index = int((start_index + int(target_count) + 1) % int(target_total))
    between_indices = {
        int((start_index + offset) % int(target_total))
        for offset in range(1, int(target_count) + 1)
    }
    dimension_scale = float(params.get("object_dimension_scale", group_default(gen_defaults, "object_dimension_scale", 0.60)))

    if str(predicate_kind) == PREDICATE_BETWEEN_OBJECT_ANCHORS:
        first_shape, target_shape_probabilities = _resolve_target_shape(
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
        active_colors = sample_named_color_palette(rng, palette_size=4)
        color_palette = [str(name) for name, _rgb in active_colors]
        shape_palette = [str(first_shape), str(second_shape), *list(filler_symbols)]
    else:
        raise ValueError(f"unsupported carousel between-anchor predicate: {predicate_kind}")
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

    belt_records = [
        {
            "belt_key": str(belt_key),
            "belt_label": str(BELT_LABELS[str(belt_key)]),
            "geometry": dict(BELT_GEOMETRY[str(belt_key)]),
            "slot_count": int(belt_counts[str(belt_key)]),
        }
        for belt_key in BELT_KEYS
    ]
    object_specs: List[Dict[str, Any]] = []
    object_sequences_by_belt: Dict[str, list[str]] = {}
    target_object_ids: list[str] = []
    start_anchor_object_id = ""
    end_anchor_object_id = ""

    for belt_key in BELT_KEYS:
        slots = _ordered_slot_positions_for_belt(
            rng=rng,
            belt_key=str(belt_key),
            count=int(belt_counts[str(belt_key)]),
        )
        if str(belt_key) == str(target_belt_key):
            sequence = tuple(target_sequence)
        else:
            sequence = tuple(str(shape_palette[int(rng.randrange(len(shape_palette)))]) for _ in slots)

        belt_object_ids: list[str] = []
        for index, slot in enumerate(slots):
            object_id = f"obj_{len(object_specs):03d}"
            belt_object_ids.append(str(object_id))
            shape_type = str(sequence[index])
            color_name = str(rng.choice(color_palette))

            is_start_anchor = str(belt_key) == str(target_belt_key) and int(index) == int(start_index)
            is_end_anchor = str(belt_key) == str(target_belt_key) and int(index) == int(end_index)
            matches_query = str(belt_key) == str(target_belt_key) and int(index) in between_indices
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
                count_role = "same_belt_distractor" if str(belt_key) == str(target_belt_key) else "belt_distractor"
            object_specs.append(
                _make_object_spec(
                    rng=rng,
                    object_id=str(object_id),
                    shape_type=str(shape_type),
                    color_name=str(color_name),
                    slot=slot,
                    belt_key=str(belt_key),
                    matches_query=bool(matches_query),
                    count_role=str(count_role),
                    dimension_scale=float(dimension_scale),
                )
            )
        object_sequences_by_belt[str(belt_key)] = list(belt_object_ids)

    if not start_anchor_object_id or not end_anchor_object_id:
        raise ValueError("between-anchor task failed to assign marked anchors")
    if len(target_object_ids) != int(target_count):
        raise ValueError("between-anchor dataset target count mismatch")

    camera, frame, camera_meta, frame_meta = _finalize_camera_and_projection(
        rng=rng,
        render_params=render_params,
        object_specs=object_specs,
    )
    finalized_specs = _screen_finalize_specs(object_specs=object_specs, camera=camera, frame=frame)
    shape_counts = Counter(str(spec["shape_type"]) for spec in finalized_specs)
    color_counts = Counter(str(spec["color_name"]) for spec in finalized_specs)
    belt_counts_final = Counter(str(spec["belt_key"]) for spec in finalized_specs)
    target_belt_object_ids = [
        str(spec["object_id"])
        for spec in finalized_specs
        if str(spec["belt_key"]) == str(target_belt_key)
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
        "layout_family": "elliptical_carousel",
        "predicate_kind": str(predicate_kind),
        "belt_records": [dict(record) for record in belt_records],
        "target_belt_key": str(target_belt_key),
        "target_belt_label": str(target_belt_label),
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
        "target_belt_object_ids": list(target_belt_object_ids),
        "object_sequences_by_belt": {str(key): list(value) for key, value in object_sequences_by_belt.items()},
        "object_count": int(len(finalized_specs)),
        "object_specs": [dict(spec) for spec in finalized_specs],
        "shape_counts": {str(key): int(value) for key, value in sorted(shape_counts.items())},
        "color_counts": {str(key): int(value) for key, value in sorted(color_counts.items())},
        "belt_counts": {str(key): int(value) for key, value in sorted(belt_counts_final.items())},
        "target_shape_type_probabilities": dict(target_shape_probabilities),
        "target_color_name_probabilities": dict(target_color_probabilities),
        "target_count_probabilities": dict(target_count_probabilities),
        "object_count_probabilities": {str(len(finalized_specs)): 1.0},
        "target_belt_key_probabilities": dict(target_belt_probabilities),
        "target_belt_probabilities": dict(target_belt_probabilities),
        "slots_per_belt": int(max(belt_counts.values())),
        "min_same_belt_angle_gap_degrees": 0.0,
        "semantic_color_palette": {str(key): list(value) for key, value in sorted(SEMANTIC_COLOR_RGB.items())},
        "camera": dict(camera_meta),
        "projection_frame": dict(frame_meta),
        "solver_trace": {
            "count_predicate": str(predicate_kind),
            "scope": {
                "belt_key": str(target_belt_key),
                "belt_label": str(target_belt_label),
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
    "build_belt_count_arithmetic_dataset",
    "build_belt_count_dataset",
    "build_ordered_pair_count_dataset",
    "build_transfer_count_dataset",
    "resolve_conveyor_axes",
]
