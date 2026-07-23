"""Sampling helpers for wire-magnetism diagrams."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.variant_sampling import apply_balanced_variant_sampling, resolve_variant

from .state import OPTION_LABELS, SUPPORTED_CURRENT_DIRECTIONS, SUPPORTED_POINT_POSITIONS, WireScenario


CURRENT_Z_SIGN: Dict[str, int] = {
    "out_of_page": 1,
    "into_page": -1,
}
POINT_OFFSETS: Dict[str, Tuple[int, int]] = {
    "east": (1, 0),
    "north": (0, 1),
    "west": (-1, 0),
    "south": (0, -1),
}
FIELD_DIRECTIONS: Tuple[str, ...] = ("north", "south", "east", "west")


def probability_map(values: Sequence[str], selected: str | None = None) -> Dict[str, float]:
    """Return a string-keyed probability map for a finite support."""

    supported = tuple(str(value) for value in values if str(value))
    if selected is not None:
        return {value: (1.0 if value == str(selected) else 0.0) for value in supported}
    probability = 1.0 / float(len(supported)) if supported else 0.0
    return {value: float(probability) for value in supported}


def _resolve_weighted_label(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    supported_values: Tuple[str, ...],
    explicit_key: str,
    weights_key: str,
    balance_flag_key: str,
    namespace: str,
) -> Tuple[str, Dict[str, float]]:
    rng = spawn_rng(int(instance_seed), f"{namespace}.choice")
    selected, probabilities = resolve_variant(
        rng,
        params=params,
        gen_defaults=defaults,
        supported_variants=supported_values,
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
    )
    selected = apply_balanced_variant_sampling(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=defaults,
        selected_variant=str(selected),
        variant_probabilities=probabilities,
        supported_variants=supported_values,
        balance_flag_key=str(balance_flag_key),
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
        sampling_namespace=f"{namespace}.choice",
    )
    return str(selected), {str(key): float(value) for key, value in sorted(probabilities.items())}


def _resolve_current_direction(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    namespace: str,
) -> Tuple[str, int, Dict[str, float]]:
    selected, probabilities = _resolve_weighted_label(
        instance_seed=int(instance_seed),
        params=params,
        defaults=defaults,
        supported_values=SUPPORTED_CURRENT_DIRECTIONS,
        explicit_key="current_direction",
        weights_key="current_direction_weights",
        balance_flag_key="balanced_current_direction_sampling",
        namespace=f"{namespace}.current_direction",
    )
    return str(selected), int(CURRENT_Z_SIGN[str(selected)]), dict(probabilities)


def _resolve_point_position(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    namespace: str,
) -> Tuple[str, Tuple[int, int], Dict[str, float]]:
    selected, probabilities = _resolve_weighted_label(
        instance_seed=int(instance_seed),
        params=params,
        defaults=defaults,
        supported_values=SUPPORTED_POINT_POSITIONS,
        explicit_key="point_position",
        weights_key="point_position_weights",
        balance_flag_key="balanced_point_position_sampling",
        namespace=f"{namespace}.point_position",
    )
    return str(selected), tuple(int(value) for value in POINT_OFFSETS[str(selected)]), dict(probabilities)


def _resolve_target_label(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    namespace: str,
) -> Tuple[str, Dict[str, float]]:
    explicit = params.get("target_label", params.get("correct_option_letter", params.get("target_answer")))
    if explicit is not None:
        selected = str(explicit).strip().upper()
        if selected not in OPTION_LABELS:
            raise ValueError(f"unsupported wire-magnetism target option: {selected}")
        return selected, probability_map(OPTION_LABELS, selected=selected)

    if bool(params.get("balanced_target_answer_sampling", group_default(defaults, "balanced_target_answer_sampling", True))):
        rng = spawn_rng(int(instance_seed), f"{namespace}.target_label")
        selected = str(rng.choice(OPTION_LABELS))
    else:
        rng = spawn_rng(int(instance_seed), f"{namespace}.target_label")
        selected = str(rng.choice(OPTION_LABELS))
    return selected, probability_map(OPTION_LABELS)


def _field_direction_for(current_z_sign: int, point_offset: Tuple[int, int]) -> str:
    # With +z current out of the page, B is tangent to the counterclockwise circle.
    x, y = int(point_offset[0]), int(point_offset[1])
    bx = int(current_z_sign) * (-y)
    by = int(current_z_sign) * x
    direction_by_vector = {
        (0, 1): "north",
        (0, -1): "south",
        (1, 0): "east",
        (-1, 0): "west",
    }
    direction = direction_by_vector.get((bx, by))
    if direction is None:
        raise ValueError(f"unsupported field vector from current={current_z_sign}, point_offset={point_offset}")
    return str(direction)


def _build_option_map(
    *,
    instance_seed: int,
    field_direction: str,
    correct_label: str,
    namespace: str,
) -> Dict[str, str]:
    remaining = [direction for direction in FIELD_DIRECTIONS if str(direction) != str(field_direction)]
    rng = spawn_rng(int(instance_seed), f"{namespace}.option_map")
    rng.shuffle(remaining)
    option_map: Dict[str, str] = {}
    distractor_cursor = 0
    for label in OPTION_LABELS:
        if str(label) == str(correct_label):
            option_map[str(label)] = str(field_direction)
        else:
            option_map[str(label)] = str(remaining[distractor_cursor])
            distractor_cursor += 1
    return option_map


def build_wire_scenario(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    namespace: str,
) -> WireScenario:
    """Resolve one physical wire scenario and unique option-letter answer."""

    current_name, current_z_sign, current_probs = _resolve_current_direction(
        instance_seed=int(instance_seed),
        params=params,
        defaults=defaults,
        namespace=str(namespace),
    )
    point_position, point_offset, point_probs = _resolve_point_position(
        instance_seed=int(instance_seed),
        params=params,
        defaults=defaults,
        namespace=str(namespace),
    )
    field_direction = _field_direction_for(int(current_z_sign), point_offset)
    correct_label, target_probs = _resolve_target_label(
        instance_seed=int(instance_seed),
        params=params,
        defaults=defaults,
        namespace=str(namespace),
    )
    option_map = _build_option_map(
        instance_seed=int(instance_seed),
        field_direction=str(field_direction),
        correct_label=str(correct_label),
        namespace=str(namespace),
    )
    return WireScenario(
        current_direction=str(current_name),
        current_z_sign=int(current_z_sign),
        point_position=str(point_position),
        point_offset_phys=tuple(int(value) for value in point_offset),
        field_direction=str(field_direction),
        option_map=dict(option_map),
        correct_label=str(correct_label),
        current_direction_probabilities=dict(current_probs),
        point_position_probabilities=dict(point_probs),
        target_answer_probabilities=dict(target_probs),
    )


__all__ = [
    "build_wire_scenario",
    "probability_map",
]
