"""Identity-free sampling helpers for measuring-tool scenes."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from trace_tasks.core.sampling import uniform_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.fixed_query import geometry_selected_probability_map

from .state import AngleMeasurementPlan, LengthMeasurementPlan


def select_supported_integer(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    explicit_key: str,
    support: Sequence[int],
    namespace: str,
) -> tuple[int, dict[str, float]]:
    """Select one integer from explicit params or balanced deterministic support."""

    supported_values = tuple(int(value) for value in support)
    if not supported_values:
        raise ValueError("integer support is empty")
    explicit = params.get(str(explicit_key))
    if explicit is not None:
        selected = int(explicit)
        if selected not in set(supported_values):
            raise ValueError(f"{explicit_key}={selected} is outside support")
        return selected, geometry_selected_probability_map(
            supported_values,
            selected,
            key_fn=lambda value: str(int(value)),
            is_selected=lambda value, target: int(value) == int(target),
        )
    rng = spawn_rng(int(instance_seed), str(namespace))
    selected = int(uniform_choice(rng, supported_values))
    probability = 1.0 / float(len(supported_values))
    return selected, {str(value): float(probability) for value in supported_values}


def select_index(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
    count: int,
) -> int:
    """Select one deterministic index without exposing task identity to shared code."""

    if int(count) <= 0:
        raise ValueError("count must be positive")
    rng = spawn_rng(int(instance_seed), str(namespace))
    return int(uniform_choice(rng, tuple(range(int(count)))))


def build_angle_measurement_plan(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    gen_defaults: Mapping[str, Any],
    measurement_kind: str,
    shape_kind: str,
    answer_namespace: str,
) -> AngleMeasurementPlan:
    """Resolve the integer angle support for one protractor objective."""

    angle_min = int(params.get("angle_min", group_default(gen_defaults, "angle_min", 30)))
    angle_max = int(params.get("angle_max", group_default(gen_defaults, "angle_max", 150)))
    angle_step = int(params.get("angle_step", group_default(gen_defaults, "angle_step", 5)))
    angle, probabilities = select_supported_integer(
        params=params,
        instance_seed=int(instance_seed),
        explicit_key="target_angle",
        support=tuple(range(angle_min, angle_max + 1, angle_step)),
        namespace=str(answer_namespace),
    )
    return AngleMeasurementPlan(
        measurement_kind=str(measurement_kind),
        shape_kind=str(shape_kind),
        target_angle_degrees=int(angle),
        answer_probabilities=dict(probabilities),
    )


def build_ruler_length_plan(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    gen_defaults: Mapping[str, Any],
    measurement_kind: str,
    shape_kind: str,
    answer_namespace: str,
    offset_namespace: str,
    shape_options: Sequence[str] = tuple(),
    shape_namespace: str | None = None,
) -> LengthMeasurementPlan:
    """Resolve the integer length, ruler offset, and optional polygon shape."""

    length_min = int(params.get("length_min", group_default(gen_defaults, "length_min", 2)))
    length_max = int(params.get("length_max", group_default(gen_defaults, "length_max", 8)))
    target, probabilities = select_supported_integer(
        params=params,
        instance_seed=int(instance_seed),
        explicit_key="target_length",
        support=tuple(range(length_min, length_max + 1)),
        namespace=str(answer_namespace),
    )
    ruler_max = int(params.get("ruler_max_cm", group_default(gen_defaults, "ruler_max_cm", 10)))
    if ruler_max <= int(target):
        raise ValueError("ruler_max_cm must exceed target_length")
    max_start = int(ruler_max) - int(target)
    explicit_start = params.get("ruler_start_cm")
    start_cm = (
        select_index(params=params, instance_seed=int(instance_seed), namespace=str(offset_namespace), count=max_start + 1)
        if explicit_start is None
        else int(explicit_start)
    )
    if start_cm < 0 or start_cm + int(target) > int(ruler_max):
        raise ValueError("ruler_start_cm plus target_length exceeds ruler range")
    resolved_shape = str(shape_kind)
    if shape_options:
        option_index = select_index(
            params=params,
            instance_seed=int(instance_seed),
            namespace=str(shape_namespace or offset_namespace),
            count=len(tuple(shape_options)),
        )
        resolved_shape = str(tuple(shape_options)[int(option_index)])
    return LengthMeasurementPlan(
        measurement_kind=str(measurement_kind),
        shape_kind=resolved_shape,
        target_length_cm=int(target),
        ruler_start_cm=int(start_cm),
        ruler_max_cm=int(ruler_max),
        answer_probabilities=dict(probabilities),
    )


__all__ = [
    "build_angle_measurement_plan",
    "build_ruler_length_plan",
    "select_index",
    "select_supported_integer",
]
