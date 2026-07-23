"""Identity-free function-family sampling primitives."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from trace_tasks.core.sampling import uniform_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.shared.deterministic_sampling import resolve_selection_index, uniform_probability_map

from .defaults import DEFAULTS, GEN_DEFAULTS, float_tuple_default, int_tuple_default
from .state import GraphPoint, GraphPolylinePoint, SampledFunctionGraph

FAMILY_SINUSOID = "sinusoid"
FAMILY_PIECEWISE_LINEAR = "piecewise_linear"


def average_rate_support() -> Tuple[float, ...]:
    """Return configured one-decimal average-rate answers."""

    values = float_tuple_default("average_rate_support", DEFAULTS.average_rate_support)
    if any(abs(float(value)) <= 1e-9 for value in values):
        raise ValueError("average_rate_support cannot include zero")
    return tuple(float(value) for value in values)


def turning_count_support_by_family() -> Dict[str, Tuple[int, ...]]:
    """Return turning-point count support for each compatible family."""

    return {
        FAMILY_SINUSOID: int_tuple_default("sinusoid_turning_support", DEFAULTS.sinusoid_turning_support),
        FAMILY_PIECEWISE_LINEAR: int_tuple_default("piecewise_turning_support", DEFAULTS.piecewise_turning_support),
    }


def local_extremum_support_by_family() -> Dict[str, Tuple[int, ...]]:
    """Return one-kind local-extremum count support for each compatible family."""

    return {
        FAMILY_SINUSOID: int_tuple_default("sinusoid_local_extremum_support", DEFAULTS.sinusoid_local_support),
        FAMILY_PIECEWISE_LINEAR: int_tuple_default(
            "piecewise_local_extremum_support",
            DEFAULTS.piecewise_local_support,
        ),
    }


def support_union(support_by_family: Mapping[str, Sequence[int]]) -> Tuple[int, ...]:
    """Return the sorted union of integer answer support across families."""

    values = {int(value) for support in support_by_family.values() for value in support}
    if not values:
        raise ValueError("count support cannot be empty")
    return tuple(sorted(values))


def resolve_numeric_target(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    support: Sequence[int],
    namespace: str,
) -> Tuple[int, Dict[str, float]]:
    """Resolve one count target uniformly from a finite answer support."""

    ordered_support = tuple(int(value) for value in support)
    explicit = params.get("target_count")
    if explicit is not None:
        selected = int(explicit)
        if selected not in set(ordered_support):
            raise ValueError(f"unsupported target_count: {selected}")
        return selected, uniform_probability_map(ordered_support, selected=selected)

    rng = spawn_rng(int(instance_seed), str(namespace))
    selected = int(uniform_choice(rng, ordered_support))
    probability = 1.0 / float(len(ordered_support))
    return selected, {str(value): float(probability) for value in ordered_support}


def resolve_rate_target(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
) -> Tuple[float, Dict[str, float]]:
    """Resolve one average-rate target uniformly from configured support."""

    support = average_rate_support()
    explicit = params.get("target_rate", params.get("average_rate"))
    if explicit is not None:
        selected = round(float(explicit), 1)
        if selected not in set(support):
            raise ValueError(f"unsupported target_rate: {selected}")
        return float(selected), {f"{float(value):.1f}": (1.0 if float(value) == selected else 0.0) for value in support}

    rng = spawn_rng(int(instance_seed), "function_graph.rate_target")
    selected = float(uniform_choice(rng, support))
    probability = 1.0 / float(len(support))
    return selected, {f"{float(value):.1f}": float(probability) for value in support}


def resolve_family_for_target(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    support_by_family: Mapping[str, Sequence[int]],
    target_count: int,
    namespace: str,
) -> Tuple[str, Dict[str, float]]:
    """Resolve one compatible function family for a requested target count."""

    all_families = tuple(str(family) for family in support_by_family)
    allowed = tuple(
        str(family)
        for family, support in support_by_family.items()
        if int(target_count) in {int(value) for value in support}
    )
    if not allowed:
        raise ValueError(f"no function family supports target_count={target_count}")
    explicit = params.get("scene_variant")
    if explicit is not None:
        selected = str(explicit)
        if selected not in set(all_families):
            raise ValueError(f"unsupported scene_variant: {selected}")
        if selected not in set(allowed):
            raise ValueError(f"scene_variant {selected} does not support target_count={target_count}")
        return selected, {family: (1.0 if family == selected else 0.0) for family in all_families}

    rng = spawn_rng(int(instance_seed), str(namespace))
    selected = str(uniform_choice(rng, allowed))
    probability = 1.0 / float(len(allowed))
    return selected, {family: (float(probability) if family in set(allowed) else 0.0) for family in all_families}


def sample_turning_scene(rng, *, family: str, target_count: int) -> SampledFunctionGraph:
    """Sample a graph with the requested number of visible turning points."""

    if str(family) == FAMILY_SINUSOID:
        return _build_sinusoid_turning_scene(rng, target_count=int(target_count))
    if str(family) == FAMILY_PIECEWISE_LINEAR:
        vertices, annotation_points = _piecewise_polyline_for_turning_points(rng, target_count=int(target_count))
        return _piecewise_scene(
            vertices=vertices,
            annotation_points=annotation_points,
            purpose="turning_points",
        )
    raise ValueError(f"unsupported function family: {family}")


def sample_local_extremum_scene(
    rng,
    *,
    family: str,
    target_count: int,
    extremum_sign: int,
) -> SampledFunctionGraph:
    """Sample a graph with the requested number of visible minima or maxima."""

    sign = -1 if int(extremum_sign) < 0 else 1
    if str(family) == FAMILY_SINUSOID:
        return _build_sinusoid_local_scene(rng, target_count=int(target_count), extremum_sign=int(sign))
    if str(family) == FAMILY_PIECEWISE_LINEAR:
        vertices, annotation_points = _piecewise_polyline_for_local_extrema(
            rng,
            target_count=int(target_count),
            extremum_sign=int(sign),
        )
        return _piecewise_scene(
            vertices=vertices,
            annotation_points=annotation_points,
            purpose="local_extrema",
        )
    raise ValueError(f"unsupported function family: {family}")


def _sample_from(values: Sequence[int], rng) -> int:
    if not values:
        raise ValueError("sampling support cannot be empty")
    return int(values[int(rng.randrange(len(values)))])


def _positions_in_window(positions: Sequence[int]) -> Tuple[int, ...]:
    return tuple(int(value) for value in positions if -9 <= int(value) <= 9)


def _sinusoid_maxima_positions(phase_shift: int) -> Tuple[int, ...]:
    return _positions_in_window([int(phase_shift + (12 * k_value)) for k_value in range(-2, 3)])


def _sinusoid_minima_positions(phase_shift: int) -> Tuple[int, ...]:
    return _positions_in_window([int(phase_shift + 6 + (12 * k_value)) for k_value in range(-2, 3)])


def _phase_shift_for_count(*, target_count: int, mode: str) -> int:
    candidates: List[int] = []
    for phase_shift in range(-5, 7):
        if str(mode) == "maxima":
            count = len(_sinusoid_maxima_positions(int(phase_shift)))
        elif str(mode) == "minima":
            count = len(_sinusoid_minima_positions(int(phase_shift)))
        elif str(mode) == "turning":
            count = len(_sinusoid_maxima_positions(int(phase_shift))) + len(_sinusoid_minima_positions(int(phase_shift)))
        else:
            raise ValueError(f"unsupported sinusoid mode: {mode}")
        if int(count) == int(target_count):
            candidates.append(int(phase_shift))
    if not candidates:
        raise ValueError(f"no sinusoid phase_shift supports {mode} target_count={target_count}")
    return int(candidates[0])


def _sinusoid_sample_points(
    *,
    amplitude: int,
    phase_shift: int,
    midline_y: int,
) -> Tuple[GraphPolylinePoint, ...]:
    return tuple(
        (
            float(x_value) / 4.0,
            float(
                (float(amplitude) * math.cos((math.pi / 6.0) * (((float(x_value) / 4.0) - float(phase_shift)))))
                + float(midline_y)
            ),
        )
        for x_value in range(-40, 41)
    )


def _build_sinusoid_turning_scene(rng, *, target_count: int) -> SampledFunctionGraph:
    if int(target_count) not in {3, 4}:
        raise ValueError(f"unsupported sinusoid turning target_count: {target_count}")
    amplitude = int(rng.randint(2, 4))
    phase_shift = _phase_shift_for_count(target_count=int(target_count), mode="turning")
    midline_y = int(_sample_from((-2, -1, 0, 1, 2), rng))
    annotation_points = tuple(
        (int(x_value), int(midline_y + amplitude)) for x_value in _sinusoid_maxima_positions(int(phase_shift))
    ) + tuple((int(x_value), int(midline_y - amplitude)) for x_value in _sinusoid_minima_positions(int(phase_shift)))
    parameters = {"amplitude": int(amplitude), "phase_shift": int(phase_shift), "midline_y": int(midline_y), "period": 12}
    return _function_scene(
        family=FAMILY_SINUSOID,
        polyline=_sinusoid_sample_points(amplitude=int(amplitude), phase_shift=int(phase_shift), midline_y=int(midline_y)),
        annotation_points=annotation_points,
        parameters=parameters,
    )


def _build_sinusoid_local_scene(rng, *, target_count: int, extremum_sign: int) -> SampledFunctionGraph:
    if int(target_count) not in {1, 2}:
        raise ValueError(f"unsupported sinusoid local-extremum target_count: {target_count}")
    amplitude = int(rng.randint(2, 4))
    sign = -1 if int(extremum_sign) < 0 else 1
    mode = "minima" if sign < 0 else "maxima"
    phase_shift = _phase_shift_for_count(target_count=int(target_count), mode=mode)
    midline_y = int(_sample_from((-2, -1, 0, 1, 2), rng))
    positions = _sinusoid_minima_positions(int(phase_shift)) if sign < 0 else _sinusoid_maxima_positions(int(phase_shift))
    annotation_points = tuple((int(x_value), int(midline_y + (sign * amplitude))) for x_value in positions)
    parameters = {"amplitude": int(amplitude), "phase_shift": int(phase_shift), "midline_y": int(midline_y), "period": 12}
    return _function_scene(
        family=FAMILY_SINUSOID,
        polyline=_sinusoid_sample_points(amplitude=int(amplitude), phase_shift=int(phase_shift), midline_y=int(midline_y)),
        annotation_points=annotation_points,
        parameters=parameters,
    )


def _piecewise_x_positions(key: str) -> Tuple[int, ...]:
    return int_tuple_default(str(key), DEFAULTS.piecewise_turning_x_positions)


def _piecewise_polyline_for_turning_points(
    rng,
    *,
    target_count: int,
) -> Tuple[Tuple[GraphPoint, ...], Tuple[GraphPoint, ...]]:
    vertex_count = int(target_count) + 2
    x_positions = list(_piecewise_x_positions("piecewise_turning_x_positions"))
    start_index = int(rng.randint(0, len(x_positions) - vertex_count))
    selected_x = x_positions[start_index : start_index + vertex_count]
    sign_start = int(rng.choice((-1, 1)))
    vertices = tuple(
        (int(x_value), int(sign_start * ((-1) ** index) * int(rng.randint(2, 5))))
        for index, x_value in enumerate(selected_x)
    )
    return vertices, tuple(vertices[1:-1])


def _piecewise_polyline_for_local_extrema(
    rng,
    *,
    target_count: int,
    extremum_sign: int,
) -> Tuple[Tuple[GraphPoint, ...], Tuple[GraphPoint, ...]]:
    sign = -1 if int(extremum_sign) < 0 else 1
    vertex_count = int((2 * int(target_count)) + 1)
    x_positions = list(_piecewise_x_positions("piecewise_turning_x_positions"))
    start_index = int(rng.randint(0, len(x_positions) - vertex_count))
    selected_x = x_positions[start_index : start_index + vertex_count]
    target_is_low = sign < 0
    vertices: List[GraphPoint] = []
    annotation_points: List[GraphPoint] = []
    for index, x_value in enumerate(selected_x):
        is_target_slot = bool(index % 2 == 1)
        high = int(rng.randint(2, 5))
        low = -int(rng.randint(2, 5))
        y_value = low if (is_target_slot == target_is_low) else high
        point = (int(x_value), int(y_value))
        vertices.append(point)
        if 0 < index < (vertex_count - 1) and is_target_slot:
            annotation_points.append(point)
    return tuple(vertices), tuple(annotation_points)


def _function_scene(
    *,
    family: str,
    polyline: Sequence[GraphPolylinePoint],
    annotation_points: Sequence[GraphPoint],
    parameters: Mapping[str, Any],
) -> SampledFunctionGraph:
    return SampledFunctionGraph(
        polyline_graph=tuple((float(x), float(y)) for x, y in polyline),
        annotation_graph_points=tuple((int(x), int(y)) for x, y in annotation_points),
        scene_entities=[
            {
                "entity_id": "function_graph",
                "entity_type": "function_graph",
                "family": str(family),
                "parameters": dict(parameters),
            }
        ],
        render_map={
            "scene_variant": str(family),
            "function_parameters": dict(parameters),
            "annotation_points_graph": [list(point) for point in annotation_points],
        },
        execution_trace={
            "family": str(family),
            "parameters": dict(parameters),
            "annotation_points_graph": [list(point) for point in annotation_points],
        },
        object_count=1,
    )


def _piecewise_scene(
    *,
    vertices: Sequence[GraphPolylinePoint],
    annotation_points: Sequence[GraphPoint],
    purpose: str,
) -> SampledFunctionGraph:
    return SampledFunctionGraph(
        polyline_graph=tuple((float(point[0]), float(point[1])) for point in vertices),
        annotation_graph_points=tuple((int(x), int(y)) for x, y in annotation_points),
        scene_entities=[
            {
                "entity_id": "function_graph",
                "entity_type": "function_graph",
                "family": FAMILY_PIECEWISE_LINEAR,
                "vertices_graph": [list(point) for point in vertices],
            }
        ],
        render_map={
            "scene_variant": FAMILY_PIECEWISE_LINEAR,
            "polyline_vertices_graph": [list(point) for point in vertices],
            "annotation_points_graph": [list(point) for point in annotation_points],
        },
        execution_trace={
            "family": FAMILY_PIECEWISE_LINEAR,
            "purpose": str(purpose),
            "polyline_vertices_graph": [list(point) for point in vertices],
            "annotation_points_graph": [list(point) for point in annotation_points],
        },
        object_count=int(len(vertices)),
    )
