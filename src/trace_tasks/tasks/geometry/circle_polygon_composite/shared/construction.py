"""Construction primitives for circle-polygon-composite diagrams."""

from __future__ import annotations

import math
from collections import defaultdict
from functools import lru_cache
from typing import Any, Dict, Mapping, Sequence

from trace_tasks.core.sampling import uniform_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.shared.fixed_query import geometry_selected_probability_map
from trace_tasks.tasks.geometry.shared.vector2d import mul as _mul

from .state import CONSTRUCTION_KINDS, Point, SIDE_KEYS


ANGLE_SUPPORT: tuple[int, ...] = tuple(range(18, 73))
SIDE_SIGN_SUPPORT: tuple[int, int] = (-1, 1)
_TANGENT_VALUE_SUPPORT: range = range(4, 37)
_TANGENT_MAX_RATIO: float = 2.8
_TANGENT_CASES_PER_SIDE_ANSWER: int = 8


def case_key(case: Sequence[int]) -> str:
    """Return a stable support key for one tangent-length case."""

    return "-".join(str(int(value)) for value in case)


def select_missing_side(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
) -> tuple[str, dict[str, float]]:
    """Select which quadrilateral side length is hidden."""

    explicit = params.get("missing_side")
    if explicit is not None:
        missing_side = str(explicit)
        if missing_side not in SIDE_KEYS:
            raise ValueError(f"unsupported missing_side: {missing_side}")
        return missing_side, geometry_selected_probability_map(SIDE_KEYS, selected=missing_side)
    rng = spawn_rng(int(instance_seed), str(namespace))
    missing_side = str(uniform_choice(rng, SIDE_KEYS))
    return missing_side, geometry_selected_probability_map(SIDE_KEYS)


def _generate_tangent_cases() -> tuple[tuple[int, int, int, int], ...]:
    """Generate a compact broad support of tangential quadrilateral cases."""

    grouped: dict[tuple[str, int], list[tuple[int, int, int, int]]] = defaultdict(list)
    cases: list[tuple[int, int, int, int]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for t_a in _TANGENT_VALUE_SUPPORT:
        for t_b in _TANGENT_VALUE_SUPPORT:
            for t_c in _TANGENT_VALUE_SUPPORT:
                for t_d in _TANGENT_VALUE_SUPPORT:
                    case = (int(t_a), int(t_b), int(t_c), int(t_d))
                    if float(max(case)) / float(min(case)) > _TANGENT_MAX_RATIO:
                        continue
                    side_lengths = side_lengths_from_vertex_tangents(case)
                    needs_case = any(
                        len(grouped[(str(side), int(answer))]) < _TANGENT_CASES_PER_SIDE_ANSWER
                        for side, answer in side_lengths.items()
                    )
                    if not needs_case:
                        continue
                    if case not in seen:
                        cases.append(case)
                        seen.add(case)
                    for side, answer in side_lengths.items():
                        key = (str(side), int(answer))
                        if len(grouped[key]) < _TANGENT_CASES_PER_SIDE_ANSWER:
                            grouped[key].append(case)
    return tuple(cases)


@lru_cache(maxsize=None)
def _tangent_cases_by_answer(missing_side: str) -> dict[str, tuple[tuple[int, int, int, int], ...]]:
    """Return tangent cases grouped by prompt-facing missing side length."""

    side = str(missing_side)
    if side not in SIDE_KEYS:
        raise ValueError(f"unsupported missing_side: {missing_side}")
    grouped: dict[str, list[tuple[int, int, int, int]]] = defaultdict(list)
    for case in TANGENT_CASES:
        answer = int(side_lengths_from_vertex_tangents(case)[side])
        grouped[str(answer)].append(case)
    return {key: tuple(values) for key, values in grouped.items()}


def select_tangent_case(
    *,
    missing_side: str,
    params: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
) -> tuple[tuple[int, int, int, int], dict[str, float], dict[str, float]]:
    """Select or validate one set of vertex tangents by final side answer first."""

    explicit = params.get("tangent_lengths")
    if explicit is not None:
        if not isinstance(explicit, Sequence) or isinstance(explicit, (str, bytes)) or len(explicit) != 4:
            raise ValueError("tangent_lengths must be a four-item sequence")
        case = tuple(int(value) for value in explicit)
        if any(value <= 0 for value in case):
            raise ValueError("tangent_lengths values must be positive")
        answer = int(side_lengths_from_vertex_tangents(case)[str(missing_side)])
        return case, {case_key(case): 1.0}, {str(answer): 1.0}

    answer_cases = _tangent_cases_by_answer(str(missing_side))
    answer_keys = tuple(sorted(answer_cases, key=lambda key: int(key)))
    if not answer_keys:
        raise ValueError("tangent answer support must not be empty")
    support_probability = 1.0 / float(len(answer_keys))
    answer_probabilities = {key: float(support_probability) for key in answer_keys}
    explicit_answer = params.get("target_answer")
    if explicit_answer is not None:
        answer_key = str(int(explicit_answer))
        if answer_key not in answer_cases:
            raise ValueError(f"target_answer={explicit_answer} is not supported for side {missing_side}")
    else:
        rng = spawn_rng(int(instance_seed), f"{namespace}.answer")
        answer_key = str(uniform_choice(rng, answer_keys))
    cases = tuple(answer_cases[str(answer_key)])
    rng = spawn_rng(int(instance_seed), f"{namespace}.case.{answer_key}")
    case = tuple(int(value) for value in uniform_choice(rng, cases))
    return case, {case_key(case): 1.0}, answer_probabilities


def side_lengths_from_vertex_tangents(case: Sequence[int]) -> dict[str, int]:
    """Convert vertex tangent lengths into side lengths for a tangential quadrilateral."""

    t_a, t_b, t_c, t_d = [int(value) for value in case]
    return {
        "AB": int(t_a + t_b),
        "BC": int(t_b + t_c),
        "CD": int(t_c + t_d),
        "DA": int(t_d + t_a),
    }


TANGENT_CASES: tuple[tuple[int, int, int, int], ...] = _generate_tangent_cases()


def vertex_tangents_from_case(case: Sequence[int]) -> dict[str, int]:
    """Return per-vertex tangent lengths keyed by quadrilateral vertex."""

    t_a, t_b, t_c, t_d = [int(value) for value in case]
    return {"A": int(t_a), "B": int(t_b), "C": int(t_c), "D": int(t_d)}


def select_angle_degrees(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    namespace: str,
) -> tuple[int, dict[str, float]]:
    """Select or validate the target angle support value."""

    explicit = params.get("target_angle")
    if explicit is not None:
        answer = int(explicit)
        if answer not in ANGLE_SUPPORT:
            raise ValueError(f"target_angle must be one of {ANGLE_SUPPORT}")
        return answer, geometry_selected_probability_map(ANGLE_SUPPORT, selected=answer)
    rng = spawn_rng(int(instance_seed), str(namespace))
    answer = int(uniform_choice(rng, ANGLE_SUPPORT))
    return answer, geometry_selected_probability_map(ANGLE_SUPPORT)


def select_side_sign(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    namespace: str,
) -> tuple[int, dict[str, float]]:
    """Select which side of the construction receives the tangent line."""

    explicit = params.get("side_sign")
    if explicit is not None:
        side_sign = int(explicit)
        if side_sign not in SIDE_SIGN_SUPPORT:
            raise ValueError("side_sign must be -1 or 1")
        return side_sign, geometry_selected_probability_map(SIDE_SIGN_SUPPORT, selected=side_sign)
    rng = spawn_rng(int(instance_seed), str(namespace))
    side_sign = int(uniform_choice(rng, SIDE_SIGN_SUPPORT))
    return side_sign, geometry_selected_probability_map(SIDE_SIGN_SUPPORT)


def select_construction_kind(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    namespace: str,
) -> tuple[str, dict[str, float]]:
    """Select which tangent-angle construction is drawn without making it a query id."""

    explicit = params.get("construction_kind")
    if explicit is not None:
        construction_kind = validate_construction_kind(str(explicit))
        return construction_kind, geometry_selected_probability_map(CONSTRUCTION_KINDS, selected=construction_kind)
    rng = spawn_rng(int(instance_seed), str(namespace))
    construction_kind = str(uniform_choice(rng, CONSTRUCTION_KINDS))
    return construction_kind, geometry_selected_probability_map(CONSTRUCTION_KINDS)


def validate_construction_kind(construction_kind: str) -> str:
    """Return a validated construction kind for angle-transfer diagrams."""

    kind = str(construction_kind)
    if kind not in CONSTRUCTION_KINDS:
        raise ValueError(f"unsupported construction_kind: {kind}")
    return kind


def _solve_inradius(tangents: Sequence[int]) -> float:
    """Solve the inradius whose tangent-angle gaps close the quadrilateral."""

    lengths = [float(value) for value in tangents]

    def total_gap(radius: float) -> float:
        return sum(2.0 * math.atan(float(radius) / length) for length in lengths)

    low = 1e-6
    high = max(lengths)
    while total_gap(high) <= 2.0 * math.pi:
        high *= 2.0
    for _ in range(90):
        mid = (low + high) / 2.0
        if total_gap(mid) < 2.0 * math.pi:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def _normal(angle: float) -> Point:
    return (math.cos(float(angle)), math.sin(float(angle)))


def _line_intersection(normal_a: Point, normal_b: Point, radius: float) -> Point:
    ax, ay = float(normal_a[0]), float(normal_a[1])
    bx, by = float(normal_b[0]), float(normal_b[1])
    det = (ax * by) - (ay * bx)
    if abs(det) <= 1e-9:
        raise ValueError("near-parallel tangent lines")
    return (float(radius) * (by - ay) / det, float(radius) * (ax - bx) / det)


def tangential_local_geometry(
    vertex_tangents: Mapping[str, int],
) -> tuple[Dict[str, Point], Dict[str, Point], float]:
    """Construct local vertices, tangency points, and inradius before rendering."""

    t_a = int(vertex_tangents["A"])
    t_b = int(vertex_tangents["B"])
    t_c = int(vertex_tangents["C"])
    t_d = int(vertex_tangents["D"])
    radius = _solve_inradius((t_a, t_b, t_c, t_d))
    gap_b = 2.0 * math.atan(float(radius) / float(t_b))
    gap_c = 2.0 * math.atan(float(radius) / float(t_c))
    gap_d = 2.0 * math.atan(float(radius) / float(t_d))
    phi_ab = 0.0
    phi_bc = phi_ab + gap_b
    phi_cd = phi_bc + gap_c
    phi_da = phi_cd + gap_d
    normals = {
        "AB": _normal(phi_ab),
        "BC": _normal(phi_bc),
        "CD": _normal(phi_cd),
        "DA": _normal(phi_da),
    }
    vertices = {
        "A": _line_intersection(normals["DA"], normals["AB"], radius),
        "B": _line_intersection(normals["AB"], normals["BC"], radius),
        "C": _line_intersection(normals["BC"], normals["CD"], radius),
        "D": _line_intersection(normals["CD"], normals["DA"], radius),
    }
    tangency_points = {side: _mul(normal, radius) for side, normal in normals.items()}
    return vertices, tangency_points, float(radius)


__all__ = [
    "ANGLE_SUPPORT",
    "SIDE_SIGN_SUPPORT",
    "TANGENT_CASES",
    "case_key",
    "select_angle_degrees",
    "select_missing_side",
    "select_side_sign",
    "select_tangent_case",
    "side_lengths_from_vertex_tangents",
    "tangential_local_geometry",
    "validate_construction_kind",
    "vertex_tangents_from_case",
]
