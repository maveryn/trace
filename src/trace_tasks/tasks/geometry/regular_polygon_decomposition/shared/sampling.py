"""Scene-neutral sampling helpers for regular-polygon measurements."""

from __future__ import annotations

import math
from typing import Any, Mapping

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.geometry.shared.measurement_rendering import round1

from .state import RegularPolygonProblem

N_SUPPORT: tuple[int, ...] = (5, 6, 8, 9, 10, 12)
WEDGE_AREA_SUPPORT: tuple[int, ...] = (8, 10, 12, 15, 18, 20, 24, 30)
SIDE_LENGTH_SUPPORT: tuple[int, ...] = (5, 6, 7, 8, 9, 10, 12)


def _rng(instance_seed: int, seed_namespace: str):
    return spawn_rng(int(instance_seed), str(seed_namespace))


def _n_sides(rng: Any, params: Mapping[str, Any]) -> int:
    return int(params.get("n_sides", N_SUPPORT[int(rng.randrange(len(N_SUPPORT)))]))


def _start_index(rng: Any, n_sides: int, wedge_count: int) -> int:
    return int(rng.randrange(max(0, int(n_sides) - int(wedge_count)) + 1))


def _clean_side_apothem_case(rng: Any, n_sides: int) -> tuple[float, float, float, float, float]:
    """Choose readout labels whose rounded formulas recover clean integer answers."""

    side_order = list(SIDE_LENGTH_SUPPORT)
    start = int(rng.randrange(len(side_order)))
    rotated_side_order = side_order[start:] + side_order[:start]
    for side_candidate in rotated_side_order:
        side_length = float(side_candidate)
        apothem = round1(float(side_length) / (2.0 * math.tan(math.pi / float(n_sides))))
        if float(apothem) <= 0.0:
            continue
        wedge_area = round1(float(side_length) * float(apothem) / 2.0)
        total_area = round1(float(n_sides) * float(wedge_area))
        perimeter = float(n_sides) * float(side_length)
        perimeter_from_labels = (2.0 * float(total_area)) / float(apothem)
        side_from_total_labels = (2.0 * float(total_area)) / (float(n_sides) * float(apothem))
        side_from_wedge_labels = (2.0 * float(wedge_area)) / float(apothem)
        if (
            math.isclose(perimeter_from_labels, perimeter, abs_tol=1e-8)
            and math.isclose(side_from_total_labels, side_length, abs_tol=1e-8)
            and math.isclose(side_from_wedge_labels, side_length, abs_tol=1e-8)
        ):
            return side_length, apothem, wedge_area, total_area, perimeter
    raise RuntimeError(f"could not find clean regular-polygon labels for n_sides={n_sides}")


def _base_problem(
    *,
    rng: Any,
    params: Mapping[str, Any],
    instance_seed: int,
    wedge_count: int,
    answer: float,
    answer_type: str,
    target_name: str,
    relation: str,
    total_area: float | None = None,
    wedge_area: float | None = None,
    side_length: float | None = None,
    apothem: float | None = None,
    perimeter: float | None = None,
    **visual_flags: bool,
) -> RegularPolygonProblem:
    n_sides = _n_sides(rng, params)
    central_angle = int(round(360.0 / float(n_sides)))
    return RegularPolygonProblem(
        n_sides=int(n_sides),
        wedge_count=int(wedge_count),
        start_index=_start_index(rng, int(n_sides), int(wedge_count)),
        answer=float(answer),
        answer_type=str(answer_type),
        target_name=str(target_name),
        relation=str(relation),
        total_area=total_area,
        wedge_area=wedge_area,
        side_length=side_length,
        apothem=apothem,
        perimeter=perimeter,
        central_angle_degrees=int(central_angle),
        case_index=int(rng.randrange(10_000_000)),
        layout_seed=int(instance_seed),
        **visual_flags,
    )


def area_from_marked_equal_pieces(instance_seed: int, params: Mapping[str, Any], *, seed_namespace: str) -> RegularPolygonProblem:
    rng = _rng(instance_seed, seed_namespace)
    n_sides = _n_sides(rng, params)
    max_marked = max(1, min(4, int(n_sides) // 2))
    wedge_count = int(params.get("wedge_count", rng.randrange(1, max_marked + 1)))
    if int(wedge_count) < 1 or int(wedge_count) > int(max_marked):
        raise ValueError(f"wedge_count must be between 1 and {max_marked} for n_sides={n_sides}")
    wedge_area = float(WEDGE_AREA_SUPPORT[int(rng.randrange(len(WEDGE_AREA_SUPPORT)))])
    return _base_problem(
        rng=rng,
        params={**dict(params), "n_sides": n_sides},
        instance_seed=int(instance_seed),
        wedge_count=int(wedge_count),
        answer=float(wedge_count) * float(wedge_area),
        answer_type="number",
        target_name="the shaded wedge" if int(wedge_count) == 1 else "the shaded region",
        relation="contiguous_wedge_group_area_from_total_regular_polygon_area",
        total_area=float(n_sides) * float(wedge_area),
        wedge_area=float(wedge_area),
        show_total_area_readout=True,
    )


def area_from_side_and_apothem(instance_seed: int, params: Mapping[str, Any], *, seed_namespace: str) -> RegularPolygonProblem:
    rng = _rng(instance_seed, seed_namespace)
    n_sides = _n_sides(rng, params)
    side_length, apothem, wedge_area, _total_area, _perimeter = _clean_side_apothem_case(rng, int(n_sides))
    return _base_problem(
        rng=rng,
        params={**dict(params), "n_sides": n_sides},
        instance_seed=int(instance_seed),
        wedge_count=1,
        answer=float(wedge_area),
        answer_type="number",
        target_name="the shaded wedge",
        relation="triangle_wedge_area_from_side_and_apothem",
        wedge_area=float(wedge_area),
        side_length=float(side_length),
        apothem=float(apothem),
        show_known_side_length=True,
        show_apothem=True,
        show_midpoint_label=True,
    )


def angle_for_marked_pieces(instance_seed: int, params: Mapping[str, Any], *, seed_namespace: str) -> RegularPolygonProblem:
    rng = _rng(instance_seed, seed_namespace)
    n_sides = _n_sides(rng, params)
    central_angle = int(round(360.0 / float(n_sides)))
    max_marked = max(1, min(2, int(n_sides) // 2))
    wedge_count = int(params.get("wedge_count", rng.randrange(1, max_marked + 1)))
    if int(wedge_count) < 1 or int(wedge_count) > int(max_marked):
        raise ValueError(f"central-angle wedge_count must be between 1 and {max_marked} for n_sides={n_sides}")
    return _base_problem(
        rng=rng,
        params={**dict(params), "n_sides": n_sides},
        instance_seed=int(instance_seed),
        wedge_count=int(wedge_count),
        answer=float(wedge_count * central_angle),
        answer_type="integer",
        target_name="AOB",
        relation="contiguous_regular_polygon_center_wedge_angle_sum",
        show_angle_unknown=True,
    )


def perimeter_from_area_apothem(instance_seed: int, params: Mapping[str, Any], *, seed_namespace: str) -> RegularPolygonProblem:
    rng = _rng(instance_seed, seed_namespace)
    n_sides = _n_sides(rng, params)
    side_length, apothem, wedge_area, total_area, perimeter = _clean_side_apothem_case(rng, int(n_sides))
    return _base_problem(
        rng=rng,
        params={**dict(params), "n_sides": n_sides},
        instance_seed=int(instance_seed),
        wedge_count=1,
        answer=float(perimeter),
        answer_type="integer",
        target_name="the regular polygon perimeter",
        relation="regular_polygon_perimeter_from_area_and_apothem",
        total_area=float(total_area),
        wedge_area=float(wedge_area),
        side_length=float(side_length),
        apothem=float(apothem),
        perimeter=float(perimeter),
        show_apothem=True,
        show_total_area_readout=True,
        show_shaded_region=False,
        show_side_endpoint_labels=False,
        show_midpoint_label=True,
    )


def side_from_perimeter(instance_seed: int, params: Mapping[str, Any], *, seed_namespace: str) -> RegularPolygonProblem:
    rng = _rng(instance_seed, seed_namespace)
    n_sides = _n_sides(rng, params)
    side_length, apothem, wedge_area, total_area, perimeter = _clean_side_apothem_case(rng, int(n_sides))
    return _base_problem(
        rng=rng,
        params={**dict(params), "n_sides": n_sides},
        instance_seed=int(instance_seed),
        wedge_count=1,
        answer=float(side_length),
        answer_type="integer",
        target_name="the marked side",
        relation="regular_polygon_side_length_from_perimeter",
        total_area=float(total_area),
        wedge_area=float(wedge_area),
        side_length=float(side_length),
        apothem=float(apothem),
        perimeter=float(perimeter),
        show_unknown_side_length=True,
        show_perimeter_readout=True,
    )


def side_from_area_apothem(instance_seed: int, params: Mapping[str, Any], *, seed_namespace: str) -> RegularPolygonProblem:
    rng = _rng(instance_seed, seed_namespace)
    n_sides = _n_sides(rng, params)
    side_length, apothem, wedge_area, total_area, perimeter = _clean_side_apothem_case(rng, int(n_sides))
    return _base_problem(
        rng=rng,
        params={**dict(params), "n_sides": n_sides},
        instance_seed=int(instance_seed),
        wedge_count=1,
        answer=float(side_length),
        answer_type="integer",
        target_name="the marked side",
        relation="regular_polygon_side_length_from_area_and_apothem",
        total_area=float(total_area),
        wedge_area=float(wedge_area),
        side_length=float(side_length),
        apothem=float(apothem),
        perimeter=float(perimeter),
        show_unknown_side_length=True,
        show_apothem=True,
        show_total_area_readout=True,
        show_midpoint_label=True,
    )


def side_from_piece_area_apothem(instance_seed: int, params: Mapping[str, Any], *, seed_namespace: str) -> RegularPolygonProblem:
    rng = _rng(instance_seed, seed_namespace)
    n_sides = _n_sides(rng, params)
    side_length, apothem, wedge_area, total_area, perimeter = _clean_side_apothem_case(rng, int(n_sides))
    return _base_problem(
        rng=rng,
        params={**dict(params), "n_sides": n_sides},
        instance_seed=int(instance_seed),
        wedge_count=1,
        answer=float(side_length),
        answer_type="integer",
        target_name="the marked side",
        relation="regular_polygon_side_length_from_wedge_area_and_apothem",
        total_area=float(total_area),
        wedge_area=float(wedge_area),
        side_length=float(side_length),
        apothem=float(apothem),
        perimeter=float(perimeter),
        show_unknown_side_length=True,
        show_apothem=True,
        show_wedge_area_readout=True,
        show_region_label=True,
        show_midpoint_label=True,
    )


__all__ = [
    "area_from_marked_equal_pieces",
    "area_from_side_and_apothem",
    "angle_for_marked_pieces",
    "perimeter_from_area_apothem",
    "side_from_area_apothem",
    "side_from_perimeter",
    "side_from_piece_area_apothem",
]
