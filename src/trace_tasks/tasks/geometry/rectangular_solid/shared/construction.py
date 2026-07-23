"""Formula resolution for rectangular-solid scene primitives."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from .sampling import (
    CUBE_EDGE_VALUES,
    CUBOID_MISSING_DIMENSION_VALUES,
    OPEN_BOX_BASE_DIMENSION_VALUES,
    CUBOID_SURFACE_AREA_VALUES,
    open_box_values_for_case,
    probability_map_for_support,
    select_cube_edge,
    select_cuboid_case_for_missing_dimension,
    select_cuboid_case_for_surface_area,
    select_open_box_case_for_dimension,
    select_open_box_dimension_role,
    select_partial_frame_edge_count,
    surface_area_for_case,
)
from .state import CubeFrameProblem, CuboidMeasureProblem, OpenBoxNetProblem

CUBOID_VOLUME_MISSING_DIMENSION_FORMULA = "missing_dimension = volume / (known_dimension_1 * known_dimension_2)"
CUBOID_SURFACE_AREA_FORMULA = "surface_area = 2 * (length*width + length*height + width*height)"
CUBE_FRAME_EDGE_FORMULA = "cube_edge = frame_length / visible_frame_edge_count"
OPEN_BOX_NET_FORMULA = (
    "base_length = sheet_length - 2*cut_size; "
    "base_width = sheet_width - 2*cut_size; "
    "volume = base_length*base_width*cut_size"
)


def cuboid_dimension_answer_support(*, selected: int, target_role: str) -> Dict[str, float]:
    """Return answer support for one cuboid missing-dimension role."""

    if str(target_role) not in {"length", "width", "height"}:
        raise ValueError("target_role must be length, width, or height")
    return probability_map_for_support(CUBOID_MISSING_DIMENSION_VALUES, selected=int(selected))


def cuboid_surface_area_answer_support(*, selected: int) -> Dict[str, float]:
    """Return answer support for cuboid surface-area values."""

    return probability_map_for_support(CUBOID_SURFACE_AREA_VALUES, selected=int(selected))


def cube_edge_answer_support(*, selected: int) -> Dict[str, float]:
    """Return answer support for cube edge values."""

    return probability_map_for_support(CUBE_EDGE_VALUES, selected=int(selected))


def open_box_answer_support(*, selected: int, target_role: str) -> Dict[str, float]:
    """Return answer support for one open-box target role."""

    if str(target_role) not in {"base_length", "base_width"}:
        raise ValueError("target_role must be base_length or base_width")
    return probability_map_for_support(OPEN_BOX_BASE_DIMENSION_VALUES, selected=int(selected))


def resolve_cuboid_missing_dimension(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    target_role: str,
    sampling_label: str,
) -> CuboidMeasureProblem:
    """Resolve a cuboid volume problem with one hidden dimension."""

    if str(target_role) not in {"length", "width", "height"}:
        raise ValueError("target_role must be length, width, or height")
    case, case_probabilities = select_cuboid_case_for_missing_dimension(
        instance_seed=int(instance_seed),
        params=params,
        sampling_label=str(sampling_label),
        target_role=str(target_role),
    )
    length, width, height = [int(value) for value in case]
    volume = int(length * width * height)
    explicit_volume = params.get("volume_units")
    if explicit_volume is not None and int(explicit_volume) != int(volume):
        raise ValueError("volume_units must equal length_units * width_units * height_units")
    answer = {"length": length, "width": width, "height": height}[str(target_role)]
    return CuboidMeasureProblem(
        target_role=str(target_role),
        length=int(length),
        width=int(width),
        height=int(height),
        volume=int(volume),
        surface_area=int(surface_area_for_case(case)),
        answer=int(answer),
        formula_family="cuboid_volume_missing_dimension",
        formula=CUBOID_VOLUME_MISSING_DIMENSION_FORMULA,
        case_probabilities=dict(case_probabilities),
        answer_support_probabilities=cuboid_dimension_answer_support(
            selected=int(answer),
            target_role=str(target_role),
        ),
    )


def resolve_cuboid_surface_area(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    sampling_label: str,
) -> CuboidMeasureProblem:
    """Resolve a cuboid total-surface-area problem."""

    case, case_probabilities = select_cuboid_case_for_surface_area(
        instance_seed=int(instance_seed),
        params=params,
        sampling_label=str(sampling_label),
    )
    length, width, height = [int(value) for value in case]
    volume = int(length * width * height)
    surface_area = int(surface_area_for_case(case))
    explicit_surface_area = params.get("surface_area_units")
    if explicit_surface_area is not None and int(explicit_surface_area) != int(surface_area):
        raise ValueError("surface_area_units must equal 2 * (length*width + length*height + width*height)")
    return CuboidMeasureProblem(
        target_role="surface_area",
        length=int(length),
        width=int(width),
        height=int(height),
        volume=int(volume),
        surface_area=int(surface_area),
        answer=int(surface_area),
        formula_family="cuboid_surface_area",
        formula=CUBOID_SURFACE_AREA_FORMULA,
        case_probabilities=dict(case_probabilities),
        answer_support_probabilities=cuboid_surface_area_answer_support(selected=int(surface_area)),
    )


def resolve_cube_frame_edge(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    frame_mode: str,
    sampling_label: str,
) -> CubeFrameProblem:
    """Resolve cube edge length from total or highlighted frame length."""

    edge, edge_probabilities = select_cube_edge(
        instance_seed=int(instance_seed),
        params=params,
        sampling_label=str(sampling_label),
    )
    if str(frame_mode) == "total":
        visible_frame_edge_count = 12
        frame_length = int(edge * visible_frame_edge_count)
        explicit_total = params.get("total_frame_length_units")
        if explicit_total is not None and int(explicit_total) != int(frame_length):
            raise ValueError("total_frame_length_units must equal 12 * edge_units")
        case_probabilities = {
            f"edge{candidate}_total12": probability
            for candidate, probability in edge_probabilities.items()
        }
    elif str(frame_mode) == "partial":
        visible_frame_edge_count, count_probabilities = select_partial_frame_edge_count(
            instance_seed=int(instance_seed),
            params=params,
            sampling_label=str(sampling_label),
        )
        frame_length = int(edge * visible_frame_edge_count)
        explicit_partial = params.get("highlighted_frame_length_units")
        if explicit_partial is not None and int(explicit_partial) != int(frame_length):
            raise ValueError("highlighted_frame_length_units must equal highlighted_edge_count * edge_units")
        case_probabilities = {
            f"edge{candidate_edge}_partial{candidate_count}": float(edge_probability) * float(count_probability)
            for candidate_edge, edge_probability in edge_probabilities.items()
            for candidate_count, count_probability in count_probabilities.items()
        }
    else:
        raise ValueError("frame_mode must be total or partial")
    explicit_edge = params.get("edge_units")
    if explicit_edge is not None and int(explicit_edge) != int(edge):
        raise ValueError("edge_units mismatch")
    return CubeFrameProblem(
        frame_mode=str(frame_mode),
        cube_edge=int(edge),
        visible_frame_edge_count=int(visible_frame_edge_count),
        frame_length=int(frame_length),
        answer=int(edge),
        formula_family="cube_edge_from_frame_length",
        formula=CUBE_FRAME_EDGE_FORMULA,
        case_probabilities=dict(case_probabilities),
        answer_support_probabilities=cube_edge_answer_support(selected=int(edge)),
    )


def resolve_open_box_dimension(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    sampling_label: str,
) -> OpenBoxNetProblem:
    """Resolve the marked base dimension of one open-box net."""

    target_role, role_probabilities = select_open_box_dimension_role(
        instance_seed=int(instance_seed),
        params=params,
        sampling_label=str(sampling_label),
    )
    case, case_probabilities = select_open_box_case_for_dimension(
        instance_seed=int(instance_seed),
        params=params,
        sampling_label=str(sampling_label),
        target_role=str(target_role),
    )
    sheet_length, sheet_width, cut_size, base_length, base_width, volume = open_box_values_for_case(case)
    answer = base_length if target_role == "base_length" else base_width
    if len(role_probabilities) > 1:
        case_probabilities = {
            f"{case_key}_{role}": float(case_probability) * float(role_probability)
            for case_key, case_probability in case_probabilities.items()
            for role, role_probability in role_probabilities.items()
        }
    else:
        case_probabilities = {
            f"{case_key}_{target_role}": probability
            for case_key, probability in case_probabilities.items()
        }
    return OpenBoxNetProblem(
        target_role=str(target_role),
        sheet_length=int(sheet_length),
        sheet_width=int(sheet_width),
        cut_size=int(cut_size),
        base_length=int(base_length),
        base_width=int(base_width),
        open_box_volume=int(volume),
        answer=int(answer),
        formula_family="open_box_net_corner_cut",
        formula=OPEN_BOX_NET_FORMULA,
        case_probabilities=dict(case_probabilities),
        answer_support_probabilities=open_box_answer_support(
            selected=int(answer),
            target_role=str(target_role),
        ),
    )


__all__ = [
    "CUBE_FRAME_EDGE_FORMULA",
    "CUBOID_SURFACE_AREA_FORMULA",
    "CUBOID_VOLUME_MISSING_DIMENSION_FORMULA",
    "OPEN_BOX_NET_FORMULA",
    "cube_edge_answer_support",
    "cuboid_dimension_answer_support",
    "cuboid_surface_area_answer_support",
    "open_box_answer_support",
    "resolve_cube_frame_edge",
    "resolve_cuboid_missing_dimension",
    "resolve_cuboid_surface_area",
    "resolve_open_box_dimension",
]
