"""Sampling and validation primitives for rectangular-solid measurements."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Mapping, Sequence, Tuple

from trace_tasks.core.sampling import uniform_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.shared.fixed_query import (
    geometry_selected_probability_map as _probability_map,
)

CUBE_EDGE_VALUES: Tuple[int, ...] = tuple(range(2, 52))
CUBOID_MISSING_DIMENSION_VALUES: Tuple[int, ...] = tuple(range(3, 53))
PARTIAL_FRAME_EDGE_COUNTS: Tuple[int, ...] = (4, 5, 6)
OPEN_BOX_BASE_DIMENSION_VALUES: Tuple[int, ...] = tuple(range(4, 54))
OPEN_BOX_DIMENSION_ROLES: Tuple[str, ...] = ("base_length", "base_width")
_KNOWN_CUBOID_DIMENSION_VALUES: Tuple[int, ...] = tuple(range(4, 19))
_OPEN_BOX_CUT_VALUES: Tuple[int, ...] = tuple(range(2, 9))
_OPEN_BOX_COMPANION_DIMENSION_VALUES: Tuple[int, ...] = tuple(range(4, 29))


def cuboid_case_key(case: Sequence[int]) -> str:
    """Return a stable trace key for one cuboid dimension case."""

    length, width, height = [int(value) for value in case]
    return f"L{length}_W{width}_H{height}"


def surface_area_for_case(case: Sequence[int]) -> int:
    """Compute total surface area for one cuboid dimension case."""

    length, width, height = [int(value) for value in case]
    return int(2 * ((length * width) + (length * height) + (width * height)))


def _take_evenly(values: Sequence[int], count: int) -> Tuple[int, ...]:
    """Return a deterministic spread of values across a sorted support."""

    sorted_values = tuple(sorted({int(value) for value in values}))
    if len(sorted_values) <= int(count):
        return sorted_values
    if int(count) <= 1:
        return (sorted_values[0],)
    step = (len(sorted_values) - 1) / float(int(count) - 1)
    selected: list[int] = []
    used: set[int] = set()
    for index in range(int(count)):
        value = sorted_values[int(round(float(index) * step))]
        if value in used:
            continue
        selected.append(value)
        used.add(value)
    cursor = 0
    while len(selected) < int(count):
        value = sorted_values[cursor]
        if value not in used:
            selected.append(value)
            used.add(value)
        cursor += 1
    return tuple(sorted(selected))


def _build_surface_area_cases_by_answer() -> Dict[int, Tuple[Tuple[int, int, int], ...]]:
    """Build a compact answer-first cuboid support for surface-area tasks."""

    grouped: dict[int, list[Tuple[int, int, int]]] = defaultdict(list)
    for length in range(5, 35):
        for width in range(4, 27):
            for height in range(3, 23):
                dimensions = (int(length), int(width), int(height))
                if max(dimensions) / float(min(dimensions)) > 5.0:
                    continue
                grouped[surface_area_for_case(dimensions)].append(dimensions)
    selected_answers = _take_evenly(tuple(grouped.keys()), 50)
    return {
        int(answer): tuple(grouped[int(answer)][:12])
        for answer in selected_answers
    }


CUBOID_SURFACE_AREA_CASES_BY_ANSWER: Dict[int, Tuple[Tuple[int, int, int], ...]] = _build_surface_area_cases_by_answer()
CUBOID_SURFACE_AREA_VALUES: Tuple[int, ...] = tuple(sorted(CUBOID_SURFACE_AREA_CASES_BY_ANSWER))
CUBOID_DIMENSION_CASES: Tuple[Tuple[int, int, int], ...] = tuple(
    cases[0] for _answer, cases in sorted(CUBOID_SURFACE_AREA_CASES_BY_ANSWER.items())
)
OPEN_BOX_CASES: Tuple[Tuple[int, int, int], ...] = tuple(
    (answer + 2 * cut_size, companion + 2 * cut_size, cut_size)
    for answer in OPEN_BOX_BASE_DIMENSION_VALUES
    for cut_size in (2,)
    for companion in (8,)
)


def open_box_values_for_case(case: Sequence[int]) -> tuple[int, int, int, int, int, int]:
    """Return sheet, cut, base, and volume values for one open-box case."""

    sheet_length, sheet_width, cut_size = [int(value) for value in case]
    base_length = int(sheet_length - 2 * cut_size)
    base_width = int(sheet_width - 2 * cut_size)
    volume = int(base_length * base_width * cut_size)
    return sheet_length, sheet_width, cut_size, base_length, base_width, volume


def open_box_case_key(case: Sequence[int]) -> str:
    """Return a stable trace key for one open-box net case."""

    sheet_length, sheet_width, cut_size = [int(value) for value in case]
    return f"sheet{sheet_length}x{sheet_width}_cut{cut_size}"


def probability_map_for_support(support: Sequence[int | str], *, selected: int | str) -> Dict[str, float]:
    """Return the selected-value probability map used in task traces."""

    resolved = tuple(dict.fromkeys((*tuple(support), selected)))
    return _probability_map(resolved, selected=selected)


def _select_answer_from_support(
    support: Sequence[int],
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    sampling_label: str,
    support_offset: int | None = None,
) -> tuple[int, Dict[str, float]]:
    """Select a target answer with deterministic finite-support cycling."""

    values = tuple(int(value) for value in support)
    if not values:
        raise ValueError("answer support must not be empty")
    explicit_answer = params.get("target_answer")
    if explicit_answer is None:
        explicit_answer = params.get("answer_value")
    if explicit_answer is not None:
        answer = int(explicit_answer)
        if answer not in values:
            raise ValueError(f"target answer {answer} is outside rectangular-solid support")
    elif params.get("_sample_cursor") is not None:
        cursor = abs(int(params["_sample_cursor"]))
        offset = (
            int(support_offset)
            if support_offset is not None
            else sum(ord(char) for char in str(sampling_label))
        ) % len(values)
        answer = int(values[(cursor + offset) % len(values)])
    else:
        rng = spawn_rng(int(instance_seed), f"{sampling_label}.answer")
        answer = int(uniform_choice(rng, values))
    return int(answer), probability_map_for_support(values, selected=int(answer))


def select_cuboid_case(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    sampling_label: str,
) -> tuple[Tuple[int, int, int], Dict[str, float]]:
    """Select or validate one cuboid dimension triple."""

    explicit_dimensions = params.get("dimensions")
    explicit_dimension_fields = any(key in params for key in ("length_units", "width_units", "height_units"))
    if explicit_dimensions is not None or explicit_dimension_fields:
        if explicit_dimensions is not None:
            if (
                not isinstance(explicit_dimensions, Sequence)
                or isinstance(explicit_dimensions, (str, bytes))
                or len(explicit_dimensions) != 3
            ):
                raise ValueError("dimensions must be [length, width, height]")
            length, width, height = [int(value) for value in explicit_dimensions]
        else:
            missing = [key for key in ("length_units", "width_units", "height_units") if key not in params]
            if missing:
                raise ValueError(f"explicit cuboid dimensions require all three fields; missing {missing}")
            length = int(params["length_units"])
            width = int(params["width_units"])
            height = int(params["height_units"])
        case = (length, width, height)
        validate_dimensions(case)
        return case, {cuboid_case_key(case): 1.0}

    rng = spawn_rng(int(instance_seed), f"{sampling_label}.cuboid_case")
    case = uniform_choice(rng, CUBOID_DIMENSION_CASES)
    probability = 1.0 / float(len(CUBOID_DIMENSION_CASES))
    return case, {cuboid_case_key(candidate): probability for candidate in CUBOID_DIMENSION_CASES}


def select_cuboid_case_for_missing_dimension(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    sampling_label: str,
    target_role: str,
) -> tuple[Tuple[int, int, int], Dict[str, float]]:
    """Select a cuboid case by the hidden dimension answer first."""

    explicit_dimensions = params.get("dimensions")
    explicit_dimension_fields = any(key in params for key in ("length_units", "width_units", "height_units"))
    if explicit_dimensions is not None or explicit_dimension_fields:
        return select_cuboid_case(
            instance_seed=int(instance_seed),
            params=params,
            sampling_label=str(sampling_label),
        )

    answer, _answer_probabilities = _select_answer_from_support(
        CUBOID_MISSING_DIMENSION_VALUES,
        instance_seed=int(instance_seed),
        params=params,
        sampling_label=str(sampling_label),
        support_offset={"length": 0, "width": 17, "height": 34}[str(target_role)],
    )
    rng = spawn_rng(int(instance_seed), f"{sampling_label}.known_dimensions.{answer}")
    known_a = int(uniform_choice(rng, _KNOWN_CUBOID_DIMENSION_VALUES))
    known_b = int(uniform_choice(rng, _KNOWN_CUBOID_DIMENSION_VALUES))
    if str(target_role) == "length":
        case = (int(answer), int(known_a), int(known_b))
    elif str(target_role) == "width":
        case = (int(known_a), int(answer), int(known_b))
    elif str(target_role) == "height":
        case = (int(known_a), int(known_b), int(answer))
    else:
        raise ValueError("target_role must be length, width, or height")
    validate_dimensions(case)
    return case, {cuboid_case_key(case): 1.0}


def select_cuboid_case_for_surface_area(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    sampling_label: str,
) -> tuple[Tuple[int, int, int], Dict[str, float]]:
    """Select a cuboid case by total surface-area answer first."""

    explicit_dimensions = params.get("dimensions")
    explicit_dimension_fields = any(key in params for key in ("length_units", "width_units", "height_units"))
    if explicit_dimensions is not None or explicit_dimension_fields:
        return select_cuboid_case(
            instance_seed=int(instance_seed),
            params=params,
            sampling_label=str(sampling_label),
        )

    answer, _answer_probabilities = _select_answer_from_support(
        CUBOID_SURFACE_AREA_VALUES,
        instance_seed=int(instance_seed),
        params=params,
        sampling_label=str(sampling_label),
    )
    variants = CUBOID_SURFACE_AREA_CASES_BY_ANSWER[int(answer)]
    rng = spawn_rng(int(instance_seed), f"{sampling_label}.surface_case.{answer}")
    case = tuple(int(value) for value in uniform_choice(rng, variants))
    validate_dimensions(case)
    return case, {cuboid_case_key(case): 1.0}


def select_cube_edge(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    sampling_label: str,
) -> tuple[int, Dict[str, float]]:
    """Select or validate one cube edge length."""

    explicit_edge = params.get("edge_units")
    if explicit_edge is not None:
        edge = int(explicit_edge)
        validate_cube_edge(edge)
        return edge, {str(edge): 1.0}
    rng = spawn_rng(int(instance_seed), f"{sampling_label}.cube_edge")
    if params.get("_sample_cursor") is not None:
        edge, probabilities = _select_answer_from_support(
            CUBE_EDGE_VALUES,
            instance_seed=int(instance_seed),
            params=params,
            sampling_label=str(sampling_label),
        )
    else:
        edge = int(uniform_choice(rng, CUBE_EDGE_VALUES))
        probabilities = probability_map_for_support(CUBE_EDGE_VALUES, selected=edge)
    return edge, probabilities


def select_partial_frame_edge_count(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    sampling_label: str,
) -> tuple[int, Dict[str, float]]:
    """Select or validate the number of highlighted cube-frame edges."""

    explicit_count = params.get("highlighted_edge_count")
    if explicit_count is not None:
        count = int(explicit_count)
        if count not in PARTIAL_FRAME_EDGE_COUNTS:
            raise ValueError("highlighted_edge_count must be one of 4, 5, or 6")
        return count, {str(count): 1.0}
    rng = spawn_rng(int(instance_seed), f"{sampling_label}.partial_frame_edge_count")
    count = int(uniform_choice(rng, PARTIAL_FRAME_EDGE_COUNTS))
    probability = 1.0 / float(len(PARTIAL_FRAME_EDGE_COUNTS))
    return count, {str(candidate): probability for candidate in PARTIAL_FRAME_EDGE_COUNTS}


def select_open_box_case(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    sampling_label: str,
) -> tuple[Tuple[int, int, int], Dict[str, float]]:
    """Select or validate one corner-cut open-box net case."""

    explicit_case = params.get("open_box_case")
    explicit_fields = any(key in params for key in ("sheet_length_units", "sheet_width_units", "cut_size_units"))
    if explicit_case is not None or explicit_fields:
        if explicit_case is not None:
            if not isinstance(explicit_case, Sequence) or isinstance(explicit_case, (str, bytes)) or len(explicit_case) != 3:
                raise ValueError("open_box_case must be [sheet_length, sheet_width, cut_size]")
            case = tuple(int(value) for value in explicit_case)
        else:
            missing = [key for key in ("sheet_length_units", "sheet_width_units", "cut_size_units") if key not in params]
            if missing:
                raise ValueError(f"explicit open-box dimensions require all three fields; missing {missing}")
            case = (int(params["sheet_length_units"]), int(params["sheet_width_units"]), int(params["cut_size_units"]))
        validate_open_box_case(case)
        return case, {open_box_case_key(case): 1.0}
    rng = spawn_rng(int(instance_seed), f"{sampling_label}.open_box_case")
    case = uniform_choice(rng, OPEN_BOX_CASES)
    probability = 1.0 / float(len(OPEN_BOX_CASES))
    return case, {open_box_case_key(candidate): probability for candidate in OPEN_BOX_CASES}


def select_open_box_case_for_dimension(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    sampling_label: str,
    target_role: str,
) -> tuple[Tuple[int, int, int], Dict[str, float]]:
    """Select a corner-cut net by the requested base-dimension answer first."""

    explicit_case = params.get("open_box_case")
    explicit_fields = any(key in params for key in ("sheet_length_units", "sheet_width_units", "cut_size_units"))
    if explicit_case is not None or explicit_fields:
        return select_open_box_case(
            instance_seed=int(instance_seed),
            params=params,
            sampling_label=str(sampling_label),
        )

    answer, _answer_probabilities = _select_answer_from_support(
        OPEN_BOX_BASE_DIMENSION_VALUES,
        instance_seed=int(instance_seed),
        params=params,
        sampling_label=str(sampling_label),
    )
    rng = spawn_rng(int(instance_seed), f"{sampling_label}.open_box_case.{target_role}.{answer}")
    cut_size = int(uniform_choice(rng, _OPEN_BOX_CUT_VALUES))
    companion = int(uniform_choice(rng, _OPEN_BOX_COMPANION_DIMENSION_VALUES))
    if str(target_role) == "base_length":
        case = (int(answer + 2 * cut_size), int(companion + 2 * cut_size), int(cut_size))
    elif str(target_role) == "base_width":
        case = (int(companion + 2 * cut_size), int(answer + 2 * cut_size), int(cut_size))
    else:
        raise ValueError("target_role must be base_length or base_width")
    validate_open_box_case(case)
    return case, {open_box_case_key(case): 1.0}


def select_open_box_dimension_role(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    sampling_label: str,
) -> tuple[str, Dict[str, float]]:
    """Select or validate the requested resulting base dimension role."""

    explicit_role = params.get("target_dimension_role")
    if explicit_role is not None:
        role = str(explicit_role)
        if role not in OPEN_BOX_DIMENSION_ROLES:
            raise ValueError("target_dimension_role must be base_length or base_width")
        return role, probability_map_for_support(OPEN_BOX_DIMENSION_ROLES, selected=role)
    rng = spawn_rng(int(instance_seed), f"{sampling_label}.target_dimension_role")
    role = str(uniform_choice(rng, OPEN_BOX_DIMENSION_ROLES))
    return str(role), _probability_map(OPEN_BOX_DIMENSION_ROLES)


def validate_dimensions(case: Sequence[int]) -> None:
    """Validate explicit cuboid dimensions."""

    length, width, height = [int(value) for value in case]
    if min(length, width, height) <= 0:
        raise ValueError("cuboid dimensions must be positive integers")
    if max(length, width, height) > 60:
        raise ValueError("cuboid dimensions are too large for this renderer")


def validate_cube_edge(edge: int) -> None:
    """Validate explicit cube edge length."""

    if int(edge) not in CUBE_EDGE_VALUES:
        raise ValueError("edge_units must be an integer from 2 to 51")


def validate_open_box_case(case: Sequence[int]) -> None:
    """Validate explicit open-box sheet and cut dimensions."""

    sheet_length, sheet_width, cut_size, base_length, base_width, _volume = open_box_values_for_case(case)
    if min(sheet_length, sheet_width, cut_size) <= 0:
        raise ValueError("open-box sheet dimensions and cut size must be positive integers")
    if base_length < 3 or base_width < 3:
        raise ValueError("open-box resulting base dimensions must each be at least 3")
    if max(sheet_length, sheet_width) > 90:
        raise ValueError("open-box sheet dimensions are too large for this renderer")


__all__ = [
    "CUBE_EDGE_VALUES",
    "CUBOID_DIMENSION_CASES",
    "CUBOID_MISSING_DIMENSION_VALUES",
    "CUBOID_SURFACE_AREA_VALUES",
    "OPEN_BOX_CASES",
    "OPEN_BOX_BASE_DIMENSION_VALUES",
    "OPEN_BOX_DIMENSION_ROLES",
    "PARTIAL_FRAME_EDGE_COUNTS",
    "cuboid_case_key",
    "open_box_case_key",
    "open_box_values_for_case",
    "probability_map_for_support",
    "select_cube_edge",
    "select_cuboid_case",
    "select_cuboid_case_for_missing_dimension",
    "select_cuboid_case_for_surface_area",
    "select_open_box_case",
    "select_open_box_case_for_dimension",
    "select_open_box_dimension_role",
    "select_partial_frame_edge_count",
    "surface_area_for_case",
    "validate_cube_edge",
    "validate_dimensions",
    "validate_open_box_case",
]
