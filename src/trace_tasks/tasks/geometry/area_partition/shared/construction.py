"""Diagram construction inputs for the area-partition scene."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from trace_tasks.core.sampling import integer_range_choice, uniform_choice
from trace_tasks.core.seed import spawn_rng

from .relations import selected_probability_map, total_area_from_unit_partition
from .state import AreaPartitionProblem, PartitionCase


PARALLELOGRAM_PARTITION_CASES: tuple[PartitionCase, ...] = (
    ("parallelogram_diagonals_midpoint_eighth", 19, 8),
    ("parallelogram_diagonals_midpoint_eighth", 23, 8),
    ("parallelogram_diagonals_midpoint_eighth", 31, 8),
    ("parallelogram_diagonals_midpoint_eighth", 43, 8),
    ("parallelogram_diagonals_quarter", 37, 4),
    ("parallelogram_diagonals_quarter", 47, 4),
    ("parallelogram_diagonals_quarter", 53, 4),
    ("parallelogram_diagonals_quarter", 61, 4),
    ("parallelogram_diagonals_quarter", 71, 4),
    ("parallelogram_diagonals_quarter", 79, 4),
)

TRIANGLE_PARTITION_CASES: tuple[PartitionCase, ...] = (
    ("triangle_median_half", 68, 2),
    ("triangle_median_half", 84, 2),
    ("triangle_midsegment_quarter", 37, 4),
    ("triangle_midsegment_quarter", 49, 4),
    ("triangle_midsegment_quarter", 62, 4),
    ("triangle_medians_sixth", 23, 6),
    ("triangle_medians_sixth", 31, 6),
    ("triangle_medians_sixth", 43, 6),
    ("triangle_medians_sixth", 57, 6),
)

AREA_PARTITION_CASES: tuple[PartitionCase, ...] = (
    *PARALLELOGRAM_PARTITION_CASES,
    *TRIANGLE_PARTITION_CASES,
)

SHADED_AREA_RANGE_BY_DENOMINATOR: dict[int, tuple[int, int]] = {
    2: (60, 172),
    4: (30, 86),
    6: (20, 57),
    8: (15, 43),
}


def _answer_support_values(partition_cases: Sequence[PartitionCase]) -> tuple[int, ...]:
    """Return all feasible total-area answers from the denominator-aware ranges."""

    values: set[int] = set()
    for _scene_variant, _shaded_area, denominator in partition_cases:
        low, high = SHADED_AREA_RANGE_BY_DENOMINATOR.get(int(denominator), (0, -1))
        values.update(int(shaded_area) * int(denominator) for shaded_area in range(int(low), int(high) + 1))
    return tuple(sorted(values))


def _sample_shaded_area(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    denominator: int,
    namespace: str,
) -> int:
    """Sample a deterministic shaded area for the selected partition denominator."""

    explicit = params.get("shaded_area")
    if explicit is not None:
        return int(explicit)
    if int(denominator) not in SHADED_AREA_RANGE_BY_DENOMINATOR:
        raise ValueError(f"unsupported area partition denominator: {denominator}")
    low, high = SHADED_AREA_RANGE_BY_DENOMINATOR[int(denominator)]
    rng = spawn_rng(int(instance_seed), str(namespace))
    value, _probabilities = integer_range_choice(rng, int(low), int(high))
    return int(value)


def resolve_area_partition_problem(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    partition_cases: Sequence[PartitionCase] = AREA_PARTITION_CASES,
    sampling_namespace: str,
) -> AreaPartitionProblem:
    """Resolve one deterministic total-area partition problem."""

    cases = tuple(partition_cases)
    if not cases:
        raise ValueError("area-partition case support must be non-empty")

    rng = spawn_rng(int(instance_seed), str(sampling_namespace))
    scene_variant, _case_shaded_area, denominator = uniform_choice(rng, cases)
    scene_variant = str(params.get("scene_variant", scene_variant))
    denominator = int(params.get("area_denominator", denominator))
    shaded_area = _sample_shaded_area(
        instance_seed=int(instance_seed),
        params=params,
        denominator=int(denominator),
        namespace=f"{sampling_namespace}.shaded_area",
    )

    allowed_variants = {str(case[0]) for case in cases}
    if scene_variant not in allowed_variants:
        raise ValueError(f"unsupported area partition scene_variant: {scene_variant}")

    answer = total_area_from_unit_partition(
        shaded_area=int(shaded_area),
        denominator=int(denominator),
    )
    support_values = tuple(sorted({*_answer_support_values(cases), int(answer)}))
    return AreaPartitionProblem(
        scene_variant=str(scene_variant),
        answer=float(answer),
        shaded_area=int(shaded_area),
        denominator=int(denominator),
        formula=f"total area = shaded area * {int(denominator)}",
        support_probabilities=selected_probability_map(support_values, float(answer)),
    )


__all__ = [
    "AREA_PARTITION_CASES",
    "PARALLELOGRAM_PARTITION_CASES",
    "SHADED_AREA_RANGE_BY_DENOMINATOR",
    "TRIANGLE_PARTITION_CASES",
    "resolve_area_partition_problem",
]
