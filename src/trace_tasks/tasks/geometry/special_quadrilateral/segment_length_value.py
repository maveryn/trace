"""Solve algebraic side or diagonal segment lengths in a special quadrilateral."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.registry import register_task

from ._lifecycle import _run_expression_relation
from .shared.rendering import (
    RENDER_KITE_ADJACENT_SIDES,
    RENDER_PARALLELOGRAM_BISECTED_DIAGONAL,
    RENDER_PARALLELOGRAM_OPPOSITE_SIDES,
    RENDER_RHOMBUS_ALL_SIDES,
)
from .shared.state import DOMAIN, LinearExpression, QuadrilateralCase

TASK_ID = "task_geometry__special_quadrilateral__segment_length_value"
SUPPORTED_QUERY_IDS: tuple[str, ...] = (
    "parallelogram_opposite_side_expression",
    "rhombus_all_sides_expression",
    "kite_adjacent_equal_side_expression",
    "parallelogram_diagonal_bisection_expression",
)
TASK_PROMPT_KEY = "segment_length_value_query"
SegmentCaseRecord = tuple[str, str, int, str, tuple[int, int], tuple[int, int], str, int]


def _segment_expression(raw: tuple[int, int]) -> LinearExpression:
    coefficient, constant = raw
    return LinearExpression(int(coefficient), int(constant))


def _segment_expression_for_value(value: int, *, x_value: int, index: int, salt: int) -> tuple[int, int]:
    """Build a compact linear expression tuple that evaluates to one length."""

    coefficients = (2, 3, 4, 5, 1)
    for offset in range(len(coefficients)):
        coefficient = coefficients[(int(index) + int(salt) + offset) % len(coefficients)]
        constant = int(value) - int(coefficient) * int(x_value)
        if -9 <= int(constant) <= 99:
            return int(coefficient), int(constant)
    return 1, int(value) - int(x_value)


def _segment_record(
    *,
    render_kind: str,
    shape_kind: str,
    answer: int,
    target_name: str,
    theorem: str,
    index: int,
) -> SegmentCaseRecord:
    x_value = max(4, min(18, int(answer) // 5 + int(index) % 3))
    target_raw = _segment_expression_for_value(
        int(answer),
        x_value=int(x_value),
        index=int(index),
        salt=0,
    )
    support_raw = _segment_expression_for_value(
        int(answer),
        x_value=int(x_value),
        index=int(index),
        salt=2,
    )
    return (
        str(render_kind),
        str(shape_kind),
        int(answer),
        str(target_name),
        target_raw,
        support_raw,
        str(theorem),
        int(x_value),
    )


def _segment_records(
    *,
    render_kind: str,
    shape_kind: str,
    answers: range,
    target_name: str,
    theorem: str,
) -> tuple[SegmentCaseRecord, ...]:
    return tuple(
        _segment_record(
            render_kind=str(render_kind),
            shape_kind=str(shape_kind),
            answer=int(answer),
            target_name=str(target_name),
            theorem=str(theorem),
            index=int(index),
        )
        for index, answer in enumerate(answers)
    )


def _segment_case_from_record(record: SegmentCaseRecord) -> QuadrilateralCase:
    render_kind, shape_kind, answer, target_name, target_raw, support_raw, theorem, x_value = record
    target_expression = _segment_expression(target_raw)
    support_expression = _segment_expression(support_raw)
    return QuadrilateralCase(
        render_kind=str(render_kind),
        shape_kind=str(shape_kind),
        answer=int(answer),
        target_name=str(target_name),
        target_label=target_expression.display(),
        support_label=support_expression.display(),
        theorem=str(theorem),
        x_value=int(x_value),
        target_expression=target_expression,
        support_expression=support_expression,
    )


def _segment_cases(*records: SegmentCaseRecord) -> tuple[QuadrilateralCase, ...]:
    return tuple(_segment_case_from_record(record) for record in records)


_CASES_BY_BRANCH: dict[str, tuple[QuadrilateralCase, ...]] = {
    "parallelogram_opposite_side_expression": _segment_cases(
        *_segment_records(
            render_kind=RENDER_PARALLELOGRAM_OPPOSITE_SIDES,
            shape_kind="parallelogram",
            answers=range(16, 31),
            target_name="segment BC",
            theorem="opposite_sides_of_a_parallelogram_are_equal",
        ),
    ),
    "rhombus_all_sides_expression": _segment_cases(
        *_segment_records(
            render_kind=RENDER_RHOMBUS_ALL_SIDES,
            shape_kind="rhombus",
            answers=range(31, 46),
            target_name="segment CD",
            theorem="all_sides_of_a_rhombus_are_equal",
        ),
    ),
    "kite_adjacent_equal_side_expression": _segment_cases(
        *_segment_records(
            render_kind=RENDER_KITE_ADJACENT_SIDES,
            shape_kind="kite",
            answers=range(46, 61),
            target_name="segment AD",
            theorem="adjacent_marked_sides_of_a_kite_are_equal",
        ),
    ),
    "parallelogram_diagonal_bisection_expression": _segment_cases(
        *_segment_records(
            render_kind=RENDER_PARALLELOGRAM_BISECTED_DIAGONAL,
            shape_kind="parallelogram",
            answers=range(61, 76),
            target_name="segment OC",
            theorem="diagonals_of_a_parallelogram_bisect_each_other",
        ),
    ),
}


def _validate_segment_case_table(case_table: Mapping[str, tuple[QuadrilateralCase, ...]]) -> None:
    """Fail fast if a segment-length equation case does not bind to its answer."""

    for branch_name, branch_cases in case_table.items():
        if not branch_cases:
            raise ValueError(f"special quadrilateral segment branch has no cases: {branch_name}")
        for case in branch_cases:
            if case.target_expression is None or case.support_expression is None or case.x_value is None:
                raise ValueError(f"segment case is missing algebraic length expressions: {branch_name}")
            x_value = int(case.x_value)
            target_value = int(case.target_expression.evaluate(x_value))
            support_value = int(case.support_expression.evaluate(x_value))
            if target_value != int(case.answer) or support_value != int(case.answer):
                raise ValueError(f"segment case does not evaluate to the bound answer: {branch_name}")
            if int(case.answer) <= 0:
                raise ValueError(f"segment length answer must be positive: {branch_name}")


_validate_segment_case_table(_CASES_BY_BRANCH)


def _build_segment_cases() -> Mapping[str, tuple[QuadrilateralCase, ...]]:
    """Return the task-owned theorem cases for algebraic segment solving."""

    return _CASES_BY_BRANCH


@register_task
class GeometrySpecialQuadrilateralSegmentLengthValueTask:
    """Solve algebraic side or diagonal segment lengths in a special quadrilateral."""

    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    domain = DOMAIN
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one segment-length instance with task-owned binding."""

        return _run_expression_relation(
            task_id=TASK_ID,
            supported_queries=SUPPORTED_QUERY_IDS,
            cases_by_branch=_build_segment_cases(),
            task_prompt_key=TASK_PROMPT_KEY,
            witness_type="special_quadrilateral_segment_length_relation",
            unsupported_query_subject="segment-length",
            instance_seed=int(instance_seed),
            params=dict(params),
            max_attempts=int(max_attempts),
        )


__all__ = ["GeometrySpecialQuadrilateralSegmentLengthValueTask", "SUPPORTED_QUERY_IDS", "TASK_ID"]
