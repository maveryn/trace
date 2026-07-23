"""Solve algebraic angle expressions in a special quadrilateral."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.registry import register_task

from ._lifecycle import _run_expression_relation
from .shared.rendering import (
    RENDER_KITE_OPPOSITE_ANGLES,
    RENDER_PARALLELOGRAM_CONSECUTIVE_ANGLES,
    RENDER_PARALLELOGRAM_OPPOSITE_ANGLES,
    RENDER_RHOMBUS_HALF_ANGLE_EXPRESSION,
)
from .shared.state import DOMAIN, LinearExpression, QuadrilateralCase

TASK_ID = "task_geometry__special_quadrilateral__algebraic_angle_value"
SUPPORTED_QUERY_IDS: tuple[str, ...] = (
    "parallelogram_opposite_angle_expression",
    "parallelogram_consecutive_angle_expression",
    "rhombus_diagonal_half_angle_expression",
    "kite_opposite_angle_expression",
)
TASK_PROMPT_KEY = "algebraic_angle_value_query"
AngleCaseRecord = tuple[str, str, int, str, int, str]


def _expr(coefficient: int, constant: int) -> LinearExpression:
    return LinearExpression(int(coefficient), int(constant))


def _expression_for_value(value: int, *, x_value: int, index: int, salt: int) -> LinearExpression:
    """Build a compact linear expression that evaluates to one visible value."""

    coefficients = (2, 3, 4, 5, 1)
    for offset in range(len(coefficients)):
        coefficient = coefficients[(int(index) + int(salt) + offset) % len(coefficients)]
        constant = int(value) - int(coefficient) * int(x_value)
        if -9 <= int(constant) <= 99:
            return _expr(int(coefficient), int(constant))
    coefficient = 1
    return _expr(int(coefficient), int(value) - int(x_value))


def _expression_pair_for_values(
    *,
    target_value: int,
    support_value: int,
    index: int,
) -> tuple[LinearExpression, LinearExpression, int]:
    """Return two same-variable expressions for one algebraic angle case."""

    x_value = max(4, min(18, min(int(target_value), int(support_value)) // 5 + int(index) % 3))
    target_expression = _expression_for_value(
        int(target_value),
        x_value=int(x_value),
        index=int(index),
        salt=0,
    )
    support_expression = _expression_for_value(
        int(support_value),
        x_value=int(x_value),
        index=int(index),
        salt=2,
    )
    return target_expression, support_expression, int(x_value)


def _case(
    *,
    render_kind: str,
    shape_kind: str,
    answer: int,
    target_name: str,
    target_expression: LinearExpression,
    support_expression: LinearExpression,
    theorem: str,
    x_value: int,
) -> QuadrilateralCase:
    return QuadrilateralCase(
        render_kind=str(render_kind),
        shape_kind=str(shape_kind),
        answer=int(answer),
        target_name=str(target_name),
        target_label=target_expression.display(degree=True),
        support_label=support_expression.display(degree=True),
        theorem=str(theorem),
        x_value=int(x_value),
        target_expression=target_expression,
        support_expression=support_expression,
    )


def _angle_cases(*records: AngleCaseRecord) -> tuple[QuadrilateralCase, ...]:
    cases: list[QuadrilateralCase] = []
    for index, record in enumerate(records):
        render_kind, shape_kind, answer, target_name, support_value, theorem = record
        target_expression, support_expression, x_value = _expression_pair_for_values(
            target_value=int(answer),
            support_value=int(support_value),
            index=int(index),
        )
        cases.append(
            _case(
                render_kind=str(render_kind),
                shape_kind=str(shape_kind),
                answer=int(answer),
                target_name=str(target_name),
                target_expression=target_expression,
                support_expression=support_expression,
                theorem=str(theorem),
                x_value=int(x_value),
            )
        )
    return tuple(cases)


def _equal_angle_cases(
    *,
    render_kind: str,
    shape_kind: str,
    answers: range,
    target_name: str,
    theorem: str,
) -> tuple[QuadrilateralCase, ...]:
    return _angle_cases(
        *(
            (str(render_kind), str(shape_kind), int(answer), str(target_name), int(answer), str(theorem))
            for answer in answers
        )
    )


def _supplementary_angle_cases(
    *,
    answers: range,
) -> tuple[QuadrilateralCase, ...]:
    return _angle_cases(
        *(
            (
                RENDER_PARALLELOGRAM_CONSECUTIVE_ANGLES,
                "parallelogram",
                int(answer),
                "angle ABC",
                180 - int(answer),
                "consecutive_angles_of_a_parallelogram_are_supplementary",
            )
            for answer in answers
        )
    )


_CASES_BY_BRANCH: dict[str, tuple[QuadrilateralCase, ...]] = {
    "parallelogram_opposite_angle_expression": _equal_angle_cases(
        render_kind=RENDER_PARALLELOGRAM_OPPOSITE_ANGLES,
        shape_kind="parallelogram",
        answers=range(36, 51),
        target_name="angle BCD",
        theorem="opposite_angles_of_a_parallelogram_are_equal",
    ),
    "parallelogram_consecutive_angle_expression": _supplementary_angle_cases(answers=range(96, 111)),
    "rhombus_diagonal_half_angle_expression": _equal_angle_cases(
        render_kind=RENDER_RHOMBUS_HALF_ANGLE_EXPRESSION,
        shape_kind="rhombus",
        answers=range(21, 36),
        target_name="angle ABO",
        theorem="rhombus_diagonal_bisects_vertex_angle",
    ),
    "kite_opposite_angle_expression": _equal_angle_cases(
        render_kind=RENDER_KITE_OPPOSITE_ANGLES,
        shape_kind="kite",
        answers=range(66, 81),
        target_name="angle ABC",
        theorem="opposite_non_vertex_angles_of_this_kite_are_equal",
    ),
}


def _validate_angle_case_table(case_table: Mapping[str, tuple[QuadrilateralCase, ...]]) -> None:
    """Fail fast if an algebraic angle case does not bind to its theorem."""

    for branch_name, branch_cases in case_table.items():
        if not branch_cases:
            raise ValueError(f"special quadrilateral angle branch has no cases: {branch_name}")
        for case in branch_cases:
            if case.target_expression is None or case.support_expression is None or case.x_value is None:
                raise ValueError(f"angle case is missing algebraic angle expressions: {branch_name}")
            x_value = int(case.x_value)
            target_value = int(case.target_expression.evaluate(x_value))
            support_value = int(case.support_expression.evaluate(x_value))
            if target_value != int(case.answer):
                raise ValueError(f"angle case target expression does not evaluate to the answer: {branch_name}")
            if str(branch_name) == "parallelogram_consecutive_angle_expression":
                if target_value + support_value != 180:
                    raise ValueError(f"consecutive-angle case is not supplementary: {branch_name}")
            elif support_value != int(case.answer):
                raise ValueError(f"angle case support expression does not match the answer: {branch_name}")
            if not 0 < int(case.answer) < 180:
                raise ValueError(f"angle answer must be between 0 and 180 degrees: {branch_name}")


_validate_angle_case_table(_CASES_BY_BRANCH)


def _build_algebraic_cases() -> Mapping[str, tuple[QuadrilateralCase, ...]]:
    """Return the task-owned theorem cases for algebraic angle solving."""

    return _CASES_BY_BRANCH


@register_task
class GeometrySpecialQuadrilateralAlgebraicAngleValueTask:
    """Solve algebraic angle expressions in a special quadrilateral."""

    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    domain = DOMAIN
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one algebraic-angle instance with task-owned binding."""

        return _run_expression_relation(
            task_id=TASK_ID,
            supported_queries=SUPPORTED_QUERY_IDS,
            cases_by_branch=_build_algebraic_cases(),
            task_prompt_key=TASK_PROMPT_KEY,
            witness_type="special_quadrilateral_algebraic_angle_relation",
            unsupported_query_subject="algebraic-angle",
            instance_seed=int(instance_seed),
            params=dict(params),
            max_attempts=int(max_attempts),
        )


__all__ = ["GeometrySpecialQuadrilateralAlgebraicAngleValueTask", "SUPPORTED_QUERY_IDS", "TASK_ID"]
