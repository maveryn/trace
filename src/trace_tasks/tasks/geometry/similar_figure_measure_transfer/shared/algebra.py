"""Expression-labeled similar-figure case builders."""

from __future__ import annotations

from .state import SimilarEquationCase


def variable_case(
    *,
    construction_family: str,
    shape_kind: str,
    answer: int,
    scale_factor: int,
    coefficient: int,
    constant: int,
    support_source: int,
) -> SimilarEquationCase:
    """Build a case where the answer is the variable value."""

    source_value = int(coefficient) * int(answer) + int(constant)
    target_value = int(scale_factor) * int(source_value)
    if str(construction_family) == "two_expression_ratio":
        target_coefficient = max(1, int(scale_factor) - 1)
        target_constant = int(target_value) - int(target_coefficient) * int(answer)
        target_label = _linear_expression(target_coefficient, target_constant)
    else:
        target_label = str(target_value)
    return SimilarEquationCase(
        construction_family=str(construction_family),
        shape_kind=str(shape_kind),
        answer=float(answer),
        target_name="x",
        variable_name="x",
        relation="similar_figure_marked_side_equation",
        source_target_label=_linear_expression(int(coefficient), int(constant)),
        target_target_label=target_label,
        support_source_label=str(int(support_source)),
        support_target_label=str(int(support_source) * int(scale_factor)),
        scale_factor=float(scale_factor),
        source_target_value=float(source_value),
        target_target_value=float(target_value),
        annotation_labels=_side_annotation_labels(str(shape_kind)),
    )


def _side_annotation_labels(shape_kind: str) -> tuple[str, ...]:
    if str(shape_kind) == "triangle":
        return ("A", "B", "B", "C", "A'", "B'", "B'", "C'")
    return ("A", "B", "C", "D", "A'", "B'", "C'", "D'")


def _linear_expression(coefficient: int, constant: int) -> str:
    if int(coefficient) == 1:
        head = "x"
    elif int(coefficient) == -1:
        head = "-x"
    else:
        head = f"{int(coefficient)}x"
    if int(constant) == 0:
        return head
    if int(constant) > 0:
        return f"{head}+{int(constant)}"
    return f"{head}{int(constant)}"
