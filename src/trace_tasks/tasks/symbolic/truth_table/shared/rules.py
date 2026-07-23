"""Boolean expression rules for symbolic truth tables."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .state import TRUTH_VARIABLES, TruthExpressionSpec, TruthRowSpec


def truth_rows() -> tuple[TruthRowSpec, ...]:
    """Return the canonical 3-variable truth-table row order."""

    rows: list[TruthRowSpec] = []
    for index in range(8):
        values = {
            "A": 1 if index & 4 else 0,
            "B": 1 if index & 2 else 0,
            "C": 1 if index & 1 else 0,
        }
        rows.append(
            TruthRowSpec(
                row_index=int(index),
                row_label=str(index + 1),
                values=dict(values),
            )
        )
    return tuple(rows)


def _var(name: str) -> tuple[str, str]:
    return ("var", str(name))


def _not(value: Any) -> tuple[str, Any]:
    return ("not", value)


def _and(left: Any, right: Any) -> tuple[str, Any, Any]:
    return ("and", left, right)


def _or(left: Any, right: Any) -> tuple[str, Any, Any]:
    return ("or", left, right)


def _xor(left: Any, right: Any) -> tuple[str, Any, Any]:
    return ("xor", left, right)


def evaluate_expression_ast(ast: Any, assignment: Mapping[str, int]) -> int:
    """Evaluate one expression AST under a 0/1 assignment."""

    if not isinstance(ast, tuple) or not ast:
        raise ValueError("invalid truth-table expression AST")
    op = str(ast[0])
    if op == "var":
        variable = str(ast[1])
        if variable not in TRUTH_VARIABLES:
            raise ValueError(f"unsupported truth-table variable: {variable!r}")
        return 1 if int(assignment[variable]) else 0
    if op == "not":
        return 0 if evaluate_expression_ast(ast[1], assignment) else 1
    if op in {"and", "or", "xor"}:
        left = evaluate_expression_ast(ast[1], assignment)
        right = evaluate_expression_ast(ast[2], assignment)
        if op == "and":
            return 1 if left and right else 0
        if op == "or":
            return 1 if left or right else 0
        return 1 if int(left) != int(right) else 0
    raise ValueError(f"unsupported truth-table operator: {op!r}")


def truth_pattern(ast: Any, rows: Sequence[TruthRowSpec] | None = None) -> tuple[int, ...]:
    """Return one output bit per canonical truth-table row."""

    active_rows = tuple(rows if rows is not None else truth_rows())
    return tuple(evaluate_expression_ast(ast, row.values) for row in active_rows)


def _spec(expression_id: str, display: str, ast: Any) -> TruthExpressionSpec:
    return TruthExpressionSpec(
        expression_id=str(expression_id),
        display=str(display),
        ast=ast,
        pattern=truth_pattern(ast),
    )


def all_expression_specs() -> tuple[TruthExpressionSpec, ...]:
    """Return a curated expression bank with useful equivalence groups."""

    a, b, c = _var("A"), _var("B"), _var("C")
    return (
        _spec("a", "A", a),
        _spec("b", "B", b),
        _spec("c", "C", c),
        _spec("not_a", "!A", _not(a)),
        _spec("not_b", "!B", _not(b)),
        _spec("not_c", "!C", _not(c)),
        _spec("a_and_b", "A&B", _and(a, b)),
        _spec("b_and_a", "B&A", _and(b, a)),
        _spec("a_or_b", "A|B", _or(a, b)),
        _spec("b_or_a", "B|A", _or(b, a)),
        _spec("a_xor_b", "A^B", _xor(a, b)),
        _spec("b_xor_a", "B^A", _xor(b, a)),
        _spec("a_and_c", "A&C", _and(a, c)),
        _spec("c_and_a", "C&A", _and(c, a)),
        _spec("b_and_c", "B&C", _and(b, c)),
        _spec("c_and_b", "C&B", _and(c, b)),
        _spec("a_or_c", "A|C", _or(a, c)),
        _spec("c_or_a", "C|A", _or(c, a)),
        _spec("b_or_c", "B|C", _or(b, c)),
        _spec("c_or_b", "C|B", _or(c, b)),
        _spec("a_and_not_b", "A&!B", _and(a, _not(b))),
        _spec("not_b_and_a", "!B&A", _and(_not(b), a)),
        _spec("not_a_and_b", "!A&B", _and(_not(a), b)),
        _spec("b_and_not_a", "B&!A", _and(b, _not(a))),
        _spec("a_or_not_b", "A|!B", _or(a, _not(b))),
        _spec("not_b_or_a", "!B|A", _or(_not(b), a)),
        _spec("not_a_or_b", "!A|B", _or(_not(a), b)),
        _spec("b_or_not_a", "B|!A", _or(b, _not(a))),
        _spec("a_and_b_or_c", "(A&B)|C", _or(_and(a, b), c)),
        _spec("c_or_a_and_b", "C|(A&B)", _or(c, _and(a, b))),
        _spec("a_and_b_or_c_group", "A&(B|C)", _and(a, _or(b, c))),
        _spec("a_and_b_or_a_and_c", "(A&B)|(A&C)", _or(_and(a, b), _and(a, c))),
        _spec("a_or_b_and_c", "A|(B&C)", _or(a, _and(b, c))),
        _spec("a_or_b_and_a_or_c", "(A|B)&(A|C)", _and(_or(a, b), _or(a, c))),
        _spec("not_a_and_b_group", "!(A&B)", _not(_and(a, b))),
        _spec("not_a_or_not_b", "!A|!B", _or(_not(a), _not(b))),
        _spec("not_a_or_b_group", "!(A|B)", _not(_or(a, b))),
        _spec("not_a_and_not_b", "!A&!B", _and(_not(a), _not(b))),
        _spec("xor_ab_and_c", "(A^B)&C", _and(_xor(a, b), c)),
        _spec("c_and_xor_ab", "C&(A^B)", _and(c, _xor(a, b))),
        _spec("xor_ab_or_c", "(A^B)|C", _or(_xor(a, b), c)),
        _spec("c_or_xor_ab", "C|(A^B)", _or(c, _xor(a, b))),
        _spec("xor_abc_left", "(A^B)^C", _xor(_xor(a, b), c)),
        _spec("xor_abc_right", "A^(B^C)", _xor(a, _xor(b, c))),
        _spec("a_and_b_and_c", "A&B&C", _and(_and(a, b), c)),
        _spec("c_and_a_and_b", "C&(A&B)", _and(c, _and(a, b))),
        _spec("a_or_b_or_c", "A|B|C", _or(_or(a, b), c)),
        _spec("c_or_a_or_b", "C|(A|B)", _or(c, _or(a, b))),
        _spec("not_a_and_not_b_and_not_c", "!A&!B&!C", _and(_and(_not(a), _not(b)), _not(c))),
        _spec("not_a_or_b_or_c_group", "!(A|B|C)", _not(_or(_or(a, b), c))),
        _spec("not_a_and_b_and_c_group", "!(A&B&C)", _not(_and(_and(a, b), c))),
    )


def expression_by_id(expression_id: str) -> TruthExpressionSpec:
    """Look up one curated expression by stable id."""

    mapping = {spec.expression_id: spec for spec in all_expression_specs()}
    key = str(expression_id)
    if key not in mapping:
        raise ValueError(f"unknown truth-table expression id: {expression_id!r}")
    return mapping[key]


def expressions_by_true_count(count: int) -> tuple[TruthExpressionSpec, ...]:
    """Return expressions with the requested number of true rows."""

    return tuple(spec for spec in all_expression_specs() if int(spec.true_count) == int(count))


def equivalent_expression_groups() -> tuple[tuple[TruthExpressionSpec, ...], ...]:
    """Group curated expressions by identical truth pattern."""

    groups: dict[tuple[int, ...], list[TruthExpressionSpec]] = {}
    for spec in all_expression_specs():
        groups.setdefault(tuple(spec.pattern), []).append(spec)
    return tuple(tuple(group) for group in groups.values() if len(group) >= 2)


def distinct_truth_patterns() -> tuple[str, ...]:
    """Return distinct output patterns from the curated expression bank."""

    return tuple(sorted({spec.pattern_string for spec in all_expression_specs()}))


def row_truth_cells(expression: TruthExpressionSpec) -> tuple[tuple[str, int], ...]:
    """Return stable output-cell ids and values for one expression column."""

    return tuple(
        (f"cell_out_{row.row_label}", int(value))
        for row, value in zip(truth_rows(), expression.pattern)
    )


__all__ = [
    "all_expression_specs",
    "distinct_truth_patterns",
    "equivalent_expression_groups",
    "evaluate_expression_ast",
    "expression_by_id",
    "expressions_by_true_count",
    "row_truth_cells",
    "truth_pattern",
    "truth_rows",
]
