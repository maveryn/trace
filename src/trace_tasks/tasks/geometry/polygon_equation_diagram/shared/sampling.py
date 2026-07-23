"""Identity-free sampling primitives for polygon equation diagrams."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.core.seed import spawn_rng

from .algebra import format_angle_expression, format_degrees, format_linear_expression, side_name
from .state import polygon_kind


def _side_count(instance_seed: int, params: Mapping[str, Any], namespace: str) -> int:
    if "side_count" in params:
        value = int(params["side_count"])
        if value not in {3, 4, 5, 6}:
            raise ValueError("side_count must be one of 3, 4, 5, 6")
        return value
    rng = spawn_rng(int(instance_seed), namespace)
    return int(rng.choice((3, 4, 5, 6)))


def _equation_pair_for_value(*, rng: Any, value: int, variable_value: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return two distinct linear expressions that evaluate to the same value."""

    coefficients = (1, 2, 3, 4)
    for _ in range(200):
        first = int(rng.choice(coefficients))
        second = int(rng.choice(coefficients))
        if first == second:
            continue
        return (
            (first, int(value) - (first * int(variable_value))),
            (second, int(value) - (second * int(variable_value))),
        )
    raise ValueError("failed to sample distinct linear expressions")


def _expression_for_value(*, rng: Any, value: int, variable_name: str, variable_value: int) -> tuple[int, int]:
    """Return one linear expression that evaluates to `value` for the hidden variable value."""

    coefficient = int(rng.choice((1, 2, 3, 4)))
    return coefficient, int(value) - (coefficient * int(variable_value))


def _unconstrained_expression(*, rng: Any) -> tuple[int, int]:
    """Return a plausible linear expression for a distractor variable."""

    coefficient = int(rng.choice((1, 2, 3, 4)))
    offset = int(rng.randint(-28, 96))
    if offset == 0:
        offset = int(rng.choice((-12, 12, 18)))
    return coefficient, offset


def _choose_distractor_count(*, rng: Any, available_count: int) -> int:
    if int(available_count) <= 0:
        return 0
    if int(available_count) == 1:
        return 1
    return int(rng.choice((1, 2)))


def _use_distractors(*, rng: Any, include_distractors: bool) -> bool:
    return bool(include_distractors) and bool(rng.randrange(2))


def _sample_distractor_value(*, rng: Any, lower: int, upper: int, forbidden: set[int]) -> int:
    for _ in range(300):
        value = int(rng.randint(int(lower), int(upper)))
        if value not in forbidden:
            return value
    raise ValueError("failed to sample distractor value")


def _format_side_measure(value: int) -> str:
    return str(int(value))


def _side_names(labels: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(side_name(labels, index) for index in range(len(labels)))


def _sample_side_expression_label(
    *,
    rng: Any,
    side_value: int,
    variable_name: str,
    variable_value: int,
    force_expression: bool = False,
) -> str:
    if bool(force_expression) or bool(rng.randrange(2)):
        expression = _expression_for_value(
            rng=rng,
            value=int(side_value),
            variable_name=str(variable_name),
            variable_value=int(variable_value),
        )
        return format_linear_expression(expression[0], variable_name, expression[1])
    return _format_side_measure(int(side_value))


def _sample_values_for_sum(*, rng: Any, count: int, total: int, min_value: int, max_value: int) -> tuple[int, ...]:
    """Sample integer angle values whose sum is exactly total."""

    for _ in range(8000):
        values = [int(rng.randint(int(min_value), int(max_value))) for _ in range(int(count) - 1)]
        last = int(total) - sum(values)
        if int(min_value) <= last <= int(max_value):
            values.append(last)
            rng.shuffle(values)
            return tuple(int(value) for value in values)
    raise ValueError("failed to sample angle values for requested sum")


def _polygon_angles(side_count: int, *, instance_seed: int, params: Mapping[str, Any], namespace: str) -> tuple[int, ...]:
    rng = spawn_rng(int(instance_seed), namespace)
    min_angle = int(params.get("angle_min", 38))
    max_angle = int(params.get("angle_max", 152))
    return _sample_values_for_sum(
        rng=rng,
        count=int(side_count),
        total=(int(side_count) - 2) * 180,
        min_value=int(min_angle),
        max_value=int(max_angle),
    )


def _target_indices(side_count: int, *, instance_seed: int, namespace: str) -> tuple[int, int]:
    rng = spawn_rng(int(instance_seed), namespace)
    first = int(rng.randrange(int(side_count)))
    gap = int(rng.randrange(1, int(side_count)))
    return first, (first + gap) % int(side_count)


def sample_equal_side_relation(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    namespace: str,
    include_distractors: bool = False,
) -> dict[str, Any]:
    """Sample a marked equal-side algebraic relation without binding an objective answer."""

    side_count = _side_count(int(instance_seed), params, f"{namespace}.side_count")
    labels = tuple(chr(ord("A") + index) for index in range(side_count))
    rng = spawn_rng(int(instance_seed), namespace)
    variable_name = str(params.get("variable_name", "x"))
    variable_value = int(rng.randint(3, 31))
    side_value = int(rng.randint(12, 96))
    side_a, side_b = _target_indices(side_count, instance_seed=int(instance_seed), namespace=f"{namespace}.sides")
    first_expr, second_expr = _equation_pair_for_value(
        rng=rng,
        value=side_value,
        variable_value=variable_value,
    )
    first_side = side_name(labels, side_a)
    second_side = side_name(labels, side_b)
    side_labels = {
        first_side: format_linear_expression(first_expr[0], variable_name, first_expr[1]),
        second_side: format_linear_expression(second_expr[0], variable_name, second_expr[1]),
    }
    side_mark_counts = {
        str(first_side): 2,
        str(second_side): 2,
    }
    side_distractors: list[dict[str, Any]] = []
    if bool(include_distractors):
        available_sides = [
            side_name(labels, index)
            for index in range(side_count)
            if side_name(labels, index) not in {first_side, second_side}
        ]
        rng.shuffle(available_sides)
        mark_counts = [1, 3]
        rng.shuffle(mark_counts)
        for side_label in available_sides[: _choose_distractor_count(rng=rng, available_count=len(available_sides))]:
            mark_count = int(mark_counts.pop(0))
            distractor_variable = str(rng.choice(("x", "y", "z")))
            if distractor_variable == variable_name:
                numeric_value = _sample_distractor_value(
                    rng=rng,
                    lower=8,
                    upper=112,
                    forbidden={int(side_value)},
                )
                expression = _expression_for_value(
                    rng=rng,
                    value=int(numeric_value),
                    variable_name=distractor_variable,
                    variable_value=int(variable_value),
                )
            else:
                numeric_value = None
                expression = _unconstrained_expression(rng=rng)
            side_labels[str(side_label)] = format_linear_expression(
                expression[0],
                distractor_variable,
                expression[1],
            )
            side_mark_counts[str(side_label)] = int(mark_count)
            side_distractors.append(
                {
                    "side": str(side_label),
                    "mark_count": int(mark_count),
                    "variable_name": str(distractor_variable),
                    "label": str(side_labels[str(side_label)]),
                    "numeric_value_under_x": numeric_value,
                }
            )
    return {
        "side_count": int(side_count),
        "variable_name": str(variable_name),
        "variable_value": int(variable_value),
        "side_value": int(side_value),
        "target_side": str(first_side),
        "side_labels": dict(side_labels),
        "side_mark_counts": dict(side_mark_counts),
        "equal_sides": (str(first_side), str(second_side)),
        "side_distractors": list(side_distractors),
        "witness": {
            "polygon_kind": polygon_kind(side_count),
            "variable_value": int(variable_value),
            "equal_side_length": int(side_value),
            "equation": f"{side_labels[first_side]} = {side_labels[second_side]}",
            "side_mark_counts": dict(side_mark_counts),
            "side_distractors": list(side_distractors),
        },
    }


def sample_equal_angle_relation(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    namespace: str,
    include_distractors: bool = False,
) -> dict[str, Any]:
    """Sample a marked equal-angle algebraic relation without binding an objective answer."""

    side_count = _side_count(int(instance_seed), params, f"{namespace}.side_count")
    labels = tuple(chr(ord("A") + index) for index in range(side_count))
    rng = spawn_rng(int(instance_seed), namespace)
    variable_name = str(params.get("variable_name", "x"))
    variable_value = int(rng.randint(4, 41))
    angle_value = int(rng.randint(42, 132))
    first_index, second_index = _target_indices(
        side_count,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.angles",
    )
    first_expr, second_expr = _equation_pair_for_value(
        rng=rng,
        value=angle_value,
        variable_value=variable_value,
    )
    first_angle = labels[first_index]
    second_angle = labels[second_index]
    angle_labels = {
        first_angle: format_angle_expression(first_expr[0], variable_name, first_expr[1]),
        second_angle: format_angle_expression(second_expr[0], variable_name, second_expr[1]),
    }
    angle_mark_counts = {
        str(first_angle): 2,
        str(second_angle): 2,
    }
    distractor_mode = _use_distractors(rng=rng, include_distractors=include_distractors)
    angle_distractors: list[dict[str, Any]] = []
    if bool(distractor_mode):
        available_angles = [str(label) for label in labels if str(label) not in {str(first_angle), str(second_angle)}]
        rng.shuffle(available_angles)
        mark_counts = [1, 3]
        rng.shuffle(mark_counts)
        for vertex_label in available_angles[: _choose_distractor_count(rng=rng, available_count=len(available_angles))]:
            mark_count = int(mark_counts.pop(0))
            distractor_variable = str(rng.choice(("x", "y", "z")))
            if distractor_variable == variable_name:
                numeric_value = _sample_distractor_value(
                    rng=rng,
                    lower=30,
                    upper=150,
                    forbidden={int(angle_value)},
                )
                expression = _expression_for_value(
                    rng=rng,
                    value=int(numeric_value),
                    variable_name=distractor_variable,
                    variable_value=int(variable_value),
                )
            else:
                numeric_value = None
                expression = _unconstrained_expression(rng=rng)
            angle_labels[str(vertex_label)] = format_angle_expression(
                expression[0],
                distractor_variable,
                expression[1],
            )
            angle_mark_counts[str(vertex_label)] = int(mark_count)
            angle_distractors.append(
                {
                    "vertex": str(vertex_label),
                    "mark_count": int(mark_count),
                    "variable_name": str(distractor_variable),
                    "label": str(angle_labels[str(vertex_label)]),
                    "numeric_value_under_x": numeric_value,
                }
            )
    return {
        "side_count": int(side_count),
        "variable_name": str(variable_name),
        "variable_value": int(variable_value),
        "angle_value": int(angle_value),
        "target_angle": str(first_angle),
        "angle_labels": dict(angle_labels),
        "angle_mark_counts": dict(angle_mark_counts),
        "equal_angles": (str(first_angle), str(second_angle)),
        "angle_distractors": list(angle_distractors),
        "distractor_mode": bool(distractor_mode),
        "witness": {
            "polygon_kind": polygon_kind(side_count),
            "variable_value": int(variable_value),
            "equal_angle_measure": int(angle_value),
            "equation": f"{angle_labels[first_angle]} = {angle_labels[second_angle]}",
            "angle_mark_counts": dict(angle_mark_counts),
            "angle_distractors": list(angle_distractors),
            "distractor_mode": bool(distractor_mode),
            "distractor_count": len(angle_distractors),
        },
    }


def sample_interior_angle_sum_relation(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    namespace: str,
) -> dict[str, Any]:
    """Sample polygon interior-angle labels without binding an objective answer."""

    side_count = _side_count(int(instance_seed), params, f"{namespace}.side_count")
    labels = tuple(chr(ord("A") + index) for index in range(side_count))
    rng = spawn_rng(int(instance_seed), namespace)
    variable_name = str(params.get("variable_name", "x"))
    angles = _polygon_angles(
        side_count,
        instance_seed=int(instance_seed),
        params={**dict(generation_defaults), **dict(params)},
        namespace=f"{namespace}.angles",
    )
    target_index = int(rng.randrange(side_count))
    variable_value = int(rng.randint(12, 96))
    expression_count = 2 if side_count <= 4 else 3
    expression_indices = [target_index]
    for offset in range(1, side_count):
        if len(expression_indices) >= expression_count:
            break
        expression_indices.append((target_index + offset) % side_count)

    angle_labels: dict[str, str] = {}
    expression_values: dict[str, int] = {}
    for index, value in enumerate(angles):
        label = labels[index]
        if index in expression_indices:
            coefficient = 1 if index == target_index else int(rng.choice((1, 2, 3)))
            offset = int(value) - (int(coefficient) * int(variable_value))
            angle_labels[label] = format_angle_expression(coefficient, variable_name, offset)
            expression_values[label] = int(value)
        else:
            angle_labels[label] = format_degrees(int(value))

    target_angle = labels[target_index]
    return {
        "side_count": int(side_count),
        "variable_name": str(variable_name),
        "variable_value": int(variable_value),
        "target_angle": str(target_angle),
        "target_angle_value": int(angles[target_index]),
        "angle_labels": dict(angle_labels),
        "witness": {
            "polygon_kind": polygon_kind(side_count),
            "variable_value": int(variable_value),
            "numeric_angle_values": [int(value) for value in angles],
            "expression_values": dict(expression_values),
            "equation": f"sum(interior_angles) = {(side_count - 2) * 180}",
        },
    }


def sample_equal_side_perimeter_relation(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    namespace: str,
    include_distractors: bool = True,
) -> dict[str, Any]:
    """Sample side labels where equal-side algebra gives x before summing perimeter."""

    relation = sample_equal_side_relation(
        instance_seed=int(instance_seed),
        params=params,
        namespace=namespace,
        include_distractors=False,
    )
    side_count = int(relation["side_count"])
    labels = tuple(chr(ord("A") + index) for index in range(side_count))
    rng = spawn_rng(int(instance_seed), f"{namespace}.perimeter")
    variable_name = str(relation["variable_name"])
    variable_value = int(relation["variable_value"])
    equal_sides = {str(side_label) for side_label in relation["equal_sides"]}
    side_values: dict[str, int] = {str(side_label): int(relation["side_value"]) for side_label in equal_sides}
    side_labels = dict(relation["side_labels"])
    side_mark_counts = dict(relation["side_mark_counts"])
    available_sides = [side_label for side_label in _side_names(labels) if str(side_label) not in equal_sides]
    rng.shuffle(available_sides)
    distractor_mode = _use_distractors(rng=rng, include_distractors=include_distractors)
    marked_distractor_sides = (
        set(available_sides[: _choose_distractor_count(rng=rng, available_count=len(available_sides))])
        if bool(distractor_mode)
        else set()
    )
    mark_counts = [1, 3]
    rng.shuffle(mark_counts)
    side_distractors: list[dict[str, Any]] = []
    for side_label in _side_names(labels):
        if str(side_label) in side_values:
            continue
        side_value = int(rng.randint(12, 86))
        side_values[str(side_label)] = int(side_value)
        force_expression = str(side_label) in marked_distractor_sides
        if bool(force_expression):
            side_labels[str(side_label)] = _sample_side_expression_label(
                rng=rng,
                side_value=int(side_value),
                variable_name=variable_name,
                variable_value=int(variable_value),
                force_expression=True,
            )
        else:
            side_labels[str(side_label)] = _format_side_measure(int(side_value))
        if str(side_label) in marked_distractor_sides:
            mark_count = int(mark_counts.pop(0))
            side_mark_counts[str(side_label)] = int(mark_count)
            side_distractors.append(
                {
                    "side": str(side_label),
                    "mark_count": int(mark_count),
                    "variable_name": str(variable_name),
                    "label": str(side_labels[str(side_label)]),
                    "numeric_value_under_x": int(side_value),
                }
            )
    perimeter_value = int(sum(side_values.values()))
    witness = {
        **dict(relation["witness"]),
        "side_mark_counts": dict(side_mark_counts),
        "side_distractors": list(side_distractors),
        "distractor_mode": bool(distractor_mode),
        "perimeter_side_values": dict(side_values),
        "perimeter_value": int(perimeter_value),
        "perimeter_equation": " + ".join(str(side_labels[side_label]) for side_label in _side_names(labels)),
        "distractor_count": len(side_distractors),
    }
    return {
        **dict(relation),
        "side_labels": dict(side_labels),
        "side_mark_counts": dict(side_mark_counts),
        "side_distractors": list(side_distractors),
        "distractor_mode": bool(distractor_mode),
        "side_values": dict(side_values),
        "perimeter_value": int(perimeter_value),
        "witness": dict(witness),
    }


__all__ = [
    "sample_equal_angle_relation",
    "sample_equal_side_perimeter_relation",
    "sample_equal_side_relation",
    "sample_interior_angle_sum_relation",
]
