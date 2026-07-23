"""Pure angle and algebra relations for the angle-relations scene."""

from __future__ import annotations


def exterior_angle_value(angle_a: int, angle_b: int) -> int:
    """Return the exterior angle equal to the sum of two remote interior angles."""

    return int(angle_a) + int(angle_b)


def supplement_angle_value(given_angle: int) -> int:
    """Return the supplement of one visible angle."""

    return 180 - int(given_angle)


def linear_expression_value(coefficient: int, constant: int, x_value: int) -> int:
    """Evaluate a linear expression at ``x``."""

    return (int(coefficient) * int(x_value)) + int(constant)


def format_linear_expression(coefficient: int, constant: int) -> str:
    """Format a compact linear expression for an angle label."""

    coeff = int(coefficient)
    const = int(constant)
    if coeff == 1:
        body = "x"
    elif coeff == -1:
        body = "-x"
    else:
        body = f"{coeff}x"
    if const > 0:
        return f"{body}+{const}"
    if const < 0:
        return f"{body}{const}"
    return body


def format_angle_expression(coefficient: int, constant: int) -> str:
    """Format a linear angle expression with a degree marker."""

    return f"({format_linear_expression(coefficient, constant)})°"


def format_degrees(value: int | str) -> str:
    """Format one integer degree label."""

    return f"{value}°"


__all__ = [
    "exterior_angle_value",
    "format_angle_expression",
    "format_degrees",
    "format_linear_expression",
    "linear_expression_value",
    "supplement_angle_value",
]
