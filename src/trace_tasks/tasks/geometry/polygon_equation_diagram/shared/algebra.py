"""Algebra formatting helpers for polygon equation diagrams."""

from __future__ import annotations


def format_linear_expression(coefficient: int, variable_name: str, offset: int) -> str:
    """Format `coefficient * variable + offset` without awkward `1x` text."""

    coefficient = int(coefficient)
    offset = int(offset)
    variable_name = str(variable_name)
    if coefficient == 1:
        text = variable_name
    elif coefficient == -1:
        text = f"-{variable_name}"
    else:
        text = f"{coefficient}{variable_name}"
    if offset > 0:
        text = f"{text}+{offset}"
    elif offset < 0:
        text = f"{text}{offset}"
    return text


def format_angle_expression(coefficient: int, variable_name: str, offset: int) -> str:
    """Format a linear expression intended as a degree angle label."""

    return f"({format_linear_expression(coefficient, variable_name, offset)})°"


def format_degrees(value: int) -> str:
    """Format an integer angle value in degrees."""

    return f"{int(value)}°"


def side_name(labels: tuple[str, ...], index: int) -> str:
    """Return the side name starting at vertex index."""

    return f"{labels[int(index)]}{labels[(int(index) + 1) % len(labels)]}"


def angle_name(labels: tuple[str, ...], index: int) -> str:
    """Return the angle name at vertex index."""

    return f"angle {labels[int(index)]}"


__all__ = [
    "angle_name",
    "format_angle_expression",
    "format_degrees",
    "format_linear_expression",
    "side_name",
]
