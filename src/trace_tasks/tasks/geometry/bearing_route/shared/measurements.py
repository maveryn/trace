"""Visible bearing and distance measurement formatting helpers."""

from __future__ import annotations

from .state import DEGREE_SYMBOL


def leg_label(distance: int, bearing: int) -> str:
    return f"{int(bearing)}{DEGREE_SYMBOL} {int(distance)} steps"


def bearing_label(bearing: int) -> str:
    return f"{int(bearing)}{DEGREE_SYMBOL}"


__all__ = ["bearing_label", "leg_label"]
