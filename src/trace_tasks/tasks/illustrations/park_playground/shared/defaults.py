"""Default ranges for park/playground count tasks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CountDefaults:
    """Numeric scene defaults shared by park/playground count tasks."""

    person_count_min: int = 7
    person_count_max: int = 12
    target_count_min: int = 1
    target_count_max: int = 6
    equipment_count_min: int = 4
    equipment_count_max: int = 7
    canvas_width: int = 1280
    canvas_height: int = 900
    render_scale: int = 2


__all__ = ["CountDefaults"]
