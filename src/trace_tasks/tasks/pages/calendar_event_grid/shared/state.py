"""Passive scene state for pages calendar-event-grid tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from trace_tasks.tasks.pages.shared.calendar_scene import CalendarEventChipSpec


@dataclass(frozen=True)
class EventGridCase:
    """Resolved scene and semantic operands for one calendar event-grid."""

    scene_variant: str
    style_variant: str
    accent_color_name: str
    layout_mode: str
    title_mode: str
    surface_mode: str
    text_color_mode: str
    year: int
    month: int
    month_name: str
    days_in_month: int
    row_count: int
    slot_id: str
    slot_label: str
    category_label: str
    event_category_labels: Tuple[str, ...]
    target_date: int | None
    target_count: int | None
    event_chips: Tuple[CalendarEventChipSpec, ...]
    matching_chip_keys: Tuple[str, ...]
    weekday_index: int | None
    weekday_label: str | None
    category_probabilities: Dict[str, float]
    slot_probabilities: Dict[str, float]
    weekday_probabilities: Dict[str, float]
    target_count_probabilities: Dict[str, float]
    scene_variant_probabilities: Dict[str, float]
    style_variant_probabilities: Dict[str, float]
    accent_color_name_probabilities: Dict[str, float]
    layout_mode_probabilities: Dict[str, float]
    title_mode_probabilities: Dict[str, float]
    surface_mode_probabilities: Dict[str, float]
    text_color_mode_probabilities: Dict[str, float]
