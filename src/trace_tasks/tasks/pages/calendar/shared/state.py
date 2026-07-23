"""Calendar scene state objects shared by public task files."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class CalendarCase:
    """Resolved scene and semantic state for one month-view calendar."""

    marked_day_class: str | None
    scene_variant: str
    layout_mode: str
    title_mode: str
    week_start: str
    first_weekday_index: int
    year: int
    month: int
    month_name: str
    row_count: int
    start_weekday_index: int
    days_in_month: int
    marked_dates: Tuple[int, ...]
    annotation_dates: Tuple[int, ...]
    answer_value: int
    weekday_index: int | None
    occurrence: int | None
    workday_direction: str | None
    workday_offset: int | None
    reference_date: int | None
    target_date: int | None
    weekend_weekday_indices: Tuple[int, ...]
    date_occurrence_support: Tuple[int, ...]
    marked_weekend_count_support: Tuple[int, ...]
    marked_weekday_count_support: Tuple[int, ...]
    marked_weekday_distractor_support: Tuple[int, ...]
    marked_weekend_distractor_support: Tuple[int, ...]
    workday_offset_support: Tuple[int, ...]
    marked_day_class_probabilities: Dict[str, float]
    scene_variant_probabilities: Dict[str, float]
    layout_mode_probabilities: Dict[str, float]
    title_mode_probabilities: Dict[str, float]
    week_start_probabilities: Dict[str, float]
