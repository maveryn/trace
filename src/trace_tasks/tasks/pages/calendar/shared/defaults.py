"""Calendar scene defaults and supported visual axes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Tuple

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.pages.shared.calendar_scene import SUPPORTED_PAGE_CALENDAR_SCENE_VARIANTS
from trace_tasks.tasks.pages.shared.visual_defaults import load_pages_noise_defaults
from trace_tasks.tasks.shared.config_defaults import split_scene_generation_rendering_prompt_defaults


DOMAIN = "pages"
SCENE = "calendar"
PROMPT_BUNDLE = "pages_calendar_v1"
PROMPT_SCENE_KEY = "month_calendar"

SUPPORTED_PAGE_CALENDAR_LAYOUT_MODES: Tuple[str, ...] = (
    "center_clean",
    "free_jitter_clean",
    "left_with_side_note",
    "right_with_side_note",
    "top_with_bottom_note",
)
SUPPORTED_PAGE_CALENDAR_TITLE_MODES: Tuple[str, ...] = (
    "none",
    "generic",
    "full_month_year",
)
SUPPORTED_MARKED_DAY_CLASSES: Tuple[str, ...] = ("weekend", "weekday")
SUPPORTED_PAGE_CALENDAR_WEEK_STARTS: Tuple[str, ...] = ("monday", "sunday")
WORKDAY_DIRECTIONS: Tuple[str, str] = ("after", "before")

GENERIC_TITLE_TEXTS: Tuple[str, ...] = (
    "Calendar",
    "Month View",
    "Planner",
)

WEEKDAY_OCCURRENCE_TASK_KEY = "calendar_weekday_occurrence_query"
DATE_WEEKDAY_LABEL_TASK_KEY = "calendar_date_weekday_label_query"
MARKED_DAY_CLASS_COUNT_TASK_KEY = "calendar_marked_day_class_count_query"
WORKDAY_OFFSET_TASK_KEY = "calendar_workday_offset_query"
DATE_RANGE_DAY_CLASS_COUNT_TASK_KEY = "calendar_date_range_day_class_count_query"


@dataclass(frozen=True)
class TaskDefaults:
    """Stable fallback defaults for month-view calendar scenes."""

    year_min: int = 2022
    year_max: int = 2030
    weekend_weekday_indices: Tuple[int, int] = (5, 6)
    date_occurrence_support: Tuple[int, ...] = (1, 2, 3, 4, 5)
    marked_weekend_count_support: Tuple[int, ...] = (0, 1, 2, 3, 4)
    marked_weekday_count_support: Tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)
    marked_weekday_distractor_support: Tuple[int, ...] = (1, 2, 3, 4)
    marked_weekend_distractor_support: Tuple[int, ...] = (1, 2, 3, 4)
    workday_offset_support: Tuple[int, ...] = (2, 3, 4, 5, 6, 7)
    date_range_day_count_support: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    date_range_min_span_days: int = 4
    date_range_max_span_days: int = 16
    canvas_width: int = 860
    canvas_height: int = 760
    outer_margin_px: int = 34
    title_height_px: int = 64
    title_bottom_gap_px: int = 14
    weekday_header_height_px: int = 34
    weekday_grid_gap_px: int = 10
    cell_gap_px: int = 8
    panel_corner_radius_px: int = 18
    panel_outline_width_px: int = 3
    cell_corner_radius_px: int = 12
    cell_outline_width_px: int = 2
    title_font_size_px: int = 30
    weekday_font_size_px: int = 16
    date_font_size_px: int = 22
    marker_inset_px: int = 10
    marker_outline_width_px: int = 3


DEFAULTS = TaskDefaults()
SCENE_DEFAULTS = get_scene_defaults(DOMAIN, SCENE)
GENERATION_DEFAULTS, RENDERING_DEFAULTS, PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    SCENE_DEFAULTS if isinstance(SCENE_DEFAULTS, Mapping) else {},
)
RENDER_FALLBACKS = asdict(DEFAULTS)
NOISE_DEFAULTS = load_pages_noise_defaults(scene_id=SCENE, apply_prob=0.0)

__all__ = [
    "DEFAULTS",
    "DATE_RANGE_DAY_CLASS_COUNT_TASK_KEY",
    "DATE_WEEKDAY_LABEL_TASK_KEY",
    "DOMAIN",
    "GENERATION_DEFAULTS",
    "GENERIC_TITLE_TEXTS",
    "MARKED_DAY_CLASS_COUNT_TASK_KEY",
    "NOISE_DEFAULTS",
    "PROMPT_BUNDLE",
    "PROMPT_DEFAULTS",
    "PROMPT_SCENE_KEY",
    "RENDERING_DEFAULTS",
    "RENDER_FALLBACKS",
    "SCENE",
    "SUPPORTED_MARKED_DAY_CLASSES",
    "SUPPORTED_PAGE_CALENDAR_LAYOUT_MODES",
    "SUPPORTED_PAGE_CALENDAR_SCENE_VARIANTS",
    "SUPPORTED_PAGE_CALENDAR_TITLE_MODES",
    "SUPPORTED_PAGE_CALENDAR_WEEK_STARTS",
    "TaskDefaults",
    "WEEKDAY_OCCURRENCE_TASK_KEY",
    "WORKDAY_DIRECTIONS",
    "WORKDAY_OFFSET_TASK_KEY",
]
