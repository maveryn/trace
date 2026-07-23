"""Calendar event-grid scene defaults and supported visual axes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Tuple

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.pages.shared.calendar_scene import SUPPORTED_PAGE_CALENDAR_SCENE_VARIANTS
from trace_tasks.tasks.pages.shared.visual_defaults import load_pages_background_defaults, load_pages_noise_defaults
from trace_tasks.tasks.shared.config_defaults import split_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.time_artifact_style import (
    SUPPORTED_TIME_ARTIFACT_CALENDAR_TEXT_COLOR_MODES,
    SUPPORTED_TIME_ARTIFACT_COLOR_NAMES,
    SUPPORTED_TIME_ARTIFACT_STYLE_VARIANTS,
)


DOMAIN = "pages"
SCENE = "calendar_event_grid"
PROMPT_BUNDLE = "pages_calendar_event_grid_v1"
PROMPT_SCENE_KEY = "calendar_event_grid"

EVENT_GRID_TASK_KEY = "calendar_event_grid_query"

SUPPORTED_EVENT_GRID_LAYOUT_MODES: Tuple[str, ...] = (
    "center_clean",
    "free_jitter_clean",
    "left_with_side_note",
    "right_with_side_note",
    "top_with_bottom_note",
)
SUPPORTED_EVENT_GRID_TITLE_MODES: Tuple[str, ...] = (
    "generic",
    "full_month_year",
)
SUPPORTED_EVENT_GRID_SURFACE_MODES: Tuple[str, ...] = ("light", "dark")
SUPPORTED_EVENT_GRID_TEXT_COLOR_MODES: Tuple[str, ...] = SUPPORTED_TIME_ARTIFACT_CALENDAR_TEXT_COLOR_MODES

EVENT_SLOT_SPECS: Tuple[Tuple[str, str], ...] = (
    ("top", "Top"),
    ("mid", "Mid"),
    ("end", "Bottom"),
)
EVENT_CATEGORY_LABELS: Tuple[str, ...] = (
    "Arts",
    "Civic",
    "Culture",
    "Finance",
    "Health",
    "Policy",
    "Science",
    "Sports",
    "Tech",
    "Travel",
)
CHIP_FILL_PALETTE: Tuple[Tuple[int, int, int], ...] = (
    (226, 240, 255),
    (222, 245, 233),
    (255, 237, 213),
    (245, 226, 255),
    (255, 228, 232),
    (224, 246, 250),
    (245, 239, 211),
    (230, 234, 255),
)
GENERIC_TITLE_TEXTS: Tuple[str, ...] = (
    "Event Calendar",
    "News Grid",
    "Daily Highlights",
    "Month Board",
)


@dataclass(frozen=True)
class EventGridDefaults:
    """Stable fallback defaults for event-grid calendar scenes."""

    year_min: int = 2022
    year_max: int = 2030
    target_count_support: Tuple[int, ...] = (2, 3, 4, 5, 6)
    date_filled_slot_count_support: Tuple[int, ...] = (0, 1, 2, 3)
    weekday_column_count_support: Tuple[int, ...] = (1, 2, 3, 4)
    weekday_column_distractor_support: Tuple[int, ...] = (3, 4, 5, 6)
    busiest_two_chip_distractor_date_count_support: Tuple[int, ...] = (5, 6, 7, 8)
    busiest_single_chip_distractor_date_count_support: Tuple[int, ...] = (8, 9, 10, 11, 12)
    min_random_extra_chips: int = 10
    max_random_extra_chips: int = 18
    canvas_width: int = 980
    canvas_height: int = 780
    outer_margin_px: int = 34
    title_height_px: int = 62
    title_bottom_gap_px: int = 12
    weekday_header_height_px: int = 32
    weekday_grid_gap_px: int = 8
    cell_gap_px: int = 7
    panel_corner_radius_px: int = 16
    panel_outline_width_px: int = 3
    cell_corner_radius_px: int = 10
    cell_outline_width_px: int = 2
    title_font_size_px: int = 30
    weekday_font_size_px: int = 15
    date_font_size_px: int = 20
    marker_inset_px: int = 8
    marker_outline_width_px: int = 2


DEFAULTS = EventGridDefaults()
SCENE_DEFAULTS = get_scene_defaults(DOMAIN, SCENE)
GENERATION_DEFAULTS, RENDERING_DEFAULTS, PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    SCENE_DEFAULTS if isinstance(SCENE_DEFAULTS, Mapping) else {},
)
RENDER_FALLBACKS = asdict(DEFAULTS)
BACKGROUND_DEFAULTS = load_pages_background_defaults(scene_id=SCENE)
NOISE_DEFAULTS = load_pages_noise_defaults(scene_id=SCENE, apply_prob=0.0)
