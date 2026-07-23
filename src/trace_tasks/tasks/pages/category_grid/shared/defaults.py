"""Category-grid scene defaults and visual axes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Tuple

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.pages.shared.visual_defaults import load_pages_noise_defaults
from trace_tasks.tasks.shared.config_defaults import split_scene_generation_rendering_prompt_defaults


DOMAIN = "pages"
SCENE = "category_grid"
PROMPT_BUNDLE = "pages_category_grid_v1"
PROMPT_SCENE_KEY = "category_grid"
PROMPT_TASK_KEY = "category_grid_lookup_query"
NAMESPACE_ROOT = "pages.category_grid"

SCENE_VARIANTS: Tuple[str, ...] = ("card_grid", "column_groups", "compact_index")
ACCENTS: Tuple[Tuple[int, int, int], ...] = (
    (55, 118, 172),
    (42, 142, 112),
    (190, 91, 76),
    (128, 102, 184),
    (198, 142, 58),
    (72, 132, 151),
    (168, 91, 132),
    (89, 122, 92),
)
ORDINALS = {
    1: "first",
    2: "second",
    3: "third",
    4: "fourth",
    5: "fifth",
    6: "sixth",
}


@dataclass(frozen=True)
class CategoryGridDefaults:
    """Stable fallback defaults for category-grid pages."""

    category_count_support: Tuple[int, ...] = (3, 4)
    subcategory_count_support: Tuple[int, ...] = (2, 3)
    item_count_support: Tuple[int, ...] = (2, 3, 4, 5, 6)
    canvas_width: int = 1120
    canvas_height: int = 900
    outer_margin_px: int = 34
    header_height_px: int = 88
    gap_px: int = 14
    corner_radius_px: int = 14
    outline_width_px: int = 2
    title_font_size_px: int = 30
    subtitle_font_size_px: int = 17
    category_title_font_size_px: int = 21
    subcategory_title_font_size_px: int = 15
    item_font_size_px: int = 14


DEFAULTS = CategoryGridDefaults()
SCENE_DEFAULTS = get_scene_defaults(DOMAIN, SCENE)
GENERATION_DEFAULTS, RENDERING_DEFAULTS, PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    SCENE_DEFAULTS if isinstance(SCENE_DEFAULTS, Mapping) else {},
)
RENDER_FALLBACKS = asdict(DEFAULTS)
POST_IMAGE_NOISE_DEFAULTS = load_pages_noise_defaults(scene_id=SCENE, apply_prob=0.0)
