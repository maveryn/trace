"""Defaults and stable scene keys for mixed infographic pages."""

from __future__ import annotations

from typing import Mapping, Tuple

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.pages.shared.visual_defaults import load_pages_scene_noise_defaults
from trace_tasks.tasks.shared.config_defaults import split_scene_generation_rendering_prompt_defaults


DOMAIN = "pages"
SCENE = "mixed_infographic_page"
NAMESPACE_ROOT = "pages.mixed_infographic_page"

PROMPT_BUNDLE = "pages_mixed_infographic_page_v1"
PROMPT_SCENE_KEY = "mixed_infographic_page"
PROMPT_TASK_KEY = "mixed_infographic_lookup_query"

SCENE_VARIANTS: Tuple[str, ...] = (
    "masonry_report",
    "dashboard_blocks",
    "poster_sections",
    "compact_newsletter",
    "collage_board",
    "radial_mosaic",
)

SCENE_DEFAULTS = get_scene_defaults(DOMAIN, SCENE)
GENERATION_DEFAULTS, RENDERING_DEFAULTS, PROMPT_DEFAULTS = (
    split_scene_generation_rendering_prompt_defaults(
        SCENE_DEFAULTS if isinstance(SCENE_DEFAULTS, Mapping) else {},
    )
)
POST_IMAGE_NOISE_DEFAULTS = load_pages_scene_noise_defaults(
    scene_id=SCENE,
    apply_prob=0.5,
)


__all__ = [
    "DOMAIN",
    "GENERATION_DEFAULTS",
    "NAMESPACE_ROOT",
    "POST_IMAGE_NOISE_DEFAULTS",
    "PROMPT_BUNDLE",
    "PROMPT_DEFAULTS",
    "PROMPT_SCENE_KEY",
    "PROMPT_TASK_KEY",
    "RENDERING_DEFAULTS",
    "SCENE",
    "SCENE_DEFAULTS",
    "SCENE_VARIANTS",
]
