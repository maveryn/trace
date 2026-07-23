"""Defaults and prompt keys for printed map scene packages."""

from __future__ import annotations

from typing import Mapping, Tuple

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.pages.shared.diagram.visual_defaults import (
    load_diagrams_scene_background_defaults,
    load_diagrams_scene_noise_defaults,
)
from trace_tasks.tasks.shared.config_defaults import split_scene_generation_rendering_prompt_defaults

from .state import MapDefaults


DOMAIN = "pages"
SCENE = "map"
NAMESPACE_ROOT = "pages.map"

PROMPT_BUNDLE = "pages_map_v1"
PROMPT_SCENE_KEY = "printed_map"
PROMPT_TASK_KEY = "map_navigation_query"

SCENE_VARIANTS: Tuple[str, ...] = ("campus_map",)

DEFAULTS = MapDefaults()
SCENE_DEFAULTS = get_scene_defaults(DOMAIN, SCENE)
GENERATION_DEFAULTS, RENDERING_DEFAULTS, PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    SCENE_DEFAULTS if isinstance(SCENE_DEFAULTS, Mapping) else {},
)
POST_IMAGE_BACKGROUND_DEFAULTS = load_diagrams_scene_background_defaults(scene_id=SCENE)
POST_IMAGE_NOISE_DEFAULTS = load_diagrams_scene_noise_defaults(scene_id=SCENE, apply_prob=0.0)


__all__ = [
    "DEFAULTS",
    "DOMAIN",
    "GENERATION_DEFAULTS",
    "NAMESPACE_ROOT",
    "POST_IMAGE_BACKGROUND_DEFAULTS",
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
