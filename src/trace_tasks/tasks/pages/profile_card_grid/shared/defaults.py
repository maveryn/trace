"""Profile-card-grid scene defaults and prompt constants."""

from __future__ import annotations

from dataclasses import asdict
from typing import Mapping

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.pages.shared.visual_defaults import load_pages_background_defaults, load_pages_noise_defaults
from trace_tasks.tasks.shared.config_defaults import split_scene_generation_rendering_prompt_defaults

from .state import ProfileCardGridDefaults


DOMAIN = "pages"
SCENE = "profile_card_grid"
PROMPT_BUNDLE = "pages_profile_card_grid_v1"
PROMPT_SCENE_KEY = "profile_card_grid"
PROMPT_TASK_KEY = "profile_attribute_lookup_query"
NAMESPACE_ROOT = "pages.profile_card_grid"

DEFAULTS = ProfileCardGridDefaults()
SCENE_DEFAULTS = get_scene_defaults(DOMAIN, SCENE)
GENERATION_DEFAULTS, RENDERING_DEFAULTS, PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    SCENE_DEFAULTS if isinstance(SCENE_DEFAULTS, Mapping) else {},
)
RENDER_FALLBACKS = asdict(DEFAULTS)
POST_IMAGE_BACKGROUND_DEFAULTS = load_pages_background_defaults(scene_id=SCENE)
POST_IMAGE_NOISE_DEFAULTS = load_pages_noise_defaults(scene_id=SCENE, apply_prob=0.0)
