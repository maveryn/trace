"""Defaults and scene axes for navigation-flow page packages."""

from __future__ import annotations

from dataclasses import asdict
from typing import Mapping, Tuple

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.pages.shared.visual_defaults import load_pages_background_defaults, load_pages_noise_defaults
from trace_tasks.tasks.shared.config_defaults import split_scene_generation_rendering_prompt_defaults

from .state import NavigationFlowDefaults


DOMAIN = "pages"
SCENE = "navigation_flow"
PROMPT_BUNDLE = "pages_navigation_flow_v1"
PROMPT_SCENE_KEY = "navigation_flow"
PROMPT_TASK_KEY = "navigation_path_query"
NAMESPACE_ROOT = "pages.navigation_flow"

SCENE_VARIANTS: Tuple[str, ...] = (
    "office_document",
    "creative_workspace",
    "developer_ide",
    "cad_workspace",
    "scientific_plotter",
    "os_file_manager",
)

DEFAULTS = NavigationFlowDefaults()
SCENE_DEFAULTS = get_scene_defaults(DOMAIN, SCENE)
GENERATION_DEFAULTS, RENDERING_DEFAULTS, PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    SCENE_DEFAULTS if isinstance(SCENE_DEFAULTS, Mapping) else {},
)
RENDER_FALLBACKS = asdict(DEFAULTS)
POST_IMAGE_NOISE_DEFAULTS = load_pages_noise_defaults(scene_id=SCENE, apply_prob=0.0)
BACKGROUND_DEFAULTS = load_pages_background_defaults(scene_id=SCENE)
