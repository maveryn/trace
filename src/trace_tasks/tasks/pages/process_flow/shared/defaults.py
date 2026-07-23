"""Defaults and prompt identifiers for process-flow page scenes."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Mapping

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.pages.shared.visual_defaults import (
    load_pages_scene_background_defaults,
    load_pages_scene_noise_defaults,
)
from trace_tasks.tasks.shared.config_defaults import (
    split_scene_generation_rendering_prompt_defaults,
)

from .state import ProcessFlowDefaults


DOMAIN = "pages"
SCENE = "process_flow"
PROMPT_BUNDLE = "pages_process_flow_v1"
PROMPT_SCENE_KEY = "process_flow_diagram"
PROMPT_TASK_KEY = "process_flow_diagram_query"
NAMESPACE_ROOT = "pages.process_flow"

DEFAULTS = ProcessFlowDefaults()
SCENE_DEFAULTS = get_scene_defaults(DOMAIN, SCENE)
GENERATION_DEFAULTS, RENDERING_DEFAULTS, PROMPT_DEFAULTS = (
    split_scene_generation_rendering_prompt_defaults(
        SCENE_DEFAULTS if isinstance(SCENE_DEFAULTS, Mapping) else {},
    )
)
RENDER_FALLBACKS: Dict[str, Any] = asdict(DEFAULTS)
POST_IMAGE_BACKGROUND_DEFAULTS = load_pages_scene_background_defaults(scene_id=SCENE)
POST_IMAGE_NOISE_DEFAULTS = load_pages_scene_noise_defaults(
    scene_id=SCENE,
    apply_prob=0.5,
)
