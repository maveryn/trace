"""Scene-local defaults for circle-theorem task packages."""

from __future__ import annotations

from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults

from .state import DOMAIN, SCENE_ID

GEN_DEFAULTS, RENDER_DEFAULTS, PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    DOMAIN,
    SCENE_ID,
)

__all__ = [
    "GEN_DEFAULTS",
    "PROMPT_DEFAULTS",
    "RENDER_DEFAULTS",
]
