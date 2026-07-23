"""Scene default loading for tower draughts board tasks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, Tuple

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.games.shared.visual_defaults import load_games_scene_noise_defaults
from trace_tasks.tasks.shared.config_defaults import split_scene_generation_rendering_prompt_defaults

from .state import DOMAIN, SCENE_ID, TowerDraughtsDefaults


DEFAULTS = TowerDraughtsDefaults()
_SCENE_DEFAULTS = get_scene_defaults(DOMAIN, SCENE_ID)
GEN_DEFAULTS, RENDER_DEFAULTS, PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    _SCENE_DEFAULTS if isinstance(_SCENE_DEFAULTS, Mapping) else {},
)
POST_IMAGE_NOISE_DEFAULTS = load_games_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)


def default_sections() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Return mutable copies of loaded generation, rendering, and prompt defaults."""

    return dict(GEN_DEFAULTS), dict(RENDER_DEFAULTS), dict(PROMPT_DEFAULTS)


__all__ = [
    "DEFAULTS",
    "GEN_DEFAULTS",
    "POST_IMAGE_NOISE_DEFAULTS",
    "PROMPT_DEFAULTS",
    "RENDER_DEFAULTS",
    "default_sections",
]
