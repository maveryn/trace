"""Scene-level config defaults for Ultimate Tic-Tac-Toe."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.games.shared.visual_defaults import load_games_scene_noise_defaults
from trace_tasks.tasks.shared.config_defaults import (
    group_default,
    split_scene_generation_rendering_prompt_defaults,
)

from .state import SCENE_ID, UltimateDefaults


DEFAULTS = UltimateDefaults()
SCENE_DEFAULTS = get_scene_defaults("games", SCENE_ID)
GEN_DEFAULTS, RENDER_DEFAULTS, PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    SCENE_DEFAULTS if isinstance(SCENE_DEFAULTS, Mapping) else {},
)
POST_IMAGE_NOISE_DEFAULTS = load_games_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)


def int_render_default(params: Mapping[str, Any], key: str, fallback: int) -> int:
    """Resolve one integer render parameter from params, scene config, or fallback."""

    if str(key) in params:
        return int(params[str(key)])
    return int(group_default(RENDER_DEFAULTS, str(key), int(fallback)))
