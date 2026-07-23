"""Config defaults for the Tetris scene package."""

from __future__ import annotations

from typing import Mapping

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.shared.config_defaults import split_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.games.shared.visual_defaults import load_games_scene_noise_defaults

from .state import SCENE_ID, TetrisDefaults

DEFAULTS = TetrisDefaults()
SCENE_DEFAULTS = get_scene_defaults("games", SCENE_ID)
GEN_DEFAULTS, RENDER_DEFAULTS, PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    SCENE_DEFAULTS if isinstance(SCENE_DEFAULTS, Mapping) else {},
    task_id="games_tetris_base",
)
POST_IMAGE_NOISE_DEFAULTS = load_games_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)
