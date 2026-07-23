"""Scene-level constants for Nine Men's Morris game tasks."""

from __future__ import annotations

from trace_tasks.tasks.games.shared.visual_defaults import load_games_scene_noise_defaults


SCENE_ID = "nine_mens_morris"
SUPPORTED_NINE_MENS_MORRIS_SCENE_VARIANTS: tuple[str, ...] = ("single_board",)
POST_IMAGE_NOISE_DEFAULTS = load_games_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)


__all__ = [
    "POST_IMAGE_NOISE_DEFAULTS",
    "SCENE_ID",
    "SUPPORTED_NINE_MENS_MORRIS_SCENE_VARIANTS",
]
