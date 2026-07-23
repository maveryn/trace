"""Config defaults for triangle-relations scene-package tasks."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.geometry.shared.noise_defaults import load_geometry_noise_defaults
from trace_tasks.tasks.shared.config_defaults import split_scene_generation_rendering_prompt_defaults

from .state import SCENE_ID

POST_IMAGE_NOISE_DEFAULTS = load_geometry_noise_defaults(scene_id=SCENE_ID)
_SCENE_DEFAULTS = get_scene_defaults("geometry", SCENE_ID)


def load_triangle_relations_defaults() -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    """Load scene defaults without task-owned routing or query overrides."""

    return split_scene_generation_rendering_prompt_defaults(_SCENE_DEFAULTS)


__all__ = ["POST_IMAGE_NOISE_DEFAULTS", "load_triangle_relations_defaults"]
