"""Defaults for the survey-traverse scene package."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.geometry.shared.noise_defaults import POST_IMAGE_NOISE_DEFAULTS
from trace_tasks.tasks.shared.config_defaults import split_scene_generation_rendering_prompt_defaults

from .state import DOMAIN, SCENE_ID

PROMPT_BUNDLE_ID = "geometry_survey_traverse_v1"
SCENE_PROMPT_KEY = "survey_traverse_scene"

_SCENE_DEFAULTS = get_scene_defaults(DOMAIN, SCENE_ID)


def load_survey_traverse_defaults() -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    """Load scene defaults without public task routing."""

    return split_scene_generation_rendering_prompt_defaults(
        _SCENE_DEFAULTS,
    )


__all__ = [
    "POST_IMAGE_NOISE_DEFAULTS",
    "PROMPT_BUNDLE_ID",
    "SCENE_PROMPT_KEY",
    "load_survey_traverse_defaults",
]
