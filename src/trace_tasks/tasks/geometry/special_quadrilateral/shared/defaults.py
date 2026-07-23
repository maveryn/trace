"""Defaults and constants for special-quadrilateral scene packages."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.geometry.shared.noise_defaults import POST_IMAGE_NOISE_DEFAULTS
from trace_tasks.tasks.shared.config_defaults import split_scene_generation_rendering_prompt_defaults

from .state import DOMAIN, SCENE_ID

PROMPT_BUNDLE_ID = "geometry_special_quadrilateral_v1"
SCENE_PROMPT_KEY = "special_quadrilateral_scene"
SCENE_KIND = "analytical_special_quadrilateral"

_SCENE_DEFAULTS = get_scene_defaults(DOMAIN, SCENE_ID)


def load_special_quadrilateral_defaults() -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    """Load scene defaults without public task routing."""

    return split_scene_generation_rendering_prompt_defaults(
        _SCENE_DEFAULTS,
    )


__all__ = [
    "DOMAIN",
    "POST_IMAGE_NOISE_DEFAULTS",
    "PROMPT_BUNDLE_ID",
    "SCENE_ID",
    "SCENE_KIND",
    "SCENE_PROMPT_KEY",
    "load_special_quadrilateral_defaults",
]
