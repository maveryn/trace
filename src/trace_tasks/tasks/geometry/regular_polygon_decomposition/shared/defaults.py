"""Defaults for the regular-polygon decomposition geometry scene."""

from __future__ import annotations

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.geometry.shared.noise_defaults import POST_IMAGE_NOISE_DEFAULTS

DOMAIN = "geometry"
SCENE_ID = "regular_polygon_decomposition"
PROMPT_BUNDLE_ID = "geometry_regular_polygon_decomposition_v1"


def raw_scene_defaults() -> dict:
    """Return merged scene defaults for generation, rendering, and prompts."""

    return dict(get_scene_defaults(DOMAIN, SCENE_ID))


__all__ = ["DOMAIN", "POST_IMAGE_NOISE_DEFAULTS", "PROMPT_BUNDLE_ID", "SCENE_ID", "raw_scene_defaults"]
