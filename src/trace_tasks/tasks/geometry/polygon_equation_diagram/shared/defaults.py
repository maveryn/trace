"""Defaults for the polygon equation diagram scene."""

from __future__ import annotations

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.geometry.shared.noise_defaults import POST_IMAGE_NOISE_DEFAULTS

DOMAIN = "geometry"
SCENE_ID = "polygon_equation_diagram"
SCENE_KIND = "polygon_equation_diagram"
SCENE_VARIANT = "algebraic_polygon_measurement"
PROMPT_BUNDLE_ID = "geometry_polygon_equation_diagram_v1"

SCENE_DEFAULTS = get_scene_defaults(DOMAIN, SCENE_ID)

__all__ = [
    "DOMAIN",
    "POST_IMAGE_NOISE_DEFAULTS",
    "PROMPT_BUNDLE_ID",
    "SCENE_DEFAULTS",
    "SCENE_ID",
    "SCENE_KIND",
    "SCENE_VARIANT",
]
