"""Defaults for the measuring-tools geometry scene package."""

from __future__ import annotations

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.geometry.shared.noise_defaults import POST_IMAGE_NOISE_DEFAULTS

DOMAIN = "geometry"
SCENE_ID = "measuring_tools"
SCENE_KIND = "geometry_measuring_tools"
SCENE_VARIANT = "visible_tool_readout"
PROMPT_BUNDLE_ID = "geometry_measuring_tools_v0"

SCENE_DEFAULTS = get_scene_defaults(DOMAIN, SCENE_ID)

__all__ = [
    "DOMAIN",
    "PROMPT_BUNDLE_ID",
    "POST_IMAGE_NOISE_DEFAULTS",
    "SCENE_DEFAULTS",
    "SCENE_ID",
    "SCENE_KIND",
    "SCENE_VARIANT",
]
