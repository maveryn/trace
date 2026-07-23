"""Defaults for the Pythagorean tree scene."""

from __future__ import annotations

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.geometry.shared.noise_defaults import load_geometry_noise_defaults

DOMAIN = "geometry"
SCENE_ID = "pythagorean_tree"
SCENE_KIND = "geometry_pythagorean_attached_square_tree"
SCENE_VARIANT = "attached_square_tree"
PROMPT_BUNDLE_ID = "geometry_pythagorean_tree_v1"

SCENE_DEFAULTS = get_scene_defaults(DOMAIN, SCENE_ID)
POST_IMAGE_NOISE_DEFAULTS = load_geometry_noise_defaults(scene_id=SCENE_ID)

__all__ = [
    "DOMAIN",
    "POST_IMAGE_NOISE_DEFAULTS",
    "PROMPT_BUNDLE_ID",
    "SCENE_DEFAULTS",
    "SCENE_ID",
    "SCENE_KIND",
    "SCENE_VARIANT",
]
