"""Defaults for the circular-sector geometry scene."""

from __future__ import annotations

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.geometry.shared.noise_defaults import POST_IMAGE_NOISE_DEFAULTS

DOMAIN = "geometry"
SCENE_ID = "sector"
PROMPT_BUNDLE_ID = "geometry_sector_formula_v1"


def raw_scene_defaults() -> dict:
    """Return merged scene defaults for sector generation/rendering/prompt settings."""

    return dict(get_scene_defaults(DOMAIN, SCENE_ID))


__all__ = ["DOMAIN", "POST_IMAGE_NOISE_DEFAULTS", "PROMPT_BUNDLE_ID", "SCENE_ID", "raw_scene_defaults"]
