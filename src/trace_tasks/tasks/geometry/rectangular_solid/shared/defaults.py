"""Defaults for the rectangular-solid geometry scene."""

from __future__ import annotations

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.geometry.shared.noise_defaults import POST_IMAGE_NOISE_DEFAULTS
from trace_tasks.tasks.shared.config_defaults import (
    load_scene_generation_rendering_prompt_defaults,
)

DOMAIN = "geometry"
SCENE_ID = "rectangular_solid"
PROMPT_BUNDLE_ID = "geometry_rectangular_solid_v1"


def load_rectangular_solid_defaults(*, public_identifier: str) -> tuple[dict, dict, dict]:
    """Load scene defaults for one public rectangular-solid objective."""

    return load_scene_generation_rendering_prompt_defaults(
        DOMAIN,
        SCENE_ID,
        task_id=str(public_identifier),
    )


def raw_scene_defaults() -> dict:
    """Return the raw merged scene defaults used by review tools."""

    return dict(get_scene_defaults(DOMAIN, SCENE_ID))


__all__ = [
    "DOMAIN",
    "POST_IMAGE_NOISE_DEFAULTS",
    "PROMPT_BUNDLE_ID",
    "SCENE_ID",
    "load_rectangular_solid_defaults",
    "raw_scene_defaults",
]
