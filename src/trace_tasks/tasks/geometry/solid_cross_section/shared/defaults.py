"""Scene defaults for solid cross-section geometry tasks."""

from __future__ import annotations

from typing import Any

from trace_tasks.tasks.geometry.shared.noise_defaults import load_geometry_noise_defaults
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults

DOMAIN = "geometry"
SCENE_ID = "solid_cross_section"
SCENE_KIND = "geometry_solid_cross_section"

POST_IMAGE_NOISE_DEFAULTS: dict[str, Any] = load_geometry_noise_defaults(scene_id=SCENE_ID)


def load_solid_cross_section_defaults(public_identifier: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load generation, rendering, and prompt defaults for one public task."""

    generation_defaults, render_defaults, prompt_defaults = load_scene_generation_rendering_prompt_defaults(
        DOMAIN,
        SCENE_ID,
        task_id=str(public_identifier),
    )
    return dict(generation_defaults), dict(render_defaults), dict(prompt_defaults)


__all__ = [
    "DOMAIN",
    "load_solid_cross_section_defaults",
    "POST_IMAGE_NOISE_DEFAULTS",
    "SCENE_ID",
    "SCENE_KIND",
]
