"""Scene defaults for container volume-transfer geometry tasks."""

from __future__ import annotations

from typing import Any

from trace_tasks.tasks.geometry.shared.noise_defaults import POST_IMAGE_NOISE_DEFAULTS
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults

DOMAIN = "geometry"
SCENE_ID = "container_volume_transfer"
PROMPT_BUNDLE_ID = "geometry_container_volume_transfer_v1"
SCENE_KIND = "geometry_container_volume_transfer"


def load_container_volume_transfer_task_defaults(public_identifier: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load scene render and prompt defaults for one public task."""

    _generation_defaults, render_defaults, prompt_defaults = load_scene_generation_rendering_prompt_defaults(
        DOMAIN,
        SCENE_ID,
        task_id=str(public_identifier),
    )
    return dict(render_defaults), dict(prompt_defaults)


__all__ = [
    "DOMAIN",
    "PROMPT_BUNDLE_ID",
    "POST_IMAGE_NOISE_DEFAULTS",
    "SCENE_ID",
    "SCENE_KIND",
    "load_container_volume_transfer_task_defaults",
]
