"""Defaults for volume-equivalence conversion scene rendering and prompts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.geometry.shared.noise_defaults import POST_IMAGE_NOISE_DEFAULTS
from trace_tasks.tasks.shared.config_defaults import split_scene_generation_rendering_prompt_defaults

DOMAIN = "geometry"
SCENE_ID = "volume_equivalence_conversion"
SCENE_DEFAULTS = get_scene_defaults(DOMAIN, SCENE_ID)


def load_volume_equivalence_defaults(public_identifier: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return generation, rendering, and prompt defaults for one public entry."""

    generation_defaults, render_defaults, prompt_defaults = split_scene_generation_rendering_prompt_defaults(
        SCENE_DEFAULTS if isinstance(SCENE_DEFAULTS, Mapping) else {},
        task_id=str(public_identifier),
    )
    return dict(generation_defaults), dict(render_defaults), dict(prompt_defaults)


__all__ = [
    "DOMAIN",
    "POST_IMAGE_NOISE_DEFAULTS",
    "SCENE_DEFAULTS",
    "SCENE_ID",
    "load_volume_equivalence_defaults",
]
