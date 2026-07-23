"""Scene defaults for cone-net geometry tasks."""

from __future__ import annotations

from typing import Any, Dict

from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults

DOMAIN = "geometry"
SCENE_ID = "cone_net"
SCENE_KIND = "geometry_cone_sector_net"
SCENE_VARIANT = "sector_net_to_cone"

POST_IMAGE_NOISE_DEFAULTS: Dict[str, Any] = {
    "apply_prob": 0.45,
    "edit_types": ["blur", "downsample", "jpeg", "noise"],
    "edit_count_range": [1, 1],
    "value_ranges": {
        "blur": {"radius": [0.08, 0.22]},
        "downsample": {"scale": [0.95, 0.99]},
        "jpeg": {"quality": [88, 96]},
        "noise": {"alpha": [0.008, 0.02]},
    },
}


def load_cone_net_task_defaults(public_identifier: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load scene render and prompt defaults for one public task."""

    _generation_defaults, render_defaults, prompt_defaults = load_scene_generation_rendering_prompt_defaults(
        DOMAIN,
        SCENE_ID,
        task_id=str(public_identifier),
    )
    return dict(render_defaults), dict(prompt_defaults)


__all__ = [
    "DOMAIN",
    "load_cone_net_task_defaults",
    "POST_IMAGE_NOISE_DEFAULTS",
    "SCENE_ID",
    "SCENE_KIND",
    "SCENE_VARIANT",
]
