"""Scene defaults for incircle-tangent geometry tasks."""

from __future__ import annotations

from typing import Any, Dict

from trace_tasks.core.scene_config import get_scene_defaults

DOMAIN = "geometry"
SCENE_ID = "incircle_tangents"
SCENE_KIND = "geometry_tangent_polygon_incircle"
SCENE_VARIANT = "triangle_incircle"
PROMPT_BUNDLE_ID = "geometry_tangent_polygon_incircle_v0"

SCENE_DEFAULTS = get_scene_defaults(DOMAIN, SCENE_ID)

BACKGROUND_DEFAULTS: Dict[str, Any] = {
    "enabled": True,
    "styles": {
        "paper_white": {"kind": "solid", "color": [255, 255, 252]},
        "cool_paper": {"kind": "solid", "color": [248, 252, 255]},
        "warm_paper": {"kind": "solid", "color": [255, 251, 246]},
    },
    "weights": {"paper_white": 1.0, "cool_paper": 1.0, "warm_paper": 1.0},
}

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


__all__ = [
    "BACKGROUND_DEFAULTS",
    "DOMAIN",
    "POST_IMAGE_NOISE_DEFAULTS",
    "PROMPT_BUNDLE_ID",
    "SCENE_DEFAULTS",
    "SCENE_ID",
    "SCENE_KIND",
    "SCENE_VARIANT",
]
