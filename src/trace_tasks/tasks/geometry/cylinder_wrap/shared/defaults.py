"""Scene defaults for cylinder-wrap geometry diagrams."""

from __future__ import annotations

from typing import Any, Dict

DOMAIN = "geometry"
SCENE_ID = "cylinder_wrap"
SCENE_KIND = "geometry_cylinder_wrap"

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

WRAP_STYLE_IDS: tuple[str, ...] = (
    "split_net",
    "drafting_strip",
    "rim_projection",
    "workshop_sheet",
)
MARKER_STYLE_IDS: tuple[str, ...] = ("ring", "target", "diamond", "square")

__all__ = [
    "DOMAIN",
    "MARKER_STYLE_IDS",
    "POST_IMAGE_NOISE_DEFAULTS",
    "SCENE_ID",
    "SCENE_KIND",
    "WRAP_STYLE_IDS",
]
