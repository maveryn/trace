"""Scene-level defaults for composite-shape geometry tasks."""

from __future__ import annotations

from typing import Any, Dict

from trace_tasks.core.scene_config import get_scene_defaults

SCENE_ID = "composite_shape"
RECTILINEAR_PROMPT_BUNDLE_ID = "geometry_analytical_measurement_v0"
CURVILINEAR_PROMPT_BUNDLE_ID = "geometry_curvilinear_composite_v0"
PI_VALUE = 3.141592653589793

SCENE_DEFAULTS = get_scene_defaults("geometry", SCENE_ID)

POST_IMAGE_NOISE_DEFAULTS: Dict[str, Any] = {
    "apply_prob": 0.5,
    "edit_types": ["blur", "downsample", "jpeg", "noise"],
    "edit_count_range": [1, 1],
    "value_ranges": {
        "blur": {"radius": [0.12, 0.32]},
        "downsample": {"scale": [0.93, 0.98]},
        "jpeg": {"quality": [84, 94]},
        "noise": {"alpha": [0.01, 0.03]},
    },
}
