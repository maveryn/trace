"""Scene-local defaults for bearing-route geometry tasks."""

from __future__ import annotations

from typing import Any


PROMPT_WIRING_KEYS: tuple[str, ...] = (
    "bundle_id",
    "scene_key",
    "task_key",
)

POST_IMAGE_NOISE_DEFAULTS: dict[str, Any] = {
    "apply_prob": 0.40,
    "edit_types": ["blur", "downsample", "jpeg", "noise"],
    "edit_count_range": [1, 1],
    "value_ranges": {
        "blur": {"radius": [0.06, 0.18]},
        "downsample": {"scale": [0.96, 0.99]},
        "jpeg": {"quality": [90, 97]},
        "noise": {"alpha": [0.006, 0.018]},
    },
}


__all__ = ["POST_IMAGE_NOISE_DEFAULTS", "PROMPT_WIRING_KEYS"]
