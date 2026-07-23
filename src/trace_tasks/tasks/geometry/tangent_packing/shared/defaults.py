"""Config defaults for tangent-packing scene-package tasks."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.shared.config_defaults import split_scene_generation_rendering_prompt_defaults

from .state import PROMPT_BUNDLE_ID, SCENE_ID, SCENE_PROMPT_KEY

_SCENE_DEFAULTS = get_scene_defaults("geometry", SCENE_ID)

BACKGROUND_DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "styles": {
        "paper_white": {"kind": "solid", "color": [255, 255, 252]},
        "cool_paper": {"kind": "solid", "color": [248, 252, 255]},
        "warm_paper": {"kind": "solid", "color": [255, 251, 246]},
    },
    "weights": {"paper_white": 1.0, "cool_paper": 1.0, "warm_paper": 1.0},
}

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


def load_tangent_packing_defaults() -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    """Load scene-level generation/rendering/prompt defaults."""

    generation_defaults, rendering_defaults, prompt_defaults = split_scene_generation_rendering_prompt_defaults(
        _SCENE_DEFAULTS,
    )
    merged_prompt_defaults = {
        "bundle_id": PROMPT_BUNDLE_ID,
        "scene_key": SCENE_PROMPT_KEY,
        **dict(prompt_defaults),
    }
    return generation_defaults, rendering_defaults, merged_prompt_defaults


__all__ = [
    "BACKGROUND_DEFAULTS",
    "POST_IMAGE_NOISE_DEFAULTS",
    "load_tangent_packing_defaults",
]
