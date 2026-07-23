"""Scene defaults for uncertainty-band charts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.charts.shared.visual_defaults import load_chart_scene_noise_defaults
from trace_tasks.tasks.shared.config_defaults import group_default, split_scene_generation_rendering_prompt_defaults

from .state import DOMAIN, PROMPT_BUNDLE_ID, SCENE_ID


_SCENE_DEFAULTS = get_scene_defaults(DOMAIN, SCENE_ID)
GEN_DEFAULTS, RENDER_DEFAULTS, PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    _SCENE_DEFAULTS if isinstance(_SCENE_DEFAULTS, Mapping) else {},
    task_id="charts_uncertainty_band_scene",
)
POST_IMAGE_NOISE_DEFAULTS = load_chart_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.0)


def generation_default(key: str, fallback: Any = None) -> Any:
    """Return a generation default for this scene."""

    return group_default(GEN_DEFAULTS, str(key), fallback)


def rendering_default(key: str, fallback: Any = None) -> Any:
    """Return a rendering default for this scene."""

    return group_default(RENDER_DEFAULTS, str(key), fallback)


def prompt_bundle_id() -> str:
    """Return the configured prompt bundle id."""

    return str(group_default(PROMPT_DEFAULTS, "bundle_id", PROMPT_BUNDLE_ID))


__all__ = [
    "GEN_DEFAULTS",
    "POST_IMAGE_NOISE_DEFAULTS",
    "PROMPT_DEFAULTS",
    "RENDER_DEFAULTS",
    "generation_default",
    "prompt_bundle_id",
    "rendering_default",
]
