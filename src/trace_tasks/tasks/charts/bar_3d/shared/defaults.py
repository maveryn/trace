"""Scene-local defaults for the 3D bar-grid chart scene."""

from __future__ import annotations

from typing import Any, Sequence

from ....shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from ...shared.visual_defaults import load_chart_scene_background_defaults, load_chart_scene_noise_defaults
from .state import DOMAIN, RGB, SCENE_ID


PROMPT_BUNDLE_ID = "charts_bar_3d_v1"

_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    DOMAIN,
    SCENE_ID,
)
POST_IMAGE_BACKGROUND_DEFAULTS = load_chart_scene_background_defaults(scene_id=SCENE_ID)
POST_IMAGE_NOISE_DEFAULTS = load_chart_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.0)

_DEFAULT_PALETTE: tuple[RGB, ...] = (
    (62, 121, 190),
    (219, 117, 68),
    (73, 157, 99),
    (151, 100, 186),
    (206, 166, 63),
    (73, 158, 176),
)


def _as_rgb(value: Any, fallback: RGB) -> RGB:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 3:
        return tuple(int(channel) for channel in fallback)
    return tuple(max(0, min(255, int(channel))) for channel in value[:3])  # type: ignore[index]
