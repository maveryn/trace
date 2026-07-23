"""Scene-level constants and semantic helpers for heatmap charts."""

from __future__ import annotations

from typing import Mapping, Tuple

from .....core.scene_config import get_scene_defaults
from ....shared.config_defaults import split_scene_generation_rendering_prompt_defaults
from ...shared.visual_defaults import load_chart_scene_background_defaults, load_chart_scene_noise_defaults


DOMAIN = "charts"
SCENE_ID = "heatmap"
SCENE_NAMESPACE = "charts_heatmap_base"
SUPPORTED_SCENE_VARIANTS: Tuple[str, ...] = (
    "intensity_heatmap",
    "signed_change_heatmap",
    "calendar_heatmap",
    "continuous_colorbar_heatmap",
)
SUPPORTED_EXTREMUM_DIRECTIONS: Tuple[str, ...] = ("hottest", "coolest")
SUPPORTED_QUERY_AXES: Tuple[str, ...] = ("row", "column")
_INTENSITY_CONDITIONS: Tuple[str, ...] = ("hot", "cool")
_SIGNED_CONDITIONS: Tuple[str, ...] = ("increase", "decrease")

_TASK_GROUP_DEFAULTS = get_scene_defaults("charts", "heatmap")
GEN_DEFAULTS, RENDER_DEFAULTS, PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    _TASK_GROUP_DEFAULTS if isinstance(_TASK_GROUP_DEFAULTS, Mapping) else {},
)
POST_IMAGE_BACKGROUND_DEFAULTS = load_chart_scene_background_defaults(scene_id="heatmap")
POST_IMAGE_NOISE_DEFAULTS = load_chart_scene_noise_defaults(scene_id="heatmap", apply_prob=0.0)

_MISSING_CONDITION_PHRASES: Tuple[str, ...] = (
    "purple-coded",
    "black-striped",
    "teal-outlined",
    "orange-dotted",
)
_WEEKDAY_LABELS: Tuple[str, ...] = ("Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat")
_TITLE_OPTIONS: Tuple[str, ...] = (
    "Activity Heatmap",
    "Change Intensity Grid",
    "Category Heat Matrix",
    "Weekly Signal Heatmap",
    "Response Pattern Grid",
)
_INTENSITY_PALETTE: Tuple[Tuple[int, int, int], ...] = (
    (247, 252, 245),
    (199, 233, 192),
    (116, 196, 118),
    (49, 163, 84),
    (0, 109, 44),
)
_SIGNED_PALETTE: Tuple[Tuple[int, int, int], ...] = (
    (49, 117, 182),
    (171, 217, 233),
    (244, 244, 244),
    (253, 174, 97),
    (215, 48, 39),
)
_CALENDAR_PALETTE: Tuple[Tuple[int, int, int], ...] = (
    (246, 245, 252),
    (218, 218, 235),
    (188, 189, 220),
    (128, 125, 186),
    (84, 39, 143),
)
BBox = Tuple[float, float, float, float]


def _condition_support(scene_variant: str) -> Tuple[str, ...]:
    if str(scene_variant) == "signed_change_heatmap":
        return _SIGNED_CONDITIONS
    return _INTENSITY_CONDITIONS

def _condition_matches(value: int, *, condition_kind: str, bin_count: int) -> bool:
    if str(condition_kind) == "hot":
        return int(value) == max(0, int(bin_count) - 1)
    if str(condition_kind) == "cool":
        return int(value) == 0
    if str(condition_kind) == "increase":
        return int(value) == max(0, int(bin_count) - 1)
    if str(condition_kind) == "decrease":
        return int(value) == 0
    raise ValueError(f"unsupported condition_kind: {condition_kind}")


def _condition_phrase(condition_kind: str, *, scene_variant: str) -> str:
    if str(scene_variant) == "calendar_heatmap":
        phrases = {
            "hot": "the highest activity level",
            "cool": "the lowest activity level",
        }
    else:
        phrases = {
            "hot": "the highest intensity level",
            "cool": "the lowest intensity level",
            "increase": "the strongest increase color",
            "decrease": "the strongest decrease color",
        }
    return str(phrases[str(condition_kind)])


def _extremum_phrase(extremum_direction: str, *, scene_variant: str) -> str:
    if str(scene_variant) == "signed_change_heatmap":
        if str(extremum_direction) == "hottest":
            return "strongest increase-colored"
        if str(extremum_direction) == "coolest":
            return "strongest decrease-colored"
    if str(scene_variant) == "calendar_heatmap":
        if str(extremum_direction) == "hottest":
            return "highest-activity"
        if str(extremum_direction) == "coolest":
            return "lowest-activity"
    if str(extremum_direction) == "hottest":
        return "highest-intensity"
    if str(extremum_direction) == "coolest":
        return "lowest-intensity"
    raise ValueError(f"unsupported extremum_direction: {extremum_direction}")
