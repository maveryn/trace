"""Default ranges for environment illustration count tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class CountContractDefaults:
    """Default sampler/render ranges for one environment count contract."""

    object_count_min: int
    object_count_max: int
    target_count_min: int
    target_count_max: int
    canvas_width: int = 1280
    canvas_height: int = 840
    object_size_min_px: int = 62
    object_size_max_px: int = 116
    min_gap_px: int = 6
    max_overlap_fraction: float = 0.02
    placement_max_attempts: int = 420
    render_scale: int = 2
    skyline_building_min: int = 7
    skyline_building_max: int = 14


FEATURE_RELATION_SIDE_DEFAULTS = CountContractDefaults(
    object_count_min=12,
    object_count_max=18,
    target_count_min=1,
    target_count_max=6,
)
FEATURE_RELATION_ON_DEFAULTS = CountContractDefaults(
    object_count_min=12,
    object_count_max=18,
    target_count_min=1,
    target_count_max=6,
)
CROSSING_DEFAULTS = CountContractDefaults(
    object_count_min=12,
    object_count_max=18,
    target_count_min=1,
    target_count_max=5,
)
LIT_WINDOW_DEFAULTS = CountContractDefaults(
    object_count_min=8,
    object_count_max=14,
    target_count_min=1,
    target_count_max=6,
    object_size_min_px=58,
    object_size_max_px=108,
    skyline_building_min=4,
    skyline_building_max=7,
)


def render_fallback(defaults: CountContractDefaults) -> Dict[str, Any]:
    """Return render fallback values consumed by the shared lifecycle."""

    return {
        "canvas_width": int(defaults.canvas_width),
        "canvas_height": int(defaults.canvas_height),
        "object_size_min_px": int(defaults.object_size_min_px),
        "object_size_max_px": int(defaults.object_size_max_px),
        "min_gap_px": int(defaults.min_gap_px),
        "max_overlap_fraction": float(defaults.max_overlap_fraction),
        "placement_max_attempts": int(defaults.placement_max_attempts),
        "render_scale": int(defaults.render_scale),
        "skyline_building_min": int(defaults.skyline_building_min),
        "skyline_building_max": int(defaults.skyline_building_max),
    }


__all__ = [
    "CROSSING_DEFAULTS",
    "FEATURE_RELATION_ON_DEFAULTS",
    "FEATURE_RELATION_SIDE_DEFAULTS",
    "LIT_WINDOW_DEFAULTS",
    "CountContractDefaults",
    "render_fallback",
]
