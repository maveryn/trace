"""Shared source-relative patch sizing for illustration missing-patch tasks."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Mapping, Sequence, Tuple

from ...shared.config_defaults import group_default


DEFAULT_PATCH_WIDTH_RATIO_MIN = 0.15
DEFAULT_PATCH_WIDTH_RATIO_MAX = 0.30
DEFAULT_PATCH_HEIGHT_RATIO_MIN = 0.15
DEFAULT_PATCH_HEIGHT_RATIO_MAX = 0.26
DEFAULT_PATCH_AREA_RATIO_MAX = 0.065


@dataclass(frozen=True)
class MissingPatchSizeSample:
    """Resolved missing-patch size and ratio diagnostics."""

    patch_size: Tuple[int, int]
    width_ratio: float
    height_ratio: float
    area_ratio: float
    width_ratio_range: Tuple[float, float]
    height_ratio_range: Tuple[float, float]
    area_ratio_max: float

    def trace(self) -> Dict[str, Any]:
        return {
            "width_ratio": round(float(self.width_ratio), 6),
            "height_ratio": round(float(self.height_ratio), 6),
            "area_ratio": round(float(self.area_ratio), 6),
            "width_ratio_range": [round(float(value), 6) for value in self.width_ratio_range],
            "height_ratio_range": [round(float(value), 6) for value in self.height_ratio_range],
            "area_ratio_max": round(float(self.area_ratio_max), 6),
        }


def _float_value(params: Mapping[str, Any], defaults: Mapping[str, Any], key: str, fallback: float) -> float:
    return float(params.get(str(key), group_default(defaults, str(key), float(fallback))))


def _ratio_range(
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    *,
    min_key: str,
    max_key: str,
    fallback_min: float,
    fallback_max: float,
) -> Tuple[float, float]:
    low = _float_value(params, defaults, str(min_key), float(fallback_min))
    high = _float_value(params, defaults, str(max_key), float(fallback_max))
    if low <= 0.0 or high <= 0.0 or low > high or high > 1.0:
        raise ValueError(f"invalid {min_key}/{max_key} ratio bounds")
    return (float(low), float(high))


def sample_missing_patch_size(
    *,
    rng: Any,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    source_size: Sequence[int],
    fallback_width_ratio_min: float = DEFAULT_PATCH_WIDTH_RATIO_MIN,
    fallback_width_ratio_max: float = DEFAULT_PATCH_WIDTH_RATIO_MAX,
    fallback_height_ratio_min: float = DEFAULT_PATCH_HEIGHT_RATIO_MIN,
    fallback_height_ratio_max: float = DEFAULT_PATCH_HEIGHT_RATIO_MAX,
    fallback_area_ratio_max: float = DEFAULT_PATCH_AREA_RATIO_MAX,
) -> MissingPatchSizeSample:
    """Sample a missing-patch size from source-image-relative ratio bounds."""

    source_w = int(source_size[0])
    source_h = int(source_size[1])
    if source_w <= 0 or source_h <= 0:
        raise ValueError("source_size must contain positive width and height")

    width_range = _ratio_range(
        params,
        defaults,
        min_key="patch_width_ratio_min",
        max_key="patch_width_ratio_max",
        fallback_min=float(fallback_width_ratio_min),
        fallback_max=float(fallback_width_ratio_max),
    )
    height_range = _ratio_range(
        params,
        defaults,
        min_key="patch_height_ratio_min",
        max_key="patch_height_ratio_max",
        fallback_min=float(fallback_height_ratio_min),
        fallback_max=float(fallback_height_ratio_max),
    )
    area_ratio_max = _float_value(params, defaults, "patch_area_ratio_max", float(fallback_area_ratio_max))
    if area_ratio_max <= 0.0 or area_ratio_max > 1.0:
        raise ValueError("patch_area_ratio_max must be in (0, 1]")
    if width_range[0] * height_range[0] > area_ratio_max:
        raise ValueError("minimum patch width/height ratios exceed patch_area_ratio_max")

    width_min = max(1, int(math.ceil(width_range[0] * source_w)))
    width_max = min(source_w, int(math.floor(width_range[1] * source_w)))
    height_min = max(1, int(math.ceil(height_range[0] * source_h)))
    height_ratio_max_px = min(source_h, int(math.floor(height_range[1] * source_h)))
    if width_min > width_max or height_min > height_ratio_max_px:
        raise ValueError("patch ratio bounds leave no feasible integer patch size")

    patch_w = int(rng.randint(width_min, width_max))
    area_height_cap = int(math.floor((float(area_ratio_max) * float(source_w) * float(source_h)) / float(patch_w)))
    height_max = min(height_ratio_max_px, area_height_cap)
    if height_max < height_min:
        raise ValueError("sampled patch width leaves no feasible height under patch_area_ratio_max")
    patch_h = int(rng.randint(height_min, height_max))

    return MissingPatchSizeSample(
        patch_size=(int(patch_w), int(patch_h)),
        width_ratio=float(patch_w) / float(source_w),
        height_ratio=float(patch_h) / float(source_h),
        area_ratio=(float(patch_w) * float(patch_h)) / (float(source_w) * float(source_h)),
        width_ratio_range=width_range,
        height_ratio_range=height_range,
        area_ratio_max=float(area_ratio_max),
    )


__all__ = [
    "DEFAULT_PATCH_AREA_RATIO_MAX",
    "DEFAULT_PATCH_HEIGHT_RATIO_MAX",
    "DEFAULT_PATCH_HEIGHT_RATIO_MIN",
    "DEFAULT_PATCH_WIDTH_RATIO_MAX",
    "DEFAULT_PATCH_WIDTH_RATIO_MIN",
    "MissingPatchSizeSample",
    "sample_missing_patch_size",
]
