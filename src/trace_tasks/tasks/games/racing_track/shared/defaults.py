"""Scene-level defaults and supported visual axes for racing-track tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


SCENE_ID = "racing_track"

SUPPORTED_SCENE_VARIANTS: Tuple[str, ...] = (
    "oval_loop",
    "rounded_loop",
    "kidney_loop",
)
SUPPORTED_STYLE_VARIANTS: Tuple[str, ...] = (
    "asphalt_day",
    "rally_sand",
    "neon_night",
    "blueprint_track",
    "paper_race",
)
CAR_LABELS: Tuple[str, ...] = ("A", "B", "C", "D", "E", "F", "G")


@dataclass(frozen=True)
class RacingTrackSceneDefaults:
    """Stable fallback defaults for visible racing-track scenes."""

    canvas_width: int = 1000
    canvas_height: int = 760
    track_width_px: int = 820
    track_height_px: int = 600
    road_width_px: int = 72
    road_border_width_px: int = 8
    car_length_px: int = 48
    car_width_px: int = 28
    marked_outline_width_px: int = 5
    label_font_size_px: int = 24
    min_progress_gap: float = 0.045


DEFAULTS = RacingTrackSceneDefaults()


__all__ = [
    "CAR_LABELS",
    "DEFAULTS",
    "RacingTrackSceneDefaults",
    "SCENE_ID",
    "SUPPORTED_SCENE_VARIANTS",
    "SUPPORTED_STYLE_VARIANTS",
]
