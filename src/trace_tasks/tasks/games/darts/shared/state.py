"""Identity-free darts scene state for games-domain tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class DartsSceneAxes:
    """Resolved scene and visual axes shared by darts tasks."""

    scene_variant: str
    style_variant: str
    scene_variant_probabilities: Dict[str, float]
    style_variant_probabilities: Dict[str, float]


@dataclass(frozen=True)
class DartsIntegerAxis:
    """Resolved integer sampling axis with trace metadata."""

    value: int
    support: Tuple[int, ...]
    probabilities: Dict[str, float]


@dataclass(frozen=True)
class DartsScoreSlot:
    """One simplified scoring area on the board."""

    area_kind: str
    sector_value: int | None
    score: int


@dataclass(frozen=True)
class DartInstance:
    """One visible dart marker before rendering."""

    dart_id: str
    label: str | None
    area_kind: str
    sector_value: int | None
    score: int
    x_px: float
    y_px: float
    is_marked: bool


@dataclass(frozen=True)
class DartsSampledScene:
    """One sampled darts scene with task-owned witness metadata."""

    darts: Tuple[DartInstance, ...]
    annotation_dart_ids: Tuple[str, ...]
    total_score: int
    target_score: int | None = None


__all__ = [
    "DartInstance",
    "DartsIntegerAxis",
    "DartsSampledScene",
    "DartsSceneAxes",
    "DartsScoreSlot",
]
