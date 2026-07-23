"""Passive dominoes scene state for games-domain tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class DominoTileInstance:
    """One visible domino tile before rendering."""

    tile_id: str
    left_value: int
    right_value: int
    role: str
    is_reference: bool = False
    highlight_right_half: bool = False
    option_label: str | None = None
    right_join_label: str | None = None


@dataclass(frozen=True)
class DominoSceneAxes:
    """Resolved scene/style axes shared by dominoes tasks."""

    scene_variant: str
    style_variant: str
    scene_variant_probabilities: Dict[str, float]
    style_variant_probabilities: Dict[str, float]


@dataclass(frozen=True)
class DominoIntegerAxis:
    """Resolved integer sampling axis with trace metadata."""

    value: int
    support: Tuple[int, ...]
    probabilities: Dict[str, float]


@dataclass(frozen=True)
class SampledDominoScene:
    """One sampled domino-chain scene plus task-owned witness metadata."""

    chain_tiles: Tuple[DominoTileInstance, ...]
    candidate_tiles: Tuple[DominoTileInstance, ...]
    annotation_tile_ids: Tuple[str, ...]
    answer_value: int | str
    reference_tile_id: str | None
    open_end_value: int | None
    reference_sum: int | None
    target_total: int | None
    first_step_tile_id: str | None
    second_step_tile_id: str | None
    bridge_value: int | None
    chain_tile_specs: Tuple[Dict[str, Any], ...]
    candidate_tile_specs: Tuple[Dict[str, Any], ...]


__all__ = [
    "DominoIntegerAxis",
    "DominoSceneAxes",
    "DominoTileInstance",
    "SampledDominoScene",
]
