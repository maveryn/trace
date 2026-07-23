"""Passive state records for marble-chain game tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

from PIL import Image


@dataclass(frozen=True)
class MarbleIntegerAxis:
    """Resolved integer sampling axis and probability trace."""

    value: int
    support: Tuple[int, ...]
    probabilities: Mapping[str, float]


@dataclass(frozen=True)
class MarbleSceneAxes:
    """Scene-level variants shared by all marble-chain objectives."""

    scene_variant: str
    scene_variant_probabilities: Mapping[str, float]
    style_variant: str
    style_variant_probabilities: Mapping[str, float]


@dataclass(frozen=True)
class MarbleOutcome:
    """Single insertion outcome for one candidate chain gap."""

    slot_index: int
    pop_count: int
    popped_indices: Tuple[int, ...]
    affected_indices: Tuple[int, ...]
    remaining_count: int


@dataclass(frozen=True)
class ShotOption:
    """One labeled shot direction shown in the scene."""

    label: str
    slot_index: int
    outcome: MarbleOutcome
    is_answer: bool

    @property
    def entity_id(self) -> str:
        return f"shot_option_{str(self.label).lower()}"


@dataclass(frozen=True)
class MarbleSample:
    """Task-owned marble-chain board plus answer witness ids."""

    scene_variant: str
    chain_colors: Tuple[str, ...]
    shooter_color: str
    answer: int | str
    answer_type: str
    option_specs: Tuple[ShotOption, ...]
    marked_slot_index: int | None
    marked_outcome: MarbleOutcome | None
    target_pop_count: int | None
    annotation_entity_ids: Tuple[str, ...]
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class RenderedMarbleScene:
    """Rendered marble-chain image plus entity and projection maps."""

    image: Image.Image
    entities: Tuple[Dict[str, Any], ...]
    render_map: Dict[str, Any]
    style_meta: Dict[str, Any]
    background_meta: Dict[str, Any]


__all__ = [
    "MarbleIntegerAxis",
    "MarbleOutcome",
    "MarbleSample",
    "MarbleSceneAxes",
    "RenderedMarbleScene",
    "ShotOption",
]
