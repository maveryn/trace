"""Passive state records for match-3 scene generation and rendering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

from PIL import Image


Coord = Tuple[int, int]
Board = Tuple[Tuple[str, ...], ...]


@dataclass(frozen=True)
class Match3IntegerAxis:
    """Resolved integer generation axis with support and probabilities."""

    value: int
    support: Tuple[int, ...]
    probabilities: Dict[str, float]


@dataclass(frozen=True)
class Match3SceneAxes:
    """Scene/style axes sampled independently of the public objective."""

    scene_variant: str
    scene_variant_probabilities: Dict[str, float]
    style_variant: str
    style_variant_probabilities: Dict[str, float]


@dataclass(frozen=True)
class Match3BoardSpec:
    """Constructed board plus generation metadata for replay and balancing."""

    board: Board
    gem_keys: Tuple[str, ...]
    rows: int
    cols: int
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class SwapMove:
    """One adjacent gem swap expressed in zero-based board coordinates."""

    a: Coord
    b: Coord

    @property
    def key(self) -> Tuple[Coord, Coord]:
        """Return a stable undirected key for de-duplicating adjacent swaps."""

        return (tuple(self.a), tuple(self.b)) if tuple(self.a) <= tuple(self.b) else (tuple(self.b), tuple(self.a))


@dataclass(frozen=True)
class MoveOutcome:
    """Immediate post-swap clear outcome without gravity, refill, or cascades."""

    move: SwapMove
    clear_count: int
    run_count: int
    cleared_cells: Tuple[Coord, ...]
    runs: Tuple[Tuple[Coord, ...], ...]


@dataclass(frozen=True)
class SwapOption:
    """One visible labeled arrow option tied to a swap outcome."""

    label: str
    outcome: MoveOutcome
    is_answer: bool

    @property
    def entity_id(self) -> str:
        """Return the rendered entity id for this option arrow."""

        return f"swap_option_{str(self.label).lower()}"


@dataclass(frozen=True)
class Match3Sample:
    """Task-owned board state and witness bindings for one match-3 instance."""

    scene_variant: str
    board: Board
    answer: int | str
    answer_type: str
    option_specs: Tuple[SwapOption, ...] = ()
    annotation_entity_ids: Tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RenderedMatch3Scene:
    """Rendered image and projection maps for task-owned annotation binding."""

    image: Image.Image
    entities: Tuple[Dict[str, Any], ...]
    render_map: Dict[str, Any]
    style_meta: Dict[str, Any]
    background_meta: Dict[str, Any]


__all__ = [
    "Board",
    "Coord",
    "Match3BoardSpec",
    "Match3IntegerAxis",
    "Match3Sample",
    "Match3SceneAxes",
    "MoveOutcome",
    "RenderedMatch3Scene",
    "SwapMove",
    "SwapOption",
]
