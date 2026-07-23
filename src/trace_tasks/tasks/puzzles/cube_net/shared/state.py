"""Passive state, constants, and dataclasses for cube-net puzzles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


SCENE_ID = "cube_net"
DOMAIN = "puzzles"

SCENE_VARIANTS: Tuple[str, ...] = ("clean_net", "paper_model", "game_mat")
FACE_IDS: Tuple[str, ...] = ("U", "D", "F", "B", "L", "R")
OPPOSITE_FACE = {"U": "D", "D": "U", "F": "B", "B": "F", "L": "R", "R": "L"}
NORMAL_BY_FACE = {
    "U": (0, 1, 0),
    "D": (0, -1, 0),
    "F": (0, 0, 1),
    "B": (0, 0, -1),
    "L": (-1, 0, 0),
    "R": (1, 0, 0),
}
FACE_BY_NORMAL = {value: key for key, value in NORMAL_BY_FACE.items()}
NET_COORDS = {
    "B": (0, -1),
    "U": (0, 0),
    "L": (-1, 1),
    "F": (0, 1),
    "R": (1, 1),
    "D": (0, 2),
}
SIDE_OFFSETS = {
    "top": (0, -1),
    "right": (1, 0),
    "bottom": (0, 1),
    "left": (-1, 0),
}
FACE_LABEL_POOL: Tuple[str, ...] = tuple("JKLMNPQRSTUVWXYZ23456789")


@dataclass(frozen=True)
class CubeNetDefaults:
    """Stable code fallbacks for scene-level cube-net generation/rendering."""

    canvas_width: int = 1100
    face_relation_canvas_height: int = 760
    equivalent_net_canvas_height: int = 860
    option_count: int = 4
    net_cell_size_px: int = 86
    line_width_px: int = 3
    title_font_size_px: int = 22
    face_font_size_px: int = 31
    option_font_size_px: int = 28


DEFAULTS = CubeNetDefaults()


@dataclass(frozen=True)
class FaceOption:
    """One labeled face-answer option card."""

    option_label: str
    face_id: str
    face_label: str


@dataclass(frozen=True)
class FaceRelationDataset:
    """Concrete cube-net relation case sampled before task answer binding."""

    relation_kind: str
    face_labels: Dict[str, str]
    reference_face: str
    marked_side: str | None
    correct_face: str
    options: Tuple[FaceOption, ...]
    correct_option_label: str


@dataclass(frozen=True)
class NetEquivalenceOption:
    """One candidate colored net for whole-cube equivalence matching."""

    option_label: str
    face_color_names: Dict[str, str]
    equivalence_kind: str
    canonical_signature: Tuple[str, ...]


@dataclass(frozen=True)
class NetEquivalenceDataset:
    """Concrete colored-net equivalence case with exactly one matching option."""

    reference_face_color_names: Dict[str, str]
    reference_signature: Tuple[str, ...]
    options: Tuple[NetEquivalenceOption, ...]
    correct_option_label: str


__all__ = [
    "DEFAULTS",
    "DOMAIN",
    "FACE_BY_NORMAL",
    "FACE_IDS",
    "FACE_LABEL_POOL",
    "FaceOption",
    "FaceRelationDataset",
    "NetEquivalenceDataset",
    "NetEquivalenceOption",
    "NET_COORDS",
    "NORMAL_BY_FACE",
    "OPPOSITE_FACE",
    "SCENE_ID",
    "SCENE_VARIANTS",
    "SIDE_OFFSETS",
]
