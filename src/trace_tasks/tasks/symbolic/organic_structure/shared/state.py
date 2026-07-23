"""Passive state objects for organic-structure notation scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

SCENE_ID = "organic_structure"

BOND_ORDER_VALUES: Mapping[str, int] = {"single": 1, "double": 2, "triple": 3}
SUPPORTED_BOND_ORDERS: Tuple[str, ...] = ("single", "double", "triple")
SUPPORTED_ORGANIC_RING_SIZES: Tuple[int, ...] = (5, 6)
ORGANIC_STRUCTURE_MAX_BOND_ORDER_COUNT = 5
ORGANIC_STRUCTURE_MAX_RING_SIZE_COUNT = 5
SCENE_VARIANTS: Tuple[str, ...] = ("clean_worksheet", "exam_scan", "notebook_problem")


@dataclass(frozen=True)
class OrganicAtom:
    item_id: str
    x: float
    y: float
    element: str = "C"
    implicit: bool = True


@dataclass(frozen=True)
class OrganicTextLabel:
    item_id: str
    text: str
    x: float
    y: float
    role: str = "substituent_label"
    anchor_atom: int | None = None


@dataclass(frozen=True)
class OrganicBond:
    item_id: str
    atom_a: int
    atom_b: int
    order: str
    role: str = "backbone"
    ring_index: int | None = None


@dataclass(frozen=True)
class OrganicStructureSpec:
    atoms: Tuple[OrganicAtom, ...]
    bonds: Tuple[OrganicBond, ...]
    ring_atom_sets: Tuple[Tuple[int, ...], ...]
    scaffold_id: str
    scaffold_family: str
    target_bond_order: str
    target_answer_value: int
    constraint_policy: str
    text_labels: Tuple[OrganicTextLabel, ...] = tuple()


@dataclass(frozen=True)
class OrganicConstraintReport:
    valence_by_atom_id: Dict[str, int]
    max_valence: int
    branch_point_atom_ids: Tuple[str, ...]
    min_branch_angle_degrees: float | None
    triple_linear_atom_ids: Tuple[str, ...]
    ring_sizes: Tuple[int, ...]
    crossing_count: int

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "valence_by_atom_id": dict(self.valence_by_atom_id),
            "max_valence": int(self.max_valence),
            "branch_point_atom_ids": list(self.branch_point_atom_ids),
            "min_branch_angle_degrees": None
            if self.min_branch_angle_degrees is None
            else round(float(self.min_branch_angle_degrees), 3),
            "triple_linear_atom_ids": list(self.triple_linear_atom_ids),
            "ring_sizes": [int(value) for value in self.ring_sizes],
            "crossing_count": int(self.crossing_count),
        }


@dataclass(frozen=True)
class OrganicProjection:
    atom_points_px: Tuple[Tuple[float, float], ...]
    metadata: Dict[str, Any]
    text_label_points_px: Tuple[Tuple[float, float], ...] = tuple()


@dataclass(frozen=True)
class OrganicRenderedStructure:
    entities: Tuple[Dict[str, Any], ...]
    item_bboxes: Dict[str, Tuple[float, float, float, float]]
    item_segments: Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]]
    item_points: Dict[str, Tuple[float, float]]
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class OrganicRenderParams:
    canvas_width: int
    canvas_height: int
    panel_padding_px: int
    panel_corner_radius_px: int
    panel_border_width_px: int
    bond_width_px: int
    bond_gap_px: int
    structure_width_px: int
    structure_height_px: int
    unit_size_jitter: Dict[str, Any]
    panel_fill_rgb: Tuple[int, int, int]
    panel_border_rgb: Tuple[int, int, int]
    bond_rgb: Tuple[int, int, int]
    annotation_rgb: Tuple[int, int, int]


@dataclass(frozen=True)
class RenderedOrganicScene:
    image: Any
    entities: Tuple[Dict[str, Any], ...]
    scene_bbox_px: Tuple[float, float, float, float]
    item_bboxes: Dict[str, Tuple[float, float, float, float]]
    item_segments: Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]]
    item_points: Dict[str, Tuple[float, float]]
    layout_jitter: Dict[str, Any]
    style_metadata: Dict[str, Any]


__all__ = [
    "BOND_ORDER_VALUES",
    "ORGANIC_STRUCTURE_MAX_BOND_ORDER_COUNT",
    "ORGANIC_STRUCTURE_MAX_RING_SIZE_COUNT",
    "OrganicAtom",
    "OrganicBond",
    "OrganicConstraintReport",
    "OrganicProjection",
    "OrganicRenderedStructure",
    "OrganicRenderParams",
    "OrganicStructureSpec",
    "OrganicTextLabel",
    "RenderedOrganicScene",
    "SCENE_ID",
    "SCENE_VARIANTS",
    "SUPPORTED_BOND_ORDERS",
    "SUPPORTED_ORGANIC_RING_SIZES",
]
