"""Passive state objects for symbolic chemical-equation scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Tuple

SCENE_ID = "chemical_equation"
SCENE_KIND = "symbolic_chemical_equation"
SCENE_VARIANTS: Tuple[str, ...] = ("clean_lab", "worksheet", "notebook_scan")
OPTION_LABELS: Tuple[str, ...] = ("A", "B", "C", "D")
COEFFICIENT_SUPPORT: Tuple[int, ...] = (1, 2, 3, 4, 5)


@dataclass(frozen=True)
class ChemicalTermDef:
    formula: str
    atom_counts: Mapping[str, int]
    element_order: Tuple[str, ...]


@dataclass(frozen=True)
class ChemicalReactionDef:
    reaction_id: str
    left_formulas: Tuple[str, ...]
    right_formulas: Tuple[str, ...]
    coefficients: Tuple[int, ...]
    family: str


@dataclass(frozen=True)
class ChemicalTermSpec:
    item_id: str
    coefficient_slot_id: str
    molecule_card_id: str
    side: str
    side_index: int
    term_index: int
    formula: str
    coefficient: int
    hidden_coefficient: bool
    atom_counts: Mapping[str, int]
    element_order: Tuple[str, ...]


@dataclass(frozen=True)
class ChemicalOptionSpec:
    item_id: str
    label: str
    coefficients: Tuple[int, ...]
    balances_equation: bool


@dataclass(frozen=True)
class ChemicalEquationDataset:
    task_kind: str
    scene_variant: str
    scene_variant_probabilities: Mapping[str, float]
    reaction: ChemicalReactionDef
    terms: Tuple[ChemicalTermSpec, ...]
    hidden_slot_index: int | None
    answer_value: int | str
    target_answer_support: Tuple[int | str, ...]
    target_answer_probabilities: Mapping[str, float]
    options: Tuple[ChemicalOptionSpec, ...] = tuple()
    correct_option_label: str = ""
    correct_option_label_probabilities: Mapping[str, float] | None = None
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class ChemicalEquationRenderParams:
    canvas_width: int
    canvas_height: int
    panel_left_px: int
    panel_top_px: int
    panel_width_px: int
    panel_height_px: int
    coefficient_box_width_px: int
    coefficient_box_height_px: int
    molecule_card_width_px: int
    molecule_card_height_px: int
    term_gap_px: int
    operator_gap_px: int
    card_corner_radius_px: int
    card_border_width_px: int
    coefficient_font_size_px: int
    atom_font_size_px: int
    option_label_font_size_px: int
    option_text_font_size_px: int
    operator_font_size_px: int
    atom_chip_diameter_px: int


@dataclass(frozen=True)
class RenderedChemicalEquationScene:
    image: Any
    entities: Tuple[dict[str, Any], ...]
    scene_bbox_px: Tuple[float, float, float, float]
    item_bboxes: Mapping[str, Tuple[float, float, float, float]]
    coefficient_slot_bboxes: Mapping[str, Tuple[float, float, float, float]]
    molecule_card_bboxes: Mapping[str, Tuple[float, float, float, float]]
    option_bboxes: Mapping[str, Tuple[float, float, float, float]]
    atom_chip_bboxes: Mapping[str, Tuple[float, float, float, float]]
    style_metadata: Mapping[str, Any]


__all__ = [
    "COEFFICIENT_SUPPORT",
    "ChemicalEquationDataset",
    "ChemicalEquationRenderParams",
    "ChemicalOptionSpec",
    "ChemicalReactionDef",
    "ChemicalTermDef",
    "ChemicalTermSpec",
    "OPTION_LABELS",
    "RenderedChemicalEquationScene",
    "SCENE_ID",
    "SCENE_KIND",
    "SCENE_VARIANTS",
]
