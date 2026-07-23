"""Passive state records for pressure-volume diagram scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image


SCENE_ID = "pv_diagram"
SCENE_NAMESPACE = "physics_pv_diagram"
SUPPORTED_SCENE_VARIANTS: tuple[str, ...] = (
    "clean_grid",
    "paper_grid",
    "bold_grid",
)
SUPPORTED_WORK_MODES: tuple[str, ...] = (
    "single_process",
    "rectangular_cycle",
)
SUPPORTED_TARGET_SIGNS: tuple[str, ...] = ("positive", "negative", "zero")
OPTION_LETTERS: tuple[str, ...] = ("A", "B", "C", "D", "E", "F", "G", "H")


@dataclass(frozen=True)
class PVDiagramTaskDefaults:
    """Stable fallback defaults for PV-diagram scenes."""

    canvas_width: int = 1180
    canvas_height: int = 760
    plot_left_px: int = 118
    plot_top_px: int = 78
    plot_width_px: int = 760
    plot_height_px: int = 560
    mini_plot_left_px: int = 58
    mini_plot_top_px: int = 88
    mini_cell_width_px: int = 262
    mini_cell_height_px: int = 196
    mini_cell_gap_x_px: int = 18
    mini_cell_gap_y_px: int = 32
    axis_width_px: int = 5
    grid_line_width_px: int = 1
    bold_grid_line_width_px: int = 2
    process_line_width_px: int = 9
    cycle_line_width_px: int = 8
    arrow_head_length_px: int = 24
    arrow_head_width_px: int = 22
    label_font_size_px: int = 24
    tick_font_size_px: int = 18
    state_font_size_px: int = 25
    option_font_size_px: int = 26
    note_font_size_px: int = 21
    label_stroke_width_px: int = 2
    pressure_max_kpa: int = 10
    volume_max_l: int = 12
    pressure_support: tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8, 9)
    volume_support: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
    min_volume_delta: int = 2
    max_volume_delta: int = 9
    work_answer_support: tuple[int, ...] = (
        -70,
        -64,
        -63,
        -60,
        -56,
        -54,
        -49,
        -48,
        -45,
        -42,
        -40,
        -36,
        -35,
        -32,
        -30,
        -28,
        -27,
        -25,
        -24,
        -21,
        -20,
        -18,
        -16,
        -15,
        -14,
        -12,
        -10,
        -9,
        -8,
        -7,
        -6,
        -5,
        -4,
        -3,
        -2,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        12,
        14,
        15,
        16,
        18,
        20,
        21,
        24,
        25,
        27,
        28,
        30,
        32,
        35,
        36,
        40,
        42,
        45,
        48,
        49,
        54,
        56,
        60,
        63,
        64,
        70,
    )


@dataclass(frozen=True)
class PVDiagramWorkAxes:
    """Resolved sampling axes for numeric PV-work scenes."""

    scene_variant: str
    work_mode: str
    accent_color_name: str
    target_answer: int
    scene_variant_probabilities: dict[str, float]
    work_mode_probabilities: dict[str, float]
    accent_color_name_probabilities: dict[str, float]
    target_answer_probabilities: dict[str, float]


@dataclass(frozen=True)
class PVDiagramSignChoiceAxes:
    """Resolved sampling axes for PV work-sign option scenes."""

    scene_variant: str
    target_sign: str
    correct_option_letter: str
    accent_color_name: str
    target_answer: str
    scene_variant_probabilities: dict[str, float]
    target_sign_probabilities: dict[str, float]
    correct_option_letter_probabilities: dict[str, float]
    accent_color_name_probabilities: dict[str, float]
    target_answer_probabilities: dict[str, float]


@dataclass(frozen=True)
class PVWorkScenario:
    """One symbolic PV-work scenario."""

    work_mode: str
    work_value: int
    pressure: int | None
    volume_start: int | None
    volume_end: int | None
    pressure_low: int | None
    pressure_high: int | None
    volume_left: int | None
    volume_right: int | None
    cycle_direction: str | None


@dataclass(frozen=True)
class PVProcessCandidate:
    """One labeled candidate PV process for a sign-choice panel."""

    letter: str
    sign: str
    pressure_start: int
    pressure_end: int
    volume_start: int
    volume_end: int


@dataclass(frozen=True)
class PVDiagramSceneSpec:
    """Resolved symbolic PV scene."""

    scene_variant: str
    diagram_kind: str
    work_mode: str | None
    target_sign: str | None
    correct_option_letter: str | None
    target_answer: int | str
    work_scenario: PVWorkScenario | None
    process_candidates: tuple[PVProcessCandidate, ...]
    annotation_entity_ids: tuple[str, ...]


@dataclass(frozen=True)
class RenderedPVDiagramScene:
    """Rendered PV diagram plus prompt-facing annotation metadata."""

    image: Image.Image
    annotation_bboxes: list[list[float]]
    annotation_entity_ids: list[str]
    scene_entities: list[dict[str, Any]]
    render_map: dict[str, Any]


__all__ = [
    "OPTION_LETTERS",
    "PVDiagramSceneSpec",
    "PVDiagramSignChoiceAxes",
    "PVDiagramTaskDefaults",
    "PVDiagramWorkAxes",
    "PVProcessCandidate",
    "PVWorkScenario",
    "RenderedPVDiagramScene",
    "SCENE_ID",
    "SCENE_NAMESPACE",
    "SUPPORTED_SCENE_VARIANTS",
    "SUPPORTED_TARGET_SIGNS",
    "SUPPORTED_WORK_MODES",
]
