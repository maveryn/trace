"""Passive state containers for symbolic Life automaton rendering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from PIL import Image


@dataclass(frozen=True)
class LifeOptionSpec:
    """One labeled candidate future-grid option."""

    option_id: str
    label: str
    grid: Tuple[Tuple[int, ...], ...]
    is_correct: bool


@dataclass(frozen=True)
class LifeSceneSpec:
    """Rendered Life automaton scene inputs."""

    rows: int
    cols: int
    initial_grid: Tuple[Tuple[int, ...], ...]
    future_grid: Tuple[Tuple[int, ...], ...]
    option_specs: Tuple[LifeOptionSpec, ...] = tuple()
    target_cells: Tuple[Tuple[int, int], ...] = tuple()
    source_marker_label: str = ""


@dataclass(frozen=True)
class LifeRenderParams:
    """Resolved rendering parameters for one Life automaton instance."""

    canvas_width: int
    canvas_height: int
    cell_size_px: int
    grid_gap_px: int
    panel_padding_px: int
    panel_corner_radius_px: int
    panel_border_width_px: int
    grid_line_width_px: int
    option_card_width_px: int
    option_card_height_px: int
    option_gap_px: int
    option_grid_cell_px: int
    label_font_size_px: int
    small_font_size_px: int
    arrow_width_px: int
    unit_size_jitter: Dict[str, Any]
    layout_seed: int
    font_family: str


@dataclass(frozen=True)
class LifeBoardVisual:
    """Resolved non-semantic Life board style and palette."""

    board_style: str
    cell_palette_id: str
    dead_rgb: Tuple[int, int, int]
    alive_rgb: Tuple[int, int, int]
    grid_rgb: Tuple[int, int, int]
    edge_rgb: Tuple[int, int, int]
    mark_rgb: Tuple[int, int, int]
    accent_rgb: Tuple[int, int, int]
    board_style_probabilities: Dict[str, float]
    cell_palette_probabilities: Dict[str, float]


@dataclass(frozen=True)
class RenderedLifeScene:
    """Rendered image metadata for the Life automaton scene."""

    image: Image.Image
    scene_bbox_px: Tuple[int, int, int, int]
    item_bboxes: Dict[str, Tuple[int, int, int, int]]
    entities: Tuple[Dict[str, Any], ...]
    layout_jitter: Dict[str, Any]
    style_metadata: Dict[str, Any]


@dataclass(frozen=True)
class LifeRenderBundle:
    """Rendered image plus reusable trace metadata."""

    image: Image.Image
    rendered: RenderedLifeScene
    render_params: LifeRenderParams
    background_meta: Dict[str, Any]
    post_noise_meta: Dict[str, Any]
    board_style: str
    board_style_probabilities: Dict[str, float]
    cell_palette_id: str
    cell_palette_probabilities: Dict[str, float]
