"""State objects and constants for graph option-panel scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Tuple

from PIL import Image


SCENE_ID = "graph_options"

SUPPORTED_SCENE_VARIANTS: Tuple[str, ...] = (
    "clean_graph_options",
    "colored_node_graph_options",
    "notebook_graph_options",
)
SUPPORTED_EDGE_MODES: Tuple[str, ...] = ("undirected", "directed")
LABEL_POOL: Tuple[str, ...] = tuple("ABCDEFGHJKLMNPQRSTUVXYZ")


@dataclass(frozen=True)
class GraphOptionsDefaults:
    """Stable fallback generation defaults for graph option scenes."""

    option_count: int = 4
    same_structure_node_count_min: int = 4
    same_structure_node_count_max: int = 6
    node_count_min: int = 5
    node_count_max: int = 7
    extra_edge_min: int = 1
    extra_edge_max: int = 3
    subgraph_node_count_min: int = 3
    subgraph_node_count_max: int = 4


@dataclass(frozen=True)
class GraphOptionsRenderParams:
    """Resolved render constants for the option-panel scene."""

    canvas_width: int = 1260
    canvas_height: int = 920
    margin_x_px: int = 66
    margin_top_px: int = 46
    reference_panel_height_px: int = 300
    reference_to_options_gap_px: int = 34
    option_gap_px: int = 26
    option_row_gap_px: int = 24
    panel_padding_px: int = 24
    panel_corner_radius_px: int = 24
    border_width_px: int = 3
    title_font_size_px: int = 28
    option_label_font_size_px: int = 30
    node_radius_px: int = 22
    edge_width_px: int = 5
    panel_fill_rgb: Tuple[int, int, int] = (250, 251, 254)
    option_fill_rgb: Tuple[int, int, int] = (255, 255, 255)
    border_rgb: Tuple[int, int, int] = (86, 96, 112)
    edge_rgb: Tuple[int, int, int] = (72, 82, 98)
    node_fill_rgb: Tuple[int, int, int] = (246, 248, 252)
    node_outline_rgb: Tuple[int, int, int] = (63, 76, 94)
    text_rgb: Tuple[int, int, int] = (30, 34, 42)
    text_stroke_rgb: Tuple[int, int, int] = (255, 255, 255)
    notebook_line_rgb: Tuple[int, int, int] = (218, 225, 235)
    font_family: str = ""
    font_asset: Dict[str, Any] | None = None
    font_asset_version: str = ""
    font_exclusion_reason: str = "readout font pool; no scene-local exclusion"


@dataclass(frozen=True)
class GraphOptionsDataset:
    """Sampled semantic graph-option data before rendering."""

    panel_title: str
    source_structure_spec: Dict[str, Any]
    answer_structure_spec: Dict[str, Any]
    option_specs: List[Dict[str, Any]]
    answer_option_label: str
    correct_option_index: int
    correct_option_panel_id: str
    option_count: int
    node_count: int
    node_count_range: List[int]
    node_count_probabilities: Dict[str, float]
    edge_mode: str
    solver_trace: Dict[str, Any]
    pattern_node_count: int | None = None
    pattern_node_count_range: List[int] | None = None
    pattern_node_count_probability_map: Dict[str, float] | None = None
    subgraph_node_count: int | None = None
    subgraph_node_count_range: List[int] | None = None
    subgraph_node_count_probability_map: Dict[str, float] | None = None


@dataclass(frozen=True)
class GraphOptionsRenderedScene:
    """Rendered image and projected geometry for one graph option scene."""

    image: Image.Image
    entities: List[Dict[str, Any]]
    scene_bbox_px: List[float]
    bbox_map: Dict[str, List[float]]
    option_panel_bbox_map: Dict[str, List[float]]
    render_params: GraphOptionsRenderParams
    scene_variant: str
    edge_mode: str
    background_meta: Mapping[str, Any]
    post_noise_meta: Mapping[str, Any]
    scene_style_meta: Mapping[str, Any]


__all__ = [
    "LABEL_POOL",
    "SCENE_ID",
    "SUPPORTED_EDGE_MODES",
    "SUPPORTED_SCENE_VARIANTS",
    "GraphOptionsDataset",
    "GraphOptionsDefaults",
    "GraphOptionsRenderedScene",
    "GraphOptionsRenderParams",
]
