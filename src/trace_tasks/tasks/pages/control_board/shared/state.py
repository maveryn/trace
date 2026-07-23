"""State containers for the pages control-board scene."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from PIL import Image


BBox = Tuple[float, float, float, float]
Color = Tuple[int, int, int]


@dataclass(frozen=True)
class ControlBoardDefaults:
    """Stable fallback defaults for grouped GUI control boards."""

    canvas_width: int = 1280
    canvas_height: int = 800
    window_margin_px: int = 42
    title_bar_height_px: int = 46
    menu_bar_height_px: int = 34
    corner_radius_px: int = 16
    control_corner_radius_px: int = 8
    control_outline_width_px: int = 2
    badge_size_px: int = 28
    title_font_size_px: int = 24
    body_font_size_px: int = 17
    small_font_size_px: int = 13
    label_font_size_px: int = 18
    group_name_pool: Tuple[str, ...] = ("Layout", "Editing", "Review", "Output")
    candidate_label_pool: Tuple[str, ...] = tuple(chr(ord("A") + idx) for idx in range(26))


@dataclass(frozen=True)
class ControlBoardRenderParams:
    """Pixel-space render parameters resolved from scene defaults."""

    canvas_width: int
    canvas_height: int
    window_margin_px: int
    title_bar_height_px: int
    menu_bar_height_px: int
    corner_radius_px: int
    control_corner_radius_px: int
    control_outline_width_px: int
    badge_size_px: int
    title_font_size_px: int
    body_font_size_px: int
    small_font_size_px: int
    label_font_size_px: int


@dataclass(frozen=True)
class ControlBoardTheme:
    """Color palette for one control-board style variant."""

    name: str
    app_fill: Color
    title_bar: Color
    title_text: Color
    chrome_line: Color
    panel_fill: Color
    panel_alt_fill: Color
    control_fill: Color
    control_outline: Color
    control_text: Color
    muted_text: Color
    disabled_fill: Color
    disabled_outline: Color
    selected_fill: Color
    selected_outline: Color
    accent: Color
    accent_alt: Color
    badge_fill: Color
    badge_text: Color
    workspace_line: Color


@dataclass(frozen=True)
class ControlBoardProfile:
    """Application chrome text and workspace labels for one scene variant."""

    app_title: str
    window_title: str
    primary_tab: str
    secondary_tab: str
    workspace_title: str
    status_text: str


@dataclass(frozen=True)
class CommandOption:
    """One visible command tile inside a grouped control board."""

    command_key: str
    display_text: str
    icon_kind: str


@dataclass(frozen=True)
class ControlSpec:
    """Semantic state for one rendered control tile."""

    control_id: str
    candidate_label: str
    group_name: str
    group_index: int
    order_in_group: int
    global_order_index: int
    command: CommandOption
    enabled: bool
    selected: bool
    is_reference: bool


@dataclass(frozen=True)
class ControlBoardCase:
    """Sampled scene state before projection to pixels."""

    count_mode: str
    scene_variant: str
    controls: Tuple[ControlSpec, ...]
    group_names: Tuple[str, ...]
    target_group_name: str
    target_group_index: int
    answer_value: int
    annotation_control_ids: Tuple[str, ...]
    answer_support: Tuple[int, ...]
    candidate_label_pool: Tuple[str, ...]
    scene_variant_probabilities: Dict[str, float]


@dataclass(frozen=True)
class RenderedControlBoard:
    """Rendered image plus pixel bboxes and render metadata."""

    image: Image.Image
    render_params: ControlBoardRenderParams
    control_bboxes_by_id: Dict[str, List[float]]
    badge_bboxes_by_id: Dict[str, List[float]]
    group_bboxes_by_name: Dict[str, List[float]]
    control_records: Tuple[Dict[str, Any], ...]
    scene_bbox_px: List[float]
    window_bbox_px: List[float]
    profile: ControlBoardProfile
    theme: ControlBoardTheme
    background_meta: Dict[str, Any]
    post_noise_meta: Dict[str, Any]
