"""Data containers for navigation-flow page scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


BBox = Tuple[float, float, float, float]
Color = Tuple[int, int, int]

MENU_SURFACE = "menu_path"
SIDEBAR_SURFACE = "sidebar_tree"
RIBBON_SURFACE = "ribbon_group"


@dataclass(frozen=True)
class NavigationFlowDefaults:
    """Stable fallback defaults for navigation-flow screens."""

    canvas_width: int = 1280
    canvas_height: int = 800
    window_margin_px: int = 42
    title_bar_height_px: int = 46
    menu_bar_height_px: int = 34
    corner_radius_px: int = 16
    control_corner_radius_px: int = 8
    control_outline_width_px: int = 2
    badge_size_px: int = 30
    title_font_size_px: int = 24
    body_font_size_px: int = 16
    small_font_size_px: int = 13
    label_font_size_px: int = 18
    candidate_label_pool: Tuple[str, ...] = tuple(chr(ord("A") + idx) for idx in range(26))
    nav_menu_pool: Tuple[str, ...] = ("File", "Edit", "View")
    nav_submenu_pool: Tuple[str, ...] = ("Arrange", "Inspect")
    nav_menu_group_pool: Tuple[str, ...] = ("Primary", "Advanced")
    nav_command_pool: Tuple[str, ...] = ("Align", "Duplicate", "Export", "Preview")
    nav_sidebar_section_pool: Tuple[str, ...] = ("Workspace", "Assets", "Settings", "Reports")
    nav_sidebar_group_pool: Tuple[str, ...] = ("Pinned", "Recent", "Shared")
    nav_sidebar_item_pool: Tuple[str, ...] = ("Overview", "Timeline", "Details")
    nav_ribbon_tab_pool: Tuple[str, ...] = ("Home", "Insert", "Review", "Analyze", "Share")
    nav_ribbon_group_pool: Tuple[str, ...] = ("Arrange", "Inspect", "Publish")
    nav_ribbon_command_pool: Tuple[str, ...] = ("Compare", "Filter", "Attach", "Sync")


@dataclass(frozen=True)
class RenderParams:
    """Resolved screen geometry and typography."""

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
class ControlSpec:
    """One visible candidate control in a navigation surface."""

    control_id: str
    candidate_label: str
    role: str
    display_text: str
    nav_kind: str
    path_keys: Tuple[str, ...]
    order_index: int


@dataclass(frozen=True)
class NavigationFlowCase:
    """Sampled navigation screen and target path."""

    navigation_surface: str
    scene_variant: str
    controls: Tuple[ControlSpec, ...]
    target_control_id: str
    target_label: str
    path_labels: Tuple[str, ...]
    path_display: str
    command_label: str
    menu_command_count: int
    menu_command_count_range: Tuple[int, int]
    ribbon_tab_count: int
    ribbon_tab_count_range: Tuple[int, int]
    ribbon_group_count: int
    ribbon_group_count_range: Tuple[int, int]
    ribbon_command_count: int
    ribbon_command_count_range: Tuple[int, int]
    candidate_label_pool: Tuple[str, ...]
    surface_probabilities: Dict[str, float]
    scene_variant_probabilities: Dict[str, float]


@dataclass(frozen=True)
class Theme:
    """Color theme for the synthetic app chrome."""

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
    selected_fill: Color
    accent: Color
    accent_alt: Color
    badge_fill: Color
    badge_text: Color


@dataclass(frozen=True)
class AppProfile:
    """Visible app labels used as scene dressing."""

    app_title: str
    window_title: str
    primary_tab: str
    secondary_tab: str
    workspace_title: str
    status_text: str


@dataclass(frozen=True)
class RenderedNavigationFlow:
    """Rendered navigation scene with projected control geometry."""

    image: Any
    control_bboxes_by_id: Dict[str, List[float]]
    badge_bboxes_by_id: Dict[str, List[float]]
    support_bboxes_by_id: Dict[str, List[float]]
    control_records: Tuple[Dict[str, Any], ...]
    support_records: Tuple[Dict[str, Any], ...]
    scene_bbox_px: List[float]
    window_bbox_px: List[float]
    background_meta: Dict[str, Any]
    post_noise_meta: Dict[str, Any]
    render_params: RenderParams
    profile: AppProfile
    theme: Theme
