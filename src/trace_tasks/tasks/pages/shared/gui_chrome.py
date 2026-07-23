"""Reusable GUI chrome drawing primitives for pages scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

from PIL import ImageDraw, ImageFont

from ...shared.text_legibility import contrast_ratio, draw_text_traced, normalize_rgb
from ...shared.text_rendering import draw_text_centered, fit_font_to_box, load_font
from .information_style import PagesInformationStyle

BBox = Tuple[float, float, float, float]
Color = Tuple[int, int, int]

SUPPORTED_GUI_SCENE_VARIANTS: Tuple[str, ...] = (
    "office_document",
    "creative_workspace",
    "developer_ide",
    "cad_workspace",
    "scientific_plotter",
    "os_file_manager",
)
SUPPORTED_GUI_STYLE_VARIANTS: Tuple[str, ...] = ("standard", "compact", "contrast", "cool", "warm", "sage")


@dataclass(frozen=True)
class GuiTheme:
    """Color theme for reusable synthetic app chrome."""

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
class GuiAppProfile:
    """Visible app labels used as non-query scene dressing."""

    app_title: str
    window_title: str
    primary_tab: str
    secondary_tab: str
    workspace_title: str
    status_text: str


GUI_APP_PROFILES: Dict[str, GuiAppProfile] = {
    "office_document": GuiAppProfile("ReviewHub", "Approvals", "Inbox", "Rules", "Workflow Admin", "Live"),
    "creative_workspace": GuiAppProfile("AssetFlow", "Library", "Assets", "Campaigns", "Content Admin", "Synced"),
    "developer_ide": GuiAppProfile("OpsBoard", "Incidents", "Deploys", "Queues", "Operations Admin", "Healthy"),
    "cad_workspace": GuiAppProfile("InventoryGrid", "Catalog", "Items", "Rules", "Product Admin", "Updated"),
    "scientific_plotter": GuiAppProfile("MetricsCloud", "Experiments", "Reports", "Segments", "Analytics Admin", "2.4k rows"),
    "os_file_manager": GuiAppProfile("PortalDesk", "Shared Files", "Files", "Teams", "Workspace Admin", "23 items"),
}


def clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def bbox_list(bbox: Sequence[float]) -> List[float]:
    return [round(float(value), 3) for value in bbox]


def measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[float, float]:
    try:
        bbox = draw.textbbox((0, 0), str(text), font=font)
        return (float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1]))
    except Exception:
        width, height = draw.textsize(str(text), font=font)
        return (float(width), float(height))


def draw_text_left(
    draw: ImageDraw.ImageDraw,
    *,
    text: str,
    bbox: Sequence[float],
    fill: Color,
    max_size_px: int,
    min_size_px: int = 8,
    bold: bool = False,
) -> None:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    font = fit_font_to_box(
        draw,
        text=str(text),
        max_width=max(1.0, float(x2 - x1)),
        max_height=max(1.0, float(y2 - y1)),
        bold=bool(bold),
        min_size_px=int(min_size_px),
        max_size_px=int(max_size_px),
        fill_ratio=0.96,
    )
    _width, height = measure_text(draw, str(text), font)
    y = float(y1) + max(0.0, (float(y2 - y1) - float(height)) / 2.0) - 1.0
    draw_text_traced(
        draw,
        (float(x1), float(y)),
        str(text),
        fill=fill,
        font=font,
        role="readout",
        required=False,
    )


def draw_text_center_fit(
    draw: ImageDraw.ImageDraw,
    *,
    text: str,
    bbox: Sequence[float],
    fill: Color,
    max_size_px: int,
    min_size_px: int = 8,
    bold: bool = False,
) -> None:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    font = fit_font_to_box(
        draw,
        text=str(text),
        max_width=max(1.0, float(x2 - x1)),
        max_height=max(1.0, float(y2 - y1)),
        bold=bool(bold),
        min_size_px=int(min_size_px),
        max_size_px=int(max_size_px),
        fill_ratio=0.90,
    )
    draw_text_centered(
        draw,
        text=str(text),
        center=((float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0),
        font=font,
        fill=fill,
        stroke_width=0,
    )


def rounded_rect(
    draw: ImageDraw.ImageDraw,
    bbox: Sequence[float],
    *,
    radius: int,
    fill: Color,
    outline: Color | None = None,
    width: int = 1,
) -> None:
    draw.rounded_rectangle(
        [float(value) for value in bbox],
        radius=max(0, int(radius)),
        fill=fill,
        outline=outline,
        width=max(1, int(width)),
    )


def resolve_gui_theme(style_variant: str) -> GuiTheme:
    """Resolve a reusable GUI theme; style variants only affect rendering."""

    if str(style_variant) == "cool":
        return GuiTheme("cool", (253, 254, 255), (49, 80, 112), (255, 255, 255), (198, 211, 224), (244, 248, 252), (235, 244, 250), (255, 255, 255), (177, 196, 213), (35, 48, 63), (85, 101, 118), (225, 241, 255), (38, 113, 171), (64, 142, 137), (35, 58, 84), (255, 255, 255))
    if str(style_variant) == "warm":
        return GuiTheme("warm", (255, 254, 250), (116, 77, 49), (255, 255, 255), (219, 207, 193), (250, 247, 241), (244, 238, 228), (255, 255, 252), (207, 186, 164), (55, 44, 34), (112, 93, 73), (255, 238, 219), (159, 90, 45), (46, 126, 119), (75, 55, 41), (255, 255, 255))
    if str(style_variant) == "sage":
        return GuiTheme("sage", (253, 255, 253), (53, 92, 79), (255, 255, 255), (198, 216, 208), (244, 250, 247), (234, 245, 240), (255, 255, 255), (174, 199, 188), (35, 53, 47), (80, 104, 96), (222, 244, 234), (41, 123, 100), (166, 91, 65), (35, 65, 55), (255, 255, 255))
    if str(style_variant) == "compact":
        return GuiTheme("compact", (251, 252, 253), (45, 53, 67), (250, 252, 255), (203, 209, 218), (242, 245, 248), (232, 240, 240), (255, 255, 255), (173, 185, 195), (38, 44, 55), (91, 99, 112), (221, 244, 242), (0, 126, 145), (225, 90, 71), (31, 39, 51), (255, 255, 255))
    if str(style_variant) == "contrast":
        return GuiTheme("contrast", (250, 250, 247), (34, 34, 38), (255, 255, 255), (184, 184, 178), (241, 241, 236), (229, 239, 246), (255, 255, 252), (83, 92, 103), (26, 28, 32), (77, 81, 88), (246, 226, 230), (184, 53, 71), (24, 121, 108), (34, 34, 38), (255, 255, 255))
    return GuiTheme("standard", (255, 255, 255), (63, 78, 104), (255, 255, 255), (205, 211, 220), (246, 248, 250), (237, 243, 248), (255, 255, 255), (186, 196, 210), (40, 48, 61), (91, 101, 117), (225, 239, 255), (43, 114, 197), (213, 111, 54), (36, 47, 64), (255, 255, 255))


def _blend_rgb(base: Sequence[int], overlay: Sequence[int], amount: float) -> Color:
    factor = clamp_unit(float(amount))
    return tuple(
        int(round((int(base_channel) * (1.0 - factor)) + (int(overlay_channel) * factor)))
        for base_channel, overlay_channel in zip(normalize_rgb(base), normalize_rgb(overlay))
    )


def _readable_on(surface_rgb: Sequence[int], preferred_rgbs: Sequence[Sequence[int]]) -> Color:
    surface = normalize_rgb(surface_rgb)
    candidates = [
        *(normalize_rgb(value) for value in preferred_rgbs),
        (255, 255, 255),
        (10, 14, 22),
        (0, 0, 0),
    ]
    unique: List[Color] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return max(unique, key=lambda candidate: contrast_ratio(candidate, surface))


def gui_theme_from_information_style(style: PagesInformationStyle) -> GuiTheme:
    """Map the shared Pages 25-treatment style into synthetic GUI chrome roles."""

    selected_mix = 0.28 if str(style.treatment).startswith("dark_") else 0.18
    selected_fill = _blend_rgb(style.panel_fill_rgb, style.highlight_rgb, selected_mix)
    badge_fill = tuple(int(value) for value in style.accent_rgb)
    badge_text = _readable_on(
        badge_fill,
        (
            style.header_text_rgb,
            style.text_rgb,
            style.text_stroke_rgb,
        ),
    )
    title_text = _readable_on(
        style.header_rgb,
        (
            style.header_text_rgb,
            style.text_rgb,
            style.text_stroke_rgb,
        ),
    )
    return GuiTheme(
        name=f"information_scene:{style.style_pack}",
        app_fill=tuple(int(value) for value in style.surface_rgb),
        title_bar=tuple(int(value) for value in style.header_rgb),
        title_text=title_text,
        chrome_line=tuple(int(value) for value in style.panel_border_rgb),
        panel_fill=tuple(int(value) for value in style.panel_fill_rgb),
        panel_alt_fill=tuple(int(value) for value in style.surface_alt_rgb),
        control_fill=tuple(int(value) for value in style.surface_rgb),
        control_outline=tuple(int(value) for value in style.guide_rgb),
        control_text=tuple(int(value) for value in style.text_rgb),
        muted_text=tuple(int(value) for value in style.muted_text_rgb),
        selected_fill=selected_fill,
        accent=tuple(int(value) for value in style.accent_rgb),
        accent_alt=tuple(int(value) for value in style.secondary_accent_rgb),
        badge_fill=badge_fill,
        badge_text=badge_text,
    )


def draw_app_chrome(
    draw: ImageDraw.ImageDraw,
    *,
    scene_variant: str,
    render_params: Any,
    theme: GuiTheme,
) -> Tuple[BBox, GuiAppProfile]:
    """Draw reusable desktop-window chrome and return the content frame."""

    profile = GUI_APP_PROFILES[str(scene_variant)]
    m = int(render_params.window_margin_px)
    width = int(render_params.canvas_width)
    height = int(render_params.canvas_height)
    window = (float(m), float(m - 6), float(width - m), float(height - m + 6))
    rounded_rect(draw, window, radius=int(render_params.corner_radius_px), fill=theme.app_fill, outline=theme.chrome_line, width=2)

    header_h = max(58, int(render_params.title_bar_height_px) + 16)
    header = (window[0], window[1], window[2], window[1] + float(header_h))
    draw.rounded_rectangle(
        [header[0], header[1], header[2], header[3] + int(render_params.corner_radius_px)],
        radius=int(render_params.corner_radius_px),
        fill=theme.app_fill,
    )
    draw.rectangle([header[0], header[3] - int(render_params.corner_radius_px), header[2], header[3]], fill=theme.app_fill)
    draw.line([header[0], header[3], header[2], header[3]], fill=theme.chrome_line, width=1)

    logo = (window[0] + 22.0, header[1] + 15.0, window[0] + 52.0, header[1] + 45.0)
    rounded_rect(draw, logo, radius=8, fill=theme.accent, outline=None)
    logo_font = load_font(int(render_params.small_font_size_px), bold=True)
    draw_text_centered(
        draw,
        text=str(profile.app_title)[:1],
        center=((logo[0] + logo[2]) / 2.0, (logo[1] + logo[3]) / 2.0),
        font=logo_font,
        fill=theme.badge_text,
    )

    title_font = load_font(int(render_params.title_font_size_px), bold=True)
    draw_text_traced(draw, (window[0] + 64.0, header[1] + 11.0), str(profile.app_title), fill=theme.control_text, font=title_font, role="readout", required=False)
    small_font = load_font(int(render_params.small_font_size_px), bold=False)
    draw_text_traced(draw, (window[0] + 66.0, header[1] + 39.0), str(profile.window_title), fill=theme.muted_text, font=small_font, role="readout", required=False)

    nav_x = window[0] + 285.0
    for idx, nav_label in enumerate((str(profile.primary_tab), str(profile.secondary_tab), "Reports", "Settings")):
        tab_w = 92.0 if len(nav_label) <= 8 else 118.0
        tab_bbox = (nav_x, header[1] + 16.0, nav_x + tab_w, header[1] + 46.0)
        if idx == 0:
            rounded_rect(draw, tab_bbox, radius=15, fill=theme.selected_fill, outline=theme.accent, width=1)
            draw_text_center_fit(draw, text=nav_label, bbox=(tab_bbox[0] + 10.0, tab_bbox[1] + 4.0, tab_bbox[2] - 10.0, tab_bbox[3] - 4.0), fill=theme.control_text, max_size_px=int(render_params.small_font_size_px), bold=True)
        else:
            draw_text_center_fit(draw, text=nav_label, bbox=(tab_bbox[0] + 8.0, tab_bbox[1] + 5.0, tab_bbox[2] - 8.0, tab_bbox[3] - 5.0), fill=theme.muted_text, max_size_px=int(render_params.small_font_size_px), bold=True)
        nav_x += tab_w + 8.0

    status_pill = (window[2] - 194.0, header[1] + 16.0, window[2] - 26.0, header[1] + 46.0)
    rounded_rect(draw, status_pill, radius=15, fill=theme.panel_alt_fill, outline=theme.chrome_line, width=1)
    draw_text_center_fit(draw, text=str(profile.status_text), bbox=(status_pill[0] + 10.0, status_pill[1] + 4.0, status_pill[2] - 10.0, status_pill[3] - 4.0), fill=theme.control_text, max_size_px=int(render_params.small_font_size_px), bold=True)

    menu_y1 = header[3]
    menu_y2 = menu_y1 + max(40, int(render_params.menu_bar_height_px) + 6)
    draw.rectangle([window[0], menu_y1, window[2], menu_y2], fill=theme.panel_fill, outline=theme.chrome_line)
    tab_font = load_font(int(render_params.small_font_size_px), bold=True)
    breadcrumb_x = window[0] + 26.0
    draw_text_traced(draw, (breadcrumb_x, menu_y1 + 12.0), "Workspace", fill=theme.muted_text, font=tab_font, role="readout", required=False)
    draw_text_traced(draw, (breadcrumb_x + 92.0, menu_y1 + 12.0), "/", fill=theme.muted_text, font=tab_font, role="readout", required=False)
    draw_text_traced(draw, (breadcrumb_x + 112.0, menu_y1 + 12.0), str(profile.window_title), fill=theme.control_text, font=tab_font, role="readout", required=False)
    filter_bbox = (window[2] - 266.0, menu_y1 + 7.0, window[2] - 172.0, menu_y2 - 7.0)
    export_bbox = (window[2] - 154.0, menu_y1 + 7.0, window[2] - 26.0, menu_y2 - 7.0)
    rounded_rect(draw, filter_bbox, radius=8, fill=theme.control_fill, outline=theme.chrome_line, width=1)
    rounded_rect(draw, export_bbox, radius=8, fill=theme.control_fill, outline=theme.chrome_line, width=1)
    draw_text_center_fit(draw, text="Filter", bbox=(filter_bbox[0] + 8.0, filter_bbox[1] + 3.0, filter_bbox[2] - 8.0, filter_bbox[3] - 3.0), fill=theme.muted_text, max_size_px=int(render_params.small_font_size_px), bold=True)
    draw_text_center_fit(draw, text="Export", bbox=(export_bbox[0] + 8.0, export_bbox[1] + 3.0, export_bbox[2] - 8.0, export_bbox[3] - 3.0), fill=theme.muted_text, max_size_px=int(render_params.small_font_size_px), bold=True)
    return (window[0] + 22, menu_y2 + 18, window[2] - 22, window[3] - 18), profile


def draw_badge(
    draw: ImageDraw.ImageDraw,
    *,
    control_bbox: Sequence[float],
    label: str,
    render_params: Any,
    theme: GuiTheme,
) -> List[float]:
    """Draw a candidate-label badge in a GUI control."""

    x1, y1, x2, y2 = [float(value) for value in control_bbox]
    available_size = max(16, int(min(float(x2 - x1), float(y2 - y1)) - 8.0))
    size = max(16, min(max(20, int(render_params.badge_size_px)), int(available_size)))
    badge = (x1 + 5.0, y1 + 5.0, x1 + 5.0 + float(size), y1 + 5.0 + float(size))
    draw.ellipse(
        [float(value) for value in badge],
        fill=theme.badge_fill,
        outline=(255, 255, 255),
        width=1,
    )
    font = fit_font_to_box(
        draw,
        text=str(label),
        max_width=float(size) * 0.60,
        max_height=float(size) * 0.60,
        bold=False,
        min_size_px=8,
        max_size_px=int(render_params.label_font_size_px),
        fill_ratio=0.90,
    )
    draw_text_centered(
        draw,
        text=str(label),
        center=((badge[0] + badge[2]) / 2.0, (badge[1] + badge[3]) / 2.0),
        font=font,
        fill=theme.badge_text,
    )
    return bbox_list(badge)


def draw_control_button(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: Sequence[float],
    control: Any,
    render_params: Any,
    theme: GuiTheme,
) -> List[float]:
    """Draw one GUI candidate control button and return its badge bbox."""

    rounded_rect(
        draw,
        bbox,
        radius=int(render_params.control_corner_radius_px),
        fill=theme.control_fill,
        outline=theme.control_outline,
        width=int(render_params.control_outline_width_px),
    )
    display_text = str(control.display_text)
    max_size_px = int(render_params.small_font_size_px)
    if str(control.role) == "panel_control":
        max_size_px = int(render_params.body_font_size_px + 2)
    draw_text_center_fit(
        draw,
        text=display_text,
        bbox=(float(bbox[0]) + 30.0, float(bbox[1]) + 4.0, float(bbox[2]) - 8.0, float(bbox[3]) - 4.0),
        fill=theme.control_text,
        max_size_px=int(max_size_px),
        bold=True,
    )
    return draw_badge(draw, control_bbox=bbox, label=str(control.candidate_label), render_params=render_params, theme=theme)


__all__ = [
    "BBox",
    "Color",
    "GUI_APP_PROFILES",
    "GuiAppProfile",
    "GuiTheme",
    "SUPPORTED_GUI_SCENE_VARIANTS",
    "SUPPORTED_GUI_STYLE_VARIANTS",
    "bbox_list",
    "clamp_unit",
    "draw_app_chrome",
    "draw_badge",
    "draw_control_button",
    "draw_text_center_fit",
    "draw_text_left",
    "gui_theme_from_information_style",
    "measure_text",
    "resolve_gui_theme",
    "rounded_rect",
]
