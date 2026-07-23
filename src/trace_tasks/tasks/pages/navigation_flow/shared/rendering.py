"""Rendering helpers for navigation-flow page scenes."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.pages.shared.gui_chrome import (
    draw_text_center_fit as _draw_text_center_fit,
    draw_text_left as _draw_text_left,
    gui_theme_from_information_style,
    rounded_rect as _rounded_rect,
)
from trace_tasks.tasks.pages.shared.information_style import make_pages_information_background, resolve_pages_information_style
from trace_tasks.tasks.shared.render_variation import resolve_render_int
from trace_tasks.tasks.shared.text_legibility import draw_text_traced
from trace_tasks.tasks.shared.text_rendering import draw_text_centered, fit_font_to_box, load_font

from .annotations import bbox_list
from .defaults import (
    NAMESPACE_ROOT,
    POST_IMAGE_NOISE_DEFAULTS,
    RENDER_FALLBACKS,
    RENDERING_DEFAULTS,
)
from .state import (
    MENU_SURFACE,
    RIBBON_SURFACE,
    SIDEBAR_SURFACE,
    AppProfile,
    BBox,
    Color,
    ControlSpec,
    NavigationFlowCase,
    RenderParams,
    RenderedNavigationFlow,
    Theme,
)


_APP_PROFILES: Dict[str, AppProfile] = {
    "office_document": AppProfile("ReviewHub", "Approvals", "Inbox", "Rules", "Workflow Admin", "Live"),
    "creative_workspace": AppProfile("AssetFlow", "Library", "Assets", "Campaigns", "Content Admin", "Synced"),
    "developer_ide": AppProfile("OpsBoard", "Incidents", "Deploys", "Queues", "Operations Admin", "Healthy"),
    "cad_workspace": AppProfile("InventoryGrid", "Catalog", "Items", "Rules", "Product Admin", "Updated"),
    "scientific_plotter": AppProfile("MetricsCloud", "Experiments", "Reports", "Segments", "Analytics Admin", "2.4k rows"),
    "os_file_manager": AppProfile("PortalDesk", "Shared Files", "Files", "Teams", "Workspace Admin", "23 items"),
}


def resolve_render_params(params: Mapping[str, Any], *, instance_seed: int, namespace: str) -> RenderParams:
    """Resolve GUI-window render parameters."""

    values = dict(RENDER_FALLBACKS)

    def _int_value(key: str) -> int:
        return resolve_render_int(
            params,
            RENDERING_DEFAULTS,
            str(key),
            int(values[str(key)]),
            instance_seed=int(instance_seed),
            namespace=f"{NAMESPACE_ROOT}.{namespace}",
        )

    return RenderParams(
        canvas_width=_int_value("canvas_width"),
        canvas_height=_int_value("canvas_height"),
        window_margin_px=_int_value("window_margin_px"),
        title_bar_height_px=_int_value("title_bar_height_px"),
        menu_bar_height_px=_int_value("menu_bar_height_px"),
        corner_radius_px=_int_value("corner_radius_px"),
        control_corner_radius_px=_int_value("control_corner_radius_px"),
        control_outline_width_px=_int_value("control_outline_width_px"),
        badge_size_px=_int_value("badge_size_px"),
        title_font_size_px=_int_value("title_font_size_px"),
        body_font_size_px=_int_value("body_font_size_px"),
        small_font_size_px=_int_value("small_font_size_px"),
        label_font_size_px=_int_value("label_font_size_px"),
    )


def _theme_from_information_style(style: Any) -> Theme:
    """Map the shared Pages information style into navigation app chrome roles."""

    theme = gui_theme_from_information_style(style)
    return Theme(
        name=str(theme.name),
        app_fill=tuple(theme.app_fill),
        title_bar=tuple(theme.title_bar),
        title_text=tuple(theme.title_text),
        chrome_line=tuple(theme.chrome_line),
        panel_fill=tuple(theme.panel_fill),
        panel_alt_fill=tuple(theme.panel_alt_fill),
        control_fill=tuple(theme.control_fill),
        control_outline=tuple(theme.control_outline),
        control_text=tuple(theme.control_text),
        muted_text=tuple(theme.muted_text),
        selected_fill=tuple(theme.selected_fill),
        accent=tuple(theme.accent),
        accent_alt=tuple(theme.accent_alt),
        badge_fill=tuple(theme.badge_fill),
        badge_text=tuple(theme.badge_text),
    )


def _draw_app_chrome(
    draw: ImageDraw.ImageDraw,
    *,
    case: NavigationFlowCase,
    render_params: RenderParams,
    theme: Theme,
) -> Tuple[BBox, AppProfile]:
    """Draw reusable desktop chrome; task annotation stays inside returned content bounds."""

    profile = _APP_PROFILES[str(case.scene_variant)]
    margin = int(render_params.window_margin_px)
    width = int(render_params.canvas_width)
    height = int(render_params.canvas_height)
    window = (float(margin), float(margin - 6), float(width - margin), float(height - margin + 6))
    _rounded_rect(
        draw,
        window,
        radius=int(render_params.corner_radius_px),
        fill=theme.app_fill,
        outline=theme.chrome_line,
        width=2,
    )

    header_h = max(58, int(render_params.title_bar_height_px) + 16)
    header = (window[0], window[1], window[2], window[1] + float(header_h))
    draw.rounded_rectangle(
        [header[0], header[1], header[2], header[3] + int(render_params.corner_radius_px)],
        radius=int(render_params.corner_radius_px),
        fill=theme.app_fill,
    )
    draw.rectangle(
        [header[0], header[3] - int(render_params.corner_radius_px), header[2], header[3]],
        fill=theme.app_fill,
    )
    draw.line([header[0], header[3], header[2], header[3]], fill=theme.chrome_line, width=1)

    logo = (window[0] + 22.0, header[1] + 15.0, window[0] + 52.0, header[1] + 45.0)
    _rounded_rect(draw, logo, radius=8, fill=theme.accent, outline=None)
    logo_font = load_font(int(render_params.small_font_size_px), bold=True)
    draw_text_centered(
        draw,
        text=str(profile.app_title)[:1],
        center=((logo[0] + logo[2]) / 2.0, (logo[1] + logo[3]) / 2.0),
        font=logo_font,
        fill=(255, 255, 255),
    )

    title_font = load_font(int(render_params.title_font_size_px), bold=True)
    draw_text_traced(
        draw,
        (window[0] + 64.0, header[1] + 11.0),
        str(profile.app_title),
        fill=theme.control_text,
        font=title_font,
        role="readout",
        required=False,
    )
    small_font = load_font(int(render_params.small_font_size_px), bold=False)
    draw_text_traced(
        draw,
        (window[0] + 66.0, header[1] + 39.0),
        str(profile.window_title),
        fill=theme.muted_text,
        font=small_font,
        role="readout",
        required=False,
    )

    nav_x = window[0] + 285.0
    for index, nav_label in enumerate((profile.primary_tab, profile.secondary_tab, "Reports", "Settings")):
        tab_w = 92.0 if len(str(nav_label)) <= 8 else 118.0
        tab_bbox = (nav_x, header[1] + 16.0, nav_x + tab_w, header[1] + 46.0)
        if int(index) == 0:
            _rounded_rect(draw, tab_bbox, radius=15, fill=theme.selected_fill, outline=theme.accent, width=1)
            _draw_text_center_fit(
                draw,
                text=str(nav_label),
                bbox=(tab_bbox[0] + 10.0, tab_bbox[1] + 4.0, tab_bbox[2] - 10.0, tab_bbox[3] - 4.0),
                fill=theme.control_text,
                max_size_px=int(render_params.small_font_size_px),
                bold=True,
            )
        else:
            _draw_text_center_fit(
                draw,
                text=str(nav_label),
                bbox=(tab_bbox[0] + 8.0, tab_bbox[1] + 5.0, tab_bbox[2] - 8.0, tab_bbox[3] - 5.0),
                fill=theme.muted_text,
                max_size_px=int(render_params.small_font_size_px),
                bold=True,
            )
        nav_x += tab_w + 8.0

    status_pill = (window[2] - 194.0, header[1] + 16.0, window[2] - 26.0, header[1] + 46.0)
    _rounded_rect(draw, status_pill, radius=15, fill=theme.panel_alt_fill, outline=theme.chrome_line, width=1)
    _draw_text_center_fit(
        draw,
        text=str(profile.status_text),
        bbox=(status_pill[0] + 10.0, status_pill[1] + 4.0, status_pill[2] - 10.0, status_pill[3] - 4.0),
        fill=theme.control_text,
        max_size_px=int(render_params.small_font_size_px),
        bold=True,
    )

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
    _rounded_rect(draw, filter_bbox, radius=8, fill=theme.control_fill, outline=theme.chrome_line, width=1)
    _rounded_rect(draw, export_bbox, radius=8, fill=theme.control_fill, outline=theme.chrome_line, width=1)
    _draw_text_center_fit(
        draw,
        text="Filter",
        bbox=(filter_bbox[0] + 8.0, filter_bbox[1] + 3.0, filter_bbox[2] - 8.0, filter_bbox[3] - 3.0),
        fill=theme.muted_text,
        max_size_px=int(render_params.small_font_size_px),
        bold=True,
    )
    _draw_text_center_fit(
        draw,
        text="Export",
        bbox=(export_bbox[0] + 8.0, export_bbox[1] + 3.0, export_bbox[2] - 8.0, export_bbox[3] - 3.0),
        fill=theme.muted_text,
        max_size_px=int(render_params.small_font_size_px),
        bold=True,
    )
    return (window[0] + 22, menu_y2 + 18, window[2] - 22, window[3] - 18), profile


def _draw_badge(
    draw: ImageDraw.ImageDraw,
    *,
    control_bbox: BBox,
    label: str,
    render_params: RenderParams,
    theme: Theme,
) -> List[float]:
    x1, y1, x2, y2 = [float(value) for value in control_bbox]
    available_size = max(16, int(min(float(x2 - x1), float(y2 - y1)) - 8.0))
    size = max(16, min(max(20, int(render_params.badge_size_px)), int(available_size)))
    badge = (x1 + 5.0, y1 + 5.0, x1 + 5.0 + float(size), y1 + 5.0 + float(size))
    draw.ellipse([float(value) for value in badge], fill=theme.badge_fill, outline=(255, 255, 255), width=1)
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


def _draw_control_button(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: BBox,
    control: ControlSpec,
    render_params: RenderParams,
    theme: Theme,
) -> List[float]:
    _rounded_rect(
        draw,
        bbox,
        radius=int(render_params.control_corner_radius_px),
        fill=theme.control_fill,
        outline=theme.control_outline,
        width=int(render_params.control_outline_width_px),
    )
    _draw_text_center_fit(
        draw,
        text=str(control.display_text),
        bbox=(bbox[0] + 30.0, bbox[1] + 4.0, bbox[2] - 8.0, bbox[3] - 4.0),
        fill=theme.control_text,
        max_size_px=int(render_params.small_font_size_px),
        bold=True,
    )
    return _draw_badge(draw, control_bbox=bbox, label=str(control.candidate_label), render_params=render_params, theme=theme)


def _fit_control_bbox(
    bbox: BBox,
    *,
    bounds: BBox,
    min_height_px: float = 24.0,
) -> BBox:
    """Return a visible control bbox whose height meets the annotation minimum."""

    x1, y1, x2, y2 = [float(value) for value in bbox]
    bx1, by1, bx2, by2 = [float(value) for value in bounds]
    current_h = max(0.0, y2 - y1)
    target_h = min(max(float(current_h), float(min_height_px)), max(0.0, by2 - by1))
    cy = 0.5 * (y1 + y2)
    ny1 = cy - (0.5 * target_h)
    ny2 = cy + (0.5 * target_h)
    if ny1 < by1:
        ny2 += by1 - ny1
        ny1 = by1
    if ny2 > by2:
        ny1 -= ny2 - by2
        ny2 = by2
    return (
        max(bx1, min(bx2, x1)),
        max(by1, min(by2, ny1)),
        max(bx1, min(bx2, x2)),
        max(by1, min(by2, ny2)),
    )


def _add_support(
    support_bboxes: Dict[str, List[float]],
    support_records: List[Dict[str, Any]],
    support_id: str,
    support_kind: str,
    display_text: str,
    bbox: BBox,
) -> None:
    support_bboxes[str(support_id)] = bbox_list(bbox)
    support_records.append(
        {
            "support_id": str(support_id),
            "support_kind": str(support_kind),
            "display_text": str(display_text),
            "bbox_px": bbox_list(bbox),
        }
    )


def _draw_menu_path_scene(
    draw: ImageDraw.ImageDraw,
    *,
    case: NavigationFlowCase,
    workspace: BBox,
    render_params: RenderParams,
    theme: Theme,
    control_bboxes: Dict[str, List[float]],
    badge_bboxes: Dict[str, List[float]],
    support_bboxes: Dict[str, List[float]],
    support_records: List[Dict[str, Any]],
) -> None:
    """Draw menu roots, submenu groups, and command candidates."""

    header_font = load_font(int(render_params.body_font_size_px), bold=True)
    draw_text_traced(draw, (workspace[0] + 18.0, workspace[1] + 14.0), "Menu Navigator", fill=theme.control_text, font=header_font, role="readout", required=False)
    controls_by_menu: Dict[str, List[ControlSpec]] = {}
    for control in case.controls:
        controls_by_menu.setdefault(str(control.path_keys[0]), []).append(control)
    command_symbols: Dict[str, str] = {}
    for control in case.controls:
        command_symbols.setdefault(str(control.path_keys[-1]), str(control.display_text))
    legend_items = list(command_symbols.items())[:4]
    legend_x1 = workspace[0] + 250.0
    legend_x2 = workspace[2] - 18.0
    legend_y1 = workspace[1] + 12.0
    legend_w = (legend_x2 - legend_x1) / max(1, len(legend_items))
    for legend_index, (command, symbol) in enumerate(legend_items):
        legend_bbox = (
            legend_x1 + legend_index * legend_w,
            legend_y1,
            legend_x1 + (legend_index + 1) * legend_w - 8.0,
            legend_y1 + 36.0,
        )
        _rounded_rect(draw, legend_bbox, radius=8, fill=theme.panel_alt_fill, outline=theme.chrome_line)
        _draw_text_center_fit(
            draw,
            text=f"{symbol} {command}",
            bbox=(legend_bbox[0] + 6.0, legend_bbox[1] + 4.0, legend_bbox[2] - 6.0, legend_bbox[3] - 4.0),
            fill=theme.control_text,
            max_size_px=int(render_params.small_font_size_px),
            bold=True,
        )
    menus = list(controls_by_menu.keys())
    menu_x1 = workspace[0] + 18.0
    menu_y1 = workspace[1] + 58.0
    menu_h = 38.0
    menu_w = (workspace[2] - workspace[0] - 36.0) / float(len(menus))
    body_y1 = menu_y1 + menu_h + 12.0
    for menu_index, menu in enumerate(menus):
        mx1 = menu_x1 + menu_index * menu_w
        menu_bbox = (mx1, menu_y1, mx1 + menu_w - 10.0, menu_y1 + menu_h)
        _rounded_rect(draw, menu_bbox, radius=8, fill=theme.selected_fill if menu_index % 2 == 0 else theme.panel_alt_fill, outline=theme.chrome_line)
        _draw_text_center_fit(draw, text=str(menu), bbox=(menu_bbox[0] + 8.0, menu_bbox[1] + 5.0, menu_bbox[2] - 8.0, menu_bbox[3] - 5.0), fill=theme.control_text, max_size_px=int(render_params.small_font_size_px), bold=True)
        _add_support(support_bboxes, support_records, f"support_menu_{menu_index}", "menu_root", str(menu), menu_bbox)
        submenu_groups: Dict[str, List[ControlSpec]] = {}
        for control in controls_by_menu[str(menu)]:
            submenu_groups.setdefault(str(control.path_keys[1]), []).append(control)
        submenu_items = list(submenu_groups.items())
        panel_bbox = (mx1, body_y1, mx1 + menu_w - 10.0, workspace[3] - 18.0)
        _rounded_rect(draw, panel_bbox, radius=8, fill=theme.control_fill, outline=theme.chrome_line)
        sub_h = (panel_bbox[3] - panel_bbox[1] - 22.0) / float(len(submenu_items))
        for submenu_index, (submenu, controls) in enumerate(submenu_items):
            sy1 = panel_bbox[1] + 12.0 + submenu_index * sub_h
            submenu_bbox = (panel_bbox[0] + 12.0, sy1, panel_bbox[2] - 12.0, sy1 + 28.0)
            _rounded_rect(draw, submenu_bbox, radius=7, fill=theme.panel_alt_fill, outline=theme.chrome_line)
            _draw_text_left(draw, text=str(submenu), bbox=(submenu_bbox[0] + 10.0, submenu_bbox[1] + 4.0, submenu_bbox[2] - 10.0, submenu_bbox[3] - 4.0), fill=theme.control_text, max_size_px=int(render_params.small_font_size_px), bold=True)
            _add_support(support_bboxes, support_records, f"support_menu_{menu_index}_submenu_{submenu_index}", "submenu", str(submenu), submenu_bbox)
            by_group: Dict[str, List[ControlSpec]] = {}
            for control in controls:
                by_group.setdefault(str(control.path_keys[2]), []).append(control)
            group_items = list(by_group.items())
            group_h = (sub_h - 44.0) / max(1, len(group_items))
            for group_index, (group, group_controls) in enumerate(group_items):
                gy1 = submenu_bbox[3] + 8.0 + group_index * group_h
                group_bbox = (submenu_bbox[0], gy1, submenu_bbox[2], gy1 + group_h - 6.0)
                _rounded_rect(draw, group_bbox, radius=7, fill=theme.panel_fill if group_index % 2 == 0 else theme.app_fill, outline=theme.chrome_line)
                _draw_text_left(draw, text=str(group), bbox=(group_bbox[0] + 8.0, group_bbox[1] + 4.0, group_bbox[0] + 112.0, group_bbox[3] - 4.0), fill=theme.muted_text, max_size_px=int(render_params.small_font_size_px), bold=True)
                _add_support(support_bboxes, support_records, f"support_menu_{menu_index}_submenu_{submenu_index}_group_{group_index}", "menu_group", str(group), group_bbox)
                controls_sorted = sorted(group_controls, key=lambda item: int(item.order_index))
                controls_x1 = group_bbox[0] + 122.0
                control_w = (group_bbox[2] - controls_x1 - 8.0) / max(1, len(controls_sorted))
                for command_index, control in enumerate(controls_sorted):
                    bx1 = controls_x1 + command_index * control_w
                    bbox = _fit_control_bbox(
                        (bx1, group_bbox[1] + 6.0, bx1 + control_w - 8.0, group_bbox[3] - 6.0),
                        bounds=group_bbox,
                    )
                    badge_bboxes[str(control.control_id)] = _draw_control_button(draw, bbox=bbox, control=control, render_params=render_params, theme=theme)
                    control_bboxes[str(control.control_id)] = bbox_list(bbox)


def _draw_sidebar_tree_scene(
    draw: ImageDraw.ImageDraw,
    *,
    case: NavigationFlowCase,
    workspace: BBox,
    render_params: RenderParams,
    theme: Theme,
    control_bboxes: Dict[str, List[float]],
    badge_bboxes: Dict[str, List[float]],
    support_bboxes: Dict[str, List[float]],
    support_records: List[Dict[str, Any]],
) -> None:
    """Draw a sidebar tree with lettered item controls."""

    header_font = load_font(int(render_params.body_font_size_px), bold=True)
    draw_text_traced(draw, (workspace[0] + 18.0, workspace[1] + 14.0), "Sidebar Tree", fill=theme.control_text, font=header_font, role="readout", required=False)
    tree_x1 = workspace[0] + 18.0
    tree_x2 = workspace[0] + 520.0
    tree_y1 = workspace[1] + 54.0
    tree_y2 = workspace[3] - 18.0
    _rounded_rect(draw, (tree_x1, tree_y1, tree_x2, tree_y2), radius=10, fill=theme.control_fill, outline=theme.chrome_line)
    preview_bbox = (tree_x2 + 18.0, tree_y1, workspace[2] - 18.0, tree_y2)
    _rounded_rect(draw, preview_bbox, radius=10, fill=theme.panel_alt_fill, outline=theme.chrome_line)
    _draw_text_center_fit(draw, text="Content Preview", bbox=(preview_bbox[0] + 30.0, preview_bbox[1] + 40.0, preview_bbox[2] - 30.0, preview_bbox[1] + 92.0), fill=theme.muted_text, max_size_px=int(render_params.body_font_size_px), bold=True)
    by_section: Dict[str, List[ControlSpec]] = {}
    for control in case.controls:
        by_section.setdefault(str(control.path_keys[0]), []).append(control)
    item_symbols: Dict[str, str] = {}
    for control in case.controls:
        item_symbols.setdefault(str(control.path_keys[-1]), str(control.display_text))
    legend_items = list(item_symbols.items())[:4]
    if legend_items:
        legend_title_bbox = (preview_bbox[0] + 28.0, preview_bbox[1] + 126.0, preview_bbox[2] - 28.0, preview_bbox[1] + 154.0)
        _draw_text_left(draw, text="Item Legend", bbox=legend_title_bbox, fill=theme.control_text, max_size_px=int(render_params.body_font_size_px), bold=True)
        legend_x1 = preview_bbox[0] + 28.0
        legend_x2 = preview_bbox[2] - 28.0
        legend_y1 = preview_bbox[1] + 166.0
        legend_w = (legend_x2 - legend_x1) / max(1, min(2, len(legend_items)))
        for legend_index, (item, symbol) in enumerate(legend_items):
            row = legend_index // 2
            col = legend_index % 2
            legend_bbox = (
                legend_x1 + col * legend_w,
                legend_y1 + row * 44.0,
                legend_x1 + (col + 1) * legend_w - 10.0,
                legend_y1 + row * 44.0 + 34.0,
            )
            _rounded_rect(draw, legend_bbox, radius=8, fill=theme.control_fill, outline=theme.chrome_line)
            _draw_text_center_fit(draw, text=f"{symbol} {item}", bbox=(legend_bbox[0] + 8.0, legend_bbox[1] + 4.0, legend_bbox[2] - 8.0, legend_bbox[3] - 4.0), fill=theme.control_text, max_size_px=int(render_params.small_font_size_px), bold=True)
    sections = list(by_section.items())
    section_h = (tree_y2 - tree_y1 - 20.0) / float(len(sections))
    for section_index, (section, section_controls) in enumerate(sections):
        sy1 = tree_y1 + 10.0 + section_index * section_h
        section_bbox = (tree_x1 + 12.0, sy1, tree_x2 - 12.0, sy1 + 28.0)
        _rounded_rect(draw, section_bbox, radius=7, fill=theme.panel_alt_fill, outline=theme.chrome_line)
        _draw_text_left(draw, text=str(section), bbox=(section_bbox[0] + 10.0, section_bbox[1] + 4.0, section_bbox[2] - 10.0, section_bbox[3] - 4.0), fill=theme.control_text, max_size_px=int(render_params.small_font_size_px), bold=True)
        _add_support(support_bboxes, support_records, f"support_sidebar_section_{section_index}", "sidebar_section", str(section), section_bbox)
        by_group: Dict[str, List[ControlSpec]] = {}
        for control in section_controls:
            by_group.setdefault(str(control.path_keys[1]), []).append(control)
        group_items = list(by_group.items())
        group_h = (section_h - 34.0) / float(len(group_items))
        for group_index, (group, controls) in enumerate(group_items):
            gy1 = section_bbox[3] + 6.0 + group_index * group_h
            group_bbox = (tree_x1 + 28.0, gy1, tree_x2 - 16.0, gy1 + group_h - 4.0)
            _rounded_rect(draw, group_bbox, radius=6, fill=theme.panel_fill if group_index % 2 == 0 else theme.app_fill, outline=theme.chrome_line)
            _draw_text_left(draw, text=f"+ {group}", bbox=(group_bbox[0] + 8.0, group_bbox[1] + 4.0, group_bbox[0] + 110.0, group_bbox[3] - 4.0), fill=theme.muted_text, max_size_px=int(render_params.small_font_size_px), bold=True)
            _add_support(support_bboxes, support_records, f"support_sidebar_section_{section_index}_group_{group_index}", "sidebar_group", str(group), group_bbox)
            items_x1 = group_bbox[0] + 122.0
            item_w = (group_bbox[2] - items_x1 - 10.0) / max(1, len(controls))
            for item_index, control in enumerate(sorted(controls, key=lambda item: int(item.order_index))):
                ix1 = items_x1 + item_index * item_w
                bbox = _fit_control_bbox(
                    (ix1, group_bbox[1] + 5.0, ix1 + item_w - 8.0, group_bbox[3] - 5.0),
                    bounds=group_bbox,
                )
                badge_bboxes[str(control.control_id)] = _draw_control_button(draw, bbox=bbox, control=control, render_params=render_params, theme=theme)
                control_bboxes[str(control.control_id)] = bbox_list(bbox)


def _draw_ribbon_group_scene(
    draw: ImageDraw.ImageDraw,
    *,
    case: NavigationFlowCase,
    workspace: BBox,
    render_params: RenderParams,
    theme: Theme,
    control_bboxes: Dict[str, List[float]],
    badge_bboxes: Dict[str, List[float]],
    support_bboxes: Dict[str, List[float]],
    support_records: List[Dict[str, Any]],
) -> None:
    """Draw ribbon tabs, groups, and command candidates."""

    header_font = load_font(int(render_params.body_font_size_px), bold=True)
    draw_text_traced(draw, (workspace[0] + 18.0, workspace[1] + 14.0), "Ribbon Workspace", fill=theme.control_text, font=header_font, role="readout", required=False)
    by_tab: Dict[str, List[ControlSpec]] = {}
    for control in case.controls:
        by_tab.setdefault(str(control.path_keys[0]), []).append(control)
    command_symbols: Dict[str, str] = {}
    for control in case.controls:
        command_symbols.setdefault(str(control.path_keys[-1]), str(control.display_text))
    legend_items = list(command_symbols.items())[:4]
    legend_x1 = workspace[0] + 250.0
    legend_x2 = workspace[2] - 18.0
    legend_y1 = workspace[1] + 12.0
    legend_w = (legend_x2 - legend_x1) / max(1, len(legend_items))
    for legend_index, (command, symbol) in enumerate(legend_items):
        legend_bbox = (
            legend_x1 + legend_index * legend_w,
            legend_y1,
            legend_x1 + (legend_index + 1) * legend_w - 8.0,
            legend_y1 + 36.0,
        )
        _rounded_rect(draw, legend_bbox, radius=8, fill=theme.panel_alt_fill, outline=theme.chrome_line)
        _draw_text_center_fit(draw, text=f"{symbol} {command}", bbox=(legend_bbox[0] + 6.0, legend_bbox[1] + 4.0, legend_bbox[2] - 6.0, legend_bbox[3] - 4.0), fill=theme.control_text, max_size_px=int(render_params.small_font_size_px), bold=True)
    tabs = list(by_tab.items())
    tab_x1 = workspace[0] + 18.0
    tab_x2 = workspace[2] - 18.0
    tab_y1 = workspace[1] + 54.0
    tab_h = 38.0
    tab_w = (tab_x2 - tab_x1) / float(len(tabs))
    ribbon_y1 = tab_y1 + tab_h + 12.0
    ribbon_y2 = workspace[3] - 185.0
    canvas_bbox = (tab_x1, ribbon_y2 + 16.0, tab_x2, workspace[3] - 18.0)
    _rounded_rect(draw, canvas_bbox, radius=10, fill=theme.control_fill, outline=theme.chrome_line)
    for tab_index, (tab, controls_for_tab) in enumerate(tabs):
        tx1 = tab_x1 + tab_index * tab_w
        tab_bbox = (tx1 + 4.0, tab_y1, tx1 + tab_w - 4.0, tab_y1 + tab_h)
        _rounded_rect(draw, tab_bbox, radius=8, fill=theme.selected_fill if tab_index % 2 == 0 else theme.panel_alt_fill, outline=theme.chrome_line)
        _draw_text_center_fit(draw, text=str(tab), bbox=(tab_bbox[0] + 8.0, tab_bbox[1] + 5.0, tab_bbox[2] - 8.0, tab_bbox[3] - 5.0), fill=theme.control_text, max_size_px=int(render_params.small_font_size_px), bold=True)
        _add_support(support_bboxes, support_records, f"support_ribbon_tab_{tab_index}", "ribbon_tab", str(tab), tab_bbox)
        tab_panel = (tx1 + 4.0, ribbon_y1, tx1 + tab_w - 4.0, ribbon_y2)
        _rounded_rect(draw, tab_panel, radius=8, fill=theme.control_fill, outline=theme.chrome_line)
        by_group: Dict[str, List[ControlSpec]] = {}
        for control in controls_for_tab:
            by_group.setdefault(str(control.path_keys[1]), []).append(control)
        group_items = list(by_group.items())
        group_h = (tab_panel[3] - tab_panel[1] - 14.0) / float(len(group_items))
        for group_index, (group, controls) in enumerate(group_items):
            gy1 = tab_panel[1] + 8.0 + group_index * group_h
            group_bbox = (tab_panel[0] + 10.0, gy1, tab_panel[2] - 10.0, gy1 + group_h - 6.0)
            _rounded_rect(draw, group_bbox, radius=7, fill=theme.panel_alt_fill if group_index % 2 == 0 else theme.panel_fill, outline=theme.chrome_line)
            _draw_text_center_fit(draw, text=str(group), bbox=(group_bbox[0] + 8.0, group_bbox[1] + 4.0, group_bbox[2] - 8.0, group_bbox[1] + 27.0), fill=theme.muted_text, max_size_px=int(render_params.small_font_size_px), bold=True)
            _add_support(support_bboxes, support_records, f"support_ribbon_tab_{tab_index}_group_{group_index}", "ribbon_group", str(group), group_bbox)
            controls_sorted = sorted(controls, key=lambda item: int(item.order_index))
            control_w = (group_bbox[2] - group_bbox[0] - 28.0) / max(1, len(controls_sorted))
            for command_index, control in enumerate(controls_sorted):
                bx1 = group_bbox[0] + 10.0 + command_index * control_w
                bbox = _fit_control_bbox(
                    (bx1, group_bbox[1] + 32.0, bx1 + control_w - 8.0, group_bbox[3] - 8.0),
                    bounds=group_bbox,
                )
                badge_bboxes[str(control.control_id)] = _draw_control_button(draw, bbox=bbox, control=control, render_params=render_params, theme=theme)
                control_bboxes[str(control.control_id)] = bbox_list(bbox)


def _render_navigation_scene(
    image: Image.Image,
    *,
    case: NavigationFlowCase,
    render_params: RenderParams,
    theme: Theme,
) -> tuple[Dict[str, List[float]], Dict[str, List[float]], Dict[str, List[float]], List[Dict[str, Any]], List[Dict[str, Any]], List[float], List[float], AppProfile, Theme]:
    """Draw the full app screen and collect projected geometry."""

    draw = ImageDraw.Draw(image)
    content_bbox, profile = _draw_app_chrome(draw, case=case, render_params=render_params, theme=theme)
    x1, y1, x2, y2 = [float(value) for value in content_bbox]
    title_bar = (x1 + 18.0, y1 + 10.0, x2 - 18.0, y1 + 50.0)
    _rounded_rect(draw, title_bar, radius=10, fill=theme.panel_alt_fill, outline=theme.chrome_line, width=1)
    _draw_text_left(draw, text=str(profile.workspace_title), bbox=(title_bar[0] + 18.0, title_bar[1] + 8.0, title_bar[0] + 440.0, title_bar[3] - 8.0), fill=theme.control_text, max_size_px=int(render_params.body_font_size_px), bold=True)
    draw_text_traced(draw, (title_bar[2] - 145.0, title_bar[1] + 13.0), str(profile.status_text), fill=theme.muted_text, font=load_font(int(render_params.small_font_size_px)), role="readout", required=False)
    workspace = (x1 + 18.0, title_bar[3] + 12.0, x2 - 18.0, y2 - 16.0)
    _rounded_rect(draw, workspace, radius=10, fill=theme.panel_fill, outline=theme.chrome_line)

    control_bboxes: Dict[str, List[float]] = {}
    badge_bboxes: Dict[str, List[float]] = {}
    support_bboxes: Dict[str, List[float]] = {}
    support_records: List[Dict[str, Any]] = []
    draw_kwargs = {
        "case": case,
        "workspace": workspace,
        "render_params": render_params,
        "theme": theme,
        "control_bboxes": control_bboxes,
        "badge_bboxes": badge_bboxes,
        "support_bboxes": support_bboxes,
        "support_records": support_records,
    }
    if str(case.navigation_surface) == MENU_SURFACE:
        _draw_menu_path_scene(draw, **draw_kwargs)
    elif str(case.navigation_surface) == SIDEBAR_SURFACE:
        _draw_sidebar_tree_scene(draw, **draw_kwargs)
    elif str(case.navigation_surface) == RIBBON_SURFACE:
        _draw_ribbon_group_scene(draw, **draw_kwargs)
    else:
        raise ValueError(f"unsupported navigation surface: {case.navigation_surface}")

    control_records: List[Dict[str, Any]] = []
    for control in case.controls:
        control_records.append(
            {
                "control_id": str(control.control_id),
                "candidate_label": str(control.candidate_label),
                "role": str(control.role),
                "display_text": str(control.display_text),
                "nav_kind": str(control.nav_kind),
                "path_keys": [str(value) for value in control.path_keys],
                "order_index": int(control.order_index),
                "bbox_px": list(control_bboxes[str(control.control_id)]),
                "candidate_label_bbox_px": list(badge_bboxes[str(control.control_id)]),
            }
        )
    margin = int(render_params.window_margin_px)
    window_bbox = [
        float(margin),
        float(margin - 6),
        float(render_params.canvas_width - margin),
        float(render_params.canvas_height - margin + 6),
    ]
    return (
        dict(control_bboxes),
        dict(badge_bboxes),
        dict(support_bboxes),
        list(control_records),
        list(support_records),
        [0.0, 0.0, float(render_params.canvas_width), float(render_params.canvas_height)],
        bbox_list(tuple(window_bbox)),
        profile,
        theme,
    )


def render_navigation_flow_case(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    case: NavigationFlowCase,
    namespace: str,
) -> RenderedNavigationFlow:
    """Render one sampled case and preserve finalized geometry for annotation."""

    render_params = resolve_render_params(params, instance_seed=int(instance_seed), namespace=str(namespace))
    information_style, _information_style_meta = resolve_pages_information_style(
        instance_seed=int(instance_seed),
        params=params,
        scene_id="navigation_flow",
    )
    background, background_meta = make_pages_information_background(
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
        style=information_style,
        instance_seed=int(instance_seed),
        namespace=f"{NAMESPACE_ROOT}.{namespace}.background",
    )
    theme = _theme_from_information_style(information_style)
    image = background.copy().convert("RGB")
    (
        control_bboxes,
        badge_bboxes,
        support_bboxes,
        control_records,
        support_records,
        scene_bbox,
        window_bbox,
        profile,
        theme,
    ) = _render_navigation_scene(image, case=case, render_params=render_params, theme=theme)
    image, post_noise_meta = apply_post_image_noise(
        image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    return RenderedNavigationFlow(
        image=image,
        control_bboxes_by_id={str(key): list(value) for key, value in control_bboxes.items()},
        badge_bboxes_by_id={str(key): list(value) for key, value in badge_bboxes.items()},
        support_bboxes_by_id={str(key): list(value) for key, value in support_bboxes.items()},
        control_records=tuple(dict(record) for record in control_records),
        support_records=tuple(dict(record) for record in support_records),
        scene_bbox_px=list(scene_bbox),
        window_bbox_px=list(window_bbox),
        background_meta=dict(background_meta),
        post_noise_meta=dict(post_noise_meta),
        render_params=render_params,
        profile=profile,
        theme=theme,
    )
