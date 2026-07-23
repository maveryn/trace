"""Rendering primitives for grouped GUI control-board pages."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Mapping, Tuple

from PIL import Image, ImageDraw, ImageFont

from trace_tasks.core.seed import hash64
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.pages.shared.gui_chrome import gui_theme_from_information_style
from trace_tasks.tasks.pages.shared.information_style import make_pages_information_background, resolve_pages_information_style
from trace_tasks.tasks.shared.render_variation import resolve_render_int
from trace_tasks.tasks.shared.text_legibility import draw_text_traced
from trace_tasks.tasks.shared.text_rendering import draw_text_centered, fit_font_to_box, load_font

from .defaults import (
    APP_PROFILES,
    DEFAULTS,
    NAMESPACE_ROOT,
    POST_IMAGE_NOISE_DEFAULTS,
    RENDERING_DEFAULTS,
)
from .state import (
    BBox,
    Color,
    ControlBoardCase,
    ControlBoardRenderParams,
    ControlBoardTheme,
    ControlSpec,
    RenderedControlBoard,
)


def _bbox_list(bbox: BBox) -> List[float]:
    return [round(float(value), 3) for value in bbox]


def _measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[float, float]:
    try:
        bbox = draw.textbbox((0, 0), str(text), font=font)
        return (float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1]))
    except Exception:
        width, height = draw.textsize(str(text), font=font)
        return (float(width), float(height))


def _draw_text_left(
    draw: ImageDraw.ImageDraw,
    *,
    text: str,
    bbox: BBox,
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
    _width, height = _measure_text(draw, str(text), font)
    y = float(y1) + max(0.0, (float(y2 - y1) - float(height)) / 2.0) - 1.0
    draw_text_traced(draw, (float(x1), float(y)), str(text), fill=fill, font=font, role="readout", required=False)


def _rounded_rect(
    draw: ImageDraw.ImageDraw,
    bbox: BBox,
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


def _theme_from_information_style(style: Any) -> ControlBoardTheme:
    """Map the shared Pages information style into control-board-specific roles."""

    theme = gui_theme_from_information_style(style)
    return ControlBoardTheme(
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
        disabled_fill=tuple(theme.panel_alt_fill),
        disabled_outline=tuple(theme.chrome_line),
        selected_fill=tuple(theme.selected_fill),
        selected_outline=tuple(theme.accent),
        accent=tuple(theme.accent),
        accent_alt=tuple(theme.accent_alt),
        badge_fill=tuple(theme.badge_fill),
        badge_text=tuple(theme.badge_text),
        workspace_line=tuple(theme.chrome_line),
    )


def resolve_control_board_render_params(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
) -> ControlBoardRenderParams:
    """Resolve scene render params without public task identity."""

    fallback = asdict(DEFAULTS)

    def int_value(key: str) -> int:
        return resolve_render_int(
            params,
            RENDERING_DEFAULTS,
            str(key),
            int(fallback[str(key)]),
            instance_seed=int(instance_seed),
            namespace=NAMESPACE_ROOT,
        )

    return ControlBoardRenderParams(
        canvas_width=int_value("canvas_width"),
        canvas_height=int_value("canvas_height"),
        window_margin_px=int_value("window_margin_px"),
        title_bar_height_px=int_value("title_bar_height_px"),
        menu_bar_height_px=int_value("menu_bar_height_px"),
        corner_radius_px=int_value("corner_radius_px"),
        control_corner_radius_px=int_value("control_corner_radius_px"),
        control_outline_width_px=int_value("control_outline_width_px"),
        badge_size_px=int_value("badge_size_px"),
        title_font_size_px=int_value("title_font_size_px"),
        body_font_size_px=int_value("body_font_size_px"),
        small_font_size_px=int_value("small_font_size_px"),
        label_font_size_px=int_value("label_font_size_px"),
    )


def _draw_app_chrome(
    draw: ImageDraw.ImageDraw,
    *,
    case: ControlBoardCase,
    render_params: ControlBoardRenderParams,
    theme: ControlBoardTheme,
) -> Tuple[BBox, Any]:
    """Draw the outer desktop-window chrome and return the content box."""

    profile = APP_PROFILES[str(case.scene_variant)]
    m = int(render_params.window_margin_px)
    width = int(render_params.canvas_width)
    height = int(render_params.canvas_height)
    window = (float(m), float(m - 6), float(width - m), float(height - m + 6))
    _rounded_rect(draw, window, radius=int(render_params.corner_radius_px), fill=theme.app_fill, outline=theme.chrome_line, width=2)

    title_bar = (window[0], window[1], window[2], window[1] + int(render_params.title_bar_height_px))
    draw.rounded_rectangle(
        [title_bar[0], title_bar[1], title_bar[2], title_bar[3] + int(render_params.corner_radius_px)],
        radius=int(render_params.corner_radius_px),
        fill=theme.title_bar,
    )
    draw.rectangle(
        [title_bar[0], title_bar[3] - int(render_params.corner_radius_px), title_bar[2], title_bar[3]],
        fill=theme.title_bar,
    )
    dot_y = (title_bar[1] + title_bar[3]) / 2.0
    for idx, color in enumerate(((221, 91, 84), (229, 174, 65), (88, 174, 104))):
        draw.ellipse([window[0] + 18 + idx * 22, dot_y - 6, window[0] + 30 + idx * 22, dot_y + 6], fill=color)
    title_font = load_font(int(render_params.title_font_size_px), bold=True)
    draw_text_traced(draw, (window[0] + 98, title_bar[1] + 10), str(profile.app_title), fill=theme.title_text, font=title_font, role="readout", required=False)
    small_font = load_font(int(render_params.small_font_size_px), bold=False)
    draw_text_traced(draw, (window[2] - 255, title_bar[1] + 16), str(profile.window_title), fill=theme.title_text, font=small_font, role="readout", required=False)

    menu_y1 = title_bar[3]
    menu_y2 = menu_y1 + int(render_params.menu_bar_height_px)
    draw.rectangle([window[0], menu_y1, window[2], menu_y2], fill=theme.panel_fill, outline=theme.chrome_line)
    tab_font = load_font(int(render_params.small_font_size_px), bold=True)
    tab_x = window[0] + 26
    for idx, tab in enumerate(("File", str(profile.primary_tab), str(profile.secondary_tab), "View", "Help")):
        fill = theme.accent if idx == 1 else theme.muted_text
        draw_text_traced(draw, (tab_x, menu_y1 + 9), tab, fill=fill, font=tab_font, role="readout", required=False)
        tab_x += 86 if idx else 64

    content_bbox = (window[0] + 22, menu_y2 + 18, window[2] - 22, window[3] - 18)
    return content_bbox, profile


def _draw_mini_icon(
    draw: ImageDraw.ImageDraw,
    *,
    kind: str,
    bbox: BBox,
    stroke: Color,
    accent: Color,
    disabled: bool,
) -> None:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    color = stroke
    accent_color = accent if not bool(disabled) else stroke
    selector = int(abs(hash64(0, str(kind), 19)) % 6)
    if selector == 0:
        draw.rectangle([x1 + 5, y1 + 6, x2 - 5, y2 - 6], outline=color, width=2)
        draw.line([(x1 + 9, cy), (x2 - 9, cy)], fill=accent_color, width=2)
    elif selector == 1:
        draw.ellipse([x1 + 5, y1 + 5, x2 - 9, y2 - 9], outline=color, width=2)
        draw.line([(cx + 6, cy + 6), (x2 - 4, y2 - 4)], fill=accent_color, width=2)
    elif selector == 2:
        draw.polygon([(cx, y1 + 4), (x2 - 5, cy), (cx, y2 - 4), (x1 + 5, cy)], outline=color)
        draw.line([(x1 + 10, cy), (x2 - 10, cy)], fill=accent_color, width=2)
    elif selector == 3:
        draw.line([(x1 + 7, y2 - 7), (x2 - 7, y1 + 7)], fill=color, width=2)
        draw.line([(x1 + 10, y1 + 10), (x2 - 10, y1 + 10)], fill=accent_color, width=2)
        draw.line([(x1 + 10, y2 - 10), (x2 - 10, y2 - 10)], fill=accent_color, width=2)
    elif selector == 4:
        for idx in range(3):
            yy = y1 + 9 + idx * 9
            draw.line([(x1 + 7, yy), (x2 - 7, yy)], fill=color if idx != 1 else accent_color, width=2)
    else:
        draw.arc([x1 + 6, y1 + 6, x2 - 6, y2 - 6], start=35, end=320, fill=color, width=2)
        draw.polygon([(x2 - 11, y1 + 10), (x2 - 5, y1 + 6), (x2 - 7, y1 + 16)], fill=accent_color)


def _draw_candidate_badge(
    draw: ImageDraw.ImageDraw,
    *,
    control_bbox: BBox,
    label: str,
    render_params: ControlBoardRenderParams,
    theme: ControlBoardTheme,
) -> List[float]:
    size = max(20, int(render_params.badge_size_px))
    x1, y1, _x2, _y2 = [float(value) for value in control_bbox]
    badge = (x1 + 7.0, y1 + 7.0, x1 + 7.0 + float(size), y1 + 7.0 + float(size))
    draw.ellipse([float(value) for value in badge], fill=theme.badge_fill, outline=(255, 255, 255), width=1)
    font = fit_font_to_box(
        draw,
        text=str(label),
        max_width=float(size) * 0.60,
        max_height=float(size) * 0.60,
        bold=False,
        min_size_px=9,
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
    return _bbox_list(badge)


def _draw_control(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: BBox,
    control: ControlSpec,
    render_params: ControlBoardRenderParams,
    theme: ControlBoardTheme,
) -> None:
    """Draw one command tile while preserving state cues and bbox ownership."""

    disabled = not bool(control.enabled)
    selected = bool(control.selected)
    fill = theme.disabled_fill if disabled else (theme.selected_fill if selected else theme.control_fill)
    outline = theme.disabled_outline if disabled else (theme.selected_outline if selected else theme.control_outline)
    width = 3 if selected or bool(control.is_reference) else int(render_params.control_outline_width_px)
    _rounded_rect(
        draw,
        bbox,
        radius=int(render_params.control_corner_radius_px),
        fill=fill,
        outline=outline,
        width=int(width),
    )
    x1, y1, x2, y2 = [float(value) for value in bbox]
    icon_box = (x1 + 12.0, y1 + 34.0, x1 + 46.0, y1 + 68.0)
    text_fill = theme.muted_text if disabled else theme.control_text
    _draw_mini_icon(
        draw,
        kind=str(control.command.icon_kind),
        bbox=icon_box,
        stroke=text_fill,
        accent=theme.accent,
        disabled=bool(disabled),
    )
    _draw_text_left(
        draw,
        text=str(control.command.display_text),
        bbox=(x1 + 54.0, y1 + 37.0, x2 - 12.0, y1 + 65.0),
        fill=text_fill,
        max_size_px=int(render_params.small_font_size_px + 2),
        bold=False,
    )
    if disabled:
        draw.line([(x1 + 10.0, y2 - 11.0), (x2 - 10.0, y1 + 11.0)], fill=theme.disabled_outline, width=3)
    if selected:
        mark_x = x2 - 30.0
        mark_y = y1 + 14.0
        draw.line([(mark_x, mark_y + 8.0), (mark_x + 7.0, mark_y + 15.0), (mark_x + 20.0, mark_y)], fill=theme.selected_outline, width=3)
    if bool(control.is_reference):
        draw.rectangle([x1 + 3.0, y1 + 3.0, x2 - 3.0, y2 - 3.0], outline=theme.accent_alt, width=2)


def _render_control_board_scene(
    image: Image.Image,
    *,
    case: ControlBoardCase,
    render_params: ControlBoardRenderParams,
    theme: ControlBoardTheme,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Draw all visible board elements and return bbox maps plus records."""

    draw = ImageDraw.Draw(image)
    content_bbox, profile = _draw_app_chrome(draw, case=case, render_params=render_params, theme=theme)
    x1, y1, x2, y2 = [float(value) for value in content_bbox]

    title_bar = (x1 + 18.0, y1 + 10.0, x2 - 18.0, y1 + 52.0)
    _rounded_rect(draw, title_bar, radius=10, fill=theme.panel_alt_fill, outline=theme.chrome_line, width=1)
    _draw_text_left(
        draw,
        text=str(profile.workspace_title),
        bbox=(title_bar[0] + 18.0, title_bar[1] + 8.0, title_bar[0] + 360.0, title_bar[3] - 8.0),
        fill=theme.control_text,
        max_size_px=int(render_params.body_font_size_px),
        bold=True,
    )
    status_font = load_font(int(render_params.small_font_size_px), bold=False)
    draw_text_traced(draw, (title_bar[2] - 150.0, title_bar[1] + 14.0), str(profile.status_text), fill=theme.muted_text, font=status_font, role="readout", required=False)

    board = (x1 + 18.0, title_bar[3] + 14.0, x2 - 18.0, y2 - 16.0)
    group_gap = 18.0
    group_w = (board[2] - board[0] - group_gap) / 2.0
    group_h = (board[3] - board[1] - group_gap) / 2.0
    group_bboxes: Dict[str, List[float]] = {}
    control_bboxes: Dict[str, List[float]] = {}
    badge_bboxes: Dict[str, List[float]] = {}

    controls_by_group: Dict[str, List[ControlSpec]] = {str(name): [] for name in case.group_names}
    for control in case.controls:
        controls_by_group[str(control.group_name)].append(control)

    group_font = load_font(int(render_params.body_font_size_px), bold=True)
    count_font = load_font(int(render_params.small_font_size_px), bold=False)
    for group_index, group_name in enumerate(case.group_names):
        row = int(group_index) // 2
        col = int(group_index) % 2
        gx1 = board[0] + col * (group_w + group_gap)
        gy1 = board[1] + row * (group_h + group_gap)
        group_bbox = (gx1, gy1, gx1 + group_w, gy1 + group_h)
        panel_fill = theme.panel_alt_fill if group_index % 2 == 0 else theme.panel_fill
        _rounded_rect(draw, group_bbox, radius=12, fill=panel_fill, outline=theme.chrome_line, width=1)
        draw_text_traced(draw, (gx1 + 18.0, gy1 + 14.0), str(group_name), fill=theme.control_text, font=group_font, role="readout", required=False)
        draw_text_traced(draw, (group_bbox[2] - 92.0, gy1 + 18.0), f"{len(controls_by_group[str(group_name)])} controls", fill=theme.muted_text, font=count_font, role="readout", required=False)
        group_bboxes[str(group_name)] = _bbox_list(group_bbox)

        controls = controls_by_group[str(group_name)]
        cols = 4
        gap = 10.0
        pad_x = 17.0
        control_w = (group_w - (2 * pad_x) - (gap * (cols - 1))) / float(cols)
        control_h = 82.0
        start_y = gy1 + 55.0
        for local_index, control in enumerate(controls):
            c_row = int(local_index) // cols
            c_col = int(local_index) % cols
            cx1 = gx1 + pad_x + c_col * (control_w + gap)
            cy1 = start_y + c_row * (control_h + gap)
            bbox = (cx1, cy1, cx1 + control_w, cy1 + control_h)
            _draw_control(draw, bbox=bbox, control=control, render_params=render_params, theme=theme)
            badge_bboxes[str(control.control_id)] = _draw_candidate_badge(
                draw,
                control_bbox=bbox,
                label=str(control.candidate_label),
                render_params=render_params,
                theme=theme,
            )
            control_bboxes[str(control.control_id)] = _bbox_list(bbox)

    control_records: List[Dict[str, Any]] = []
    for control in case.controls:
        control_records.append(
            {
                "control_id": str(control.control_id),
                "candidate_label": str(control.candidate_label),
                "group_name": str(control.group_name),
                "group_index": int(control.group_index),
                "order_in_group": int(control.order_in_group),
                "global_order_index": int(control.global_order_index),
                "command_key": str(control.command.command_key),
                "display_text": str(control.command.display_text),
                "icon_kind": str(control.command.icon_kind),
                "enabled": bool(control.enabled),
                "selected": bool(control.selected),
                "is_reference": bool(control.is_reference),
                "bbox_px": list(control_bboxes[str(control.control_id)]),
                "candidate_label_bbox_px": list(badge_bboxes[str(control.control_id)]),
            }
        )

    m = int(render_params.window_margin_px)
    window_bbox = [float(m), float(m - 6), float(render_params.canvas_width - m), float(render_params.canvas_height - m + 6)]
    return (
        {
            "control_bboxes_by_id": {str(key): list(value) for key, value in control_bboxes.items()},
            "badge_bboxes_by_id": {str(key): list(value) for key, value in badge_bboxes.items()},
            "group_bboxes_by_name": {str(key): list(value) for key, value in group_bboxes.items()},
            "control_records": tuple(control_records),
            "scene_bbox_px": [0.0, 0.0, float(render_params.canvas_width), float(render_params.canvas_height)],
            "window_bbox_px": _bbox_list(tuple(window_bbox)),
        },
        {"profile": profile, "theme": theme},
    )


def render_control_board_case(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    case: ControlBoardCase,
) -> RenderedControlBoard:
    """Render a sampled board case into an image and projection metadata."""

    render_params = resolve_control_board_render_params(params=params, instance_seed=int(instance_seed))
    information_style, _information_style_meta = resolve_pages_information_style(
        instance_seed=int(instance_seed),
        params=params,
        scene_id="control_board",
    )
    background, background_meta = make_pages_information_background(
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
        style=information_style,
        instance_seed=int(instance_seed),
        namespace=f"{NAMESPACE_ROOT}.background",
    )
    theme = _theme_from_information_style(information_style)
    image = background.copy().convert("RGB")
    bbox_payload, style_payload = _render_control_board_scene(
        image,
        case=case,
        render_params=render_params,
        theme=theme,
    )
    image, post_noise_meta = apply_post_image_noise(
        image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    return RenderedControlBoard(
        image=image,
        render_params=render_params,
        control_bboxes_by_id=dict(bbox_payload["control_bboxes_by_id"]),
        badge_bboxes_by_id=dict(bbox_payload["badge_bboxes_by_id"]),
        group_bboxes_by_name=dict(bbox_payload["group_bboxes_by_name"]),
        control_records=tuple(bbox_payload["control_records"]),
        scene_bbox_px=list(bbox_payload["scene_bbox_px"]),
        window_bbox_px=list(bbox_payload["window_bbox_px"]),
        profile=style_payload["profile"],
        theme=style_payload["theme"],
        background_meta=dict(background_meta),
        post_noise_meta=dict(post_noise_meta),
    )
