"""Shared GUI rendering helper bindings for workspace page tasks."""

from __future__ import annotations

from typing import Any, List, Mapping, Sequence, Tuple

from PIL import ImageDraw

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.pages.shared.gui_chrome import (
    BBox,
    Color,
    GUI_APP_PROFILES,
    GuiAppProfile,
    GuiTheme,
    SUPPORTED_GUI_SCENE_VARIANTS,
    bbox_list,
    clamp_unit,
    draw_app_chrome,
    draw_control_button,
    draw_text_center_fit,
    draw_text_left,
    gui_theme_from_information_style,
    rounded_rect,
)
from trace_tasks.tasks.shared.config_defaults import group_default, split_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.text_rendering import draw_text_centered, fit_font_to_box


SUPPORTED_SCENE_VARIANTS: Tuple[str, ...] = SUPPORTED_GUI_SCENE_VARIANTS

_TASK_GROUP_DEFAULTS = get_scene_defaults("pages", "workspace")
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = split_generation_rendering_prompt_defaults(
    _TASK_GROUP_DEFAULTS if isinstance(_TASK_GROUP_DEFAULTS, Mapping) else {},
)

_Theme = GuiTheme
_AppProfile = GuiAppProfile
_APP_PROFILES = GUI_APP_PROFILES
_clamp_unit = clamp_unit
_bbox_list = bbox_list
_draw_text_left = draw_text_left
_draw_text_center_fit = draw_text_center_fit
_rounded_rect = rounded_rect
_theme_from_information_style = gui_theme_from_information_style


def _normalize_str_support(params: Mapping[str, Any], key: str, fallback: Sequence[str]) -> Tuple[str, ...]:
    raw_values = params.get(str(key), group_default(_GEN_DEFAULTS, str(key), fallback))
    support: List[str] = []
    for raw_value in raw_values:
        value = str(raw_value).strip()
        if value and value not in support:
            support.append(value)
    if not support:
        raise ValueError(f"{key} must not be empty for GUI scene tasks")
    return tuple(str(value) for value in support)


def _draw_app_chrome(
    draw: ImageDraw.ImageDraw,
    *,
    query: Any,
    render_params: Any,
    theme: _Theme,
) -> Tuple[BBox, _AppProfile]:
    return draw_app_chrome(
        draw,
        scene_variant=str(query.scene_variant),
        render_params=render_params,
        theme=theme,
    )


def _draw_badge(
    draw: ImageDraw.ImageDraw,
    *,
    control_bbox: BBox,
    label: str,
    render_params: Any,
    theme: _Theme,
) -> List[float]:
    """Draw a lighter workspace option badge than the generic GUI badge."""

    x1, y1, x2, y2 = [float(value) for value in control_bbox]
    available_size = max(16, int(min(float(x2 - x1), float(y2 - y1)) - 8.0))
    size = max(16, min(max(20, int(render_params.badge_size_px)), int(available_size)))
    badge = (x1 + 5.0, y1 + 5.0, x1 + 5.0 + float(size), y1 + 5.0 + float(size))
    draw.ellipse(
        [float(value) for value in badge],
        fill=theme.badge_fill,
        outline=theme.control_outline,
        width=1,
    )
    label_box = float(size) * 0.58
    max_label_size = max(10, min(int(render_params.label_font_size_px), int(round(label_box))))
    font = fit_font_to_box(
        draw,
        text=str(label),
        max_width=label_box,
        max_height=label_box,
        bold=False,
        min_size_px=8,
        max_size_px=int(max_label_size),
        fill_ratio=0.9,
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
    control: Any,
    render_params: Any,
    theme: _Theme,
) -> List[float]:
    return draw_control_button(
        draw,
        bbox=bbox,
        control=control,
        render_params=render_params,
        theme=theme,
    )
