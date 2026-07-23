"""Chart-domain adapter for shared structured-information styling."""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from typing import Any, Mapping, Sequence

from ....core.scene_config import get_scene_defaults
from ...shared.config_defaults import split_scene_generation_rendering_prompt_defaults
from ...shared.visual_style.information_scene import (
    Color,
    INFORMATION_SCENE_TREATMENT_IDS,
    InformationSceneStyle,
    make_information_scene_background,
    resolve_information_scene_style_from_request,
)
from ...shared.visual_style.request import (
    build_visual_style_request,
    resolve_style_bool,
)
from .chart_scene_types import ChartRenderParams


ChartInformationStyle = InformationSceneStyle

INFORMATION_STYLE_RENDER_ROLE_MAP: dict[str, str] = {
    "axis_color_rgb": "axis_rgb",
    "axis_rgb": "axis_rgb",
    "grid_color_rgb": "grid_rgb",
    "grid_rgb": "grid_rgb",
    "guide_line_color_rgb": "guide_rgb",
    "guide_rgb": "guide_rgb",
    "plot_fill_rgb": "surface_rgb",
    "surface_rgb": "surface_rgb",
    "panel_rgb": "surface_rgb",
    "panel_fill_rgb": "panel_fill_rgb",
    "panel_border_rgb": "panel_border_rgb",
    "panel_outline_rgb": "panel_border_rgb",
    "map_border_rgb": "panel_border_rgb",
    "region_border_rgb": "guide_rgb",
    "legend_border_rgb": "panel_border_rgb",
    "legend_fill_rgb": "surface_alt_rgb",
    "mark_outline_rgb": "axis_rgb",
    "marker_outline_rgb": "axis_rgb",
    "bar_edge_rgb": "axis_rgb",
    "card_fill_rgb": "panel_fill_rgb",
    "card_alt_fill_rgb": "surface_alt_rgb",
    "card_outline_rgb": "panel_border_rgb",
    "border_color_rgb": "panel_border_rgb",
    "header_fill_rgb": "header_rgb",
    "header_dark_fill_rgb": "header_rgb",
    "header_dark_text_rgb": "header_text_rgb",
    "zebra_row_fill_rgb": "surface_alt_rgb",
    "shadow_color_rgb": "shadow_rgb",
    "separator_rgb": "guide_rgb",
    "grid_border_rgb": "guide_rgb",
    "cell_border_rgb": "guide_rgb",
    "row_line_rgb": "guide_rgb",
    "spoke_rgb": "guide_rgb",
    "track_rgb": "guide_rgb",
    "tick_rgb": "muted_text_rgb",
    "needle_rgb": "axis_rgb",
    "node_fill_rgb": "panel_fill_rgb",
    "node_border_rgb": "panel_border_rgb",
    "node_text_rgb": "text_rgb",
    "source_node_fill_rgb": "header_rgb",
    "target_node_fill_rgb": "panel_fill_rgb",
    "ring_line_rgb": "guide_rgb",
    "value_label_fill_rgb": "panel_fill_rgb",
    "value_label_border_rgb": "panel_border_rgb",
    "value_label_text_rgb": "text_rgb",
    "connector_rgb": "connector_rgb",
    "text_color_rgb": "text_rgb",
    "text_rgb": "text_rgb",
    "muted_text_rgb": "muted_text_rgb",
    "muted_rgb": "muted_text_rgb",
    "axis_text_rgb": "text_rgb",
    "legend_text_rgb": "text_rgb",
    "header_text_rgb": "header_text_rgb",
    "subtitle_rgb": "muted_text_rgb",
    "title_rgb": "text_rgb",
    "title_color_rgb": "text_rgb",
    "text_stroke_rgb": "text_stroke_rgb",
    "label_stroke_rgb": "text_stroke_rgb",
    "threshold_rgb": "highlight_rgb",
    "threshold_line_rgb": "highlight_rgb",
    "threshold_label_rgb": "highlight_rgb",
    "threshold_guide_fill_rgb": "panel_fill_rgb",
    "hex_outline_rgb": "surface_rgb",
    "reference_rgb": "highlight_rgb",
}


@lru_cache(maxsize=64)
def _chart_scene_render_defaults(scene_id: str) -> dict[str, Any]:
    """Return shared chart rendering defaults for scene-style resolution."""

    try:
        defaults = get_scene_defaults("charts", str(scene_id))
    except Exception:
        return {}
    if not isinstance(defaults, Mapping):
        return {}
    _gen, rendering, _prompt = split_scene_generation_rendering_prompt_defaults(
        defaults,
        task_id=f"charts_{str(scene_id)}_style_defaults",
    )
    return dict(rendering)


def resolve_chart_information_style(
    *,
    instance_seed: int,
    params: Mapping[str, Any] | None,
    scene_id: str,
    protected_colors: Sequence[Color] | None = None,
    allow_dark: bool | None = None,
    allow_colored_surface: bool | None = None,
) -> tuple[ChartInformationStyle, dict[str, Any]]:
    """Resolve one chart presentation style without changing semantic marks."""

    routing_key = str(scene_id)
    default_params = _chart_scene_render_defaults(str(scene_id))
    resolved_params = {**default_params, **dict(params or {})}
    resolved_allow_dark = (
        resolve_style_bool(resolved_params, "information_scene_allow_dark", False)
        if allow_dark is None
        else bool(allow_dark)
    )
    resolved_allow_colored_surface = (
        resolve_style_bool(resolved_params, "information_scene_allow_colored_surface", True)
        if allow_colored_surface is None
        else bool(allow_colored_surface)
    )
    request = build_visual_style_request(
        domain="charts",
        scene_id=str(scene_id),
        routing_key=str(routing_key),
        instance_seed=int(instance_seed),
        params=resolved_params,
        style_family="information_scene",
        allow_dark=bool(resolved_allow_dark),
        allow_colored_surface=bool(resolved_allow_colored_surface),
        protected_colors=protected_colors or (),
        required_text_roles=("chart_label", "axis_tick", "legend_label"),
    )
    return resolve_information_scene_style_from_request(
        request,
        treatments=resolved_params.get("information_scene_treatments"),
        treatment_weights=resolved_params.get("information_scene_treatment_weights", {}),
        palettes=resolved_params.get("information_scene_palettes"),
        palette_weights=resolved_params.get("information_scene_palette_weights", {}),
        chrome_modes=resolved_params.get("information_scene_chrome_modes"),
        chrome_mode_weights=resolved_params.get("information_scene_chrome_mode_weights", {}),
    )


def make_chart_information_background(
    *,
    canvas_width: int,
    canvas_height: int,
    style: ChartInformationStyle,
    instance_seed: int,
    namespace: str,
) -> tuple[Any, dict[str, Any]]:
    """Create a chart background from the shared information style."""

    return make_information_scene_background(
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        style=style,
        instance_seed=int(instance_seed),
        namespace=str(namespace),
    )


def make_chart_background_canvas(
    *,
    canvas_width: int | None = None,
    canvas_height: int | None = None,
    canvas_size: int | None = None,
    instance_seed: int,
    params: Mapping[str, Any] | None,
    scene_id: str,
    protected_colors: Sequence[Color] | None = None,
    default_config: Mapping[str, Any] | None = None,
    fallback_color: Sequence[int] | None = None,
    allow_dark: bool | None = None,
    allow_colored_surface: bool | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Create a chart-domain information-scene background.

    This mirrors the old background helper's call shape for chart renderers,
    while making the chart-domain information-scene treatment inventory the
    default background source. ``default_config`` is accepted for transitional
    scene callers; visual background styles now resolve through chart
    information-scene config instead of per-scene solid/grid background lists.
    """

    del default_config, fallback_color
    if canvas_size is not None:
        width = max(1, int(canvas_size))
        height = max(1, int(canvas_size))
    else:
        width = max(1, int(canvas_width) if canvas_width is not None else 1)
        height = max(1, int(canvas_height) if canvas_height is not None else width)
    style, style_meta = resolve_chart_information_style(
        instance_seed=int(instance_seed),
        params=params or {},
        scene_id=str(scene_id),
        protected_colors=protected_colors or (),
        allow_dark=allow_dark,
        allow_colored_surface=allow_colored_surface,
    )
    image, metadata = make_chart_information_background(
        canvas_width=int(width),
        canvas_height=int(height),
        style=style,
        instance_seed=int(instance_seed),
        namespace=f"charts.{str(scene_id)}.information_scene_background",
    )
    out = dict(metadata)
    out.setdefault("available_styles", [f"information_scene_style:{item}" for item in INFORMATION_SCENE_TREATMENT_IDS])
    out["information_scene_style"] = dict(style_meta)
    return image, out


def apply_chart_information_style(
    render_params: ChartRenderParams,
    style: ChartInformationStyle,
) -> ChartRenderParams:
    """Map non-semantic shared style roles into chart render parameters."""

    return apply_chart_information_style_roles(render_params, style)


def apply_chart_information_style_roles(
    render_params: Any,
    style: ChartInformationStyle,
    *,
    role_map: Mapping[str, str] | None = None,
) -> Any:
    """Apply information-scene roles to matching non-semantic render fields."""

    updates: dict[str, Any] = {}
    for field_name, role_name in dict(role_map or INFORMATION_STYLE_RENDER_ROLE_MAP).items():
        if not hasattr(render_params, str(field_name)):
            continue
        role_value = getattr(style, str(role_name))
        updates[str(field_name)] = tuple(int(value) for value in role_value)
    if not updates:
        return render_params
    return replace(render_params, **updates)


def prepare_chart_information_scene(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    scene_id: str,
    render_params: ChartRenderParams,
    protected_colors: Sequence[Color] | None = None,
    allow_dark: bool | None = None,
    allow_colored_surface: bool | None = None,
) -> tuple[ChartRenderParams, Any, dict[str, Any], dict[str, Any]]:
    """Resolve chart information style, apply it, and create the background."""

    style, style_meta = resolve_chart_information_style(
        instance_seed=int(instance_seed),
        params=params,
        scene_id=str(scene_id),
        protected_colors=protected_colors or (),
        allow_dark=allow_dark,
        allow_colored_surface=allow_colored_surface,
    )
    styled_render_params = apply_chart_information_style(render_params, style)
    background, background_meta = make_chart_information_background(
        canvas_width=int(styled_render_params.canvas_width),
        canvas_height=int(styled_render_params.canvas_height),
        style=style,
        instance_seed=int(instance_seed),
        namespace=f"charts.{str(scene_id)}.information_scene_background",
    )
    return styled_render_params, background, background_meta, style_meta


__all__ = [
    "ChartInformationStyle",
    "apply_chart_information_style",
    "apply_chart_information_style_roles",
    "INFORMATION_STYLE_RENDER_ROLE_MAP",
    "make_chart_background_canvas",
    "make_chart_information_background",
    "prepare_chart_information_scene",
    "resolve_chart_information_style",
]
