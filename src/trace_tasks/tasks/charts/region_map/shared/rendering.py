"""Base choropleth map rendering helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from trace_tasks.tasks.charts.shared.dense_text import dense_fit_bold, dense_stroke_width
from trace_tasks.tasks.charts.shared.information_style import prepare_chart_information_scene
from .....core.visual.noise import apply_post_image_noise
from .....core.seed import spawn_rng
from ....shared.font_assets import font_asset_version, sample_font_family
from ....shared.bbox_projection import bbox_union_raw as _bbox_union, round_bbox as _round_bbox
from ....shared.color_distance import coerce_rgb as _rgb
from ....shared.config_defaults import required_group_defaults, resolve_required_int_bounds
from ....shared.deterministic_sampling import resolve_selection_index, uniform_probability_map
from ....shared.drawing import draw_centered_text, draw_rounded_rect
from ....shared.render_variation import apply_layout_jitter_to_margins
from ....shared.text_rendering import fit_font_to_box, load_font, temporary_default_font_family
from ...shared.label_assets import resolve_chart_category_labels
from ...shared.visual_defaults import (
    render_style_seed as _render_style_seed,
    resolve_chart_render_rgb,
)
from .assets import (
    GEOGRAPHIC_MAP_ASSETS as _GEOGRAPHIC_MAP_ASSETS,
    WORLD_CATEGORY_TITLE_OPTIONS as _WORLD_CATEGORY_TITLE_OPTIONS,
    WORLD_MAP_ASSET_ID as _WORLD_MAP_ASSET_ID,
    WORLD_TITLE_OPTIONS as _WORLD_TITLE_OPTIONS,
    load_geographic_map_asset as _load_geographic_map_asset,
    load_world_map_asset as _load_world_map_asset,
)
from .defaults import (
    POST_IMAGE_NOISE_DEFAULTS,
    SCENE_NAMESPACE,
    _RENDER_DEFAULTS,
)
from .spatial_primitives import (
    _balanced_int,
    _choose_random,
    _grid_pair_support,
    _grid_points,
    _neighbors,
    _polygon_bbox,
    _polygon_center,
    _reading_order_region_ids,
    _region_polygon,
    _region_sort_key,
    _sample_connected_cells,
    _shrink_polygon,
)
from .projection import (
    WORLD_FILTERED_CONTINENTS as _WORLD_FILTERED_CONTINENTS,
    _border_segment_key,
    _border_segment_length,
    _centroid_lonlat_from_rings,
    _geographic_border_neighbors,
    _geographic_shared_border_lengths,
    _region_boundary_segments,
    _selected_geographic_region_adjacency,
    _synthetic_region_adjacency,
    _world_country_shared_border_lengths,
    _world_filtered_region_candidates,
)
from .styles import (
    resolve_choropleth_legend_position as _resolve_legend_position,
    resolve_choropleth_marker_style as _resolve_marker_style,
    resolve_choropleth_palette as _resolve_palette,
    resolve_choropleth_world_map_style as _resolve_world_map_style,
)

Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]

@dataclass(frozen=True)
class _MapRenderParams:
    canvas_width: int
    canvas_height: int
    outer_margin_px: int
    panel_padding_px: int
    title_band_height_px: int
    legend_width_px: int
    legend_height_px: int
    map_legend_gap_px: int
    region_gap_px: int
    region_border_width_px: int
    label_font_size_px: int
    legend_font_size_px: int
    title_font_size_px: int
    legend_position: str
    legend_position_probabilities: Dict[str, float]
    panel_fill_rgb: Tuple[int, int, int]
    panel_border_rgb: Tuple[int, int, int]
    title_color_rgb: Tuple[int, int, int]
    map_border_rgb: Tuple[int, int, int]
    region_border_rgb: Tuple[int, int, int]
    legend_fill_rgb: Tuple[int, int, int]
    legend_text_rgb: Tuple[int, int, int]
    map_palette_rgb: Tuple[Tuple[int, int, int], ...]
    map_palette_variant: str
    map_palette_variant_probabilities: Dict[str, float]
    world_map_style_id: str
    world_map_style_probabilities: Dict[str, float]
    world_map_style: Dict[str, Any]
    layout_offset_x_px: int
    layout_offset_y_px: int
    layout_jitter_meta: Dict[str, Any]

@dataclass(frozen=True)
class _RenderedChoroplethMap:
    image: Image.Image
    entities: Tuple[Dict[str, Any], ...]
    panel_bbox_px: List[float]
    title_bbox_px: List[float]
    map_bbox_px: List[float]
    legend_bbox_px: List[float]
    region_bbox_map: Dict[str, List[float]]
    region_center_map: Dict[str, List[float]]
    legend_entry_bbox_map: Dict[str, List[float]]
    render_meta: Dict[str, Any]

def _rgb_param(params: Mapping[str, Any], key: str, fallback: Tuple[int, int, int]) -> Tuple[int, int, int]:
    return resolve_chart_render_rgb(params, _RENDER_DEFAULTS, str(key), fallback, namespace=SCENE_NAMESPACE)

def _int_param(params: Mapping[str, Any], key: str, fallback: int) -> int:
    raw = params.get(key, _RENDER_DEFAULTS.get(key, fallback))
    return int(raw)

def _resolve_render_params(
    params: Mapping[str, Any],
    *,
    legend_count: int,
    categorical: bool = False,
) -> _MapRenderParams:
    """Resolve scene-wide render style while preserving map/legend layout invariants."""

    outer = _int_param(params, "outer_margin_px", 42)
    jitter_left, _jitter_right, jitter_top, _jitter_bottom, layout_jitter_meta = apply_layout_jitter_to_margins(
        left_px=int(outer),
        right_px=int(outer),
        top_px=int(outer),
        bottom_px=int(outer),
        params=params,
        defaults=_RENDER_DEFAULTS,
        instance_seed=_render_style_seed(params),
        namespace=f"{SCENE_NAMESPACE}.layout",
    )
    palette_variant, palette_probabilities, palette = _resolve_palette(
        params,
        render_defaults=_RENDER_DEFAULTS,
        namespace=SCENE_NAMESPACE,
        style_seed=_render_style_seed(params),
        required_palette_count=int(legend_count),
        categorical=bool(categorical),
    )
    legend_position, legend_position_probabilities = _resolve_legend_position(
        params,
        render_defaults=_RENDER_DEFAULTS,
        namespace=SCENE_NAMESPACE,
        instance_seed=_render_style_seed(params),
    )
    world_map_style_id, world_map_style_probabilities, world_map_style = _resolve_world_map_style(
        params,
        render_defaults=_RENDER_DEFAULTS,
        namespace=SCENE_NAMESPACE,
        instance_seed=_render_style_seed(params),
    )
    return _MapRenderParams(
        canvas_width=_int_param(params, "canvas_width", 1180),
        canvas_height=_int_param(params, "canvas_height", 760),
        outer_margin_px=int(outer),
        panel_padding_px=_int_param(params, "panel_padding_px", 26),
        title_band_height_px=_int_param(params, "title_band_height_px", 62),
        legend_width_px=_int_param(params, "legend_width_px", 238),
        legend_height_px=_int_param(params, "legend_height_px", 118),
        map_legend_gap_px=_int_param(params, "map_legend_gap_px", 32),
        region_gap_px=_int_param(params, "region_gap_px", 3),
        region_border_width_px=_int_param(params, "region_border_width_px", 3),
        label_font_size_px=_int_param(params, "label_font_size_px", 18),
        legend_font_size_px=_int_param(params, "legend_font_size_px", 17),
        title_font_size_px=_int_param(params, "title_font_size_px", 28),
        legend_position=str(legend_position),
        legend_position_probabilities=dict(legend_position_probabilities),
        panel_fill_rgb=_rgb_param(params, "panel_fill_rgb", (252, 253, 251)),
        panel_border_rgb=_rgb_param(params, "panel_border_rgb", (72, 82, 92)),
        title_color_rgb=_rgb_param(params, "title_color_rgb", (35, 42, 50)),
        map_border_rgb=_rgb_param(params, "map_border_rgb", (74, 84, 94)),
        region_border_rgb=_rgb_param(params, "region_border_rgb", (255, 255, 255)),
        legend_fill_rgb=_rgb_param(params, "legend_fill_rgb", (255, 255, 255)),
        legend_text_rgb=_rgb_param(params, "legend_text_rgb", (36, 42, 50)),
        map_palette_rgb=tuple(palette),
        map_palette_variant=str(palette_variant),
        map_palette_variant_probabilities=dict(palette_probabilities),
        world_map_style_id=str(world_map_style_id),
        world_map_style_probabilities=dict(world_map_style_probabilities),
        world_map_style=dict(world_map_style),
        layout_offset_x_px=int(jitter_left) - int(outer),
        layout_offset_y_px=int(jitter_top) - int(outer),
        layout_jitter_meta=dict(layout_jitter_meta),
    )

def _color_luminance(color: Sequence[int]) -> float:
    red, green, blue = (int(color[0]), int(color[1]), int(color[2]))
    return (0.299 * float(red)) + (0.587 * float(green)) + (0.114 * float(blue))

def _text_fill_for_color(color: Sequence[int]) -> Tuple[int, int, int]:
    return (255, 255, 255) if _color_luminance(color) < 145.0 else (26, 35, 44)

def _text_stroke_for_color(color: Sequence[int]) -> Tuple[int, int, int]:
    return (24, 32, 40) if _color_luminance(color) < 145.0 else (255, 255, 255)

def _region_value_label_text(region: Mapping[str, Any]) -> str:
    value = region.get("region_value")
    if value is None:
        return ""
    label = str(region.get("region_label") or "").strip()
    if label:
        return f"{label}:{int(value)}"
    return str(int(value))

def _draw_region_value_label(
    draw: ImageDraw.ImageDraw,
    *,
    region: Mapping[str, Any],
    center: Sequence[float],
    bbox: Sequence[float],
    fill_color: Sequence[int],
    render_params: _MapRenderParams,
) -> List[float] | None:
    text = _region_value_label_text(region)
    if not text:
        return None
    width = max(1.0, float(bbox[2]) - float(bbox[0]))
    height = max(1.0, float(bbox[3]) - float(bbox[1]))
    max_width = max(18.0, min(float(width) - 4.0, 74.0))
    max_height = max(10.0, min(float(height) - 4.0, 24.0))
    font = fit_font_to_box(
        draw,
        text=str(text),
        max_width=float(max_width),
        max_height=float(max_height),
        bold=dense_fit_bold(),
        min_size_px=7,
        max_size_px=max(8, int(render_params.label_font_size_px)),
        fill_ratio=0.95,
    )
    return draw_centered_text(
        draw,
        text=str(text),
        center=(float(center[0]), float(center[1])),
        font=font,
        fill=_text_fill_for_color(fill_color),
        stroke_fill=_text_stroke_for_color(fill_color),
        stroke_width=dense_stroke_width(),
    )

def _draw_region_reference_label(
    draw: ImageDraw.ImageDraw,
    *,
    region: Mapping[str, Any],
    center: Sequence[float],
    bbox: Sequence[float],
    fill_color: Sequence[int],
    render_params: _MapRenderParams,
) -> List[float] | None:
    text = str(region.get("region_label") or "").strip()
    if not text:
        return None
    width = max(1.0, float(bbox[2]) - float(bbox[0]))
    height = max(1.0, float(bbox[3]) - float(bbox[1]))
    font = fit_font_to_box(
        draw,
        text=str(text),
        max_width=max(14.0, min(float(width) - 5.0, 58.0)),
        max_height=max(10.0, min(float(height) - 5.0, 26.0)),
        bold=dense_fit_bold(),
        min_size_px=8,
        max_size_px=max(10, int(render_params.label_font_size_px) + 2),
        fill_ratio=0.95,
    )
    return draw_centered_text(
        draw,
        text=str(text),
        center=(float(center[0]), float(center[1])),
        font=font,
        fill=_text_fill_for_color(fill_color),
        stroke_fill=_text_stroke_for_color(fill_color),
        stroke_width=dense_stroke_width(),
    )

def _layout_bboxes(render_params: _MapRenderParams) -> Tuple[BBox, BBox, BBox, BBox]:
    """Compute stable panel, title, map, and legend boxes from resolved margins."""

    outer = float(render_params.outer_margin_px)
    offset_x = float(render_params.layout_offset_x_px)
    offset_y = float(render_params.layout_offset_y_px)
    panel_bbox: BBox = (
        outer + offset_x,
        outer + offset_y,
        float(render_params.canvas_width) - outer + offset_x,
        float(render_params.canvas_height) - outer + offset_y,
    )
    title_bbox: BBox = (
        panel_bbox[0] + float(render_params.panel_padding_px),
        panel_bbox[1] + 10.0,
        panel_bbox[2] - float(render_params.panel_padding_px),
        panel_bbox[1] + float(render_params.title_band_height_px),
    )
    content_top = float(title_bbox[3] + 12.0)
    content_bottom = float(panel_bbox[3] - render_params.panel_padding_px)
    if str(render_params.legend_position) == "none":
        legend_bbox = (0.0, 0.0, 0.0, 0.0)
        map_bbox = (
            float(panel_bbox[0] + render_params.panel_padding_px),
            content_top,
            float(panel_bbox[2] - render_params.panel_padding_px),
            content_bottom,
        )
    elif str(render_params.legend_position) == "bottom":
        legend_bbox = (
            float(panel_bbox[0] + render_params.panel_padding_px),
            float(content_bottom - render_params.legend_height_px),
            float(panel_bbox[2] - render_params.panel_padding_px),
            float(content_bottom),
        )
        map_bbox = (
            float(panel_bbox[0] + render_params.panel_padding_px),
            content_top,
            float(panel_bbox[2] - render_params.panel_padding_px),
            float(legend_bbox[1] - render_params.map_legend_gap_px),
        )
    elif str(render_params.legend_position) == "top":
        legend_bbox = (
            float(panel_bbox[0] + render_params.panel_padding_px),
            content_top,
            float(panel_bbox[2] - render_params.panel_padding_px),
            float(content_top + render_params.legend_height_px),
        )
        map_bbox = (
            float(panel_bbox[0] + render_params.panel_padding_px),
            float(legend_bbox[3] + render_params.map_legend_gap_px),
            float(panel_bbox[2] - render_params.panel_padding_px),
            content_bottom,
        )
    else:
        legend_left = float(panel_bbox[2] - render_params.panel_padding_px - render_params.legend_width_px)
        map_bbox = (
            float(panel_bbox[0] + render_params.panel_padding_px),
            content_top,
            float(legend_left - render_params.map_legend_gap_px),
            content_bottom,
        )
        legend_bbox = (
            legend_left,
            content_top,
            float(panel_bbox[2] - render_params.panel_padding_px),
            content_bottom,
        )
    return panel_bbox, title_bbox, map_bbox, legend_bbox

def _draw_legend(
    draw: ImageDraw.ImageDraw,
    *,
    legend_bins: Sequence[Mapping[str, Any]],
    legend_bbox: BBox,
    render_params: _MapRenderParams,
    categorical: bool,
) -> Dict[str, List[float]]:
    """Draw a value/category legend and return swatch bboxes keyed by legend bin id."""

    if str(render_params.legend_position) == "none":
        return {}
    draw_rounded_rect(
        draw,
        legend_bbox,
        radius=12,
        fill=render_params.legend_fill_rgb,
        outline=render_params.map_border_rgb,
        width=2,
    )
    palette = list(render_params.map_palette_rgb)
    legend_entry_bbox_map: Dict[str, List[float]] = {}
    title = "Category legend" if bool(categorical) else "Value legend"
    title_bbox = (
        legend_bbox[0] + 12.0,
        legend_bbox[1] + 10.0,
        legend_bbox[2] - 12.0,
        legend_bbox[1] + 42.0,
    )
    draw_centered_text(
        draw,
        text=title,
        center=(0.5 * (title_bbox[0] + title_bbox[2]), 0.5 * (title_bbox[1] + title_bbox[3])),
        font=load_font(int(render_params.legend_font_size_px) + 2, bold=False),
        fill=render_params.legend_text_rgb,
        stroke_fill=render_params.legend_fill_rgb,
        stroke_width=dense_stroke_width(),
    )

    if str(render_params.legend_position) in {"bottom", "top"}:
        count = max(1, len(legend_bins))
        start_x = legend_bbox[0] + 18.0
        usable_w = max(1.0, legend_bbox[2] - legend_bbox[0] - 36.0)
        cell_w = usable_w / float(count)
        swatch_h = 26.0
        swatch_y0 = legend_bbox[1] + 52.0
        label_y0 = swatch_y0 + swatch_h + 3.0
        for index, bin_spec in enumerate(legend_bins):
            left = float(start_x + (index * cell_w) + 4.0)
            right = float(start_x + ((index + 1) * cell_w) - 4.0)
            swatch_bbox = (left, swatch_y0, right, swatch_y0 + swatch_h)
            fill = tuple(int(channel) for channel in palette[int(index) % len(palette)])
            draw.rounded_rectangle(
                swatch_bbox,
                radius=5,
                fill=fill,
                outline=tuple(int(channel) for channel in render_params.map_border_rgb),
                width=1,
            )
            text_bbox = (left - 2.0, label_y0, right + 2.0, legend_bbox[3] - 10.0)
            draw_centered_text(
                draw,
                text=str(bin_spec["bin_label"]),
                center=(0.5 * (text_bbox[0] + text_bbox[2]), 0.5 * (text_bbox[1] + text_bbox[3])),
                font=fit_font_to_box(
                    draw,
                    text=str(bin_spec["bin_label"]),
                    max_width=max(1.0, float(text_bbox[2] - text_bbox[0])),
                    max_height=max(1.0, float(text_bbox[3] - text_bbox[1])),
                    bold=dense_fit_bold(),
                    min_size_px=9,
                    max_size_px=int(render_params.legend_font_size_px),
                    fill_ratio=0.92,
                ),
                fill=render_params.legend_text_rgb,
                stroke_fill=render_params.legend_fill_rgb,
                stroke_width=dense_stroke_width(),
            )
            legend_entry_bbox_map[str(bin_spec["bin_id"])] = _round_bbox(swatch_bbox)
    else:
        row_top = legend_bbox[1] + 56.0
        row_step = max(38.0, min(55.0, (legend_bbox[3] - row_top - 14.0) / max(1.0, float(len(legend_bins)))))
        for index, bin_spec in enumerate(legend_bins):
            top = float(row_top + (float(index) * row_step))
            swatch_bbox = (
                float(legend_bbox[0] + 18.0),
                top,
                float(legend_bbox[0] + 66.0),
                top + min(31.0, row_step - 8.0),
            )
            fill = tuple(int(channel) for channel in palette[int(index) % len(palette)])
            draw.rounded_rectangle(
                swatch_bbox,
                radius=6,
                fill=fill,
                outline=tuple(int(channel) for channel in render_params.map_border_rgb),
                width=1,
            )
            text_bbox = (
                float(swatch_bbox[2] + 10.0),
                float(top - 2.0),
                float(legend_bbox[2] - 10.0),
                float(top + min(35.0, row_step - 4.0)),
            )
            draw_centered_text(
                draw,
                text=str(bin_spec["bin_label"]),
                center=(0.5 * (text_bbox[0] + text_bbox[2]), 0.5 * (text_bbox[1] + text_bbox[3])),
                font=fit_font_to_box(
                    draw,
                    text=str(bin_spec["bin_label"]),
                    max_width=max(1.0, float(text_bbox[2] - text_bbox[0])),
                    max_height=max(1.0, float(text_bbox[3] - text_bbox[1])),
                    bold=dense_fit_bold(),
                    min_size_px=9,
                    max_size_px=int(render_params.legend_font_size_px),
                    fill_ratio=0.94,
                ),
                fill=render_params.legend_text_rgb,
                stroke_fill=render_params.legend_fill_rgb,
                stroke_width=dense_stroke_width(),
            )
            legend_entry_bbox_map[str(bin_spec["bin_id"])] = _round_bbox(swatch_bbox)
    return dict(legend_entry_bbox_map)

def _project_world_point(
    point: Sequence[float],
    *,
    map_bbox: BBox,
    lon_bounds: Sequence[float],
    lat_bounds: Sequence[float],
) -> Point:
    lon = float(point[0])
    lat = float(point[1])
    lon_min, lon_max = float(lon_bounds[0]), float(lon_bounds[1])
    lat_min, lat_max = float(lat_bounds[0]), float(lat_bounds[1])
    lon = max(lon_min, min(lon_max, lon))
    lat = max(lat_min, min(lat_max, lat))
    x = float(map_bbox[0]) + ((lon - lon_min) / max(1e-9, lon_max - lon_min)) * float(map_bbox[2] - map_bbox[0])
    y = float(map_bbox[1]) + ((lat_max - lat) / max(1e-9, lat_max - lat_min)) * float(map_bbox[3] - map_bbox[1])
    return (float(x), float(y))

def _project_world_rings(
    rings: Sequence[Sequence[Sequence[float]]],
    *,
    map_bbox: BBox,
    lon_bounds: Sequence[float],
    lat_bounds: Sequence[float],
) -> List[List[Point]]:
    projected: List[List[Point]] = []
    for ring in rings:
        points = [
            _project_world_point(point, map_bbox=map_bbox, lon_bounds=lon_bounds, lat_bounds=lat_bounds)
            for point in ring
            if len(point) >= 2
        ]
        if len(points) >= 3:
            projected.append(points)
    return projected


def _polygon_area_px(points: Sequence[Point]) -> float:
    """Return unsigned projected polygon area in square pixels."""

    if len(points) < 3:
        return 0.0
    total = 0.0
    closed_points = list(points)
    for point_a, point_b in zip(closed_points, closed_points[1:] + closed_points[:1]):
        total += (float(point_a[0]) * float(point_b[1])) - (float(point_b[0]) * float(point_a[1]))
    return abs(float(total)) / 2.0


def _fit_bbox_to_aspect(container_bbox: BBox, *, target_aspect: float) -> BBox:
    left, top, right, bottom = [float(value) for value in container_bbox]
    width = max(1.0, float(right - left))
    height = max(1.0, float(bottom - top))
    aspect = width / height
    if aspect > float(target_aspect):
        fitted_width = height * float(target_aspect)
        x0 = left + ((width - fitted_width) / 2.0)
        return (float(x0), top, float(x0 + fitted_width), bottom)
    fitted_height = width / float(target_aspect)
    y0 = top + ((height - fitted_height) / 2.0)
    return (left, float(y0), right, float(y0 + fitted_height))

def _draw_world_graticule(
    draw: ImageDraw.ImageDraw,
    *,
    map_bbox: BBox,
    lon_bounds: Sequence[float],
    lat_bounds: Sequence[float],
    color: Tuple[int, int, int],
    width_px: int = 1,
) -> None:
    for lon in range(-120, 181, 60):
        p0 = _project_world_point((float(lon), float(lat_bounds[0])), map_bbox=map_bbox, lon_bounds=lon_bounds, lat_bounds=lat_bounds)
        p1 = _project_world_point((float(lon), float(lat_bounds[1])), map_bbox=map_bbox, lon_bounds=lon_bounds, lat_bounds=lat_bounds)
        draw.line([p0, p1], fill=color, width=max(1, int(width_px)))
    for lat in range(-30, 61, 30):
        p0 = _project_world_point((float(lon_bounds[0]), float(lat)), map_bbox=map_bbox, lon_bounds=lon_bounds, lat_bounds=lat_bounds)
        p1 = _project_world_point((float(lon_bounds[1]), float(lat)), map_bbox=map_bbox, lon_bounds=lon_bounds, lat_bounds=lat_bounds)
        draw.line([p0, p1], fill=color, width=max(1, int(width_px)))


def _world_projection_context(asset: Mapping[str, Any], *, render_params: _MapRenderParams) -> tuple[BBox, list[float], list[float], float]:
    """Return the projected map bbox and lon/lat bounds used for geographic maps."""

    _panel_bbox, _title_bbox, map_bbox_raw, _legend_bbox = _layout_bboxes(render_params)
    lon_bounds = [float(value) for value in asset.get("lon_bounds", [-180.0, 180.0])]
    lat_bounds = [float(value) for value in asset.get("lat_bounds", [-58.0, 84.0])]
    world_aspect = abs(float(lon_bounds[1] - lon_bounds[0])) / max(1e-9, abs(float(lat_bounds[1] - lat_bounds[0])))
    map_bbox = _fit_bbox_to_aspect(
        (
            float(map_bbox_raw[0] + 18.0),
            float(map_bbox_raw[1] + 18.0),
            float(map_bbox_raw[2] - 18.0),
            float(map_bbox_raw[3] - 16.0),
        ),
        target_aspect=float(world_aspect),
    )
    return map_bbox, lon_bounds, lat_bounds, float(world_aspect)


def geographic_visible_component_metadata(
    asset_region: Mapping[str, Any],
    *,
    asset: Mapping[str, Any],
    render_params: _MapRenderParams,
) -> Dict[str, Any]:
    """Return projected largest-component metadata for one geographic region."""

    map_bbox, lon_bounds, lat_bounds, _world_aspect = _world_projection_context(asset, render_params=render_params)
    projected_rings = _project_world_rings(
        asset_region.get("rings", []),
        map_bbox=map_bbox,
        lon_bounds=lon_bounds,
        lat_bounds=lat_bounds,
    )
    component_rows: list[dict[str, Any]] = []
    for index, ring in enumerate(projected_rings):
        component_rows.append(
            {
                "component_index": int(index),
                "area_px": round(float(_polygon_area_px(ring)), 3),
                "bbox_px": _round_bbox(_polygon_bbox(ring)),
                "center_px": [round(float(value), 3) for value in _polygon_center(ring)],
            }
        )
    if not component_rows:
        return {
            "component_count": 0,
            "visible_component_index": -1,
            "visible_component_area_px": 0.0,
            "visible_component_bbox_px": [0.0, 0.0, 0.0, 0.0],
            "visible_component_center_px": [0.0, 0.0],
        }
    largest = max(component_rows, key=lambda row: (float(row["area_px"]), -int(row["component_index"])))
    return {
        "component_count": int(len(component_rows)),
        "visible_component_index": int(largest["component_index"]),
        "visible_component_area_px": float(largest["area_px"]),
        "visible_component_bbox_px": list(largest["bbox_px"]),
        "visible_component_center_px": list(largest["center_px"]),
    }

def _render_world_choropleth_map(
    background: Image.Image,
    *,
    scene_title: str,
    map_asset_id: str,
    regions: Sequence[Mapping[str, Any]],
    legend_bins: Sequence[Mapping[str, Any]],
    render_params: _MapRenderParams,
    instance_seed: int,
    categorical: bool,
    draw_color_legend: bool = True,
    neutral_regions: bool = False,
    show_region_value_labels: bool = False,
    show_region_reference_labels: bool = False,
) -> _RenderedChoroplethMap:
    """Render selected geographic asset regions with projected region bboxes for annotation."""

    image = background.copy()
    draw = ImageDraw.Draw(image)
    entities: List[Dict[str, Any]] = []
    region_bbox_map: Dict[str, List[float]] = {}
    region_center_map: Dict[str, List[float]] = {}

    panel_bbox, title_bbox, map_bbox_raw, legend_bbox = _layout_bboxes(render_params)
    asset = _load_geographic_map_asset(str(map_asset_id or "world_countries"))
    map_bbox, lon_bounds, lat_bounds, world_aspect = _world_projection_context(asset, render_params=render_params)
    style_id = str(render_params.world_map_style_id)
    style_probabilities = dict(render_params.world_map_style_probabilities)
    style = dict(render_params.world_map_style)
    ocean_rgb = _rgb(style.get("ocean_rgb"), (230, 239, 245))
    land_fill = _rgb(style.get("land_fill_rgb"), (215, 221, 219))
    land_outline = _rgb(style.get("land_outline_rgb"), (132, 144, 148))
    selected_outline = _rgb(style.get("selected_outline_rgb"), (38, 45, 52))
    reference_fill = _rgb(style.get("reference_fill_rgb"), (255, 240, 138))
    reference_outline = _rgb(style.get("reference_outline_rgb"), (20, 28, 36))
    neutral_selected_fill = _rgb(style.get("marker_region_fill_rgb"), (235, 238, 232))
    graticule_rgb = _rgb(style.get("graticule_rgb"), (200, 214, 222))
    graticule_width = max(1, int(style.get("graticule_width_px", 1)))
    selected_outline_width = max(1, int(style.get("selected_outline_width_px", 2)))
    show_graticule = bool(style.get("show_graticule", True))
    draw_rounded_rect(
        draw,
        panel_bbox,
        radius=16,
        fill=render_params.panel_fill_rgb,
        outline=render_params.panel_border_rgb,
        width=2,
    )
    draw.rounded_rectangle(
        map_bbox_raw,
        radius=12,
        fill=ocean_rgb,
        outline=tuple(int(channel) for channel in render_params.map_border_rgb),
        width=2,
    )
    title_text_bbox = draw_centered_text(
        draw,
        text=str(scene_title),
        center=(0.5 * (title_bbox[0] + title_bbox[2]), 0.5 * (title_bbox[1] + title_bbox[3])),
        font=load_font(int(render_params.title_font_size_px), bold=False),
        fill=render_params.title_color_rgb,
        stroke_fill=render_params.panel_fill_rgb,
        stroke_width=dense_stroke_width(),
    )

    selected_by_asset_id = {str(region["asset_region_id"]): dict(region) for region in regions}
    selected_region_id_by_asset_id = {
        str(region["asset_region_id"]): str(region["region_id"])
        for region in regions
    }
    border_reference_scene = (
        str(render_params.legend_position) == "none"
        and any(bool(region.get("is_reference_region")) for region in regions)
    )
    palette = list(render_params.map_palette_rgb)
    selected_fill_by_region_id: Dict[str, Tuple[int, int, int]] = {}

    if show_graticule:
        _draw_world_graticule(
            draw,
            map_bbox=map_bbox,
            lon_bounds=lon_bounds,
            lat_bounds=lat_bounds,
            color=graticule_rgb,
            width_px=graticule_width,
        )

    selected_projected: Dict[str, List[List[Point]]] = {}
    all_region_boxes: List[List[float]] = []
    for asset_region in asset.get("regions", []):
        if not isinstance(asset_region, Mapping):
            continue
        asset_region_id = str(asset_region.get("region_id"))
        projected_rings = _project_world_rings(
            asset_region.get("rings", []),
            map_bbox=map_bbox,
            lon_bounds=lon_bounds,
            lat_bounds=lat_bounds,
        )
        if not projected_rings:
            continue
        region_spec = selected_by_asset_id.get(asset_region_id)
        fill = land_fill
        outline = land_outline
        line_width = 1
        if region_spec is not None:
            if bool(region_spec.get("is_reference_region")) and not bool(show_region_reference_labels):
                fill = reference_fill
                outline = reference_outline
                line_width = max(4, int(selected_outline_width) + 2)
            elif bool(border_reference_scene):
                fill = land_fill
                outline = land_outline
                line_width = 1
            elif bool(neutral_regions):
                fill = neutral_selected_fill
                outline = selected_outline
                line_width = selected_outline_width
            else:
                fill = tuple(int(channel) for channel in palette[int(region_spec["bin_index"]) % len(palette)])
                outline = selected_outline
                line_width = selected_outline_width
            selected_fill_by_region_id[selected_region_id_by_asset_id[asset_region_id]] = tuple(int(channel) for channel in fill)
            component_index = int(region_spec.get("visible_component_index", -1))
            if not (0 <= int(component_index) < len(projected_rings)):
                component_index = max(
                    range(len(projected_rings)),
                    key=lambda index: _polygon_area_px(projected_rings[int(index)]),
                )
            selected_projected[selected_region_id_by_asset_id[asset_region_id]] = [list(projected_rings[int(component_index)])]
        for ring in projected_rings:
            draw.polygon(ring, fill=land_fill, outline=land_outline)
            draw.line(ring + [ring[0]], fill=land_outline, width=1, joint="curve")
            all_region_boxes.append(_polygon_bbox(ring))
        if region_spec is not None:
            selected_ring = selected_projected[selected_region_id_by_asset_id[asset_region_id]][0]
            draw.polygon(selected_ring, fill=fill, outline=outline)
            draw.line(selected_ring + [selected_ring[0]], fill=outline, width=line_width, joint="curve")

    for region in regions:
        region_id = str(region["region_id"])
        rings = selected_projected.get(region_id, [])
        bbox = _bbox_union(_polygon_bbox(ring) for ring in rings) if rings else [0.0, 0.0, 0.0, 0.0]
        center = _polygon_center(rings[0]) if rings else (0.0, 0.0)
        region_bbox_map[region_id] = list(bbox)
        region_center_map[region_id] = [round(float(center[0]), 3), round(float(center[1]), 3)]
        value_label_bbox = None
        if bool(show_region_value_labels):
            value_label_bbox = _draw_region_value_label(
                draw,
                region=region,
                center=center,
                bbox=bbox,
                fill_color=selected_fill_by_region_id.get(str(region_id), land_fill),
                render_params=render_params,
            )
        elif bool(show_region_reference_labels):
            value_label_bbox = _draw_region_reference_label(
                draw,
                region=region,
                center=center,
                bbox=bbox,
                fill_color=selected_fill_by_region_id.get(str(region_id), land_fill),
                render_params=render_params,
            )
        entities.append(
            {
                "entity_id": region_id,
                "entity_type": "world_choropleth_country",
                "bbox_xyxy": list(bbox),
                "attrs": {
                    "asset_region_id": str(region["asset_region_id"]),
                    "display_name": str(region["display_name"]),
                    "continent": str(region.get("continent") or ""),
                    "bin_index": int(region["bin_index"]),
                    "bin_label": str(region["bin_label"]),
                    "bin_lower": region.get("bin_lower"),
                    "bin_upper": region.get("bin_upper"),
                    "category": str(region.get("category") or ""),
                    "region_label": str(region.get("region_label") or ""),
                    "region_value": region.get("region_value"),
                    "is_reference_region": bool(region.get("is_reference_region")),
                    "center_px": list(region_center_map[region_id]),
                    "centroid_lonlat": list(region.get("centroid_lonlat", [])),
                    "visible_component_index": int(region.get("visible_component_index", -1)),
                    "visible_component_area_px": float(region.get("visible_component_area_px", 0.0)),
                    "visible_component_bbox_px": list(region.get("visible_component_bbox_px", [])),
                    "component_count": int(region.get("component_count", 0)),
                },
            }
        )
        if value_label_bbox is not None:
            entities.append(
                {
                    "entity_id": f"{region_id}_value_label",
                    "entity_type": "map_region_value_label" if bool(show_region_value_labels) else "map_region_reference_label",
                    "bbox_xyxy": list(value_label_bbox),
                    "attrs": {
                        "region_id": str(region_id),
                        "region_label": str(region.get("region_label") or ""),
                        "region_value": (int(region.get("region_value")) if region.get("region_value") is not None else None),
                    },
                }
            )

    legend_entry_bbox_map = (
        _draw_legend(
            draw,
            legend_bins=legend_bins,
            legend_bbox=legend_bbox,
            render_params=render_params,
            categorical=bool(categorical),
        )
        if bool(draw_color_legend)
        else {}
    )
    map_shape_bbox = _bbox_union(all_region_boxes) if all_region_boxes else _round_bbox(map_bbox)
    entities.insert(0, {"entity_id": "map_panel", "entity_type": "map_panel", "bbox_xyxy": _round_bbox(panel_bbox)})
    entities.insert(
        1,
        {
            "entity_id": "map_title",
            "entity_type": "map_title",
            "bbox_xyxy": list(title_text_bbox),
            "attrs": {"title": str(scene_title)},
        },
    )
    entities.insert(2, {"entity_id": "world_choropleth_map_shape", "entity_type": "world_choropleth_map_shape", "bbox_xyxy": list(map_shape_bbox)})
    return _RenderedChoroplethMap(
        image=image,
        entities=tuple(dict(item) for item in entities),
        panel_bbox_px=_round_bbox(panel_bbox),
        title_bbox_px=list(title_text_bbox),
        map_bbox_px=list(map_shape_bbox),
        legend_bbox_px=_round_bbox(legend_bbox),
        region_bbox_map=dict(region_bbox_map),
        region_center_map=dict(region_center_map),
        legend_entry_bbox_map=dict(legend_entry_bbox_map),
        render_meta={
            "world_map_style": {
                "style_id": str(style_id),
                "style_probabilities": dict(style_probabilities),
                "ocean_rgb": [int(channel) for channel in ocean_rgb],
                "land_fill_rgb": [int(channel) for channel in land_fill],
                "land_outline_rgb": [int(channel) for channel in land_outline],
                "selected_outline_rgb": [int(channel) for channel in selected_outline],
                "reference_fill_rgb": [int(channel) for channel in reference_fill],
                "reference_outline_rgb": [int(channel) for channel in reference_outline],
                "graticule_rgb": [int(channel) for channel in graticule_rgb],
                "show_graticule": bool(show_graticule),
            },
            "world_projection_bbox_px": _round_bbox(map_bbox),
            "world_projection_aspect": round(float(world_aspect), 6),
            "geographic_visible_component_policy": "largest_component_only",
        },
    )

def _render_choropleth_map(
    background: Image.Image,
    *,
    scene_title: str,
    rows: int,
    cols: int,
    regions: Sequence[Mapping[str, Any]],
    legend_bins: Sequence[Mapping[str, Any]],
    render_params: _MapRenderParams,
    instance_seed: int,
    categorical: bool,
    draw_color_legend: bool = True,
    neutral_regions: bool = False,
    show_region_value_labels: bool = False,
    show_region_reference_labels: bool = False,
) -> _RenderedChoroplethMap:
    """Render synthetic connected-cell regions with bbox/center maps for downstream tasks."""

    image = background.copy()
    draw = ImageDraw.Draw(image)
    entities: List[Dict[str, Any]] = []
    region_bbox_map: Dict[str, List[float]] = {}
    region_center_map: Dict[str, List[float]] = {}

    panel_bbox, title_bbox, map_bbox, legend_bbox = _layout_bboxes(render_params)
    draw_rounded_rect(
        draw,
        panel_bbox,
        radius=16,
        fill=render_params.panel_fill_rgb,
        outline=render_params.panel_border_rgb,
        width=2,
    )
    title_text_bbox = draw_centered_text(
        draw,
        text=str(scene_title),
        center=(0.5 * (title_bbox[0] + title_bbox[2]), 0.5 * (title_bbox[1] + title_bbox[3])),
        font=load_font(int(render_params.title_font_size_px), bold=False),
        fill=render_params.title_color_rgb,
        stroke_fill=render_params.panel_fill_rgb,
        stroke_width=dense_stroke_width(),
    )

    grid_points = _grid_points(
        rows=int(rows),
        cols=int(cols),
        map_bbox=(
            map_bbox[0] + 20.0,
            map_bbox[1] + 22.0,
            map_bbox[2] - 20.0,
            map_bbox[3] - 18.0,
        ),
        instance_seed=int(instance_seed),
    )

    palette = list(render_params.map_palette_rgb)
    reference_fill = _rgb(render_params.world_map_style.get("reference_fill_rgb"), (255, 240, 138))
    reference_outline = _rgb(render_params.world_map_style.get("reference_outline_rgb"), (20, 28, 36))
    neutral_fills = (
        (235, 238, 232),
        (229, 235, 236),
        (238, 235, 229),
        (233, 232, 240),
    )
    polygons: Dict[str, List[Point]] = {}
    for region in regions:
        region_id = str(region["region_id"])
        polygon = _region_polygon(row=int(region["row"]), col=int(region["col"]), grid_points=grid_points)
        display_polygon = _shrink_polygon(polygon, gap_px=float(render_params.region_gap_px))
        polygons[region_id] = list(display_polygon)

    map_shape_bbox = _bbox_union(_polygon_bbox(points) for points in polygons.values())

    for region in regions:
        region_id = str(region["region_id"])
        display_polygon = polygons[str(region_id)]
        bin_index = int(region["bin_index"])
        fill = tuple(int(channel) for channel in palette[int(bin_index) % len(palette)])
        outline = tuple(int(channel) for channel in render_params.map_border_rgb)
        line_width = max(1, int(render_params.region_border_width_px) - 1)
        if bool(region.get("is_reference_region")) and not bool(show_region_reference_labels):
            fill = reference_fill
            outline = reference_outline
            line_width = max(4, int(render_params.region_border_width_px) + 1)
        elif bool(neutral_regions):
            fill = neutral_fills[int(_region_sort_key(region_id)[1]) % len(neutral_fills)] if isinstance(_region_sort_key(region_id)[1], int) else neutral_fills[0]
            outline = tuple(int(channel) for channel in render_params.map_border_rgb)
            line_width = max(2, int(render_params.region_border_width_px) - 1)
        draw.polygon(
            display_polygon,
            fill=fill,
            outline=tuple(int(channel) for channel in render_params.region_border_rgb),
        )
        draw.line(
            display_polygon + [display_polygon[0]],
            fill=outline,
            width=int(line_width),
            joint="curve",
        )
        bbox = _polygon_bbox(display_polygon)
        center = _polygon_center(display_polygon)
        region_bbox_map[region_id] = list(bbox)
        region_center_map[region_id] = [round(float(center[0]), 3), round(float(center[1]), 3)]
        value_label_bbox = None
        if bool(show_region_value_labels):
            value_label_bbox = _draw_region_value_label(
                draw,
                region=region,
                center=center,
                bbox=bbox,
                fill_color=fill,
                render_params=render_params,
            )
        elif bool(show_region_reference_labels):
            value_label_bbox = _draw_region_reference_label(
                draw,
                region=region,
                center=center,
                bbox=bbox,
                fill_color=fill,
                render_params=render_params,
            )
        entities.append(
            {
                "entity_id": region_id,
                "entity_type": "choropleth_region",
                "bbox_xyxy": list(bbox),
                "attrs": {
                    "row": int(region["row"]),
                    "col": int(region["col"]),
                    "bin_index": int(bin_index),
                    "bin_label": str(region["bin_label"]),
                    "bin_lower": region.get("bin_lower"),
                    "bin_upper": region.get("bin_upper"),
                    "category": str(region.get("category") or ""),
                    "region_label": str(region.get("region_label") or ""),
                    "region_value": region.get("region_value"),
                    "is_reference_region": bool(region.get("is_reference_region")),
                    "center_px": list(region_center_map[region_id]),
                },
            }
        )
        if value_label_bbox is not None:
            entities.append(
                {
                    "entity_id": f"{region_id}_value_label",
                    "entity_type": "map_region_value_label" if bool(show_region_value_labels) else "map_region_reference_label",
                    "bbox_xyxy": list(value_label_bbox),
                    "attrs": {
                        "region_id": str(region_id),
                        "region_label": str(region.get("region_label") or ""),
                        "region_value": (int(region.get("region_value")) if region.get("region_value") is not None else None),
                    },
                }
            )

    legend_entry_bbox_map = (
        _draw_legend(
            draw,
            legend_bins=legend_bins,
            legend_bbox=legend_bbox,
            render_params=render_params,
            categorical=bool(categorical),
        )
        if bool(draw_color_legend)
        else {}
    )

    entities.insert(0, {"entity_id": "map_panel", "entity_type": "map_panel", "bbox_xyxy": _round_bbox(panel_bbox)})
    entities.insert(
        1,
        {
            "entity_id": "map_title",
            "entity_type": "map_title",
            "bbox_xyxy": list(title_text_bbox),
            "attrs": {"title": str(scene_title)},
        },
    )
    entities.insert(2, {"entity_id": "choropleth_map_shape", "entity_type": "choropleth_map_shape", "bbox_xyxy": list(map_shape_bbox)})
    return _RenderedChoroplethMap(
        image=image,
        entities=tuple(dict(item) for item in entities),
        panel_bbox_px=_round_bbox(panel_bbox),
        title_bbox_px=list(title_text_bbox),
        map_bbox_px=list(map_shape_bbox),
        legend_bbox_px=_round_bbox(legend_bbox),
        region_bbox_map=dict(region_bbox_map),
        region_center_map=dict(region_center_map),
        legend_entry_bbox_map=dict(legend_entry_bbox_map),
        render_meta={
            "map_style": {"style_id": "synthetic_irregular_default"},
        },
    )


@dataclass(frozen=True)
class RegionMapRenderResult:
    """Rendered region-map scene plus reusable render metadata."""

    image: Image.Image
    rendered_scene: _RenderedChoroplethMap
    render_params: _MapRenderParams
    background_meta: Dict[str, Any]
    post_noise_meta: Dict[str, Any]
    chart_font_family: str


def render_region_map(
    *,
    dataset: Mapping[str, Any],
    params: Mapping[str, Any],
    instance_seed: int,
    categorical: bool,
    show_region_value_labels: bool,
    draw_color_legend: bool = True,
    neutral_regions: bool = False,
) -> RegionMapRenderResult:
    """Render a complete region-map scene from semantic dataset fields."""

    render_style_params = {**dict(params), "_render_style_seed": int(instance_seed)}
    resolved_params = _resolve_render_params(
        render_style_params,
        legend_count=len(dataset["legend_bins"]),
        categorical=bool(categorical),
    )
    chart_font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.chart_font",
        params=params,
        explicit_key="chart_font_family",
        weights_key="chart_font_family_weights",
    )
    render_params, background, background_meta, information_style_meta = prepare_chart_information_scene(
        instance_seed=int(instance_seed),
        params=params,
        scene_id="region_map",
        render_params=resolved_params,
        protected_colors=resolved_params.map_palette_rgb,
    )
    with temporary_default_font_family(str(chart_font_family)):
        show_region_reference_labels = bool(dataset.get("show_region_reference_labels"))
        if str(dataset["scene_variant"]) == "geographic_region_map":
            rendered_scene = _render_world_choropleth_map(
                background,
                scene_title=str(dataset["scene_title"]),
                map_asset_id=str(dataset.get("map_asset_id") or _WORLD_MAP_ASSET_ID),
                regions=list(dataset["regions"]),
                legend_bins=list(dataset["legend_bins"]),
                render_params=render_params,
                instance_seed=int(instance_seed),
                categorical=bool(categorical),
                draw_color_legend=bool(draw_color_legend),
                neutral_regions=bool(neutral_regions),
                show_region_value_labels=bool(show_region_value_labels),
                show_region_reference_labels=bool(show_region_reference_labels),
            )
        else:
            rendered_scene = _render_choropleth_map(
                background,
                scene_title=str(dataset["scene_title"]),
                rows=int(dataset["rows"]),
                cols=int(dataset["cols"]),
                regions=list(dataset["regions"]),
                legend_bins=list(dataset["legend_bins"]),
                render_params=render_params,
                instance_seed=int(instance_seed),
                categorical=bool(categorical),
                draw_color_legend=bool(draw_color_legend),
                neutral_regions=bool(neutral_regions),
                show_region_value_labels=bool(show_region_value_labels),
                show_region_reference_labels=bool(show_region_reference_labels),
            )
    image, post_noise_meta = apply_post_image_noise(
        rendered_scene.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    return RegionMapRenderResult(
        image=image,
        rendered_scene=rendered_scene,
        render_params=render_params,
        background_meta={**dict(background_meta), "information_scene_style": dict(information_style_meta)},
        post_noise_meta=dict(post_noise_meta),
        chart_font_family=str(chart_font_family),
    )


def _clamp_marker_bbox(bbox: Sequence[float], *, width: int, height: int) -> List[float]:
    x0, y0, x1, y1 = [float(value) for value in bbox]
    return _round_bbox(
        [
            max(0.0, min(float(width) - 1.0, x0)),
            max(0.0, min(float(height) - 1.0, y0)),
            max(1.0, min(float(width), x1)),
            max(1.0, min(float(height), y1)),
        ]
    )


def _draw_marker_label(
    draw: ImageDraw.ImageDraw,
    *,
    label: str,
    center: Sequence[float],
    radius: float,
    render_params: _MapRenderParams,
    style: Mapping[str, Tuple[int, int, int]],
) -> List[float]:
    """Draw one marker label near its bubble group without occluding the marker center."""

    x, y = float(center[0]), float(center[1])
    label_w = 28.0 if len(str(label)) <= 1 else 38.0
    label_h = 24.0
    left = x + float(radius) + 5.0
    top = y - float(radius) - 4.0
    if left + label_w > float(render_params.canvas_width) - 8.0:
        left = x - float(radius) - label_w - 5.0
    if top < 8.0:
        top = y + float(radius) + 4.0
    bbox = (
        float(left),
        float(top),
        float(left + label_w),
        float(top + label_h),
    )
    draw.rounded_rectangle(
        bbox,
        radius=6,
        fill=style["label_fill"],
        outline=style["label_outline"],
        width=2,
    )
    draw_centered_text(
        draw,
        text=str(label),
        center=(0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3])),
        font=load_font(15, bold=False),
        fill=style["label_outline"],
        stroke_fill=style["label_fill"],
        stroke_width=dense_stroke_width(),
    )
    return _clamp_marker_bbox(bbox, width=int(render_params.canvas_width), height=int(render_params.canvas_height))


def _draw_marker_legend(
    draw: ImageDraw.ImageDraw,
    *,
    render_params: _MapRenderParams,
    value_min: int,
    value_max: int,
    style: Mapping[str, Tuple[int, int, int]],
) -> Dict[str, List[float]]:
    """Draw the marker-size legend that explains bubble radius encoding."""

    legend_bbox = tuple(float(value) for value in _layout_bboxes(render_params)[3])
    if str(render_params.legend_position) == "none":
        return {}
    draw_rounded_rect(
        draw,
        legend_bbox,
        radius=12,
        fill=render_params.legend_fill_rgb,
        outline=render_params.map_border_rgb,
        width=2,
    )
    title_bbox = (
        legend_bbox[0] + 12.0,
        legend_bbox[1] + 8.0,
        legend_bbox[2] - 12.0,
        legend_bbox[1] + 38.0,
    )
    draw_centered_text(
        draw,
        text="Marker value",
        center=(0.5 * (title_bbox[0] + title_bbox[2]), 0.5 * (title_bbox[1] + title_bbox[3])),
        font=load_font(int(render_params.legend_font_size_px) + 2, bold=False),
        fill=render_params.legend_text_rgb,
        stroke_fill=render_params.legend_fill_rgb,
        stroke_width=dense_stroke_width(),
    )
    entries: Dict[str, List[float]] = {}
    values = list(range(int(value_min), int(value_max) + 1))
    count = len(values)
    if str(render_params.legend_position) in {"bottom", "top"}:
        start_x = legend_bbox[0] + 30.0
        usable_w = max(1.0, legend_bbox[2] - legend_bbox[0] - 60.0)
        row_y = legend_bbox[1] + 72.0
        for index, value in enumerate(values):
            x = start_x + (usable_w * float(index) / max(1.0, float(count - 1)))
            radius = 5.5 + ((float(value) - float(value_min)) / max(1.0, float(value_max - value_min))) * 11.0
            bbox = (x - radius, row_y - radius, x + radius, row_y + radius)
            draw.ellipse(bbox, fill=style["fill"], outline=style["outline"], width=2)
            draw_centered_text(
                draw,
                text=str(value),
                center=(float(x), float(row_y + 24.0)),
                font=load_font(12, bold=False),
                fill=render_params.legend_text_rgb,
                stroke_fill=render_params.legend_fill_rgb,
                stroke_width=dense_stroke_width(),
            )
            entries[f"marker_value_{value}"] = _round_bbox(bbox)
    else:
        start_y = legend_bbox[1] + 60.0
        step = max(30.0, min(42.0, (legend_bbox[3] - start_y - 16.0) / max(1.0, float(count))))
        for index, value in enumerate(values):
            y = start_y + (float(index) * step)
            x = legend_bbox[0] + 45.0
            radius = 5.0 + ((float(value) - float(value_min)) / max(1.0, float(value_max - value_min))) * 10.0
            bbox = (x - radius, y - radius, x + radius, y + radius)
            draw.ellipse(bbox, fill=style["fill"], outline=style["outline"], width=2)
            draw_centered_text(
                draw,
                text=str(value),
                center=(legend_bbox[0] + 96.0, y),
                font=load_font(int(render_params.legend_font_size_px), bold=False),
                fill=render_params.legend_text_rgb,
                stroke_fill=render_params.legend_fill_rgb,
                stroke_width=dense_stroke_width(),
            )
            entries[f"marker_value_{value}"] = _round_bbox(bbox)
    return entries


def _render_marker_layer(
    rendered_scene: _RenderedChoroplethMap,
    *,
    dataset: Mapping[str, Any],
    render_params: _MapRenderParams,
    params: Mapping[str, Any],
    instance_seed: int,
) -> Tuple[_RenderedChoroplethMap, Dict[str, List[List[float]]], Dict[str, List[float]], Dict[str, Any]]:
    """Overlay marker bubbles, labels, and marker legend on a rendered base map."""

    image = rendered_scene.image.copy()
    draw = ImageDraw.Draw(image)
    style_id, style_probabilities, style = _resolve_marker_style(
        params,
        render_defaults=_RENDER_DEFAULTS,
        namespace=SCENE_NAMESPACE,
        instance_seed=int(instance_seed),
    )
    marker_render_variant = str(dataset.get("marker_render_variant") or "proportional_bubble")
    show_marker_labels = bool(dataset.get("show_marker_labels", False))
    value_min = int(dataset.get("marker_value_min", 1))
    value_max = int(dataset.get("marker_value_max", 5))
    marker_bboxes_by_region: Dict[str, List[List[float]]] = {}
    marker_group_bbox_map: Dict[str, List[float]] = {}
    marker_entities: List[Dict[str, Any]] = []
    region_specs = [dict(region) for region in dataset.get("regions", []) if isinstance(region, Mapping)]

    min_radius = float(params.get("marker_min_radius_px", _RENDER_DEFAULTS.get("marker_min_radius_px", 8)))
    max_radius = float(params.get("marker_max_radius_px", _RENDER_DEFAULTS.get("marker_max_radius_px", 22)))
    for region in region_specs:
        region_id = str(region["region_id"])
        if not bool(region.get("has_marker", True)):
            continue
        center = rendered_scene.region_center_map.get(str(region_id))
        if not center:
            continue
        value = int(region.get("marker_value", value_min))
        label = str(region.get("marker_label", ""))
        radius = float(min_radius) + (
            (float(value) - float(value_min)) / max(1.0, float(value_max - value_min))
        ) * max(1.0, float(max_radius - min_radius))
        bbox = (
            float(center[0] - radius),
            float(center[1] - radius),
            float(center[0] + radius),
            float(center[1] + radius),
        )
        draw.ellipse(bbox, fill=style["fill"], outline=style["outline"], width=3)
        rounded = _clamp_marker_bbox(bbox, width=int(render_params.canvas_width), height=int(render_params.canvas_height))
        marker_entities.append(
            {
                "entity_id": f"{region_id}_marker",
                "entity_type": "map_marker_bubble",
                "bbox_xyxy": list(rounded),
                "attrs": {
                    "region_id": str(region_id),
                    "marker_label": str(label),
                    "marker_value": int(value),
                    "marker_render_variant": str(marker_render_variant),
                },
            }
        )
        if bool(show_marker_labels):
            label_bbox = _draw_marker_label(
                draw,
                label=str(label),
                center=center,
                radius=float(radius),
                render_params=render_params,
                style=style,
            )
            marker_entities.append(
                {
                    "entity_id": f"{region_id}_marker_label",
                    "entity_type": "map_marker_label",
                    "bbox_xyxy": list(label_bbox),
                    "attrs": {
                        "region_id": str(region_id),
                        "marker_label": str(label),
                        "marker_value": int(value),
                    },
                }
            )
        marker_bboxes_by_region[str(region_id)] = [list(rounded)]
        marker_group_bbox_map[str(region_id)] = _bbox_union([rounded])

    marker_legend_bbox_map = _draw_marker_legend(
        draw,
        render_params=render_params,
        value_min=int(value_min),
        value_max=int(value_max),
        style=style,
    )
    marker_meta = {
        "marker_render_variant": str(marker_render_variant),
        "marker_render_variant_probabilities": dict(dataset.get("marker_render_variant_probabilities", {})),
        "marker_style_variant": str(style_id),
        "marker_style_variant_probabilities": dict(style_probabilities),
        "marker_value_min": int(value_min),
        "marker_value_max": int(value_max),
        "marker_fill_rgb": [int(channel) for channel in style["fill"]],
        "marker_outline_rgb": [int(channel) for channel in style["outline"]],
        "show_marker_labels": bool(show_marker_labels),
    }
    return (
        _RenderedChoroplethMap(
            image=image,
            entities=tuple([dict(entity) for entity in rendered_scene.entities] + [dict(entity) for entity in marker_entities]),
            panel_bbox_px=list(rendered_scene.panel_bbox_px),
            title_bbox_px=list(rendered_scene.title_bbox_px),
            map_bbox_px=list(rendered_scene.map_bbox_px),
            legend_bbox_px=list(rendered_scene.legend_bbox_px),
            region_bbox_map=dict(rendered_scene.region_bbox_map),
            region_center_map=dict(rendered_scene.region_center_map),
            legend_entry_bbox_map=dict(marker_legend_bbox_map),
            render_meta={**dict(rendered_scene.render_meta), "marker_layer": dict(marker_meta)},
        ),
        {str(key): [list(bbox) for bbox in value] for key, value in marker_bboxes_by_region.items()},
        {str(key): list(value) for key, value in marker_group_bbox_map.items()},
        dict(marker_meta),
    )


@dataclass(frozen=True)
class MarkerMapRenderResult:
    """Rendered region-map scene with an overlaid marker layer."""

    image: Image.Image
    rendered_scene: _RenderedChoroplethMap
    render_params: _MapRenderParams
    marker_bboxes_by_region: Dict[str, List[List[float]]]
    marker_group_bbox_map: Dict[str, List[float]]
    marker_render_meta: Dict[str, Any]
    background_meta: Dict[str, Any]
    post_noise_meta: Dict[str, Any]
    chart_font_family: str


def render_region_marker_layer(
    *,
    dataset: Mapping[str, Any],
    params: Mapping[str, Any],
    instance_seed: int,
) -> MarkerMapRenderResult:
    """Render a complete region-map scene with neutral regions and marker bubbles."""

    base_result = render_region_map(
        dataset=dataset,
        params=params,
        instance_seed=int(instance_seed),
        categorical=False,
        show_region_value_labels=False,
        draw_color_legend=False,
        neutral_regions=True,
    )
    render_style_params = {**dict(params), "_render_style_seed": int(instance_seed)}
    with temporary_default_font_family(str(base_result.chart_font_family)):
        rendered_scene, marker_bboxes_by_region, marker_group_bbox_map, marker_render_meta = _render_marker_layer(
            base_result.rendered_scene,
            dataset=dataset,
            render_params=base_result.render_params,
            params=render_style_params,
            instance_seed=int(instance_seed),
        )
    image, post_noise_meta = apply_post_image_noise(
        rendered_scene.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    return MarkerMapRenderResult(
        image=image,
        rendered_scene=rendered_scene,
        render_params=base_result.render_params,
        marker_bboxes_by_region={str(key): [list(bbox) for bbox in value] for key, value in marker_bboxes_by_region.items()},
        marker_group_bbox_map={str(key): list(value) for key, value in marker_group_bbox_map.items()},
        marker_render_meta=dict(marker_render_meta),
        background_meta=dict(base_result.background_meta),
        post_noise_meta=dict(post_noise_meta),
        chart_font_family=str(base_result.chart_font_family),
    )


def font_assets_payload(chart_font_family: str) -> Dict[str, str]:
    return {
        "font_asset_version": font_asset_version(),
        "chart_font_family": str(chart_font_family),
    }


__all__ = [
    'BBox',
    'MarkerMapRenderResult',
    'Point',
    'RegionMapRenderResult',
    'font_assets_payload',
    'geographic_visible_component_metadata',
    'render_region_marker_layer',
    'render_region_map',
    '_MapRenderParams',
    '_RenderedChoroplethMap',
    '_render_choropleth_map',
    '_render_world_choropleth_map',
    '_resolve_render_params',
]
