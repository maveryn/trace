"""Rendering helpers for the metro graph scene."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from ....shared.font_assets import font_asset_version, get_font_family_record
from ....shared.text_legibility import (
    draw_centered_readable_text,
    draw_readable_text,
    resolve_readable_text_style,
    text_legibility_summary_from_records,
)
from ....shared.text_rendering import fit_font_to_box, load_font
from ...shared.graph_scene import (
    GraphRenderParams,
    Point,
    apply_graph_content_layout_jitter,
    draw_graph_context_text_chips,
)
from .algorithms import station_sort_key
from .state import (
    GridPoint,
    MetroRouteNetworkSample,
    MetroRouteTemplate,
    RenderedMetroRoute,
    RenderedMetroRouteScene,
    RenderedMetroStation,
)


def _resolve_metro_panel_geometry(render_params: GraphRenderParams) -> Dict[str, Any]:
    width = int(render_params.canvas_width)
    height = int(render_params.canvas_height)
    margin = int(render_params.outer_margin_px)
    panel = (margin, margin, width - margin, height - margin)
    title_band_height = max(40, int(round(float(render_params.panel_title_font_size_px) * 1.8)))
    title_band = (panel[0], panel[1], panel[2], panel[1] + title_band_height)
    legend_band = (panel[0] + 18, title_band[3] + 4, panel[2] - 18, title_band[3] + 36)
    content = (
        panel[0] + max(36, int(render_params.panel_padding_px) + 18),
        legend_band[3] + 18,
        panel[2] - max(36, int(render_params.panel_padding_px) + 18),
        panel[3] - max(30, int(render_params.panel_padding_px)),
    )
    return {
        "canvas_size": [int(width), int(height)],
        "scene_panel_xyxy": [int(value) for value in panel],
        "title_band_xyxy": [int(value) for value in title_band],
        "legend_band_xyxy": [int(value) for value in legend_band],
        "scene_content_xyxy": [int(value) for value in content],
    }


def _grid_to_pixel(coord: GridPoint, content_bbox: Sequence[int]) -> Point:
    x0, y0, x1, y1 = [int(value) for value in content_bbox]
    x, y = (int(coord[0]), int(coord[1]))
    px = x0 + ((x1 - x0) * ((float(x) - 1.0) / 8.0))
    py = y0 + ((y1 - y0) * ((float(y) - 1.0) / 8.0))
    return (int(round(px)), int(round(py)))


def _text_bbox(draw: ImageDraw.ImageDraw, text: str, font: Any, *, stroke_width: int = 0) -> Tuple[int, int, int, int]:
    try:
        bbox = draw.textbbox((0, 0), str(text), font=font, stroke_width=max(0, int(stroke_width)))
        return (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
    except Exception:
        width, height = draw.textsize(str(text), font=font)
        return (0, 0, int(width), int(height))


def _draw_panel(
    image: Image.Image,
    *,
    panel_geometry: Mapping[str, Any],
    render_params: GraphRenderParams,
    scene_title: str,
    layout_seed: int,
) -> None:
    draw = ImageDraw.Draw(image)
    panel = tuple(int(value) for value in panel_geometry["scene_panel_xyxy"])
    draw.rounded_rectangle(
        panel,
        radius=max(0, int(render_params.panel_corner_radius_px)),
        fill=tuple(int(value) for value in render_params.panel_fill_rgb),
        outline=tuple(int(value) for value in render_params.panel_border_rgb),
        width=2,
    )
    title_band = tuple(int(value) for value in panel_geometry["title_band_xyxy"])
    title_style = resolve_readable_text_style(
        instance_seed=int(layout_seed),
        namespace="graph.metro.panel_title_text",
        role="graph_panel_title_text",
        surface_rgbs=(tuple(int(value) for value in render_params.panel_fill_rgb),),
        preferred_rgbs=(tuple(int(value) for value in render_params.title_color_rgb),),
        min_contrast_ratio=4.5,
        min_lab_distance=28.0,
    )
    draw_centered_readable_text(
        draw,
        text=str(scene_title),
        center=(0.5 * float(title_band[0] + title_band[2]), 0.5 * float(title_band[1] + title_band[3])),
        font=load_font(
            int(render_params.panel_title_font_size_px),
            bold=True,
            font_family=str(render_params.font_family or ""),
        ),
        style=title_style,
        stroke_width=2,
    )


def _draw_route_legend(
    draw: ImageDraw.ImageDraw,
    *,
    routes: Sequence[MetroRouteTemplate],
    legend_bbox: Sequence[int],
    render_params: GraphRenderParams,
    layout_seed: int,
) -> None:
    x0, y0, x1, y1 = [int(value) for value in legend_bbox]
    if not routes:
        return
    font = load_font(13, bold=True, font_family=str(render_params.font_family or ""))
    legend_style = resolve_readable_text_style(
        instance_seed=int(layout_seed),
        namespace="graph.metro.route_legend_text",
        role="metro_route_legend_text",
        surface_rgbs=(tuple(int(value) for value in render_params.panel_fill_rgb),),
        preferred_rgbs=(tuple(int(value) for value in render_params.title_color_rgb),),
        min_contrast_ratio=4.5,
        min_lab_distance=28.0,
    )
    slot_width = max(1, int((x1 - x0) / len(routes)))
    for index, route in enumerate(routes):
        left = x0 + (index * slot_width) + 8
        cy = int(round(0.5 * (y0 + y1)))
        color = tuple(int(value) for value in route.color_rgb)
        draw.line((left, cy, left + 30, cy), fill=color, width=8)
        draw.rounded_rectangle((left - 2, cy - 7, left + 32, cy + 7), radius=6, outline=color, width=1)
        draw_readable_text(
            draw,
            xy=(left + 40, cy - 8),
            text=str(route.route_name),
            font=font,
            style=legend_style,
        )


def _label_anchor_for_station(center: Point, content_bbox: Sequence[int]) -> Point:
    x0, y0, x1, y1 = [int(value) for value in content_bbox]
    cx, cy = int(center[0]), int(center[1])
    horizontal = -1 if cx > (x0 + x1) / 2 else 1
    vertical = -1 if cy > (y0 + y1) / 2 else 1
    return (int(cx + horizontal * 18), int(cy + vertical * 17))


def _metro_text_legibility_records(
    render_params: GraphRenderParams,
    *,
    layout_seed: int,
) -> Tuple[Dict[str, Any], ...]:
    """Return metro-specific required text styles for panel metadata."""

    existing_records: list[Dict[str, Any]] = []
    if isinstance(render_params.text_legibility, Mapping):
        raw_records = render_params.text_legibility.get("records")
        if isinstance(raw_records, list):
            existing_records = [
                dict(record)
                for record in raw_records
                if isinstance(record, Mapping) and str(record.get("role")) != "graph_node_label_text"
            ]
    title_style = resolve_readable_text_style(
        instance_seed=int(layout_seed),
        namespace="graph.metro.panel_title_text",
        role="graph_panel_title_text",
        surface_rgbs=(tuple(int(value) for value in render_params.panel_fill_rgb),),
        preferred_rgbs=(tuple(int(value) for value in render_params.title_color_rgb),),
        min_contrast_ratio=4.5,
        min_lab_distance=28.0,
    )
    legend_style = resolve_readable_text_style(
        instance_seed=int(layout_seed),
        namespace="graph.metro.route_legend_text",
        role="metro_route_legend_text",
        surface_rgbs=(tuple(int(value) for value in render_params.panel_fill_rgb),),
        preferred_rgbs=(tuple(int(value) for value in render_params.title_color_rgb),),
        min_contrast_ratio=4.5,
        min_lab_distance=28.0,
    )
    station_style = resolve_readable_text_style(
        instance_seed=int(layout_seed),
        namespace="graph.metro.station_label_text",
        role="metro_station_label_text",
        surface_rgbs=((255, 255, 255),),
        preferred_rgbs=(tuple(int(value) for value in render_params.label_text_rgb), (43, 50, 61)),
        min_contrast_ratio=7.0,
        min_lab_distance=38.0,
    )
    return tuple(
        [
            *existing_records,
            title_style.metadata(),
            legend_style.metadata(),
            station_style.metadata(),
        ]
    )


def render_metro_scene(
    *,
    metro_sample: MetroRouteNetworkSample,
    render_params: GraphRenderParams,
    base_image: Image.Image,
    scene_title: str = "Metro Route Graph",
    layout_seed: int = 0,
) -> RenderedMetroRouteScene:
    """Render one sampled metro-route graph scene."""

    image = base_image.convert("RGB")
    draw = ImageDraw.Draw(image)
    panel_geometry = _resolve_metro_panel_geometry(render_params)
    if isinstance(render_params.information_scene_style, Mapping):
        panel_geometry["information_scene_style"] = dict(render_params.information_scene_style)
    panel_geometry["text_legibility"] = text_legibility_summary_from_records(
        _metro_text_legibility_records(render_params, layout_seed=int(layout_seed))
    )
    panel_geometry["font_family"] = str(render_params.font_family or "")
    panel_geometry["font_asset"] = (
        dict(render_params.font_asset)
        if isinstance(render_params.font_asset, Mapping)
        else dict(get_font_family_record(str(render_params.font_family)).to_trace())
        if str(render_params.font_family or "").strip()
        else {}
    )
    panel_geometry["font_asset_version"] = str(render_params.font_asset_version or font_asset_version())
    panel_geometry["font_exclusion_reason"] = str(render_params.font_exclusion_reason)
    _draw_panel(
        image,
        panel_geometry=panel_geometry,
        render_params=render_params,
        scene_title=str(scene_title),
        layout_seed=int(layout_seed),
    )
    _draw_route_legend(
        draw,
        routes=metro_sample.route_templates,
        legend_bbox=panel_geometry["legend_band_xyxy"],
        render_params=render_params,
        layout_seed=int(layout_seed),
    )
    chip_context_elements = draw_graph_context_text_chips(
        image,
        panel_geometry=panel_geometry,
        render_params=render_params,
        layout_seed=int(layout_seed),
    )
    if chip_context_elements:
        panel_geometry["context_text_elements"] = [dict(element) for element in chip_context_elements]
    apply_graph_content_layout_jitter(
        panel_geometry,
        render_params=render_params,
        layout_seed=int(layout_seed),
    )

    content_bbox = tuple(int(value) for value in panel_geometry["scene_content_xyxy"])
    route_line_width = max(8, int(render_params.edge_width_px) * 2 + 2)
    station_radius = max(8, int(round(float(render_params.node_radius_px) * 0.48)))
    transfer_radius = max(station_radius + 7, int(round(float(render_params.node_radius_px) * 0.78)))
    coord_centers = {
        tuple(coord): _grid_to_pixel(tuple(coord), content_bbox)
        for coord in metro_sample.label_by_coord
    }

    rendered_routes: list[RenderedMetroRoute] = []
    for route in metro_sample.route_templates:
        polyline = tuple(_grid_to_pixel(tuple(coord), content_bbox) for coord in route.grid_points)
        for left, right in zip(polyline, polyline[1:]):
            draw.line(
                (left[0], left[1], right[0], right[1]),
                fill=(255, 255, 255),
                width=route_line_width + 6,
            )
        draw.line(
            tuple(point for xy in polyline for point in xy),
            fill=tuple(int(value) for value in route.color_rgb),
            width=route_line_width,
        )
        rendered_routes.append(
            RenderedMetroRoute(
                route_id=str(route.route_id),
                route_name=str(route.route_name),
                color_rgb=tuple(int(value) for value in route.color_rgb),
                station_labels=tuple(str(value) for value in metro_sample.route_station_labels[str(route.route_id)]),
                polyline_px=tuple(tuple(int(v) for v in point) for point in polyline),
            )
        )

    max_label = max(metro_sample.station_labels, key=len)
    label_font = fit_font_to_box(
        draw,
        text=str(max_label),
        max_width=34,
        max_height=22,
        max_size_px=max(13, int(render_params.label_font_size_px) - 3),
        min_size_px=10,
        bold=True,
        font_family=str(render_params.font_family or ""),
    )
    label_font_size = int(getattr(label_font, "size", 13))
    label_stroke_width = max(1, int(round(float(label_font_size) * 0.08)))
    label_style = resolve_readable_text_style(
        instance_seed=int(layout_seed),
        namespace="graph.metro.station_label_text",
        role="metro_station_label_text",
        surface_rgbs=((255, 255, 255),),
        preferred_rgbs=(tuple(int(value) for value in render_params.label_text_rgb), (43, 50, 61)),
        min_contrast_ratio=7.0,
        min_lab_distance=38.0,
    )
    rendered_stations: list[RenderedMetroStation] = []
    for coord in sorted(metro_sample.label_by_coord, key=station_sort_key):
        label = str(metro_sample.label_by_coord[coord])
        route_ids = tuple(str(value) for value in metro_sample.station_route_ids_by_label[label])
        is_transfer = len(route_ids) >= 2
        center = tuple(int(value) for value in coord_centers[coord])
        radius = int(transfer_radius if is_transfer else station_radius)
        if str(metro_sample.query_label) and str(label) == str(metro_sample.query_label):
            halo_radius = int(radius + 8)
            halo_bbox = (
                center[0] - halo_radius,
                center[1] - halo_radius,
                center[0] + halo_radius,
                center[1] + halo_radius,
            )
            draw.ellipse(halo_bbox, outline=(22, 25, 31), width=4)
        bbox = (center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius)
        draw.ellipse(
            bbox,
            fill=(255, 255, 255),
            outline=(48, 55, 67) if is_transfer else (88, 98, 114),
            width=4 if is_transfer else 2,
        )
        if is_transfer:
            inner = (center[0] - station_radius, center[1] - station_radius, center[0] + station_radius, center[1] + station_radius)
            draw.ellipse(inner, outline=(48, 55, 67), width=3)

        anchor = _label_anchor_for_station(center, content_bbox)
        text_bbox = _text_bbox(draw, label, label_font, stroke_width=label_stroke_width)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        label_box = (
            anchor[0] - int(text_width // 2) - 5,
            anchor[1] - int(text_height // 2) - 4,
            anchor[0] + int(math.ceil(text_width / 2.0)) + 5,
            anchor[1] + int(math.ceil(text_height / 2.0)) + 4,
        )
        draw.rounded_rectangle(label_box, radius=5, fill=(255, 255, 255), outline=(213, 219, 229), width=1)
        draw_centered_readable_text(
            draw,
            text=label,
            center=(float(anchor[0]), float(anchor[1])),
            font=label_font,
            style=label_style,
            stroke_width=label_stroke_width,
        )
        rendered_stations.append(
            RenderedMetroStation(
                label=str(label),
                grid_point=tuple(int(value) for value in coord),
                route_ids=tuple(route_ids),
                is_transfer=bool(is_transfer),
                center_xy=tuple(center),
                bbox_xyxy=tuple(int(value) for value in bbox),
            )
        )

    return RenderedMetroRouteScene(
        image=image,
        panel_geometry=dict(panel_geometry),
        stations=tuple(rendered_stations),
        routes=tuple(rendered_routes),
        resolved_label_font_size_px=int(label_font_size),
        route_line_width_px=int(route_line_width),
        station_radius_px=int(station_radius),
        transfer_station_radius_px=int(transfer_radius),
    )


__all__ = ["render_metro_scene"]
