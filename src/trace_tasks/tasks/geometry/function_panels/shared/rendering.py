"""Rendering primitives for function-panel scenes."""

from __future__ import annotations

from typing import Any, Mapping

from PIL import ImageDraw

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.geometry.shared.coordinate_panel_grid import (
    BBox,
    CoordinatePanelConfig,
    coordinate_panel_layout,
    draw_coordinate_panel_grid,
    draw_endpoint,
    graph_point_to_panel_pixel,
    panel_bbox_for_index,
)
from trace_tasks.tasks.geometry.shared.diagram_style import (
    geometry_coordinate_panel_style_from_diagram_style,
    prepare_geometry_diagram_style_and_background,
)
from trace_tasks.tasks.geometry.shared.noise_defaults import load_geometry_noise_defaults
from trace_tasks.tasks.geometry.shared.option_count import panel_grid_shape_for_option_count

from .defaults import canvas_size
from .sampling import format_interval
from .state import (
    GRID_MAX,
    GRID_MIN,
    Color,
    IntersectionPanelSpec,
    IntersectionSelection,
    Point,
    PropertySelection,
    RelationSpec,
    RenderedIntersectionScene,
    RenderedPropertyScene,
    SCENE_ID,
)


_POST_NOISE_DEFAULTS = load_geometry_noise_defaults(scene_id=SCENE_ID)
_LINE_PALETTES: tuple[tuple[Color, ...], ...] = (
    ((35, 93, 165), (190, 74, 60), (43, 127, 89), (132, 82, 174), (184, 126, 35), (21, 126, 139)),
    ((25, 106, 149), (181, 83, 121), (84, 117, 40), (112, 89, 172), (205, 96, 45), (25, 133, 109)),
    ((53, 84, 151), (178, 91, 52), (31, 127, 127), (145, 72, 157), (65, 120, 70), (194, 69, 90)),
)
_OBJECT_COLOR_PALETTES: tuple[tuple[Color, Color], ...] = (
    ((27, 96, 168), (196, 82, 42)),
    ((32, 126, 88), (147, 74, 165)),
    ((8, 119, 153), (205, 73, 89)),
    ((83, 87, 176), (184, 126, 26)),
    ((29, 131, 130), (178, 62, 123)),
    ((92, 104, 38), (185, 78, 45)),
    ((24, 105, 164), (183, 91, 42)),
    ((22, 129, 114), (117, 80, 178)),
    ((68, 112, 132), (176, 66, 104)),
)
_INTERSECTION_COLORS: tuple[Color, ...] = ((36, 42, 52), (54, 58, 68), (45, 67, 63))


def _panel_config(option_count: int) -> CoordinatePanelConfig:
    columns, rows = panel_grid_shape_for_option_count(int(option_count))
    return CoordinatePanelConfig(grid_min=GRID_MIN, grid_max=GRID_MAX, columns=int(columns), rows=int(rows))


def _function_panel_layout(canvas_width: int, canvas_height: int, *, config: CoordinatePanelConfig) -> dict[str, int]:
    if int(config.columns) != 2 or int(config.rows) != 2:
        return coordinate_panel_layout(int(canvas_width), int(canvas_height), config=config)

    gap = max(18, int(round(float(min(int(canvas_width), int(canvas_height))) * 0.025)))
    margin_y = max(12, int(round(float(canvas_height) * 0.018)))
    max_panel_from_height = (int(canvas_height) - (2 * margin_y) - gap) // 2
    max_panel_from_width = (int(canvas_width) - 56 - gap) // 2
    panel_size = int(max(240, min(max_panel_from_height, max_panel_from_width)))
    content_width = (2 * panel_size) + gap
    content_height = (2 * panel_size) + gap
    return {
        "margin_x": int((int(canvas_width) - content_width) // 2),
        "margin_y": int((int(canvas_height) - content_height) // 2),
        "gap_x": int(gap),
        "gap_y": int(gap),
        "panel_width": int(panel_size),
        "panel_height": int(panel_size),
    }


def _plot_bbox_for_function_panel(panel_bbox: BBox) -> BBox:
    left, top, right, bottom = panel_bbox
    panel_w = int(right) - int(left)
    panel_h = int(bottom) - int(top)
    plot_size = int(max(160, min(panel_w - 44, panel_h - 44)))
    plot_left = int(left) + ((panel_w - plot_size) // 2)
    plot_top = int(top) + max(24, ((panel_h - plot_size) // 2) - 2)
    if plot_top + plot_size > int(bottom) - 18:
        plot_top = int(bottom) - 18 - plot_size
    return (int(plot_left), int(plot_top), int(plot_left + plot_size), int(plot_top + plot_size))


def _relation_pixels(relation: RelationSpec, *, plot_bbox, config: CoordinatePanelConfig) -> list[Point]:
    return [
        graph_point_to_panel_pixel(point, plot_bbox=plot_bbox, config=config)
        for point in relation.points
    ]


def _draw_relation(
    draw: ImageDraw.ImageDraw,
    relation: RelationSpec,
    *,
    plot_bbox,
    config: CoordinatePanelConfig,
    color: Color,
    line_width: int,
) -> None:
    if relation.draw_kind == "ellipse" and relation.center is not None and relation.radii is not None:
        center_px = graph_point_to_panel_pixel(relation.center, plot_bbox=plot_bbox, config=config)
        radius_x_px = abs(
            graph_point_to_panel_pixel((relation.center[0] + relation.radii[0], relation.center[1]), plot_bbox=plot_bbox, config=config)[0]
            - center_px[0]
        )
        radius_y_px = abs(
            graph_point_to_panel_pixel((relation.center[0], relation.center[1] + relation.radii[1]), plot_bbox=plot_bbox, config=config)[1]
            - center_px[1]
        )
        draw.ellipse(
            (
                center_px[0] - radius_x_px,
                center_px[1] - radius_y_px,
                center_px[0] + radius_x_px,
                center_px[1] + radius_y_px,
            ),
            outline=tuple(int(value) for value in color),
            width=max(2, int(line_width)),
        )
        return

    pixels = _relation_pixels(relation, plot_bbox=plot_bbox, config=config)
    if len(pixels) >= 2:
        draw.line(pixels, fill=tuple(int(value) for value in color), width=max(2, int(line_width)), joint="curve")
        for point in pixels:
            draw_endpoint(draw, point, color=tuple(int(value) for value in color), radius=max(3, int(line_width)))


def _draw_intersection_panel(
    draw: ImageDraw.ImageDraw,
    panel: IntersectionPanelSpec,
    *,
    plot_bbox,
    config: CoordinatePanelConfig,
    object_colors: tuple[Color, Color],
    intersection_color: Color,
    line_width: int,
) -> list[list[int]]:
    for index, (center, radius) in enumerate(panel.circles):
        center_px = graph_point_to_panel_pixel(center, plot_bbox=plot_bbox, config=config)
        edge_px = graph_point_to_panel_pixel((center[0] + radius, center[1]), plot_bbox=plot_bbox, config=config)
        radius_px = abs(edge_px[0] - center_px[0])
        draw.ellipse(
            (center_px[0] - radius_px, center_px[1] - radius_px, center_px[0] + radius_px, center_px[1] + radius_px),
            outline=tuple(int(value) for value in object_colors[index % 2]),
            width=max(2, int(line_width)),
        )
    for index, (start, end) in enumerate(panel.line_segments):
        start_px = graph_point_to_panel_pixel(start, plot_bbox=plot_bbox, config=config)
        end_px = graph_point_to_panel_pixel(end, plot_bbox=plot_bbox, config=config)
        draw.line((start_px, end_px), fill=tuple(int(value) for value in object_colors[index % 2]), width=max(2, int(line_width)))
    point_boxes: list[list[int]] = []
    radius_px = max(6, int(line_width) + 4)
    for point in panel.intersection_points:
        point_px = graph_point_to_panel_pixel(point, plot_bbox=plot_bbox, config=config)
        draw_endpoint(draw, point_px, color=tuple(int(value) for value in intersection_color), radius=radius_px)
        point_boxes.append(
            [
                int(round(point_px[0] - radius_px)),
                int(round(point_px[1] - radius_px)),
                int(round(point_px[0] + radius_px)),
                int(round(point_px[1] + radius_px)),
            ]
        )
    return point_boxes


def render_property_scene(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    selection: PropertySelection,
    relations_by_label: Mapping[str, RelationSpec],
) -> RenderedPropertyScene:
    """Render a selected-property panel grid after objective sampling is complete."""

    width, height = canvas_size(params, render_defaults=render_defaults, fallback_height=720)
    image, background_meta, diagram_style, diagram_style_meta = prepare_geometry_diagram_style_and_background(
        instance_seed=int(instance_seed),
        params=params,
        scene_id=SCENE_ID,
        canvas_width=int(width),
        canvas_height=int(height),
        style_profile="coordinate_grid",
        namespace_suffix="function_panel_background",
    )
    draw = ImageDraw.Draw(image)
    config = _panel_config(len(selection.label_pool))
    panel_style = geometry_coordinate_panel_style_from_diagram_style(diagram_style)
    layout = _function_panel_layout(int(width), int(height), config=config)
    rng = spawn_rng(int(instance_seed), "geometry.function_panels.property.render")
    palette_index = int(rng.randrange(len(_LINE_PALETTES)))
    line_colors = tuple(_LINE_PALETTES[palette_index])
    line_width = int(params.get("line_width", group_default(render_defaults, "line_width", 4)))
    panel_bboxes: dict[str, list[int]] = {}
    plot_bboxes: dict[str, list[int]] = {}
    for index, label in enumerate(selection.label_pool):
        panel_bbox = panel_bbox_for_index(layout, int(index), config=config)
        plot_bbox = _plot_bbox_for_function_panel(panel_bbox)
        draw_coordinate_panel_grid(draw, panel_bbox=panel_bbox, plot_bbox=plot_bbox, label=str(label), config=config, style=panel_style)
        _draw_relation(
            draw,
            relations_by_label[str(label)],
            plot_bbox=plot_bbox,
            config=config,
            color=line_colors[int(index) % len(line_colors)],
            line_width=int(line_width),
        )
        panel_bboxes[str(label)] = [int(value) for value in panel_bbox]
        plot_bboxes[str(label)] = [int(value) for value in plot_bbox]
    image, post_noise_meta = apply_post_image_noise(
        image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=_POST_NOISE_DEFAULTS,
    )
    selected_relation = relations_by_label[str(selection.selected_label)]
    return RenderedPropertyScene(
        image=image,
        background_meta=dict(background_meta),
        post_noise_meta=dict(post_noise_meta),
        diagram_style_meta=dict(diagram_style_meta),
        panel_style_meta=panel_style.to_trace_dict(),
        line_color_meta={"palette_index": int(palette_index), "palette": [list(color) for color in line_colors]},
        line_colors=tuple(line_colors),
        relations_by_label=dict(relations_by_label),
        panel_bboxes=panel_bboxes,
        plot_bboxes=plot_bboxes,
        panel_columns=int(config.columns),
        panel_rows=int(config.rows),
        panel_count_probabilities=dict(selection.panel_count_probabilities),
        target_range=format_interval(selected_relation.range),
        target_interval=format_interval(selection.target_interval) if selection.target_interval is not None else "",
    )


def render_intersection_scene(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    selection: IntersectionSelection,
    panels_by_label: Mapping[str, IntersectionPanelSpec],
) -> RenderedIntersectionScene:
    """Render a primitive-intersection panel grid after objective sampling is complete."""

    width = int(params.get("canvas_width", group_default(render_defaults, "canvas_width", 1024)))
    height = int(params.get("canvas_height", 1024))
    image, background_meta, diagram_style, diagram_style_meta = prepare_geometry_diagram_style_and_background(
        instance_seed=int(instance_seed),
        params=params,
        scene_id=SCENE_ID,
        canvas_width=max(720, int(width)),
        canvas_height=max(720, int(height)),
        style_profile="coordinate_grid",
        namespace_suffix="intersection_panel_background",
    )
    draw = ImageDraw.Draw(image)
    config = _panel_config(len(selection.label_pool))
    panel_style = geometry_coordinate_panel_style_from_diagram_style(diagram_style)
    layout = _function_panel_layout(image.width, image.height, config=config)
    rng = spawn_rng(int(instance_seed), "geometry.function_panels.intersection.render")
    palette_index = int(rng.randrange(len(_OBJECT_COLOR_PALETTES)))
    object_colors = tuple(_OBJECT_COLOR_PALETTES[palette_index])
    intersection_index = int(rng.randrange(len(_INTERSECTION_COLORS)))
    intersection_color = tuple(_INTERSECTION_COLORS[intersection_index])
    line_width = int(params.get("line_width", group_default(render_defaults, "line_width", 4)))
    panel_bboxes: dict[str, list[int]] = {}
    plot_bboxes: dict[str, list[int]] = {}
    point_bboxes: dict[str, list[list[int]]] = {}
    for index, label in enumerate(selection.label_pool):
        panel_bbox = panel_bbox_for_index(layout, int(index), config=config)
        plot_bbox = _plot_bbox_for_function_panel(panel_bbox)
        draw_coordinate_panel_grid(draw, panel_bbox=panel_bbox, plot_bbox=plot_bbox, label=str(label), config=config, style=panel_style)
        point_bboxes[str(label)] = _draw_intersection_panel(
            draw,
            panels_by_label[str(label)],
            plot_bbox=plot_bbox,
            config=config,
            object_colors=object_colors,
            intersection_color=intersection_color,
            line_width=int(line_width),
        )
        panel_bboxes[str(label)] = [int(value) for value in panel_bbox]
        plot_bboxes[str(label)] = [int(value) for value in plot_bbox]
    image, post_noise_meta = apply_post_image_noise(
        image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=_POST_NOISE_DEFAULTS,
    )
    return RenderedIntersectionScene(
        image=image,
        background_meta=dict(background_meta),
        post_noise_meta=dict(post_noise_meta),
        panel_style_meta=panel_style.to_trace_dict(),
        panels_by_label=dict(panels_by_label),
        panel_bboxes=panel_bboxes,
        plot_bboxes=plot_bboxes,
        intersection_point_bboxes=point_bboxes,
        panel_columns=int(config.columns),
        panel_rows=int(config.rows),
        panel_count_probabilities=dict(selection.panel_count_probabilities),
        object_color_meta={"palette_index": int(palette_index), "palette": [list(color) for color in object_colors]},
        object_colors=object_colors,
        intersection_color_meta={"palette_index": int(intersection_index), "color": list(intersection_color)},
        intersection_color=intersection_color,
    )
