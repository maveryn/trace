"""Shared graph-paper polygon scene helpers for geometry transformation families."""

from __future__ import annotations

from typing import Sequence, Tuple

from ...shared.geometry_primitives import Point, point_inside_square_canvas
from ...shared.text_rendering import draw_text_centered, load_font
from .graph_rendering import graph_units_to_pixel, scale_point
from .single_object_scene import GraphSceneContext


def pixel_point_from_graph_units(point: Point, *, context: GraphSceneContext) -> Point:
    """Project one lattice graph point into canonical pixel coordinates."""

    return graph_units_to_pixel(
        (int(round(float(point[0]))), int(round(float(point[1])))),
        graph_origin=context.graph_origin,
        spacing=int(context.graph_spacing),
    )


def pixel_polygon_from_graph_units(vertices: Sequence[Point], *, context: GraphSceneContext) -> Tuple[Point, ...]:
    """Project one lattice polygon into canonical pixel coordinates."""

    return tuple(pixel_point_from_graph_units(point, context=context) for point in vertices)


def graph_polygon_inside_canvas(
    vertices_graph: Sequence[Point],
    *,
    context: GraphSceneContext,
    padding_px: float,
) -> bool:
    """Return whether every graph vertex stays inside the visible canvas."""

    for point in vertices_graph:
        pixel_point = pixel_point_from_graph_units(point, context=context)
        if not point_inside_square_canvas(
            pixel_point,
            canvas_size=int(context.canvas_size),
            padding=float(padding_px),
        ):
            return False
    return True


def polygon_bbox(vertices_px: Sequence[Point]) -> list[float]:
    """Return one canonical pixel bbox for the provided polygon vertices."""

    min_x = min(float(point[0]) for point in vertices_px)
    max_x = max(float(point[0]) for point in vertices_px)
    min_y = min(float(point[1]) for point in vertices_px)
    max_y = max(float(point[1]) for point in vertices_px)
    return [round(float(min_x), 3), round(float(min_y), 3), round(float(max_x), 3), round(float(max_y), 3)]


def draw_reference_polygon(
    draw,
    *,
    vertices_px: Sequence[Point],
    scene_scale: int,
    line_width: int,
    label_font_size_px: int,
    label_stroke_width: int,
    label_gap_px: float,
    line_color: Sequence[int],
    label_color: Sequence[int],
    label_stroke_color: Sequence[int],
) -> None:
    """Draw the reference polygon plus its `Reference` label."""

    scaled_vertices = [scale_point(point, int(scene_scale)) for point in vertices_px]
    draw.line(
        [*scaled_vertices, scaled_vertices[0]],
        fill=tuple(int(value) for value in line_color),
        width=max(1, int(line_width)),
        joint="curve",
    )
    min_y = min(float(point[1]) for point in scaled_vertices)
    avg_x = sum(float(point[0]) for point in scaled_vertices) / float(len(scaled_vertices))
    label_center = (
        float(avg_x),
        max(float(label_font_size_px), float(min_y - (float(label_gap_px) * float(scene_scale)))),
    )
    font = load_font(int(label_font_size_px), bold=True)
    draw_text_centered(
        draw,
        text="Reference",
        center=label_center,
        font=font,
        fill=tuple(int(value) for value in label_color),
        stroke_fill=tuple(int(value) for value in label_stroke_color),
        stroke_width=int(label_stroke_width),
    )


__all__ = [
    "draw_reference_polygon",
    "graph_polygon_inside_canvas",
    "pixel_point_from_graph_units",
    "pixel_polygon_from_graph_units",
    "polygon_bbox",
]
