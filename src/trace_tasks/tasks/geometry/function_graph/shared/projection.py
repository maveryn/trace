"""Projection and drawing primitives for function-graph scenes."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

from PIL import ImageDraw

PointF = Tuple[float, float]


def graph_units_to_pixel_float(
    point: Sequence[float],
    *,
    graph_origin: Sequence[float],
    graph_spacing: int,
) -> PointF:
    """Project one graph-unit point into canonical pixel coordinates."""

    return (
        float(graph_origin[0]) + (float(point[0]) * float(graph_spacing)),
        float(graph_origin[1]) - (float(point[1]) * float(graph_spacing)),
    )


def scale_polyline(points: Iterable[PointF], *, scene_scale: int) -> List[PointF]:
    """Scale canonical pixel points into render-space coordinates."""

    scale = float(max(1, int(scene_scale)))
    return [(float(point[0]) * scale, float(point[1]) * scale) for point in points]


def draw_function_polyline(
    draw: ImageDraw.ImageDraw,
    *,
    polyline_graph: Sequence[Sequence[float]],
    graph_origin: Sequence[float],
    graph_spacing: int,
    scene_scale: int,
    line_width: int,
    line_color: Sequence[int],
) -> List[PointF]:
    """Draw one plotted function polyline and return render-space points."""

    canonical_points = [
        graph_units_to_pixel_float(point, graph_origin=graph_origin, graph_spacing=int(graph_spacing))
        for point in polyline_graph
    ]
    render_points = scale_polyline(canonical_points, scene_scale=int(scene_scale))
    if len(render_points) >= 2:
        draw.line(
            list(render_points),
            fill=tuple(int(value) for value in line_color),
            width=max(1, int(line_width)),
            joint="curve",
        )
    return render_points
