"""Scene-local rendering helpers for object-scene 3D layouts."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

from PIL import ImageDraw

from ...shared.camera_projection import (
    CameraSpec,
    ProjectionFrame,
    canvas_floor_polygon_xy,
    grid_values_for_range,
    polygon_axis_line_segment,
    project_xy,
)
from ...shared.object_scene_rendering import _draw_line, _shade, _tint


def draw_object_scene_room(
    draw: ImageDraw.ImageDraw,
    *,
    camera: CameraSpec,
    frame: ProjectionFrame,
    render_params: Any,
    scene_variant: str,
) -> Tuple[List[float], List[Dict[str, Any]]]:
    """Draw the object-scene floor/platform shell and return its witness entity."""
    extent = float(render_params.room_extent)
    floor = [
        project_xy((-extent, -extent, 0.0), camera, frame),
        project_xy((extent, -extent, 0.0), camera, frame),
        project_xy((extent, extent, 0.0), camera, frame),
        project_xy((-extent, extent, 0.0), camera, frame),
    ]
    if str(scene_variant) == "tabletop_room":
        floor_fill = _tint(render_params.floor_rgb, 0.035)
        border_rgb = _shade(render_params.edge_rgb, 0.95)
        grid_rgb = _tint(render_params.grid_rgb, 0.025)
    elif str(scene_variant) == "studio_platform":
        floor_fill = _tint(render_params.floor_rgb, 0.06)
        border_rgb = render_params.edge_rgb
        grid_rgb = _tint(render_params.grid_rgb, 0.04)
    else:
        floor_fill = render_params.floor_rgb
        border_rgb = render_params.edge_rgb
        grid_rgb = render_params.grid_rgb
    full_bleed = bool(render_params.full_bleed_floor)
    if full_bleed:
        draw.rectangle(
            (0, 0, int(render_params.canvas_width), int(render_params.canvas_height)),
            fill=floor_fill,
        )
    else:
        draw.polygon(floor, fill=floor_fill)
        draw.line(floor + [floor[0]], fill=border_rgb, width=max(1, int(render_params.line_width_px) + 1))

    draw_grid = float(render_params.grid_step) > 0.0
    grid_mode = "bounded_stage" if bool(draw_grid) else "none"
    grid_extent = 0.0
    grid_world_bbox: List[float] | None = None
    if not draw_grid:
        grid_extent = 0.0
    elif full_bleed:
        floor_polygon_xy = canvas_floor_polygon_xy(camera=camera, frame=frame, render_params=render_params)
        if floor_polygon_xy:
            grid_mode = "screen_ray_floor_plane"
            min_x = min(float(point[0]) for point in floor_polygon_xy)
            max_x = max(float(point[0]) for point in floor_polygon_xy)
            min_y = min(float(point[1]) for point in floor_polygon_xy)
            max_y = max(float(point[1]) for point in floor_polygon_xy)
            grid_world_bbox = [round(min_x, 4), round(min_y, 4), round(max_x, 4), round(max_y, 4)]
            grid_x_values = grid_values_for_range(min_x, max_x, float(render_params.grid_step))
            grid_y_values = grid_values_for_range(min_y, max_y, float(render_params.grid_step))
            for value in grid_y_values:
                segment = polygon_axis_line_segment(floor_polygon_xy, axis="y", value=float(value))
                if segment is None:
                    continue
                _draw_line(
                    draw,
                    project_xy((segment[0][0], segment[0][1], 0.0), camera, frame),
                    project_xy((segment[1][0], segment[1][1], 0.0), camera, frame),
                    fill=grid_rgb,
                    width=render_params.line_width_px,
                )
            for value in grid_x_values:
                segment = polygon_axis_line_segment(floor_polygon_xy, axis="x", value=float(value))
                if segment is None:
                    continue
                _draw_line(
                    draw,
                    project_xy((segment[0][0], segment[0][1], 0.0), camera, frame),
                    project_xy((segment[1][0], segment[1][1], 0.0), camera, frame),
                    fill=grid_rgb,
                    width=render_params.line_width_px,
                )
            grid_extent = max(abs(min_x), abs(max_x), abs(min_y), abs(max_y))
        else:
            grid_mode = "bounded_stage_fallback"
            grid_extent = min(
                float(extent) * max(1.0, float(render_params.full_bleed_floor_extent_multiplier)),
                max(float(extent), float(camera.distance) * 0.74),
            )
            grid_count = int(math.ceil((2.0 * grid_extent) / float(render_params.grid_step)))
            grid_values = [round(-grid_extent + index * float(render_params.grid_step), 6) for index in range(grid_count + 1)]
            if grid_values[-1] < grid_extent:
                grid_values.append(grid_extent)
            grid_values = [max(-grid_extent, min(grid_extent, float(value))) for value in grid_values]
            for value in grid_values:
                _draw_line(
                    draw,
                    project_xy((-grid_extent, value, 0.0), camera, frame),
                    project_xy((grid_extent, value, 0.0), camera, frame),
                    fill=grid_rgb,
                    width=render_params.line_width_px,
                )
                _draw_line(
                    draw,
                    project_xy((value, -grid_extent, 0.0), camera, frame),
                    project_xy((value, grid_extent, 0.0), camera, frame),
                    fill=grid_rgb,
                    width=render_params.line_width_px,
                )
    else:
        grid_extent = float(extent)
        grid_count = int(math.ceil((2.0 * grid_extent) / float(render_params.grid_step)))
        grid_values = [round(-grid_extent + index * float(render_params.grid_step), 6) for index in range(grid_count + 1)]
        if grid_values[-1] < grid_extent:
            grid_values.append(grid_extent)
        grid_values = [max(-grid_extent, min(grid_extent, float(value))) for value in grid_values]
        for value in grid_values:
            _draw_line(
                draw,
                project_xy((-grid_extent, value, 0.0), camera, frame),
                project_xy((grid_extent, value, 0.0), camera, frame),
                fill=grid_rgb,
                width=render_params.line_width_px,
            )
            _draw_line(
                draw,
                project_xy((value, -grid_extent, 0.0), camera, frame),
                project_xy((value, grid_extent, 0.0), camera, frame),
                fill=grid_rgb,
                width=render_params.line_width_px,
            )
    platform_points: List[Tuple[float, float]] = []
    if str(scene_variant) == "studio_platform" and not full_bleed:
        platform_points = [
            project_xy((-2.4, -2.35, 0.03), camera, frame),
            project_xy((2.4, -2.35, 0.03), camera, frame),
            project_xy((2.4, 2.35, 0.03), camera, frame),
            project_xy((-2.4, 2.35, 0.03), camera, frame),
        ]
        draw.line(platform_points + [platform_points[0]], fill=(78, 90, 106), width=max(2, int(render_params.line_width_px) + 1))

    if full_bleed:
        room_bbox = [0.0, 0.0, float(render_params.canvas_width), float(render_params.canvas_height)]
    else:
        all_points = list(floor) + list(platform_points)
        room_bbox = [
            round(float(min(point[0] for point in all_points)), 3),
            round(float(min(point[1] for point in all_points)), 3),
            round(float(max(point[0] for point in all_points)), 3),
            round(float(max(point[1] for point in all_points)), 3),
        ]
    return room_bbox, [
        {
            "entity_id": "open_floor_stage",
            "entity_type": "three_d_open_floor_stage",
            "bbox_px": list(room_bbox),
            "attrs": {
                "scene_variant": str(scene_variant),
                "room_extent": float(extent),
                "full_bleed_floor": bool(full_bleed),
                "grid_extent": float(grid_extent),
                "grid_mode": str(grid_mode),
                "grid_world_bbox": list(grid_world_bbox) if grid_world_bbox is not None else None,
                "has_walls": False,
            },
        }
    ]


__all__ = ["draw_object_scene_room"]
