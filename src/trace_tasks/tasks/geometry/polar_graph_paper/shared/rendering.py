"""Rendering for polar graph paper readout tasks."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from PIL import ImageDraw

from trace_tasks.tasks.geometry.shared.diagram_style import (
    GEOMETRY_STYLE_PROFILE_ANALYTICAL_DIAGRAM,
    prepare_geometry_diagram_style_and_background,
)
from trace_tasks.tasks.shared.text_rendering import draw_text_centered, load_font

from .state import (
    PolarCoordinateCountCase,
    PolarDifferenceCase,
    PolarPointSpec,
    PolarReadoutCase,
    RenderedPolarGraphPaperScene,
)


def _int_default(defaults: Mapping[str, Any], name: str, fallback: int) -> int:
    return int(defaults.get(name, fallback))


def _polar_point(center: tuple[float, float], scale: float, radius: float, theta_degrees: float) -> tuple[float, float]:
    theta = math.radians(theta_degrees)
    return (center[0] + scale * radius * math.cos(theta), center[1] - scale * radius * math.sin(theta))


def _draw_polar_grid(
    draw: ImageDraw.ImageDraw,
    *,
    center: tuple[float, float],
    plot_radius_px: float,
    radius_max: int,
    minor_spoke_step: int,
    major_spoke_step: int,
    radius_label_step: int,
    style: Any,
) -> dict[str, Any]:
    """Draw reusable polar graph paper while preserving radius/spoke scale."""

    scale = plot_radius_px / radius_max
    grid_major = tuple(style.grid_major_rgb)
    grid_minor = tuple(style.grid_minor_rgb)
    axis_color = tuple(style.axis_rgb)
    text_color = tuple(style.label_rgb)
    small_font = load_font(13, bold=False)
    angle_font = load_font(15, bold=False)

    for ring in range(1, radius_max + 1):
        ring_radius = ring * scale
        color = grid_major if ring % 2 == 0 or ring == radius_max else grid_minor
        width = 2 if ring % 2 == 0 or ring == radius_max else 1
        draw.ellipse(
            (
                center[0] - ring_radius,
                center[1] - ring_radius,
                center[0] + ring_radius,
                center[1] + ring_radius,
            ),
            outline=color,
            width=width,
        )

    for theta in range(0, 360, minor_spoke_step):
        endpoint = _polar_point(center, scale, radius_max, theta)
        is_axis = theta % 90 == 0
        is_major = theta % major_spoke_step == 0
        color = axis_color if is_axis else (grid_major if is_major else grid_minor)
        width = 3 if is_axis else (2 if is_major else 1)
        draw.line((center, endpoint), fill=color, width=width)

    for ring in range(radius_label_step, radius_max + 1, radius_label_step):
        ring_point = _polar_point(center, scale, ring, 0)
        draw_text_centered(
            draw,
            center=(ring_point[0], ring_point[1] + 16),
            text=str(ring),
            font=small_font,
            fill=text_color,
            stroke_width=0,
        )

    for theta in range(0, 360, major_spoke_step):
        label_point = _polar_point(center, scale, radius_max + 0.58, theta)
        draw_text_centered(
            draw,
            center=label_point,
            text=f"{theta}\N{DEGREE SIGN}",
            font=angle_font,
            fill=text_color,
            stroke_width=0,
        )

    origin_radius = 4
    draw.ellipse(
        (
            center[0] - origin_radius,
            center[1] - origin_radius,
            center[0] + origin_radius,
            center[1] + origin_radius,
        ),
        fill=axis_color,
    )
    draw_text_centered(
        draw,
        center=(center[0] - 18, center[1] + 20),
        text="O",
        font=small_font,
        fill=text_color,
        stroke_width=0,
    )
    return {"plot_scale_px_per_radius": scale}


def _draw_labeled_polar_point(
    draw: ImageDraw.ImageDraw,
    *,
    center: tuple[float, float],
    scale: float,
    point: PolarPointSpec,
    marker_radius: int,
    label_font_size: int,
    marker_fill: tuple[int, int, int],
    style: Any,
    label_offset_degrees: int = 180,
    draw_label: bool = True,
    draw_radius_line: bool = True,
) -> dict[str, Any]:
    """Draw one labeled polar point and return its projected witness geometry."""

    point_xy = _polar_point(center, scale, point.radius, point.theta_degrees)
    if draw_radius_line:
        draw.line((center, point_xy), fill=marker_fill, width=2)
    point_bbox = [
        round(point_xy[0] - marker_radius),
        round(point_xy[1] - marker_radius),
        round(point_xy[0] + marker_radius),
        round(point_xy[1] + marker_radius),
    ]
    draw.ellipse(
        point_bbox,
        fill=marker_fill,
        outline=tuple(style.stroke_rgb),
        width=3,
    )

    if draw_label:
        label_font = load_font(label_font_size, bold=True)
        label_offset = _polar_point(
            (0.0, 0.0),
            1.0,
            marker_radius + 22,
            point.theta_degrees + label_offset_degrees,
        )
        draw_text_centered(
            draw,
            center=(point_xy[0] + label_offset[0], point_xy[1] + label_offset[1]),
            text=point.label,
            font=label_font,
            fill=tuple(style.label_rgb),
            stroke_width=0,
        )
    return {
        "point": [round(point_xy[0], 3), round(point_xy[1], 3)],
        "bbox": point_bbox,
        "polar": {
            "radius": int(point.radius),
            "theta_degrees": int(point.theta_degrees),
        },
    }


def _base_polar_canvas(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    scene_id: str,
    rendering_defaults: Mapping[str, Any],
) -> tuple[Any, ImageDraw.ImageDraw, tuple[float, float], float, dict[str, Any], dict[str, Any], Any, dict[str, Any]]:
    """Prepare a polar grid canvas shared by all polar graph task renderers."""

    canvas_width = _int_default(rendering_defaults, "canvas_width", 820)
    canvas_height = _int_default(rendering_defaults, "canvas_height", 780)
    radius_max = _int_default(rendering_defaults, "polar_radius_max", 8)
    minor_spoke_step = _int_default(rendering_defaults, "minor_spoke_step_degrees", 15)
    major_spoke_step = _int_default(rendering_defaults, "major_spoke_step_degrees", 30)
    radius_label_step = _int_default(rendering_defaults, "radius_label_step", 1)

    background, background_meta, style, style_meta = prepare_geometry_diagram_style_and_background(
        instance_seed=int(instance_seed),
        params=params,
        scene_id=scene_id,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        style_profile=GEOMETRY_STYLE_PROFILE_ANALYTICAL_DIAGRAM,
        allow_dark=True,
        require_grid=False,
    )
    image = background
    draw = ImageDraw.Draw(image)

    plot_panel = (42, 32, canvas_width - 42, canvas_height - 42)
    draw.rounded_rectangle(
        plot_panel,
        radius=18,
        fill=tuple(style.panel_fill_rgb),
        outline=tuple(style.panel_border_rgb),
        width=2,
    )

    center = (canvas_width / 2, canvas_height / 2 + 8)
    plot_radius_px = min(
        (plot_panel[2] - plot_panel[0]) / 2 - 78,
        (plot_panel[3] - plot_panel[1]) / 2 - 58,
    )
    grid_spec = _draw_polar_grid(
        draw,
        center=center,
        plot_radius_px=plot_radius_px,
        radius_max=radius_max,
        minor_spoke_step=minor_spoke_step,
        major_spoke_step=major_spoke_step,
        radius_label_step=radius_label_step,
        style=style,
    )
    render_spec = {
        "canvas_width": canvas_width,
        "canvas_height": canvas_height,
        "polar_radius_max": radius_max,
        "minor_spoke_step_degrees": minor_spoke_step,
        "major_spoke_step_degrees": major_spoke_step,
        "style_profile": GEOMETRY_STYLE_PROFILE_ANALYTICAL_DIAGRAM,
    }
    style_metadata = {"background": background_meta, "diagram_style": style_meta}
    return image, draw, center, float(grid_spec["plot_scale_px_per_radius"]), render_spec, style_metadata, style, {
        "plot_radius_px": round(plot_radius_px, 3),
    }


def render_polar_graph_paper_scene(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    scene_id: str,
    case: PolarReadoutCase,
    rendering_defaults: Mapping[str, Any],
) -> RenderedPolarGraphPaperScene:
    """Render the polar grid and point witness from one sampled case."""

    canvas_width = _int_default(rendering_defaults, "canvas_width", 820)
    canvas_height = _int_default(rendering_defaults, "canvas_height", 780)
    radius_max = _int_default(rendering_defaults, "polar_radius_max", 8)
    minor_spoke_step = _int_default(rendering_defaults, "minor_spoke_step_degrees", 15)
    major_spoke_step = _int_default(rendering_defaults, "major_spoke_step_degrees", 30)
    radius_label_step = _int_default(rendering_defaults, "radius_label_step", 1)
    marker_radius = _int_default(rendering_defaults, "marker_radius_px", 9)
    label_font_size = _int_default(rendering_defaults, "label_font_size", 28)

    background, background_meta, style, style_meta = prepare_geometry_diagram_style_and_background(
        instance_seed=int(instance_seed),
        params=params,
        scene_id=scene_id,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        style_profile=GEOMETRY_STYLE_PROFILE_ANALYTICAL_DIAGRAM,
        allow_dark=True,
        require_grid=False,
    )
    image = background
    draw = ImageDraw.Draw(image)

    plot_panel = (42, 32, canvas_width - 42, canvas_height - 42)
    draw.rounded_rectangle(
        plot_panel,
        radius=18,
        fill=tuple(style.panel_fill_rgb),
        outline=tuple(style.panel_border_rgb),
        width=2,
    )

    center = (canvas_width / 2, canvas_height / 2 + 8)
    plot_radius_px = min(
        (plot_panel[2] - plot_panel[0]) / 2 - 78,
        (plot_panel[3] - plot_panel[1]) / 2 - 58,
    )
    grid_spec = _draw_polar_grid(
        draw,
        center=center,
        plot_radius_px=plot_radius_px,
        radius_max=radius_max,
        minor_spoke_step=minor_spoke_step,
        major_spoke_step=major_spoke_step,
        radius_label_step=radius_label_step,
        style=style,
    )
    scale = grid_spec["plot_scale_px_per_radius"]
    point_render = _draw_labeled_polar_point(
        draw,
        center=center,
        scale=scale,
        point=PolarPointSpec(label="P", radius=int(case.radius), theta_degrees=int(case.theta_degrees)),
        marker_radius=marker_radius,
        label_font_size=label_font_size,
        marker_fill=tuple(style.accent_rgb),
        style=style,
    )

    render_map = {
        "point_p": point_render["point"],
        "point_p_bbox": point_render["bbox"],
        "point_p_polar": point_render["polar"],
        "plot_center": [round(center[0], 3), round(center[1], 3)],
        "plot_radius_px": round(plot_radius_px, 3),
        "plot_scale_px_per_radius": round(scale, 3),
    }
    render_spec = {
        "canvas_width": canvas_width,
        "canvas_height": canvas_height,
        "polar_radius_max": radius_max,
        "minor_spoke_step_degrees": minor_spoke_step,
        "major_spoke_step_degrees": major_spoke_step,
        "style_profile": GEOMETRY_STYLE_PROFILE_ANALYTICAL_DIAGRAM,
    }

    return RenderedPolarGraphPaperScene(
        image=image,
        render_map=render_map,
        render_spec=render_spec,
        style_metadata={"background": background_meta, "diagram_style": style_meta},
    )


def render_polar_graph_paper_difference_scene(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    scene_id: str,
    case: PolarDifferenceCase,
    rendering_defaults: Mapping[str, Any],
) -> RenderedPolarGraphPaperScene:
    """Render the polar grid and two labeled point witnesses."""

    canvas_width = _int_default(rendering_defaults, "canvas_width", 820)
    canvas_height = _int_default(rendering_defaults, "canvas_height", 780)
    radius_max = _int_default(rendering_defaults, "polar_radius_max", 8)
    minor_spoke_step = _int_default(rendering_defaults, "minor_spoke_step_degrees", 15)
    major_spoke_step = _int_default(rendering_defaults, "major_spoke_step_degrees", 30)
    radius_label_step = _int_default(rendering_defaults, "radius_label_step", 1)
    marker_radius = _int_default(rendering_defaults, "marker_radius_px", 9)
    label_font_size = _int_default(rendering_defaults, "label_font_size", 28)

    background, background_meta, style, style_meta = prepare_geometry_diagram_style_and_background(
        instance_seed=int(instance_seed),
        params=params,
        scene_id=scene_id,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        style_profile=GEOMETRY_STYLE_PROFILE_ANALYTICAL_DIAGRAM,
        allow_dark=True,
        require_grid=False,
    )
    image = background
    draw = ImageDraw.Draw(image)

    plot_panel = (42, 32, canvas_width - 42, canvas_height - 42)
    draw.rounded_rectangle(
        plot_panel,
        radius=18,
        fill=tuple(style.panel_fill_rgb),
        outline=tuple(style.panel_border_rgb),
        width=2,
    )

    center = (canvas_width / 2, canvas_height / 2 + 8)
    plot_radius_px = min(
        (plot_panel[2] - plot_panel[0]) / 2 - 78,
        (plot_panel[3] - plot_panel[1]) / 2 - 58,
    )
    grid_spec = _draw_polar_grid(
        draw,
        center=center,
        plot_radius_px=plot_radius_px,
        radius_max=radius_max,
        minor_spoke_step=minor_spoke_step,
        major_spoke_step=major_spoke_step,
        radius_label_step=radius_label_step,
        style=style,
    )
    scale = grid_spec["plot_scale_px_per_radius"]
    point_p = _draw_labeled_polar_point(
        draw,
        center=center,
        scale=scale,
        point=case.point_p,
        marker_radius=marker_radius,
        label_font_size=label_font_size,
        marker_fill=tuple(style.accent_rgb),
        style=style,
    )
    point_q = _draw_labeled_polar_point(
        draw,
        center=center,
        scale=scale,
        point=case.point_q,
        marker_radius=marker_radius,
        label_font_size=label_font_size,
        marker_fill=tuple(style.secondary_accent_rgb),
        style=style,
    )

    render_map = {
        "point_p": point_p["point"],
        "point_q": point_q["point"],
        "point_p_bbox": point_p["bbox"],
        "point_q_bbox": point_q["bbox"],
        "point_p_polar": point_p["polar"],
        "point_q_polar": point_q["polar"],
        "plot_center": [round(center[0], 3), round(center[1], 3)],
        "plot_radius_px": round(plot_radius_px, 3),
        "plot_scale_px_per_radius": round(scale, 3),
    }
    render_spec = {
        "canvas_width": canvas_width,
        "canvas_height": canvas_height,
        "polar_radius_max": radius_max,
        "minor_spoke_step_degrees": minor_spoke_step,
        "major_spoke_step_degrees": major_spoke_step,
        "style_profile": GEOMETRY_STYLE_PROFILE_ANALYTICAL_DIAGRAM,
    }

    return RenderedPolarGraphPaperScene(
        image=image,
        render_map=render_map,
        render_spec=render_spec,
        style_metadata={"background": background_meta, "diagram_style": style_meta},
    )


def render_polar_graph_paper_coordinate_count_scene(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    scene_id: str,
    case: PolarCoordinateCountCase,
    rendering_defaults: Mapping[str, Any],
) -> RenderedPolarGraphPaperScene:
    """Render a polar graph with 8-12 marked points for coordinate-value counting."""

    marker_radius = _int_default(rendering_defaults, "marker_radius_px", 9)
    label_font_size = _int_default(rendering_defaults, "label_font_size", 28)
    image, draw, center, scale, render_spec, style_metadata, style, grid_meta = _base_polar_canvas(
        instance_seed=instance_seed,
        params=params,
        scene_id=scene_id,
        rendering_defaults=rendering_defaults,
    )

    points_by_label: dict[str, Any] = {}
    point_bboxes_by_label: dict[str, Any] = {}
    polar_by_label: dict[str, Any] = {}
    for point in case.points:
        point_render = _draw_labeled_polar_point(
            draw,
            center=center,
            scale=scale,
            point=point,
            marker_radius=marker_radius,
            label_font_size=max(20, label_font_size - 4),
            marker_fill=tuple(style.accent_rgb),
            style=style,
            label_offset_degrees=0,
            draw_label=False,
            draw_radius_line=False,
        )
        points_by_label[str(point.label)] = point_render["point"]
        point_bboxes_by_label[str(point.label)] = point_render["bbox"]
        polar_by_label[str(point.label)] = point_render["polar"]

    render_map = {
        "points_by_label": points_by_label,
        "point_bboxes_by_label": point_bboxes_by_label,
        "polar_by_label": polar_by_label,
        "matching_labels": list(case.matching_labels),
        "plot_center": [round(center[0], 3), round(center[1], 3)],
        "plot_radius_px": grid_meta["plot_radius_px"],
        "plot_scale_px_per_radius": round(scale, 3),
    }

    return RenderedPolarGraphPaperScene(
        image=image,
        render_map=render_map,
        render_spec=render_spec,
        style_metadata=style_metadata,
    )
