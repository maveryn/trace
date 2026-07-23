"""Rendering helpers for circle-centerline-overlap diagrams."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.shared.color_distance import color_distance
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.deterministic_sampling import resolve_selection_index
from trace_tasks.tasks.shared.text_legibility import (
    READ_REQUIRED_TEXT_MIN_CONTRAST_RATIO,
    READ_REQUIRED_TEXT_MIN_LAB_DISTANCE,
    contrast_ratio,
)
from trace_tasks.tasks.shared.text_rendering import load_font
from trace_tasks.tasks.geometry.shared.diagram_style import (
    GEOMETRY_STYLE_PROFILE_ANALYTICAL_DIAGRAM,
    prepare_geometry_diagram_style_and_background,
)
from trace_tasks.tasks.geometry.shared.measurement_rendering import (
    assert_bboxes_inside,
    bbox_to_list,
    draw_dimension_line,
    draw_readout_centered,
)
from trace_tasks.tasks.geometry.shared.scene_transform import LazySceneTransform
from trace_tasks.tasks.geometry.shared.vector2d import (
    add,
    mul,
    perp,
    point_to_list,
    sub,
    unit,
)

from .state import (
    BBox,
    BOUNDARY_POINT_LABELS,
    CENTER_LABELS,
    SCENE_ID,
    CenterlineOverlapDiagramSpec,
    CenterlineOverlapRenderContext,
    Point,
    RenderedCenterlineOverlapScene,
)

def create_centerline_overlap_render_context(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
) -> CenterlineOverlapRenderContext:
    """Create a styled PIL render context for one overlap diagram."""

    width = int(params.get("canvas_width", group_default(rendering_defaults, "canvas_width", 820)))
    height = int(params.get("canvas_height", group_default(rendering_defaults, "canvas_height", 600)))
    image, background_meta, diagram_style, diagram_style_meta = prepare_geometry_diagram_style_and_background(
        instance_seed=int(instance_seed),
        params=params,
        scene_id=SCENE_ID,
        canvas_width=int(width),
        canvas_height=int(height),
        allow_dark=False,
        require_grid=False,
        style_profile=GEOMETRY_STYLE_PROFILE_ANALYTICAL_DIAGRAM,
    )
    fill_palettes = (
        ((238, 246, 255), (255, 247, 231), (241, 248, 239)),
        ((241, 248, 239), (246, 242, 255), (255, 242, 235)),
        ((255, 242, 235), (232, 246, 250), (248, 247, 240)),
        ((248, 247, 240), (234, 242, 255), (246, 242, 255)),
    )
    fill_index = resolve_selection_index(
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_ID}.fill_palette",
    )
    font_size = int(params.get("label_font_size", group_default(rendering_defaults, "label_font_size", 22)))
    small_font_size = int(params.get("small_label_font_size", group_default(rendering_defaults, "small_label_font_size", 18)))
    line_width = int(params.get("line_width", group_default(rendering_defaults, "line_width", 3)))
    label_stroke_width = int(
        params.get(
            "label_stroke_width",
            group_default(rendering_defaults, "label_stroke_width", 0),
        )
    )
    return CenterlineOverlapRenderContext(
        image=image,
        draw=ImageDraw.Draw(image),
        width=int(width),
        height=int(height),
        line_color=tuple(int(value) for value in diagram_style.stroke_rgb),
        secondary_color=tuple(int(value) for value in diagram_style.secondary_stroke_rgb),
        label_color=tuple(int(value) for value in diagram_style.label_rgb),
        label_stroke_color=tuple(int(value) for value in diagram_style.label_stroke_rgb),
        label_backing_color=tuple(int(value) for value in diagram_style.panel_fill_rgb),
        fill_colors=tuple(fill_palettes[int(fill_index) % len(fill_palettes)]),
        accent_color=tuple(int(value) for value in diagram_style.accent_rgb),
        line_width=max(2, int(line_width)),
        label_stroke_width=max(0, int(label_stroke_width)),
        font=load_font(max(12, int(font_size)), bold=False),
        small_font=load_font(max(10, int(small_font_size)), bold=False),
        diagram_style_meta=dict(diagram_style_meta),
        background_meta=dict(background_meta),
        scene_transform=LazySceneTransform(
            spawn_rng(int(instance_seed), f"{SCENE_ID}.scene_transform"),
            params=params,
            render_defaults=rendering_defaults,
            canvas_width=int(width),
            canvas_height=int(height),
        ),
    )


def _circle_bbox(center: Point, radius: float) -> BBox:
    return (
        float(center[0]) - float(radius),
        float(center[1]) - float(radius),
        float(center[0]) + float(radius),
        float(center[1]) + float(radius),
    )


def _circle_measure_label(spec: CenterlineOverlapDiagramSpec, label: str) -> str:
    radius = {
        "A": int(spec.case.radius_a),
        "B": int(spec.case.radius_b),
        "C": int(spec.case.radius_c),
    }[str(label)]
    if str(spec.label_mode) == "diameter":
        return f"d{label}={2 * radius}"
    return f"r{label}={radius}"


def _active_center_labels(spec: CenterlineOverlapDiagramSpec) -> tuple[str, ...]:
    return CENTER_LABELS[: int(spec.case.circle_count)]


def _active_boundary_labels(spec: CenterlineOverlapDiagramSpec) -> tuple[str, ...]:
    return BOUNDARY_POINT_LABELS[: 2 if int(spec.case.circle_count) == 2 else 4]


def _rgb_tuple(value: Sequence[int] | Any) -> tuple[int, int, int]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 3:
        return tuple(max(0, min(255, int(channel))) for channel in value[:3])  # type: ignore[return-value]
    return (255, 255, 255)


def _readout_surface_metadata(ctx: CenterlineOverlapRenderContext) -> dict[str, Any]:
    """Describe the intended plain-background surface under unbacked labels."""

    fill_rgb = _rgb_tuple(ctx.label_color)
    surfaces = (_rgb_tuple(ctx.label_backing_color),)
    min_contrast = min(float(contrast_ratio(fill_rgb, surface)) for surface in surfaces)
    min_lab = min(float(color_distance(fill_rgb, surface, distance_space="lab")) for surface in surfaces)
    return {
        "surface_rgbs": [list(surface) for surface in surfaces],
        "surface_sample_method": "geometry_scene_plain_background_anchors",
        "min_contrast_ratio": round(float(min_contrast), 3),
        "min_lab_distance": round(float(min_lab), 3),
        "min_contrast_required": round(float(READ_REQUIRED_TEXT_MIN_CONTRAST_RATIO), 3),
        "min_lab_distance_required": round(float(READ_REQUIRED_TEXT_MIN_LAB_DISTANCE), 3),
        "passes": bool(
            min_contrast >= float(READ_REQUIRED_TEXT_MIN_CONTRAST_RATIO)
            and min_lab >= float(READ_REQUIRED_TEXT_MIN_LAB_DISTANCE)
        ),
    }


def render_centerline_overlap_scene(
    ctx: CenterlineOverlapRenderContext,
    spec: CenterlineOverlapDiagramSpec,
    *,
    instance_seed: int,
    render_namespace: str,
) -> RenderedCenterlineOverlapScene:
    """Draw one collinear overlap diagram and project annotation roles."""

    rng = spawn_rng(int(instance_seed), str(render_namespace))
    case = spec.case
    centers_local: dict[str, Point] = {
        "A": (0.0, 0.0),
        "B": (float(case.distance_ab), 0.0),
    }
    radii = {"A": int(case.radius_a), "B": int(case.radius_b)}
    points_local: dict[str, Point] = {
        "A": centers_local["A"],
        "B": centers_local["B"],
        "P": (centers_local["B"][0] - float(case.radius_b), 0.0),
        "Q": (centers_local["A"][0] + float(case.radius_a), 0.0),
    }
    if int(case.circle_count) == 3:
        centers_local["C"] = (float(case.distance_ac), 0.0)
        radii["C"] = int(case.radius_c)
        points_local["C"] = centers_local["C"]
        points_local["R"] = (centers_local["C"][0] - float(case.radius_c), 0.0)
        points_local["S"] = (centers_local["B"][0] + float(case.radius_b), 0.0)
    active_center_labels = _active_center_labels(spec)
    active_boundary_labels = _active_boundary_labels(spec)
    local_min_x = min(centers_local[key][0] - radii[key] for key in centers_local)
    local_max_x = max(centers_local[key][0] + radii[key] for key in centers_local)
    local_max_r = max(radii.values())
    span_x = max(1.0, float(local_max_x - local_min_x))
    span_y = max(1.0, float(local_max_r * 2.0))
    scale = min((float(ctx.width) - 190.0) / span_x, (float(ctx.height) - 180.0) / span_y)
    if spec.show_overlap_dimensions:
        scale *= float(rng.uniform(0.78, 0.86))
    else:
        scale *= float(rng.uniform(0.86, 0.95))
    angle = math.radians(float(rng.uniform(-5.0, 5.0)))
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    local_center = ((local_min_x + local_max_x) / 2.0, 0.0)
    target_center = (
        float(ctx.width) / 2.0 + float(rng.uniform(-22.0, 22.0)),
        float(ctx.height) / 2.0
        + (-34.0 if spec.show_overlap_dimensions else 0.0)
        + float(rng.uniform(-12.0, 12.0) if spec.show_overlap_dimensions else rng.uniform(-18.0, 18.0)),
    )

    def transform(point: Point) -> Point:
        x = (float(point[0]) - float(local_center[0])) * float(scale)
        y = (float(point[1]) - float(local_center[1])) * float(scale)
        return (
            x * cos_a - y * sin_a + target_center[0],
            x * sin_a + y * cos_a + target_center[1],
        )

    points = {key: transform(value) for key, value in points_local.items()}
    radius_px = {key: float(value) * float(scale) for key, value in radii.items()}
    ctx.scene_transform.resolve(tuple(points.values()))
    points = ctx.scene_transform.keyed_points(points)
    radius_px = {
        key: float(value) * float(ctx.scene_transform.transform.scale)
        for key, value in radius_px.items()
    }
    rightmost_center = "C" if int(case.circle_count) == 3 else "B"
    centerline = unit(sub(points[rightmost_center], points["A"]))
    normal = perp(centerline)

    circle_bboxes: dict[str, BBox] = {}
    for index, label in enumerate(active_center_labels):
        bbox = _circle_bbox(points[label], radius_px[label])
        circle_bboxes[label] = bbox
        ctx.draw.ellipse(bbox, outline=ctx.line_color, width=ctx.line_width)
    assert_bboxes_inside(
        circle_bboxes.values(),
        width=ctx.width,
        height=ctx.height,
        error_message="circle centerline overlap circle too close to canvas edge",
    )

    line_start = add(points["A"], mul(centerline, -radius_px["A"] - 16.0))
    line_end = add(points[rightmost_center], mul(centerline, radius_px[rightmost_center] + 16.0))
    ctx.draw.line([line_start, line_end], fill=ctx.secondary_color, width=max(2, ctx.line_width - 1))

    label_bboxes: dict[str, BBox] = {}
    readout_metadata = _readout_surface_metadata(ctx)
    dot_radius = max(4, int(ctx.line_width + 1))
    boundary_dot_radius = max(3, int(ctx.line_width))
    if spec.show_overlap_dimensions:
        boundary_label_side = -24.0
        boundary_label_spread = 14.0
    else:
        boundary_label_side = 25.0
        boundary_label_spread = 12.0
    point_offsets = {
        "A": add(mul(normal, -26.0), mul(centerline, -15.0)),
        "B": mul(normal, -30.0),
        "C": add(mul(normal, -26.0), mul(centerline, 15.0)),
        "P": add(mul(normal, boundary_label_side), mul(centerline, -boundary_label_spread)),
        "Q": add(mul(normal, boundary_label_side), mul(centerline, boundary_label_spread)),
        "R": add(mul(normal, boundary_label_side), mul(centerline, -boundary_label_spread)),
        "S": add(mul(normal, boundary_label_side), mul(centerline, boundary_label_spread)),
    }
    for label in active_center_labels:
        point = points[label]
        ctx.draw.ellipse(
            (point[0] - dot_radius, point[1] - dot_radius, point[0] + dot_radius, point[1] + dot_radius),
            outline=ctx.line_color,
            width=max(1, ctx.line_width - 1),
        )
        label_bboxes[f"{label}_label"] = draw_readout_centered(
            ctx,
            label,
            add(point, point_offsets[label]),
            small=True,
            extra_metadata=readout_metadata,
        )
    for label in active_boundary_labels:
        point = points[label]
        ctx.draw.ellipse(
            (
                point[0] - boundary_dot_radius,
                point[1] - boundary_dot_radius,
                point[0] + boundary_dot_radius,
                point[1] + boundary_dot_radius,
            ),
            outline=ctx.accent_color,
            width=max(1, ctx.line_width - 1),
        )
        label_bboxes[f"{label}_label"] = draw_readout_centered(
            ctx,
            label,
            add(point, point_offsets[label]),
            small=True,
            extra_metadata=readout_metadata,
        )

    for label in active_center_labels:
        text_point = add(points[label], mul(normal, -radius_px[label] - 22.0))
        label_bboxes[f"{label}_measure"] = draw_readout_centered(
            ctx,
            _circle_measure_label(spec, label),
            text_point,
            small=True,
            extra_metadata=readout_metadata,
        )

    if spec.show_overlap_dimensions:
        overlap_label_offset = max(96.0, max(radius_px.values()) + 30.0)
        label_bboxes["overlap_ab"] = draw_dimension_line(
            ctx,
            points["P"],
            points["Q"],
            f"PQ={int(case.overlap_ab)}",
            label_offset=mul(normal, overlap_label_offset),
            color=ctx.accent_color,
            extra_metadata=readout_metadata,
        )
        if int(case.circle_count) == 3:
            label_bboxes["overlap_bc"] = draw_dimension_line(
                ctx,
                points["R"],
                points["S"],
                f"RS={int(case.overlap_bc)}",
                label_offset=mul(normal, overlap_label_offset),
                color=ctx.accent_color,
                extra_metadata=readout_metadata,
            )
        label_bboxes["target_center_distance"] = draw_dimension_line(
            ctx,
            points["A"],
            points[rightmost_center],
            f"A{rightmost_center}=?",
            label_offset=mul(normal, -62.0),
            color=ctx.secondary_color,
            extra_metadata=readout_metadata,
        )
    else:
        known_start, known_end = spec.known_segment_points
        target_start, target_end = spec.target_segment_points
        label_bboxes["known_segment"] = draw_dimension_line(
            ctx,
            points[known_start],
            points[known_end],
            f"{spec.known_segment_name}={int(spec.known_segment_value)}",
            label_offset=mul(normal, 46.0),
            color=ctx.accent_color,
            extra_metadata=readout_metadata,
        )
        label_bboxes["target_segment"] = draw_dimension_line(
            ctx,
            points[target_start],
            points[target_end],
            f"{spec.target_name}=?",
            label_offset=mul(normal, -52.0),
            color=ctx.secondary_color,
            extra_metadata=readout_metadata,
        )

    assert_bboxes_inside(
        label_bboxes.values(),
        width=ctx.width,
        height=ctx.height,
        error_message="circle centerline overlap label too close to canvas edge",
    )
    role_points = {label: points[label] for label in (*active_center_labels, *active_boundary_labels)}
    scene_entities = tuple(
        {
            "entity_id": f"circle_{label.lower()}",
            "entity_type": "circle",
            "label": label,
            "center": point_to_list(points[label]),
            "radius_units": int(radii[label]),
            "diameter_units": int(2 * radii[label]),
            "radius_px": round(float(radius_px[label]), 3),
            "bbox": bbox_to_list(circle_bboxes[label]),
        }
        for label in active_center_labels
    )
    render_map = {
        "coord_space": "pixel",
        "centers": {label: point_to_list(points[label]) for label in active_center_labels},
        "boundary_points": {label: point_to_list(points[label]) for label in active_boundary_labels},
        "circle_bboxes": {label: bbox_to_list(circle_bboxes[label]) for label in active_center_labels},
        "label_bboxes": {key: bbox_to_list(value) for key, value in label_bboxes.items()},
        "scale_px_per_unit": round(float(scale), 3),
        "centerline_angle_degrees": round(math.degrees(angle), 3),
        "target_name": str(spec.target_name),
        "label_mode": str(spec.label_mode),
        "circle_count": int(case.circle_count),
    }
    return RenderedCenterlineOverlapScene(
        image=ctx.image,
        annotation_keyed_points={key: tuple(role_points[key]) for key in spec.annotation_roles},
        annotation_roles=tuple(spec.annotation_roles),
        label_bboxes=dict(label_bboxes),
        scene_entities=scene_entities,
        render_map=render_map,
    )


__all__ = [
    "create_centerline_overlap_render_context",
    "render_centerline_overlap_scene",
]
