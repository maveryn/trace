"""Rendering helpers for circle-pair external tangent diagrams."""

from __future__ import annotations

from typing import Any, Mapping

from PIL import ImageDraw

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.shared.drawing import draw_dashed_line
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.text_rendering import load_font
from trace_tasks.tasks.geometry.shared.diagram_style import (
    GEOMETRY_STYLE_PROFILE_ANALYTICAL_DIAGRAM,
    prepare_geometry_diagram_style_and_background,
)
from trace_tasks.tasks.geometry.shared.measurement_rendering import (
    assert_bboxes_inside,
    bbox_from_points,
    bbox_to_list,
    pad_bbox,
    draw_dimension_line,
    draw_readout_centered,
    draw_right_angle_marker,
)
from trace_tasks.tasks.geometry.shared.scene_transform import LazySceneTransform
from trace_tasks.tasks.geometry.shared.vector2d import (
    add,
    distance,
    mid,
    mul,
    perp,
    point_to_list,
    sub,
    unit,
)

from .state import (
    BBox,
    PairTangentDiagramSpec,
    PairTangentRenderContext,
    Point,
    RenderedPairTangentScene,
    SCENE_ID,
)


def create_pair_tangent_render_context(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
) -> PairTangentRenderContext:
    """Create a styled PIL render context for one pair-tangent diagram."""

    width = int(
        params.get(
            "canvas_width", group_default(rendering_defaults, "canvas_width", 820)
        )
    )
    height = int(
        params.get(
            "canvas_height", group_default(rendering_defaults, "canvas_height", 600)
        )
    )
    image, background_meta, diagram_style, diagram_style_meta = (
        prepare_geometry_diagram_style_and_background(
            instance_seed=int(instance_seed),
            params=params,
            scene_id=SCENE_ID,
            canvas_width=int(width),
            canvas_height=int(height),
            allow_dark=False,
            require_grid=None,
            style_profile=GEOMETRY_STYLE_PROFILE_ANALYTICAL_DIAGRAM,
        )
    )
    font_size = int(
        params.get(
            "label_font_size", group_default(rendering_defaults, "label_font_size", 22)
        )
    )
    small_font_size = int(
        params.get(
            "small_label_font_size",
            group_default(rendering_defaults, "small_label_font_size", 18),
        )
    )
    line_width = int(
        params.get("line_width", group_default(rendering_defaults, "line_width", 3))
    )
    label_stroke_width = int(
        params.get(
            "label_stroke_width",
            group_default(rendering_defaults, "label_stroke_width", 0),
        )
    )
    return PairTangentRenderContext(
        image=image,
        draw=ImageDraw.Draw(image),
        width=int(width),
        height=int(height),
        line_color=tuple(int(value) for value in diagram_style.stroke_rgb),
        secondary_color=tuple(
            int(value) for value in diagram_style.secondary_stroke_rgb
        ),
        label_color=tuple(int(value) for value in diagram_style.label_rgb),
        label_stroke_color=tuple(
            int(value) for value in diagram_style.label_stroke_rgb
        ),
        label_backing_color=tuple(int(value) for value in diagram_style.panel_fill_rgb),
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


def _ellipse_bbox(center: Point, radius: float) -> BBox:
    return (
        float(center[0]) - float(radius),
        float(center[1]) - float(radius),
        float(center[0]) + float(radius),
        float(center[1]) + float(radius),
    )


def _bboxes_overlap(a: BBox, b: BBox, *, pad: float = 0.0) -> bool:
    return not (
        float(a[2]) + float(pad) < float(b[0])
        or float(b[2]) + float(pad) < float(a[0])
        or float(a[3]) + float(pad) < float(b[1])
        or float(b[3]) + float(pad) < float(a[1])
    )


def _label_bbox(ctx: PairTangentRenderContext, text: str, center: Point) -> BBox:
    bbox = ctx.draw.textbbox(
        (float(center[0]), float(center[1])),
        str(text),
        anchor="mm",
        font=ctx.small_font,
        stroke_width=max(0, int(ctx.label_stroke_width)),
    )
    return pad_bbox(bbox, 4.0, width=int(ctx.width), height=int(ctx.height))


def _bbox_outside_circle(bbox: BBox, *, center: Point, radius_px: float) -> bool:
    x = max(float(bbox[0]), min(float(center[0]), float(bbox[2])))
    y = max(float(bbox[1]), min(float(center[1]), float(bbox[3])))
    return distance((x, y), center) >= float(radius_px) + 3.0


def _bbox_edge_toward(bbox: BBox, target: Point) -> Point:
    cx = (float(bbox[0]) + float(bbox[2])) / 2.0
    cy = (float(bbox[1]) + float(bbox[3])) / 2.0
    dx = float(target[0]) - cx
    dy = float(target[1]) - cy
    half_w = max(1.0, (float(bbox[2]) - float(bbox[0])) / 2.0)
    half_h = max(1.0, (float(bbox[3]) - float(bbox[1])) / 2.0)
    if abs(dx) <= 1e-6 and abs(dy) <= 1e-6:
        return (cx, cy)
    scale = min(half_w / max(1e-6, abs(dx)), half_h / max(1e-6, abs(dy)))
    return (cx + dx * scale, cy + dy * scale)


def _draw_dimension_segment_without_label(
    ctx: PairTangentRenderContext,
    start: Point,
    end: Point,
    *,
    color: tuple[int, int, int],
    tick_px: float = 7.0,
) -> BBox:
    ctx.draw.line([start, end], fill=color, width=max(2, int(ctx.line_width) - 1))
    direction = unit(sub(end, start))
    normal = perp(direction)
    tick = float(tick_px)
    for point in (start, end):
        ctx.draw.line(
            [add(point, mul(normal, -tick)), add(point, mul(normal, tick))],
            fill=color,
            width=max(1, int(ctx.line_width) - 2),
        )
    return bbox_from_points((start, end), width=int(ctx.width), height=int(ctx.height), pad=tick + 3.0)


def _radius_callout_candidates(
    *,
    center: Point,
    tangent_point: Point,
    radius_px: float,
) -> tuple[Point, ...]:
    radial = unit(sub(tangent_point, center))
    side = unit(perp(radial))
    return (
        add(center, mul(radial, float(radius_px) + 42.0)),
        add(add(center, mul(radial, float(radius_px) + 44.0)), mul(side, 44.0)),
        add(add(center, mul(radial, float(radius_px) + 44.0)), mul(side, -44.0)),
        add(center, mul(radial, float(radius_px) + 64.0)),
        add(add(center, mul(side, float(radius_px) + 48.0)), mul(radial, 28.0)),
        add(add(center, mul(side, -(float(radius_px) + 48.0))), mul(radial, 28.0)),
        add(add(center, mul(radial, float(radius_px) + 72.0)), mul(side, 72.0)),
        add(add(center, mul(radial, float(radius_px) + 72.0)), mul(side, -72.0)),
    )


def _draw_radius_callout(
    ctx: PairTangentRenderContext,
    *,
    text: str,
    center: Point,
    tangent_point: Point,
    radius_px: float,
    occupied_bboxes: list[BBox],
) -> BBox:
    target = mid(center, tangent_point)
    candidates = _radius_callout_candidates(
        center=center,
        tangent_point=tangent_point,
        radius_px=float(radius_px),
    )
    chosen_center: Point | None = None
    chosen_bbox: BBox | None = None
    for candidate in candidates:
        bbox = _label_bbox(ctx, text, candidate)
        if not _bbox_outside_circle(bbox, center=center, radius_px=float(radius_px)):
            continue
        if any(_bboxes_overlap(bbox, occupied, pad=6.0) for occupied in occupied_bboxes):
            continue
        chosen_center = candidate
        chosen_bbox = bbox
        break
    if chosen_center is None or chosen_bbox is None:
        raise ValueError("circle-pair tangent radius callout has no collision-free placement")

    leader_start = _bbox_edge_toward(chosen_bbox, target)
    ctx.draw.line(
        [target, leader_start],
        fill=ctx.secondary_color,
        width=max(1, int(ctx.line_width) - 2),
    )
    return draw_readout_centered(ctx, text, chosen_center, small=True, backed=True)


def render_pair_tangent_scene(
    ctx: PairTangentRenderContext,
    spec: PairTangentDiagramSpec,
    *,
    instance_seed: int,
    render_namespace: str,
) -> RenderedPairTangentScene:
    """Draw two separated circles and their external common tangent."""

    rng = spawn_rng(int(instance_seed), str(render_namespace))
    radius_o1 = float(spec.radius_o1)
    radius_o2 = float(spec.radius_o2)
    center_distance = float(spec.center_distance)
    tangent_length = float(spec.tangent_length)
    delta = radius_o2 - radius_o1
    side_sign = -1.0 if str(spec.tangent_side) == "above" else 1.0

    # For an external common tangent, the center vector decomposes into
    # tangent length plus the signed radius-difference normal component.
    # This keeps AB perpendicular to both radii and outside both interiors.
    u_unit = (tangent_length / center_distance, side_sign * delta / center_distance)
    n_unit = (-delta / center_distance, side_sign * tangent_length / center_distance)

    o1_local = (0.0, 0.0)
    o2_local = (center_distance, 0.0)
    t1_local = add(o1_local, mul(n_unit, radius_o1))
    t2_local = add(o2_local, mul(n_unit, radius_o2))

    local_points = (
        add(o1_local, (-radius_o1, -radius_o1)),
        add(o1_local, (radius_o1, radius_o1)),
        add(o2_local, (-radius_o2, -radius_o2)),
        add(o2_local, (radius_o2, radius_o2)),
        t1_local,
        t2_local,
    )
    min_x = min(point[0] for point in local_points)
    max_x = max(point[0] for point in local_points)
    min_y = min(point[1] for point in local_points)
    max_y = max(point[1] for point in local_points)
    span_x = max(1e-6, float(max_x - min_x))
    span_y = max(1e-6, float(max_y - min_y))
    scale = min((ctx.width - 190.0) / span_x, (ctx.height - 160.0) / span_y)
    scale *= float(rng.uniform(0.86, 0.95))
    local_center = ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0)
    target_center = (
        (ctx.width / 2.0) + float(rng.uniform(-28.0, 28.0)),
        (ctx.height / 2.0) + float(rng.uniform(-22.0, 22.0)),
    )

    def transform(point: Point) -> Point:
        return (
            (float(point[0]) - float(local_center[0])) * float(scale)
            + float(target_center[0]),
            (float(point[1]) - float(local_center[1])) * float(scale)
            + float(target_center[1]),
        )

    o1 = transform(o1_local)
    o2 = transform(o2_local)
    t1 = transform(t1_local)
    t2 = transform(t2_local)
    aux_e = transform(add(o1_local, sub(t2_local, t1_local)))
    radius_o1_px = radius_o1 * scale
    radius_o2_px = radius_o2 * scale
    ctx.scene_transform.resolve((o1, o2, t1, t2, aux_e))
    o1, o2, t1, t2, aux_e = ctx.scene_transform.points((o1, o2, t1, t2, aux_e))
    radius_o1_px *= float(ctx.scene_transform.transform.scale)
    radius_o2_px *= float(ctx.scene_transform.transform.scale)
    u_px = sub(t2, t1)
    n_px = sub(t1, o1)

    o1_bbox = _ellipse_bbox(o1, radius_o1_px)
    o2_bbox = _ellipse_bbox(o2, radius_o2_px)
    assert_bboxes_inside(
        (o1_bbox, o2_bbox),
        width=ctx.width,
        height=ctx.height,
        error_message="circle-pair tangent label too close to canvas edge",
    )

    ctx.draw.ellipse(o1_bbox, outline=ctx.line_color, width=ctx.line_width)
    ctx.draw.ellipse(o2_bbox, outline=ctx.line_color, width=ctx.line_width)

    center_label_offset = (0.0, 26.0 if str(spec.tangent_side) == "above" else -26.0)
    tangent_label_offset = (0.0, -28.0 if str(spec.tangent_side) == "above" else 28.0)

    label_bboxes: dict[str, BBox] = {}
    label_bboxes["center_distance"] = draw_dimension_line(
        ctx,
        o1,
        o2,
        str(spec.center_segment_label),
        label_offset=center_label_offset,
        color=ctx.secondary_color,
        tick_px=7.0,
        backed=True,
    )
    radius_o1_segment_bbox = _draw_dimension_segment_without_label(
        ctx,
        o1,
        t1,
        color=ctx.secondary_color,
        tick_px=7.0,
    )
    radius_o2_segment_bbox = _draw_dimension_segment_without_label(
        ctx,
        o2,
        t2,
        color=ctx.secondary_color,
        tick_px=7.0,
    )

    aux_color = ctx.secondary_color
    draw_dashed_line(
        ctx.draw,
        start=o1,
        end=aux_e,
        fill=aux_color,
        width=max(1, int(ctx.line_width) - 2),
        dash_px=10.0,
        gap_px=7.0,
    )
    label_bboxes["auxiliary_radius_difference"] = draw_dimension_line(
        ctx,
        aux_e,
        o2,
        f"ED={abs(int(spec.radius_o2) - int(spec.radius_o1))}",
        label_offset=mul(unit(u_px), 50.0 if str(spec.larger_circle_side) == "right" else -50.0),
        color=aux_color,
        tick_px=7.0,
        backed=True,
    )
    label_bboxes["right_angle_aux_e"] = draw_right_angle_marker(
        ctx,
        aux_e,
        arm_a=sub(o1, aux_e),
        arm_b=sub(o2, aux_e),
        side_px=16.0,
        color=aux_color,
        width=max(1, int(ctx.line_width) - 2),
    )

    extension = max(18.0, ctx.line_width * 7.0)
    tangent_start = add(t1, mul(unit(u_px), -extension))
    tangent_end = add(t2, mul(unit(u_px), extension))
    ctx.draw.line(
        [tangent_start, tangent_end], fill=ctx.accent_color, width=ctx.line_width + 1
    )
    label_bboxes["tangent_length"] = draw_dimension_line(
        ctx,
        t1,
        t2,
        str(spec.tangent_segment_label),
        label_offset=tangent_label_offset,
        color=ctx.accent_color,
        tick_px=7.0,
        backed=True,
    )

    label_bboxes["right_angle_t1"] = draw_right_angle_marker(
        ctx, t1, arm_a=mul(n_px, -1.0), arm_b=u_px
    )
    label_bboxes["right_angle_t2"] = draw_right_angle_marker(
        ctx, t2, arm_a=mul(n_px, -1.0), arm_b=u_px
    )

    dot_radius = max(3, int(ctx.line_width + 1))
    point_label_offsets = {
        "C": (-18.0, 18.0),
        "D": (18.0, 18.0),
        "A": (-22.0, -20.0 if str(spec.tangent_side) == "above" else 20.0),
        "B": (22.0, -20.0 if str(spec.tangent_side) == "above" else 20.0),
        "E": (20.0, 20.0 if str(spec.tangent_side) == "above" else -20.0),
    }
    for label, point in (("C", o1), ("D", o2), ("A", t1), ("B", t2)):
        ctx.draw.ellipse(
            (
                point[0] - dot_radius,
                point[1] - dot_radius,
                point[0] + dot_radius,
                point[1] + dot_radius,
            ),
            fill=ctx.line_color if label in {"C", "D"} else ctx.accent_color,
            outline=ctx.line_color,
            width=1,
        )
        label_bboxes[f"{label}_label"] = draw_readout_centered(
            ctx,
            label,
            add(point, point_label_offsets[label]),
            small=True,
            backed=True,
        )
    label_bboxes["E_label"] = draw_readout_centered(
        ctx,
        "E",
        add(aux_e, point_label_offsets["E"]),
        small=True,
        backed=True,
    )
    occupied_for_radius_labels = list(label_bboxes.values())
    occupied_for_radius_labels.extend((radius_o1_segment_bbox, radius_o2_segment_bbox))
    label_bboxes["radius_o1"] = _draw_radius_callout(
        ctx,
        text=f"CA={int(spec.radius_o1)}",
        center=o1,
        tangent_point=t1,
        radius_px=float(radius_o1_px),
        occupied_bboxes=occupied_for_radius_labels,
    )
    occupied_for_radius_labels.append(label_bboxes["radius_o1"])
    label_bboxes["radius_o2"] = _draw_radius_callout(
        ctx,
        text=f"DB={int(spec.radius_o2)}",
        center=o2,
        tangent_point=t2,
        radius_px=float(radius_o2_px),
        occupied_bboxes=occupied_for_radius_labels,
    )
    assert_bboxes_inside(
        label_bboxes.values(),
        width=ctx.width,
        height=ctx.height,
        error_message="circle-pair tangent label too close to canvas edge",
    )

    annotation = {"C": o1, "D": o2, "A": t1, "B": t2}
    scene_entities = (
        {
            "entity_id": "circle_c",
            "entity_type": "circle",
            "center": point_to_list(o1),
            "radius_units": int(spec.radius_o1),
            "radius_px": round(float(radius_o1_px), 3),
            "bbox": bbox_to_list(o1_bbox),
        },
        {
            "entity_id": "circle_d",
            "entity_type": "circle",
            "center": point_to_list(o2),
            "radius_units": int(spec.radius_o2),
            "radius_px": round(float(radius_o2_px), 3),
            "bbox": bbox_to_list(o2_bbox),
        },
        {
            "entity_id": "common_tangent_segment",
            "entity_type": "segment",
            "endpoints": [point_to_list(t1), point_to_list(t2)],
            "length_units": int(spec.tangent_length),
            "bbox": bbox_to_list(
                bbox_from_points((t1, t2), width=ctx.width, height=ctx.height, pad=16.0)
            ),
        },
    )
    render_map = {
        "coord_space": "pixel",
        "centers": {"C": point_to_list(o1), "D": point_to_list(o2)},
        "tangent_points": {"A": point_to_list(t1), "B": point_to_list(t2)},
        "circle_bboxes": {"C": bbox_to_list(o1_bbox), "D": bbox_to_list(o2_bbox)},
        "tangent_segment": [point_to_list(t1), point_to_list(t2)],
        "center_segment": [point_to_list(o1), point_to_list(o2)],
        "radii_segments": {
            "CA": [point_to_list(o1), point_to_list(t1)],
            "DB": [point_to_list(o2), point_to_list(t2)],
        },
        "auxiliary_right_triangle": {
            "E": point_to_list(aux_e),
            "CE": [point_to_list(o1), point_to_list(aux_e)],
            "ED": [point_to_list(aux_e), point_to_list(o2)],
            "CD": [point_to_list(o1), point_to_list(o2)],
            "ED_length_units": abs(int(spec.radius_o2) - int(spec.radius_o1)),
        },
        "label_bboxes": {
            key: bbox_to_list(value) for key, value in label_bboxes.items()
        },
        "scale_px_per_unit": round(float(scale), 3),
        "tangent_side": str(spec.tangent_side),
        "larger_circle_side": str(spec.larger_circle_side),
    }
    return RenderedPairTangentScene(
        image=ctx.image,
        annotation_keyed_points={
            key: tuple(value) for key, value in annotation.items()
        },
        annotation_roles=tuple(spec.annotation_roles),
        label_bboxes=dict(label_bboxes),
        scene_entities=scene_entities,
        render_map=render_map,
    )


__all__ = [
    "create_pair_tangent_render_context",
    "render_pair_tangent_scene",
]
