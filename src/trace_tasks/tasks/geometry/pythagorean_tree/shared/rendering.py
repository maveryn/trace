"""Rendering primitives for Pythagorean attached-square tree diagrams."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from PIL import ImageDraw

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.geometry.shared.diagram_style import prepare_geometry_diagram_style_and_background
from trace_tasks.tasks.geometry.shared.measurement_rendering import bbox_from_points, bbox_to_list, pad_bbox
from trace_tasks.tasks.geometry.shared.readout_fill_style import resolve_readout_fill_style
from trace_tasks.tasks.geometry.shared.scene_transform import LazySceneTransform
from trace_tasks.tasks.geometry.shared.vector2d import add, mul, sub, unit
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.font_assets import font_asset_version, get_font_family_record, sample_font_family
from trace_tasks.tasks.shared.text_legibility import draw_text_traced
from trace_tasks.tasks.shared.text_rendering import load_font

from .defaults import SCENE_ID
from .construction import polygon_center, rotate_point, square_on_segment, transform_points
from .state import BBox, Color, Point, Polygon, PythagoreanTreePlan, RenderContext, RenderedPythagoreanTreeScene


def make_render_context(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
) -> tuple[RenderContext, dict[str, Any]]:
    """Resolve deterministic theme, font, canvas, and transform state."""

    width = int(params.get("canvas_width", group_default(rendering_defaults, "canvas_width", 820)))
    height = int(params.get("canvas_height", group_default(rendering_defaults, "canvas_height", 600)))
    image, background_meta, diagram_style, diagram_style_meta = prepare_geometry_diagram_style_and_background(
        instance_seed=int(instance_seed),
        params=params,
        scene_id=SCENE_ID,
        canvas_width=int(width),
        canvas_height=int(height),
        allow_dark=True,
        require_grid=False,
        style_profile="analytical_diagram",
    )
    fills: tuple[tuple[Color, Color, Color, Color], ...] = (
        (
            tuple(int(value) for value in diagram_style.panel_fill_rgb),
            tuple(int(value) for value in diagram_style.panel_alt_fill_rgb),
            tuple(int(value) for value in diagram_style.option_fill_rgb),
            tuple(int(value) for value in diagram_style.muted_fill_rgb),
        ),
        (
            tuple(int(value) for value in diagram_style.panel_alt_fill_rgb),
            tuple(int(value) for value in diagram_style.option_fill_rgb),
            tuple(int(value) for value in diagram_style.muted_fill_rgb),
            tuple(int(value) for value in diagram_style.panel_fill_rgb),
        ),
        (
            tuple(int(value) for value in diagram_style.option_fill_rgb),
            tuple(int(value) for value in diagram_style.panel_fill_rgb),
            tuple(int(value) for value in diagram_style.panel_alt_fill_rgb),
            tuple(int(value) for value in diagram_style.muted_fill_rgb),
        ),
    )
    readout_style = resolve_readout_fill_style(
        instance_seed=int(instance_seed),
        namespace=f"geometry.{SCENE_ID}.readout_fill",
        diagram_style=diagram_style,
        background_meta=background_meta,
        candidate_palettes=fills,
        params=params,
    )
    triangle_fill, leg_fill, other_leg_fill, hyp_fill = readout_style.fill_colors
    font_family = str(
        params.get(
            "readout_font_family",
            sample_font_family(
                role="readout",
                instance_seed=int(instance_seed),
                namespace=f"geometry.{SCENE_ID}.font_family",
                params=params,
            ),
        )
    )
    font_record = get_font_family_record(str(font_family))
    font_size = int(params.get("label_font_size", group_default(rendering_defaults, "label_font_size", 22)))
    small_font_size = int(
        params.get("small_label_font_size", group_default(rendering_defaults, "small_label_font_size", 18))
    )
    line_width = int(params.get("line_width", group_default(rendering_defaults, "line_width", 3)))
    label_stroke_width = int(
        params.get(
            "label_stroke_width",
            group_default(rendering_defaults, "label_stroke_width", int(diagram_style.label_stroke_width_px)),
        )
    )
    font_meta = {
        "font_family": str(font_family),
        "font_role": "readout",
        "font_source": getattr(font_record, "source", ""),
        "font_asset_version": font_asset_version(),
    }
    return (
        RenderContext(
            image=image,
            draw=ImageDraw.Draw(image),
            width=int(width),
            height=int(height),
            line_color=tuple(int(value) for value in diagram_style.stroke_rgb),
            secondary_color=tuple(int(value) for value in diagram_style.secondary_stroke_rgb),
            label_color=tuple(int(value) for value in readout_style.label_color),
            label_stroke_color=tuple(int(value) for value in readout_style.label_stroke_color),
            triangle_fill=tuple(int(value) for value in triangle_fill),
            leg_square_fill=tuple(int(value) for value in leg_fill),
            other_leg_square_fill=tuple(int(value) for value in other_leg_fill),
            hypotenuse_square_fill=tuple(int(value) for value in hyp_fill),
            accent_color=tuple(int(value) for value in diagram_style.accent_rgb),
            line_width=max(2, int(line_width)),
            label_stroke_width=max(0, int(label_stroke_width)),
            font=load_font(max(12, int(font_size)), bold=False, font_family=str(font_family)),
            small_font=load_font(max(10, int(small_font_size)), bold=False, font_family=str(font_family)),
            readout_text_metadata=dict(readout_style.metadata["readout_text_style"]),
            diagram_style_meta=dict(diagram_style_meta),
            background_meta=dict(background_meta),
            font_meta=dict(font_meta),
            scene_transform=LazySceneTransform(
                spawn_rng(int(instance_seed), f"geometry.{SCENE_ID}.scene_transform"),
                params=params,
                render_defaults=rendering_defaults,
                canvas_width=int(width),
                canvas_height=int(height),
            ),
        ),
        {
            "diagram_style": dict(diagram_style_meta),
            "background": dict(background_meta),
            "font": dict(font_meta),
            "readout_fill_style": dict(readout_style.metadata),
        },
    )


def _draw_text_centered(ctx: RenderContext, text: str, center: Point, *, small: bool = True) -> BBox:
    font = ctx.small_font if bool(small) else ctx.font
    x, y = float(center[0]), float(center[1])
    draw_text_traced(
        ctx.draw,
        (x, y),
        str(text),
        anchor="mm",
        font=font,
        fill=ctx.label_color,
        stroke_width=max(0, int(ctx.label_stroke_width)),
        stroke_fill=ctx.label_stroke_color,
        role="readout",
        required=True,
        extra_metadata=ctx.readout_text_metadata,
    )
    bbox = ctx.draw.textbbox(
        (x, y),
        str(text),
        anchor="mm",
        font=font,
        stroke_width=max(0, int(ctx.label_stroke_width)),
    )
    return pad_bbox(bbox, 3.0, width=ctx.width, height=ctx.height)


def _bbox_dict(polygons: Mapping[str, Polygon], *, width: int, height: int) -> dict[str, BBox]:
    return {
        str(key): bbox_from_points(points, width=int(width), height=int(height), pad=2.0)
        for key, points in polygons.items()
    }


def _assert_bboxes_inside(bboxes: Sequence[BBox], *, width: int, height: int) -> None:
    for bbox in bboxes:
        x0, y0, x1, y1 = [float(value) for value in bbox]
        if x0 <= 4 or y0 <= 4 or x1 >= float(width) - 4 or y1 >= float(height) - 4:
            raise ValueError("pythagorean tree label or square too close to canvas edge")


def _draw_right_angle_marker(ctx: RenderContext, *, vertex: Point, arm_a: Point, arm_b: Point) -> BBox:
    size = 18.0
    u = unit(sub(arm_a, vertex))
    v = unit(sub(arm_b, vertex))
    p1 = add(vertex, mul(u, size))
    p2 = add(p1, mul(v, size))
    p3 = add(vertex, mul(v, size))
    ctx.draw.line([p1, p2, p3], fill=ctx.secondary_color, width=max(2, ctx.line_width - 1))
    return bbox_from_points((p1, p2, p3), width=ctx.width, height=ctx.height, pad=ctx.line_width + 2)


def render_pythagorean_tree_scene(
    ctx: RenderContext,
    plan: PythagoreanTreePlan,
    *,
    instance_seed: int,
) -> RenderedPythagoreanTreeScene:
    """Render one attached-square tree diagram and expose role-bound geometry."""

    rng = spawn_rng(int(instance_seed), f"geometry.{SCENE_ID}.render.scene")
    a = (0.0, 0.0)
    b = (float(plan.triple.leg_b), 0.0)
    c = (0.0, -float(plan.triple.leg_a))
    leg_square_1 = square_on_segment(a, c, away_from=b)
    leg_square_2 = square_on_segment(a, b, away_from=c)
    hypotenuse_square = square_on_segment(b, c, away_from=a)
    all_local_points = (a, b, c, *leg_square_1, *leg_square_2, *hypotenuse_square)

    base_rotation = float(rng.choice((0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0)))
    angle = base_rotation + math.radians(float(rng.uniform(-7.0, 7.0)))
    rotated = [rotate_point(point, angle) for point in all_local_points]
    min_x = min(point[0] for point in rotated)
    max_x = max(point[0] for point in rotated)
    min_y = min(point[1] for point in rotated)
    max_y = max(point[1] for point in rotated)
    span_x = max(1e-6, max_x - min_x)
    span_y = max(1e-6, max_y - min_y)
    scale = min((ctx.width - 130.0) / span_x, (ctx.height - 120.0) / span_y)
    scale *= float(rng.uniform(0.91, 0.98))
    center_after_scale = ((min_x + max_x) * scale / 2.0, (min_y + max_y) * scale / 2.0)
    target_center = (
        (ctx.width / 2.0) + float(rng.uniform(-26.0, 26.0)),
        (ctx.height / 2.0) + float(rng.uniform(-18.0, 18.0)),
    )
    offset = (target_center[0] - center_after_scale[0], target_center[1] - center_after_scale[1])

    a_px, b_px, c_px = transform_points((a, b, c), angle_radians=angle, scale=scale, offset=offset)
    square_polygons = {
        "leg_square_1": transform_points(leg_square_1, angle_radians=angle, scale=scale, offset=offset),
        "leg_square_2": transform_points(leg_square_2, angle_radians=angle, scale=scale, offset=offset),
        "hypotenuse_square": transform_points(hypotenuse_square, angle_radians=angle, scale=scale, offset=offset),
    }
    transform_points_for_fit = (
        a_px,
        b_px,
        c_px,
        *(point for polygon in square_polygons.values() for point in polygon),
    )
    ctx.scene_transform.resolve(transform_points_for_fit)
    a_px, b_px, c_px = ctx.scene_transform.points((a_px, b_px, c_px))
    square_polygons = {key: ctx.scene_transform.points(polygon) for key, polygon in square_polygons.items()}
    square_bboxes = _bbox_dict(square_polygons, width=ctx.width, height=ctx.height)
    _assert_bboxes_inside(square_bboxes.values(), width=ctx.width, height=ctx.height)

    for key, fill in (
        ("leg_square_1", ctx.leg_square_fill),
        ("leg_square_2", ctx.other_leg_square_fill),
        ("hypotenuse_square", ctx.hypotenuse_square_fill),
    ):
        polygon = square_polygons[key]
        ctx.draw.polygon(polygon, fill=fill)
        ctx.draw.line([*polygon, polygon[0]], fill=ctx.line_color, width=ctx.line_width, joint="curve")

    triangle = (a_px, b_px, c_px)
    ctx.draw.polygon(triangle, fill=ctx.triangle_fill)
    ctx.draw.line([a_px, b_px, c_px, a_px], fill=ctx.line_color, width=ctx.line_width, joint="curve")
    right_angle_bbox = _draw_right_angle_marker(ctx, vertex=a_px, arm_a=b_px, arm_b=c_px)

    label_bboxes: dict[str, BBox] = {}
    for role, text in plan.known_area_labels.items():
        label_bboxes[f"{role}_label"] = _draw_text_centered(
            ctx,
            str(text),
            polygon_center(square_polygons[str(role)]),
            small=True,
        )
    _assert_bboxes_inside(label_bboxes.values(), width=ctx.width, height=ctx.height)

    render_map = {
        "coord_space": "pixel",
        "triangle_vertices": {
            "right_angle_vertex": [round(a_px[0], 3), round(a_px[1], 3)],
            "leg_square_2_endpoint": [round(b_px[0], 3), round(b_px[1], 3)],
            "leg_square_1_endpoint": [round(c_px[0], 3), round(c_px[1], 3)],
        },
        "square_vertices": {
            key: [[round(x, 3), round(y, 3)] for x, y in polygon]
            for key, polygon in square_polygons.items()
        },
        "square_bboxes": {key: bbox_to_list(value) for key, value in square_bboxes.items()},
        "label_bboxes": {key: bbox_to_list(value) for key, value in label_bboxes.items()},
        "marker_bboxes": {"right_angle": bbox_to_list(right_angle_bbox)},
        "rotation_degrees": round(math.degrees(angle), 3),
        "scale_px_per_unit": round(float(scale), 3),
    }
    witness = {
        "rotation_degrees": round(math.degrees(angle), 3),
        "scale_px_per_unit": round(float(scale), 3),
    }
    scene_entities = (
        {"role": "right_triangle", "vertices": dict(render_map["triangle_vertices"])},
        {
            "role": "attached_squares",
            "square_bboxes": dict(render_map["square_bboxes"]),
            "square_vertices": dict(render_map["square_vertices"]),
        },
    )
    return RenderedPythagoreanTreeScene(
        image=ctx.image,
        square_polygons={key: tuple(value) for key, value in square_polygons.items()},
        square_bboxes={key: tuple(value) for key, value in square_bboxes.items()},
        triangle_vertices={
            "right_angle_vertex": a_px,
            "leg_square_2_endpoint": b_px,
            "leg_square_1_endpoint": c_px,
        },
        label_bboxes=dict(label_bboxes),
        marker_bboxes={"right_angle": right_angle_bbox},
        render_map=dict(render_map),
        witness=dict(witness),
        scene_entities=tuple(scene_entities),
    )


__all__ = ["make_render_context", "render_pythagorean_tree_scene"]
