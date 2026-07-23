"""Rendering primitives for tangent-packing diagrams."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from PIL import Image, ImageChops, ImageDraw

from trace_tasks.core.sampling import uniform_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.geometry.shared.diagram_style import (
    geometry_diagram_style_metadata,
    prepare_geometry_diagram_style_and_background,
)
from trace_tasks.tasks.geometry.shared.measurement_rendering import (
    bbox_from_points,
    bbox_to_list,
    draw_readout_centered,
    fmt_measure,
    pad_bbox,
)
from trace_tasks.tasks.geometry.shared.scene_transform import LazySceneTransform
from trace_tasks.tasks.geometry.shared.shape_style import (
    extract_background_anchor_colors,
    sample_geometry_shape_style,
)
from trace_tasks.tasks.shared.config_defaults import group_default

from .state import SCENE_ID
from .measurements import case_trace_values
from .state import BBox, Color, Point, RenderContext, RenderedTangentPackingScene, TangentPackingProblem


def _union_bboxes(bboxes: Sequence[BBox], *, width: int, height: int, pad: float = 0.0) -> BBox:
    if not bboxes:
        return (0.0, 0.0, 1.0, 1.0)
    return pad_bbox(
        (
            min(bbox[0] for bbox in bboxes),
            min(bbox[1] for bbox in bboxes),
            max(bbox[2] for bbox in bboxes),
            max(bbox[3] for bbox in bboxes),
        ),
        pad,
        width=width,
        height=height,
    )


def _ellipse_bbox(center: Point, radius: float) -> BBox:
    return (
        float(center[0]) - float(radius),
        float(center[1]) - float(radius),
        float(center[0]) + float(radius),
        float(center[1]) + float(radius),
    )


def _rect_points(rect: BBox) -> tuple[Point, Point, Point, Point]:
    return (
        (float(rect[0]), float(rect[1])),
        (float(rect[2]), float(rect[1])),
        (float(rect[2]), float(rect[3])),
        (float(rect[0]), float(rect[3])),
    )


def _closed(points: Sequence[Point]) -> list[Point]:
    return list(points) + [points[0]] if points else []


def _blend_rgb(foreground: Color, background: Color, foreground_weight: float) -> Color:
    weight = max(0.0, min(1.0, float(foreground_weight)))
    return tuple(
        int(round(float(fg) * weight + float(bg) * (1.0 - weight)))
        for fg, bg in zip(foreground, background)
    )


def _background_fill_color(ctx: RenderContext) -> Color:
    """Estimate the current page/background color without depending on style internals."""

    sample_points = (
        (8, 8),
        (ctx.width - 9, 8),
        (8, ctx.height - 9),
        (ctx.width - 9, ctx.height - 9),
    )
    samples: list[Color] = []
    for x, y in sample_points:
        pixel = ctx.image.getpixel(
            (max(0, min(ctx.width - 1, x)), max(0, min(ctx.height - 1, y)))
        )
        if isinstance(pixel, int):
            samples.append((int(pixel), int(pixel), int(pixel)))
        else:
            samples.append(tuple(int(value) for value in pixel[:3]))
    return tuple(int(round(sum(sample[index] for sample in samples) / len(samples))) for index in range(3))


def _squared_color_distance(color_a: Color, color_b: Color) -> int:
    return sum((int(channel_a) - int(channel_b)) ** 2 for channel_a, channel_b in zip(color_a, color_b))


def _visible_hatch_color(ctx: RenderContext, shaded_fill: Color) -> Color:
    darker = tuple(max(0, int(channel) - 96) for channel in shaded_fill)
    lighter = tuple(min(255, int(channel) + 96) for channel in shaded_fill)
    candidates = (
        ctx.accent_color,
        ctx.line_color,
        ctx.label_color,
        darker,
        lighter,
    )
    return max(candidates, key=lambda candidate: _squared_color_distance(candidate, shaded_fill))


def _apply_gap_shading(ctx: RenderContext, mask: Image.Image) -> None:
    """Fill only the actual container-minus-packed-shape gap region."""

    background = _background_fill_color(ctx)
    shaded_fill = _blend_rgb(ctx.shaded_color, background, 0.72)
    hatch_color = _visible_hatch_color(ctx, shaded_fill)
    ctx.image.paste(Image.new("RGB", ctx.image.size, shaded_fill), (0, 0), mask)

    line_mask = Image.new("L", ctx.image.size, 0)
    line_draw = ImageDraw.Draw(line_mask)
    scale = float(ctx.scene_transform.transform.scale)
    spacing = max(10, int(round(14.0 * scale)))
    hatch_width = max(2, int(round(float(ctx.line_width) / 2.0)))
    for offset in range(-ctx.height, ctx.width + ctx.height + spacing, spacing):
        line_draw.line(
            [(offset, ctx.height + 8), (offset + ctx.height + 8, -8)],
            fill=210,
            width=hatch_width,
        )
    clipped_line_mask = ImageChops.multiply(mask, line_mask)
    ctx.image.paste(Image.new("RGB", ctx.image.size, hatch_color), (0, 0), clipped_line_mask)
    ctx.draw = ImageDraw.Draw(ctx.image)


def _draw_measure_label(ctx: RenderContext, text: str, center: Point, *, small: bool = True) -> BBox:
    return draw_readout_centered(ctx, text, center, small=small, backed=True)


def _gap_shading_render_map() -> dict[str, Any]:
    return {
        "gap_shading_mode": "container_minus_packed_shape_mask",
        "packed_region_fill_mode": "background_unshaded",
        "gap_texture": "high_contrast_diagonal_hatch",
    }


def _transformed_radius(ctx: RenderContext, radius: float) -> float:
    return float(radius) * float(ctx.scene_transform.transform.scale)


def _draw_dimension(
    ctx: RenderContext,
    start: Point,
    end: Point,
    label: str,
    *,
    label_center: Point | None = None,
    label_offset: Point = (0.0, 0.0),
    color: Color | None = None,
) -> BBox:
    """Draw one dimension segment plus backed label; return the combined visual bbox."""

    draw_color = color if color is not None else ctx.label_color
    ctx.draw.line([start, end], fill=draw_color, width=max(2, ctx.line_width - 1))
    tick = 8.0
    dx = float(end[0]) - float(start[0])
    dy = float(end[1]) - float(start[1])
    length = math.hypot(dx, dy)
    if length > 1e-9:
        nx = -dy / length
        ny = dx / length
        for point in (start, end):
            ctx.draw.line(
                [
                    (point[0] - nx * tick, point[1] - ny * tick),
                    (point[0] + nx * tick, point[1] + ny * tick),
                ],
                fill=draw_color,
                width=max(2, ctx.line_width - 1),
            )
    center = label_center
    if center is None:
        center = (
            (float(start[0]) + float(end[0])) / 2.0 + float(label_offset[0]),
            (float(start[1]) + float(end[1])) / 2.0 + float(label_offset[1]),
        )
    label_bbox = _draw_measure_label(
        ctx,
        label,
        center,
        small=True,
    )
    line_bbox = bbox_from_points((start, end), width=ctx.width, height=ctx.height, pad=10.0)
    return _union_bboxes((line_bbox, label_bbox), width=ctx.width, height=ctx.height)


def create_render_context(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    namespace: str,
) -> tuple[RenderContext, dict[str, Any]]:
    """Create a styled PIL render context for one tangent-packing sample."""

    rng = spawn_rng(int(instance_seed), f"{namespace}.render")
    width = int(params.get("canvas_width", group_default(render_defaults, "canvas_width", 780)))
    height = int(params.get("canvas_height", group_default(render_defaults, "canvas_height", 560)))
    image, background_meta, diagram_style, diagram_style_meta = prepare_geometry_diagram_style_and_background(
        canvas_width=int(width),
        canvas_height=int(height),
        scene_id=SCENE_ID,
        instance_seed=int(instance_seed),
        params=params,
        namespace_suffix="tangent_packing_background",
    )
    shape_style = sample_geometry_shape_style(
        rng,
        params=params,
        render_defaults=render_defaults,
        anchor_colors=extract_background_anchor_colors(background_meta),
    )
    palettes: tuple[tuple[Color, Color, Color, Color], ...] = (
        ((229, 240, 255), (247, 222, 186), (26, 123, 185), (126, 143, 156)),
        ((236, 247, 232), (255, 224, 210), (38, 143, 104), (150, 142, 132)),
        ((248, 238, 252), (226, 242, 255), (123, 95, 190), (143, 154, 172)),
        ((255, 244, 224), (226, 240, 255), (196, 102, 44), (132, 146, 160)),
    )
    palette_rng = spawn_rng(int(instance_seed), f"{namespace}.palette")
    fill_color, shaded_color, accent_color, muted_color = uniform_choice(
        palette_rng,
        palettes,
    )
    font_size = int(params.get("label_font_size", group_default(render_defaults, "label_font_size", 22)))
    small_font_size = int(params.get("small_label_font_size", group_default(render_defaults, "small_label_font_size", 18)))
    line_width = int(params.get("line_width", group_default(render_defaults, "line_width", 4)))
    from trace_tasks.tasks.shared.text_rendering import load_font

    ctx = RenderContext(
        rng=rng,
        image=image,
        draw=ImageDraw.Draw(image),
        width=int(width),
        height=int(height),
        line_color=shape_style.line_color,
        label_color=shape_style.label_color,
        label_stroke_color=shape_style.label_stroke_color,
        fill_color=fill_color,
        shaded_color=shaded_color,
        accent_color=accent_color,
        muted_color=muted_color,
        line_width=max(2, int(line_width)),
        font=load_font(max(12, int(font_size)), bold=False),
        small_font=load_font(max(10, int(small_font_size)), bold=False),
        scene_transform=LazySceneTransform(
            rng,
            params=params,
            render_defaults=render_defaults,
            canvas_width=int(width),
            canvas_height=int(height),
        ),
    )
    return ctx, {
        "background_style": dict(background_meta),
        "technical_diagram_style": geometry_diagram_style_metadata(diagram_style),
        "technical_diagram_style_resolution": dict(diagram_style_meta),
        "shape_style": shape_style.to_trace_dict(),
        "line_width": int(ctx.line_width),
        "label_font_size": int(font_size),
        "small_label_font_size": int(small_font_size),
        "fill_color": list(fill_color),
        "shaded_color": list(shaded_color),
        "accent_color": list(accent_color),
        "muted_color": list(muted_color),
    }


def _build_rendered_scene(
    *,
    ctx: RenderContext,
    problem: TangentPackingProblem,
    scene_bbox: BBox,
    annotation_bbox: BBox,
    label_bboxes: Mapping[str, BBox],
    scene_entities: tuple[dict[str, Any], ...],
    render_map: Mapping[str, Any],
    witness: Mapping[str, Any],
) -> RenderedTangentPackingScene:
    target_bbox = pad_bbox(
        annotation_bbox,
        2.0,
        width=ctx.width,
        height=ctx.height,
    )
    return RenderedTangentPackingScene(
        image=ctx.image,
        answer=float(problem.answer),
        annotation_bboxes={"diagram": target_bbox},
        annotation_roles=("diagram",),
        label_bboxes=dict(label_bboxes),
        scene_entities=tuple(scene_entities),
        render_map={
            "coord_space": "pixel",
            "label_bboxes": {key: bbox_to_list(value) for key, value in label_bboxes.items()},
            "diagram_bbox": bbox_to_list(scene_bbox),
            "target_annotation_bbox": bbox_to_list(target_bbox),
            **dict(render_map),
        },
        witness=dict(witness),
    )


def render_circle_in_square_scene(
    ctx: RenderContext,
    problem: TangentPackingProblem,
) -> RenderedTangentPackingScene:
    """Render a circle tangent inside a square container."""

    case = problem.case
    square = (185.0, 125.0, 485.0, 425.0)
    square_dim_y_raw = square[3] + 30.0
    square_dim_label_raw = ((square[0] + square[2]) / 2.0, square[3] + 76.0)
    center_raw = (335.0, 275.0)
    radius_raw = 150.0
    circle_extents = (
        (center_raw[0] - radius_raw, center_raw[1]),
        (center_raw[0] + radius_raw, center_raw[1]),
        (center_raw[0], center_raw[1] - radius_raw),
        (center_raw[0], center_raw[1] + radius_raw),
    )
    ctx.scene_transform.resolve(
        _rect_points(square)
        + circle_extents
        + (
            (580.0, 104.0),
            (560.0, 274.0),
            (590.0, 104.0),
            (square[0], square_dim_y_raw),
            (square[2], square_dim_y_raw),
            square_dim_label_raw,
        )
    )
    square_points = ctx.scene_transform.points(_rect_points(square))
    center = ctx.scene_transform.point(center_raw)
    radius_px = _transformed_radius(ctx, radius_raw)
    circle_bbox = _ellipse_bbox(center, radius_px)
    label_bboxes: dict[str, BBox] = {}
    use_gap_shading = problem.target_kind == "shaded_area" or problem.support_kind == "shaded_area"
    if use_gap_shading:
        gap_mask = Image.new("L", ctx.image.size, 0)
        gap_draw = ImageDraw.Draw(gap_mask)
        gap_draw.polygon(square_points, fill=255)
        gap_draw.ellipse(circle_bbox, fill=0)
        _apply_gap_shading(ctx, gap_mask)
    else:
        ctx.draw.polygon(square_points, fill=ctx.fill_color)
        ctx.draw.ellipse(circle_bbox, outline=ctx.accent_color, width=ctx.line_width)
    ctx.draw.line(_closed(square_points), fill=ctx.line_color, width=ctx.line_width, joint="curve")
    ctx.draw.ellipse(circle_bbox, outline=ctx.accent_color, width=ctx.line_width)
    if problem.support_kind == "shaded_area":
        label_bboxes["support"] = _draw_measure_label(ctx, problem.support_text, ctx.scene_transform.point((580.0, 104.0)))
    else:
        label_bboxes["support"] = _draw_dimension(
            ctx,
            ctx.scene_transform.point((square[0], square_dim_y_raw)),
            ctx.scene_transform.point((square[2], square_dim_y_raw)),
            problem.support_text,
            label_center=ctx.scene_transform.point(square_dim_label_raw),
        )
    if problem.target_kind == "radius":
        radius_endpoint = ctx.scene_transform.point((center_raw[0] + radius_raw, center_raw[1]))
        ctx.draw.line([center, radius_endpoint], fill=ctx.accent_color, width=max(2, ctx.line_width - 1))
        label_bboxes["target"] = _draw_measure_label(ctx, problem.target_text, ctx.scene_transform.point((560.0, 274.0)))
    else:
        label_bboxes["target"] = _draw_measure_label(ctx, problem.target_text, ctx.scene_transform.point((590.0, 104.0)))
    scene_bbox = bbox_from_points(square_points, width=ctx.width, height=ctx.height, pad=10.0)
    circle_padded = pad_bbox(circle_bbox, 4.0, width=ctx.width, height=ctx.height)
    witness = {
        "scene_variant": "circle_in_square",
        "formula": str(problem.formula_text),
        "answer_value": float(problem.answer),
        **case_trace_values(case),
    }
    return _build_rendered_scene(
        ctx=ctx,
        problem=problem,
        scene_bbox=scene_bbox,
        annotation_bbox=circle_padded if problem.target_kind == "radius" else scene_bbox,
        label_bboxes=label_bboxes,
        scene_entities=(
            {"entity_id": "square_container", "entity_type": "square", "bbox": bbox_to_list(scene_bbox)},
            {
                "entity_id": "inscribed_circle",
                "entity_type": "circle",
                "bbox": bbox_to_list(circle_padded),
                "center": [round(center[0], 3), round(center[1], 3)],
                "radius_px": round(radius_px, 3),
            },
        ),
        render_map={
            "scene_variant": "circle_in_square",
            "square_bbox": bbox_to_list(scene_bbox),
            "circle_bbox": bbox_to_list(circle_padded),
            "shaded_region_bbox": bbox_to_list(scene_bbox),
            **(_gap_shading_render_map() if use_gap_shading else {}),
        },
        witness=witness,
    )


def render_square_in_circle_scene(
    ctx: RenderContext,
    problem: TangentPackingProblem,
) -> RenderedTangentPackingScene:
    """Render a square tangent inside a circle container."""

    case = problem.case
    center_raw = (350.0, 280.0)
    radius_raw = 170.0
    square_points_raw = ((350.0, 110.0), (520.0, 280.0), (350.0, 450.0), (180.0, 280.0))
    ctx.scene_transform.resolve(
        tuple(square_points_raw)
        + (
            (center_raw[0] - radius_raw, center_raw[1]),
            (center_raw[0] + radius_raw, center_raw[1]),
            (center_raw[0], center_raw[1] - radius_raw),
            (center_raw[0], center_raw[1] + radius_raw),
            (575.0, 104.0),
            (570.0, 454.0),
            (440.0, 252.0),
        )
    )
    center = ctx.scene_transform.point(center_raw)
    radius_px = _transformed_radius(ctx, radius_raw)
    square_points = ctx.scene_transform.points(square_points_raw)
    circle_bbox = _ellipse_bbox(center, radius_px)
    label_bboxes: dict[str, BBox] = {}
    use_gap_shading = problem.target_kind == "shaded_area" or problem.support_kind == "shaded_area"
    if use_gap_shading:
        gap_mask = Image.new("L", ctx.image.size, 0)
        gap_draw = ImageDraw.Draw(gap_mask)
        gap_draw.ellipse(circle_bbox, fill=255)
        gap_draw.polygon(square_points, fill=0)
        _apply_gap_shading(ctx, gap_mask)
    else:
        ctx.draw.ellipse(circle_bbox, fill=ctx.fill_color)
    ctx.draw.ellipse(circle_bbox, outline=ctx.line_color, width=ctx.line_width)
    ctx.draw.line(_closed(square_points), fill=ctx.accent_color, width=ctx.line_width, joint="curve")
    radius_endpoint = ctx.scene_transform.point((center_raw[0] + radius_raw, center_raw[1]))
    ctx.draw.line([center, radius_endpoint], fill=ctx.accent_color, width=max(2, ctx.line_width - 1))
    if problem.support_kind == "shaded_area":
        label_bboxes["support"] = _draw_measure_label(ctx, problem.support_text, ctx.scene_transform.point((575.0, 104.0)))
    else:
        label_bboxes["support"] = _draw_measure_label(ctx, problem.support_text, ctx.scene_transform.point((440.0, 252.0)))
    if problem.target_kind == "square_side":
        label_bboxes["target"] = _draw_measure_label(ctx, problem.target_text, ctx.scene_transform.point((570.0, 454.0)))
    else:
        label_bboxes["target"] = _draw_measure_label(ctx, problem.target_text, ctx.scene_transform.point((575.0, 104.0)))
    scene_bbox = pad_bbox(circle_bbox, 10.0, width=ctx.width, height=ctx.height)
    square_bbox = bbox_from_points(square_points, width=ctx.width, height=ctx.height, pad=4.0)
    witness = {
        "scene_variant": "square_in_circle",
        "formula": str(problem.formula_text),
        "answer_value": float(problem.answer),
        **case_trace_values(case),
    }
    return _build_rendered_scene(
        ctx=ctx,
        problem=problem,
        scene_bbox=scene_bbox,
        annotation_bbox=square_bbox if problem.target_kind == "square_side" else scene_bbox,
        label_bboxes=label_bboxes,
        scene_entities=(
            {
                "entity_id": "circle_container",
                "entity_type": "circle",
                "bbox": bbox_to_list(scene_bbox),
                "center": [round(center[0], 3), round(center[1], 3)],
                "radius_px": round(radius_px, 3),
            },
            {
                "entity_id": "inscribed_square",
                "entity_type": "square",
                "bbox": bbox_to_list(square_bbox),
                "points": [[round(x, 3), round(y, 3)] for x, y in square_points],
            },
        ),
        render_map={
            "scene_variant": "square_in_circle",
            "circle_bbox": bbox_to_list(scene_bbox),
            "square_points": [[round(x, 3), round(y, 3)] for x, y in square_points],
            **(_gap_shading_render_map() if use_gap_shading else {}),
        },
        witness=witness,
    )


def render_two_circles_rectangle_scene(
    ctx: RenderContext,
    problem: TangentPackingProblem,
) -> RenderedTangentPackingScene:
    """Render two equal tangent circles inside a rectangle."""

    case = problem.case
    rect = (110.0, 160.0, 670.0, 440.0)
    rect_dim_y_raw = rect[3] + 30.0
    rect_dim_label_raw = ((rect[0] + rect[2]) / 2.0, rect[3] + 76.0)
    c1_raw = (250.0, 300.0)
    c2_raw = (530.0, 300.0)
    radius_raw = 140.0
    ctx.scene_transform.resolve(
        _rect_points(rect)
        + (
            (c1_raw[0] - radius_raw, c1_raw[1]),
            (c1_raw[0] + radius_raw, c1_raw[1]),
            (c1_raw[0], c1_raw[1] - radius_raw),
            (c1_raw[0], c1_raw[1] + radius_raw),
            (c2_raw[0] - radius_raw, c2_raw[1]),
            (c2_raw[0] + radius_raw, c2_raw[1]),
            (c2_raw[0], c2_raw[1] - radius_raw),
            (c2_raw[0], c2_raw[1] + radius_raw),
            (570.0, 112.0),
            (250.0, 262.0),
            (rect[0], rect_dim_y_raw),
            (rect[2], rect_dim_y_raw),
            rect_dim_label_raw,
        )
    )
    rect_points = ctx.scene_transform.points(_rect_points(rect))
    c1 = ctx.scene_transform.point(c1_raw)
    c2 = ctx.scene_transform.point(c2_raw)
    radius_px = _transformed_radius(ctx, radius_raw)
    circle1_bbox = _ellipse_bbox(c1, radius_px)
    circle2_bbox = _ellipse_bbox(c2, radius_px)
    label_bboxes: dict[str, BBox] = {}
    use_gap_shading = problem.target_kind == "shaded_area" or problem.support_kind == "shaded_area"
    if use_gap_shading:
        gap_mask = Image.new("L", ctx.image.size, 0)
        gap_draw = ImageDraw.Draw(gap_mask)
        gap_draw.polygon(rect_points, fill=255)
        gap_draw.ellipse(circle1_bbox, fill=0)
        gap_draw.ellipse(circle2_bbox, fill=0)
        _apply_gap_shading(ctx, gap_mask)
    else:
        ctx.draw.polygon(rect_points, fill=ctx.fill_color)
    ctx.draw.line(_closed(rect_points), fill=ctx.line_color, width=ctx.line_width, joint="curve")
    ctx.draw.ellipse(circle1_bbox, outline=ctx.accent_color, width=ctx.line_width)
    ctx.draw.ellipse(circle2_bbox, outline=ctx.accent_color, width=ctx.line_width)
    ctx.draw.line([c1, ctx.scene_transform.point((c1_raw[0] + radius_raw, c1_raw[1]))], fill=ctx.accent_color, width=max(2, ctx.line_width - 1))
    if problem.support_kind == "shaded_area":
        label_bboxes["support"] = _draw_measure_label(ctx, problem.support_text, ctx.scene_transform.point((570.0, 112.0)))
    else:
        label_bboxes["support"] = _draw_dimension(
            ctx,
            ctx.scene_transform.point((rect[0], rect_dim_y_raw)),
            ctx.scene_transform.point((rect[2], rect_dim_y_raw)),
            problem.support_text,
            label_center=ctx.scene_transform.point(rect_dim_label_raw),
        )
    if problem.target_kind == "radius":
        label_bboxes["target"] = _draw_measure_label(ctx, problem.target_text, ctx.scene_transform.point((250.0, 262.0)))
    else:
        label_bboxes["target"] = _draw_measure_label(ctx, problem.target_text, ctx.scene_transform.point((570.0, 112.0)))
    scene_bbox = bbox_from_points(rect_points, width=ctx.width, height=ctx.height, pad=10.0)
    circle1_padded = pad_bbox(circle1_bbox, 4.0, width=ctx.width, height=ctx.height)
    circle2_padded = pad_bbox(circle2_bbox, 4.0, width=ctx.width, height=ctx.height)
    witness = {
        "scene_variant": "two_circles_in_rectangle",
        "formula": str(problem.formula_text),
        "answer_value": float(problem.answer),
        **case_trace_values(case),
    }
    return _build_rendered_scene(
        ctx=ctx,
        problem=problem,
        scene_bbox=scene_bbox,
        annotation_bbox=circle1_padded if problem.target_kind == "radius" else scene_bbox,
        label_bboxes=label_bboxes,
        scene_entities=(
            {"entity_id": "rectangle_container", "entity_type": "rectangle", "bbox": bbox_to_list(scene_bbox)},
            {
                "entity_id": "left_circle",
                "entity_type": "circle",
                "bbox": bbox_to_list(circle1_padded),
                "center": [round(c1[0], 3), round(c1[1], 3)],
                "radius_px": round(radius_px, 3),
            },
            {
                "entity_id": "right_circle",
                "entity_type": "circle",
                "bbox": bbox_to_list(circle2_padded),
                "center": [round(c2[0], 3), round(c2[1], 3)],
                "radius_px": round(radius_px, 3),
            },
        ),
        render_map={
            "scene_variant": "two_circles_in_rectangle",
            "rectangle_bbox": bbox_to_list(scene_bbox),
            "circle_bboxes": [bbox_to_list(circle1_padded), bbox_to_list(circle2_padded)],
            **(_gap_shading_render_map() if use_gap_shading else {}),
        },
        witness=witness,
    )


__all__ = [
    "create_render_context",
    "render_circle_in_square_scene",
    "render_square_in_circle_scene",
    "render_two_circles_rectangle_scene",
]
