"""Rendering helpers for the area-partition geometry scene."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from trace_tasks.core.sampling import uniform_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.geometry.shared.diagram_style import (
    GEOMETRY_STYLE_PROFILE_ANALYTICAL_DIAGRAM,
    geometry_diagram_style_metadata,
    geometry_shape_style_from_diagram_style,
    prepare_geometry_diagram_style_and_background,
)
from trace_tasks.tasks.geometry.shared.measurement_rendering import (
    bbox_from_points,
    bbox_to_list,
    draw_readout_centered,
)
from trace_tasks.tasks.geometry.shared.scene_transform import LazySceneTransform
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.font_assets import font_asset_version, get_font_family_record, sample_font_family
from trace_tasks.tasks.shared.text_legibility import contrast_ratio
from trace_tasks.tasks.shared.text_rendering import load_font

from .relations import centroid, midpoint
from .state import (
    ANNOTATION_ROLES,
    OUTER_SHAPE_ID,
    SCENE_ID,
    SHADED_REGION_ID,
    AreaPartitionProblem,
    AreaPartitionWitness,
    BBox,
    Color,
    Point,
)


@dataclass
class AreaPartitionRenderContext:
    """Low-level drawing context for one rendered partition diagram."""

    rng: Any
    image: Image.Image
    draw: ImageDraw.ImageDraw
    width: int
    height: int
    line_color: Color
    label_color: Color
    label_stroke_color: Color
    fill_color: Color
    shaded_color: Color
    accent_color: Color
    muted_color: Color
    line_width: int
    font: Any
    small_font: Any
    layout_offset: Point = (0.0, 0.0)
    font_family: str = ""
    scene_transform: LazySceneTransform | None = None


@dataclass(frozen=True)
class RenderedAreaPartitionScene:
    """Rendered area-partition scene plus trace-friendly geometry."""

    image: Image.Image
    answer: float
    annotation_bboxes: Tuple[BBox, ...]
    annotation_keyed_bboxes: Mapping[str, BBox]
    annotation_roles: Tuple[str, ...]
    label_bboxes: Dict[str, BBox]
    scene_entities: Tuple[Dict[str, Any], ...]
    render_map: Dict[str, Any]
    witness: AreaPartitionWitness


def _offset_points(ctx: AreaPartitionRenderContext, points: Sequence[Point]) -> tuple[Point, ...]:
    shifted = tuple(
        (
            float(point[0]) + float(ctx.layout_offset[0]),
            float(point[1]) + float(ctx.layout_offset[1]),
        )
        for point in points
    )
    if ctx.scene_transform is not None:
        return ctx.scene_transform.points(shifted)
    return shifted


def _clamped_readout_center(
    ctx: AreaPartitionRenderContext,
    text: str,
    center: Point,
    *,
    small: bool = True,
) -> Point:
    """Clamp a readout center so the backed text remains fully on canvas."""

    font = ctx.small_font if bool(small) else ctx.font
    stroke_width = max(0, int(getattr(ctx, "label_stroke_width", 0)))
    bbox = ctx.draw.textbbox((0, 0), str(text), font=font, stroke_width=stroke_width)
    half_w = (float(bbox[2] - bbox[0]) / 2.0) + 18.0
    half_h = (float(bbox[3] - bbox[1]) / 2.0) + 18.0
    min_x, max_x = half_w, float(ctx.width) - half_w
    min_y, max_y = half_h, float(ctx.height) - half_h
    if min_x > max_x:
        x = float(ctx.width) / 2.0
    else:
        x = min(max(float(center[0]), min_x), max_x)
    if min_y > max_y:
        y = float(ctx.height) / 2.0
    else:
        y = min(max(float(center[1]), min_y), max_y)
    return (x, y)


def _readout_text_size(
    ctx: AreaPartitionRenderContext,
    text: str,
    *,
    small: bool = True,
) -> tuple[float, float]:
    """Return one readout's approximate rendered size."""

    font = ctx.small_font if bool(small) else ctx.font
    stroke_width = max(0, int(getattr(ctx, "label_stroke_width", 0)))
    bbox = ctx.draw.textbbox((0, 0), str(text), font=font, stroke_width=stroke_width)
    return (float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1]))


def _readout_stack_centers(
    *,
    ctx: AreaPartitionRenderContext,
    texts: Sequence[str],
    x_center: float,
    y_center: float,
) -> tuple[Point, ...]:
    sizes = [_readout_text_size(ctx, text, small=True) for text in texts]
    gap = 12.0
    total_height = sum(size[1] for size in sizes) + gap * max(0, len(sizes) - 1)
    top = float(y_center) - total_height / 2.0
    centers: list[Point] = []
    cursor = top
    for _width, height in sizes:
        centers.append(
            _clamped_readout_center(
                ctx,
                texts[len(centers)],
                (x_center, cursor + height / 2.0),
                small=True,
            )
        )
        cursor += height + gap
    return tuple(centers)


def _readout_row_centers(
    *,
    ctx: AreaPartitionRenderContext,
    texts: Sequence[str],
    x_center: float,
    y_center: float,
) -> tuple[Point, ...]:
    sizes = [_readout_text_size(ctx, text, small=True) for text in texts]
    gap = 28.0
    total_width = sum(size[0] for size in sizes) + gap * max(0, len(sizes) - 1)
    left = float(x_center) - total_width / 2.0
    centers: list[Point] = []
    cursor = left
    for width, _height in sizes:
        centers.append(
            _clamped_readout_center(
                ctx,
                texts[len(centers)],
                (cursor + width / 2.0, y_center),
                small=True,
            )
        )
        cursor += width + gap
    return tuple(centers)


def _external_readout_centers(
    ctx: AreaPartitionRenderContext,
    outer_bbox: BBox,
    texts: Sequence[str],
) -> tuple[Point, ...]:
    """Place readouts outside the partition geometry when canvas whitespace allows."""

    x0, y0, x1, y1 = [float(value) for value in outer_bbox]
    max_text_width = max(_readout_text_size(ctx, text, small=True)[0] for text in texts)
    max_text_height = max(_readout_text_size(ctx, text, small=True)[1] for text in texts)
    pad = 28.0
    right_gap = float(ctx.width) - x1
    left_gap = x0
    below_gap = float(ctx.height) - y1
    above_gap = y0
    shape_center_y = (y0 + y1) / 2.0
    shape_center_x = (x0 + x1) / 2.0

    if right_gap >= max_text_width + pad * 2:
        return _readout_stack_centers(
            ctx=ctx,
            texts=texts,
            x_center=(x1 + float(ctx.width)) / 2.0,
            y_center=shape_center_y,
        )
    if left_gap >= max_text_width + pad * 2:
        return _readout_stack_centers(
            ctx=ctx,
            texts=texts,
            x_center=x0 / 2.0,
            y_center=shape_center_y,
        )
    if below_gap >= max_text_height + pad:
        return _readout_row_centers(
            ctx=ctx,
            texts=texts,
            x_center=shape_center_x,
            y_center=(y1 + float(ctx.height)) / 2.0,
        )
    if above_gap >= max_text_height + pad:
        return _readout_row_centers(
            ctx=ctx,
            texts=texts,
            x_center=shape_center_x,
            y_center=y0 / 2.0,
        )
    return _readout_row_centers(
        ctx=ctx,
        texts=texts,
        x_center=float(ctx.width) / 2.0,
        y_center=float(ctx.height) - pad,
    )


def _readout_ink_for_background(surfaces: Sequence[Color]) -> Color:
    """Choose black or white readout ink for the sampled diagram background."""

    candidates: tuple[Color, Color] = ((0, 0, 0), (255, 255, 255))
    return max(
        candidates,
        key=lambda color: min(float(contrast_ratio(color, surface)) for surface in surfaces),
    )


def _draw_equal_ticks(
    ctx: AreaPartitionRenderContext,
    segments: Sequence[Tuple[Point, Point]],
) -> BBox:
    tick_bboxes: list[BBox] = []
    for start, end in segments:
        mid = midpoint(start, end)
        dx = float(end[0]) - float(start[0])
        dy = float(end[1]) - float(start[1])
        length = max(1e-9, (dx**2 + dy**2) ** 0.5)
        nx = -dy / length
        ny = dx / length
        half = 8.0
        p0 = (mid[0] - nx * half, mid[1] - ny * half)
        p1 = (mid[0] + nx * half, mid[1] + ny * half)
        ctx.draw.line([p0, p1], fill=ctx.accent_color, width=max(2, ctx.line_width - 1))
        tick_bboxes.append(
            bbox_from_points((p0, p1), width=ctx.width, height=ctx.height, pad=4.0)
        )
    return (
        min(bbox[0] for bbox in tick_bboxes),
        min(bbox[1] for bbox in tick_bboxes),
        max(bbox[2] for bbox in tick_bboxes),
        max(bbox[3] for bbox in tick_bboxes),
    )


def _draw_parallelogram(
    ctx: AreaPartitionRenderContext,
    problem: AreaPartitionProblem,
) -> tuple[Tuple[Point, ...], Tuple[Point, ...], BBox, BBox, BBox]:
    """Draw reusable parallelogram partition variants without task routing."""

    a = (130.0, 408.0)
    b = (534.0, 408.0)
    c = (646.0, 152.0)
    d = (242.0, 152.0)
    a, b, c, d = _offset_points(ctx, (a, b, c, d))
    o = midpoint(a, c)
    outer = (a, b, c, d)
    if problem.scene_variant == "parallelogram_diagonal_half":
        shaded = (a, b, c)
        partition_points = (a, c)
        ctx.draw.polygon(outer, fill=ctx.fill_color)
        ctx.draw.polygon(shaded, fill=ctx.shaded_color)
        ctx.draw.line([a, b, c, d, a], fill=ctx.line_color, width=ctx.line_width)
        ctx.draw.line([a, c], fill=ctx.accent_color, width=ctx.line_width)
        partition_bbox = bbox_from_points(
            partition_points,
            width=ctx.width,
            height=ctx.height,
            pad=10.0,
        )
    elif problem.scene_variant == "parallelogram_diagonals_quarter":
        shaded = (a, b, o)
        partition_points = (a, b, c, d)
        ctx.draw.polygon(outer, fill=ctx.fill_color)
        ctx.draw.polygon(shaded, fill=ctx.shaded_color)
        ctx.draw.line([a, b, c, d, a], fill=ctx.line_color, width=ctx.line_width)
        ctx.draw.line([a, c], fill=ctx.accent_color, width=ctx.line_width)
        ctx.draw.line([b, d], fill=ctx.accent_color, width=ctx.line_width)
        ctx.draw.ellipse((o[0] - 5.0, o[1] - 5.0, o[0] + 5.0, o[1] + 5.0), fill=ctx.accent_color)
        partition_bbox = bbox_from_points(
            partition_points,
            width=ctx.width,
            height=ctx.height,
            pad=10.0,
        )
    else:
        m = midpoint(a, b)
        shaded = (a, m, o)
        ctx.draw.polygon(outer, fill=ctx.fill_color)
        ctx.draw.polygon(shaded, fill=ctx.shaded_color)
        ctx.draw.line([a, b, c, d, a], fill=ctx.line_color, width=ctx.line_width)
        ctx.draw.line([a, c], fill=ctx.accent_color, width=ctx.line_width)
        ctx.draw.line([b, d], fill=ctx.accent_color, width=ctx.line_width)
        ctx.draw.line([m, o], fill=ctx.accent_color, width=ctx.line_width)
        ctx.draw.ellipse((o[0] - 5.0, o[1] - 5.0, o[0] + 5.0, o[1] + 5.0), fill=ctx.accent_color)
        ticks_bbox = _draw_equal_ticks(ctx, ((a, m), (m, b)))
        partition_lines_bbox = bbox_from_points(
            (a, b, c, d, m, o),
            width=ctx.width,
            height=ctx.height,
            pad=10.0,
        )
        partition_bbox = (
            min(ticks_bbox[0], partition_lines_bbox[0]),
            min(ticks_bbox[1], partition_lines_bbox[1]),
            max(ticks_bbox[2], partition_lines_bbox[2]),
            max(ticks_bbox[3], partition_lines_bbox[3]),
        )
    outer_bbox = bbox_from_points(outer, width=ctx.width, height=ctx.height, pad=8.0)
    shaded_bbox = bbox_from_points(shaded, width=ctx.width, height=ctx.height, pad=8.0)
    return outer, shaded, outer_bbox, shaded_bbox, partition_bbox


def _draw_triangle(
    ctx: AreaPartitionRenderContext,
    problem: AreaPartitionProblem,
) -> tuple[Tuple[Point, ...], Tuple[Point, ...], BBox, BBox, BBox]:
    """Draw reusable triangle partition variants without task routing."""

    a = (382.0, 118.0)
    b = (116.0, 430.0)
    c = (654.0, 430.0)
    a, b, c = _offset_points(ctx, (a, b, c))
    outer = (a, b, c)
    if problem.scene_variant == "triangle_median_half":
        d = midpoint(b, c)
        shaded = (a, b, d)
        ctx.draw.polygon(outer, fill=ctx.fill_color)
        ctx.draw.polygon(shaded, fill=ctx.shaded_color)
        ctx.draw.line([a, b, c, a], fill=ctx.line_color, width=ctx.line_width)
        ctx.draw.line([a, d], fill=ctx.accent_color, width=ctx.line_width)
        ticks_bbox = _draw_equal_ticks(ctx, ((b, d), (d, c)))
        median_bbox = bbox_from_points((a, d), width=ctx.width, height=ctx.height, pad=10.0)
        partition_bbox = (
            min(ticks_bbox[0], median_bbox[0]),
            min(ticks_bbox[1], median_bbox[1]),
            max(ticks_bbox[2], median_bbox[2]),
            max(ticks_bbox[3], median_bbox[3]),
        )
    elif problem.scene_variant == "triangle_midsegment_quarter":
        e = midpoint(c, a)
        f = midpoint(a, b)
        shaded = (a, f, e)
        ctx.draw.polygon(outer, fill=ctx.fill_color)
        ctx.draw.polygon(shaded, fill=ctx.shaded_color)
        ctx.draw.line([a, b, c, a], fill=ctx.line_color, width=ctx.line_width)
        ctx.draw.line([f, e], fill=ctx.accent_color, width=ctx.line_width)
        ticks_bbox = _draw_equal_ticks(ctx, ((a, f), (f, b), (a, e), (e, c)))
        midsegment_bbox = bbox_from_points((f, e), width=ctx.width, height=ctx.height, pad=10.0)
        partition_bbox = (
            min(ticks_bbox[0], midsegment_bbox[0]),
            min(ticks_bbox[1], midsegment_bbox[1]),
            max(ticks_bbox[2], midsegment_bbox[2]),
            max(ticks_bbox[3], midsegment_bbox[3]),
        )
    else:
        d = midpoint(b, c)
        e = midpoint(c, a)
        f = midpoint(a, b)
        g = centroid(a, b, c)
        shaded = (a, f, g)
        ctx.draw.polygon(outer, fill=ctx.fill_color)
        ctx.draw.polygon(shaded, fill=ctx.shaded_color)
        ctx.draw.line([a, b, c, a], fill=ctx.line_color, width=ctx.line_width)
        for start, end in ((a, d), (b, e), (c, f)):
            ctx.draw.line([start, end], fill=ctx.accent_color, width=ctx.line_width)
        ctx.draw.ellipse((g[0] - 5.0, g[1] - 5.0, g[0] + 5.0, g[1] + 5.0), fill=ctx.accent_color)
        partition_bbox = bbox_from_points(
            (a, b, c, d, e, f, g),
            width=ctx.width,
            height=ctx.height,
            pad=10.0,
        )
    outer_bbox = bbox_from_points(outer, width=ctx.width, height=ctx.height, pad=8.0)
    shaded_bbox = bbox_from_points(shaded, width=ctx.width, height=ctx.height, pad=8.0)
    return outer, shaded, outer_bbox, shaded_bbox, partition_bbox


def create_area_partition_render_context(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
) -> tuple[AreaPartitionRenderContext, Dict[str, Any]]:
    """Create scene-level drawing context and render metadata."""

    rng = spawn_rng(int(instance_seed), "geometry.area_partition.render")
    width = int(params.get("canvas_width", group_default(render_defaults, "canvas_width", 760)))
    height = int(params.get("canvas_height", group_default(render_defaults, "canvas_height", 560)))
    image, background_meta, diagram_style, diagram_style_resolution = prepare_geometry_diagram_style_and_background(
        canvas_width=int(width),
        canvas_height=int(height),
        instance_seed=int(instance_seed),
        scene_id=SCENE_ID,
        params=params,
        allow_dark=True,
        style_profile=GEOMETRY_STYLE_PROFILE_ANALYTICAL_DIAGRAM,
    )
    shape_style = geometry_shape_style_from_diagram_style(diagram_style)
    font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace="geometry.area_partition.font_family",
        params=params,
    )
    font_record = get_font_family_record(str(font_family))
    palettes: Tuple[Tuple[Color, Color, Color, Color], ...] = (
        (
            tuple(int(value) for value in diagram_style.fill_rgb),
            tuple(int(value) for value in diagram_style.accent_rgb),
            tuple(int(value) for value in diagram_style.secondary_accent_rgb),
            tuple(int(value) for value in diagram_style.secondary_stroke_rgb),
        ),
        (
            tuple(int(value) for value in diagram_style.muted_fill_rgb),
            tuple(int(value) for value in diagram_style.highlight_rgb),
            tuple(int(value) for value in diagram_style.accent_rgb),
            tuple(int(value) for value in diagram_style.guide_rgb),
        ),
        (
            tuple(int(value) for value in diagram_style.option_fill_rgb),
            tuple(int(value) for value in diagram_style.secondary_accent_rgb),
            tuple(int(value) for value in diagram_style.accent_rgb),
            tuple(int(value) for value in diagram_style.secondary_stroke_rgb),
        ),
        (
            tuple(int(value) for value in diagram_style.panel_alt_fill_rgb),
            tuple(int(value) for value in diagram_style.accent_rgb),
            tuple(int(value) for value in diagram_style.highlight_rgb),
            tuple(int(value) for value in diagram_style.guide_rgb),
        ),
    )
    palette_rng = spawn_rng(int(instance_seed), "geometry.area_partition.palette")
    fill_color, shaded_color, accent_color, muted_color = uniform_choice(
        palette_rng,
        palettes,
    )
    font_size = int(params.get("label_font_size", group_default(render_defaults, "label_font_size", 22)))
    small_font_size = int(
        params.get(
            "small_label_font_size",
            group_default(render_defaults, "small_label_font_size", 18),
        )
    )
    line_width = int(params.get("line_width", group_default(render_defaults, "line_width", 4)))
    readout_label_color = _readout_ink_for_background(
        (
            tuple(int(value) for value in diagram_style.canvas_rgb),
            tuple(int(value) for value in diagram_style.paper_rgb),
            tuple(int(value) for value in diagram_style.panel_fill_rgb),
            tuple(int(value) for value in diagram_style.panel_alt_fill_rgb),
        )
    )
    ctx = AreaPartitionRenderContext(
        rng=rng,
        image=image,
        draw=ImageDraw.Draw(image),
        width=int(width),
        height=int(height),
        line_color=shape_style.line_color,
        label_color=readout_label_color,
        label_stroke_color=readout_label_color,
        fill_color=fill_color,
        shaded_color=shaded_color,
        accent_color=accent_color,
        muted_color=muted_color,
        line_width=max(2, int(line_width)),
        font=load_font(max(12, int(font_size)), bold=True, font_family=font_family),
        small_font=load_font(max(10, int(small_font_size)), bold=True, font_family=font_family),
        layout_offset=(float(rng.randint(-26, 24)), float(rng.randint(-14, 16))),
        font_family=str(font_family),
        scene_transform=LazySceneTransform(
            rng,
            params=params,
            render_defaults=render_defaults,
            canvas_width=int(width),
            canvas_height=int(height),
        ),
    )
    render_meta = {
        "background_style": dict(background_meta),
        "technical_diagram_style": geometry_diagram_style_metadata(diagram_style),
        "technical_diagram_style_resolution": dict(diagram_style_resolution),
        "shape_style": shape_style.to_trace_dict(),
        "line_width": int(ctx.line_width),
        "label_font_size": int(font_size),
        "small_label_font_size": int(small_font_size),
        "fill_color": list(fill_color),
        "shaded_color": list(shaded_color),
        "accent_color": list(accent_color),
        "muted_color": list(muted_color),
        "readout_label_color": list(readout_label_color),
        "font_family": font_record.to_trace(),
        "font_asset_version": font_asset_version(),
        "layout_jitter": {
            "offset_px": [round(float(ctx.layout_offset[0]), 3), round(float(ctx.layout_offset[1]), 3)],
            "offset_range_px": [-26, 24, -14, 16],
            "applied_before_annotation_projection": True,
        },
    }
    return ctx, render_meta


def render_area_partition_scene(
    ctx: AreaPartitionRenderContext,
    problem: AreaPartitionProblem,
) -> RenderedAreaPartitionScene:
    """Render the selected partition problem without choosing the answer."""

    if problem.scene_variant.startswith("parallelogram"):
        outer_points, shaded_points, outer_bbox, shaded_bbox, partition_bbox = _draw_parallelogram(ctx, problem)
        shape_type = "parallelogram"
    else:
        outer_points, shaded_points, outer_bbox, shaded_bbox, partition_bbox = _draw_triangle(ctx, problem)
        shape_type = "triangle"

    label_bboxes: Dict[str, BBox] = {}
    shaded_area_text = f"shaded area = {problem.shaded_area}"
    target_text = "total area = ?"
    shaded_center, target_center = _external_readout_centers(
        ctx,
        outer_bbox,
        (shaded_area_text, target_text),
    )
    label_bboxes["given_area"] = draw_readout_centered(
        ctx,
        shaded_area_text,
        shaded_center,
        small=True,
        required=True,
        backed=False,
    )
    label_bboxes["target"] = draw_readout_centered(
        ctx,
        target_text,
        target_center,
        small=True,
        required=True,
        backed=False,
    )
    scene_entities = (
        {
            "entity_id": OUTER_SHAPE_ID,
            "entity_type": shape_type,
            "bbox": bbox_to_list(outer_bbox),
            "points": [[round(x, 3), round(y, 3)] for x, y in outer_points],
        },
        {
            "entity_id": SHADED_REGION_ID,
            "entity_type": "area_partition_region",
            "bbox": bbox_to_list(shaded_bbox),
            "points": [[round(x, 3), round(y, 3)] for x, y in shaded_points],
        },
    )
    witness = AreaPartitionWitness(
        scene_variant=str(problem.scene_variant),
        shape_type=shape_type,
        shaded_area=int(problem.shaded_area),
        shaded_fraction_numerator=1,
        shaded_fraction_denominator=int(problem.denominator),
        formula=str(problem.formula),
        answer_value=float(problem.answer),
    )
    return RenderedAreaPartitionScene(
        image=ctx.image,
        answer=float(problem.answer),
        annotation_bboxes=(outer_bbox, shaded_bbox),
        annotation_keyed_bboxes={
            OUTER_SHAPE_ID: outer_bbox,
            SHADED_REGION_ID: shaded_bbox,
        },
        annotation_roles=ANNOTATION_ROLES,
        label_bboxes=dict(label_bboxes),
        scene_entities=scene_entities,
        render_map={
            OUTER_SHAPE_ID: {
                "type": shape_type,
                "points": [[round(x, 3), round(y, 3)] for x, y in outer_points],
                "bbox": bbox_to_list(outer_bbox),
            },
            SHADED_REGION_ID: {
                "points": [[round(x, 3), round(y, 3)] for x, y in shaded_points],
                "bbox": bbox_to_list(shaded_bbox),
            },
            "partition_bbox": bbox_to_list(partition_bbox),
            "label_bboxes": {
                key: bbox_to_list(value) for key, value in label_bboxes.items()
            },
            "coord_space": "pixel",
        },
        witness=witness,
    )


__all__ = [
    "AreaPartitionRenderContext",
    "RenderedAreaPartitionScene",
    "create_area_partition_render_context",
    "render_area_partition_scene",
]
