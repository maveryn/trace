"""Rendering primitives for trapezoid-extension diagrams."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from PIL import ImageDraw

from trace_tasks.core.sampling import uniform_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.geometry.shared.diagram_style import (
    geometry_diagram_style_metadata,
    prepare_geometry_diagram_style_and_background,
)
from trace_tasks.tasks.geometry.shared.measurement_rendering import (
    bbox_from_points,
    bbox_to_list,
    draw_label,
    fmt_measure,
    pad_bbox,
)
from trace_tasks.tasks.geometry.shared.scene_transform import LazySceneTransform
from trace_tasks.tasks.geometry.shared.shape_style import (
    extract_background_anchor_colors,
    sample_geometry_shape_style,
)
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.text_rendering import load_font

from .state import SCENE_ID
from .measurements import case_trace_values
from .state import (
    BBox,
    Color,
    Point,
    RenderContext,
    RenderedTrapezoidExtensionScene,
    TrapezoidExtensionProblem,
)


_LAYOUT_CENTER: Point = (390.0, 280.0)
_LAYOUT_SCALE = 0.88
_LAYOUT_SHIFT: Point = (22.0, 0.0)
_HEIGHT_LABEL_OFFSET = 40.0
_HEIGHT_LABEL_HALF_WIDTH_GUARD = 44.0
_HEIGHT_LABEL_HALF_HEIGHT_GUARD = 18.0
_SIDE_LABEL_OFFSET = 58.0
_SIDE_LABEL_HALF_WIDTH_GUARD = 48.0
_SIDE_LABEL_HALF_HEIGHT_GUARD = 20.0


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


def _draw_dashed_line(
    ctx: RenderContext,
    start: Point,
    end: Point,
    *,
    fill: Color | None = None,
    width: int | None = None,
    dash: float = 14.0,
    gap: float = 8.0,
) -> None:
    color = fill if fill is not None else ctx.muted_color
    line_width = int(width if width is not None else ctx.line_width)
    dx = float(end[0]) - float(start[0])
    dy = float(end[1]) - float(start[1])
    length = math.hypot(dx, dy)
    if length <= 1e-9:
        return
    ux = dx / length
    uy = dy / length
    distance = 0.0
    while distance < length:
        next_distance = min(length, distance + dash)
        p0 = (float(start[0]) + ux * distance, float(start[1]) + uy * distance)
        p1 = (float(start[0]) + ux * next_distance, float(start[1]) + uy * next_distance)
        ctx.draw.line([p0, p1], fill=color, width=line_width)
        distance += dash + gap


def _draw_height_marker(ctx: RenderContext, top: Point, bottom: Point, label: str) -> BBox:
    tick = 11.0 * float(ctx.scene_transform.transform.scale)
    dx = float(bottom[0]) - float(top[0])
    dy = float(bottom[1]) - float(top[1])
    length = max(1e-9, math.hypot(dx, dy))
    nx = -dy / length
    ny = dx / length
    ctx.draw.line([top, bottom], fill=ctx.accent_color, width=max(2, ctx.line_width - 1))
    ctx.draw.line(
        [(top[0] - nx * tick, top[1] - ny * tick), (top[0] + nx * tick, top[1] + ny * tick)],
        fill=ctx.accent_color,
        width=max(2, ctx.line_width - 1),
    )
    ctx.draw.line(
        [
            (bottom[0] - nx * tick, bottom[1] - ny * tick),
            (bottom[0] + nx * tick, bottom[1] + ny * tick),
        ],
        fill=ctx.accent_color,
        width=max(2, ctx.line_width - 1),
    )
    midpoint = ((float(top[0]) + float(bottom[0])) / 2.0, (float(top[1]) + float(bottom[1])) / 2.0)
    label_offset = float(_HEIGHT_LABEL_OFFSET) * float(ctx.scene_transform.transform.scale)
    left_candidate = (midpoint[0] + nx * label_offset, midpoint[1] + ny * label_offset)
    right_candidate = (midpoint[0] - nx * label_offset, midpoint[1] - ny * label_offset)
    label_center = left_candidate if float(left_candidate[0]) <= float(right_candidate[0]) else right_candidate
    label_bbox = draw_label(ctx, label, label_center, small=True)
    marker_bbox = bbox_from_points(
        (
            (top[0] - nx * tick, top[1] - ny * tick),
            (top[0] + nx * tick, top[1] + ny * tick),
            (bottom[0] - nx * tick, bottom[1] - ny * tick),
            (bottom[0] + nx * tick, bottom[1] + ny * tick),
        ),
        width=ctx.width,
        height=ctx.height,
        pad=5.0,
    )
    return _union_bboxes((marker_bbox, label_bbox), width=ctx.width, height=ctx.height)


def _compact_layout_point(point: Point) -> Point:
    """Shrink the canonical construction to leave room for external labels."""

    return (
        float(_LAYOUT_CENTER[0])
        + ((float(point[0]) - float(_LAYOUT_CENTER[0])) * float(_LAYOUT_SCALE))
        + float(_LAYOUT_SHIFT[0]),
        float(_LAYOUT_CENTER[1])
        + ((float(point[1]) - float(_LAYOUT_CENTER[1])) * float(_LAYOUT_SCALE))
        + float(_LAYOUT_SHIFT[1]),
    )


def _compact_layout_points(points: Mapping[str, Point]) -> dict[str, Point]:
    return {str(key): _compact_layout_point(point) for key, point in points.items()}


def _height_label_fit_points(top: Point, bottom: Point) -> tuple[Point, ...]:
    """Return invisible fit witnesses for the externally placed height label."""

    dx = float(bottom[0]) - float(top[0])
    dy = float(bottom[1]) - float(top[1])
    length = max(1e-9, math.hypot(dx, dy))
    tx = dx / length
    ty = dy / length
    nx = -dy / length
    ny = dx / length
    midpoint = ((float(top[0]) + float(bottom[0])) / 2.0, (float(top[1]) + float(bottom[1])) / 2.0)
    fit_points: list[Point] = []
    for side in (-1.0, 1.0):
        label_center = (
            midpoint[0] + (nx * float(_HEIGHT_LABEL_OFFSET) * side),
            midpoint[1] + (ny * float(_HEIGHT_LABEL_OFFSET) * side),
        )
        for normal_sign in (-1.0, 1.0):
            for tangent_sign in (-1.0, 1.0):
                fit_points.append(
                    (
                        label_center[0]
                        + (nx * float(_HEIGHT_LABEL_HALF_WIDTH_GUARD) * normal_sign)
                        + (tx * float(_HEIGHT_LABEL_HALF_HEIGHT_GUARD) * tangent_sign),
                        label_center[1]
                        + (ny * float(_HEIGHT_LABEL_HALF_WIDTH_GUARD) * normal_sign)
                        + (ty * float(_HEIGHT_LABEL_HALF_HEIGHT_GUARD) * tangent_sign),
                    )
                )
    return tuple(fit_points)


def _side_label_point(start: Point, end: Point) -> Point:
    """Place the AD side label beside the side edge instead of on top of it."""

    dx = float(end[0]) - float(start[0])
    dy = float(end[1]) - float(start[1])
    length = max(1e-9, math.hypot(dx, dy))
    nx = -dy / length
    ny = dx / length
    midpoint = ((float(start[0]) + float(end[0])) / 2.0, (float(start[1]) + float(end[1])) / 2.0)
    candidate_a = (
        midpoint[0] + (nx * float(_SIDE_LABEL_OFFSET)),
        midpoint[1] + (ny * float(_SIDE_LABEL_OFFSET)),
    )
    candidate_b = (
        midpoint[0] - (nx * float(_SIDE_LABEL_OFFSET)),
        midpoint[1] - (ny * float(_SIDE_LABEL_OFFSET)),
    )
    return candidate_a if float(candidate_a[0]) >= float(candidate_b[0]) else candidate_b


def _axis_aligned_label_fit_points(center: Point, *, half_width: float, half_height: float) -> tuple[Point, ...]:
    cx, cy = float(center[0]), float(center[1])
    return (
        (cx - float(half_width), cy - float(half_height)),
        (cx + float(half_width), cy - float(half_height)),
        (cx - float(half_width), cy + float(half_height)),
        (cx + float(half_width), cy + float(half_height)),
    )


def _draw_vertex_markers(ctx: RenderContext, vertices: Mapping[str, Point]) -> dict[str, BBox]:
    """Draw visible vertex dots and letter labels for named construction points."""

    label_offsets: dict[str, Point] = {
        "A": (-22.0, -25.0),
        "B": (0.0, -27.0),
        "C": (24.0, 23.0),
        "D": (-24.0, 23.0),
        "E": (23.0, -25.0),
    }
    radius = max(4.0, 5.0 * float(ctx.scene_transform.transform.scale))
    bboxes: dict[str, BBox] = {}
    for label, point in vertices.items():
        px, py = float(point[0]), float(point[1])
        dot_bbox = (px - radius, py - radius, px + radius, py + radius)
        ctx.draw.ellipse(dot_bbox, fill=(255, 255, 255), outline=ctx.line_color, width=max(2, ctx.line_width - 1))
        offset = label_offsets[str(label)]
        text_bbox = draw_label(ctx, str(label), (px + float(offset[0]), py + float(offset[1])), small=True)
        bboxes[f"point_{label}"] = _union_bboxes((dot_bbox, text_bbox), width=ctx.width, height=ctx.height, pad=2.0)
    return bboxes


def create_render_context(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    namespace: str,
) -> tuple[RenderContext, dict[str, Any]]:
    """Create a styled PIL render context for one trapezoid-extension sample."""

    rng = spawn_rng(int(instance_seed), f"{namespace}.render")
    width = int(params.get("canvas_width", group_default(render_defaults, "canvas_width", 780)))
    height = int(params.get("canvas_height", group_default(render_defaults, "canvas_height", 560)))
    image, background_meta, diagram_style, diagram_style_meta = prepare_geometry_diagram_style_and_background(
        canvas_width=int(width),
        canvas_height=int(height),
        scene_id=SCENE_ID,
        instance_seed=int(instance_seed),
        params=params,
        namespace_suffix="trapezoid_extension_background",
    )
    shape_style = sample_geometry_shape_style(
        rng,
        params=params,
        render_defaults=render_defaults,
        anchor_colors=extract_background_anchor_colors(background_meta),
    )
    palettes: tuple[tuple[Color, Color, Color, Color], ...] = (
        ((229, 240, 255), (255, 226, 199), (26, 123, 185), (126, 143, 156)),
        ((236, 247, 232), (255, 224, 210), (38, 143, 104), (150, 142, 132)),
        ((248, 238, 252), (226, 242, 255), (123, 95, 190), (143, 154, 172)),
        ((255, 244, 224), (226, 240, 255), (196, 102, 44), (132, 146, 160)),
    )
    palette_rng = spawn_rng(int(instance_seed), f"{namespace}.palette")
    fill_color, extension_fill_color, accent_color, muted_color = uniform_choice(
        palette_rng,
        palettes,
    )
    font_size = int(params.get("label_font_size", group_default(render_defaults, "label_font_size", 22)))
    small_font_size = int(params.get("small_label_font_size", group_default(render_defaults, "small_label_font_size", 18)))
    line_width = int(params.get("line_width", group_default(render_defaults, "line_width", 4)))
    label_stroke_width = int(params.get("label_stroke_width", group_default(render_defaults, "label_stroke_width", 0)))
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
        extension_fill_color=extension_fill_color,
        accent_color=accent_color,
        muted_color=muted_color,
        line_width=max(2, int(line_width)),
        label_stroke_width=max(0, int(label_stroke_width)),
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
        "extension_fill_color": list(extension_fill_color),
        "accent_color": list(accent_color),
        "muted_color": list(muted_color),
    }


def render_trapezoid_extension_scene(
    ctx: RenderContext,
    problem: TrapezoidExtensionProblem,
) -> RenderedTrapezoidExtensionScene:
    """Draw one resolved trapezoid-completion construction and projected witnesses."""

    case = problem.case
    raw_vertices = {
        "A": (135.0, 170.0),
        "B": (355.0, 170.0),
        "E": (685.0, 170.0),
        "D": (95.0, 420.0),
        "C": (645.0, 420.0),
    }
    raw_label_points = {
        "top_base": (245.0, 132.0),
        "height_top": (56.0, 170.0),
        "height_bottom": (56.0, 420.0),
        "parallelogram_area": (520.0, 84.0),
        "parallelogram_perimeter": (520.0, 84.0),
        "extension": (520.0, 132.0),
        "bottom_base": (370.0, 458.0),
        "target_extension": (520.0, 202.0),
        "target_area": (520.0, 505.0),
    }
    layout_vertices = _compact_layout_points(raw_vertices)
    raw_label_points = _compact_layout_points(raw_label_points)
    raw_label_points["side"] = _side_label_point(layout_vertices["A"], layout_vertices["D"])
    height_fit_points = _height_label_fit_points(raw_label_points["height_top"], raw_label_points["height_bottom"])
    side_fit_points = _axis_aligned_label_fit_points(
        raw_label_points["side"],
        half_width=_SIDE_LABEL_HALF_WIDTH_GUARD,
        half_height=_SIDE_LABEL_HALF_HEIGHT_GUARD,
    )
    ctx.scene_transform.resolve(
        (*layout_vertices.values(), *raw_label_points.values(), *height_fit_points, *side_fit_points)
    )
    vertex_points = ctx.scene_transform.keyed_points(layout_vertices)
    a = vertex_points["A"]
    b = vertex_points["B"]
    e = vertex_points["E"]
    d = vertex_points["D"]
    c = vertex_points["C"]
    label_points = ctx.scene_transform.keyed_points(raw_label_points)

    trapezoid_points = (a, b, c, d)
    extension_points = (b, e, c)
    parallelogram_points = (a, e, c, d)
    ctx.draw.polygon(trapezoid_points, fill=ctx.fill_color)
    ctx.draw.polygon(extension_points, fill=ctx.extension_fill_color)
    ctx.draw.line([a, b, c, d, a], fill=ctx.line_color, width=ctx.line_width)
    _draw_dashed_line(ctx, b, e, fill=ctx.muted_color, width=ctx.line_width)
    _draw_dashed_line(ctx, e, c, fill=ctx.muted_color, width=ctx.line_width)
    ctx.draw.line([d, c], fill=ctx.line_color, width=ctx.line_width)

    label_bboxes: dict[str, BBox] = {}
    label_bboxes.update(_draw_vertex_markers(ctx, {"A": a, "B": b, "C": c, "D": d, "E": e}))
    label_bboxes["top_base"] = draw_label(ctx, f"AB={case.top_base}", label_points["top_base"], small=True)
    label_bboxes["height"] = _draw_height_marker(
        ctx,
        label_points["height_top"],
        label_points["height_bottom"],
        f"h={case.height}",
    )
    for label in problem.support_labels:
        label_bboxes[str(label.role)] = draw_label(
            ctx,
            str(label.text),
            label_points[str(label.position_key)],
            small=True,
        )
    label_bboxes["target"] = draw_label(
        ctx,
        str(problem.target_text),
        label_points[str(problem.target_position_key)],
        small=True,
    )

    support_bboxes = [label_bboxes["top_base"]]
    if bool(problem.include_height_in_support):
        support_bboxes.append(label_bboxes["height"])
    support_bboxes.extend(label_bboxes[str(label.role)] for label in problem.support_labels)
    original_bbox = bbox_from_points(trapezoid_points, width=ctx.width, height=ctx.height, pad=10.0)
    dashed_completion_bbox = bbox_from_points(extension_points, width=ctx.width, height=ctx.height, pad=10.0)
    completed_bbox = bbox_from_points(parallelogram_points, width=ctx.width, height=ctx.height, pad=10.0)
    supporting_bbox = _union_bboxes(tuple(support_bboxes), width=ctx.width, height=ctx.height, pad=4.0)

    if str(problem.annotation_mode) == "extension_segment":
        annotation_bboxes = {"BE": bbox_from_points((b, e), width=ctx.width, height=ctx.height, pad=8.0)}
        annotation_roles = ("BE",)
        annotation_segment = (b, e)
        annotation_bbox = None
    else:
        annotation_bboxes = {"original_trapezoid": original_bbox}
        annotation_roles = ("original_trapezoid",)
        annotation_segment = None
        annotation_bbox = original_bbox
    scene_entities = (
        {
            "entity_id": "original_trapezoid",
            "entity_type": "trapezoid",
            "bbox": bbox_to_list(original_bbox),
            "points": [[round(x, 3), round(y, 3)] for x, y in trapezoid_points],
        },
        {
            "entity_id": "completion_triangle",
            "entity_type": "dashed_extension_region",
            "bbox": bbox_to_list(dashed_completion_bbox),
            "points": [[round(x, 3), round(y, 3)] for x, y in extension_points],
        },
        {
            "entity_id": "completed_parallelogram",
            "entity_type": "parallelogram",
            "bbox": bbox_to_list(completed_bbox),
            "points": [[round(x, 3), round(y, 3)] for x, y in parallelogram_points],
        },
    )
    witness = {
        "formula_family": str(problem.formula_family),
        "formula": str(problem.formula_text),
        "answer_value": float(problem.answer),
        "target_support_probabilities": dict(problem.target_support_probabilities),
        **case_trace_values(case),
    }
    return RenderedTrapezoidExtensionScene(
        image=ctx.image,
        answer=float(problem.answer),
        annotation_bboxes=dict(annotation_bboxes),
        annotation_roles=annotation_roles,
        label_bboxes=dict(label_bboxes),
        scene_entities=scene_entities,
        render_map={
            "coord_space": "pixel",
            "original_trapezoid": {
                "points": [[round(x, 3), round(y, 3)] for x, y in trapezoid_points],
                "bbox": bbox_to_list(original_bbox),
            },
            "completion_triangle": {
                "points": [[round(x, 3), round(y, 3)] for x, y in extension_points],
                "bbox": bbox_to_list(dashed_completion_bbox),
            },
            "label_bboxes": {key: bbox_to_list(value) for key, value in label_bboxes.items()},
            "supporting_visible_labels_bbox": bbox_to_list(supporting_bbox),
            "target_cue_bbox": bbox_to_list(label_bboxes["target"]),
            "dashed_parallelogram_completion_bbox": bbox_to_list(dashed_completion_bbox),
            "completed_parallelogram_bbox": bbox_to_list(completed_bbox),
            "vertex_points": {
                key: [round(float(point[0]), 3), round(float(point[1]), 3)]
                for key, point in {"A": a, "B": b, "C": c, "D": d, "E": e}.items()
            },
        },
        witness=witness,
        annotation_mode=str(problem.annotation_mode),
        annotation_segment=annotation_segment,
        annotation_bbox=annotation_bbox,
    )


__all__ = ["create_render_context", "fmt_measure", "render_trapezoid_extension_scene"]
