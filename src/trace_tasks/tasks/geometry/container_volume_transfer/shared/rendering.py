"""Rendering primitives for container volume-transfer diagrams."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from PIL import ImageDraw

from trace_tasks.core.sampling import uniform_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.geometry.shared.diagram_style import prepare_geometry_diagram_style_and_background
from trace_tasks.tasks.geometry.shared.measurement_rendering import bbox_to_list, bbox_union_from_bboxes as _bbox_union, pad_bbox
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.text_legibility import draw_text_traced
from trace_tasks.tasks.shared.text_rendering import load_font

from .measurements import fmt_number
from .state import BBox, Color, Point, RenderContext, RenderedScene, ResolvedProblem

PANEL_FILL_COLOR: Color = (252, 253, 250)
PANEL_LABEL_COLOR: Color = (10, 14, 22)
PANEL_LINE_COLOR: Color = (31, 52, 75)
PANEL_SECONDARY_COLOR: Color = (91, 111, 132)
PANEL_BORDER_COLOR: Color = (94, 111, 134)


def draw_text_centered(ctx: RenderContext, text: str, center: Point, *, small: bool = False) -> BBox:
    font = ctx.small_font if bool(small) else ctx.font
    bbox = ctx.draw.textbbox((0, 0), str(text), font=font, stroke_width=ctx.label_stroke_width)
    text_w = float(bbox[2] - bbox[0])
    text_h = float(bbox[3] - bbox[1])
    left = float(center[0]) - text_w / 2.0
    top = float(center[1]) - text_h / 2.0
    draw_text_traced(
        ctx.draw,
        (left, top),
        str(text),
        font=font,
        fill=ctx.label_color,
        stroke_width=ctx.label_stroke_width,
        stroke_fill=ctx.label_stroke_color,
        role="readout",
        required=False,
    )
    return pad_bbox((left, top, left + text_w, top + text_h), 4.0, width=ctx.width, height=ctx.height)


def draw_value_box(ctx: RenderContext, text: str, center: Point) -> BBox:
    font = ctx.font
    bbox = ctx.draw.textbbox((0, 0), str(text), font=font, stroke_width=ctx.label_stroke_width)
    text_w = float(bbox[2] - bbox[0])
    text_h = float(bbox[3] - bbox[1])
    left = float(center[0]) - text_w / 2.0 - 14.0
    top = float(center[1]) - text_h / 2.0 - 8.0
    right = left + text_w + 28.0
    bottom = top + text_h + 16.0
    ctx.draw.rounded_rectangle(
        (left, top, right, bottom),
        radius=7,
        fill=ctx.panel_fill_color,
        outline=ctx.muted_color,
        width=max(1, ctx.line_width - 1),
    )
    draw_text_traced(
        ctx.draw,
        (left + 14.0, top + 8.0),
        str(text),
        font=font,
        fill=ctx.label_color,
        stroke_width=ctx.label_stroke_width,
        stroke_fill=ctx.label_stroke_color,
        role="readout",
        required=False,
    )
    return pad_bbox((left, top, right, bottom), 2.0, width=ctx.width, height=ctx.height)


def draw_arrow(ctx: RenderContext, start: Point, end: Point) -> BBox:
    ctx.draw.line([start, end], fill=ctx.accent_color, width=max(3, ctx.line_width + 1))
    dx = float(end[0]) - float(start[0])
    dy = float(end[1]) - float(start[1])
    length = max(1e-6, (dx * dx + dy * dy) ** 0.5)
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    head = 18.0
    wing = 8.0
    p1 = (float(end[0]) - ux * head + px * wing, float(end[1]) - uy * head + py * wing)
    p2 = (float(end[0]) - ux * head - px * wing, float(end[1]) - uy * head - py * wing)
    ctx.draw.polygon([end, p1, p2], fill=ctx.accent_color)
    return _bbox_union(
        (
            (float(start[0]), float(start[1]), float(start[0]), float(start[1])),
            (float(end[0]), float(end[1]), float(end[0]), float(end[1])),
            (float(p1[0]), float(p1[1]), float(p1[0]), float(p1[1])),
            (float(p2[0]), float(p2[1]), float(p2[0]), float(p2[1])),
        ),
        width=ctx.width,
        height=ctx.height,
        pad=8.0,
    )


def draw_cylinder(
    ctx: RenderContext,
    bbox: BBox,
    *,
    fill: Color,
    liquid: bool = True,
    liquid_fraction: float = 0.72,
    mark_fill_line: bool = False,
) -> BBox | None:
    x0, y0, x1, y1 = [float(value) for value in bbox]
    ellipse_h = min(42.0, max(24.0, (y1 - y0) * 0.18))
    body_top = y0 + ellipse_h / 2.0
    body_bottom = y1 - ellipse_h / 2.0
    ctx.draw.rectangle((x0, body_top, x1, body_bottom), fill=fill, outline=ctx.line_color, width=ctx.line_width)
    fill_mark_bbox: BBox | None = None
    if liquid:
        fraction = min(0.95, max(0.08, float(liquid_fraction)))
        fill_y = body_bottom - (body_bottom - body_top) * fraction
        ctx.draw.rectangle((x0 + 4.0, fill_y, x1 - 4.0, body_bottom - 2.0), fill=ctx.liquid_fill)
        if mark_fill_line:
            ctx.draw.line((x0 + 2.0, fill_y, x1 - 2.0, fill_y), fill=ctx.accent_color, width=max(3, ctx.line_width + 1))
            fill_mark_bbox = pad_bbox((x0 + 2.0, fill_y - 1.0, x1 - 2.0, fill_y + 1.0), 7.0, width=ctx.width, height=ctx.height)
    ctx.draw.ellipse((x0, y0, x1, y0 + ellipse_h), fill=fill, outline=ctx.line_color, width=ctx.line_width)
    if liquid:
        top_liquid_ellipse = (x0 + 4.0, fill_y - ellipse_h / 2.0, x1 - 4.0, fill_y + ellipse_h / 2.0)
        ctx.draw.arc(top_liquid_ellipse, start=0, end=180, fill=ctx.accent_color if mark_fill_line else ctx.secondary_color, width=max(1, ctx.line_width - 1))
    ctx.draw.arc((x0, y1 - ellipse_h, x1, y1), start=0, end=180, fill=ctx.line_color, width=ctx.line_width)
    ctx.draw.arc((x0, y1 - ellipse_h, x1, y1), start=180, end=360, fill=ctx.secondary_color, width=max(1, ctx.line_width - 1))
    ctx.draw.line((x0, body_top, x0, body_bottom), fill=ctx.line_color, width=ctx.line_width)
    ctx.draw.line((x1, body_top, x1, body_bottom), fill=ctx.line_color, width=ctx.line_width)
    return fill_mark_bbox


def draw_cone(ctx: RenderContext, bbox: BBox) -> None:
    x0, y0, x1, y1 = [float(value) for value in bbox]
    apex = ((x0 + x1) / 2.0, y0)
    base_top = y1 - min(38.0, max(24.0, (y1 - y0) * 0.16))
    ctx.draw.polygon([apex, (x0, base_top), (x1, base_top)], fill=ctx.source_fill, outline=ctx.line_color)
    ctx.draw.line([apex, (x0, base_top)], fill=ctx.line_color, width=ctx.line_width)
    ctx.draw.line([apex, (x1, base_top)], fill=ctx.line_color, width=ctx.line_width)
    ctx.draw.ellipse((x0, base_top - 14.0, x1, y1), fill=ctx.source_fill, outline=ctx.line_color, width=ctx.line_width)
    ctx.draw.arc((x0, base_top - 14.0, x1, y1), start=180, end=360, fill=ctx.secondary_color, width=max(1, ctx.line_width - 1))


def draw_cuboid_tank(
    ctx: RenderContext,
    bbox: BBox,
    *,
    liquid_fraction: float = 0.72,
    mark_fill_line: bool = False,
) -> tuple[BBox, BBox | None]:
    x0, y0, x1, y1 = [float(value) for value in bbox]
    depth = min(48.0, max(28.0, (x1 - x0) * 0.18))
    top_shift = (depth, -depth * 0.55)
    front = (x0, y0 + depth * 0.45, x1 - depth, y1)
    back = (front[0] + top_shift[0], front[1] + top_shift[1], front[2] + top_shift[0], front[3] + top_shift[1])
    ctx.draw.polygon(
        [(back[0], back[1]), (back[2], back[1]), (front[2], front[1]), (front[0], front[1])],
        fill=ctx.target_fill,
        outline=ctx.line_color,
    )
    ctx.draw.polygon(
        [(front[2], front[1]), (back[2], back[1]), (back[2], back[3]), (front[2], front[3])],
        fill=ctx.target_fill,
        outline=ctx.line_color,
    )
    ctx.draw.rectangle(front, fill=ctx.target_fill, outline=ctx.line_color, width=ctx.line_width)
    fraction = min(0.95, max(0.08, float(liquid_fraction)))
    fill_y = front[3] - (front[3] - front[1]) * fraction
    ctx.draw.rectangle((front[0] + 8.0, fill_y, front[2] - 8.0, front[3] - 6.0), fill=ctx.liquid_fill, outline=None)
    fill_mark_bbox: BBox | None = None
    if mark_fill_line:
        ctx.draw.line((front[0] + 6.0, fill_y, front[2] - 6.0, fill_y), fill=ctx.accent_color, width=max(3, ctx.line_width + 1))
        fill_mark_bbox = pad_bbox((front[0] + 6.0, fill_y - 1.0, front[2] - 6.0, fill_y + 1.0), 7.0, width=ctx.width, height=ctx.height)
    ctx.draw.rectangle(front, outline=ctx.line_color, width=ctx.line_width)
    return pad_bbox((x0, y0, x1, y1), 4.0, width=ctx.width, height=ctx.height), fill_mark_bbox


def create_render_context(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    random_namespace: str,
) -> tuple[RenderContext, Dict[str, Any]]:
    """Resolve one visual style/background/font palette without task routing."""

    width = int(params.get("canvas_width", group_default(render_defaults, "canvas_width", 820)))
    height = int(params.get("canvas_height", group_default(render_defaults, "canvas_height", 600)))
    image, background_meta, diagram_style, diagram_style_meta = prepare_geometry_diagram_style_and_background(
        instance_seed=int(instance_seed),
        params=params,
        scene_id="container_volume_transfer",
        canvas_width=int(width),
        canvas_height=int(height),
        require_grid=False,
    )
    fill_palettes: Tuple[Tuple[Color, Color, Color, Color], ...] = (
        ((236, 245, 255), (238, 246, 236), (201, 231, 255), (41, 122, 184)),
        ((255, 241, 230), (237, 241, 255), (218, 236, 255), (177, 93, 48)),
        ((239, 235, 255), (235, 248, 246), (220, 239, 249), (111, 92, 190)),
        ((232, 248, 237), (255, 241, 222), (214, 237, 229), (38, 137, 95)),
    )
    palette_rng = spawn_rng(int(instance_seed), f"{random_namespace}.palette")
    source_fill, target_fill, liquid_fill, accent_color = uniform_choice(
        palette_rng,
        fill_palettes,
    )
    font_size = int(params.get("label_font_size", group_default(render_defaults, "label_font_size", 22)))
    small_font_size = int(params.get("small_label_font_size", group_default(render_defaults, "small_label_font_size", 18)))
    line_width = int(params.get("line_width", group_default(render_defaults, "line_width", 3)))
    label_stroke_width = int(params.get("label_stroke_width", group_default(render_defaults, "label_stroke_width", 0)))
    return RenderContext(
        image=image,
        draw=ImageDraw.Draw(image),
        width=int(width),
        height=int(height),
        line_color=PANEL_LINE_COLOR,
        secondary_color=PANEL_SECONDARY_COLOR,
        label_color=PANEL_LABEL_COLOR,
        label_stroke_color=tuple(int(value) for value in diagram_style.label_stroke_rgb),
        source_fill=source_fill,
        target_fill=target_fill,
        liquid_fill=liquid_fill,
        accent_color=accent_color,
        muted_color=PANEL_BORDER_COLOR,
        panel_fill_color=PANEL_FILL_COLOR,
        line_width=max(2, int(line_width)),
        label_stroke_width=max(0, int(label_stroke_width)),
        font=load_font(max(12, int(font_size)), bold=False),
        small_font=load_font(max(10, int(small_font_size)), bold=False),
        diagram_style_meta=dict(diagram_style_meta),
        background_meta=dict(background_meta),
    ), {"technical_diagram": dict(diagram_style_meta), "background": dict(background_meta)}


def assert_bboxes_inside(bboxes: Sequence[BBox], *, width: int, height: int) -> None:
    for bbox in bboxes:
        x0, y0, x1, y1 = [float(value) for value in bbox]
        if x0 <= 2.0 or y0 <= 2.0 or x1 >= float(width) - 2.0 or y1 >= float(height) - 2.0:
            raise ValueError("container-volume-transfer label too close to canvas edge")


def render_container_volume_transfer_scene(ctx: RenderContext, problem: ResolvedProblem, *, instance_seed: int) -> RenderedScene:
    """Render one two-container transfer diagram and return projected witnesses."""

    rng = spawn_rng(int(instance_seed), f"geometry.container_volume_transfer.{problem.objective}.render.scene")
    is_resulting_height = str(problem.diagram_mode) == "resulting_fill_height"
    source_center_x = float(ctx.width) * 0.27 + float(rng.uniform(-14.0, 14.0))
    target_center_x = float(ctx.width) * 0.72 + float(rng.uniform(-14.0, 14.0))
    base_y = float(ctx.height) * 0.71 + float(rng.uniform(-8.0, 10.0))
    source_bbox = (source_center_x - 70.0, base_y - 230.0, source_center_x + 70.0, base_y)
    target_bbox = (target_center_x - 90.0, base_y - 245.0, target_center_x + 90.0, base_y + 6.0)
    panel_bbox = (
        min(source_bbox[0], target_bbox[0]) - 132.0,
        min(source_bbox[1], target_bbox[1]) - 42.0,
        max(source_bbox[2], target_bbox[2]) + 148.0,
        max(source_bbox[3], target_bbox[3]) + 64.0,
    )
    ctx.draw.rounded_rectangle(
        panel_bbox,
        radius=10,
        fill=ctx.panel_fill_color,
        outline=ctx.muted_color,
        width=max(1, ctx.line_width - 1),
    )
    if problem.source_shape == "cone":
        draw_cone(ctx, source_bbox)
    else:
        draw_cylinder(ctx, source_bbox, fill=ctx.source_fill, liquid_fraction=0.78)
    target_fill_fraction = float(problem.resulting_height) / float(problem.target_height) if is_resulting_height else 0.82
    fill_mark_bbox: BBox | None = None
    if problem.target_shape == "cylinder":
        fill_mark_bbox = draw_cylinder(
            ctx,
            target_bbox,
            fill=ctx.target_fill,
            liquid_fraction=target_fill_fraction,
            mark_fill_line=is_resulting_height,
        )
    else:
        target_bbox, fill_mark_bbox = draw_cuboid_tank(
            ctx,
            target_bbox,
            liquid_fraction=target_fill_fraction,
            mark_fill_line=is_resulting_height,
        )

    label_bboxes: Dict[str, BBox] = {}
    label_bboxes["source_title"] = draw_text_centered(ctx, f"source {problem.source_shape}", (source_center_x, source_bbox[1] - 26.0), small=True)
    label_bboxes["target_title"] = draw_text_centered(ctx, f"target {problem.target_shape}", (target_center_x, target_bbox[1] - 26.0), small=True)
    label_bboxes["source_base_area"] = draw_text_centered(ctx, f"base area={problem.source_base_area}", (source_center_x, source_bbox[3] + 28.0), small=True)
    label_bboxes["source_height"] = draw_text_centered(ctx, f"height={problem.source_height}", (source_bbox[0] - 56.0, (source_bbox[1] + source_bbox[3]) / 2.0), small=True)
    if problem.target_shape == "cylinder":
        label_bboxes["target_base_area"] = draw_text_centered(ctx, f"base area={problem.target_base_area}", (target_center_x, target_bbox[3] + 28.0), small=True)
        if not is_resulting_height:
            label_bboxes["target_height"] = draw_text_centered(ctx, f"height={problem.target_height}", (target_bbox[2] + 62.0, (target_bbox[1] + target_bbox[3]) / 2.0), small=True)
    else:
        label_bboxes["target_length"] = draw_text_centered(ctx, f"L={problem.target_length}", (target_center_x - 4.0, target_bbox[3] + 30.0), small=True)
        label_bboxes["target_width"] = draw_text_centered(ctx, f"W={problem.target_width}", (target_bbox[2] + 38.0, target_bbox[1] + 22.0), small=True)
        if not is_resulting_height:
            label_bboxes["target_height"] = draw_text_centered(ctx, f"H={problem.target_height}", (target_bbox[2] + 64.0, (target_bbox[1] + target_bbox[3]) / 2.0 + 22.0), small=True)
    if is_resulting_height:
        label_bboxes["transfer_count"] = draw_value_box(ctx, f"{problem.pour_count} full pours", (ctx.width / 2.0, 78.0 + float(rng.uniform(-4.0, 5.0))))
        if fill_mark_bbox is None:
            raise ValueError("resulting-height task requires a visible fill mark")
        fill_label_x = float(target_bbox[2]) + (72.0 if problem.target_shape == "cuboid" else 64.0)
        label_bboxes["fill_mark_label"] = draw_text_centered(ctx, "height=?", (fill_label_x, (fill_mark_bbox[1] + fill_mark_bbox[3]) / 2.0), small=True)
        fill_mark_bbox = _bbox_union((fill_mark_bbox, label_bboxes["fill_mark_label"]), width=ctx.width, height=ctx.height, pad=5.0)
    else:
        label_bboxes["question"] = draw_value_box(ctx, "full pours ?", (ctx.width / 2.0, 78.0 + float(rng.uniform(-4.0, 5.0))))
    arrow_bbox = draw_arrow(ctx, (source_bbox[2] + 26.0, (source_bbox[1] + source_bbox[3]) / 2.0 - 8.0), (target_bbox[0] - 26.0, (target_bbox[1] + target_bbox[3]) / 2.0 - 8.0))
    source_dimension_region = _bbox_union((label_bboxes["source_base_area"], label_bboxes["source_height"]), width=ctx.width, height=ctx.height, pad=6.0)
    if is_resulting_height:
        target_dimension_keys = ("target_base_area",) if problem.target_shape == "cylinder" else ("target_length", "target_width")
    else:
        target_dimension_keys = ("target_base_area", "target_height") if problem.target_shape == "cylinder" else ("target_length", "target_width", "target_height")
    target_dimension_region = (
        _bbox_union(tuple(label_bboxes[key] for key in target_dimension_keys), width=ctx.width, height=ctx.height, pad=6.0)
        if target_dimension_keys
        else None
    )
    annotation_bboxes = {
        "source_container_bbox": pad_bbox(source_bbox, 5.0, width=ctx.width, height=ctx.height),
        "target_container_bbox": pad_bbox(target_bbox, 5.0, width=ctx.width, height=ctx.height),
        "source_dimension_region_bbox": source_dimension_region,
    }
    if is_resulting_height:
        annotation_bboxes.update(
            {
                "target_base_dimension_region_bbox": target_dimension_region if target_dimension_region is not None else source_dimension_region,
                "transfer_count_bbox": label_bboxes["transfer_count"],
                "fill_mark_bbox": fill_mark_bbox if fill_mark_bbox is not None else target_dimension_region,
            }
        )
    else:
        annotation_bboxes.update(
            {
                "target_dimension_region_bbox": target_dimension_region if target_dimension_region is not None else source_dimension_region,
                "transfer_arrow_bbox": arrow_bbox,
            }
        )
    assert_bboxes_inside(tuple(annotation_bboxes.values()) + tuple(label_bboxes.values()), width=ctx.width, height=ctx.height)
    scene_entities = (
        {
            "entity_id": "source_container",
            "entity_type": str(problem.source_shape),
            "base_area_units": int(problem.source_base_area),
            "height_units": int(problem.source_height),
            "volume_units": int(problem.source_volume),
            "bbox": bbox_to_list(annotation_bboxes["source_container_bbox"]),
        },
        {
            "entity_id": "target_container",
            "entity_type": str(problem.target_shape),
            "base_area_units": int(problem.target_base_area),
            "length_units": int(problem.target_length),
            "width_units": int(problem.target_width),
            "height_units": int(problem.target_height),
            "volume_units": int(problem.target_volume),
            "resulting_height_units": float(problem.resulting_height),
            "pour_count": int(problem.pour_count),
            "bbox": bbox_to_list(annotation_bboxes["target_container_bbox"]),
        },
    )
    render_map = {
        "coord_space": "pixel",
        "source_shape": str(problem.source_shape),
        "target_shape": str(problem.target_shape),
        "source_container_bbox": bbox_to_list(annotation_bboxes["source_container_bbox"]),
        "target_container_bbox": bbox_to_list(annotation_bboxes["target_container_bbox"]),
        "annotation_bboxes": {key: bbox_to_list(value) for key, value in annotation_bboxes.items()},
        "label_bboxes": {key: bbox_to_list(value) for key, value in label_bboxes.items()},
        "pour_count": int(problem.pour_count),
        "resulting_height_units": float(problem.resulting_height),
    }
    return RenderedScene(
        image=ctx.image,
        annotation_bboxes=annotation_bboxes,
        label_bboxes=label_bboxes,
        scene_entities=scene_entities,
        render_map=render_map,
    )


def render_container_volume_transfer_with_retries(
    *,
    problem: ResolvedProblem,
    instance_seed: int,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    max_attempts: int,
    random_namespace: str,
) -> tuple[RenderedScene, Dict[str, Any]]:
    """Retry layout jitter until all labels and annotation boxes fit in-frame."""

    last_error: Exception | None = None
    for attempt_index in range(max(1, int(max_attempts))):
        try:
            render_seed = int(instance_seed) + int(attempt_index) * 9973
            ctx, render_meta = create_render_context(
                instance_seed=render_seed,
                params=params,
                render_defaults=render_defaults,
                random_namespace=str(random_namespace),
            )
            rendered = render_container_volume_transfer_scene(ctx, problem, instance_seed=render_seed)
            return rendered, dict(render_meta)
        except Exception as exc:
            last_error = exc
    raise RuntimeError("failed to render container volume-transfer scene") from last_error


__all__ = ["render_container_volume_transfer_with_retries"]
