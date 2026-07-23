"""Rendering primitives for volume-equivalence conversion diagrams."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from PIL import ImageDraw

from trace_tasks.core.sampling import uniform_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.geometry.shared.diagram_style import (
    prepare_geometry_diagram_style_and_background,
)
from trace_tasks.tasks.geometry.shared.measurement_rendering import (
    bbox_to_list,
    bbox_union_from_bboxes,
    pad_bbox,
)
from trace_tasks.tasks.shared.text_legibility import draw_text_traced
from trace_tasks.tasks.shared.text_rendering import load_font

from .defaults import SCENE_ID
from .construction import solid_volume
from .state import BBox, Point, RenderContext, RenderedScene, ResolvedProblem, SolidSpec

_PALETTES = (
    ((224, 239, 255), (255, 238, 218), (239, 248, 238), (28, 106, 176)),
    ((238, 234, 255), (231, 248, 244), (255, 242, 222), (112, 82, 190)),
    ((229, 246, 236), (255, 235, 229), (235, 241, 255), (42, 128, 90)),
)


def make_render_context(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    namespace: str,
) -> RenderContext:
    """Create the styled diagram canvas for one render attempt."""

    width = int(params.get("canvas_width", render_defaults.get("canvas_width", 860)))
    height = int(params.get("canvas_height", render_defaults.get("canvas_height", 640)))
    image, background_meta, diagram_style, diagram_style_meta = (
        prepare_geometry_diagram_style_and_background(
            instance_seed=int(instance_seed),
            params=params,
            scene_id=SCENE_ID,
            canvas_width=int(width),
            canvas_height=int(height),
            allow_dark=False,
            require_grid=False,
        )
    )
    palette_rng = spawn_rng(int(instance_seed), f"{namespace}.palette")
    source_fill, target_fill, option_fill, accent = uniform_choice(
        palette_rng,
        _PALETTES,
    )
    line_width = max(
        2, int(params.get("line_width", render_defaults.get("line_width", 3)))
    )
    font_size = int(
        params.get("label_font_size", render_defaults.get("label_font_size", 22))
    )
    small_font_size = int(
        params.get(
            "small_label_font_size", render_defaults.get("small_label_font_size", 19)
        )
    )
    diagram_style_meta = dict(diagram_style_meta)
    diagram_style_meta["volume_equivalence_text"] = {
        "label_stroke_width_px": 0,
        "label_rgb": [10, 14, 22],
        "line_rgb": [35, 48, 62],
        "small_font_bold": False,
        "small_font_size_px": int(small_font_size),
    }
    return RenderContext(
        image=image,
        draw=ImageDraw.Draw(image),
        width=width,
        height=height,
        line_color=(35, 48, 62),
        secondary_color=(96, 110, 124),
        label_color=(10, 14, 22),
        label_stroke_color=(255, 255, 255),
        source_fill=source_fill,
        target_fill=target_fill,
        option_fill=option_fill,
        accent_color=accent,
        muted_color=tuple(int(value) for value in diagram_style.panel_border_rgb),
        line_width=max(2, line_width),
        label_stroke_width=0,
        font=load_font(max(12, font_size), bold=True),
        small_font=load_font(max(10, small_font_size), bold=False),
        diagram_style_meta=diagram_style_meta,
        background_meta=dict(background_meta),
    )


def _draw_text(
    ctx: RenderContext, text: str, center: Point, *, small: bool = False
) -> BBox:
    font = ctx.small_font if small else ctx.font
    bbox = ctx.draw.textbbox(
        (0, 0), str(text), font=font, stroke_width=ctx.label_stroke_width
    )
    text_width = float(bbox[2] - bbox[0])
    text_height = float(bbox[3] - bbox[1])
    x = float(center[0]) - (float(bbox[0]) + float(bbox[2])) / 2.0
    y = float(center[1]) - (float(bbox[1]) + float(bbox[3])) / 2.0
    draw_text_traced(
        ctx.draw,
        (x, y),
        str(text),
        font=font,
        fill=ctx.label_color,
        stroke_width=ctx.label_stroke_width,
        stroke_fill=ctx.label_stroke_color,
        role="readout",
        required=False,
    )
    actual_bbox = (
        x + float(bbox[0]),
        y + float(bbox[1]),
        x + float(bbox[2]),
        y + float(bbox[3]),
    )
    return pad_bbox(actual_bbox, 4.0, width=ctx.width, height=ctx.height)


def _draw_value_box(
    ctx: RenderContext, text: str, center: Point, *, small: bool = False
) -> BBox:
    font = ctx.small_font if small else ctx.font
    bbox = ctx.draw.textbbox(
        (0, 0), str(text), font=font, stroke_width=ctx.label_stroke_width
    )
    text_width = float(bbox[2] - bbox[0])
    text_height = float(bbox[3] - bbox[1])
    x0 = float(center[0]) - text_width / 2.0 - 10.0
    y0 = float(center[1]) - text_height / 2.0 - 6.0
    x1 = x0 + text_width + 20.0
    y1 = y0 + text_height + 12.0
    ctx.draw.rounded_rectangle(
        (x0, y0, x1, y1),
        radius=6,
        fill=(255, 255, 255),
        outline=ctx.muted_color,
        width=1,
    )
    text_x = x0 + 10.0 - float(bbox[0])
    text_y = y0 + 6.0 - float(bbox[1])
    draw_text_traced(
        ctx.draw,
        (text_x, text_y),
        str(text),
        font=font,
        fill=ctx.label_color,
        stroke_width=ctx.label_stroke_width,
        stroke_fill=ctx.label_stroke_color,
        role="readout",
        required=False,
    )
    return pad_bbox((x0, y0, x1, y1), 3.0, width=ctx.width, height=ctx.height)


def _draw_cylinder(
    ctx: RenderContext, bbox: BBox, *, fill: tuple[int, int, int]
) -> None:
    x0, y0, x1, y1 = [float(value) for value in bbox]
    ellipse_h = min(34.0, max(18.0, (y1 - y0) * 0.17))
    body_top = y0 + ellipse_h / 2.0
    body_bottom = y1 - ellipse_h / 2.0
    ctx.draw.rectangle(
        (x0, body_top, x1, body_bottom),
        fill=fill,
        outline=ctx.line_color,
        width=ctx.line_width,
    )
    ctx.draw.ellipse(
        (x0, y0, x1, y0 + ellipse_h),
        fill=fill,
        outline=ctx.line_color,
        width=ctx.line_width,
    )
    ctx.draw.arc(
        (x0, y1 - ellipse_h, x1, y1),
        start=0,
        end=180,
        fill=ctx.line_color,
        width=ctx.line_width,
    )
    ctx.draw.arc(
        (x0, y1 - ellipse_h, x1, y1),
        start=180,
        end=360,
        fill=ctx.secondary_color,
        width=max(1, ctx.line_width - 1),
    )


def _draw_cone(ctx: RenderContext, bbox: BBox, *, fill: tuple[int, int, int]) -> None:
    x0, y0, x1, y1 = [float(value) for value in bbox]
    apex = ((x0 + x1) / 2.0, y0)
    base_y = y1 - min(30.0, max(18.0, (y1 - y0) * 0.16))
    ctx.draw.polygon(
        [apex, (x0, base_y), (x1, base_y)], fill=fill, outline=ctx.line_color
    )
    ctx.draw.line([apex, (x0, base_y)], fill=ctx.line_color, width=ctx.line_width)
    ctx.draw.line([apex, (x1, base_y)], fill=ctx.line_color, width=ctx.line_width)
    ctx.draw.ellipse(
        (x0, base_y - 12.0, x1, y1),
        fill=fill,
        outline=ctx.line_color,
        width=ctx.line_width,
    )
    ctx.draw.arc(
        (x0, base_y - 12.0, x1, y1),
        start=180,
        end=360,
        fill=ctx.secondary_color,
        width=max(1, ctx.line_width - 1),
    )


def _draw_cuboid(ctx: RenderContext, bbox: BBox, *, fill: tuple[int, int, int]) -> None:
    x0, y0, x1, y1 = [float(value) for value in bbox]
    depth = min(42.0, max(24.0, (x1 - x0) * 0.18))
    front = (x0, y0 + depth * 0.55, x1 - depth, y1)
    back = (
        front[0] + depth,
        front[1] - depth * 0.55,
        front[2] + depth,
        front[3] - depth * 0.55,
    )
    ctx.draw.polygon(
        [
            (front[0], front[1]),
            (back[0], back[1]),
            (back[2], back[1]),
            (front[2], front[1]),
        ],
        fill=fill,
        outline=ctx.line_color,
    )
    ctx.draw.polygon(
        [
            (front[2], front[1]),
            (back[2], back[1]),
            (back[2], back[3]),
            (front[2], front[3]),
        ],
        fill=tuple(max(0, color - 18) for color in fill),
        outline=ctx.line_color,
    )
    ctx.draw.rectangle(front, fill=fill, outline=ctx.line_color, width=ctx.line_width)


def _draw_solid(
    ctx: RenderContext, spec: SolidSpec, bbox: BBox, *, fill: tuple[int, int, int]
) -> None:
    if spec.shape == "cuboid":
        _draw_cuboid(ctx, bbox, fill=fill)
    elif spec.shape == "cylinder":
        _draw_cylinder(ctx, bbox, fill=fill)
    elif spec.shape == "cone":
        _draw_cone(ctx, bbox, fill=fill)
    else:
        raise ValueError(f"unsupported shape: {spec.shape}")


def _dimension_labels(spec: SolidSpec, *, unknown_role: str = "") -> tuple[str, ...]:
    if spec.shape == "cuboid":
        height = "H=?" if unknown_role == "cuboid_height" else f"H={spec.height}"
        return (f"L={spec.length}", f"W={spec.width}", height)
    if spec.shape == "cylinder":
        height = "H=?" if unknown_role == "cylinder_height" else f"H={spec.height}"
        return (f"base area={spec.base_area}", height)
    if spec.shape == "cone":
        height = "H=?" if unknown_role == "cone_height" else f"H={spec.height}"
        return (f"base area={spec.base_area}", height)
    return ()


def _draw_option_dimension_band(
    ctx: RenderContext, labels: tuple[str, ...], card: BBox
) -> BBox:
    """Draw compact option measurements in a reserved high-contrast band."""

    band_height = max(36.0, 17.0 * float(len(labels)) + 10.0)
    band = (card[0] + 7.0, card[3] - band_height - 7.0, card[2] - 7.0, card[3] - 7.0)
    ctx.draw.rounded_rectangle(
        band, radius=5, fill=(255, 255, 255), outline=ctx.muted_color, width=1
    )
    line_gap = band_height / float(len(labels) + 1)
    label_boxes = []
    for index, text in enumerate(labels):
        label_boxes.append(
            _draw_text(
                ctx,
                text,
                ((band[0] + band[2]) / 2.0, band[1] + line_gap * float(index + 1)),
                small=True,
            )
        )
    return bbox_union_from_bboxes(
        label_boxes, width=ctx.width, height=ctx.height, pad=3.0
    )


def _draw_arrow(ctx: RenderContext, start: Point, end: Point) -> BBox:
    ctx.draw.line([start, end], fill=ctx.accent_color, width=max(3, ctx.line_width + 1))
    dx = float(end[0]) - float(start[0])
    dy = float(end[1]) - float(start[1])
    length = max(1e-6, (dx * dx + dy * dy) ** 0.5)
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    head = 16.0
    wing = 8.0
    p1 = (float(end[0]) - ux * head + px * wing, float(end[1]) - uy * head + py * wing)
    p2 = (float(end[0]) - ux * head - px * wing, float(end[1]) - uy * head - py * wing)
    ctx.draw.polygon([end, p1, p2], fill=ctx.accent_color)
    return bbox_union_from_bboxes(
        (
            (start[0], start[1], start[0], start[1]),
            (end[0], end[1], end[0], end[1]),
            (p1[0], p1[1], p1[0], p1[1]),
            (p2[0], p2[1], p2[0], p2[1]),
        ),
        width=ctx.width,
        height=ctx.height,
        pad=8.0,
    )


def _option_positions(option_count: int) -> tuple[Point, ...]:
    if int(option_count) == 4:
        return ((290.0, 360.0), (560.0, 360.0), (290.0, 520.0), (560.0, 520.0))
    if int(option_count) == 6:
        return (
            (160.0, 360.0),
            (425.0, 360.0),
            (690.0, 360.0),
            (160.0, 520.0),
            (425.0, 520.0),
            (690.0, 520.0),
        )
    raise ValueError(f"unsupported volume option_count={option_count}")


def _scene_entities(problem: ResolvedProblem) -> tuple[dict[str, Any], ...]:
    entities = [
        {
            "entity_id": "source_solid",
            "entity_type": problem.source.shape,
            "base_area_units": int(problem.source.base_area),
            "height_units": int(problem.source.height),
            "length_units": int(problem.source.length),
            "width_units": int(problem.source.width),
            "volume_units": int(solid_volume(problem.source)),
        },
        {
            "entity_id": "target_solid",
            "entity_type": problem.target.shape,
            "base_area_units": int(problem.target.base_area),
            "height_units": int(problem.target.height),
            "length_units": int(problem.target.length),
            "width_units": int(problem.target.width),
            "volume_units": int(solid_volume(problem.target)),
        },
    ]
    for option in problem.option_specs:
        entities.append(
            {
                "entity_id": f"option_{option.label}",
                "entity_type": option.solid.shape,
                "option_label": str(option.label),
                "base_area_units": int(option.solid.base_area),
                "height_units": int(option.solid.height),
                "length_units": int(option.solid.length),
                "width_units": int(option.solid.width),
                "volume_units": int(option.volume),
                "is_answer": str(option.label) == str(problem.selected_option_label),
            }
        )
    return tuple(entities)


def render_missing_dimension_scene(
    ctx: RenderContext,
    problem: ResolvedProblem,
    *,
    instance_seed: int,
    random_namespace: str,
) -> RenderedScene:
    """Render the two-solid conversion layout and role boxes for the missing target dimension."""

    rng = spawn_rng(int(instance_seed), f"{random_namespace}.missing")
    source_bbox = (
        120.0 + rng.uniform(-8.0, 8.0),
        205.0,
        310.0 + rng.uniform(-8.0, 8.0),
        420.0,
    )
    target_bbox = (
        535.0 + rng.uniform(-8.0, 8.0),
        205.0,
        720.0 + rng.uniform(-8.0, 8.0),
        420.0,
    )
    ctx.draw.rounded_rectangle(
        (72, 116, 780, 500),
        radius=10,
        fill=(255, 255, 255),
        outline=ctx.muted_color,
        width=1,
    )
    _draw_solid(ctx, problem.source, source_bbox, fill=ctx.source_fill)
    _draw_solid(ctx, problem.target, target_bbox, fill=ctx.target_fill)
    label_bboxes: dict[str, BBox] = {}
    label_bboxes["source_title"] = _draw_text(
        ctx,
        f"source {problem.source.shape}",
        ((source_bbox[0] + source_bbox[2]) / 2.0, 166.0),
        small=True,
    )
    label_bboxes["target_title"] = _draw_text(
        ctx,
        f"target {problem.target.shape}",
        ((target_bbox[0] + target_bbox[2]) / 2.0, 166.0),
        small=True,
    )
    source_label_boxes = [
        _draw_value_box(
            ctx,
            text,
            ((source_bbox[0] + source_bbox[2]) / 2.0, 450.0 + index * 34.0),
            small=True,
        )
        for index, text in enumerate(_dimension_labels(problem.source))
    ]
    target_label_boxes: list[BBox] = []
    target_known_label_boxes: list[BBox] = []
    unknown_label_boxes: list[BBox] = []
    for index, text in enumerate(
        _dimension_labels(problem.target, unknown_role=problem.target_unknown_role)
    ):
        bbox = _draw_value_box(
            ctx,
            text,
            ((target_bbox[0] + target_bbox[2]) / 2.0, 450.0 + index * 34.0),
            small=True,
        )
        target_label_boxes.append(bbox)
        if "?" in text:
            unknown_label_boxes.append(bbox)
        else:
            target_known_label_boxes.append(bbox)
    arrow_bbox = _draw_arrow(
        ctx, (source_bbox[2] + 42.0, 300.0), (target_bbox[0] - 42.0, 300.0)
    )
    label_bboxes["equal_volume"] = _draw_text(
        ctx, "same volume", (ctx.width / 2.0, 260.0), small=True
    )
    annotation_bboxes = {
        "source_solid_bbox": pad_bbox(
            source_bbox, 8.0, width=ctx.width, height=ctx.height
        ),
        "target_solid_bbox": pad_bbox(
            target_bbox, 8.0, width=ctx.width, height=ctx.height
        ),
    }
    source_inputs_bbox = bbox_union_from_bboxes(
        source_label_boxes, width=ctx.width, height=ctx.height, pad=4.0
    )
    target_inputs_bbox = bbox_union_from_bboxes(
        target_known_label_boxes, width=ctx.width, height=ctx.height, pad=4.0
    )
    render_map = {
        "coord_space": "pixel",
        "source_solid_bbox": bbox_to_list(annotation_bboxes["source_solid_bbox"]),
        "target_solid_bbox": bbox_to_list(annotation_bboxes["target_solid_bbox"]),
        "source_inputs_bbox": bbox_to_list(source_inputs_bbox),
        "target_inputs_bbox": bbox_to_list(target_inputs_bbox),
        "source_dimension_region_bbox": bbox_to_list(source_inputs_bbox),
        "target_dimension_region_bbox": bbox_to_list(
            bbox_union_from_bboxes(
                target_label_boxes, width=ctx.width, height=ctx.height, pad=4.0
            )
        ),
        "target_unknown_region_bbox": bbox_to_list(
            bbox_union_from_bboxes(
                unknown_label_boxes, width=ctx.width, height=ctx.height, pad=4.0
            )
        ),
        "conversion_arrow_bbox": bbox_to_list(arrow_bbox),
        "annotation_bboxes": {
            key: bbox_to_list(value) for key, value in annotation_bboxes.items()
        },
    }
    return RenderedScene(
        image=ctx.image,
        annotation_bboxes=annotation_bboxes,
        label_bboxes=label_bboxes,
        scene_entities=_scene_entities(problem),
        render_map=render_map,
        render_meta={
            "style": {
                "technical_diagram": dict(ctx.diagram_style_meta),
                "background": dict(ctx.background_meta),
            }
        },
    )


def render_option_scene(
    ctx: RenderContext,
    problem: ResolvedProblem,
    *,
    instance_seed: int,
    random_namespace: str,
) -> RenderedScene:
    """Render the source solid plus option-card layout and boxes for the selected equal-volume option."""

    rng = spawn_rng(int(instance_seed), f"{random_namespace}.option")
    source_bbox = (
        320.0 + rng.uniform(-8.0, 8.0),
        90.0,
        510.0 + rng.uniform(-8.0, 8.0),
        280.0,
    )
    ctx.draw.rounded_rectangle(
        (72, 42, 790, 585),
        radius=10,
        fill=(255, 255, 255),
        outline=ctx.muted_color,
        width=1,
    )
    _draw_solid(ctx, problem.source, source_bbox, fill=ctx.source_fill)
    label_bboxes: dict[str, BBox] = {
        "source_title": _draw_text(
            ctx,
            f"source {problem.source.shape}",
            ((source_bbox[0] + source_bbox[2]) / 2.0, 62.0),
            small=True,
        )
    }
    source_label_boxes = [
        _draw_value_box(ctx, text, (190.0, 118.0 + index * 36.0), small=True)
        for index, text in enumerate(_dimension_labels(problem.source))
    ]
    option_bboxes: dict[str, BBox] = {}
    option_dimension_bboxes: dict[str, BBox] = {}
    option_count = len(problem.option_specs)
    card_half_width = 90.0 if option_count == 6 else 102.0
    for option, center in zip(problem.option_specs, _option_positions(option_count)):
        card = (
            center[0] - card_half_width,
            center[1] - 80.0,
            center[0] + card_half_width,
            center[1] + 72.0,
        )
        ctx.draw.rounded_rectangle(
            card, radius=8, fill=(250, 252, 255), outline=ctx.muted_color, width=1
        )
        label_bboxes[f"option_{option.label}_label"] = _draw_text(
            ctx, option.label, (card[0] + 22.0, card[1] + 20.0), small=False
        )
        solid_bbox = (
            center[0] - 44.0,
            card[1] + 22.0,
            center[0] + 50.0,
            card[1] + 94.0,
        )
        _draw_solid(ctx, option.solid, solid_bbox, fill=ctx.option_fill)
        dimension_bbox = _draw_option_dimension_band(
            ctx, _dimension_labels(option.solid), card
        )
        option_bboxes[str(option.label)] = pad_bbox(
            card, 4.0, width=ctx.width, height=ctx.height
        )
        option_dimension_bboxes[str(option.label)] = dimension_bbox
    selected = str(problem.selected_option_label)
    annotation_bboxes = {
        "source_solid_bbox": pad_bbox(
            source_bbox, 8.0, width=ctx.width, height=ctx.height
        ),
        "source_dimension_region_bbox": bbox_union_from_bboxes(
            source_label_boxes, width=ctx.width, height=ctx.height, pad=5.0
        ),
        "selected_option_bbox": option_bboxes[selected],
        "selected_option_dimension_region_bbox": option_dimension_bboxes[selected],
    }
    render_map = {
        "coord_space": "pixel",
        "source_solid_bbox": bbox_to_list(annotation_bboxes["source_solid_bbox"]),
        "option_bboxes": {
            key: bbox_to_list(value) for key, value in option_bboxes.items()
        },
        "option_count": int(len(problem.option_specs)),
        "selected_option_label": selected,
        "annotation_bboxes": {
            key: bbox_to_list(value) for key, value in annotation_bboxes.items()
        },
    }
    return RenderedScene(
        image=ctx.image,
        annotation_bboxes=annotation_bboxes,
        label_bboxes=label_bboxes,
        scene_entities=_scene_entities(problem),
        render_map=render_map,
        render_meta={
            "option_count": int(len(problem.option_specs)),
            "style": {
                "technical_diagram": dict(ctx.diagram_style_meta),
                "background": dict(ctx.background_meta),
            },
        },
    )


def render_volume_equivalence_with_retries(
    *,
    problem: ResolvedProblem,
    render_scene: Callable[..., RenderedScene],
    instance_seed: int,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    max_attempts: int,
    random_namespace: str,
) -> RenderedScene:
    """Render the bound problem, retrying only layout failures."""

    last_error: Exception | None = None
    for attempt in range(max(1, int(max_attempts))):
        try:
            render_seed = int(instance_seed) + int(attempt) * 9973
            ctx = make_render_context(
                instance_seed=render_seed,
                params=params,
                render_defaults=render_defaults,
                namespace=str(random_namespace),
            )
            return render_scene(
                ctx,
                problem,
                instance_seed=render_seed,
                random_namespace=str(random_namespace),
            )
        except Exception as exc:
            last_error = exc
    raise RuntimeError("failed to render volume-equivalence scene") from last_error


__all__ = [
    "make_render_context",
    "render_missing_dimension_scene",
    "render_option_scene",
    "render_volume_equivalence_with_retries",
]
