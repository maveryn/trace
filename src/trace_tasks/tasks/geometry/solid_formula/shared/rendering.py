"""Rendering primitives for solid-formula diagrams."""

from __future__ import annotations

from typing import Any, MutableMapping, Mapping, Sequence

from PIL import ImageDraw

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.geometry.shared.diagram_style import (
    geometry_shape_style_from_diagram_style,
    prepare_geometry_diagram_style_and_background,
)
from trace_tasks.tasks.geometry.shared.measurement_rendering import (
    bbox_from_points,
    bbox_to_list,
    draw_dimension_line,
    draw_readout_centered,
    pad_bbox,
)
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.deterministic_sampling import resolve_selection_index
from trace_tasks.tasks.shared.font_assets import font_role_trace, sample_font_family
from trace_tasks.tasks.shared.text_legibility import contrast_ratio
from trace_tasks.tasks.shared.text_rendering import load_font, temporary_default_font_family

from .defaults import SCENE_ID
from .measurements import format_measure, format_pi_multiple
from .state import BBox, Color, Point, RenderContext, RenderedSolidFormulaScene, SolidFormulaProblem

_DARK_LABEL_RGB: Color = (10, 14, 22)
_LIGHT_LABEL_RGB: Color = (250, 252, 255)


def _min_contrast(color: Color, surfaces: tuple[Color, ...]) -> float:
    if not surfaces:
        return float("inf")
    return min(float(contrast_ratio(color, surface)) for surface in surfaces)


def _resolve_measurement_label_color(*, preferred: Color, surfaces: tuple[Color, ...]) -> tuple[Color, str, float]:
    """Use a neutral ink that stays readable on the sampled solid-formula treatment."""

    candidates = (preferred, _DARK_LABEL_RGB, _LIGHT_LABEL_RGB)
    best = max(candidates, key=lambda candidate: (_min_contrast(candidate, surfaces), candidate == preferred))
    policy = "preferred" if best == preferred else "neutral_high_contrast"
    return best, policy, _min_contrast(best, surfaces)


def create_render_context(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    random_namespace: str,
) -> RenderContext:
    """Resolve deterministic style, font, canvas, and color state."""

    rng = spawn_rng(int(instance_seed), str(random_namespace))
    width = int(params.get("canvas_width", group_default(rendering_defaults, "canvas_width", 820)))
    height = int(params.get("canvas_height", group_default(rendering_defaults, "canvas_height", 580)))
    image, background_meta, diagram_style, diagram_style_meta = prepare_geometry_diagram_style_and_background(
        instance_seed=int(instance_seed),
        params=params,
        scene_id=SCENE_ID,
        canvas_width=int(width),
        canvas_height=int(height),
        allow_dark=True,
    )
    shape_style = geometry_shape_style_from_diagram_style(diagram_style)
    palettes: tuple[tuple[Color, Color, Color, Color], ...] = (
        (
            tuple(int(v) for v in diagram_style.panel_fill_rgb),
            tuple(int(v) for v in diagram_style.panel_alt_fill_rgb),
            tuple(int(v) for v in diagram_style.accent_rgb),
            tuple(int(v) for v in diagram_style.guide_rgb),
        ),
        (
            tuple(int(v) for v in diagram_style.option_fill_rgb),
            tuple(int(v) for v in diagram_style.muted_fill_rgb),
            tuple(int(v) for v in diagram_style.secondary_accent_rgb),
            tuple(int(v) for v in diagram_style.secondary_stroke_rgb),
        ),
        (
            tuple(int(v) for v in diagram_style.panel_alt_fill_rgb),
            tuple(int(v) for v in diagram_style.panel_fill_rgb),
            tuple(int(v) for v in diagram_style.highlight_rgb),
            tuple(int(v) for v in diagram_style.guide_rgb),
        ),
    )
    palette_index = resolve_selection_index(
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{random_namespace}.palette",
    )
    fill_color, secondary_fill_color, accent_color, muted_color = palettes[
        int(palette_index) % len(palettes)
    ]
    label_surfaces: tuple[Color, ...] = (
        tuple(int(v) for v in diagram_style.canvas_rgb),
        tuple(int(v) for v in diagram_style.paper_rgb),
        tuple(int(v) for v in diagram_style.panel_fill_rgb),
        tuple(int(v) for v in diagram_style.panel_alt_fill_rgb),
        fill_color,
        secondary_fill_color,
    )
    label_color, label_color_policy, label_min_surface_contrast = _resolve_measurement_label_color(
        preferred=shape_style.label_color,
        surfaces=label_surfaces,
    )
    font_size = int(params.get("label_font_size", group_default(rendering_defaults, "label_font_size", 22)))
    small_font_size = int(
        params.get("small_label_font_size", group_default(rendering_defaults, "small_label_font_size", 18))
    )
    font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace=f"{random_namespace}.font",
        params=params,
    )
    with temporary_default_font_family(str(font_family)):
        font = load_font(max(12, int(font_size)), bold=False)
        small_font = load_font(max(10, int(small_font_size)), bold=False)
    requested_label_stroke_width = int(
        params.get("label_stroke_width", group_default(rendering_defaults, "label_stroke_width", 0))
    )
    if "label_stroke_width" not in params and label_color == _LIGHT_LABEL_RGB:
        requested_label_stroke_width = max(
            int(requested_label_stroke_width),
            int(group_default(rendering_defaults, "dark_measurement_label_stroke_width", 1)),
        )
    label_stroke_width = max(0, min(1, int(requested_label_stroke_width)))
    return RenderContext(
        rng=rng,
        image=image,
        draw=ImageDraw.Draw(image),
        width=int(width),
        height=int(height),
        line_color=shape_style.line_color,
        label_color=label_color,
        label_stroke_color=label_color,
        fill_color=fill_color,
        secondary_fill_color=secondary_fill_color,
        accent_color=accent_color,
        muted_color=muted_color,
        line_width=max(2, int(params.get("line_width", group_default(rendering_defaults, "line_width", 4)))),
        font=font,
        small_font=small_font,
        label_stroke_width=int(label_stroke_width),
        diagram_style_meta=dict(diagram_style_meta),
        background_meta=dict(background_meta),
        font_meta=font_role_trace(str(font_family), role="readout"),
        palette_meta={
            "fill_color": list(fill_color),
            "secondary_fill_color": list(secondary_fill_color),
            "accent_color": list(accent_color),
            "muted_color": list(muted_color),
            "palette_index": int(palette_index) % len(palettes),
            "measurement_label_color": list(label_color),
            "measurement_label_color_policy": str(label_color_policy),
            "measurement_label_min_surface_contrast": round(float(label_min_surface_contrast), 3),
            "measurement_label_surface_rgbs": [list(surface) for surface in label_surfaces],
        },
    )


def _dimension_segment(start: Point, end: Point) -> list[list[float]]:
    return [
        [round(float(start[0]), 3), round(float(start[1]), 3)],
        [round(float(end[0]), 3), round(float(end[1]), 3)],
    ]


def _dimension_bbox(ctx: RenderContext, start: Point, end: Point) -> BBox:
    return bbox_from_points(
        (start, end),
        width=int(ctx.width),
        height=int(ctx.height),
        pad=max(12.0, float(ctx.line_width) * 4.0),
    )


def _draw_dimension(
    ctx: RenderContext,
    start: Point,
    end: Point,
    label: str,
    *,
    label_offset: Point = (0.0, 0.0),
    target: bool = False,
) -> BBox:
    return draw_dimension_line(
        ctx,
        start,
        end,
        label,
        label_offset=label_offset,
        color=ctx.accent_color if bool(target) else ctx.label_color,
        backed=True,
    )


def _add_dimension(
    ctx: RenderContext,
    *,
    label_bboxes: MutableMapping[str, BBox],
    dimension_bboxes: MutableMapping[str, BBox],
    dimension_segments: MutableMapping[str, list[list[float]]],
    label_key: str,
    witness_key: str,
    start: Point,
    end: Point,
    label: str,
    label_offset: Point = (0.0, 0.0),
    target: bool = False,
) -> None:
    label_bboxes[str(label_key)] = _draw_dimension(
        ctx,
        start,
        end,
        label,
        label_offset=label_offset,
        target=bool(target),
    )
    dimension_bboxes[str(witness_key)] = _dimension_bbox(ctx, start, end)
    dimension_segments[str(witness_key)] = _dimension_segment(start, end)


def _draw_dashed_line(
    ctx: RenderContext,
    start: Point,
    end: Point,
    *,
    fill: Color,
    width: int,
    dash: float = 10.0,
    gap: float = 7.0,
) -> None:
    dx = float(end[0]) - float(start[0])
    dy = float(end[1]) - float(start[1])
    length = (dx**2 + dy**2) ** 0.5
    if length <= 1e-9:
        return
    ux = dx / length
    uy = dy / length
    distance = 0.0
    while distance < length:
        seg_start = distance
        seg_end = min(length, distance + dash)
        ctx.draw.line(
            [
                (float(start[0]) + ux * seg_start, float(start[1]) + uy * seg_start),
                (float(start[0]) + ux * seg_end, float(start[1]) + uy * seg_end),
            ],
            fill=fill,
            width=width,
        )
        distance += dash + gap


def _solid_entity(kind: str, bbox: BBox) -> tuple[dict[str, Any], ...]:
    return (
        {
            "entity_id": "solid",
            "entity_type": str(kind),
            "bbox": bbox_to_list(bbox),
        },
    )


def _render_map(
    kind: str,
    solid_bbox: BBox,
    label_bboxes: Mapping[str, BBox],
    *,
    dimension_bboxes: Mapping[str, BBox] | None = None,
    dimension_segments: Mapping[str, Sequence[Sequence[float]]] | None = None,
) -> dict[str, Any]:
    return {
        "coord_space": "pixel",
        "solid": {
            "kind": str(kind),
            "bbox": bbox_to_list(solid_bbox),
        },
        "label_bboxes": {str(key): bbox_to_list(value) for key, value in label_bboxes.items()},
        "dimension_bboxes": {
            str(key): bbox_to_list(value)
            for key, value in dict(dimension_bboxes or {}).items()
        },
        "dimension_segments": {
            str(key): [[round(float(point[0]), 3), round(float(point[1]), 3)] for point in segment[:2]]
            for key, segment in dict(dimension_segments or {}).items()
        },
    }


def _with_scene_payload(
    *,
    ctx: RenderContext,
    problem: SolidFormulaProblem,
    solid_bbox: BBox,
    label_bboxes: Mapping[str, BBox],
    annotation_bboxes: Mapping[str, BBox],
    dimension_segments: Mapping[str, Sequence[Sequence[float]]],
    annotation_roles: tuple[str, ...],
) -> RenderedSolidFormulaScene:
    """Pack rendered labels and measurements into scene-neutral output state."""

    annotation_bboxes = {
        str(role): annotation_bboxes[str(role)]
        for role in tuple(str(value) for value in annotation_roles)
    }
    measurements = {
        "formula_family": str(problem.formula_family),
        "solid_kind": str(problem.solid_kind),
        "unknown_dimension": str(problem.unknown_dimension),
        "radius": None if problem.radius is None else float(problem.radius),
        "total_height": None if problem.total_height is None else float(problem.total_height),
        "cylinder_height": None if problem.cylinder_height is None else float(problem.cylinder_height),
        "cone_height": None if problem.cone_height is None else float(problem.cone_height),
        "volume": None if problem.volume is None else float(problem.volume),
        "volume_pi_multiple": None
        if problem.volume_pi_multiple is None
        else float(problem.volume_pi_multiple),
        "side_a": None if problem.side_a is None else float(problem.side_a),
        "side_b": None if problem.side_b is None else float(problem.side_b),
        "prism_height": None if problem.prism_height is None else float(problem.prism_height),
        "pyramid_height": None if problem.pyramid_height is None else float(problem.pyramid_height),
        "triangle_base": None if problem.triangle_base is None else float(problem.triangle_base),
        "prism_length": None if problem.prism_length is None else float(problem.prism_length),
        "wall_height": None if problem.wall_height is None else float(problem.wall_height),
        "roof_height": None if problem.roof_height is None else float(problem.roof_height),
        "formula": str(problem.formula),
        "answer_value": float(problem.answer),
    }
    return RenderedSolidFormulaScene(
        image=ctx.image,
        annotation_bboxes=dict(annotation_bboxes),
        annotation_roles=tuple(annotation_roles),
        label_bboxes=dict(label_bboxes),
        scene_entities=_solid_entity(problem.solid_kind, solid_bbox),
        render_map=_render_map(
            problem.solid_kind,
            solid_bbox,
            label_bboxes,
            dimension_bboxes=annotation_bboxes,
            dimension_segments=dimension_segments,
        ),
        measurements=measurements,
    )


def _draw_cylinder_cone_body(ctx: RenderContext) -> tuple[BBox, dict[str, Point]]:
    """Draw the shared cylinder-cone body used by both cylinder-cone tasks."""

    x0, x1 = 250.0, 570.0
    cyl_top_y, bottom_y = 252.0, 438.0
    apex = (410.0, 84.0)
    ellipse_h = 66.0
    cx = (x0 + x1) / 2.0

    ctx.draw.rectangle((x0, cyl_top_y, x1, bottom_y), fill=ctx.fill_color)
    ctx.draw.line([(x0, cyl_top_y), (x0, bottom_y)], fill=ctx.line_color, width=ctx.line_width)
    ctx.draw.line([(x1, cyl_top_y), (x1, bottom_y)], fill=ctx.line_color, width=ctx.line_width)
    ctx.draw.arc(
        (x0, bottom_y - ellipse_h / 2.0, x1, bottom_y + ellipse_h / 2.0),
        0,
        180,
        fill=ctx.line_color,
        width=ctx.line_width,
    )
    ctx.draw.arc(
        (x0, bottom_y - ellipse_h / 2.0, x1, bottom_y + ellipse_h / 2.0),
        180,
        360,
        fill=ctx.muted_color,
        width=max(2, ctx.line_width - 1),
    )
    ctx.draw.polygon((apex, (x0, cyl_top_y), (x1, cyl_top_y)), fill=ctx.secondary_fill_color)
    ctx.draw.line([apex, (x0, cyl_top_y)], fill=ctx.line_color, width=ctx.line_width)
    ctx.draw.line([apex, (x1, cyl_top_y)], fill=ctx.line_color, width=ctx.line_width)
    ctx.draw.ellipse(
        (x0, cyl_top_y - ellipse_h / 2.0, x1, cyl_top_y + ellipse_h / 2.0),
        fill=None,
        outline=ctx.line_color,
        width=ctx.line_width,
    )
    _draw_dashed_line(ctx, apex, (cx, cyl_top_y), fill=ctx.accent_color, width=3)
    bbox = pad_bbox(
        (x0, apex[1], x1, bottom_y + ellipse_h / 2.0),
        42.0,
        width=ctx.width,
        height=ctx.height,
    )
    return bbox, {
        "apex": apex,
        "base_center": (cx, cyl_top_y),
        "base_right": (x1, cyl_top_y),
        "bottom_left": (x0, bottom_y),
        "bottom_right": (x1, bottom_y),
        "cyl_top_right": (x1, cyl_top_y),
    }


def render_cylinder_cone_radius(ctx: RenderContext, problem: SolidFormulaProblem) -> RenderedSolidFormulaScene:
    """Render labels needed to solve the cylinder-cone shared-radius formula."""

    solid_bbox, points = _draw_cylinder_cone_body(ctx)
    label_bboxes: dict[str, BBox] = {}
    dimension_bboxes: dict[str, BBox] = {}
    dimension_segments: dict[str, list[list[float]]] = {}
    label_bboxes["volume_label"] = draw_readout_centered(
        ctx,
        f"V = {format_pi_multiple(problem.volume_pi_multiple or 0)}",
        (120.0, 92.0),
        small=False,
        backed=True,
    )
    _add_dimension(
        ctx,
        label_bboxes=label_bboxes,
        dimension_bboxes=dimension_bboxes,
        dimension_segments=dimension_segments,
        label_key="total_height_label",
        witness_key="total_height_segment",
        start=(points["bottom_left"][0] - 50.0, points["apex"][1]),
        end=(points["bottom_left"][0] - 50.0, points["bottom_left"][1]),
        label=f"H = {format_measure(problem.total_height or 0)}",
        label_offset=(-42.0, 0.0),
    )
    _add_dimension(
        ctx,
        label_bboxes=label_bboxes,
        dimension_bboxes=dimension_bboxes,
        dimension_segments=dimension_segments,
        label_key="cone_height_label",
        witness_key="cone_height_segment",
        start=(points["cyl_top_right"][0] + 38.0, points["apex"][1]),
        end=(points["cyl_top_right"][0] + 38.0, points["cyl_top_right"][1]),
        label=f"c = {format_measure(problem.cone_height or 0)}",
        label_offset=(36.0, 0.0),
    )
    _add_dimension(
        ctx,
        label_bboxes=label_bboxes,
        dimension_bboxes=dimension_bboxes,
        dimension_segments=dimension_segments,
        label_key="target_radius_label",
        witness_key="target_radius_segment",
        start=points["base_center"],
        end=points["base_right"],
        label="r = ?",
        label_offset=(0.0, 72.0),
        target=True,
    )
    return _with_scene_payload(
        ctx=ctx,
        problem=problem,
        solid_bbox=solid_bbox,
        label_bboxes=label_bboxes,
        annotation_bboxes=dimension_bboxes,
        dimension_segments=dimension_segments,
        annotation_roles=(
            "target_radius_segment",
            "total_height_segment",
            "cone_height_segment",
        ),
    )


def render_cylinder_cone_height(ctx: RenderContext, problem: SolidFormulaProblem) -> RenderedSolidFormulaScene:
    """Render labels needed to solve the cylinder-cone cylinder-height formula."""

    solid_bbox, points = _draw_cylinder_cone_body(ctx)
    label_bboxes: dict[str, BBox] = {}
    dimension_bboxes: dict[str, BBox] = {}
    dimension_segments: dict[str, list[list[float]]] = {}
    label_bboxes["volume_label"] = draw_readout_centered(
        ctx,
        f"V = {format_pi_multiple(problem.volume_pi_multiple or 0)}",
        (120.0, 92.0),
        small=False,
        backed=True,
    )
    _add_dimension(
        ctx,
        label_bboxes=label_bboxes,
        dimension_bboxes=dimension_bboxes,
        dimension_segments=dimension_segments,
        label_key="radius_label",
        witness_key="radius_segment",
        start=points["base_center"],
        end=points["base_right"],
        label=f"r = {format_measure(problem.radius or 0)}",
        label_offset=(0.0, 72.0),
    )
    _add_dimension(
        ctx,
        label_bboxes=label_bboxes,
        dimension_bboxes=dimension_bboxes,
        dimension_segments=dimension_segments,
        label_key="cone_height_label",
        witness_key="cone_height_segment",
        start=(points["cyl_top_right"][0] + 38.0, points["apex"][1]),
        end=(points["cyl_top_right"][0] + 38.0, points["cyl_top_right"][1]),
        label=f"c = {format_measure(problem.cone_height or 0)}",
        label_offset=(36.0, 0.0),
    )
    _add_dimension(
        ctx,
        label_bboxes=label_bboxes,
        dimension_bboxes=dimension_bboxes,
        dimension_segments=dimension_segments,
        label_key="target_cylinder_height_label",
        witness_key="target_cylinder_height_segment",
        start=(points["bottom_right"][0] + 42.0, points["cyl_top_right"][1]),
        end=(points["bottom_right"][0] + 42.0, points["bottom_right"][1]),
        label="x = ?",
        label_offset=(38.0, 0.0),
        target=True,
    )
    return _with_scene_payload(
        ctx=ctx,
        problem=problem,
        solid_bbox=solid_bbox,
        label_bboxes=label_bboxes,
        annotation_bboxes=dimension_bboxes,
        dimension_segments=dimension_segments,
        annotation_roles=(
            "target_cylinder_height_segment",
            "radius_segment",
            "cone_height_segment",
        ),
    )


def render_prism_pyramid(ctx: RenderContext, problem: SolidFormulaProblem) -> RenderedSolidFormulaScene:
    """Render a rectangular prism with a pyramid cap."""

    front = ((236.0, 232.0), (516.0, 232.0), (516.0, 424.0), (236.0, 424.0))
    offset = (96.0, -62.0)
    back = tuple((x + offset[0], y + offset[1]) for x, y in front)
    front_tl, front_tr, front_br, front_bl = front
    back_tl, back_tr, back_br, back_bl = back
    roof_apex = (
        (front_tl[0] + front_tr[0] + back_tl[0] + back_tr[0]) / 4.0,
        90.0,
    )
    roof_center = (
        (front_tl[0] + front_tr[0] + back_tl[0] + back_tr[0]) / 4.0,
        (front_tl[1] + front_tr[1] + back_tl[1] + back_tr[1]) / 4.0,
    )

    ctx.draw.polygon((front_tl, front_tr, back_tr, back_tl), fill=ctx.secondary_fill_color)
    ctx.draw.polygon((front_tr, front_br, back_br, back_tr), fill=ctx.fill_color)
    ctx.draw.polygon((front_bl, front_br, back_br, back_bl), fill=ctx.secondary_fill_color)
    ctx.draw.polygon(front, fill=ctx.fill_color)
    for start, end in (
        (front_tl, front_tr),
        (front_tr, front_br),
        (front_br, front_bl),
        (front_bl, front_tl),
        (back_tl, back_tr),
        (back_tr, back_br),
        (back_br, back_bl),
        (back_bl, back_tl),
        (front_tl, back_tl),
        (front_tr, back_tr),
        (front_br, back_br),
        (front_bl, back_bl),
    ):
        ctx.draw.line([start, end], fill=ctx.line_color, width=ctx.line_width)
    roof_faces = (
        (front_tl, front_tr, roof_apex),
        (front_tr, back_tr, roof_apex),
        (back_tr, back_tl, roof_apex),
        (back_tl, front_tl, roof_apex),
    )
    for face in roof_faces:
        ctx.draw.polygon(face, fill=ctx.secondary_fill_color)
    for point in (front_tl, front_tr, back_tr, back_tl):
        ctx.draw.line([roof_apex, point], fill=ctx.line_color, width=ctx.line_width)
    _draw_dashed_line(ctx, roof_apex, roof_center, fill=ctx.accent_color, width=3)

    label_bboxes: dict[str, BBox] = {}
    dimension_bboxes: dict[str, BBox] = {}
    dimension_segments: dict[str, list[list[float]]] = {}
    label_bboxes["volume_label"] = draw_readout_centered(
        ctx,
        f"V = {format_measure(problem.volume or 0)}",
        (154.0, 94.0),
        small=False,
        backed=True,
    )
    _add_dimension(
        ctx,
        label_bboxes=label_bboxes,
        dimension_bboxes=dimension_bboxes,
        dimension_segments=dimension_segments,
        label_key="known_length_label",
        witness_key="base_length_segment",
        start=(front_bl[0], front_bl[1] + 36.0),
        end=(front_br[0], front_br[1] + 36.0),
        label=f"l = {format_measure(problem.side_a or 0)}",
        label_offset=(0.0, 28.0),
    )
    _add_dimension(
        ctx,
        label_bboxes=label_bboxes,
        dimension_bboxes=dimension_bboxes,
        dimension_segments=dimension_segments,
        label_key="known_width_label",
        witness_key="base_width_segment",
        start=(front_br[0] + 18.0, front_br[1] + 18.0),
        end=(back_br[0] + 18.0, back_br[1] + 18.0),
        label=f"w = {format_measure(problem.side_b or 0)}",
        label_offset=(30.0, 20.0),
    )
    _add_dimension(
        ctx,
        label_bboxes=label_bboxes,
        dimension_bboxes=dimension_bboxes,
        dimension_segments=dimension_segments,
        label_key="pyramid_height_label",
        witness_key="pyramid_height_segment",
        start=(roof_center[0] + 32.0, roof_apex[1]),
        end=(roof_center[0] + 32.0, roof_center[1]),
        label=f"p = {format_measure(problem.pyramid_height or 0)}",
        label_offset=(36.0, 0.0),
    )
    _add_dimension(
        ctx,
        label_bboxes=label_bboxes,
        dimension_bboxes=dimension_bboxes,
        dimension_segments=dimension_segments,
        label_key="target_prism_height_label",
        witness_key="target_prism_height_segment",
        start=(front_tl[0] - 34.0, front_tl[1]),
        end=(front_bl[0] - 34.0, front_bl[1]),
        label="x = ?",
        label_offset=(-34.0, 0.0),
        target=True,
    )
    solid_bbox = bbox_from_points(tuple(front) + tuple(back) + (roof_apex,), width=ctx.width, height=ctx.height, pad=54.0)
    return _with_scene_payload(
        ctx=ctx,
        problem=problem,
        solid_bbox=solid_bbox,
        label_bboxes=label_bboxes,
        annotation_bboxes=dimension_bboxes,
        dimension_segments=dimension_segments,
        annotation_roles=(
            "target_prism_height_segment",
            "base_length_segment",
            "base_width_segment",
            "pyramid_height_segment",
        ),
    )


def render_house_prism(ctx: RenderContext, problem: SolidFormulaProblem) -> RenderedSolidFormulaScene:
    """Render a house-shaped prism with rectangular wall and triangular roof."""

    front_a = (260.0, 420.0)
    front_b = (520.0, 420.0)
    front_c = (520.0, 262.0)
    front_d = (390.0, 148.0)
    front_e = (260.0, 262.0)
    offset = (116.0, -76.0)
    back_a = (front_a[0] + offset[0], front_a[1] + offset[1])
    back_b = (front_b[0] + offset[0], front_b[1] + offset[1])
    back_c = (front_c[0] + offset[0], front_c[1] + offset[1])
    back_d = (front_d[0] + offset[0], front_d[1] + offset[1])
    back_e = (front_e[0] + offset[0], front_e[1] + offset[1])
    front_poly = (front_a, front_b, front_c, front_d, front_e)
    back_poly = (back_a, back_b, back_c, back_d, back_e)

    ctx.draw.polygon((front_a, front_b, back_b, back_a), fill=ctx.secondary_fill_color)
    ctx.draw.polygon((front_b, front_c, back_c, back_b), fill=ctx.fill_color)
    ctx.draw.polygon((front_c, front_d, back_d, back_c), fill=ctx.secondary_fill_color)
    ctx.draw.polygon(front_poly, fill=ctx.fill_color)
    for start, end in (
        (front_a, front_b),
        (front_b, front_c),
        (front_c, front_d),
        (front_d, front_e),
        (front_e, front_a),
        (back_a, back_b),
        (back_b, back_c),
        (back_c, back_d),
        (back_d, back_e),
        (back_e, back_a),
        (front_a, back_a),
        (front_b, back_b),
        (front_c, back_c),
        (front_d, back_d),
        (front_e, back_e),
    ):
        ctx.draw.line([start, end], fill=ctx.line_color, width=ctx.line_width)
    roof_foot = (front_d[0], front_c[1])
    _draw_dashed_line(ctx, front_d, roof_foot, fill=ctx.accent_color, width=3)

    label_bboxes: dict[str, BBox] = {}
    dimension_bboxes: dict[str, BBox] = {}
    dimension_segments: dict[str, list[list[float]]] = {}
    label_bboxes["volume_label"] = draw_readout_centered(
        ctx,
        f"V = {format_measure(problem.volume or 0)}",
        (154.0, 94.0),
        small=False,
        backed=True,
    )
    _add_dimension(
        ctx,
        label_bboxes=label_bboxes,
        dimension_bboxes=dimension_bboxes,
        dimension_segments=dimension_segments,
        label_key="triangle_base_label",
        witness_key="base_width_segment",
        start=(front_a[0], front_a[1] + 34.0),
        end=(front_b[0], front_b[1] + 34.0),
        label=f"b = {format_measure(problem.triangle_base or 0)}",
        label_offset=(0.0, 28.0),
    )
    _add_dimension(
        ctx,
        label_bboxes=label_bboxes,
        dimension_bboxes=dimension_bboxes,
        dimension_segments=dimension_segments,
        label_key="wall_height_label",
        witness_key="wall_height_segment",
        start=(front_a[0] - 34.0, front_e[1]),
        end=(front_a[0] - 34.0, front_a[1]),
        label=f"h = {format_measure(problem.wall_height or 0)}",
        label_offset=(-36.0, 0.0),
    )
    _add_dimension(
        ctx,
        label_bboxes=label_bboxes,
        dimension_bboxes=dimension_bboxes,
        dimension_segments=dimension_segments,
        label_key="roof_height_label",
        witness_key="roof_height_segment",
        start=(roof_foot[0] + 34.0, front_d[1]),
        end=(roof_foot[0] + 34.0, roof_foot[1]),
        label=f"t = {format_measure(problem.roof_height or 0)}",
        label_offset=(64.0, 0.0),
    )
    _add_dimension(
        ctx,
        label_bboxes=label_bboxes,
        dimension_bboxes=dimension_bboxes,
        dimension_segments=dimension_segments,
        label_key="target_length_label",
        witness_key="target_length_segment",
        start=(front_b[0] + 20.0, front_b[1] + 16.0),
        end=(back_b[0] + 20.0, back_b[1] + 16.0),
        label="L = ?",
        label_offset=(34.0, 20.0),
        target=True,
    )
    solid_bbox = bbox_from_points(front_poly + back_poly, width=ctx.width, height=ctx.height, pad=54.0)
    return _with_scene_payload(
        ctx=ctx,
        problem=problem,
        solid_bbox=solid_bbox,
        label_bboxes=label_bboxes,
        annotation_bboxes=dimension_bboxes,
        dimension_segments=dimension_segments,
        annotation_roles=(
            "target_length_segment",
            "base_width_segment",
            "wall_height_segment",
            "roof_height_segment",
        ),
    )


__all__ = [
    "create_render_context",
    "render_cylinder_cone_height",
    "render_cylinder_cone_radius",
    "render_house_prism",
    "render_prism_pyramid",
]
