"""Rendering for incircle-tangent diagrams."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping

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
from trace_tasks.tasks.shared.text_rendering import load_font, symbol_safe_font_for_text

from .defaults import SCENE_ID
from .measurements import triangle_layout
from .state import Color, IncircleDiagramSpec, Point, RenderContext, RenderedIncircleScene


def _transform_layout(layout: Mapping[str, Point | float], ctx: RenderContext) -> Dict[str, Point | float]:
    points = [layout[key] for key in ("A", "B", "C")]
    xs = [float(point[0]) for point in points if isinstance(point, tuple)]
    ys = [float(point[1]) for point in points if isinstance(point, tuple)]
    margin_x = 112.0
    margin_top = 86.0
    margin_bottom = 98.0
    scale = min(
        (float(ctx.width) - 2.0 * margin_x) / max(1.0, max(xs) - min(xs)),
        (float(ctx.height) - margin_top - margin_bottom) / max(1.0, max(ys) - min(ys)),
    )
    left = (float(ctx.width) - (max(xs) - min(xs)) * scale) / 2.0
    bottom = float(ctx.height) - margin_bottom

    def tx(point: Point) -> Point:
        return (
            left + (float(point[0]) - min(xs)) * scale,
            bottom - (float(point[1]) - min(ys)) * scale,
        )

    transformed: Dict[str, Point | float] = {}
    for key, value in layout.items():
        transformed[key] = tx(value) if isinstance(value, tuple) else float(value) * scale
    return transformed


def _unit_vector(start: Point, end: Point) -> Point:
    dx = float(end[0]) - float(start[0])
    dy = float(end[1]) - float(start[1])
    length = math.hypot(dx, dy)
    if length <= 1e-9:
        return (0.0, -1.0)
    return (dx / length, dy / length)


def _draw_tick(ctx: RenderContext, start: Point, end: Point, *, count: int, color: Color) -> None:
    ux, uy = _unit_vector(start, end)
    nx, ny = -uy, ux
    mid = ((float(start[0]) + float(end[0])) / 2.0, (float(start[1]) + float(end[1])) / 2.0)
    spacing = 5.0
    for index in range(int(count)):
        shift = (float(index) - (float(count) - 1.0) / 2.0) * spacing
        cx = mid[0] + ux * shift
        cy = mid[1] + uy * shift
        ctx.draw.line(
            [(cx - nx * 7.0, cy - ny * 7.0), (cx + nx * 7.0, cy + ny * 7.0)],
            fill=color,
            width=2,
        )


def _draw_contact_point_label(ctx: RenderContext, *, label: str, point: Point, incenter: Point) -> None:
    """Draw a labeled tangent-contact point with the label offset away from the incircle."""

    ux, uy = _unit_vector(incenter, point)
    label_center = (float(point[0]) + ux * 22.0, float(point[1]) + uy * 22.0)
    ctx.draw.ellipse((point[0] - 4.5, point[1] - 4.5, point[0] + 4.5, point[1] + 4.5), fill=ctx.accent_color)
    draw_label(ctx, label, label_center, small=True)


def _text_bbox_at(ctx: RenderContext, text: str, center: Point, *, small: bool = True) -> tuple[float, float, float, float]:
    """Return the unclamped padded bbox for centered label text."""

    font = symbol_safe_font_for_text(str(text), ctx.small_font if bool(small) else ctx.font)
    stroke_width = max(0, int(ctx.label_stroke_width))
    bbox = ctx.draw.textbbox((0, 0), str(text), font=font, stroke_width=stroke_width)
    text_w = float(bbox[2] - bbox[0])
    text_h = float(bbox[3] - bbox[1])
    left = float(center[0]) - text_w / 2.0
    top = float(center[1]) - text_h / 2.0
    return (left - 4.0, top - 4.0, left + text_w + 4.0, top + text_h + 4.0)


def _bbox_inside_canvas(ctx: RenderContext, bbox: tuple[float, float, float, float], *, margin: float = 8.0) -> bool:
    """Return whether a label bbox fits inside the canvas with a small margin."""

    return (
        float(bbox[0]) >= float(margin)
        and float(bbox[1]) >= float(margin)
        and float(bbox[2]) <= float(ctx.width) - float(margin)
        and float(bbox[3]) <= float(ctx.height) - float(margin)
    )


def _clamp_label_center(ctx: RenderContext, text: str, center: Point, *, small: bool = True, margin: float = 12.0) -> Point:
    """Move a label center just enough that the rendered label remains on-canvas."""

    bbox = _text_bbox_at(ctx, text, center, small=small)
    dx = 0.0
    dy = 0.0
    if bbox[0] < margin:
        dx = float(margin) - bbox[0]
    elif bbox[2] > float(ctx.width) - margin:
        dx = float(ctx.width) - float(margin) - bbox[2]
    if bbox[1] < margin:
        dy = float(margin) - bbox[1]
    elif bbox[3] > float(ctx.height) - margin:
        dy = float(ctx.height) - float(margin) - bbox[3]
    return (float(center[0]) + dx, float(center[1]) + dy)


def _bbox_overlap_area(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    """Return the overlap area between two unclamped bboxes."""

    x0 = max(float(left[0]), float(right[0]))
    y0 = max(float(left[1]), float(right[1]))
    x1 = min(float(left[2]), float(right[2]))
    y1 = min(float(left[3]), float(right[3]))
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _legend_centers(ctx: RenderContext, *, corner: str) -> tuple[Point, Point, Point]:
    """Return three row centers for a compact tangent-equality legend."""

    x = 126.0 if "left" in corner else float(ctx.width) - 126.0
    if "top" in corner:
        ys = (36.0, 60.0, 84.0)
    else:
        ys = (float(ctx.height) - 84.0, float(ctx.height) - 60.0, float(ctx.height) - 36.0)
    return tuple((x, y) for y in ys)  # type: ignore[return-value]


def _draw_tangent_legend(
    ctx: RenderContext,
    *,
    rows: tuple[tuple[str, str], ...],
    occupied_bbox: tuple[float, float, float, float],
) -> dict[str, tuple[float, float, float, float]]:
    """Draw all tangent lengths as equality rows in the clearest image corner."""

    ranked_candidates: list[tuple[float, str, tuple[Point, ...]]] = []
    for corner in ("top_left", "top_right", "bottom_left", "bottom_right"):
        centers = _legend_centers(ctx, corner=corner)
        bboxes = tuple(
            _text_bbox_at(ctx, text, center, small=True)
            for (_key, text), center in zip(rows, centers)
        )
        outside_penalty = sum(0.0 if _bbox_inside_canvas(ctx, bbox, margin=10.0) else 10_000.0 for bbox in bboxes)
        overlap_penalty = sum(_bbox_overlap_area(bbox, occupied_bbox) for bbox in bboxes)
        ranked_candidates.append((outside_penalty + overlap_penalty, corner, centers))
    _score, _corner, centers = min(ranked_candidates, key=lambda item: item[0])
    label_bboxes: dict[str, tuple[float, float, float, float]] = {}
    for (key, text), center in zip(rows, centers):
        label_bboxes[key] = draw_label(ctx, text, _clamp_label_center(ctx, text, center, small=True), small=True)
    return label_bboxes


def _draw_center_label(ctx: RenderContext, *, center: Point, tangent_point: Point) -> tuple[float, float, float, float]:
    """Draw the incenter point and offset label after radius geometry is visible."""

    ctx.draw.ellipse((center[0] - 3.8, center[1] - 3.8, center[0] + 3.8, center[1] + 3.8), fill=ctx.line_color)
    ux, uy = _unit_vector(tangent_point, center)
    px, py = -uy, ux
    label_center = (float(center[0]) + px * 22.0 + ux * 12.0, float(center[1]) + py * 22.0 + uy * 12.0)
    return draw_label(ctx, "O", _clamp_label_center(ctx, "O", label_center, small=True), small=True)


def _unknown_radius_label_center(*, center: Point, tangent_point: Point) -> Point:
    """Place r=? near the radius segment without covering the incenter label."""

    ux, uy = _unit_vector(center, tangent_point)
    px, py = -uy, ux
    midpoint = ((float(center[0]) + float(tangent_point[0])) / 2.0, (float(center[1]) + float(tangent_point[1])) / 2.0)
    return (
        midpoint[0] - ux * 7.0 - px * 22.0,
        midpoint[1] - uy * 7.0 - py * 22.0,
    )


def _make_render_context(
    *,
    random_namespace: str,
    instance_seed: int,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
) -> tuple[RenderContext, Dict[str, Any]]:
    """Create canvas, style, font, and transform state for one diagram."""

    rng = spawn_rng(int(instance_seed), f"{random_namespace}.render")
    width = int(params.get("canvas_width", group_default(render_defaults, "canvas_width", 760)))
    height = int(params.get("canvas_height", group_default(render_defaults, "canvas_height", 560)))
    image, background_meta, diagram_style, diagram_style_meta = prepare_geometry_diagram_style_and_background(
        canvas_width=width,
        canvas_height=height,
        scene_id=SCENE_ID,
        instance_seed=int(instance_seed),
        params=params,
        namespace_suffix="incircle_tangents_background",
    )
    shape_style = sample_geometry_shape_style(
        rng,
        params=params,
        render_defaults=render_defaults,
        anchor_colors=extract_background_anchor_colors(background_meta),
    )
    accents: tuple[Color, ...] = (
        (27, 113, 191),
        (189, 91, 37),
        (111, 92, 190),
        (30, 132, 92),
    )
    accent_rng = spawn_rng(int(instance_seed), f"{random_namespace}.accent")
    accent_color = uniform_choice(accent_rng, accents)
    fill_color = (
        min(255, int(accent_color[0] * 0.18 + 255 * 0.82)),
        min(255, int(accent_color[1] * 0.18 + 255 * 0.82)),
        min(255, int(accent_color[2] * 0.18 + 255 * 0.82)),
    )
    font_size = int(params.get("label_font_size", group_default(render_defaults, "label_font_size", 22)))
    small_font_size = int(
        params.get("small_label_font_size", group_default(render_defaults, "small_label_font_size", 18))
    )
    line_width = int(params.get("line_width", group_default(render_defaults, "line_width", 4)))
    ctx = RenderContext(
        rng=rng,
        image=image,
        draw=ImageDraw.Draw(image),
        width=width,
        height=height,
        line_color=shape_style.line_color,
        label_color=shape_style.label_color,
        label_stroke_color=shape_style.label_stroke_color,
        accent_color=accent_color,
        fill_color=fill_color,
        line_width=max(2, int(line_width)),
        label_stroke_width=max(
            0,
            int(params.get("label_stroke_width", group_default(render_defaults, "label_stroke_width", 0))),
        ),
        font=load_font(max(12, int(font_size)), bold=False),
        small_font=load_font(max(10, int(small_font_size)), bold=False),
        scene_transform=LazySceneTransform(
            rng,
            params=params,
            render_defaults=render_defaults,
            canvas_width=width,
            canvas_height=height,
        ),
    )
    render_meta = {
        "background_style": dict(background_meta),
        "technical_diagram_style": geometry_diagram_style_metadata(diagram_style),
        "technical_diagram_style_resolution": dict(diagram_style_meta),
        "shape_style": shape_style.to_trace_dict(),
        "line_width": int(ctx.line_width),
        "label_stroke_width": int(ctx.label_stroke_width),
        "label_font_size": int(font_size),
        "small_label_font_size": int(small_font_size),
        "accent_color": list(accent_color),
        "fill_color": list(fill_color),
    }
    return ctx, render_meta


def _render_incircle_scene(ctx: RenderContext, spec: IncircleDiagramSpec) -> RenderedIncircleScene:
    """Draw the triangle grammar and return labels projected after final transform."""

    raw_layout = triangle_layout(spec)
    layout = _transform_layout(raw_layout, ctx)
    point_layout = {key: value for key, value in layout.items() if isinstance(value, tuple)}
    ctx.scene_transform.resolve(tuple(point_layout.values()))
    point_layout = ctx.scene_transform.keyed_points(point_layout)
    for key, value in point_layout.items():
        layout[key] = value
    inradius_px = float(layout["inradius"]) * float(ctx.scene_transform.transform.scale)

    a = point_layout["A"]
    b = point_layout["B"]
    c = point_layout["C"]
    d = point_layout["D"]
    e = point_layout["E"]
    f = point_layout["F"]
    o = point_layout["O"]

    ctx.draw.polygon([a, b, c], fill=ctx.fill_color)
    ctx.draw.line([a, b, c, a], fill=ctx.line_color, width=ctx.line_width, joint="curve")
    ctx.draw.ellipse(
        (o[0] - inradius_px, o[1] - inradius_px, o[0] + inradius_px, o[1] + inradius_px),
        outline=ctx.accent_color,
        width=max(2, ctx.line_width - 1),
    )
    for label, point in (("D", d), ("E", e), ("F", f)):
        _draw_contact_point_label(ctx, label=label, point=point, incenter=o)
    for label, point, offset in (
        ("A", a, (-20.0, 22.0)),
        ("B", b, (20.0, 22.0)),
        ("C", c, (0.0, -24.0)),
    ):
        ctx.draw.ellipse((point[0] - 3.5, point[1] - 3.5, point[0] + 3.5, point[1] + 3.5), fill=ctx.line_color)
        draw_label(ctx, label, (point[0] + offset[0], point[1] + offset[1]), small=True)

    _draw_tick(ctx, a, d, count=1, color=ctx.accent_color)
    _draw_tick(ctx, a, f, count=1, color=ctx.accent_color)
    _draw_tick(ctx, b, d, count=2, color=ctx.accent_color)
    _draw_tick(ctx, b, e, count=2, color=ctx.accent_color)
    _draw_tick(ctx, c, e, count=3, color=ctx.accent_color)
    _draw_tick(ctx, c, f, count=3, color=ctx.accent_color)

    triangle_bbox = bbox_from_points((a, b, c), width=ctx.width, height=ctx.height, pad=10.0)
    incircle_bbox = pad_bbox(
        (o[0] - inradius_px, o[1] - inradius_px, o[0] + inradius_px, o[1] + inradius_px),
        6.0,
        width=ctx.width,
        height=ctx.height,
    )
    occupied_bbox = (
        min(float(triangle_bbox[0]), float(incircle_bbox[0])),
        min(float(triangle_bbox[1]), float(incircle_bbox[1])),
        max(float(triangle_bbox[2]), float(incircle_bbox[2])),
        max(float(triangle_bbox[3]), float(incircle_bbox[3])),
    )
    label_bboxes: Dict[str, tuple[float, float, float, float]] = _draw_tangent_legend(
        ctx,
        rows=(
            ("AD_AF", f"AD = AF = {fmt_measure(spec.tangent_a)}"),
            ("BD_BE", f"BD = BE = {fmt_measure(spec.tangent_b)}"),
            ("CE_CF", f"CE = CF = {fmt_measure(spec.tangent_c)}"),
        ),
        occupied_bbox=occupied_bbox,
    )
    if spec.show_area_label:
        label_bboxes["area"] = draw_label(ctx, f"Area={fmt_measure(spec.displayed_area)}", (596.0, 96.0), small=True)
    if spec.show_radius_segment:
        ctx.draw.line([o, d], fill=ctx.accent_color, width=max(2, ctx.line_width - 1))
    if spec.unknown_label:
        if spec.show_radius_segment:
            label_center = _unknown_radius_label_center(center=o, tangent_point=d)
        else:
            label_center = (o[0], o[1] - 62.0)
        label_bboxes["unknown"] = draw_label(
            ctx,
            spec.unknown_label,
            _clamp_label_center(ctx, spec.unknown_label, label_center, small=True),
            small=True,
        )
    label_bboxes["O"] = _draw_center_label(ctx, center=o, tangent_point=d)
    scene_entities = (
        {
            "entity_id": "triangle_ABC",
            "entity_type": "triangle",
            "vertices": {
                "A": [round(a[0], 3), round(a[1], 3)],
                "B": [round(b[0], 3), round(b[1], 3)],
                "C": [round(c[0], 3), round(c[1], 3)],
            },
            "tangent_points": {
                "D": [round(d[0], 3), round(d[1], 3)],
                "E": [round(e[0], 3), round(e[1], 3)],
                "F": [round(f[0], 3), round(f[1], 3)],
            },
            "side_lengths": {
                "AB": round(float(spec.side_ab), 3),
                "BC": round(float(spec.side_bc), 3),
                "CA": round(float(spec.side_ca), 3),
            },
            "bbox": bbox_to_list(triangle_bbox),
        },
        {
            "entity_id": "incircle",
            "entity_type": "circle",
            "center": [round(o[0], 3), round(o[1], 3)],
            "radius_px": round(inradius_px, 3),
            "radius_units": round(float(spec.inradius), 3),
            "bbox": bbox_to_list(incircle_bbox),
        },
    )
    render_map = {
        "coord_space": "pixel",
        "points": {
            key: [round(value[0], 3), round(value[1], 3)]
            for key, value in point_layout.items()
        },
        "incircle_radius_px": round(inradius_px, 3),
        "label_bboxes": {key: bbox_to_list(value) for key, value in label_bboxes.items()},
    }
    return RenderedIncircleScene(
        image=ctx.image,
        label_bboxes=dict(label_bboxes),
        annotation_points={key: tuple(value) for key, value in point_layout.items()},
        scene_entities=scene_entities,
        render_map=render_map,
        annotation_roles=tuple(spec.annotation_roles),
    )


def render_incircle_scene_with_retries(
    *,
    random_namespace: str,
    spec: IncircleDiagramSpec,
    instance_seed: int,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    max_attempts: int,
) -> tuple[RenderedIncircleScene, Dict[str, Any]]:
    """Render one incircle scene, retrying if a sampled layout fails."""

    last_error: Exception | None = None
    for _attempt in range(max(1, int(max_attempts))):
        try:
            ctx, render_meta = _make_render_context(
                random_namespace=str(random_namespace),
                instance_seed=int(instance_seed),
                params=params,
                render_defaults=render_defaults,
            )
            rendered = _render_incircle_scene(ctx, spec)
            render_meta = dict(render_meta)
            render_meta["single_object_scene_rotation"] = ctx.scene_transform.metadata()
            return rendered, render_meta
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"failed to render incircle_tangents scene for {random_namespace}") from last_error


__all__ = ["render_incircle_scene_with_retries"]
