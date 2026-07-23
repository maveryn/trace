"""Registered vector renderers for reusable illustration objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence, Tuple

from .object_schema import BBox
from .render_geometry import scale_bbox as _scale_bbox, scale_points as _scale_points
from .style_registry import style_outline_params
from .vector_person_rendering import VectorPersonRenderResult, draw_vector_person


RGB = Tuple[int, int, int]


@dataclass(frozen=True)
class VectorObjectRenderResult:
    """Rendered vector object geometry emitted by registered renderers."""

    bbox_xyxy: BBox
    parts: Tuple[Mapping[str, Any], ...] = ()
    visual_attributes: Mapping[str, Any] = field(default_factory=dict)
    support_items: Tuple[Mapping[str, Any], ...] = ()


PERSON_RENDERER_IDS = frozenset(
    {
        "construction_worker",
        "object_library_person",
        "park_person",
    }
)
BENCH_RENDERER_IDS = frozenset({"fixture_bench"})
PARK_EQUIPMENT_RENDERER_IDS = frozenset({"park_equipment"})
CONSTRUCTION_MATERIAL_RENDERER_IDS = frozenset({"construction_material"})
CONSTRUCTION_EQUIPMENT_RENDERER_IDS = frozenset({"construction_equipment"})
INDOOR_FURNITURE_RENDERER_IDS = frozenset({"indoor_furniture"})
INDOOR_SURFACE_RENDERER_IDS = frozenset({"indoor_surface"})
INDOOR_CONTAINER_RENDERER_IDS = frozenset({"indoor_container"})


def render_registered_vector_object(spec: Any, context: Any) -> VectorPersonRenderResult | VectorObjectRenderResult | None:
    """Draw a vector object with a registered renderer, if one is selected."""

    renderer_id = _renderer_id(spec)
    if renderer_id in PERSON_RENDERER_IDS:
        return _render_person(spec, context, renderer_id)
    if renderer_id in BENCH_RENDERER_IDS:
        return _render_bench(spec, context, renderer_id)
    if renderer_id in PARK_EQUIPMENT_RENDERER_IDS:
        return _render_park_equipment(spec, context, renderer_id)
    if renderer_id in CONSTRUCTION_MATERIAL_RENDERER_IDS:
        return _render_construction_material(spec, context, renderer_id)
    if renderer_id in CONSTRUCTION_EQUIPMENT_RENDERER_IDS:
        return _render_construction_equipment(spec, context, renderer_id)
    if renderer_id in INDOOR_FURNITURE_RENDERER_IDS:
        return _render_indoor_furniture(spec, context, renderer_id)
    if renderer_id in INDOOR_SURFACE_RENDERER_IDS:
        return _render_indoor_surface(spec, context, renderer_id)
    if renderer_id in INDOOR_CONTAINER_RENDERER_IDS:
        return _render_indoor_container(spec, context, renderer_id)
    return None


def _render_person(spec: Any, context: Any, renderer_id: str) -> VectorPersonRenderResult:
    if getattr(context, "draw", None) is None:
        raise ValueError("registered vector person rendering requires RenderContext.draw")
    bbox = getattr(spec, "bbox_xyxy", None)
    if bbox is None:
        raise ValueError("registered vector person rendering requires IllustrationObjectSpec.bbox_xyxy")
    visual = _mapping(getattr(spec, "visual_attributes", {}))
    semantic = _mapping(getattr(spec, "semantic_attributes", {}))
    return draw_vector_person(
        context.draw,
        object_id=str(getattr(spec, "object_id")),
        object_type=str(getattr(spec, "object_type")),
        bbox_xyxy=tuple(float(value) for value in bbox),  # type: ignore[arg-type]
        renderer_id=str(renderer_id),
        renderer_variant_id=_renderer_variant_id(spec, visual, semantic),
        style_id=str(visual.get("style_id", getattr(context, "style_id", "flat_vector"))),
        render_scale=int(getattr(context, "render_scale", 1)),
        primary_color_rgb=_rgb(visual.get("primary_color_rgb", getattr(context, "primary_color_rgb", (72, 115, 166)))),
        accent_color_rgb=_rgb(visual.get("accent_color_rgb", getattr(context, "accent_color_rgb", (245, 190, 88)))),
        skin_color_rgb=_optional_rgb(visual.get("skin_color_rgb", visual.get("skin_rgb"))),
        gender_id=str(visual.get("gender_id", "")) if visual.get("gender_id") is not None else None,
        person_variant_id=str(visual.get("person_variant_id", getattr(spec, "variant_id", "") or "adult")),
        hard_hat_color_rgb=_optional_rgb(visual.get("hard_hat_color_rgb")),
        vest_color_rgb=_optional_rgb(visual.get("vest_color_rgb")),
        tool_type=str(semantic.get("tool_type", "")) if semantic.get("tool_type") else None,
    )


def _render_bench(spec: Any, context: Any, renderer_id: str) -> VectorObjectRenderResult:
    draw, box, visual, semantic = _required_draw_box(spec, context)
    scale = max(1, int(getattr(context, "render_scale", 1)))
    style_id = str(visual.get("style_id", getattr(context, "style_id", "flat_vector")))
    outline_style, line_width, _shadow = style_outline_params(style_id)
    outline = _rgb(visual.get("outline_color_rgb", outline_style or (75, 56, 43)))
    wood = _rgb(visual.get("wood_color_rgb", visual.get("primary_color_rgb", (146, 92, 58))))
    x0, y0, x1, y1 = box
    height = y1 - y0
    _rect(draw, (x0, y0 + 0.20 * height, x1, y0 + 0.44 * height), fill=wood, outline=outline, width=max(1, int(line_width)), scale=scale, radius=5)
    _rect(draw, (x0 + 8.0, y0 + 0.52 * height, x1 - 8.0, y0 + 0.72 * height), fill=wood, outline=outline, width=max(1, int(line_width)), scale=scale, radius=5)
    parts: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for lx in (x0 + 18.0, x1 - 30.0):
        leg = (lx, y0 + 0.70 * height, lx + 10.0, y1)
        _rect(draw, leg, fill=outline, outline=None, width=1, scale=scale, radius=2)
        _part(parts, counts, str(getattr(spec, "object_id")), "leg", leg)
    return _result(
        box,
        renderer_id=renderer_id,
        renderer_variant_id=_renderer_variant_id(spec, visual, semantic) or "bench",
        visual_attributes={
            "wood_color_rgb": [int(v) for v in wood],
            "outline_color_rgb": [int(v) for v in outline],
            "style_id": style_id,
        },
        parts=parts,
    )


def _render_park_equipment(spec: Any, context: Any, renderer_id: str) -> VectorObjectRenderResult:
    draw, box, visual, semantic = _required_draw_box(spec, context)
    scale = max(1, int(getattr(context, "render_scale", 1)))
    equipment_type = _renderer_variant_id(spec, visual, semantic) or str(semantic.get("equipment_type", "climbing_frame"))
    outline = _rgb(visual.get("outline_color_rgb", (61, 71, 82)))
    primary = _rgb(visual.get("primary_color_rgb", (216, 96, 82)))
    accent = _rgb(visual.get("accent_color_rgb", (229, 168, 67)))
    x0, y0, x1, y1 = box
    width = x1 - x0
    height = y1 - y0
    if equipment_type == "slide":
        ladder = _rel(box, 0.08, 0.14, 0.34, 1.00)
        platform = _rel(box, 0.16, 0.08, 0.46, 0.28)
        slide = [(x0 + 0.40 * width, y0 + 0.24 * height), (x1 - 0.06 * width, y1 - 0.06 * height), (x1 - 0.28 * width, y1 - 0.06 * height), (x0 + 0.28 * width, y0 + 0.28 * height)]
        _rect(draw, platform, fill=primary, outline=outline, width=2, scale=scale, radius=6)
        for lx in (ladder[0], ladder[2]):
            _line(draw, [(lx, ladder[1]), (lx, ladder[3])], fill=outline, width=4, scale=scale)
        for rung_y in (0.35, 0.52, 0.69, 0.86):
            y = y0 + rung_y * height
            _line(draw, [(ladder[0], y), (ladder[2], y)], fill=outline, width=3, scale=scale)
        _poly(draw, slide, fill=accent, outline=outline, width=2, scale=scale)
    elif equipment_type == "swing_set":
        top = y0 + 0.12 * height
        _line(draw, [(x0 + 0.10 * width, y1), (x0 + 0.28 * width, top), (x0 + 0.72 * width, top), (x0 + 0.90 * width, y1)], fill=outline, width=5, scale=scale)
        for center in (0.40, 0.62):
            sx = x0 + center * width
            seat_y = y0 + 0.68 * height
            _line(draw, [(sx - 22.0, top), (sx - 13.0, seat_y)], fill=outline, width=2, scale=scale)
            _line(draw, [(sx + 22.0, top), (sx + 13.0, seat_y)], fill=outline, width=2, scale=scale)
            _rect(draw, (sx - 30.0, seat_y, sx + 30.0, seat_y + 12.0), fill=primary, outline=outline, width=1, scale=scale, radius=3)
    elif equipment_type == "seesaw":
        pivot = [(0.5 * (x0 + x1), y0 + 0.48 * height), (x0 + 0.39 * width, y1), (x0 + 0.61 * width, y1)]
        _poly(draw, pivot, fill=accent, outline=outline, width=2, scale=scale)
        _line(draw, [(x0 + 0.10 * width, y0 + 0.40 * height), (x1 - 0.10 * width, y0 + 0.64 * height)], fill=primary, width=13, scale=scale)
    else:
        cx = 0.5 * (x0 + x1)
        base_y = y1
        top = (cx, y0 + 0.12 * height)
        base = [(x0 + 0.12 * width, base_y), (x1 - 0.12 * width, base_y), top]
        for point in base[:2]:
            _line(draw, [top, point], fill=outline, width=4, scale=scale)
        for frac in (0.32, 0.50, 0.68, 0.84):
            y = y0 + frac * height
            span = (y - top[1]) / max(1.0, base_y - top[1])
            left_x = cx - span * (cx - base[0][0])
            right_x = cx + span * (base[1][0] - cx)
            _line(draw, [(left_x, y), (right_x, y)], fill=outline, width=3, scale=scale)
    return _result(
        box,
        renderer_id=renderer_id,
        renderer_variant_id=equipment_type,
        visual_attributes={
            "primary_color_rgb": [int(v) for v in primary],
            "accent_color_rgb": [int(v) for v in accent],
            "outline_color_rgb": [int(v) for v in outline],
        },
    )


def _render_construction_material(spec: Any, context: Any, renderer_id: str) -> VectorObjectRenderResult:
    draw, box, visual, semantic = _required_draw_box(spec, context)
    scale = max(1, int(getattr(context, "render_scale", 1)))
    style_id = str(visual.get("style_id", getattr(context, "style_id", "flat_vector")))
    outline_raw, outline_w, shadow = style_outline_params(style_id)
    outline = tuple(outline_raw or (66, 73, 84))  # type: ignore[arg-type]
    if shadow:
        _ellipse(draw, (box[0] + 4.0, box[3] - 10.0, box[2] + 8.0, box[3] + 8.0), fill=(150, 139, 119), outline=None, width=1, scale=scale)
    x0, y0, x1, y1 = box
    width = x1 - x0
    height = y1 - y0
    material_type = _renderer_variant_id(spec, visual, semantic) or str(semantic.get("material_type", "cement_bag_stack"))
    if material_type == "brick_stack":
        for row in range(4):
            for col in range(5):
                bx0 = x0 + width * (0.05 + col * 0.18 + (0.08 if row % 2 else 0.0))
                by0 = y0 + height * (0.12 + row * 0.19)
                _rect(draw, (bx0, by0, bx0 + width * 0.16, by0 + height * 0.16), fill=(179, 83, 58), outline=outline, width=1, scale=scale, radius=2)
    elif material_type == "pipe_bundle":
        for row in range(3):
            for col in range(4):
                cx = x0 + width * (0.18 + col * 0.19 + (0.08 if row % 2 else 0.0))
                cy = y0 + height * (0.25 + row * 0.20)
                radius = min(width, height) * 0.09
                _ellipse(draw, (cx - radius, cy - radius, cx + radius, cy + radius), fill=(202, 209, 211), outline=(104, 125, 133), width=2, scale=scale)
    elif material_type == "lumber_stack":
        for row in range(5):
            by0 = y0 + height * (0.14 + row * 0.15)
            _rect(draw, (x0 + width * 0.06, by0, x1 - width * 0.06, by0 + height * 0.10), fill=(177, 123, 68), outline=outline, width=1, scale=scale, radius=3)
            _line(draw, [(x0 + width * 0.12, by0 + height * 0.05), (x1 - width * 0.12, by0 + height * 0.05)], fill=(132, 88, 45), width=1, scale=scale)
    else:
        for row in range(3):
            for col in range(3):
                bx0 = x0 + width * (0.10 + col * 0.28 + (0.06 if row % 2 else 0.0))
                by0 = y0 + height * (0.14 + row * 0.23)
                _rect(draw, (bx0, by0, bx0 + width * 0.23, by0 + height * 0.17), fill=(217, 207, 181), outline=outline, width=int(outline_w), scale=scale, radius=8)
    return _result(box, renderer_id=renderer_id, renderer_variant_id=material_type, visual_attributes={"style_id": style_id})


def _render_construction_equipment(spec: Any, context: Any, renderer_id: str) -> VectorObjectRenderResult:
    draw, box, visual, semantic = _required_draw_box(spec, context)
    scale = max(1, int(getattr(context, "render_scale", 1)))
    style_id = str(visual.get("style_id", getattr(context, "style_id", "flat_vector")))
    outline_raw, outline_w, shadow = style_outline_params(style_id)
    outline = tuple(outline_raw or (66, 73, 84))  # type: ignore[arg-type]
    yellow = _rgb(visual.get("primary_color_rgb", (229, 169, 48)))
    orange = _rgb(visual.get("accent_color_rgb", (218, 119, 48)))
    bluegray = (87, 107, 124)
    tire = (45, 49, 55)
    x0, y0, x1, y1 = box
    width = x1 - x0
    height = y1 - y0
    if shadow:
        _ellipse(draw, (x0 + width * 0.04, y1 - height * 0.10, x1 - width * 0.02, y1 + height * 0.03), fill=(137, 128, 111), outline=None, width=1, scale=scale)
    equipment_type = _renderer_variant_id(spec, visual, semantic) or str(semantic.get("equipment_type", "forklift"))
    if equipment_type == "excavator":
        _rect(draw, _rel(box, 0.06, 0.70, 0.72, 0.90), fill=(57, 61, 65), outline=outline, width=int(outline_w), scale=scale, radius=12)
        _rect(draw, _rel(box, 0.18, 0.42, 0.56, 0.72), fill=yellow, outline=outline, width=int(outline_w), scale=scale, radius=8)
        _rect(draw, _rel(box, 0.36, 0.28, 0.62, 0.58), fill=(91, 134, 157), outline=outline, width=int(outline_w), scale=scale, radius=6)
        _line(draw, [(x0 + width * 0.56, y0 + height * 0.45), (x0 + width * 0.82, y0 + height * 0.25), (x0 + width * 0.94, y0 + height * 0.60)], fill=yellow, width=10, scale=scale)
        _poly(draw, [(x0 + width * 0.88, y0 + height * 0.60), (x0 + width * 1.02, y0 + height * 0.62), (x0 + width * 0.92, y0 + height * 0.77)], fill=(83, 77, 65), outline=outline, width=int(outline_w), scale=scale)
    elif equipment_type == "dump_truck":
        _rect(draw, _rel(box, 0.08, 0.44, 0.58, 0.72), fill=orange, outline=outline, width=int(outline_w), scale=scale, radius=6)
        _poly(draw, [(x0 + width * 0.58, y0 + height * 0.38), (x0 + width * 0.84, y0 + height * 0.42), (x0 + width * 0.88, y0 + height * 0.72), (x0 + width * 0.58, y0 + height * 0.72)], fill=yellow, outline=outline, width=int(outline_w), scale=scale)
        _rect(draw, _rel(box, 0.66, 0.46, 0.80, 0.60), fill=(112, 154, 173), outline=outline, width=1, scale=scale, radius=3)
        for cx in (x0 + width * 0.24, x0 + width * 0.70):
            _ellipse(draw, (cx - width * 0.07, y0 + height * 0.66, cx + width * 0.07, y0 + height * 0.84), fill=tire, outline=outline, width=int(outline_w), scale=scale)
    elif equipment_type == "cement_mixer":
        _rect(draw, _rel(box, 0.10, 0.56, 0.88, 0.74), fill=bluegray, outline=outline, width=int(outline_w), scale=scale, radius=8)
        _ellipse(draw, _rel(box, 0.26, 0.28, 0.66, 0.68), fill=(217, 213, 196), outline=outline, width=int(outline_w), scale=scale)
        _line(draw, [(x0 + width * 0.34, y0 + height * 0.36), (x0 + width * 0.60, y0 + height * 0.60)], fill=(152, 147, 132), width=4, scale=scale)
        _rect(draw, _rel(box, 0.66, 0.38, 0.88, 0.62), fill=yellow, outline=outline, width=int(outline_w), scale=scale, radius=5)
        for cx in (x0 + width * 0.28, x0 + width * 0.74):
            _ellipse(draw, (cx - width * 0.07, y0 + height * 0.66, cx + width * 0.07, y0 + height * 0.84), fill=tire, outline=outline, width=int(outline_w), scale=scale)
    else:
        mast = (59, 63, 68)
        _rect(draw, _rel(box, 0.14, 0.48, 0.68, 0.74), fill=yellow, outline=outline, width=int(outline_w), scale=scale, radius=6)
        _rect(draw, _rel(box, 0.26, 0.24, 0.56, 0.52), fill=(111, 152, 170), outline=outline, width=int(outline_w), scale=scale, radius=4)
        _rect(draw, _rel(box, 0.58, 0.52, 0.74, 0.72), fill=yellow, outline=outline, width=int(outline_w), scale=scale, radius=3)
        _line(draw, [(x0 + width * 0.70, y0 + height * 0.20), (x0 + width * 0.70, y0 + height * 0.80)], fill=mast, width=4, scale=scale)
        _line(draw, [(x0 + width * 0.77, y0 + height * 0.20), (x0 + width * 0.77, y0 + height * 0.80)], fill=mast, width=4, scale=scale)
        _line(draw, [(x0 + width * 0.70, y0 + height * 0.40), (x0 + width * 0.77, y0 + height * 0.40)], fill=mast, width=3, scale=scale)
        _line(draw, [(x0 + width * 0.70, y0 + height * 0.62), (x0 + width * 0.77, y0 + height * 0.62)], fill=mast, width=3, scale=scale)
        _line(draw, [(x0 + width * 0.70, y0 + height * 0.74), (x0 + width * 0.98, y0 + height * 0.74)], fill=mast, width=3, scale=scale)
        _line(draw, [(x0 + width * 0.70, y0 + height * 0.81), (x0 + width * 0.96, y0 + height * 0.81)], fill=mast, width=3, scale=scale)
        for cx in (x0 + width * 0.28, x0 + width * 0.56):
            _ellipse(draw, (cx - width * 0.06, y0 + height * 0.67, cx + width * 0.06, y0 + height * 0.83), fill=tire, outline=outline, width=int(outline_w), scale=scale)
    return _result(
        box,
        renderer_id=renderer_id,
        renderer_variant_id=equipment_type,
        visual_attributes={
            "primary_color_rgb": [int(v) for v in yellow],
            "accent_color_rgb": [int(v) for v in orange],
            "style_id": style_id,
        },
    )


def _render_indoor_furniture(spec: Any, context: Any, renderer_id: str) -> VectorObjectRenderResult:
    draw, box, visual, semantic = _required_draw_box(spec, context)
    scale = max(1, int(getattr(context, "render_scale", 1)))
    style_id = str(visual.get("style_id", getattr(context, "style_id", "flat_vector")))
    furniture_type = _renderer_variant_id(spec, visual, semantic) or str(semantic.get("furniture_type", "table"))
    draw_phase = str(visual.get("draw_phase", "all"))
    if furniture_type == "table":
        _draw_indoor_table(draw, box, visual, scale=scale, draw_phase=draw_phase)
    elif furniture_type == "sofa":
        _draw_indoor_sofa(draw, box, visual, scale=scale)
    elif furniture_type == "cabinet":
        _draw_indoor_cabinet(draw, box, visual, scale=scale)
    return _result(box, renderer_id=renderer_id, renderer_variant_id=furniture_type, visual_attributes={"style_id": style_id, "draw_phase": draw_phase})


def _render_indoor_surface(spec: Any, context: Any, renderer_id: str) -> VectorObjectRenderResult:
    draw, box, visual, semantic = _required_draw_box(spec, context)
    scale = max(1, int(getattr(context, "render_scale", 1)))
    style_id = str(visual.get("style_id", getattr(context, "style_id", "flat_vector")))
    surface_type = _renderer_variant_id(spec, visual, semantic) or str(semantic.get("surface_type", "table"))
    if surface_type == "shelf":
        board_bbox = _optional_bbox(visual.get("board_bbox"))
        if board_bbox is not None:
            board_fill = _rgb(visual.get("top_fill_rgb"), (170, 132, 88))
            board_outline = _rgb(visual.get("outline_rgb"), (91, 70, 52))
            _rect(draw, board_bbox, fill=board_fill, outline=board_outline, width=3, scale=scale)
            bx0, by0, bx1, by1 = [float(v) for v in board_bbox]
            shelf_style = str(visual.get("shelf_style", "plank"))
            if shelf_style == "brackets":
                for bx in (bx0 + 0.18 * (bx1 - bx0), bx1 - 0.18 * (bx1 - bx0)):
                    bracket = [(bx - 15.0, by1), (bx + 15.0, by1), (bx, by1 + 38.0)]
                    _poly(draw, bracket, fill=_blend_rgb(board_fill, board_outline, 0.18), outline=board_outline, width=1, scale=scale)
            elif shelf_style == "cubby":
                for frac in (0.33, 0.66):
                    x = bx0 + frac * (bx1 - bx0)
                    _line(draw, [(x, by0 + 3.0), (x, by1 + 30.0)], fill=board_outline, width=2, scale=scale)
    _draw_indoor_surface_plane(
        draw,
        _surface_plane_from_visual(visual, box),
        top_fill=_rgb(visual.get("top_fill_rgb"), (198, 154, 98)),
        lip_fill=_rgb(visual.get("lip_fill_rgb"), (112, 83, 56)),
        outline=_rgb(visual.get("outline_rgb"), (68, 51, 37)),
        scale=scale,
        lip_bottom_y=_optional_float(visual.get("lip_bottom_y")),
    )
    return _result(box, renderer_id=renderer_id, renderer_variant_id=surface_type, visual_attributes={"style_id": style_id})


def _render_indoor_container(spec: Any, context: Any, renderer_id: str) -> VectorObjectRenderResult:
    draw, box, visual, semantic = _required_draw_box(spec, context)
    scale = max(1, int(getattr(context, "render_scale", 1)))
    style_id = str(visual.get("style_id", getattr(context, "style_id", "flat_vector")))
    container_type = _renderer_variant_id(spec, visual, semantic) or str(semantic.get("container_type", "basket"))
    fill_rgb = _rgb(visual.get("fill_rgb"), (193, 145, 86))
    outline_rgb = _rgb(visual.get("outline_rgb"), (94, 69, 42))
    container_style = str(visual.get("container_style", "plain"))
    x0, y0, x1, y1 = [float(v) for v in box]
    width = x1 - x0
    height = y1 - y0
    if container_type == "basket":
        _rect(draw, box, fill=fill_rgb, outline=outline_rgb, width=3, scale=scale, radius=22)
        draw.arc(_scale_bbox((x0 + 20.0, y0 - 52.0, x1 - 20.0, y0 + 38.0), scale), 180, 360, fill=outline_rgb, width=max(1, 5 * scale))
        if container_style in {"slatted", "woven"}:
            for frac in (0.28, 0.45, 0.62, 0.79):
                x = x0 + frac * width
                _line(draw, [(x, y0 + 16.0), (x, y1 - 10.0)], fill=_blend_rgb(outline_rgb, fill_rgb, 0.35), width=1, scale=scale)
    elif container_type == "box":
        _poly(draw, [(x0, y0 + 34.0), (x1, y0 + 34.0), (x1 - 20.0, y1), (x0 + 20.0, y1)], fill=fill_rgb, outline=outline_rgb, width=1, scale=scale)
        _line(draw, [(x0 + 18.0, y0 + 34.0), (x0 + 2.0, y0), (x1 - 2.0, y0), (x1 - 18.0, y0 + 34.0)], fill=outline_rgb, width=3, scale=scale)
        if container_style == "slatted":
            for frac in (0.44, 0.62, 0.80):
                y = y0 + frac * height
                _line(draw, [(x0 + 18.0, y), (x1 - 18.0, y)], fill=_blend_rgb(outline_rgb, fill_rgb, 0.35), width=1, scale=scale)
    else:
        _rect(draw, box, fill=fill_rgb, outline=outline_rgb, width=3, scale=scale)
        _line(draw, [(x0 + 18.0, y0 + 20.0), (x1 - 18.0, y0 + 20.0)], fill=outline_rgb, width=3, scale=scale)
        if container_style in {"slatted", "woven"}:
            _line(draw, [(x0 + 18.0, y0 + 0.58 * height), (x1 - 18.0, y0 + 0.58 * height)], fill=outline_rgb, width=2, scale=scale)
        if container_style == "woven":
            for frac in (0.30, 0.50, 0.70):
                x = x0 + frac * width
                _line(draw, [(x, y0 + 24.0), (x, y1 - 8.0)], fill=_blend_rgb(outline_rgb, fill_rgb, 0.35), width=1, scale=scale)
    return _result(box, renderer_id=renderer_id, renderer_variant_id=container_type, visual_attributes={"style_id": style_id})


def _draw_indoor_table(draw: Any, box: BBox, visual: Mapping[str, Any], *, scale: int, draw_phase: str) -> None:
    x0, y0, x1, y1 = [float(v) for v in box]
    width = x1 - x0
    if draw_phase in {"all", "rug"}:
        rug_bbox = _optional_bbox(visual.get("rug_bbox"))
        if rug_bbox is not None:
            rug_fill = _rgb(visual.get("rug_fill_rgb"), (153, 130, 166))
            rug_outline = _rgb(visual.get("rug_outline_rgb"), (111, 95, 124))
            _rect(draw, rug_bbox, fill=rug_fill, outline=rug_outline, width=3, scale=scale, radius=34)
            rx0, ry0, rx1, ry1 = [float(v) for v in rug_bbox]
            rug_pattern = str(visual.get("rug_pattern", "plain"))
            if rug_pattern == "border":
                pad = 15.0
                draw.rounded_rectangle(_scale_bbox((rx0 + pad, ry0 + pad, rx1 - pad, ry1 - pad), scale), radius=max(1, 22 * scale), outline=_blend_rgb(rug_outline, rug_fill, 0.30), width=max(1, 3 * scale))
            elif rug_pattern == "stripes":
                stripe = _blend_rgb(rug_fill, rug_outline, 0.18)
                for y in (ry0 + 0.25 * (ry1 - ry0), ry0 + 0.50 * (ry1 - ry0), ry0 + 0.75 * (ry1 - ry0)):
                    _line(draw, [(rx0 + 18.0, y), (rx1 - 18.0, y)], fill=stripe, width=4, scale=scale)
            elif rug_pattern == "dots":
                dot = _blend_rgb(rug_fill, (255, 255, 255), 0.28)
                for col in range(4):
                    for row in range(2):
                        cx = rx0 + (0.22 + 0.18 * col) * (rx1 - rx0)
                        cy = ry0 + (0.34 + 0.28 * row) * (ry1 - ry0)
                        _ellipse(draw, (cx - 5.0, cy - 5.0, cx + 5.0, cy + 5.0), fill=dot, outline=None, width=1, scale=scale)
    if draw_phase not in {"all", "legs"}:
        return
    table_surface_bbox = _optional_bbox(visual.get("surface_bbox"))
    surface_bottom = float(table_surface_bbox[3]) if table_surface_bbox is not None else y0 + 0.42 * (y1 - y0)
    leg_width = float(visual.get("leg_width", 34.0))
    table_dark = _rgb(visual.get("dark_rgb"), (132, 86, 55))
    table_outline = _rgb(visual.get("outline_rgb"), (91, 62, 41))
    table_style = str(visual.get("table_style", "straight_legs"))
    if table_style == "trestle":
        center_x = 0.5 * (x0 + x1)
        _rect(draw, (center_x - 0.58 * leg_width, surface_bottom - 4.0, center_x + 0.58 * leg_width, y1 - 14.0), fill=table_dark, outline=table_outline, width=2, scale=scale)
        _rect(draw, (x0 + 0.18 * width, y1 - 22.0, x1 - 0.18 * width, y1), fill=table_dark, outline=table_outline, width=2, scale=scale)
    elif table_style == "tapered_legs":
        for leg_x in (x0 + 0.13 * width, x1 - 0.13 * width - leg_width):
            points = [(leg_x, surface_bottom - 4.0), (leg_x + leg_width, surface_bottom - 4.0), (leg_x + 1.24 * leg_width, y1), (leg_x - 0.20 * leg_width, y1)]
            _poly(draw, points, fill=table_dark, outline=table_outline, width=1, scale=scale)
    else:
        for leg_x in (x0 + 0.13 * width, x1 - 0.13 * width - leg_width):
            _rect(draw, (leg_x, surface_bottom - 4.0, leg_x + leg_width, y1), fill=table_dark, outline=table_outline, width=2, scale=scale)


def _draw_indoor_sofa(draw: Any, box: BBox, visual: Mapping[str, Any], *, scale: int) -> None:
    x0, y0, x1, y1 = [float(v) for v in box]
    width = x1 - x0
    height = y1 - y0
    outline = _rgb(visual.get("outline_rgb"), (63, 83, 104))
    fill = _rgb(visual.get("fill_rgb"), (111, 142, 167))
    back = _rgb(visual.get("back_fill_rgb"), (126, 158, 183))
    sofa_style = str(visual.get("sofa_style", "block"))
    _rect(draw, (x0, y0 + 0.18 * height, x1, y1), fill=fill, outline=outline, width=3, scale=scale, radius=28)
    _rect(draw, (x0 + 0.08 * width, y0, x1 - 0.08 * width, y0 + 0.50 * height), fill=back, outline=outline, width=3, scale=scale, radius=26)
    if sofa_style == "rounded_arms":
        arm_w = 0.16 * width
        arm_fill = _blend_rgb(fill, back, 0.35)
        for arm in ((x0, y0 + 0.26 * height, x0 + arm_w, y1), (x1 - arm_w, y0 + 0.26 * height, x1, y1)):
            _rect(draw, arm, fill=arm_fill, outline=outline, width=3, scale=scale, radius=24)
    elif sofa_style == "split_cushions":
        seam = _blend_rgb(outline, fill, 0.35)
        for frac in (0.36, 0.64):
            x = x0 + frac * width
            _line(draw, [(x, y0 + 0.27 * height), (x, y1 - 10.0)], fill=seam, width=2, scale=scale)
        _line(draw, [(x0 + 0.08 * width, y0 + 0.59 * height), (x1 - 0.08 * width, y0 + 0.59 * height)], fill=seam, width=2, scale=scale)
    else:
        leg_fill = _blend_rgb(outline, (35, 30, 27), 0.35)
        for lx in (x0 + 0.16 * width, x1 - 0.19 * width):
            _rect(draw, (lx, y1 - 6.0, lx + 18.0, y1 + 18.0), fill=leg_fill, outline=None, width=1, scale=scale)


def _draw_indoor_cabinet(draw: Any, box: BBox, visual: Mapping[str, Any], *, scale: int) -> None:
    x0, y0, x1, y1 = [float(v) for v in box]
    width = x1 - x0
    height = y1 - y0
    outline = _rgb(visual.get("outline_rgb"), (82, 60, 43))
    fill = _rgb(visual.get("fill_rgb"), (164, 122, 82))
    panel_fill = _rgb(visual.get("panel_rgb"), (187, 143, 94))
    cabinet_style = str(visual.get("cabinet_style", "panel_doors"))
    _rect(draw, box, fill=fill, outline=outline, width=3, scale=scale)
    if cabinet_style == "open_shelves":
        inner = (x0 + 0.08 * width, y0 + 0.14 * height, x0 + 0.92 * width, y0 + 0.88 * height)
        _rect(draw, inner, fill=_blend_rgb(fill, (255, 255, 255), 0.10), outline=outline, width=2, scale=scale)
        for frac in (0.40, 0.66):
            y = y0 + frac * height
            _line(draw, [(inner[0], y), (inner[2], y)], fill=outline, width=2, scale=scale)
        x = x0 + 0.50 * width
        _line(draw, [(x, inner[1]), (x, inner[3])], fill=outline, width=2, scale=scale)
    elif cabinet_style == "mixed_drawers":
        _rect(draw, _rel(box, 0.08, 0.15, 0.91, 0.34), fill=panel_fill, outline=outline, width=2, scale=scale)
        _rect(draw, _rel(box, 0.08, 0.39, 0.46, 0.88), fill=panel_fill, outline=outline, width=2, scale=scale)
        _rect(draw, _rel(box, 0.52, 0.39, 0.91, 0.88), fill=panel_fill, outline=outline, width=2, scale=scale)
        for frac in (0.22, 0.50, 0.74):
            y = y0 + frac * height
            _line(draw, [(x0 + 0.17 * width, y), (x0 + 0.82 * width, y)], fill=_blend_rgb(outline, panel_fill, 0.35), width=1, scale=scale)
    else:
        _rect(draw, _rel(box, 0.08, 0.16, 0.46, 0.48), fill=panel_fill, outline=outline, width=2, scale=scale)
        _rect(draw, _rel(box, 0.51, 0.16, 0.91, 0.48), fill=panel_fill, outline=outline, width=2, scale=scale)
        _rect(draw, _rel(box, 0.08, 0.56, 0.91, 0.90), fill=panel_fill, outline=outline, width=2, scale=scale)


def _renderer_id(spec: Any) -> str:
    direct = str(getattr(spec, "renderer_id", "") or "")
    if direct:
        return direct
    visual = _mapping(getattr(spec, "visual_attributes", {}))
    return str(visual.get("renderer_id", ""))


def _renderer_variant_id(spec: Any, visual: Mapping[str, Any], semantic: Mapping[str, Any]) -> str:
    direct = str(getattr(spec, "renderer_variant_id", "") or "")
    if direct:
        return direct
    for key in ("renderer_variant_id", "pose_id", "pose", "activity", "equipment_type", "material_type", "luggage_type", "furniture_type", "surface_type", "container_type", "decor_type", "fixture_type"):
        value = visual.get(key, semantic.get(key))
        if value:
            return str(value)
    return ""


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _rgb(value: Any, fallback: RGB = (0, 0, 0)) -> RGB:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 3:
        return tuple(max(0, min(255, int(channel))) for channel in value)  # type: ignore[return-value]
    return tuple(int(channel) for channel in fallback)  # type: ignore[return-value]


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_bbox(value: Any) -> BBox | None:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 4:
        return tuple(float(v) for v in value)  # type: ignore[return-value]
    return None


def _optional_rgb(value: Any) -> RGB | None:
    if value is None:
        return None
    return _rgb(value)


def _required_draw_box(spec: Any, context: Any) -> tuple[Any, BBox, Mapping[str, Any], Mapping[str, Any]]:
    if getattr(context, "draw", None) is None:
        raise ValueError("registered vector rendering requires RenderContext.draw")
    bbox = getattr(spec, "bbox_xyxy", None)
    if bbox is None:
        raise ValueError("registered vector rendering requires IllustrationObjectSpec.bbox_xyxy")
    return (
        context.draw,
        tuple(float(value) for value in bbox),  # type: ignore[return-value]
        _mapping(getattr(spec, "visual_attributes", {})),
        _mapping(getattr(spec, "semantic_attributes", {})),
    )


def _result(
    bbox: BBox,
    *,
    renderer_id: str,
    renderer_variant_id: str,
    visual_attributes: Mapping[str, Any] | None = None,
    parts: Sequence[Mapping[str, Any]] = (),
) -> VectorObjectRenderResult:
    attrs = dict(visual_attributes or {})
    attrs["renderer_id"] = str(renderer_id)
    attrs["renderer_variant_id"] = str(renderer_variant_id)
    return VectorObjectRenderResult(
        bbox_xyxy=tuple(round(float(value), 3) for value in bbox),  # type: ignore[arg-type]
        parts=tuple(dict(part) for part in parts),
        visual_attributes=attrs,
    )


def _rel(box: BBox, x0: float, y0: float, x1: float, y1: float) -> BBox:
    bx0, by0, bx1, by1 = [float(value) for value in box]
    width = bx1 - bx0
    height = by1 - by0
    return (bx0 + x0 * width, by0 + y0 * height, bx0 + x1 * width, by0 + y1 * height)


def _expand(box: Sequence[float], pad: float) -> BBox:
    values = tuple(float(value) for value in box)
    if len(values) == 4:
        return (values[0] - pad, values[1] - pad, values[2] + pad, values[3] + pad)
    xs = values[0::2]
    ys = values[1::2]
    return (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)


def _rect(draw: Any, box: BBox, *, fill: RGB, outline: RGB | None, width: int, scale: int, radius: float = 0.0) -> None:
    scaled = _scale_bbox(box, scale)
    if float(radius) > 0.0:
        draw.rounded_rectangle(scaled, radius=max(1, int(round(float(radius) * int(scale)))), fill=tuple(fill), outline=tuple(outline) if outline else None, width=max(1, int(width) * int(scale)))
    else:
        draw.rectangle(scaled, fill=tuple(fill), outline=tuple(outline) if outline else None, width=max(1, int(width) * int(scale)))


def _ellipse(draw: Any, box: BBox, *, fill: RGB, outline: RGB | None, width: int, scale: int) -> None:
    draw.ellipse(_scale_bbox(box, scale), fill=tuple(fill), outline=tuple(outline) if outline else None, width=max(1, int(width) * int(scale)))


def _poly(draw: Any, points: Sequence[Tuple[float, float]], *, fill: RGB, outline: RGB | None, width: int, scale: int) -> None:
    draw.polygon(_scale_points(points, scale), fill=tuple(fill))
    if outline:
        draw.line(_scale_points([*points, points[0]], scale), fill=tuple(outline), width=max(1, int(width) * int(scale)), joint="curve")


def _line(draw: Any, points: Sequence[Tuple[float, float]], *, fill: RGB, width: int, scale: int) -> None:
    draw.line(_scale_points(points, scale), fill=tuple(fill), width=max(1, int(width) * int(scale)), joint="curve")


def _blend_rgb(a: RGB, b: RGB, t: float) -> RGB:
    mix = max(0.0, min(1.0, float(t)))
    return tuple(int(round((1.0 - mix) * int(left) + mix * int(right))) for left, right in zip(a, b))  # type: ignore[return-value]


def _surface_plane_from_visual(visual: Mapping[str, Any], box: BBox) -> Mapping[str, Tuple[float, float]]:
    raw = visual.get("plane")
    if isinstance(raw, Mapping):
        plane: dict[str, Tuple[float, float]] = {}
        for key in ("back_left", "back_right", "front_left", "front_right"):
            value = raw.get(key)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 2:
                plane[key] = (float(value[0]), float(value[1]))
        if len(plane) == 4:
            return plane
    return {
        "back_left": (box[0] + 0.07 * (box[2] - box[0]), box[1]),
        "back_right": (box[2] - 0.07 * (box[2] - box[0]), box[1]),
        "front_left": (box[0], box[1] + 0.46 * (box[3] - box[1])),
        "front_right": (box[2], box[1] + 0.46 * (box[3] - box[1])),
    }


def _draw_indoor_surface_plane(
    draw: Any,
    plane: Mapping[str, Tuple[float, float]],
    *,
    top_fill: RGB,
    lip_fill: RGB,
    outline: RGB,
    scale: int,
    lip_bottom_y: float | None,
) -> None:
    top = [plane["back_left"], plane["back_right"], plane["front_right"], plane["front_left"]]
    draw.polygon(_scale_points(top, scale), fill=tuple(top_fill))
    draw.line(_scale_points([*top, top[0]], scale), fill=tuple(outline), width=max(1, 3 * scale), joint="curve")
    front_left = plane["front_left"]
    front_right = plane["front_right"]
    bottom_y = float(lip_bottom_y) if lip_bottom_y is not None else float(front_left[1]) + 32.0
    lip = [front_left, front_right, (front_right[0], bottom_y), (front_left[0], bottom_y)]
    draw.polygon(_scale_points(lip, scale), fill=tuple(lip_fill))
    draw.line(_scale_points([*lip, lip[0]], scale), fill=tuple(outline), width=max(1, 3 * scale), joint="curve")


def _shadow(draw: Any, box: BBox, *, scale: int) -> None:
    x0, y0, x1, y1 = [float(value) for value in box]
    _ellipse(draw, (x0 + 0.08 * (x1 - x0), y1 - 0.06 * (y1 - y0), x1 - 0.04 * (x1 - x0), y1 + 0.04 * (y1 - y0)), fill=(132, 128, 119), outline=None, width=1, scale=scale)


def _part(parts: list[dict[str, Any]], counts: dict[str, int], object_id: str, kind: str, box: BBox) -> None:
    index = int(counts.get(str(kind), 0))
    counts[str(kind)] = index + 1
    parts.append({"part_id": f"{object_id}_{kind}_{index}", "part_kind": str(kind), "bbox": [round(float(value), 3) for value in box], "attributes": {}})


__all__ = [
    "BENCH_RENDERER_IDS",
    "CONSTRUCTION_EQUIPMENT_RENDERER_IDS",
    "CONSTRUCTION_MATERIAL_RENDERER_IDS",
    "INDOOR_CONTAINER_RENDERER_IDS",
    "INDOOR_FURNITURE_RENDERER_IDS",
    "INDOOR_SURFACE_RENDERER_IDS",
    "LUGGAGE_RENDERER_IDS",
    "PARK_EQUIPMENT_RENDERER_IDS",
    "PERSON_RENDERER_IDS",
    "VectorObjectRenderResult",
    "render_registered_vector_object",
]
