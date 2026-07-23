"""Rendering primitives for rectangular-solid measurement diagrams."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from PIL import ImageDraw

from trace_tasks.core.sampling import uniform_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.geometry.shared.diagram_style import (
    prepare_geometry_diagram_style_and_background,
)
from trace_tasks.tasks.geometry.shared.measurement_rendering import (
    bbox_from_points,
    bbox_to_list,
    draw_label_backplate,
    pad_bbox,
    readout_text_fill,
    readout_text_metadata,
)
from trace_tasks.tasks.geometry.shared.vector2d import (
    add as _add,
    mul as _mul,
    perp as _perp,
    point_to_list as _point_to_list,
    sub as _sub,
    unit as _unit,
)
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.text_legibility import draw_text_traced
from trace_tasks.tasks.shared.text_rendering import load_font

from .defaults import SCENE_ID
from .state import (
    BBox,
    Color,
    CubeFrameProblem,
    CuboidMeasureProblem,
    OpenBoxNetProblem,
    Point,
    RenderContext,
    RenderedRectangularSolidScene,
)

CUBOID_DIMENSION_ANNOTATION_KEYS: Tuple[str, ...] = (
    "target_dimension",
)
FRAME_ANNOTATION_KEYS: Tuple[str, ...] = (
    "highlighted_frame_region",
)
NET_ANNOTATION_KEYS: Tuple[str, ...] = (
    "target_dimension",
)


def create_render_context(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
) -> RenderContext:
    """Create a styled canvas and reusable drawing context."""

    width = int(params.get("canvas_width", group_default(rendering_defaults, "canvas_width", 820)))
    height = int(params.get("canvas_height", group_default(rendering_defaults, "canvas_height", 600)))
    image, background_meta, diagram_style, diagram_style_meta = prepare_geometry_diagram_style_and_background(
        instance_seed=int(instance_seed),
        params=params,
        scene_id=SCENE_ID,
        canvas_width=int(width),
        canvas_height=int(height),
        allow_dark=False,
        require_grid=None,
    )
    face_palettes: Tuple[Tuple[Color, Color, Color], ...] = (
        ((237, 246, 255), (218, 235, 249), (246, 251, 255)),
        ((244, 248, 238), (224, 239, 219), (251, 253, 246)),
        ((255, 242, 232), (247, 222, 208), (255, 249, 243)),
        ((245, 241, 255), (227, 220, 246), (252, 249, 255)),
    )
    palette_rng = spawn_rng(int(instance_seed), f"{SCENE_ID}.face_palette")
    face_front, face_side, face_top = uniform_choice(palette_rng, face_palettes)
    font_size = int(params.get("label_font_size", group_default(rendering_defaults, "label_font_size", 22)))
    small_font_size = int(params.get("small_label_font_size", group_default(rendering_defaults, "small_label_font_size", 18)))
    line_width = int(params.get("line_width", group_default(rendering_defaults, "line_width", 3)))
    label_stroke_width = int(
        params.get(
            "label_stroke_width",
            group_default(rendering_defaults, "label_stroke_width", int(diagram_style.label_stroke_width_px)),
        )
    )
    return RenderContext(
        image=image,
        draw=ImageDraw.Draw(image),
        width=int(width),
        height=int(height),
        line_color=tuple(int(value) for value in diagram_style.stroke_rgb),
        secondary_color=tuple(int(value) for value in diagram_style.secondary_stroke_rgb),
        label_color=tuple(int(value) for value in diagram_style.label_rgb),
        label_stroke_color=tuple(int(value) for value in diagram_style.label_stroke_rgb),
        label_backing_color=tuple(int(value) for value in diagram_style.panel_fill_rgb),
        face_front=tuple(int(value) for value in face_front),
        face_side=tuple(int(value) for value in face_side),
        face_top=tuple(int(value) for value in face_top),
        accent_color=tuple(int(value) for value in diagram_style.accent_rgb),
        muted_color=tuple(int(value) for value in diagram_style.secondary_stroke_rgb),
        line_width=max(2, int(line_width)),
        label_stroke_width=max(0, int(label_stroke_width)),
        font=load_font(max(12, int(font_size)), bold=False),
        small_font=load_font(max(10, int(small_font_size)), bold=False),
        diagram_style_meta=dict(diagram_style_meta),
        background_meta=dict(background_meta),
    )


def _draw_text_centered(ctx: RenderContext, text: str, center: Point, *, small: bool = True) -> BBox:
    """Draw one centered readout label and return its padded bbox."""

    font = ctx.small_font if bool(small) else ctx.font
    bbox = ctx.draw.textbbox(
        (float(center[0]), float(center[1])),
        str(text),
        anchor="mm",
        font=font,
        stroke_width=max(0, int(ctx.label_stroke_width)),
    )
    draw_label_backplate(ctx, bbox)
    fill = readout_text_fill(ctx, ctx.label_color)
    stroke_fill = readout_text_fill(ctx, ctx.label_stroke_color)
    draw_text_traced(
        ctx.draw,
        (float(center[0]), float(center[1])),
        str(text),
        anchor="mm",
        font=font,
        fill=fill,
        stroke_width=max(0, int(ctx.label_stroke_width)),
        stroke_fill=stroke_fill,
        role="readout",
        required=True,
        extra_metadata=readout_text_metadata(ctx, fill),
    )
    return pad_bbox(bbox, 4.0, width=ctx.width, height=ctx.height)


def _draw_value_box(ctx: RenderContext, text: str, center: Point) -> BBox:
    """Draw a prominent readout box used for volume and frame totals."""

    font = ctx.font
    bbox = ctx.draw.textbbox((0, 0), str(text), font=font, stroke_width=max(0, int(ctx.label_stroke_width)))
    text_w = float(bbox[2] - bbox[0])
    text_h = float(bbox[3] - bbox[1])
    left = float(center[0]) - text_w / 2.0 - 14.0
    top = float(center[1]) - text_h / 2.0 - 9.0
    right = left + text_w + 28.0
    bottom = top + text_h + 18.0
    ctx.draw.rounded_rectangle(
        (left, top, right, bottom),
        radius=6,
        fill=ctx.label_backing_color,
        outline=ctx.muted_color,
        width=max(1, ctx.line_width - 1),
    )
    fill = readout_text_fill(ctx, ctx.label_color)
    stroke_fill = readout_text_fill(ctx, ctx.label_stroke_color)
    draw_text_traced(
        ctx.draw,
        ((left + right) / 2.0, (top + bottom) / 2.0),
        str(text),
        anchor="mm",
        font=font,
        fill=fill,
        stroke_width=max(0, int(ctx.label_stroke_width)),
        stroke_fill=stroke_fill,
        role="readout",
        required=True,
        extra_metadata=readout_text_metadata(ctx, fill),
    )
    return pad_bbox((left, top, right, bottom), 2.0, width=ctx.width, height=ctx.height)


def _draw_dimension(
    ctx: RenderContext,
    start: Point,
    end: Point,
    label: str,
    *,
    offset: Point,
    label_offset: Point,
    color: Color | None = None,
) -> tuple[Point, Point, BBox]:
    """Draw a dimension guide segment with endpoint ticks."""

    line_color = color or ctx.secondary_color
    dim_start = _add(start, offset)
    dim_end = _add(end, offset)
    ctx.draw.line([dim_start, dim_end], fill=line_color, width=max(2, ctx.line_width - 1))
    direction = _unit(_sub(dim_end, dim_start))
    normal = _perp(direction)
    tick = 7.0
    for point in (dim_start, dim_end):
        ctx.draw.line(
            [
                _add(point, _mul(normal, -tick)),
                _add(point, _mul(normal, tick)),
            ],
            fill=line_color,
            width=max(1, ctx.line_width - 2),
        )
    midpoint = ((float(dim_start[0]) + float(dim_end[0])) / 2.0, (float(dim_start[1]) + float(dim_end[1])) / 2.0)
    label_bbox = _draw_text_centered(ctx, label, _add(midpoint, label_offset), small=True)
    return dim_start, dim_end, label_bbox


def _assert_bboxes_inside(bboxes: Sequence[BBox], *, width: int, height: int) -> None:
    """Reject labels or witnesses that land too close to the canvas edge."""

    for bbox in bboxes:
        x0, y0, x1, y1 = [float(value) for value in bbox]
        if x0 <= 3.0 or y0 <= 3.0 or x1 >= float(width) - 3.0 or y1 >= float(height) - 3.0:
            raise ValueError("rectangular-solid label too close to canvas edge")


def _mix_colors(foreground: Color, background: Color, foreground_weight: float) -> Color:
    """Blend two RGB colors with an explicit foreground weight."""

    weight = max(0.0, min(1.0, float(foreground_weight)))
    return tuple(
        int(round(float(foreground[index]) * weight + float(background[index]) * (1.0 - weight)))
        for index in range(3)
    )


def _segment_bbox(start: Point, end: Point, *, width: int, height: int, pad: float = 8.0) -> BBox:
    """Return a padded bbox around one drawn segment."""

    return pad_bbox(
        (
            min(float(start[0]), float(end[0])),
            min(float(start[1]), float(end[1])),
            max(float(start[0]), float(end[0])),
            max(float(start[1]), float(end[1])),
        ),
        float(pad),
        width=int(width),
        height=int(height),
    )


def _draw_hatched_cutout(ctx: RenderContext, bbox: BBox) -> None:
    """Draw one removable corner square in the open-box net."""

    x0, y0, x1, y1 = [float(value) for value in bbox]
    ctx.draw.rectangle((x0, y0, x1, y1), fill=(255, 248, 244), outline=ctx.secondary_color, width=max(1, ctx.line_width - 2))
    step = 10.0
    offset = -float(y1 - y0)
    while offset < float(x1 - x0):
        start = (x0 + max(0.0, offset), y1 if offset < 0 else y0)
        end = (
            x0 + min(float(x1 - x0), offset + float(y1 - y0)),
            y0 if offset < 0 else y0 + min(float(y1 - y0), float(x1 - x0) - offset),
        )
        ctx.draw.line([start, end], fill=ctx.muted_color, width=1)
        offset += step


def render_open_box_net_scene(
    ctx: RenderContext,
    problem: OpenBoxNetProblem,
    *,
    instance_seed: int,
) -> RenderedRectangularSolidScene:
    """Render the corner-cut sheet and the resulting base-panel target."""

    rng = spawn_rng(int(instance_seed), f"{problem.formula_family}.render.open_box_net_scene")
    sheet_length = float(problem.sheet_length)
    sheet_width = float(problem.sheet_width)
    cut_size = float(problem.cut_size)
    scale = min((ctx.width - 210.0) / sheet_length, (ctx.height - 190.0) / sheet_width)
    scale *= float(rng.uniform(0.88, 0.97))
    sheet_px_w = sheet_length * scale
    sheet_px_h = sheet_width * scale
    left = (ctx.width - sheet_px_w) / 2.0 + float(rng.uniform(-24.0, 24.0))
    top = (ctx.height - sheet_px_h) / 2.0 + float(rng.uniform(8.0, 30.0))
    right = left + sheet_px_w
    bottom = top + sheet_px_h
    cut = cut_size * scale
    sheet_bbox = pad_bbox((left, top, right, bottom), 4.0, width=ctx.width, height=ctx.height)
    base_bbox = (left + cut, top + cut, right - cut, bottom - cut)
    cutout_bboxes: Dict[str, BBox] = {
        "top_left": (left, top, left + cut, top + cut),
        "top_right": (right - cut, top, right, top + cut),
        "bottom_left": (left, bottom - cut, left + cut, bottom),
        "bottom_right": (right - cut, bottom - cut, right, bottom),
    }
    flap_bboxes: Dict[str, BBox] = {
        "top": (left + cut, top, right - cut, top + cut),
        "bottom": (left + cut, bottom - cut, right - cut, bottom),
        "left": (left, top + cut, left + cut, bottom - cut),
        "right": (right - cut, top + cut, right, bottom - cut),
    }
    ctx.draw.rounded_rectangle(
        (left - 16.0, top - 16.0, right + 16.0, bottom + 16.0),
        radius=8,
        fill=(255, 255, 255),
        outline=ctx.muted_color,
        width=max(1, ctx.line_width - 2),
    )
    for bbox in cutout_bboxes.values():
        _draw_hatched_cutout(ctx, bbox)
    for bbox in flap_bboxes.values():
        ctx.draw.rectangle(bbox, fill=ctx.face_top, outline=ctx.line_color, width=max(2, ctx.line_width - 1))
    ctx.draw.rectangle(base_bbox, fill=ctx.face_front, outline=ctx.line_color, width=ctx.line_width)
    for x0, y0, x1, y1 in flap_bboxes.values():
        ctx.draw.rectangle((x0, y0, x1, y1), outline=ctx.secondary_color, width=1)

    label_bboxes: Dict[str, BBox] = {}
    label_bboxes["sheet_length"] = _draw_text_centered(
        ctx,
        f"sheet length {int(problem.sheet_length)}",
        ((left + right) / 2.0, bottom + 28.0),
        small=True,
    )
    label_bboxes["sheet_width"] = _draw_text_centered(
        ctx,
        f"sheet width {int(problem.sheet_width)}",
        (left - 48.0, (top + bottom) / 2.0),
        small=True,
    )
    label_bboxes["cut_size"] = _draw_text_centered(
        ctx,
        f"cut {int(problem.cut_size)}",
        (
            (cutout_bboxes["top_left"][0] + cutout_bboxes["top_left"][2]) / 2.0,
            (cutout_bboxes["top_left"][1] + cutout_bboxes["top_left"][3]) / 2.0,
        ),
        small=True,
    )
    if problem.target_role == "base_length":
        target_segment = (
            (float(base_bbox[0]), float(base_bbox[3])),
            (float(base_bbox[2]), float(base_bbox[3])),
        )
        target_bbox = _segment_bbox(
            target_segment[0],
            target_segment[1],
            width=ctx.width,
            height=ctx.height,
            pad=12.0,
        )
        label_bboxes["target"] = _draw_text_centered(ctx, "?", ((base_bbox[0] + base_bbox[2]) / 2.0, base_bbox[3] - 18.0), small=True)
    elif problem.target_role == "base_width":
        target_segment = (
            (float(base_bbox[2]), float(base_bbox[1])),
            (float(base_bbox[2]), float(base_bbox[3])),
        )
        target_bbox = _segment_bbox(
            target_segment[0],
            target_segment[1],
            width=ctx.width,
            height=ctx.height,
            pad=12.0,
        )
        label_bboxes["target"] = _draw_text_centered(ctx, "?", (base_bbox[2] - 18.0, (base_bbox[1] + base_bbox[3]) / 2.0), small=True)
    else:
        raise ValueError("open-box net target_role must be base_length or base_width")

    annotation_bboxes = {
        "sheet_bbox": sheet_bbox,
        "cutout_bbox": pad_bbox(cutout_bboxes["top_left"], 4.0, width=ctx.width, height=ctx.height),
        "base_panel_bbox": pad_bbox(base_bbox, 4.0, width=ctx.width, height=ctx.height),
        "target_region_bbox": target_bbox,
    }
    _assert_bboxes_inside(tuple(annotation_bboxes.values()) + tuple(label_bboxes.values()), width=ctx.width, height=ctx.height)
    scene_entities = (
        {
            "entity_id": "open_box_net",
            "entity_type": "corner_cut_open_box_net",
            "sheet_length_units": int(problem.sheet_length),
            "sheet_width_units": int(problem.sheet_width),
            "cut_size_units": int(problem.cut_size),
            "base_length_units": int(problem.base_length),
            "base_width_units": int(problem.base_width),
            "height_units": int(problem.cut_size),
            "open_box_volume_units": int(problem.open_box_volume),
            "bbox": bbox_to_list(sheet_bbox),
        },
    )
    render_map = {
        "coord_space": "pixel",
        "target_role": str(problem.target_role),
        "sheet_bbox": bbox_to_list(sheet_bbox),
        "cutout_bboxes": {key: bbox_to_list(value) for key, value in cutout_bboxes.items()},
        "flap_bboxes": {key: bbox_to_list(value) for key, value in flap_bboxes.items()},
        "base_panel_bbox": bbox_to_list(base_bbox),
        "target_dimension_segment": [_point_to_list(target_segment[0]), _point_to_list(target_segment[1])],
        "annotation_bboxes": {key: bbox_to_list(value) for key, value in annotation_bboxes.items()},
        "label_bboxes": {key: bbox_to_list(value) for key, value in label_bboxes.items()},
        "scale_px_per_unit": round(float(scale), 3),
    }
    return RenderedRectangularSolidScene(
        image=ctx.image,
        annotation_type="segment",
        annotation_keyed_points={},
        annotation_keyed_bboxes=dict(annotation_bboxes),
        annotation_roles=NET_ANNOTATION_KEYS,
        label_bboxes=dict(label_bboxes),
        scene_entities=scene_entities,
        render_map=render_map,
    )


def render_cube_frame_scene(
    ctx: RenderContext,
    problem: CubeFrameProblem,
    *,
    instance_seed: int,
) -> RenderedRectangularSolidScene:
    """Render a cube wire frame with total or highlighted length readout."""

    rng = spawn_rng(int(instance_seed), f"{problem.formula_family}.render.frame_scene")
    depth_vec = (0.68, -0.44)
    local_points = {
        "front_bottom_left": (0.0, 0.0),
        "front_bottom_right": (1.0, 0.0),
        "back_bottom_left": depth_vec,
        "back_bottom_right": (1.0 + depth_vec[0], depth_vec[1]),
        "front_top_left": (0.0, -1.0),
        "front_top_right": (1.0, -1.0),
        "back_top_left": (depth_vec[0], depth_vec[1] - 1.0),
        "back_top_right": (1.0 + depth_vec[0], depth_vec[1] - 1.0),
    }
    min_x = min(point[0] for point in local_points.values())
    max_x = max(point[0] for point in local_points.values())
    min_y = min(point[1] for point in local_points.values())
    max_y = max(point[1] for point in local_points.values())
    span_x = max(1e-6, max_x - min_x)
    span_y = max(1e-6, max_y - min_y)
    scale = min((ctx.width - 300.0) / span_x, (ctx.height - 220.0) / span_y)
    scale *= float(rng.uniform(0.88, 0.98))
    local_center = ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0)
    target_center = (
        (ctx.width / 2.0) + float(rng.uniform(-28.0, 28.0)),
        (ctx.height / 2.0) + float(rng.uniform(-8.0, 24.0)),
    )

    def transform(point: Point) -> Point:
        return (
            (float(point[0]) - float(local_center[0])) * float(scale) + float(target_center[0]),
            (float(point[1]) - float(local_center[1])) * float(scale) + float(target_center[1]),
        )

    points = {key: transform(point) for key, point in local_points.items()}
    edge_defs: Dict[str, Tuple[str, str]] = {
        "front_bottom": ("front_bottom_left", "front_bottom_right"),
        "right_depth_bottom": ("front_bottom_right", "back_bottom_right"),
        "back_bottom": ("back_bottom_left", "back_bottom_right"),
        "left_depth_bottom": ("front_bottom_left", "back_bottom_left"),
        "front_left_vertical": ("front_bottom_left", "front_top_left"),
        "front_right_vertical": ("front_bottom_right", "front_top_right"),
        "back_right_vertical": ("back_bottom_right", "back_top_right"),
        "back_left_vertical": ("back_bottom_left", "back_top_left"),
        "front_top": ("front_top_left", "front_top_right"),
        "right_depth_top": ("front_top_right", "back_top_right"),
        "back_top": ("back_top_left", "back_top_right"),
        "left_depth_top": ("front_top_left", "back_top_left"),
    }
    draw_order = (
        "back_bottom",
        "back_left_vertical",
        "back_top",
        "left_depth_bottom",
        "left_depth_top",
        "right_depth_bottom",
        "right_depth_top",
        "back_right_vertical",
        "front_bottom",
        "front_left_vertical",
        "front_right_vertical",
        "front_top",
    )
    partial_paths: Tuple[Tuple[str, ...], ...] = (
        (
            "front_bottom",
            "right_depth_bottom",
            "back_right_vertical",
            "right_depth_top",
            "front_top",
            "front_left_vertical",
            "left_depth_bottom",
            "back_bottom",
        ),
        (
            "front_left_vertical",
            "front_top",
            "right_depth_top",
            "back_right_vertical",
            "right_depth_bottom",
            "front_bottom",
            "left_depth_bottom",
            "back_left_vertical",
        ),
        (
            "left_depth_top",
            "back_top",
            "back_right_vertical",
            "right_depth_bottom",
            "front_bottom",
            "front_left_vertical",
            "front_top",
            "right_depth_top",
        ),
    )
    path_rng = spawn_rng(int(instance_seed), f"{problem.formula_family}.partial_frame_path")
    if problem.frame_mode == "partial":
        highlighted_path = uniform_choice(path_rng, partial_paths)
        highlighted_edges = tuple(highlighted_path[: int(problem.visible_frame_edge_count)])
    else:
        highlighted_edges = tuple(edge_defs.keys())

    full_frame_bbox = bbox_from_points(tuple(points.values()), width=ctx.width, height=ctx.height, pad=10.0)
    faint_frame_color = _mix_colors(ctx.secondary_color, ctx.label_backing_color, 0.20)
    highlight_halo_color = ctx.label_backing_color
    ctx.draw.rounded_rectangle(
        (full_frame_bbox[0] - 18.0, full_frame_bbox[1] - 18.0, full_frame_bbox[2] + 18.0, full_frame_bbox[3] + 18.0),
        radius=8,
        fill=ctx.label_backing_color,
        outline=faint_frame_color,
        width=max(1, ctx.line_width - 2),
    )
    for edge_name in draw_order:
        if edge_name in highlighted_edges:
            continue
        start_key, end_key = edge_defs[edge_name]
        ctx.draw.line([points[start_key], points[end_key]], fill=faint_frame_color, width=max(1, ctx.line_width - 1))
    for edge_name in draw_order:
        if edge_name not in highlighted_edges:
            continue
        start_key, end_key = edge_defs[edge_name]
        ctx.draw.line(
            [points[start_key], points[end_key]],
            fill=highlight_halo_color,
            width=max(14, ctx.line_width + 10),
        )
    for edge_name in draw_order:
        if edge_name not in highlighted_edges:
            continue
        start_key, end_key = edge_defs[edge_name]
        ctx.draw.line([points[start_key], points[end_key]], fill=ctx.accent_color, width=max(9, ctx.line_width + 6))
    for point in points.values():
        radius = 3.5
        ctx.draw.ellipse(
            (point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius),
            fill=faint_frame_color,
            outline=ctx.label_backing_color,
            width=1,
        )
    highlighted_endpoint_keys = {
        endpoint_key
        for edge_name in highlighted_edges
        for endpoint_key in edge_defs[edge_name]
    }
    for point_key in highlighted_endpoint_keys:
        point = points[point_key]
        radius = 7.0
        ctx.draw.ellipse(
            (point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius),
            fill=ctx.accent_color,
            outline=ctx.label_backing_color,
            width=2,
        )

    target_start_key, target_end_key = edge_defs["front_bottom"]
    target_start = points[target_start_key]
    target_end = points[target_end_key]
    target_mid = ((target_start[0] + target_end[0]) / 2.0, (target_start[1] + target_end[1]) / 2.0)
    target_edge_bbox = _segment_bbox(target_start, target_end, width=ctx.width, height=ctx.height, pad=10.0)
    label_bboxes: Dict[str, BBox] = {
        "target_edge": _draw_text_centered(ctx, "?", (target_mid[0], target_mid[1] + 26.0), small=True)
    }
    readout_text = (
        f"Total frame length {int(problem.frame_length)}"
        if problem.frame_mode == "total"
        else f"Highlighted length {int(problem.frame_length)}"
    )
    label_bboxes["frame_length"] = _draw_value_box(ctx, readout_text, (ctx.width / 2.0, 52.0 + float(rng.uniform(-5.0, 7.0))))

    highlighted_points = tuple(
        point
        for edge_name in highlighted_edges
        for point in (points[edge_defs[edge_name][0]], points[edge_defs[edge_name][1]])
    )
    highlighted_bbox = bbox_from_points(highlighted_points, width=ctx.width, height=ctx.height, pad=12.0)
    given_length_bbox = full_frame_bbox if problem.frame_mode == "total" else highlighted_bbox
    annotation_bboxes = {
        "frame_bbox": full_frame_bbox,
        "given_length_region_bbox": given_length_bbox,
        "target_edge_bbox": target_edge_bbox,
    }
    _assert_bboxes_inside(tuple(annotation_bboxes.values()) + tuple(label_bboxes.values()), width=ctx.width, height=ctx.height)
    scene_entities = (
        {
            "entity_id": "cube_frame",
            "entity_type": "cube_wire_frame",
            "edge_units": int(problem.cube_edge),
            "visible_frame_edge_count": int(problem.visible_frame_edge_count),
            "frame_length_units": int(problem.frame_length),
            "total_frame_length_units": int(problem.cube_edge * 12),
            "bbox": bbox_to_list(full_frame_bbox),
        },
    )
    render_map = {
        "coord_space": "pixel",
        "target_role": "cube_edge",
        "frame_mode": str(problem.frame_mode),
        "cube_vertices": {key: _point_to_list(value) for key, value in points.items()},
        "frame_edges": {
            edge_name: [_point_to_list(points[start_key]), _point_to_list(points[end_key])]
            for edge_name, (start_key, end_key) in edge_defs.items()
        },
        "highlighted_edges": list(highlighted_edges),
        "target_edge": "front_bottom",
        "annotation_bboxes": {key: bbox_to_list(value) for key, value in annotation_bboxes.items()},
        "label_bboxes": {key: bbox_to_list(value) for key, value in label_bboxes.items()},
        "scale_px_per_edge": round(float(scale), 3),
    }
    return RenderedRectangularSolidScene(
        image=ctx.image,
        annotation_type="bbox",
        annotation_keyed_points={},
        annotation_keyed_bboxes=dict(annotation_bboxes),
        annotation_roles=FRAME_ANNOTATION_KEYS,
        label_bboxes=dict(label_bboxes),
        scene_entities=scene_entities,
        render_map=render_map,
    )


def render_cuboid_measure_scene(
    ctx: RenderContext,
    problem: CuboidMeasureProblem,
    *,
    instance_seed: int,
) -> RenderedRectangularSolidScene:
    """Render a labeled cuboid for volume or surface-area measurement."""

    rng = spawn_rng(int(instance_seed), f"{problem.formula_family}.render.scene")
    length = float(problem.length)
    width_units = float(problem.width)
    height_units = float(problem.height)
    depth_vec = (0.64 * width_units, -0.42 * width_units)

    local_points = {
        "front_bottom_left": (0.0, 0.0),
        "front_bottom_right": (length, 0.0),
        "back_bottom_left": depth_vec,
        "back_bottom_right": (length + depth_vec[0], depth_vec[1]),
        "front_top_left": (0.0, -height_units),
        "front_top_right": (length, -height_units),
        "back_top_left": (depth_vec[0], depth_vec[1] - height_units),
        "back_top_right": (length + depth_vec[0], depth_vec[1] - height_units),
    }
    min_x = min(point[0] for point in local_points.values())
    max_x = max(point[0] for point in local_points.values())
    min_y = min(point[1] for point in local_points.values())
    max_y = max(point[1] for point in local_points.values())
    span_x = max(1e-6, max_x - min_x)
    span_y = max(1e-6, max_y - min_y)
    scale = min((ctx.width - 230.0) / span_x, (ctx.height - 210.0) / span_y)
    scale *= float(rng.uniform(0.84, 0.93))
    local_center = ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0)
    target_center = (
        (ctx.width / 2.0) + float(rng.uniform(-26.0, 26.0)),
        (ctx.height / 2.0) + float(rng.uniform(-18.0, 18.0)),
    )

    def transform(point: Point) -> Point:
        return (
            (float(point[0]) - float(local_center[0])) * float(scale) + float(target_center[0]),
            (float(point[1]) - float(local_center[1])) * float(scale) + float(target_center[1]),
        )

    points = {key: transform(point) for key, point in local_points.items()}
    fbl = points["front_bottom_left"]
    fbr = points["front_bottom_right"]
    bbl = points["back_bottom_left"]
    bbr = points["back_bottom_right"]
    ftl = points["front_top_left"]
    ftr = points["front_top_right"]
    btl = points["back_top_left"]
    btr = points["back_top_right"]

    top_face = (ftl, ftr, btr, btl)
    side_face = (fbr, bbr, btr, ftr)
    front_face = (fbl, fbr, ftr, ftl)
    for polygon, fill in ((top_face, ctx.face_top), (side_face, ctx.face_side), (front_face, ctx.face_front)):
        ctx.draw.polygon(list(polygon), fill=fill)
        ctx.draw.line(list(polygon) + [polygon[0]], fill=ctx.line_color, width=ctx.line_width)
    for edge in ((bbl, bbr), (bbl, btl), (bbl, fbl), (bbr, btr)):
        ctx.draw.line(edge, fill=ctx.line_color, width=max(1, ctx.line_width - 1))

    labels = {
        "length": "L ?" if problem.target_role == "length" else f"L {int(problem.length)}",
        "width": "W ?" if problem.target_role == "width" else f"W {int(problem.width)}",
        "height": "H ?" if problem.target_role == "height" else f"H {int(problem.height)}",
    }
    label_bboxes: Dict[str, BBox] = {}
    length_start, length_end, label_bboxes["length"] = _draw_dimension(
        ctx,
        fbl,
        fbr,
        labels["length"],
        offset=(0.0, 34.0),
        label_offset=(0.0, 19.0),
        color=ctx.secondary_color,
    )
    width_start, width_end, label_bboxes["width"] = _draw_dimension(
        ctx,
        fbr,
        bbr,
        labels["width"],
        offset=(26.0, 15.0),
        label_offset=(24.0, 5.0),
        color=ctx.secondary_color,
    )
    height_start, height_end, label_bboxes["height"] = _draw_dimension(
        ctx,
        fbl,
        ftl,
        labels["height"],
        offset=(-34.0, 0.0),
        label_offset=(-27.0, 0.0),
        color=ctx.secondary_color,
    )

    readout_center = (ctx.width / 2.0, 50.0 + float(rng.uniform(-6.0, 8.0)))
    if problem.target_role == "surface_area":
        label_bboxes["surface_area"] = _draw_value_box(ctx, "Surface area ?", readout_center)
    else:
        label_bboxes["volume"] = _draw_value_box(ctx, f"Volume {int(problem.volume)}", readout_center)
    _assert_bboxes_inside(label_bboxes.values(), width=ctx.width, height=ctx.height)

    annotation = {
        "length_segment_start": length_start,
        "length_segment_end": length_end,
        "width_segment_start": width_start,
        "width_segment_end": width_end,
        "height_segment_start": height_start,
        "height_segment_end": height_end,
    }
    entity_bbox = bbox_from_points(tuple(points.values()), width=ctx.width, height=ctx.height, pad=10.0)
    scene_entities = (
        {
            "entity_id": "cuboid",
            "entity_type": "rectangular_prism",
            "length_units": int(problem.length),
            "width_units": int(problem.width),
            "height_units": int(problem.height),
            "volume_units": int(problem.volume),
            "surface_area_units": int(problem.surface_area),
            "bbox": bbox_to_list(entity_bbox),
        },
    )
    render_map = {
        "coord_space": "pixel",
        "target_role": str(problem.target_role),
        "cuboid_vertices": {key: _point_to_list(value) for key, value in points.items()},
        "dimension_segments": {
            "length": [_point_to_list(length_start), _point_to_list(length_end)],
            "width": [_point_to_list(width_start), _point_to_list(width_end)],
            "height": [_point_to_list(height_start), _point_to_list(height_end)],
        },
        "face_bboxes": {
            "front": bbox_to_list(bbox_from_points(front_face, width=ctx.width, height=ctx.height, pad=4.0)),
            "side": bbox_to_list(bbox_from_points(side_face, width=ctx.width, height=ctx.height, pad=4.0)),
            "top": bbox_to_list(bbox_from_points(top_face, width=ctx.width, height=ctx.height, pad=4.0)),
        },
        "cuboid_bbox": bbox_to_list(entity_bbox),
        "label_bboxes": {key: bbox_to_list(value) for key, value in label_bboxes.items()},
        "scale_px_per_unit": round(float(scale), 3),
    }
    return RenderedRectangularSolidScene(
        image=ctx.image,
        annotation_type="segment",
        annotation_keyed_points={key: tuple(value) for key, value in annotation.items()},
        annotation_keyed_bboxes={},
        annotation_roles=CUBOID_DIMENSION_ANNOTATION_KEYS,
        label_bboxes=dict(label_bboxes),
        scene_entities=scene_entities,
        render_map=render_map,
    )


__all__ = [
    "CUBOID_DIMENSION_ANNOTATION_KEYS",
    "FRAME_ANNOTATION_KEYS",
    "NET_ANNOTATION_KEYS",
    "create_render_context",
    "render_cube_frame_scene",
    "render_cuboid_measure_scene",
    "render_open_box_net_scene",
]
