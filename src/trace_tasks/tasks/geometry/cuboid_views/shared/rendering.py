"""Identity-free rendering primitives for cuboid orthographic-view scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

from PIL import Image, ImageDraw

from trace_tasks.core.sampling import uniform_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.geometry.shared.diagram_style import (
    geometry_diagram_style_metadata,
    prepare_geometry_diagram_style_and_background,
)
from trace_tasks.tasks.geometry.shared.measurement_rendering import bbox_to_list, draw_label
from trace_tasks.tasks.geometry.shared.shape_style import (
    extract_background_anchor_colors,
    sample_geometry_shape_style,
)
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.text_rendering import load_font

from .state import BBox, Color, CuboidDimensions, RenderedCuboidViewsScene

SCENE_ID = "cuboid_views"

NOISE_DEFAULTS: Dict[str, Any] = {
    "apply_prob": 0.40,
    "edit_types": ["blur", "downsample", "jpeg", "noise"],
    "edit_count_range": [1, 1],
    "value_ranges": {
        "blur": {"radius": [0.06, 0.18]},
        "downsample": {"scale": [0.96, 0.99]},
        "jpeg": {"quality": [90, 97]},
        "noise": {"alpha": [0.006, 0.018]},
    },
}

_PALETTES: Tuple[Tuple[Color, Color, Color, Color], ...] = (
    ((225, 239, 255), (238, 246, 236), (27, 113, 191), (160, 176, 190)),
    ((255, 237, 222), (236, 240, 255), (189, 91, 37), (164, 150, 136)),
    ((237, 232, 255), (235, 248, 246), (111, 92, 190), (158, 152, 178)),
    ((230, 247, 235), (255, 239, 219), (30, 132, 92), (144, 168, 150)),
)


@dataclass
class _RenderContext:
    rng: Any
    image: Image.Image
    draw: ImageDraw.ImageDraw
    width: int
    height: int
    line_color: Color
    label_color: Color
    label_stroke_color: Color
    label_stroke_width: int
    fill_color: Color
    secondary_fill_color: Color
    muted_color: Color
    line_width: int
    font: Any
    small_font: Any


def _rect_in_slot(unit_w: float, unit_h: float, slot: BBox, *, scale: float) -> BBox:
    x0, y0, x1, y1 = [float(value) for value in slot]
    slot_w = x1 - x0
    slot_h = y1 - y0
    rect_w = min(slot_w, float(unit_w) * scale)
    rect_h = min(slot_h, float(unit_h) * scale)
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    return (cx - rect_w / 2.0, cy - rect_h / 2.0, cx + rect_w / 2.0, cy + rect_h / 2.0)


def _draw_view_rect(
    ctx: _RenderContext,
    bbox: BBox,
    *,
    title: str,
    title_y: float,
    fill: Color,
) -> None:
    ctx.draw.rounded_rectangle(
        bbox,
        radius=4,
        fill=fill,
        outline=ctx.line_color,
        width=ctx.line_width,
    )
    draw_label(ctx, title, ((float(bbox[0]) + float(bbox[2])) / 2.0, title_y), small=True)


def _make_render_context(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
) -> tuple[_RenderContext, Dict[str, Any]]:
    """Create one deterministic drawing context; no task/query identity enters rendering."""

    rng = spawn_rng(int(instance_seed), "cuboid_views.render")
    width = int(params.get("canvas_width", group_default(render_defaults, "canvas_width", 820)))
    height = int(params.get("canvas_height", group_default(render_defaults, "canvas_height", 580)))
    image, background_meta, diagram_style, diagram_style_meta = prepare_geometry_diagram_style_and_background(
        canvas_width=int(width),
        canvas_height=int(height),
        scene_id=SCENE_ID,
        instance_seed=int(instance_seed),
        params=params,
        namespace_suffix="cuboid_views_background",
    )
    shape_style = sample_geometry_shape_style(
        rng,
        params=params,
        render_defaults=render_defaults,
        anchor_colors=extract_background_anchor_colors(background_meta),
    )
    palette_rng = spawn_rng(int(instance_seed), "cuboid_views.palette")
    fill_color, secondary_fill_color, _accent_color, muted_color = uniform_choice(
        palette_rng,
        _PALETTES,
    )
    font_size = int(params.get("label_font_size", group_default(render_defaults, "label_font_size", 22)))
    small_font_size = int(
        params.get("small_label_font_size", group_default(render_defaults, "small_label_font_size", 18))
    )
    line_width = int(params.get("line_width", group_default(render_defaults, "line_width", 4)))
    ctx = _RenderContext(
        rng=rng,
        image=image,
        draw=ImageDraw.Draw(image),
        width=int(width),
        height=int(height),
        line_color=shape_style.line_color,
        label_color=shape_style.label_color,
        label_stroke_color=shape_style.label_stroke_color,
        label_stroke_width=0,
        fill_color=fill_color,
        secondary_fill_color=secondary_fill_color,
        muted_color=muted_color,
        line_width=max(2, int(line_width)),
        font=load_font(max(12, int(font_size)), bold=False),
        small_font=load_font(max(10, int(small_font_size)), bold=False),
    )
    render_meta = {
        "background_style": dict(background_meta),
        "technical_diagram_style": geometry_diagram_style_metadata(diagram_style),
        "technical_diagram_style_resolution": dict(diagram_style_meta),
        "shape_style": shape_style.to_trace_dict(),
        "line_width": int(ctx.line_width),
        "label_font_size": int(font_size),
        "small_label_font_size": int(small_font_size),
        "fill_color": list(fill_color),
        "secondary_fill_color": list(secondary_fill_color),
        "muted_color": list(muted_color),
    }
    return ctx, render_meta


def render_cuboid_views_scene(
    *,
    dimensions: CuboidDimensions,
    instance_seed: int,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
) -> tuple[RenderedCuboidViewsScene, Dict[str, Any]]:
    """Render the three orthographic rectangles and return projected view bboxes."""

    ctx, render_meta = _make_render_context(
        instance_seed=int(instance_seed),
        params=params,
        render_defaults=render_defaults,
    )
    top_slot = (120.0, 64.0, 454.0, 222.0)
    front_slot = (120.0, 298.0, 454.0, 504.0)
    right_slot = (536.0, 298.0, 760.0, 504.0)
    scale = min(
        (top_slot[2] - top_slot[0]) / float(dimensions.length),
        (top_slot[3] - top_slot[1]) / float(dimensions.width),
        (front_slot[2] - front_slot[0]) / float(dimensions.length),
        (front_slot[3] - front_slot[1]) / float(dimensions.height),
        (right_slot[2] - right_slot[0]) / float(dimensions.width),
        (right_slot[3] - right_slot[1]) / float(dimensions.height),
    )
    top_rect = _rect_in_slot(dimensions.length, dimensions.width, top_slot, scale=scale)
    front_rect = _rect_in_slot(dimensions.length, dimensions.height, front_slot, scale=scale)
    right_rect = _rect_in_slot(dimensions.width, dimensions.height, right_slot, scale=scale)

    label_bboxes: Dict[str, BBox] = {}
    _draw_view_rect(ctx, top_rect, title="Top view", title_y=38.0, fill=ctx.secondary_fill_color)
    _draw_view_rect(ctx, front_rect, title="Front view", title_y=270.0, fill=ctx.fill_color)
    _draw_view_rect(ctx, right_rect, title="Right view", title_y=270.0, fill=ctx.fill_color)
    label_bboxes["top_perimeter"] = draw_label(
        ctx,
        f"P={dimensions.top_view_perimeter}",
        ((top_rect[0] + top_rect[2]) / 2.0, (top_rect[1] + top_rect[3]) / 2.0),
        small=True,
    )
    label_bboxes["front_perimeter"] = draw_label(
        ctx,
        f"P={dimensions.front_view_perimeter}",
        ((front_rect[0] + front_rect[2]) / 2.0, (front_rect[1] + front_rect[3]) / 2.0),
        small=True,
    )
    label_bboxes["right_perimeter"] = draw_label(
        ctx,
        f"P={dimensions.right_view_perimeter}",
        ((right_rect[0] + right_rect[2]) / 2.0, (right_rect[1] + right_rect[3]) / 2.0),
        small=True,
    )
    # Alignment guides visually bind the views while staying non-semantic.
    for x in (top_rect[0], top_rect[2]):
        ctx.draw.line([(x, top_rect[3] + 10.0), (x, front_rect[1] - 14.0)], fill=ctx.muted_color, width=2)
    for y in (front_rect[1], front_rect[3]):
        ctx.draw.line([(front_rect[2] + 16.0, y), (right_rect[0] - 16.0, y)], fill=ctx.muted_color, width=2)
    label_bboxes["target"] = draw_label(ctx, "SA=?", (642.0, 112.0), small=False)

    annotation_bboxes = {
        "top_view": top_rect,
        "front_view": front_rect,
        "right_view": right_rect,
    }
    scene_entities = (
        {
            "entity_id": "top_view",
            "entity_type": "orthographic_rectangle",
            "bbox": bbox_to_list(top_rect),
            "dimensions": ["length", "width"],
        },
        {
            "entity_id": "front_view",
            "entity_type": "orthographic_rectangle",
            "bbox": bbox_to_list(front_rect),
            "dimensions": ["length", "height"],
        },
        {
            "entity_id": "right_view",
            "entity_type": "orthographic_rectangle",
            "bbox": bbox_to_list(right_rect),
            "dimensions": ["width", "height"],
        },
    )
    return (
        RenderedCuboidViewsScene(
            image=ctx.image,
            annotation_bboxes=dict(annotation_bboxes),
            annotation_roles=("top_view", "front_view", "right_view"),
            label_bboxes=dict(label_bboxes),
            scene_entities=scene_entities,
            render_map={
                "views": {
                    "top": bbox_to_list(top_rect),
                    "front": bbox_to_list(front_rect),
                    "right": bbox_to_list(right_rect),
                },
                "label_bboxes": {key: bbox_to_list(value) for key, value in label_bboxes.items()},
                "coord_space": "pixel",
            },
        ),
        render_meta,
    )
