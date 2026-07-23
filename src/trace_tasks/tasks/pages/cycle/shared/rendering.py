"""Rendering for directed cycle page diagrams."""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, Sequence

from PIL import Image, ImageDraw

from trace_tasks.core.visual.background import make_background_canvas
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.pages.shared.diagram.common import (
    draw_diagram_text_in_box,
    resolve_diagram_panel_geometry,
    resolve_jittered_diagram_panel_geometry,
    resolve_diagrams_int_param,
    resolve_diagrams_rgb_triple,
    round_diagram_bbox,
)
from trace_tasks.tasks.shared.drawing import draw_arrow, draw_rounded_rect
from trace_tasks.tasks.shared.render_variation import resolve_layout_jitter

from .defaults import (
    NAMESPACE_ROOT,
    POST_IMAGE_BACKGROUND_DEFAULTS,
    POST_IMAGE_NOISE_DEFAULTS,
    RENDER_FALLBACKS,
    RENDERING_DEFAULTS,
)
from .state import BBox, CycleCase, CycleRenderParams, Point, RenderedCycleScene


def resolve_cycle_render_params(
    params: Mapping[str, object],
    *,
    instance_seed: int | None = None,
) -> CycleRenderParams:
    """Resolve render parameters for one cycle diagram."""

    def _int(key: str) -> int:
        return resolve_diagrams_int_param(
            params,
            RENDERING_DEFAULTS,
            key,
            int(RENDER_FALLBACKS[key]),
            instance_seed=instance_seed,
            namespace=NAMESPACE_ROOT,
        )

    def _triple(key: str) -> tuple[int, int, int]:
        return resolve_diagrams_rgb_triple(
            params,
            RENDERING_DEFAULTS,
            key,
            tuple(int(value) for value in RENDER_FALLBACKS[key]),
            instance_seed=instance_seed,
            namespace=NAMESPACE_ROOT,
        )

    layout_jitter_meta = resolve_layout_jitter(
        params,
        RENDERING_DEFAULTS,
        instance_seed=instance_seed,
        namespace=f"{NAMESPACE_ROOT}.layout",
    )
    return CycleRenderParams(
        canvas_width=_int("canvas_width"),
        canvas_height=_int("canvas_height"),
        outer_margin_px=_int("outer_margin_px"),
        panel_padding_px=_int("panel_padding_px"),
        panel_corner_radius_px=_int("panel_corner_radius_px"),
        title_font_size_px=_int("title_font_size_px"),
        title_band_height_px=_int("title_band_height_px"),
        node_width_px=_int("node_width_px"),
        node_height_px=_int("node_height_px"),
        node_corner_radius_px=_int("node_corner_radius_px"),
        node_border_width_px=_int("node_border_width_px"),
        ring_radius_x_px=_int("ring_radius_x_px"),
        ring_radius_y_px=_int("ring_radius_y_px"),
        edge_width_px=_int("edge_width_px"),
        arrow_head_length_px=_int("arrow_head_length_px"),
        arrow_head_width_px=_int("arrow_head_width_px"),
        label_font_size_px=_int("label_font_size_px"),
        panel_fill_rgb=_triple("panel_fill_rgb"),
        panel_border_rgb=_triple("panel_border_rgb"),
        title_color_rgb=_triple("title_color_rgb"),
        node_fill_rgb=_triple("node_fill_rgb"),
        node_border_rgb=_triple("node_border_rgb"),
        label_color_rgb=_triple("label_color_rgb"),
        label_stroke_rgb=_triple("label_stroke_rgb"),
        edge_color_rgb=_triple("edge_color_rgb"),
        layout_jitter_meta=dict(layout_jitter_meta),
    )


def _clip_point_to_bbox(center: Point, toward: Point, bbox: BBox) -> Point:
    """Move from a node center to its bbox boundary along one ray."""

    cx, cy = float(center[0]), float(center[1])
    tx, ty = float(toward[0]), float(toward[1])
    dx = float(tx - cx)
    dy = float(ty - cy)
    if abs(dx) <= 1e-6 and abs(dy) <= 1e-6:
        return (cx, cy)
    half_w = 0.5 * float(bbox[2] - bbox[0])
    half_h = 0.5 * float(bbox[3] - bbox[1])
    scale_x = float("inf") if abs(dx) <= 1e-6 else float(half_w / abs(dx))
    scale_y = float("inf") if abs(dy) <= 1e-6 else float(half_h / abs(dy))
    scale = min(float(scale_x), float(scale_y))
    return (float(cx + (dx * scale)), float(cy + (dy * scale)))


def _stage_bbox(center: Point, *, render_params: CycleRenderParams) -> BBox:
    """Resolve one stage box from its center."""

    cx, cy = float(center[0]), float(center[1])
    half_w = 0.5 * float(render_params.node_width_px)
    half_h = 0.5 * float(render_params.node_height_px)
    return (cx - half_w, cy - half_h, cx + half_w, cy + half_h)


def _edge_bbox(start: Point, end: Point) -> List[float]:
    """Return one directed stage arrow bbox."""

    return round_diagram_bbox(
        (
            min(float(start[0]), float(end[0])),
            min(float(start[1]), float(end[1])),
            max(float(start[0]), float(end[0])),
            max(float(start[1]), float(end[1])),
        )
    )


def _draw_cycle_on_background(
    background: Image.Image,
    *,
    scene_title: str,
    stage_specs: Sequence[Mapping[str, object]],
    edge_specs: Sequence[Mapping[str, object]],
    render_params: CycleRenderParams,
) -> RenderedCycleScene:
    """Draw the panel, directed edges, and stage boxes while preserving bbox ids."""

    image = background.copy()
    draw = ImageDraw.Draw(image)
    entities: List[Dict[str, object]] = []
    stage_bbox_map: Dict[str, List[float]] = {}
    stage_label_bbox_map: Dict[str, List[float]] = {}
    edge_bbox_map: Dict[str, List[float]] = {}

    if render_params.layout_jitter_meta:
        panel_bbox, title_bbox, content_bbox, layout_jitter_meta = resolve_jittered_diagram_panel_geometry(
            canvas_width=int(render_params.canvas_width),
            canvas_height=int(render_params.canvas_height),
            outer_margin_px=int(render_params.outer_margin_px),
            title_band_height_px=int(render_params.title_band_height_px),
            panel_padding_px=int(render_params.panel_padding_px),
            layout_jitter_meta=render_params.layout_jitter_meta,
        )
    else:
        panel_bbox, title_bbox, content_bbox = resolve_diagram_panel_geometry(
            canvas_width=int(render_params.canvas_width),
            canvas_height=int(render_params.canvas_height),
            outer_margin_px=int(render_params.outer_margin_px),
            title_band_height_px=int(render_params.title_band_height_px),
            panel_padding_px=int(render_params.panel_padding_px),
        )
        layout_jitter_meta = {}

    draw_rounded_rect(
        draw,
        panel_bbox,
        radius=int(render_params.panel_corner_radius_px),
        fill=render_params.panel_fill_rgb,
        outline=render_params.panel_border_rgb,
        width=2,
    )
    title_text_bbox = draw_diagram_text_in_box(
        draw,
        bbox=title_bbox,
        text=str(scene_title),
        font_size_px=int(render_params.title_font_size_px),
        bold=True,
        fill=render_params.title_color_rgb,
        stroke_fill=render_params.panel_fill_rgb,
        padding_px=12,
    )
    entities.append({"entity_id": "diagram_panel", "entity_type": "diagram_panel", "bbox_xyxy": round_diagram_bbox(panel_bbox)})
    entities.append(
        {
            "entity_id": "diagram_title",
            "entity_type": "diagram_title",
            "bbox_xyxy": list(title_text_bbox),
            "text": str(scene_title),
        }
    )

    content_left, content_top, content_right, content_bottom = [float(value) for value in content_bbox]
    content_cx = 0.5 * float(content_left + content_right)
    content_cy = 0.5 * float(content_top + content_bottom)
    usable_half_width = max(140.0, 0.5 * float(content_right - content_left) - (0.5 * float(render_params.node_width_px)) - 24.0)
    usable_half_height = max(110.0, 0.5 * float(content_bottom - content_top) - (0.5 * float(render_params.node_height_px)) - 24.0)
    radius_x = min(float(render_params.ring_radius_x_px), usable_half_width)
    radius_y = min(float(render_params.ring_radius_y_px), usable_half_height - 18.0)

    centers_by_stage_id: Dict[str, Point] = {}
    boxes_by_stage_id: Dict[str, BBox] = {}
    stage_count = max(1, len(stage_specs))
    for stage_spec in stage_specs:
        index = int(stage_spec["order_index"])
        angle = float((-0.5 * math.pi) + ((2.0 * math.pi * index) / float(stage_count)))
        center = (
            float(content_cx + (radius_x * math.cos(angle))),
            float(content_cy + (radius_y * math.sin(angle))),
        )
        stage_id = str(stage_spec["stage_id"])
        centers_by_stage_id[stage_id] = center
        boxes_by_stage_id[stage_id] = _stage_bbox(center, render_params=render_params)

    for edge_spec in edge_specs:
        source_id = str(edge_spec["source_stage_id"])
        target_id = str(edge_spec["target_stage_id"])
        source_center = centers_by_stage_id[source_id]
        target_center = centers_by_stage_id[target_id]
        source_box = boxes_by_stage_id[source_id]
        target_box = boxes_by_stage_id[target_id]
        start = _clip_point_to_bbox(source_center, target_center, source_box)
        end = _clip_point_to_bbox(target_center, source_center, target_box)
        draw_arrow(
            draw,
            start=start,
            end=end,
            fill=tuple(int(value) for value in render_params.edge_color_rgb),
            width=max(1, int(render_params.edge_width_px)),
            head_length_px=float(render_params.arrow_head_length_px),
            head_width_px=float(render_params.arrow_head_width_px),
        )
        edge_bbox = _edge_bbox(start, end)
        edge_bbox_map[str(edge_spec["edge_id"])] = edge_bbox
        entities.append(
            {
                "entity_id": str(edge_spec["edge_id"]),
                "entity_type": "diagram_edge",
                "bbox_xyxy": list(edge_bbox),
                "source_stage_id": source_id,
                "target_stage_id": target_id,
                "direction": str(edge_spec.get("direction", "")),
            }
        )

    for stage_spec in stage_specs:
        stage_id = str(stage_spec["stage_id"])
        stage_bbox_id = str(stage_spec["stage_bbox_id"])
        stage_label_bbox_id = str(stage_spec["stage_label_bbox_id"])
        stage_box = boxes_by_stage_id[stage_id]
        draw_rounded_rect(
            draw,
            stage_box,
            radius=int(render_params.node_corner_radius_px),
            fill=render_params.node_fill_rgb,
            outline=render_params.node_border_rgb,
            width=int(render_params.node_border_width_px),
        )
        label_bbox = draw_diagram_text_in_box(
            draw,
            bbox=stage_box,
            text=str(stage_spec["stage_label"]),
            font_size_px=int(render_params.label_font_size_px),
            bold=True,
            fill=render_params.label_color_rgb,
            stroke_fill=render_params.label_stroke_rgb,
            padding_px=8,
        )
        stage_bbox_map[stage_bbox_id] = round_diagram_bbox(stage_box)
        stage_label_bbox_map[stage_label_bbox_id] = list(label_bbox)
        entities.append(
            {
                "entity_id": stage_bbox_id,
                "entity_type": "diagram_stage",
                "bbox_xyxy": round_diagram_bbox(stage_box),
                "stage_id": stage_id,
                "text": str(stage_spec["stage_label"]),
                "order_index": int(stage_spec["order_index"]),
            }
        )
        entities.append(
            {
                "entity_id": stage_label_bbox_id,
                "entity_type": "diagram_stage_label",
                "bbox_xyxy": list(label_bbox),
                "stage_id": stage_id,
                "text": str(stage_spec["stage_label"]),
            }
        )

    return RenderedCycleScene(
        image=image,
        render_params=render_params,
        entities=entities,
        panel_bbox_px=round_diagram_bbox(panel_bbox),
        title_bbox_px=list(title_text_bbox),
        stage_bbox_map=stage_bbox_map,
        stage_label_bbox_map=stage_label_bbox_map,
        edge_bbox_map=edge_bbox_map,
        layout_jitter_meta=dict(layout_jitter_meta),
        background_meta={},
        post_noise_meta={},
    )


def render_cycle_case(*, instance_seed: int, params: Mapping[str, object], case: CycleCase) -> RenderedCycleScene:
    """Render a sampled cycle case and attach post-render metadata."""

    render_params = resolve_cycle_render_params(params, instance_seed=int(instance_seed))
    background, background_meta = make_background_canvas(
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_BACKGROUND_DEFAULTS,
    )
    rendered = _draw_cycle_on_background(
        background,
        scene_title=str(case.scene_title),
        stage_specs=tuple(case.stage_specs),
        edge_specs=tuple(case.edge_specs),
        render_params=render_params,
    )
    image, post_noise_meta = apply_post_image_noise(
        rendered.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    return RenderedCycleScene(
        image=image,
        render_params=render_params,
        entities=list(rendered.entities),
        panel_bbox_px=list(rendered.panel_bbox_px),
        title_bbox_px=list(rendered.title_bbox_px),
        stage_bbox_map=dict(rendered.stage_bbox_map),
        stage_label_bbox_map=dict(rendered.stage_label_bbox_map),
        edge_bbox_map=dict(rendered.edge_bbox_map),
        layout_jitter_meta=dict(rendered.layout_jitter_meta),
        background_meta=dict(background_meta),
        post_noise_meta=dict(post_noise_meta),
    )
