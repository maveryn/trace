"""Rendering primitives for the sunburst chart scene."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from trace_tasks.tasks.charts.shared.information_style import prepare_chart_information_scene
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.charts.shared.visual_defaults import chart_font_asset_metadata, sample_chart_font_family
from trace_tasks.tasks.shared.bbox_projection import bbox_union_raw
from trace_tasks.tasks.shared.text_legibility import (
    READ_REQUIRED_TEXT_MIN_CONTRAST_RATIO,
    READ_REQUIRED_TEXT_MIN_LAB_DISTANCE,
    draw_text_traced,
    resolve_readable_text_style,
)
from trace_tasks.tasks.shared.text_rendering import load_font, temporary_default_font_family

from .defaults import POST_IMAGE_NOISE_DEFAULTS, canvas_size, generation_value, render_int, render_rgb, rendering_value
from .sampling import descendant_leaf_ids, lighten, nodes_by_id
from .state import BBox, RGB, RenderedSunburst, RenderParams, SunburstNode, SunburstTree


SUNBURST_LABEL_FILL_RGB: RGB = (22, 28, 38)
SUNBURST_LABEL_STROKE_RGB: RGB = (248, 252, 255)
SUNBURST_CHART_FONT_FAMILY_WEIGHTS = {
    "roboto": 1.0,
    "source_sans_3": 1.0,
    "nunito_sans": 1.0,
    "fira_sans": 1.0,
    "barlow": 1.0,
    "karla": 1.0,
    "cabin": 1.0,
}


def render_scene(
    tree: SunburstTree,
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    font_namespace: str,
) -> RenderedSunburst:
    """Apply chart-wide visual variation before drawing the deterministic tree."""

    resolved_params = _render_params(params, instance_seed=int(instance_seed))
    protected_colors = [tuple(int(channel) for channel in node.color_rgb) for node in tree.nodes]
    render_params, background, background_meta, information_style_meta = prepare_chart_information_scene(
        instance_seed=int(instance_seed),
        params=dict(params),
        scene_id="sunburst",
        render_params=resolved_params,
        protected_colors=protected_colors,
    )
    font_params = dict(params)
    font_params.setdefault("chart_font_family_weights", dict(SUNBURST_CHART_FONT_FAMILY_WEIGHTS))
    chart_font_family = sample_chart_font_family(
        instance_seed=int(instance_seed),
        namespace=str(font_namespace),
        params=font_params,
    )
    with temporary_default_font_family(str(chart_font_family)):
        rendered = _render_sunburst(
            background,
            tree=tree,
            params=params,
            instance_seed=int(instance_seed),
            render_params=render_params,
        )
    image, post_noise_meta = apply_post_image_noise(
        rendered.image,
        instance_seed=int(instance_seed),
        params=dict(params),
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    return RenderedSunburst(
        image=image,
        entities=rendered.entities,
        node_traces=rendered.node_traces,
        leaf_value_bbox_by_node_id=rendered.leaf_value_bbox_by_node_id,
        chart_bbox_px=rendered.chart_bbox_px,
        render_meta={
            **dict(rendered.render_meta),
            "background_style": {**dict(background_meta), "information_scene_style": dict(information_style_meta)},
            "information_scene_style": dict(information_style_meta),
            "post_image_noise": dict(post_noise_meta),
        },
        background_meta={**dict(background_meta), "information_scene_style": dict(information_style_meta)},
        post_noise_meta=dict(post_noise_meta),
        font_assets=chart_font_asset_metadata(str(chart_font_family)),
    )


def _render_sunburst(
    background: Image.Image,
    *,
    tree: SunburstTree,
    params: Mapping[str, Any],
    instance_seed: int,
    render_params: RenderParams | None = None,
) -> RenderedSunburst:
    """Draw concentric rings and record leaf-value bboxes for annotation."""

    render_params = render_params or _render_params(params, instance_seed=int(instance_seed))
    image = background.convert("RGB")
    draw = ImageDraw.Draw(image)
    center = (float(render_params.center_x_px), float(render_params.center_y_px))
    chart_bbox = _bbox(
        (
            center[0] - render_params.leaf_outer_radius_px,
            center[1] - render_params.leaf_outer_radius_px,
            center[0] + render_params.leaf_outer_radius_px,
            center[1] + render_params.leaf_outer_radius_px,
        )
    )
    panel_pad = 34
    panel_bbox = [
        chart_bbox[0] - panel_pad,
        chart_bbox[1] - panel_pad,
        chart_bbox[2] + panel_pad,
        chart_bbox[3] + panel_pad,
    ]
    draw.rounded_rectangle(
        panel_bbox,
        radius=8,
        fill=render_params.panel_fill_rgb,
        outline=render_params.panel_border_rgb,
        width=2,
    )

    parent_font = _font(render_params.parent_font_size_px, bold=False)
    subgroup_font = _font(render_params.subgroup_font_size_px, bold=False)
    leaf_font = _font(render_params.leaf_font_size_px, bold=False)
    value_font = _font(render_params.value_font_size_px, bold=False)

    node_lookup = nodes_by_id(tree)
    spans = _node_angle_spans(tree)
    node_traces: list[dict[str, Any]] = []
    entities: list[dict[str, Any]] = []
    leaf_value_bbox_by_node_id: dict[str, BBox] = {}
    parent_label_max_chars = int(generation_value(params, "sunburst_parent_label_max_chars", 10))
    subgroup_label_max_chars = int(generation_value(params, "sunburst_subgroup_label_max_chars", 8))
    leaf_label_max_chars = int(generation_value(params, "sunburst_leaf_label_max_chars", 7))

    for level in ("leaf", "subgroup", "parent"):
        for node in [item for item in tree.nodes if item.level == level]:
            start, end = spans[str(node.node_id)]
            inner_radius, outer_radius = _ring_for_level(render_params, str(level))
            fill = _display_fill_for_node(node)
            _draw_ring_wedge(
                draw,
                center=center,
                inner_radius=int(inner_radius),
                outer_radius=int(outer_radius),
                start_angle=float(start),
                end_angle=float(end),
                fill=fill,
                hole_fill=render_params.plot_fill_rgb,
                outline=render_params.separator_rgb,
            )

    draw.ellipse(
        (
            center[0] - render_params.inner_radius_px,
            center[1] - render_params.inner_radius_px,
            center[0] + render_params.inner_radius_px,
            center[1] + render_params.inner_radius_px,
        ),
        fill=lighten(node_lookup["root"].color_rgb, 0.30),
        outline=render_params.separator_rgb,
        width=2,
    )

    for node in tree.nodes:
        if node.level == "root":
            continue
        start, end = spans[str(node.node_id)]
        inner_radius, outer_radius = _ring_for_level(render_params, str(node.level))
        label_xy = _label_center(
            center=center,
            inner_radius=int(inner_radius),
            outer_radius=int(outer_radius),
            start_angle=float(start),
            end_angle=float(end),
        )
        sweep = abs(float(end) - float(start))
        if node.level == "parent":
            label_lines = [_truncate_label(node.label, max_chars=int(parent_label_max_chars))]
            font = parent_font
        elif node.level == "subgroup":
            label_lines = [_truncate_label(node.label, max_chars=int(subgroup_label_max_chars))]
            font = subgroup_font
        else:
            label_lines = [_truncate_label(node.label, max_chars=int(leaf_label_max_chars)), str(int(node.value))]
            font = leaf_font if sweep >= 16.0 else value_font
        text_boxes = _draw_multiline_centered_text(
            draw,
            label_xy,
            label_lines,
            font=font,
            fill=SUNBURST_LABEL_FILL_RGB,
            stroke_fill=SUNBURST_LABEL_STROKE_RGB,
            stroke_width=0,
            line_gap_px=2,
            surface_rgb=_display_fill_for_node(node),
            instance_seed=int(instance_seed),
            namespace=f"charts.sunburst.label.{node.node_id}",
        )
        text_bbox = _bbox_union(text_boxes)
        value_bbox = list(text_boxes[-1]) if node.level == "leaf" and text_boxes else list(text_bbox)
        if node.level == "leaf":
            leaf_value_bbox_by_node_id[str(node.node_id)] = list(value_bbox)
        wedge_bbox = _bbox(
            (
                center[0] - outer_radius,
                center[1] - outer_radius,
                center[0] + outer_radius,
                center[1] + outer_radius,
            )
        )
        trace = {
            "node_id": str(node.node_id),
            "entity_id": str(node.node_id),
            "label": str(node.label),
            "level": str(node.level),
            "parent_id": str(node.parent_id) if node.parent_id is not None else None,
            "child_ids": [str(child_id) for child_id in node.child_ids],
            "value": int(node.value),
            "start_angle_deg": round(float(start), 3),
            "end_angle_deg": round(float(end), 3),
            "label_center_px": [round(float(label_xy[0]), 3), round(float(label_xy[1]), 3)],
            "label_bbox_px": list(text_bbox),
            "value_bbox_px": list(value_bbox) if node.level == "leaf" else None,
            "ring_bbox_px": list(wedge_bbox),
            "fill_rgb": [int(channel) for channel in node.color_rgb],
        }
        node_traces.append(dict(trace))
        entities.append(
            {
                "entity_id": str(node.node_id),
                "entity_type": f"sunburst_{node.level}",
                "bbox_xyxy": list(text_bbox),
                "attrs": dict(trace),
            }
        )

    return RenderedSunburst(
        image=image,
        entities=tuple(dict(entity) for entity in entities),
        node_traces=tuple(dict(trace) for trace in node_traces),
        leaf_value_bbox_by_node_id=dict(leaf_value_bbox_by_node_id),
        chart_bbox_px=list(chart_bbox),
        render_meta={
            "not_to_scale": True,
            "center_px": [int(render_params.center_x_px), int(render_params.center_y_px)],
            "radii_px": {
                "inner": int(render_params.inner_radius_px),
                "parent_outer": int(render_params.parent_outer_radius_px),
                "subgroup_outer": int(render_params.subgroup_outer_radius_px),
                "leaf_outer": int(render_params.leaf_outer_radius_px),
            },
            "value_display_policy": "outer_leaf_values_only",
            "label_text_style": {
                "fill_policy": "per_wedge_readable_text_style",
                "stroke_width_px": 0,
                "font_weight": "regular",
            },
        },
        background_meta={},
        post_noise_meta={},
        font_assets={},
    )


def _render_params(params: Mapping[str, Any], *, instance_seed: int) -> RenderParams:
    canvas_width, canvas_height = canvas_size(params)
    base_center_x = int(rendering_value(params, "center_x_px", canvas_width // 2))
    base_center_y = int(rendering_value(params, "center_y_px", canvas_height // 2 + 18))
    center_x = base_center_x + render_int(params, "center_x_jitter_px", 0, instance_seed=int(instance_seed))
    center_y = base_center_y + render_int(params, "center_y_jitter_px", 0, instance_seed=int(instance_seed))
    return RenderParams(
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        center_x_px=int(center_x),
        center_y_px=int(center_y),
        inner_radius_px=render_int(params, "inner_radius_px", 94, instance_seed=int(instance_seed)),
        parent_outer_radius_px=render_int(params, "parent_outer_radius_px", 210, instance_seed=int(instance_seed)),
        subgroup_outer_radius_px=render_int(params, "subgroup_outer_radius_px", 334, instance_seed=int(instance_seed)),
        leaf_outer_radius_px=render_int(params, "leaf_outer_radius_px", 456, instance_seed=int(instance_seed)),
        parent_font_size_px=render_int(params, "parent_font_size_px", 18, instance_seed=int(instance_seed)),
        subgroup_font_size_px=render_int(params, "subgroup_font_size_px", 16, instance_seed=int(instance_seed)),
        leaf_font_size_px=render_int(params, "leaf_font_size_px", 14, instance_seed=int(instance_seed)),
        value_font_size_px=render_int(params, "value_font_size_px", 15, instance_seed=int(instance_seed)),
        panel_fill_rgb=render_rgb(params, "panel_fill_rgb", (252, 253, 250), instance_seed=int(instance_seed)),
        panel_border_rgb=render_rgb(params, "panel_border_rgb", (198, 204, 210), instance_seed=int(instance_seed)),
        plot_fill_rgb=render_rgb(params, "plot_fill_rgb", (255, 255, 255), instance_seed=int(instance_seed)),
        text_color_rgb=render_rgb(params, "text_color_rgb", (32, 37, 44), instance_seed=int(instance_seed)),
        separator_rgb=render_rgb(params, "separator_rgb", (255, 255, 255), instance_seed=int(instance_seed)),
        text_stroke_rgb=render_rgb(params, "text_stroke_rgb", (255, 255, 255), instance_seed=int(instance_seed)),
        label_stroke_width_px=render_int(params, "label_stroke_width_px", 1, instance_seed=int(instance_seed)),
    )


def _node_angle_spans(tree: SunburstTree) -> dict[str, tuple[float, float]]:
    node_lookup = nodes_by_id(tree)
    spans: dict[str, tuple[float, float]] = {"root": (0.0, 360.0)}
    cursor = -90.0
    total_leaf_count = max(1, len(tree.leaf_ids))
    degrees_per_leaf = 360.0 / float(total_leaf_count)
    for parent_id in tree.parent_ids:
        parent_leaf_count = len(descendant_leaf_ids(node_lookup, parent_id))
        parent_start = cursor
        parent_end = parent_start + float(parent_leaf_count) * float(degrees_per_leaf)
        spans[parent_id] = (float(parent_start), float(parent_end))
        sub_cursor = float(parent_start)
        for subgroup_id in node_lookup[parent_id].child_ids:
            subgroup_leaf_count = len(descendant_leaf_ids(node_lookup, subgroup_id))
            subgroup_start = sub_cursor
            subgroup_end = subgroup_start + float(subgroup_leaf_count) * float(degrees_per_leaf)
            spans[str(subgroup_id)] = (float(subgroup_start), float(subgroup_end))
            leaf_cursor = float(subgroup_start)
            for leaf_id in node_lookup[str(subgroup_id)].child_ids:
                spans[str(leaf_id)] = (float(leaf_cursor), float(leaf_cursor + degrees_per_leaf))
                leaf_cursor += float(degrees_per_leaf)
            sub_cursor = float(subgroup_end)
        cursor = float(parent_end)
    return spans


def _ring_for_level(render_params: RenderParams, level: str) -> tuple[int, int]:
    if str(level) == "parent":
        return int(render_params.inner_radius_px), int(render_params.parent_outer_radius_px)
    if str(level) == "subgroup":
        return int(render_params.parent_outer_radius_px), int(render_params.subgroup_outer_radius_px)
    if str(level) == "leaf":
        return int(render_params.subgroup_outer_radius_px), int(render_params.leaf_outer_radius_px)
    return 0, int(render_params.inner_radius_px)


def _draw_ring_wedge(
    draw: ImageDraw.ImageDraw,
    *,
    center: tuple[float, float],
    inner_radius: int,
    outer_radius: int,
    start_angle: float,
    end_angle: float,
    fill: RGB,
    hole_fill: RGB,
    outline: RGB,
) -> None:
    cx, cy = float(center[0]), float(center[1])
    outer_box = (cx - outer_radius, cy - outer_radius, cx + outer_radius, cy + outer_radius)
    inner_box = (cx - inner_radius, cy - inner_radius, cx + inner_radius, cy + inner_radius)
    draw.pieslice(outer_box, start=float(start_angle), end=float(end_angle), fill=fill, outline=outline, width=2)
    if int(inner_radius) > 0:
        draw.pieslice(inner_box, start=float(start_angle), end=float(end_angle), fill=hole_fill)


def _label_center(
    *,
    center: tuple[float, float],
    inner_radius: int,
    outer_radius: int,
    start_angle: float,
    end_angle: float,
) -> tuple[float, float]:
    angle = math.radians((float(start_angle) + float(end_angle)) / 2.0)
    radius = (float(inner_radius) + float(outer_radius)) / 2.0
    return float(center[0]) + math.cos(angle) * radius, float(center[1]) + math.sin(angle) * radius


def _draw_multiline_centered_text(
    draw: ImageDraw.ImageDraw,
    center: tuple[float, float],
    lines: Sequence[str],
    *,
    font: ImageFont.ImageFont,
    fill: RGB,
    stroke_fill: RGB,
    stroke_width: int,
    line_gap_px: int = 2,
    surface_rgb: RGB = (255, 255, 255),
    instance_seed: int = 0,
    namespace: str = "charts.sunburst.label",
) -> list[BBox]:
    clean_lines = [str(line) for line in lines if str(line)]
    if not clean_lines:
        return [[float(center[0]), float(center[1]), float(center[0]), float(center[1])]]
    line_boxes = [draw.textbbox((0, 0), line, font=font, stroke_width=max(0, int(stroke_width))) for line in clean_lines]
    heights = [float(box[3] - box[1]) for box in line_boxes]
    total_h = sum(heights) + max(0, len(clean_lines) - 1) * int(line_gap_px)
    y = float(center[1]) - total_h / 2.0
    boxes: list[BBox] = []
    for line_index, (line, height) in enumerate(zip(clean_lines, heights)):
        xy = (float(center[0]), float(y + height / 2.0))
        style = resolve_readable_text_style(
            instance_seed=int(instance_seed),
            namespace=f"{namespace}.{line_index}",
            role="readout",
            surface_rgbs=(tuple(int(channel) for channel in surface_rgb),),
            preferred_rgbs=(tuple(int(channel) for channel in fill), tuple(int(channel) for channel in stroke_fill)),
            min_contrast_ratio=READ_REQUIRED_TEXT_MIN_CONTRAST_RATIO,
            min_lab_distance=READ_REQUIRED_TEXT_MIN_LAB_DISTANCE,
            required=True,
        )
        record = draw_text_traced(
            draw,
            xy,
            line,
            font=font,
            fill=style.fill_rgb,
            anchor="mm",
            stroke_fill=style.stroke_rgb,
            stroke_width=max(0, int(stroke_width)),
            role="readout",
            required=True,
            extra_metadata={**style.metadata(), "surface_rgb": [int(channel) for channel in surface_rgb]},
        )
        boxes.append([round(float(value), 3) for value in record["bbox_px"]])
        y += float(height) + int(line_gap_px)
    return boxes


def _display_fill_for_node(node: SunburstNode) -> RGB:
    fill = tuple(int(channel) for channel in node.color_rgb)
    if str(node.level) == "parent":
        return lighten(fill, 0.55)
    if str(node.level) == "subgroup":
        return lighten(fill, 0.62)
    if str(node.level) == "leaf":
        return lighten(fill, 0.70)
    return fill


def _truncate_label(label: str, *, max_chars: int) -> str:
    text = str(label)
    if len(text) <= int(max_chars):
        return text
    return text[: max(1, int(max_chars) - 1)] + "."


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    return load_font(max(8, int(size)), bold=bool(bold))


def _bbox(values: Sequence[float]) -> BBox:
    return [round(float(value), 3) for value in values]


def _bbox_union(values: Sequence[Sequence[float]]) -> BBox:
    return [round(float(value), 3) for value in bbox_union_raw(values)]


__all__ = ["render_scene"]
