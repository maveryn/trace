"""Rendering primitives for equivalent circuit diagrams."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Tuple

from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.physics.shared.circuit_scene import render_component_network_scene
from trace_tasks.tasks.physics.shared.diagram_style import prepare_physics_diagram_style_and_background
from trace_tasks.tasks.physics.shared.visual_defaults import load_physics_noise_defaults
from trace_tasks.tasks.shared.font_assets import sample_font_family
from trace_tasks.tasks.shared.render_variation import resolve_layout_jitter, resolve_render_int

from .annotations import clip_bbox, normalize_component_bbox_map
from .state import (
    DEFAULT_RENDERING,
    SCENE_ID,
    EquivalentCircuitLayout,
    EquivalentCircuitScenario,
    RenderedEquivalentCircuitScene,
)


POST_IMAGE_NOISE_DEFAULTS = load_physics_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)


def render_defaults(
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    *,
    instance_seed: int,
    namespace: str,
) -> Dict[str, int]:
    """Resolve diagram dimensions, stroke widths, and label sizes."""

    keys = (
        "canvas_width",
        "canvas_height",
        "terminal_left_x_px",
        "terminal_radius_px",
        "terminal_font_size_px",
        "wire_width_px",
        "component_symbol_width_px",
        "component_symbol_height_px",
        "component_label_font_size_px",
        "label_stroke_width_px",
        "parallel_rail_left_x_px",
        "parallel_branch_top_y_px",
        "parallel_branch_bottom_y_px",
    )
    fallback_by_key = {
        "canvas_width": DEFAULT_RENDERING.canvas_width,
        "canvas_height": DEFAULT_RENDERING.canvas_height,
        "terminal_left_x_px": DEFAULT_RENDERING.terminal_left_x_px,
        "terminal_radius_px": DEFAULT_RENDERING.terminal_radius_px,
        "terminal_font_size_px": DEFAULT_RENDERING.terminal_font_size_px,
        "wire_width_px": DEFAULT_RENDERING.wire_width_px,
        "component_symbol_width_px": DEFAULT_RENDERING.component_symbol_width_px,
        "component_symbol_height_px": DEFAULT_RENDERING.component_symbol_height_px,
        "component_label_font_size_px": DEFAULT_RENDERING.component_label_font_size_px,
        "label_stroke_width_px": DEFAULT_RENDERING.label_stroke_width_px,
        "parallel_rail_left_x_px": DEFAULT_RENDERING.parallel_rail_left_x_px,
        "parallel_branch_top_y_px": DEFAULT_RENDERING.parallel_branch_top_y_px,
        "parallel_branch_bottom_y_px": DEFAULT_RENDERING.parallel_branch_bottom_y_px,
    }
    return {
        key: int(
            resolve_render_int(
                params,
                defaults,
                key,
                int(fallback_by_key[key]),
                instance_seed=int(instance_seed),
                namespace=str(namespace),
            )
        )
        for key in keys
    }


def circuit_content_bbox(
    *,
    resolved_defaults: Mapping[str, int],
) -> list[float]:
    """Return the default circuit content box before whole-scene jitter."""

    canvas_width = int(resolved_defaults["canvas_width"])
    canvas_height = int(resolved_defaults["canvas_height"])
    mid_y = float(0.5 * canvas_height)
    left = float(resolved_defaults["terminal_left_x_px"]) - float(resolved_defaults["terminal_radius_px"]) - 26.0
    right = float(canvas_width - int(resolved_defaults["terminal_left_x_px"])) + float(resolved_defaults["terminal_radius_px"]) + 26.0
    top = min(
        float(resolved_defaults["parallel_branch_top_y_px"]) - 82.0,
        float(mid_y - float(resolved_defaults["terminal_radius_px"]) - 52.0),
    )
    bottom = float(resolved_defaults["parallel_branch_bottom_y_px"]) + 48.0
    return [round(left, 3), round(top, 3), round(right, 3), round(bottom, 3)]


def resolve_layout_placement(
    *,
    resolved_defaults: Mapping[str, int],
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
) -> Tuple[Tuple[float, float], Dict[str, Any]]:
    """Resolve whole-circuit offset and describe the final projected region."""

    canvas_width = int(resolved_defaults["canvas_width"])
    canvas_height = int(resolved_defaults["canvas_height"])
    content_bbox = circuit_content_bbox(resolved_defaults=resolved_defaults)
    content_left, content_top, content_right, content_bottom = [float(value) for value in content_bbox]
    jitter = resolve_layout_jitter(
        params,
        defaults,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.layout",
    )
    min_margin = int(jitter.get("min_margin_px", 18))
    requested_dx = int(jitter.get("requested_dx_px", 0))
    requested_dy = int(jitter.get("requested_dy_px", 0))
    min_dx = int(math.ceil(float(min_margin) - float(content_left)))
    max_dx = int(math.floor(float(canvas_width) - float(min_margin) - float(content_right)))
    min_dy = int(math.ceil(float(min_margin) - float(content_top)))
    max_dy = int(math.floor(float(canvas_height) - float(min_margin) - float(content_bottom)))
    if int(min_dx) > int(max_dx):
        min_dx = 0
        max_dx = 0
    if int(min_dy) > int(max_dy):
        min_dy = 0
        max_dy = 0
    if not bool(jitter.get("enabled", False)):
        requested_dx = 0
        requested_dy = 0
    dx = max(int(min_dx), min(int(max_dx), int(requested_dx)))
    dy = max(int(min_dy), min(int(max_dy), int(requested_dy)))
    content_width = round(float(content_right) - float(content_left), 3)
    content_height = round(float(content_bottom) - float(content_top), 3)
    final_bbox = [
        round(float(content_left) + float(dx), 3),
        round(float(content_top) + float(dy), 3),
        round(float(content_right) + float(dx), 3),
        round(float(content_bottom) + float(dy), 3),
    ]
    placement = dict(jitter)
    placement.update(
        {
            "mode": "whole_equivalent_circuit_diagram_offset",
            "content_bbox_px": list(content_bbox),
            "content_size_px": [float(content_width), float(content_height)],
            "final_content_bbox_px": list(final_bbox),
            "canvas_size_px": [int(canvas_width), int(canvas_height)],
            "free_space_px": [
                round(float(canvas_width) - float(content_width), 3),
                round(float(canvas_height) - float(content_height), 3),
            ],
            "available_offset_x_px": [int(min_dx), int(max_dx)],
            "available_offset_y_px": [int(min_dy), int(max_dy)],
            "sampled_offset_px": [int(requested_dx), int(requested_dy)],
            "final_offset_px": [int(dx), int(dy)],
            "default_origin_px": [round(float(content_left), 3), round(float(content_top), 3)],
            "final_origin_px": [
                round(float(content_left) + float(dx), 3),
                round(float(content_top) + float(dy), 3),
            ],
            "dx_px": int(dx),
            "dy_px": int(dy),
        }
    )
    return (float(dx), float(dy)), placement


def render_equivalent_circuit(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    scenario: EquivalentCircuitScenario,
    layout: EquivalentCircuitLayout,
    render_config: Mapping[str, Any],
    namespace: str,
) -> RenderedEquivalentCircuitScene:
    """Render one equivalent circuit and normalize component annotations."""

    resolved_defaults = render_defaults(
        params,
        render_config,
        instance_seed=int(instance_seed),
        namespace=str(namespace),
    )
    origin_offset_px, layout_meta = resolve_layout_placement(
        resolved_defaults=resolved_defaults,
        params=params,
        defaults=render_config,
        instance_seed=int(instance_seed),
        namespace=str(namespace),
    )
    background, background_meta, diagram_style, diagram_style_meta = prepare_physics_diagram_style_and_background(
        scene_id=SCENE_ID,
        canvas_width=int(resolved_defaults["canvas_width"]),
        canvas_height=int(resolved_defaults["canvas_height"]),
        instance_seed=int(instance_seed),
        params=params,
    )
    font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.render.font",
        params=params,
    )
    rendered_scene = render_component_network_scene(
        scene_variant=str(layout.scene_variant),
        component_kind=str(layout.component_kind),
        series_values=list(layout.series_values),
        parallel_values=list(layout.parallel_values),
        parallel_blocks=list(layout.parallel_blocks) or None,
        inter_block_series_values=list(layout.inter_block_series_values) or None,
        outer_series_values=list(layout.outer_series_values),
        background=background,
        render_defaults=resolved_defaults,
        accent_color_name=str(scenario.accent_color_name),
        origin_offset_px=origin_offset_px,
        diagram_style=diagram_style,
        font_family=str(font_family),
    )
    image, post_noise_meta = apply_post_image_noise(
        rendered_scene.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    component_prefix = "R" if str(scenario.component_kind) == "resistor" else "C"
    annotation_bbox_map = normalize_component_bbox_map(
        rendered_scene.annotation_bbox_map,
        component_prefix=component_prefix,
        width=int(image.size[0]),
        height=int(image.size[1]),
    )
    render_map = dict(rendered_scene.render_map)
    render_map.pop("annotation_bbox_map_px", None)
    annotation_bbox = clip_bbox(
        layout_meta["final_content_bbox_px"],
        width=int(image.size[0]),
        height=int(image.size[1]),
    )
    render_map["component_bboxes_px"] = dict(annotation_bbox_map)
    render_map["annotation_bbox_px"] = list(annotation_bbox)
    render_spec = {
        "scene_variant": str(scenario.scene_variant),
        "component_kind": str(scenario.component_kind),
        "component_symbol_style": "ansi_zigzag" if str(scenario.component_kind) == "resistor" else "parallel_plates",
        "canvas_width": int(image.size[0]),
        "canvas_height": int(image.size[1]),
        "accent_color_name": str(scenario.accent_color_name),
        "technical_diagram_style": dict(diagram_style_meta),
        "background_style": dict(background_meta),
        "layout_placement": dict(layout_meta),
        "post_image_noise": dict(post_noise_meta),
    }
    return RenderedEquivalentCircuitScene(
        image=image,
        component_specs=list(rendered_scene.component_specs),
        annotation_bbox=list(annotation_bbox),
        annotation_bbox_map=dict(annotation_bbox_map),
        annotation_entity_id_map=dict(rendered_scene.annotation_entity_id_map),
        scene_entities=list(rendered_scene.scene_entities),
        render_map=dict(render_map),
        render_spec=dict(render_spec),
        font_family=str(font_family),
    )


__all__ = [
    "circuit_content_bbox",
    "render_defaults",
    "render_equivalent_circuit",
    "resolve_layout_placement",
]
