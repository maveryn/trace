"""Rendering helpers for the geometry angle-relations scene."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

from PIL import Image, ImageDraw

from trace_tasks.core.sampling import uniform_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.font_assets import font_asset_version, get_font_family_record, sample_font_family
from trace_tasks.tasks.shared.text_rendering import load_font

from ...shared.diagram_style import (
    GEOMETRY_STYLE_PROFILE_ANALYTICAL_DIAGRAM,
    geometry_diagram_style_metadata,
    geometry_shape_style_from_diagram_style,
    prepare_geometry_diagram_style_and_background,
)
from ...shared.scene_transform import LazySceneTransform
from .spatial_primitives import ANGLE_RELATIONS_NOISE_DEFAULTS, RenderContext
from .state import SCENE_ID, AngleRelationCase, RenderedAngleRelationScene


@dataclass(frozen=True)
class RenderedAngleRelationContext:
    """Rendered image plus trace metadata for one angle-relations sample."""

    rendered_scene: RenderedAngleRelationScene
    image: Image.Image
    render_meta: Dict[str, Any]
    post_noise_meta: Dict[str, Any]


def make_angle_relation_render_context(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
) -> tuple[RenderContext, Dict[str, Any]]:
    """Resolve style and canvas inputs for one angle-relations rendering attempt."""

    rng = spawn_rng(int(instance_seed), "geometry.angle_relations.render")
    width = int(params.get("canvas_width", group_default(render_defaults, "canvas_width", 720)))
    height = int(params.get("canvas_height", group_default(render_defaults, "canvas_height", 560)))
    image, background_meta, diagram_style, diagram_style_resolution = prepare_geometry_diagram_style_and_background(
        instance_seed=int(instance_seed),
        params=params,
        scene_id=SCENE_ID,
        canvas_width=int(width),
        canvas_height=int(height),
        allow_dark=True,
        style_profile=GEOMETRY_STYLE_PROFILE_ANALYTICAL_DIAGRAM,
    )
    shape_style = geometry_shape_style_from_diagram_style(diagram_style)
    line_width = int(params.get("line_width", group_default(render_defaults, "line_width", 4)))
    font_size = int(params.get("label_font_size", group_default(render_defaults, "label_font_size", 22)))
    small_font_size = int(params.get("point_label_font_size", group_default(render_defaults, "point_label_font_size", 18)))
    label_stroke_width = int(params.get("label_stroke_width", group_default(render_defaults, "label_stroke_width", 1)))
    fill_choices = (
        tuple(int(value) for value in diagram_style.fill_rgb),
        tuple(int(value) for value in diagram_style.muted_fill_rgb),
        tuple(int(value) for value in diagram_style.option_fill_rgb),
        tuple(int(value) for value in diagram_style.panel_alt_fill_rgb),
    )
    accent_choices = (
        tuple(int(value) for value in diagram_style.accent_rgb),
        tuple(int(value) for value in diagram_style.secondary_accent_rgb),
        tuple(int(value) for value in diagram_style.highlight_rgb),
        tuple(int(value) for value in diagram_style.guide_rgb),
    )
    font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace=f"geometry.{SCENE_ID}.font_family",
        params=params,
    )
    font_record = get_font_family_record(str(font_family))
    color_rng = spawn_rng(int(instance_seed), "geometry.angle_relations.accent")
    color_idx = int(uniform_choice(color_rng, tuple(range(len(fill_choices)))))
    layout_offset = (
        float(rng.randint(-32, 32)),
        float(rng.randint(-20, 22)),
    )
    ctx = RenderContext(
        rng=rng,
        image=image,
        draw=ImageDraw.Draw(image),
        width=int(width),
        height=int(height),
        line_color=shape_style.line_color,
        label_color=shape_style.label_color,
        label_stroke_color=shape_style.label_stroke_color,
        accent_color=accent_choices[color_idx],
        fill_color=fill_choices[color_idx],
        line_width=max(2, int(line_width)),
        label_stroke_width=max(0, int(label_stroke_width)),
        font=load_font(max(12, int(font_size)), bold=True, font_family=str(font_family)),
        small_font=load_font(max(10, int(small_font_size)), bold=True, font_family=str(font_family)),
        layout_offset=(float(layout_offset[0]), float(layout_offset[1])),
        font_family=str(font_family),
        scene_transform=LazySceneTransform(
            rng,
            params=params,
            render_defaults=render_defaults,
            canvas_width=int(width),
            canvas_height=int(height),
        ),
    )
    render_meta = {
        "background_style": dict(background_meta),
        "shape_style": shape_style.to_trace_dict(),
        "line_width": int(ctx.line_width),
        "label_font_size": int(font_size),
        "point_label_font_size": int(small_font_size),
        "label_stroke_width": int(ctx.label_stroke_width),
        "accent_color": list(ctx.accent_color),
        "fill_color": list(ctx.fill_color),
        "layout_jitter": {
            "offset_px": [round(float(ctx.layout_offset[0]), 3), round(float(ctx.layout_offset[1]), 3)],
            "offset_range_px": [-32, 32, -20, 22],
            "applied_before_annotation_projection": True,
        },
        "technical_diagram_style": geometry_diagram_style_metadata(diagram_style),
        "technical_diagram_style_resolution": dict(diagram_style_resolution),
        "font_family": font_record.to_trace(),
        "font_asset_version": font_asset_version(),
    }
    return ctx, render_meta


def render_angle_relation_case(
    *,
    case: AngleRelationCase,
    instance_seed: int,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    max_attempts: int,
) -> RenderedAngleRelationContext:
    """Render one already-selected angle-relations case."""

    last_error: Exception | None = None
    for attempt_index in range(max(1, int(max_attempts))):
        try:
            attempt_params = dict(params)
            attempt_params["_render_attempt"] = int(attempt_index)
            ctx, render_meta = make_angle_relation_render_context(
                instance_seed=int(instance_seed) + int(attempt_index),
                params=attempt_params,
                render_defaults=render_defaults,
            )
            rendered_scene = case.build(ctx)
            if ctx.scene_transform is not None:
                render_meta["single_object_scene_rotation"] = ctx.scene_transform.metadata()
            image, post_noise_meta = apply_post_image_noise(
                rendered_scene.image,
                instance_seed=int(instance_seed),
                params=params,
                default_config=ANGLE_RELATIONS_NOISE_DEFAULTS,
            )
            return RenderedAngleRelationContext(
                rendered_scene=rendered_scene,
                image=image,
                render_meta=dict(render_meta),
                post_noise_meta=dict(post_noise_meta),
            )
        except Exception as exc:
            last_error = exc
            continue
    raise RuntimeError("failed to render angle-relations scene") from last_error


__all__ = [
    "RenderedAngleRelationContext",
    "make_angle_relation_render_context",
    "render_angle_relation_case",
]
