"""Render and trace-output helpers for the spring physics scene."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from PIL import Image

from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.physics.shared.diagram_style import prepare_physics_diagram_style_and_background
from trace_tasks.tasks.physics.shared.support_sampling import resolve_integer_support
from trace_tasks.tasks.physics.shared.visual_defaults import load_physics_noise_defaults
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.font_assets import font_asset_version, get_font_family_record, sample_font_family
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec
from trace_tasks.tasks.shared.render_variation import resolve_render_int

from .layout import resolve_spring_layout_placement
from .rendering import render_spring_scene
from .state import SCENE_ID, SPRING_SEMANTIC_COLORS, RenderedSpringScene, SpringSceneSpec, SpringTaskDefaults


_RENDER_INT_KEYS = (
    "canvas_width",
    "canvas_height",
    "card_width_px",
    "card_height_px",
    "card_left_px",
    "card_top_px",
    "card_gap_px",
    "stagger_offset_y_px",
    "card_corner_radius_px",
    "card_outline_width_px",
    "support_width_px",
    "support_height_px",
    "support_corner_radius_px",
    "anchor_y_gap_px",
    "hanger_line_width_px",
    "ruler_top_gap_px",
    "ruler_right_gap_px",
    "ruler_value_max",
    "ruler_unit_px",
    "ruler_width_px",
    "ruler_tick_long_px",
    "ruler_tick_short_px",
    "ruler_font_size_px",
    "spring_neutral_units",
    "spring_line_width_px",
    "spring_half_width_px",
    "spring_turn_count",
    "weight_box_width_px",
    "weight_box_height_px",
    "weight_font_size_px",
    "marker_height_px",
    "marker_width_px",
    "missing_tag_width_px",
    "missing_tag_height_px",
    "missing_tag_top_gap_px",
    "label_stroke_width_px",
    "texture_spacing_px",
    "texture_line_width_px",
)


@dataclass(frozen=True)
class SpringPreparedRender:
    """Rendered spring scene plus metadata produced after final placement."""

    image: Image.Image
    rendered_scene: RenderedSpringScene
    layout_placement_meta: Mapping[str, Any]
    image_size: tuple[int, int]
    background_meta: Mapping[str, Any]
    diagram_style_meta: Mapping[str, Any]
    post_noise_meta: Mapping[str, Any]
    font_family: str
    target_support: tuple[int, ...]


def resolve_spring_render_defaults(
    *,
    params: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    defaults: SpringTaskDefaults,
    instance_seed: int,
    namespace: str,
) -> dict[str, Any]:
    """Resolve all render integer knobs with deterministic variation."""

    return {
        key: resolve_render_int(
            params,
            rendering_defaults,
            key,
            int(getattr(defaults, key)),
            instance_seed=int(instance_seed),
            namespace=str(namespace),
        )
        for key in _RENDER_INT_KEYS
    }


def prepare_spring_render(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    fallback_defaults: SpringTaskDefaults,
    namespace: str,
    scene_variant: str,
    accent_color_name: str,
    scene_spec: SpringSceneSpec,
    support_key: str,
    support_fallback: Sequence[int],
) -> SpringPreparedRender:
    """Render one spring scene after objective-specific symbolic construction."""

    render_defaults = resolve_spring_render_defaults(
        params=params,
        rendering_defaults=rendering_defaults,
        defaults=fallback_defaults,
        instance_seed=int(instance_seed),
        namespace=str(namespace),
    )
    render_defaults, layout_placement_meta = resolve_spring_layout_placement(
        render_defaults=render_defaults,
        rendering_defaults=rendering_defaults,
        params=params,
        instance_seed=int(instance_seed),
        scene_variant=str(scene_variant),
        namespace=str(namespace),
    )
    canvas_width = int(render_defaults["canvas_width"])
    canvas_height = int(render_defaults["canvas_height"])
    background, background_meta, diagram_style, diagram_style_meta = prepare_physics_diagram_style_and_background(
        scene_id=SCENE_ID,
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        instance_seed=int(instance_seed),
        params=params,
        protected_colors=SPRING_SEMANTIC_COLORS,
    )
    font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.render.font",
        params=params,
    )
    rendered_scene = render_spring_scene(
        background=background,
        render_defaults=render_defaults,
        accent_color_name=str(accent_color_name),
        scene_spec=scene_spec,
        diagram_style=diagram_style,
        font_family=str(font_family),
    )
    image, post_noise_meta = apply_post_image_noise(
        rendered_scene.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=load_physics_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5),
    )
    target_support = resolve_integer_support(
        params,
        gen_defaults=generation_defaults,
        key=str(support_key),
        fallback=tuple(int(value) for value in support_fallback),
    )
    return SpringPreparedRender(
        image=image,
        rendered_scene=rendered_scene,
        layout_placement_meta=dict(layout_placement_meta),
        image_size=(int(image.size[0]), int(image.size[1])),
        background_meta=dict(background_meta),
        diagram_style_meta=dict(diagram_style_meta),
        post_noise_meta=dict(post_noise_meta),
        font_family=str(font_family),
        target_support=tuple(int(value) for value in target_support),
    )


def spring_font_trace(font_family: str) -> dict[str, Any]:
    """Return font metadata shared by spring render specs."""

    font_record = get_font_family_record(str(font_family))
    return {
        "font_family": str(font_family),
        "font_asset_version": font_asset_version(),
        "font_asset": font_record.to_trace(),
        "scope": "spring_diagram",
        "selection_policy": {
            "pool": "global_approved_font_pool",
            "include_tags": [],
            "exclude_tags": [],
            "exclusion_reason": "",
        },
    }


def build_spring_trace_payload(
    *,
    scene_variant: str,
    public_branch: str,
    public_branch_probabilities: Mapping[str, float],
    internal_branch: str,
    solve_for: str | None,
    accent_color_name: str,
    target_answer: int,
    scale_factor: int,
    scale_factor_support: Sequence[int],
    scene_variant_probabilities: Mapping[str, float],
    accent_color_name_probabilities: Mapping[str, float],
    target_answer_probabilities: Mapping[str, float],
    rendered_scene: RenderedSpringScene,
    prompt_artifacts: Any,
    layout_placement_meta: Mapping[str, Any],
    image_size: tuple[int, int],
    background_meta: Mapping[str, Any],
    diagram_style_meta: Mapping[str, Any],
    post_noise_meta: Mapping[str, Any],
    font_family: str,
    target_support: Sequence[int],
    projected_annotation: Mapping[str, Any],
    prompt_extra_fields: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build spring trace metadata from one sampled symbolic execution."""

    branch_field = "query" + "_id"
    prompt_params = {
        "scene_variant": str(scene_variant),
        "internal_query_id": str(internal_branch),
        "solve_for": None if solve_for is None else str(solve_for),
        "accent_color_name": str(accent_color_name),
        "scene_variant_probabilities": dict(scene_variant_probabilities),
        "query_id_probabilities": dict(public_branch_probabilities),
        "accent_color_name_probabilities": dict(accent_color_name_probabilities),
        "target_answer": int(target_answer),
        "target_answer_probabilities": dict(target_answer_probabilities),
    }
    if prompt_extra_fields:
        prompt_params.update(dict(prompt_extra_fields))
    annotation_ids = [str(item) for item in rendered_scene.annotation_entity_ids]
    execution_trace = {
        "scene_variant": str(scene_variant),
        branch_field: str(public_branch),
        "internal_query_id": str(internal_branch),
        "solve_for": None if solve_for is None else str(solve_for),
        "accent_color_name": str(accent_color_name),
        "target_answer": int(target_answer),
        "target_answer_support": [int(value) for value in target_support],
        "scale_factor_support": [int(value) for value in scale_factor_support],
        "scale_factor": int(scale_factor),
        "left_measurement": {
            "shown_weight_value": None
            if rendered_scene.render_map["columns"]["left"]["shown_weight_value"] is None
            else int(rendered_scene.render_map["columns"]["left"]["shown_weight_value"]),
            "true_weight_value": int(rendered_scene.render_map["columns"]["left"]["true_weight_value"]),
            "shown_extension_value": None
            if rendered_scene.render_map["columns"]["left"]["shown_extension_value"] is None
            else int(rendered_scene.render_map["columns"]["left"]["shown_extension_value"]),
            "true_extension_value": int(rendered_scene.render_map["columns"]["left"]["true_extension_value"]),
        },
        "right_measurement": {
            "shown_weight_value": None
            if rendered_scene.render_map["columns"]["right"]["shown_weight_value"] is None
            else int(rendered_scene.render_map["columns"]["right"]["shown_weight_value"]),
            "true_weight_value": int(rendered_scene.render_map["columns"]["right"]["true_weight_value"]),
            "shown_extension_value": None
            if rendered_scene.render_map["columns"]["right"]["shown_extension_value"] is None
            else int(rendered_scene.render_map["columns"]["right"]["shown_extension_value"]),
            "true_extension_value": int(rendered_scene.render_map["columns"]["right"]["true_extension_value"]),
        },
        "annotation_entity_ids": list(annotation_ids),
        "witness_entity_ids": list(annotation_ids),
    }
    return {
        "scene_ir": {
            "scene_kind": f"physics_spring_{str(scene_variant)}",
            "entities": [dict(entity) for entity in rendered_scene.scene_entities],
            "relations": {
                "scene_variant": str(scene_variant),
                branch_field: str(public_branch),
                "internal_query_id": str(internal_branch),
                "solve_for": None if solve_for is None else str(solve_for),
                "accent_color_name": str(accent_color_name),
                "scale_factor": int(scale_factor),
                "target_answer": int(target_answer),
                "annotation_entity_ids": list(annotation_ids),
                "witness_entity_ids": list(annotation_ids),
            },
        },
        "query_spec": build_prompt_query_spec(
            prompt_artifacts=prompt_artifacts,
            query_id=str(public_branch),
            params=prompt_params,
        ),
        "render_spec": {
            "scene_variant": str(scene_variant),
            "canvas_width": int(image_size[0]),
            "canvas_height": int(image_size[1]),
            "accent_color_name": str(accent_color_name),
            "font": spring_font_trace(str(font_family)),
            "technical_diagram_style": dict(diagram_style_meta),
            "background_style": dict(background_meta),
            "layout_placement": dict(layout_placement_meta),
            "post_image_noise": dict(post_noise_meta),
        },
        "render_map": dict(rendered_scene.render_map),
        "execution_trace": execution_trace,
        "witness_symbolic": {
            "type": "object_set",
            "ids": list(annotation_ids),
        },
        "projected_annotation": dict(projected_annotation),
        "background": dict(background_meta),
        "post_image_noise": dict(post_noise_meta),
    }


__all__ = [
    "SpringPreparedRender",
    "build_spring_trace_payload",
    "prepare_spring_render",
    "resolve_spring_render_defaults",
    "spring_font_trace",
]
