"""Output-support helpers shared by lever objective files."""

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

from .defaults import resolve_render_defaults
from .layout import resolve_lever_layout_placement
from .rendering import render_lever_scene
from .state import LEVER_SEMANTIC_COLORS, SCENE_ID, LeverTaskDefaults, LeverWeightSlot, RenderedLeverScene


@dataclass(frozen=True)
class LeverPreparedRender:
    """Rendered lever scene plus metadata produced after final layout placement."""

    image: Image.Image
    rendered_scene: RenderedLeverScene
    layout_placement_meta: Mapping[str, Any]
    image_size: tuple[int, int]
    background_meta: Mapping[str, Any]
    diagram_style_meta: Mapping[str, Any]
    post_noise_meta: Mapping[str, Any]
    font_family: str
    target_support: tuple[int, ...]


def prepare_lever_render(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    fallback_defaults: LeverTaskDefaults,
    namespace: str,
    scene_variant: str,
    accent_color_name: str,
    placements: Sequence[LeverWeightSlot],
    support_key: str,
    support_fallback: Sequence[int],
) -> LeverPreparedRender:
    """Render one lever layout after objective-specific placement has been sampled."""

    canvas_width = int(
        params.get(
            "canvas_width",
            group_default(rendering_defaults, "canvas_width", fallback_defaults.canvas_width),
        )
    )
    canvas_height = int(
        params.get(
            "canvas_height",
            group_default(rendering_defaults, "canvas_height", fallback_defaults.canvas_height),
        )
    )
    background, background_meta, diagram_style, diagram_style_meta = prepare_physics_diagram_style_and_background(
        scene_id=SCENE_ID,
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        instance_seed=int(instance_seed),
        params=params,
        protected_colors=LEVER_SEMANTIC_COLORS,
    )
    font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.render.font",
        params=params,
    )
    render_defaults = resolve_render_defaults(
        params=params,
        rendering_defaults=rendering_defaults,
        defaults=fallback_defaults,
        instance_seed=int(instance_seed),
        namespace=str(namespace),
    )
    render_defaults, layout_placement_meta = resolve_lever_layout_placement(
        render_defaults=render_defaults,
        params=params,
        instance_seed=int(instance_seed),
        scene_variant=str(scene_variant),
        placements=list(placements),
        namespace=f"{namespace}.lever_layout",
    )
    rendered_scene = render_lever_scene(
        scene_variant=str(scene_variant),
        accent_color_name=str(accent_color_name),
        placements=list(placements),
        render_defaults=render_defaults,
        background=background,
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
    return LeverPreparedRender(
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


def _lever_font_trace(font_family: str) -> dict[str, Any]:
    """Return the font metadata shared by lever render specs."""

    font_record = get_font_family_record(str(font_family))
    return {
        "font_family": str(font_family),
        "font_asset_version": font_asset_version(),
        "font_asset": font_record.to_trace(),
        "scope": "lever_balance_diagram",
        "selection_policy": {
            "pool": "global_approved_font_pool",
            "include_tags": [],
            "exclude_tags": [],
            "exclusion_reason": "",
        },
    }


def _lever_weight_records(rendered_scene: RenderedLeverScene) -> list[dict[str, Any]]:
    """Return compact symbolic records for all rendered lever weights."""

    return [
        {
            "weight_id": str(spec.weight_id),
            "side": str(spec.side),
            "distance_units": int(spec.distance_units),
            "value": None if spec.value is None else int(spec.value),
            "missing": bool(spec.missing),
            "relevant_to_query": bool(spec.relevant),
        }
        for spec in rendered_scene.weight_specs
    ]


def build_lever_trace_payload(
    *,
    scene_variant: str,
    public_branch: str,
    public_branch_probabilities: Mapping[str, float],
    internal_branch: str,
    torque_side: str | None,
    accent_color_name: str,
    target_answer: int,
    scene_variant_probabilities: Mapping[str, float],
    accent_color_name_probabilities: Mapping[str, float],
    target_answer_probabilities: Mapping[str, float],
    rendered_scene: RenderedLeverScene,
    prompt_artifacts: Any,
    layout_metadata: Mapping[str, Any],
    layout_placement_meta: Mapping[str, Any],
    image_size: tuple[int, int],
    background_meta: Mapping[str, Any],
    diagram_style_meta: Mapping[str, Any],
    post_noise_meta: Mapping[str, Any],
    font_family: str,
    target_support: Sequence[int],
    annotation_entity_ids: Sequence[str],
    projected_annotation_type: str,
    projected_bbox_set: Sequence[Sequence[float]],
    projected_bbox_set_map: Mapping[str, Any],
    extra_prompt_fields: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build common lever trace metadata from one sampled symbolic execution."""

    branch_field = "query" + "_id"
    prompt_params = {
        "scene_variant": str(scene_variant),
        "internal_query_id": str(internal_branch),
        "torque_side": None if torque_side is None else str(torque_side),
        "accent_color_name": str(accent_color_name),
        "scene_variant_probabilities": dict(scene_variant_probabilities),
        "query_id_probabilities": dict(public_branch_probabilities),
        "accent_color_name_probabilities": dict(accent_color_name_probabilities),
        "target_answer": int(target_answer),
        "target_answer_probabilities": dict(target_answer_probabilities),
    }
    if extra_prompt_fields:
        prompt_params.update(dict(extra_prompt_fields))
    annotation_ids = [str(item) for item in annotation_entity_ids]
    relevant_weight_ids = [str(item) for item in rendered_scene.relevant_weight_ids]
    execution_trace = {
        "scene_variant": str(scene_variant),
        branch_field: str(public_branch),
        "internal_query_id": str(internal_branch),
        "torque_side": None if torque_side is None else str(torque_side),
        "accent_color_name": str(accent_color_name),
        "target_answer": int(target_answer),
        "target_answer_support": [int(value) for value in target_support],
        "query_side": layout_metadata.get("query_side"),
        "placeholder_side": layout_metadata.get("placeholder_side"),
        "placeholder_distance_units": layout_metadata.get("placeholder_distance_units"),
        "known_torque_left": int(layout_metadata["known_torque_left"]),
        "known_torque_right": int(layout_metadata["known_torque_right"]),
        "weight_specs": _lever_weight_records(rendered_scene),
        "relevant_weight_ids": list(relevant_weight_ids),
        "annotation_entity_ids": list(annotation_ids),
        "witness_entity_ids": list(relevant_weight_ids),
    }
    return {
        "scene_ir": {
            "scene_kind": f"physics_lever_balance_{str(scene_variant)}",
            "entities": [dict(entity) for entity in rendered_scene.scene_entities],
            "relations": {
                "scene_variant": str(scene_variant),
                branch_field: str(public_branch),
                "internal_query_id": str(internal_branch),
                "torque_side": None if torque_side is None else str(torque_side),
                "accent_color_name": str(accent_color_name),
                "target_answer": int(target_answer),
                "relevant_weight_ids": list(relevant_weight_ids),
                "annotation_entity_ids": list(annotation_ids),
                "witness_entity_ids": list(relevant_weight_ids),
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
            "font": _lever_font_trace(str(font_family)),
            "technical_diagram_style": dict(diagram_style_meta),
            "background_style": dict(background_meta),
            "layout_placement": dict(layout_placement_meta),
            "post_image_noise": dict(post_noise_meta),
        },
        "render_map": dict(rendered_scene.render_map),
        "execution_trace": execution_trace,
        "witness_symbolic": {
            "type": "object_set",
            "ids": list(relevant_weight_ids),
        },
        "projected_annotation": {
            "type": str(projected_annotation_type),
            "bbox_set": [list(bbox) for bbox in projected_bbox_set],
            "bbox_set_map": dict(projected_bbox_set_map),
            "pixel_bbox_set_map": dict(projected_bbox_set_map),
        },
        "background": dict(background_meta),
        "post_image_noise": dict(post_noise_meta),
    }


__all__ = ["LeverPreparedRender", "build_lever_trace_payload", "prepare_lever_render"]
