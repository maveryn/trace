"""Scene-private lifecycle helpers for ray-optics tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from PIL import Image

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.physics.shared.diagram_style import (
    prepare_physics_diagram_style_and_background,
)
from trace_tasks.tasks.shared.config_defaults import required_group_defaults
from trace_tasks.tasks.shared.font_assets import font_asset_version, get_font_family_record
from trace_tasks.tasks.shared.font_assets import sample_font_family
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec

from .shared.annotations import (
    build_prompt_examples,
    pixel_point_set_annotation_artifacts,
)
from .shared.layout import (
    resolve_board_render_defaults,
    resolve_optics_layout_placement,
)
from .shared.mechanics import sample_scene_layout
from .shared.prompts import (
    PROMPT_BUNDLE_ID as DEFAULT_PROMPT_BUNDLE_ID,
    build_ray_optics_prompt_artifacts,
)
from .shared.rendering import render_optics_ray_scene
from .shared.sampling import answer_support, resolve_ray_axes
from .shared.state import (
    RAY_EVENT_BOUNCE,
    RAY_EVENT_TARGET_HIT,
    RayOpticsTaskDefaults,
    RaySceneLayout,
    RenderedOpticsScene,
    SCENE_ID,
)


@dataclass(frozen=True)
class RayOpticsLifecyclePlan:
    """Public objective bindings for the ray-optics lifecycle."""

    task_identifier: str
    namespace: str
    public_branch_id: str
    public_branch_probabilities: Mapping[str, float]
    ray_event_kind: str
    prompt_branch_key: str
    fallback_defaults: RayOpticsTaskDefaults
    generation_defaults: Mapping[str, Any]
    rendering_defaults: Mapping[str, Any]
    prompt_defaults_group: Mapping[str, Any]
    post_noise_defaults: Mapping[str, Any]


@dataclass(frozen=True)
class RayOpticsRenderedAssets:
    """Rendered image and metadata for one ray-optics scene."""

    rendered_scene: RenderedOpticsScene
    image: Image.Image
    background_meta: dict[str, Any]
    diagram_style_meta: dict[str, Any]
    layout_placement_meta: dict[str, Any]
    post_noise_meta: dict[str, Any]
    font_family: str


def _render_scene_assets(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    plan: RayOpticsLifecyclePlan,
    scene_layout: RaySceneLayout,
    accent_color_name: str,
    render_defaults: Mapping[str, Any],
    layout_placement_meta: Mapping[str, Any],
) -> RayOpticsRenderedAssets:
    """Render one optics board and project final-image metadata."""

    background, background_meta, diagram_style, diagram_style_meta = (
        prepare_physics_diagram_style_and_background(
            scene_id=SCENE_ID,
            canvas_width=int(render_defaults["canvas_width"]),
            canvas_height=int(render_defaults["canvas_height"]),
            instance_seed=int(instance_seed),
            params=params,
        )
    )
    font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace=f"{plan.namespace}.render.font",
        params=params,
    )
    rendered_scene = render_optics_ray_scene(
        background=background,
        render_defaults=render_defaults,
        accent_color_name=str(accent_color_name),
        scene_variant=str(scene_layout.scene_variant),
        ray_event_kind=str(plan.ray_event_kind),
        source_row=int(scene_layout.source_row),
        mirrors=[
            {
                "col": int(mirror.col),
                "row": int(mirror.row),
                "orientation": str(mirror.orientation),
                "hit": bool(mirror.hit),
            }
            for mirror in scene_layout.mirrors
        ],
        targets=[
            {
                "target_id": str(target.target_id),
                "col": int(target.col),
                "row": int(target.row),
                "label": int(target.label),
                "hit": bool(target.hit),
            }
            for target in scene_layout.targets
        ],
        bounce_cells=list(scene_layout.bounce_cells),
        ray_polyline_cells=list(scene_layout.path_cells),
        source_point_px=tuple(scene_layout.source_point_px),
        exit_point_px=tuple(scene_layout.exit_point_px),
        annotation_entity_ids=list(scene_layout.annotation_entity_ids),
        diagram_style=diagram_style,
        font_family=str(font_family),
    )
    image, post_noise_meta = apply_post_image_noise(
        rendered_scene.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=plan.post_noise_defaults,
    )
    return RayOpticsRenderedAssets(
        rendered_scene=rendered_scene,
        image=image,
        background_meta=dict(background_meta),
        diagram_style_meta=dict(diagram_style_meta),
        layout_placement_meta=dict(layout_placement_meta),
        post_noise_meta=dict(post_noise_meta),
        font_family=str(font_family),
    )


def _annotation_artifacts(
    *,
    plan: RayOpticsLifecyclePlan,
    rendered_scene: RenderedOpticsScene,
) -> dict[str, Any]:
    """Bind the objective-specific point witnesses from rendered scene specs."""

    if str(plan.ray_event_kind) == RAY_EVENT_BOUNCE:
        points_by_label = {
            str(spec.bounce_id): list(spec.point_px)
            for spec in rendered_scene.bounce_specs
            if str(spec.bounce_id) in set(rendered_scene.annotation_entity_ids)
        }
        witness_type = "physics_optics_bounce_points"
    elif str(plan.ray_event_kind) == RAY_EVENT_TARGET_HIT:
        points_by_label = {
            str(spec.target_id): list(spec.point_px)
            for spec in rendered_scene.target_specs
            if str(spec.target_id) in set(rendered_scene.annotation_entity_ids)
        }
        witness_type = "physics_optics_hit_target_points"
    else:
        raise ValueError(f"unsupported ray event kind: {plan.ray_event_kind}")
    return pixel_point_set_annotation_artifacts(
        points_by_label=points_by_label,
        graph_origin=rendered_scene.graph_origin_px,
        graph_spacing=int(rendered_scene.graph_spacing_px),
        witness_type=str(witness_type),
        ordered_labels=tuple(str(item) for item in rendered_scene.annotation_entity_ids),
    )


def run_ray_optics_lifecycle(
    *,
    domain: str,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    plan: RayOpticsLifecyclePlan,
) -> TaskOutput:
    """Sample, render, bind answer/annotation, and return one task output."""

    axes = resolve_ray_axes(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=plan.generation_defaults,
        fallback_defaults=plan.fallback_defaults,
        ray_event_kind=str(plan.ray_event_kind),
        namespace=str(plan.namespace),
    )
    render_defaults, layout_placement_meta = resolve_optics_layout_placement(
        render_defaults=resolve_board_render_defaults(
            params=params,
            rendering_defaults=plan.rendering_defaults,
            fallback_defaults=plan.fallback_defaults,
            instance_seed=int(instance_seed),
            namespace=str(plan.namespace),
        ),
        rendering_defaults=plan.rendering_defaults,
        params=params,
        instance_seed=int(instance_seed),
        namespace=str(plan.namespace),
    )

    for attempt_index in range(max(1, int(max_attempts))):
        attempt_rng = spawn_rng(
            int(instance_seed),
            f"{plan.namespace}.attempt.{int(attempt_index)}",
        )
        try:
            scene_layout = sample_scene_layout(
                attempt_rng,
                scene_variant=str(axes.scene_variant),
                ray_event_kind=str(axes.ray_event_kind),
                target_answer=int(axes.target_answer),
                params=params,
                gen_defaults=plan.generation_defaults,
                fallback_defaults=plan.fallback_defaults,
                render_defaults=render_defaults,
            )
        except ValueError:
            continue

        rendered_assets = _render_scene_assets(
            instance_seed=int(instance_seed),
            params=params,
            plan=plan,
            scene_layout=scene_layout,
            accent_color_name=str(axes.accent_color_name),
            render_defaults=render_defaults,
            layout_placement_meta=layout_placement_meta,
        )
        prompt_defaults = required_group_defaults(
            plan.prompt_defaults_group,
            (
                "bundle_id",
                "task_key",
            ),
            context=f"prompt defaults for {plan.task_identifier}",
        )
        json_example, json_example_answer_only = build_prompt_examples(
            str(plan.ray_event_kind)
        )
        prompt_artifacts = build_ray_optics_prompt_artifacts(
            domain=str(domain),
            bundle_id=str(prompt_defaults.get("bundle_id", DEFAULT_PROMPT_BUNDLE_ID)),
            task_key=str(prompt_defaults["task_key"]),
            prompt_branch_key=str(plan.prompt_branch_key),
            dynamic_slots={
                "json_example": str(json_example),
                "json_example_answer_only": str(json_example_answer_only),
            },
            instance_seed=int(instance_seed),
        )

        answer_gt = TypedValue(type="integer", value=int(axes.target_answer))
        annotation_artifacts = _annotation_artifacts(
            plan=plan,
            rendered_scene=rendered_assets.rendered_scene,
        )
        annotation_gt = TypedValue(
            type=str(annotation_artifacts["annotation_type"]),
            value=list(annotation_artifacts["annotation_value"]),
        )
        render_map = dict(rendered_assets.rendered_scene.render_map)
        render_map["annotation_point_map_px"] = dict(
            annotation_artifacts["projected_annotation"].get("pixel_point_map", {})
        )
        render_map["annotation_point_set_px"] = list(
            annotation_artifacts["annotation_value"]
        )
        font_record = get_font_family_record(str(rendered_assets.font_family))
        internal_branch_id = str(plan.prompt_branch_key)
        trace_payload = {
            "scene_ir": {
                "scene_kind": f"physics_ray_optics_{str(axes.scene_variant)}",
                "entities": [
                    dict(entity)
                    for entity in rendered_assets.rendered_scene.scene_entities
                ],
                "relations": {
                    "scene_variant": str(axes.scene_variant),
                    "ray_event_kind": str(axes.ray_event_kind),
                    "target_answer": int(axes.target_answer),
                    "accent_color_name": str(axes.accent_color_name),
                    "annotation_entity_ids": list(
                        rendered_assets.rendered_scene.annotation_entity_ids
                    ),
                },
            },
            "query_spec": build_prompt_query_spec(
                prompt_artifacts=prompt_artifacts,
                query_id=str(internal_branch_id),
                params={
                    "internal_query_id": str(internal_branch_id),
                    "public_query_id": str(plan.public_branch_id),
                    "ray_event_kind": str(axes.ray_event_kind),
                    "scene_variant": str(axes.scene_variant),
                    "accent_color_name": str(axes.accent_color_name),
                    "scene_variant_probabilities": dict(
                        axes.scene_variant_probabilities
                    ),
                    "accent_color_name_probabilities": dict(
                        axes.accent_color_name_probabilities
                    ),
                    "target_answer": int(axes.target_answer),
                    "target_answer_probabilities": dict(
                        axes.target_answer_probabilities
                    ),
                    "target_answer_support": answer_support(
                        params=params,
                        gen_defaults=plan.generation_defaults,
                        fallback_defaults=plan.fallback_defaults,
                        scene_variant=str(axes.scene_variant),
                        ray_event_kind=str(axes.ray_event_kind),
                    ),
                    "public_query_id_probabilities": dict(
                        plan.public_branch_probabilities
                    ),
                },
            ),
            "render_spec": {
                "scene_variant": str(axes.scene_variant),
                "ray_event_kind": str(axes.ray_event_kind),
                "canvas_width": int(rendered_assets.image.size[0]),
                "canvas_height": int(rendered_assets.image.size[1]),
                "accent_color_name": str(axes.accent_color_name),
                "font": {
                    "font_family": str(rendered_assets.font_family),
                    "font_asset_version": font_asset_version(),
                    "font_asset": font_record.to_trace(),
                    "scope": "ray_optics_board",
                    "selection_policy": {
                        "pool": "global_approved_font_pool",
                        "include_tags": [],
                        "exclude_tags": [],
                        "exclusion_reason": "",
                    },
                },
                "technical_diagram_style": dict(rendered_assets.diagram_style_meta),
                "background_style": dict(rendered_assets.background_meta),
                "layout_placement": dict(rendered_assets.layout_placement_meta),
                "post_image_noise": dict(rendered_assets.post_noise_meta),
            },
            "render_map": dict(render_map),
            "execution_trace": {
                "query_id": str(internal_branch_id),
                "internal_query_id": str(internal_branch_id),
                "public_query_id": str(plan.public_branch_id),
                "ray_event_kind": str(axes.ray_event_kind),
                "scene_variant": str(axes.scene_variant),
                "accent_color_name": str(axes.accent_color_name),
                "target_answer": int(axes.target_answer),
                "target_answer_support": answer_support(
                    params=params,
                    gen_defaults=plan.generation_defaults,
                    fallback_defaults=plan.fallback_defaults,
                    scene_variant=str(axes.scene_variant),
                    ray_event_kind=str(axes.ray_event_kind),
                ),
                "source_row": int(scene_layout.source_row),
                "mirror_specs": [
                    {
                        "mirror_id": str(mirror.mirror_id),
                        "col": int(mirror.col),
                        "row": int(mirror.row),
                        "orientation": str(mirror.orientation),
                        "hit": bool(mirror.hit),
                    }
                    for mirror in scene_layout.mirrors
                ],
                "target_specs": [
                    {
                        "target_id": str(target.target_id),
                        "col": int(target.col),
                        "row": int(target.row),
                        "hit": bool(target.hit),
                    }
                    for target in scene_layout.targets
                ],
                "path_cells": [
                    [int(col), int(row)] for col, row in scene_layout.path_cells
                ],
                "bounce_cells": [
                    [int(col), int(row)] for col, row in scene_layout.bounce_cells
                ],
                "annotation_pixel_points": [list(point) for point in annotation_gt.value],
                "annotation_entity_ids": list(
                    rendered_assets.rendered_scene.annotation_entity_ids
                ),
            },
            "witness_symbolic": dict(annotation_artifacts["witness_symbolic"]),
            "projected_annotation": dict(annotation_artifacts["projected_annotation"]),
            "background": dict(rendered_assets.background_meta),
            "post_image_noise": dict(rendered_assets.post_noise_meta),
        }
        return TaskOutput(
            prompt=str(prompt_artifacts.prompt),
            prompt_variants=dict(prompt_artifacts.prompt_variants),
            answer_gt=answer_gt,
            annotation_gt=annotation_gt,
            image=rendered_assets.image,
            image_id="img0",
            trace_payload=trace_payload,
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=str(internal_branch_id),
        )

    raise RuntimeError(
        f"{plan.task_identifier} failed to generate a valid scene after "
        f"{max_attempts} attempts"
    )


__all__ = ["RayOpticsLifecyclePlan", "run_ray_optics_lifecycle"]
