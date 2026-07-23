"""Mechanical-advantage task for ideal pulley diagrams."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.physics.shared.diagram_style import prepare_physics_diagram_style_and_background
from trace_tasks.tasks.physics.shared.support_sampling import resolve_integer_support
from trace_tasks.tasks.physics.shared.visual_defaults import load_physics_noise_defaults
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import (
    load_scene_generation_rendering_prompt_defaults,
    required_group_defaults,
)
from trace_tasks.tasks.shared.fixed_query import explicit_query_id_param, select_task_query_id
from trace_tasks.tasks.shared.font_assets import sample_font_family
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec
from trace_tasks.tasks.shared.render_variation import resolve_render_int

from .shared.annotations import build_prompt_examples, projected_bbox_map
from .shared.output import build_font_trace, build_object_witness, cut_segment_records
from .shared.prompts import build_pulley_prompt_artifacts
from .shared.rendering import (
    RENDER_DEFAULT_KEYS,
    render_pulley_scene,
    resolve_pulley_layout_placement,
)
from .shared.sampling import (
    answer_support_key,
    connected_support_count_support,
    disconnected_segment_count_support,
    effort_force_bounds,
    fallback_support,
    normalize_solve_for,
    resolve_axes,
    sample_scene_spec,
)
from .shared.state import PULLEY_SEMANTIC_COLORS, PulleyTaskDefaults, RenderedPulleyScene, SCENE_ID


TASK_ID = "task_physics__pulley__pulley_mechanical_advantage"
TASK_NAMESPACE = "physics_pulley_pulley_mechanical_advantage"
MISSING_EFFORT_FORCE_QUERY_ID = "missing_effort_force_value"
MISSING_LOAD_FORCE_QUERY_ID = "missing_load_force_value"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (
    MISSING_EFFORT_FORCE_QUERY_ID,
    MISSING_LOAD_FORCE_QUERY_ID,
)
QUERY_ID_TO_SOLVE_FOR = {
    MISSING_EFFORT_FORCE_QUERY_ID: "effort_force",
    MISSING_LOAD_FORCE_QUERY_ID: "load_force",
}
SOLVE_FOR_TO_QUERY_ID = {value: key for key, value in QUERY_ID_TO_SOLVE_FOR.items()}

_DEFAULTS = PulleyTaskDefaults()
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    "physics",
    SCENE_ID,
    task_id=TASK_ID,
)
POST_IMAGE_NOISE_DEFAULTS = load_physics_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)


def _resolve_render_defaults(
    *,
    instance_seed: int,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Resolve integer rendering defaults for the pulley scene."""

    return {
        key: resolve_render_int(
            params,
            _RENDER_DEFAULTS,
            key,
            int(getattr(_DEFAULTS, key)),
            instance_seed=int(instance_seed),
            namespace=TASK_NAMESPACE,
        )
        for key in RENDER_DEFAULT_KEYS
    }


@register_task
class PhysicsPulleyMechanicalAdvantageTask:
    """Return one ideal pulley mechanical-advantage question."""

    task_id = TASK_ID
    reasoning_operations = ('counting', 'formula_evaluation')
    domain = "physics"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(
        self,
        instance_seed: int,
        *,
        params: Dict[str, Any],
        max_attempts: int,
    ) -> TaskOutput:
        """Own the pulley objective: sample operands, bind answer/annotation, and build output.

        The invariant is that the rendered full supporting strands, visible
        known force label, unknown force label, and integer answer all come
        from the same sampled ideal-pulley execution trace.
        """

        input_params = dict(params or {})
        explicit_solve_for = input_params.get("solve_for")
        explicit_query_id = explicit_query_id_param(input_params, allow_default=False)
        if explicit_solve_for is not None:
            solve_for_query_id = SOLVE_FOR_TO_QUERY_ID[normalize_solve_for(explicit_solve_for)]
            if explicit_query_id is not None and str(explicit_query_id) != str(solve_for_query_id):
                raise ValueError(
                    "solve_for conflicts with query_id: "
                    f"{explicit_solve_for} implies {solve_for_query_id}, got {explicit_query_id}"
                )
            input_params["query_id"] = str(solve_for_query_id)

        selected_branch, branch_probabilities, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=input_params,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=MISSING_EFFORT_FORCE_QUERY_ID,
            task_id=TASK_ID,
            namespace=f"{TASK_NAMESPACE}.query",
        )
        task_params["solve_for"] = str(QUERY_ID_TO_SOLVE_FOR[str(selected_branch)])
        axes = resolve_axes(
            int(instance_seed),
            params=task_params,
            generation_defaults=_GEN_DEFAULTS,
            namespace=TASK_NAMESPACE,
        )
        rendered_scene: RenderedPulleyScene | None = None
        scene_spec = None

        for attempt_index in range(max(1, int(max_attempts))):
            attempt_rng = spawn_rng(int(instance_seed), f"{TASK_NAMESPACE}.attempt.{int(attempt_index)}")
            try:
                scene_spec = sample_scene_spec(
                    attempt_rng,
                    instance_seed=int(instance_seed),
                    scene_variant=str(axes.scene_variant),
                    solve_for=str(axes.solve_for),
                    target_answer=int(axes.target_answer),
                    params=task_params,
                    generation_defaults=_GEN_DEFAULTS,
                    namespace=TASK_NAMESPACE,
                )
            except ValueError:
                continue

            render_defaults = _resolve_render_defaults(
                instance_seed=int(instance_seed),
                params=task_params,
            )
            render_defaults, layout_placement_meta = resolve_pulley_layout_placement(
                render_defaults=render_defaults,
                rendering_defaults=_RENDER_DEFAULTS,
                params=task_params,
                instance_seed=int(instance_seed),
                scene_spec=scene_spec,
                namespace=TASK_NAMESPACE,
            )
            background, background_meta, diagram_style, diagram_style_meta = prepare_physics_diagram_style_and_background(
                scene_id=SCENE_ID,
                canvas_width=int(render_defaults["canvas_width"]),
                canvas_height=int(render_defaults["canvas_height"]),
                instance_seed=int(instance_seed),
                params=task_params,
                protected_colors=PULLEY_SEMANTIC_COLORS,
            )
            font_family = sample_font_family(
                role="readout",
                instance_seed=int(instance_seed),
                namespace=f"{TASK_NAMESPACE}.render.font",
                params=task_params,
            )
            rendered_scene = render_pulley_scene(
                background=background,
                render_defaults=render_defaults,
                accent_color_name=str(axes.accent_color_name),
                scene_spec=scene_spec,
                diagram_style=diagram_style,
                font_family=str(font_family),
            )
            image, post_noise_meta = apply_post_image_noise(
                rendered_scene.image,
                instance_seed=int(instance_seed),
                params=task_params,
                default_config=POST_IMAGE_NOISE_DEFAULTS,
            )

            prompt_defaults = required_group_defaults(
                _PROMPT_DEFAULTS,
                (
                    "bundle_id",
                    "task_key",
                ),
                context=f"prompt defaults for {self.task_id}",
            )
            json_example, json_example_answer_only = build_prompt_examples()
            prompt_artifacts = build_pulley_prompt_artifacts(
                domain=self.domain,
                bundle_id=str(prompt_defaults["bundle_id"]),
                task_key=str(prompt_defaults["task_key"]),
                prompt_query_key=str(selected_branch),
                dynamic_slots={
                    "json_example": str(json_example),
                    "json_example_answer_only": str(json_example_answer_only),
                },
                instance_seed=int(instance_seed),
            )

            answer_gt = TypedValue(type="integer", value=int(axes.target_answer))
            annotation_value = {
                str(key): [float(value) for value in bbox]
                for key, bbox in rendered_scene.annotation_bbox_map.items()
            }
            annotation_gt = TypedValue(type="bbox_map", value=dict(annotation_value))
            target_support_key = answer_support_key(str(axes.solve_for))
            cut_segment_payload = cut_segment_records(scene_spec.cut_segments)
            query_params = {
                "query_id": str(selected_branch),
                "solve_for": str(axes.solve_for),
                "accent_color_name": str(axes.accent_color_name),
                "scene_variant": str(axes.scene_variant),
                "scene_variant_probabilities": dict(axes.scene_variant_probabilities),
                "query_id_probabilities": dict(branch_probabilities),
                "solve_for_probabilities": dict(axes.solve_for_probabilities),
                "accent_color_name_probabilities": dict(axes.accent_color_name_probabilities),
                "target_answer": int(axes.target_answer),
                "target_answer_probabilities": dict(axes.target_answer_probabilities),
            }
            trace_payload = {
                "scene_ir": {
                    "scene_kind": f"physics_pulley_mechanical_advantage_{str(axes.scene_variant)}",
                    "entities": [dict(entity) for entity in rendered_scene.scene_entities],
                    "relations": {
                        "scene_variant": str(axes.scene_variant),
                        "query_id": str(selected_branch),
                        "solve_for": str(axes.solve_for),
                        "accent_color_name": str(axes.accent_color_name),
                        "support_segment_count": int(scene_spec.support_segment_count),
                        "connected_support_count": int(scene_spec.support_segment_count),
                        "disconnected_segment_count": int(scene_spec.disconnected_segment_count),
                        "effort_force_value": int(scene_spec.effort_force_value),
                        "load_force_value": int(scene_spec.load_force_value),
                        "shown_effort_force_value": None
                        if scene_spec.shown_effort_force_value is None
                        else int(scene_spec.shown_effort_force_value),
                        "shown_load_force_value": None
                        if scene_spec.shown_load_force_value is None
                        else int(scene_spec.shown_load_force_value),
                        "connected_slot_indices": [int(value) for value in scene_spec.connected_slot_indices],
                        "cut_segments": [dict(item) for item in cut_segment_payload],
                        "target_answer": int(axes.target_answer),
                        "annotation_entity_ids": list(rendered_scene.annotation_entity_ids),
                    },
                },
                "query_spec": build_prompt_query_spec(
                    prompt_artifacts=prompt_artifacts,
                    query_id=str(selected_branch),
                    params=query_params,
                ),
                "render_spec": {
                    "scene_variant": str(axes.scene_variant),
                    "canvas_width": int(image.size[0]),
                    "canvas_height": int(image.size[1]),
                    "accent_color_name": str(axes.accent_color_name),
                    "font": build_font_trace(font_family=str(font_family)),
                    "technical_diagram_style": dict(diagram_style_meta),
                    "background_style": dict(background_meta),
                    "layout_placement": dict(layout_placement_meta),
                    "post_image_noise": dict(post_noise_meta),
                },
                "render_map": dict(rendered_scene.render_map),
                "execution_trace": {
                    "query_id": str(selected_branch),
                    "solve_for": str(axes.solve_for),
                    "accent_color_name": str(axes.accent_color_name),
                    "scene_variant": str(axes.scene_variant),
                    "target_answer": int(axes.target_answer),
                    "target_answer_support": list(
                        resolve_integer_support(
                            task_params,
                            gen_defaults=_GEN_DEFAULTS,
                            key=str(target_support_key),
                            fallback=fallback_support(str(axes.solve_for)),
                        )
                    ),
                    "connected_support_count_support": list(
                        connected_support_count_support(task_params, _GEN_DEFAULTS)
                    ),
                    "disconnected_segment_count_support": list(
                        disconnected_segment_count_support(task_params, _GEN_DEFAULTS)
                    ),
                    "effort_force_min": int(effort_force_bounds(task_params, _GEN_DEFAULTS)[0]),
                    "effort_force_max": int(effort_force_bounds(task_params, _GEN_DEFAULTS)[1]),
                    "support_segment_count": int(scene_spec.support_segment_count),
                    "connected_support_count": int(scene_spec.support_segment_count),
                    "disconnected_segment_count": int(scene_spec.disconnected_segment_count),
                    "effort_force_value": int(scene_spec.effort_force_value),
                    "load_force_value": int(scene_spec.load_force_value),
                    "shown_effort_force_value": None
                    if scene_spec.shown_effort_force_value is None
                    else int(scene_spec.shown_effort_force_value),
                    "shown_load_force_value": None
                    if scene_spec.shown_load_force_value is None
                    else int(scene_spec.shown_load_force_value),
                    "connected_slot_indices": [int(value) for value in scene_spec.connected_slot_indices],
                    "cut_segments": [dict(item) for item in cut_segment_payload],
                    "annotation_entity_ids": list(rendered_scene.annotation_entity_ids),
                },
                "witness_symbolic": build_object_witness(rendered_scene.annotation_entity_id_map),
                "projected_annotation": projected_bbox_map(annotation_value),
                "background": dict(background_meta),
                "post_image_noise": dict(post_noise_meta),
            }
            return TaskOutput(
                prompt=str(prompt_artifacts.prompt),
                prompt_variants=dict(prompt_artifacts.prompt_variants),
                answer_gt=answer_gt,
                annotation_gt=annotation_gt,
                image=image,
                image_id="img0",
                trace_payload=trace_payload,
                task_versions=default_task_versions(),
                scene_id=SCENE_ID,
                query_id=str(selected_branch),
            )

        raise RuntimeError(f"{self.task_id} failed to generate a valid scene after {max_attempts} attempts")


__all__ = ["PhysicsPulleyMechanicalAdvantageTask"]
