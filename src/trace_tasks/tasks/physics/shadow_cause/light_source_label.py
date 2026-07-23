"""Light-source label task for the shadow-cause scene."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.types import TypedValue
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.physics.shared.diagram_style import prepare_physics_diagram_style_and_background
from trace_tasks.tasks.physics.shared.visual_defaults import load_physics_noise_defaults
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import (
    load_scene_generation_rendering_prompt_defaults,
    required_group_defaults,
)
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.font_assets import font_asset_version, get_font_family_record, sample_font_family
from trace_tasks.tasks.shared.output_metadata import default_task_versions

from .shared.annotations import normalize_shadow_annotation_bbox
from .shared.prompts import build_shadow_cause_prompt_artifacts
from .shared.rendering import render_shadow_cause_scene
from .shared.sampling import make_shadow_scene_spec, resolve_canvas_size, resolve_shadow_cause_axes, resolve_shadow_render_defaults
from .shared.state import OBJECT_SHAPES, OPPOSITE_DIRECTION, OPTION_LETTERS, SCENE_ID, SHADOW_DIRECTIONS, ShadowCauseTaskDefaults


TASK_ID = "task_physics__shadow_cause__light_source_label"
TASK_NAMESPACE = "physics_shadow_cause_light_source_label"
INTERNAL_QUERY_ID = "source_from_shadow_label"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (SINGLE_QUERY_ID,)

_DEFAULTS = ShadowCauseTaskDefaults()
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    "physics",
    SCENE_ID,
    task_id=TASK_ID,
)
POST_IMAGE_NOISE_DEFAULTS = load_physics_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)


@register_task
class PhysicsShadowCauseLightSourceLabelTask:
    """Choose which labeled light source caused the cast shadow."""

    task_id = TASK_ID
    reasoning_operations = ('spatial_relations',)
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
        """Own objective sampling, prompt binding, and output assembly."""

        params = dict(params or {})
        selected_query, query_probs, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=params,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=SINGLE_QUERY_ID,
            task_id=TASK_ID,
            namespace=f"{TASK_NAMESPACE}.query",
        )

        last_error: Exception | None = None
        for attempt_index in range(max(1, int(max_attempts))):
            attempt_seed = int(instance_seed) + (attempt_index * 7919)
            try:
                axes = resolve_shadow_cause_axes(
                    instance_seed=int(attempt_seed),
                    params=task_params,
                    generation_defaults=_GEN_DEFAULTS,
                    namespace=TASK_NAMESPACE,
                )
                render_defaults = resolve_shadow_render_defaults(
                    task_params,
                    _RENDER_DEFAULTS,
                    fallback_defaults=_DEFAULTS,
                    instance_seed=int(attempt_seed),
                    namespace=TASK_NAMESPACE,
                )
                spec = make_shadow_scene_spec(
                    instance_seed=int(attempt_seed),
                    axes=axes,
                    render_defaults=render_defaults,
                )
            except Exception as exc:  # pragma: no cover - surfaced if all attempts fail.
                last_error = exc
                continue

            canvas_width, canvas_height = resolve_canvas_size(
                task_params,
                _RENDER_DEFAULTS,
                fallback_defaults=_DEFAULTS,
            )
            background, background_meta, diagram_style, diagram_style_meta = prepare_physics_diagram_style_and_background(
                instance_seed=int(attempt_seed),
                params=task_params,
                scene_id=SCENE_ID,
                canvas_width=int(canvas_width),
                canvas_height=int(canvas_height),
                require_grid=True,
            )
            font_family = sample_font_family(
                role="readout",
                instance_seed=int(attempt_seed),
                namespace=f"{TASK_NAMESPACE}.font",
                params=task_params,
            )
            font_record = get_font_family_record(str(font_family))
            rendered = render_shadow_cause_scene(
                image=background,
                spec=spec,
                render_defaults=render_defaults,
                font_family=str(font_family),
                style=diagram_style,
            )
            image, post_noise_meta = apply_post_image_noise(
                rendered.image,
                instance_seed=int(attempt_seed),
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
            prompt_artifacts = build_shadow_cause_prompt_artifacts(
                domain=self.domain,
                bundle_id=str(prompt_defaults["bundle_id"]),
                task_key=str(prompt_defaults["task_key"]),
                query_key=str(selected_query),
                dynamic_slots={},
                instance_seed=int(attempt_seed),
            )
            answer_gt = TypedValue(type="option_letter", value=str(spec.correct_option_letter))
            annotation_bbox = normalize_shadow_annotation_bbox(
                rendered.render_map["candidate_light_sources"],
                selected_label=str(spec.correct_option_letter),
            )
            annotation_gt = TypedValue(type="bbox", value=list(annotation_bbox))

            trace_payload = {
                "scene_ir": {
                    "scene_kind": "physics_shadow_cause_light_source_candidates",
                    "entities": [dict(entity) for entity in rendered.scene_entities],
                    "relations": {
                        "query_id": str(selected_query),
                        "internal_query_id": INTERNAL_QUERY_ID,
                        "shadow_direction": str(spec.shadow_direction),
                        "source_direction": str(spec.source_direction),
                        "correct_option_letter": str(spec.correct_option_letter),
                    },
                },
                "query_spec": {
                    "query_id": str(selected_query),
                    "template_id": str(prompt_defaults["bundle_id"]),
                    "prompt_variant": dict(prompt_artifacts.prompt_variant),
                    "prompt_variant_active_key": str(prompt_artifacts.prompt_variant_active_key),
                    "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
                    "params": {
                        "query_id": str(selected_query),
                        "internal_query_id": INTERNAL_QUERY_ID,
                        "query_id_probabilities": dict(query_probs),
                        "target_answer": str(spec.correct_option_letter),
                        "answer_support": list(OPTION_LETTERS),
                        "shadow_direction": str(spec.shadow_direction),
                        "source_direction": str(spec.source_direction),
                        "shadow_direction_support": list(SHADOW_DIRECTIONS),
                        "object_shape": str(spec.object_shape),
                        "object_shape_support": list(OBJECT_SHAPES),
                    },
                },
                "render_spec": {
                    "canvas_width": int(image.size[0]),
                    "canvas_height": int(image.size[1]),
                    "font": {
                        "font_family": str(font_family),
                        "font_asset_version": font_asset_version(),
                        "font_asset": font_record.to_trace(),
                        "scope": "shadow_cause_diagram",
                        "selection_policy": {
                            "pool": "global_approved_font_pool",
                            "include_tags": [],
                            "exclude_tags": [],
                            "exclusion_reason": "",
                        },
                    },
                    "technical_diagram_style": dict(diagram_style_meta),
                    "background_style": background_meta,
                    "render_defaults": dict(render_defaults),
                    "post_image_noise": post_noise_meta,
                },
                "render_map": dict(rendered.render_map),
                "execution_trace": {
                    "query_id": str(selected_query),
                    "internal_query_id": INTERNAL_QUERY_ID,
                    "correct_option_letter": str(spec.correct_option_letter),
                    "shadow_direction": str(spec.shadow_direction),
                    "source_direction": str(spec.source_direction),
                    "candidate_directions": dict(rendered.render_map["candidate_directions"]),
                    "object_shape": str(spec.object_shape),
                    "annotation_entity_ids": [f"lamp_{spec.correct_option_letter}"],
                },
                "sampling": {
                    "query_id_probabilities": dict(query_probs),
                    "correct_option_letter_probabilities": dict(axes.correct_option_letter_probabilities),
                    "shadow_direction_probabilities": dict(axes.shadow_direction_probabilities),
                    "object_shape_probabilities": dict(axes.object_shape_probabilities),
                },
                "witness_symbolic": {
                    "type": "bbox",
                    "entity_id": f"lamp_{spec.correct_option_letter}",
                },
                "projected_annotation": {
                    "type": "bbox",
                    "bbox": list(annotation_gt.value),
                    "pixel_bbox": list(annotation_gt.value),
                },
                "background": background_meta,
                "post_image_noise": post_noise_meta,
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
                query_id=str(selected_query),
            )
        raise RuntimeError(f"failed to generate shadow-cause instance after {max_attempts} attempts: {last_error}")


__all__ = ["OPPOSITE_DIRECTION", "PhysicsShadowCauseLightSourceLabelTask"]
