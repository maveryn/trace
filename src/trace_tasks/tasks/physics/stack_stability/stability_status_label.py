"""Public task for center-of-mass stack-stability option selection."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.physics.shared.diagram_style import prepare_physics_diagram_style_and_background
from trace_tasks.tasks.physics.shared.visual_defaults import load_physics_noise_defaults
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import (
    group_default,
    required_group_defaults,
    split_generation_rendering_prompt_defaults,
)
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.font_assets import (
    font_asset_version,
    get_font_family_record,
    sample_font_family,
)
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.variant_sampling import (
    apply_balanced_variant_sampling,
    resolve_variant,
)

from .shared.annotations import normalize_stack_annotation_bbox
from .shared.prompts import build_stack_prompt_artifacts
from .shared.rendering import render_stack_scene, resolve_stack_render_defaults
from .shared.sampling import make_stack_scene_spec
from .shared.state import (
    OPTION_LETTERS,
    SCENE_ID,
    STATUS_STABLE,
    STATUS_TIPPING,
    StackTaskDefaults,
)


TASK_ID = "task_physics__stack_stability__stability_status_label"
TASK_NAMESPACE = "physics_stack_stability_stability_status_label"
TASK_PROMPT_KEY = "stack_stability_status_label_query"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = ("stable_stack_label", "tipping_stack_label")
_BRANCH_TO_STATUS = {
    "stable_stack_label": STATUS_STABLE,
    "tipping_stack_label": STATUS_TIPPING,
}

_DEFAULTS = StackTaskDefaults()
_TASK_GROUP_DEFAULTS = get_scene_defaults("physics", SCENE_ID)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = (
    split_generation_rendering_prompt_defaults(
        _TASK_GROUP_DEFAULTS if isinstance(_TASK_GROUP_DEFAULTS, Mapping) else {},
        task_id=TASK_ID,
    )
)
POST_IMAGE_NOISE_DEFAULTS = load_physics_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)


def _resolve_correct_option_letter(
    instance_seed: int,
    *,
    params: Mapping[str, Any],
) -> Tuple[str, Dict[str, float]]:
    selected, probabilities = resolve_variant(
        spawn_rng(int(instance_seed), f"{TASK_NAMESPACE}.correct_option_letter"),
        params=params,
        gen_defaults=_GEN_DEFAULTS,
        supported_variants=OPTION_LETTERS,
        explicit_key="correct_option_letter",
        weights_key="correct_option_letter_weights",
    )
    selected = apply_balanced_variant_sampling(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=_GEN_DEFAULTS,
        selected_variant=str(selected),
        variant_probabilities=probabilities,
        supported_variants=OPTION_LETTERS,
        balance_flag_key="balanced_correct_option_letter_sampling",
        explicit_key="correct_option_letter",
        weights_key="correct_option_letter_weights",
        sampling_namespace=f"{TASK_NAMESPACE}.correct_option_letter",
    )
    return str(selected), {str(key): float(value) for key, value in probabilities.items()}


@register_task
class PhysicsStackStabilityStatusLabelTask:
    """Choose the stack whose COM projection has the queried stability status."""

    task_id = TASK_ID
    reasoning_operations = ('aggregation', 'spatial_relations', 'formula_evaluation')
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
        """Generate one option-selection instance with public answer binding."""

        params = dict(params or {})
        last_error: Exception | None = None
        for attempt_index in range(max(1, int(max_attempts))):
            attempt_seed = int(instance_seed) + (attempt_index * 7919)
            try:
                selected_query, query_probs, task_params = select_task_query_id(
                    instance_seed=int(attempt_seed),
                    params=params,
                    supported_query_ids=SUPPORTED_QUERY_IDS,
                    default_query_id=SUPPORTED_QUERY_IDS[0],
                    task_id=TASK_ID,
                    namespace=f"{TASK_NAMESPACE}.query",
                )
                target_status = _BRANCH_TO_STATUS[str(selected_query)]
                correct_letter, letter_probs = _resolve_correct_option_letter(
                    attempt_seed,
                    params=task_params,
                )
                spec = make_stack_scene_spec(
                    instance_seed=attempt_seed,
                    target_status=str(target_status),
                    correct_option_letter=str(correct_letter),
                )
            except Exception as exc:  # pragma: no cover - surfaced below if all attempts fail.
                last_error = exc
                continue

            canvas_width = int(
                task_params.get(
                    "canvas_width",
                    group_default(_RENDER_DEFAULTS, "canvas_width", _DEFAULTS.canvas_width),
                )
            )
            canvas_height = int(
                task_params.get(
                    "canvas_height",
                    group_default(_RENDER_DEFAULTS, "canvas_height", _DEFAULTS.canvas_height),
                )
            )
            background, background_meta, diagram_style, diagram_style_meta = (
                prepare_physics_diagram_style_and_background(
                    instance_seed=attempt_seed,
                    params=task_params,
                    scene_id=SCENE_ID,
                    canvas_width=int(canvas_width),
                    canvas_height=int(canvas_height),
                    require_grid=True,
                )
            )
            font_family = sample_font_family(
                role="readout",
                instance_seed=attempt_seed,
                namespace=f"{TASK_NAMESPACE}.font",
                params=task_params,
            )
            font_record = get_font_family_record(str(font_family))
            render_defaults = resolve_stack_render_defaults(
                task_params,
                _RENDER_DEFAULTS,
                instance_seed=attempt_seed,
                defaults=_DEFAULTS,
            )
            rendered = render_stack_scene(
                image=background,
                spec=spec,
                render_defaults=render_defaults,
                font_family=str(font_family),
                style=diagram_style,
                instance_seed=attempt_seed,
            )
            image, post_noise_meta = apply_post_image_noise(
                rendered.image,
                instance_seed=attempt_seed,
                params=task_params,
                default_config=POST_IMAGE_NOISE_DEFAULTS,
            )

            prompt_defaults = required_group_defaults(
                _PROMPT_DEFAULTS,
                ("bundle_id", "task_key"),
                context=f"prompt defaults for {self.task_id}",
            )
            prompt_artifacts = build_stack_prompt_artifacts(
                domain=self.domain,
                bundle_id=str(prompt_defaults["bundle_id"]),
                task_key=str(prompt_defaults["task_key"]),
                query_key=str(selected_query),
                dynamic_slots={},
                instance_seed=attempt_seed,
            )

            answer_gt = TypedValue(type="option_letter", value=str(spec.correct_option_letter))
            annotation_value = normalize_stack_annotation_bbox(rendered.annotation_bbox_px)
            annotation_gt = TypedValue(type="bbox", value=annotation_value)
            trace_payload = {
                "scene_ir": {
                    "scene_kind": "physics_stack_stability_brick_stacks",
                    "entities": [dict(entity) for entity in rendered.scene_entities],
                    "relations": {
                        "query_id": str(selected_query),
                        "correct_option_letter": str(spec.correct_option_letter),
                        "target_status": str(target_status),
                    },
                },
                "query_spec": {
                    "query_id": str(selected_query),
                    "template_id": str(prompt_defaults["bundle_id"]),
                    "prompt_variant": dict(prompt_artifacts.prompt_variant),
                    "prompt_variant_active_key": str(
                        prompt_artifacts.prompt_variant_active_key
                    ),
                    "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
                    "params": {
                        "query_id": str(selected_query),
                        "query_id_probabilities": dict(query_probs),
                        "target_answer": str(spec.correct_option_letter),
                        "answer_support": list(OPTION_LETTERS),
                        "target_status": str(target_status),
                    },
                },
                "render_spec": {
                    "canvas_width": int(image.size[0]),
                    "canvas_height": int(image.size[1]),
                    "font": {
                        "font_family": str(font_family),
                        "font_asset_version": font_asset_version(),
                        "font_asset": font_record.to_trace(),
                        "scope": "stack_stability_diagram",
                    },
                    "technical_diagram_style": dict(diagram_style_meta),
                    "background_style": background_meta,
                    "render_defaults": dict(render_defaults),
                    "post_image_noise": post_noise_meta,
                },
                "render_map": dict(rendered.render_map),
                "execution_trace": {
                    "query_id": str(selected_query),
                    "correct_option_letter": str(spec.correct_option_letter),
                    "target_status": str(target_status),
                    "candidate_statuses": dict(rendered.render_map["candidate_statuses"]),
                    "candidate_tip_directions": dict(
                        rendered.render_map["candidate_tip_directions"]
                    ),
                    "annotation_entity_id": f"stack_{spec.correct_option_letter}",
                    "annotation_role": "selected_stack_stability_witness",
                },
                "sampling": {
                    "query_id_probabilities": dict(query_probs),
                    "correct_option_letter_probabilities": dict(letter_probs),
                },
                "witness_symbolic": {
                    "type": "bbox",
                    "role": "selected_stack_stability_witness",
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
        raise RuntimeError(
            "failed to generate stack-stability instance "
            f"after {max_attempts} attempts: {last_error}"
        )


__all__ = ["PhysicsStackStabilityStatusLabelTask"]
