"""Choose the candidate arrow matching the net force direction."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import (
    load_scene_generation_rendering_prompt_defaults,
    required_group_defaults,
)
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec

from .shared.annotations import projected_bbox_map
from .shared.output import build_render_spec, force_payload
from .shared.prompts import PROMPT_BUNDLE_ID as DEFAULT_PROMPT_BUNDLE_ID
from .shared.prompts import build_free_body_prompt_artifacts
from .shared.rendering import render_free_body_forces
from .shared.sampling import make_force_scenario, resolve_sampling_axes
from .shared.state import (
    DIRECTION_NAMES,
    OPTION_LETTERS,
    SCENE_ID,
    SCENE_NAMESPACE,
)


TASK_ID = "task_physics__free_body_forces__net_force_direction_choice"
TASK_NAMESPACE = "physics_free_body_forces_net_force_direction_choice"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (SINGLE_QUERY_ID,)
TASK_PROMPT_KEY = "net_force_direction_choice_query"

_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    "physics",
    SCENE_ID,
    task_id=TASK_ID,
)


@register_task
class PhysicsFreeBodyForcesNetForceDirectionChoiceTask:
    """Choose the candidate arrow showing the resultant of visible applied forces."""

    task_id = TASK_ID
    reasoning_operations = ('aggregation', 'formula_evaluation')
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
        """Generate one force diagram and bind answer/annotation."""

        _ = int(max_attempts)
        query_id, query_probabilities, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=dict(params or {}),
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=SINGLE_QUERY_ID,
            task_id=TASK_ID,
            namespace=f"{TASK_NAMESPACE}.query",
        )
        axes = resolve_sampling_axes(
            instance_seed=int(instance_seed),
            params=task_params,
            generation_defaults=_GEN_DEFAULTS,
            namespace=SCENE_NAMESPACE,
        )
        scenario = make_force_scenario(
            instance_seed=int(instance_seed),
            axes=axes,
            params=task_params,
            generation_defaults=_GEN_DEFAULTS,
            namespace=SCENE_NAMESPACE,
        )
        rendered = render_free_body_forces(
            instance_seed=int(instance_seed),
            params=task_params,
            scenario=scenario,
            axes=axes,
            rendering_defaults=_RENDER_DEFAULTS,
            namespace=SCENE_NAMESPACE,
        )
        prompt_defaults = required_group_defaults(
            _PROMPT_DEFAULTS,
            ("bundle_id", "task_key"),
            context=f"prompt defaults for {TASK_ID}",
        )
        prompt_artifacts = build_free_body_prompt_artifacts(
            domain=self.domain,
            bundle_id=str(prompt_defaults.get("bundle_id", DEFAULT_PROMPT_BUNDLE_ID)),
            task_key=str(prompt_defaults.get("task_key", TASK_PROMPT_KEY)),
            prompt_query_key=str(query_id),
            dynamic_slots={},
            instance_seed=int(instance_seed),
        )
        answer_gt = TypedValue(type="option_letter", value=str(scenario.correct_option_letter))
        annotation_value = {
            "force_diagram": list(rendered.annotation_bbox_map["force_diagram"]),
            "selected_candidate": list(rendered.annotation_bbox_map["selected_candidate"]),
        }
        annotation_gt = TypedValue(
            type="bbox_map",
            value=annotation_value,
        )
        forces = force_payload(scenario)
        prompt_query_spec = build_prompt_query_spec(
            prompt_artifacts=prompt_artifacts,
            query_id=str(query_id),
            params={
                "query_id": str(query_id),
                "scene_variant": str(scenario.scene_variant),
                "net_force_direction": str(scenario.net_force_direction),
                "correct_option_letter": str(scenario.correct_option_letter),
                "accent_color_name": str(axes.accent_color_name),
                "target_answer": str(scenario.correct_option_letter),
                "query_id_probabilities": dict(query_probabilities),
                "scene_variant_probabilities": dict(axes.scene_variant_probabilities),
                "net_force_direction_probabilities": dict(axes.net_force_direction_probabilities),
                "correct_option_letter_probabilities": dict(axes.correct_option_letter_probabilities),
                "accent_color_name_probabilities": dict(axes.accent_color_name_probabilities),
            },
        )
        render_map = dict(rendered.render_map)
        render_map["query_id"] = str(query_id)
        annotation_entity_ids = ["force_diagram", "selected_candidate"]
        trace_payload = {
            "scene_ir": {
                "scene_kind": f"physics_free_body_forces_{scenario.scene_variant}",
                "entities": [dict(entity) for entity in rendered.scene_entities],
                "relations": {
                    "scene_variant": str(scenario.scene_variant),
                    "query_id": str(query_id),
                    "net_force_direction": str(scenario.net_force_direction),
                    "resultant_vector": [
                        int(scenario.resultant_vector[0]),
                        int(scenario.resultant_vector[1]),
                    ],
                    "correct_option_letter": str(scenario.correct_option_letter),
                    "option_directions": dict(scenario.option_directions),
                    "annotation_entity_ids": list(annotation_entity_ids),
                },
            },
            "query_spec": prompt_query_spec,
            "render_spec": build_render_spec(rendered=rendered, axes=axes),
            "render_map": dict(render_map),
            "execution_trace": {
                "scene_variant": str(scenario.scene_variant),
                "query_id": str(query_id),
                "net_force_direction": str(scenario.net_force_direction),
                "resultant_vector": [
                    int(scenario.resultant_vector[0]),
                    int(scenario.resultant_vector[1]),
                ],
                "correct_option_letter": str(scenario.correct_option_letter),
                "option_letters": list(OPTION_LETTERS),
                "option_directions": dict(scenario.option_directions),
                "force_specs": list(forces),
                "annotation_entity_ids": list(annotation_entity_ids),
                "sampling_probabilities": {
                    "query_id": dict(query_probabilities),
                    "scene_variant": dict(axes.scene_variant_probabilities),
                    "net_force_direction": dict(axes.net_force_direction_probabilities),
                    "correct_option_letter": dict(axes.correct_option_letter_probabilities),
                    "accent_color_name": dict(axes.accent_color_name_probabilities),
                },
            },
            "witness_symbolic": {
                "type": "object_map",
                "ids": list(annotation_entity_ids),
                "key_to_entity_id": {
                    "force_diagram": "force_diagram",
                    "selected_candidate": "selected_candidate",
                },
            },
            "projected_annotation": projected_bbox_map(annotation_gt.value),
            "background": dict(rendered.background_meta),
            "post_image_noise": dict(rendered.post_noise_meta),
        }
        return TaskOutput(
            prompt=str(prompt_artifacts.prompt),
            prompt_variants=dict(prompt_artifacts.prompt_variants),
            answer_gt=answer_gt,
            annotation_gt=annotation_gt,
            image=rendered.image,
            image_id="img0",
            trace_payload=trace_payload,
            task_versions=default_task_versions(),
            query_id=str(query_id),
            scene_id=SCENE_ID,
        )


__all__ = [
    "DIRECTION_NAMES",
    "OPTION_LETTERS",
    "PhysicsFreeBodyForcesNetForceDirectionChoiceTask",
]
