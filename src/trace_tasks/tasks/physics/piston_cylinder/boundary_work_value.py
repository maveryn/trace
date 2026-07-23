"""Compute signed boundary work for a constant-pressure piston-cylinder process."""

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
from trace_tasks.tasks.shared.prompt_json_example import build_prompt_json_examples
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec

from .shared.annotations import ANNOTATION_KEYS, projected_bbox_map
from .shared.output import build_object_witness, build_render_spec
from .shared.prompts import build_piston_cylinder_prompt_artifacts
from .shared.rendering import render_piston_cylinder_scene
from .shared.sampling import resolve_scenario
from .shared.state import SCENE_ID, SCENE_NAMESPACE


TASK_ID = "task_physics__piston_cylinder__boundary_work_value"
TASK_NAMESPACE = "physics_piston_cylinder_boundary_work_value"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (SINGLE_QUERY_ID,)
TASK_PROMPT_KEY = "boundary_work_value_query"

_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    "physics",
    SCENE_ID,
    task_id=TASK_ID,
)


@register_task
class PhysicsPistonCylinderBoundaryWorkValueTask:
    """Compute signed boundary work from pressure and visible volume change."""

    domain = "physics"
    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(
        self,
        instance_seed: int,
        *,
        params: Dict[str, Any],
        max_attempts: int,
    ) -> TaskOutput:
        """Select the single query branch and bind answer plus annotation."""

        _ = int(max_attempts)
        selected_branch, branch_probabilities, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=dict(params or {}),
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=SINGLE_QUERY_ID,
            task_id=TASK_ID,
            namespace=f"{TASK_NAMESPACE}.query",
        )
        scenario = resolve_scenario(
            int(instance_seed),
            task_params,
            _GEN_DEFAULTS,
            namespace=SCENE_NAMESPACE,
        )
        render_params = dict(task_params)
        render_params.setdefault(
            "volume_l_support",
            [int(value) for value in scenario.final_volume_probabilities.keys()],
        )
        rendered = render_piston_cylinder_scene(
            instance_seed=int(instance_seed),
            params=render_params,
            scenario=scenario,
            render_defaults=_RENDER_DEFAULTS,
            namespace=SCENE_NAMESPACE,
        )
        prompt_defaults = required_group_defaults(
            _PROMPT_DEFAULTS,
            (
                "bundle_id",
                "task_key",
            ),
            context=f"prompt defaults for {TASK_ID}",
        )
        answer_gt = TypedValue(type="integer", value=int(scenario.boundary_work_kj))
        annotation_value = {
            str(key): list(rendered.annotation_bbox_map[str(key)])
            for key in ANNOTATION_KEYS
        }
        annotation_gt = TypedValue(type="bbox_map", value=annotation_value)
        json_example, json_example_answer_only = build_prompt_json_examples(
            annotation_value=annotation_gt.value,
            answer_type=str(answer_gt.type),
        )
        prompt_artifacts = build_piston_cylinder_prompt_artifacts(
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
        annotation_ids = list(annotation_gt.value.keys())
        query_params = {
            "query_id": str(selected_branch),
            "pressure_mpa": int(scenario.pressure_mpa),
            "initial_volume_l": int(scenario.initial_volume_l),
            "final_volume_l": int(scenario.final_volume_l),
            "target_answer": int(scenario.boundary_work_kj),
            "orientation": str(scenario.orientation),
            "query_id_probabilities": dict(branch_probabilities),
            "orientation_probabilities": dict(scenario.orientation_probabilities),
            "pressure_mpa_probabilities": dict(scenario.pressure_probabilities),
            "initial_volume_l_probabilities": dict(scenario.initial_volume_probabilities),
            "final_volume_l_probabilities": dict(scenario.final_volume_probabilities),
            "target_answer_probabilities": dict(scenario.target_answer_probabilities),
        }
        render_map = dict(rendered.render_map)
        render_map["query_id"] = str(selected_branch)
        trace_payload = {
            "scene_ir": {
                "scene_kind": "physics_piston_cylinder_constant_pressure",
                "entities": list(rendered.scene_entities),
                "relations": {
                    "query_id": str(selected_branch),
                    "pressure_mpa": int(scenario.pressure_mpa),
                    "initial_volume_l": int(scenario.initial_volume_l),
                    "final_volume_l": int(scenario.final_volume_l),
                    "boundary_work_kj": int(scenario.boundary_work_kj),
                    "orientation": str(scenario.orientation),
                },
            },
            "query_spec": build_prompt_query_spec(
                prompt_artifacts=prompt_artifacts,
                query_id=str(selected_branch),
                params=query_params,
            ),
            "render_spec": build_render_spec(rendered),
            "render_map": render_map,
            "execution_trace": {
                "query_id": str(selected_branch),
                "pressure_mpa": int(scenario.pressure_mpa),
                "initial_volume_l": int(scenario.initial_volume_l),
                "final_volume_l": int(scenario.final_volume_l),
                "delta_volume_l": int(scenario.final_volume_l - scenario.initial_volume_l),
                "boundary_work_kj": int(scenario.boundary_work_kj),
                "target_answer": int(scenario.boundary_work_kj),
                "annotation_entity_ids": sorted(annotation_ids),
            },
            "witness_symbolic": build_object_witness(sorted(annotation_ids)),
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
            scene_id=SCENE_ID,
            query_id=str(selected_branch),
        )


__all__ = ["PhysicsPistonCylinderBoundaryWorkValueTask"]
