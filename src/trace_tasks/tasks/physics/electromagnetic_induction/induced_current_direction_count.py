"""Count induction panels by current direction."""

from __future__ import annotations

from typing import Any

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

from .shared.output import (
    build_render_spec,
    panel_current_classes,
    panel_field_orientations,
    panel_flux_changes,
    panel_mechanisms,
    projected_bbox_set,
)
from .shared.prompts import PROMPT_BUNDLE_ID, build_induction_prompt_artifacts
from .shared.rendering import render_induction_scene
from .shared.sampling import make_induction_scenario
from .shared.state import ANSWER_SUPPORT, PANEL_COUNT, SCENE_ID, SCENE_NAMESPACE


TASK_ID = "task_physics__electromagnetic_induction__induced_current_direction_count"
TASK_NAMESPACE = "physics_electromagnetic_induction_direction_count"
SUPPORTED_QUERY_IDS = (
    "clockwise_induced_current_count",
    "counterclockwise_induced_current_count",
    "no_induced_current_count",
)
TASK_PROMPT_KEY = "induced_current_direction_count_query"
QUERY_TO_CURRENT_CLASS = {
    "clockwise_induced_current_count": "clockwise",
    "counterclockwise_induced_current_count": "counterclockwise",
    "no_induced_current_count": "no_current",
}
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    "physics",
    SCENE_ID,
    task_id=TASK_ID,
)


@register_task
class PhysicsElectromagneticInductionDirectionCountTask:
    """Count mini-panels by induced-current direction from visible flux-change cues."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'formula_evaluation')
    domain = "physics"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate the six-panel induction grid and bind matching full-panel boxes."""

        _ = int(max_attempts)
        selected_query, branch_probabilities, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=dict(params or {}),
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=SUPPORTED_QUERY_IDS[0],
            task_id=TASK_ID,
            namespace=f"{TASK_NAMESPACE}.branch",
        )
        target_current_class = str(QUERY_TO_CURRENT_CLASS[str(selected_query)])
        scenario = make_induction_scenario(
            instance_seed=int(instance_seed),
            params=task_params,
            generation_defaults=_GEN_DEFAULTS,
            rendering_defaults=_RENDER_DEFAULTS,
            target_current_class=target_current_class,
            namespace=SCENE_NAMESPACE,
        )
        rendered = render_induction_scene(
            instance_seed=int(instance_seed),
            params=task_params,
            scenario=scenario,
            rendering_defaults=_RENDER_DEFAULTS,
            namespace=SCENE_NAMESPACE,
        )
        prompt_defaults = required_group_defaults(
            _PROMPT_DEFAULTS,
            ("bundle_id", "task_key"),
            context=f"prompt defaults for {TASK_ID}",
        )
        prompt_artifacts = build_induction_prompt_artifacts(
            domain=self.domain,
            bundle_id=str(prompt_defaults.get("bundle_id", PROMPT_BUNDLE_ID)),
            task_key=str(prompt_defaults.get("task_key", TASK_PROMPT_KEY)),
            prompt_key=str(selected_query),
            dynamic_slots={},
            instance_seed=int(instance_seed),
        )
        answer_gt = TypedValue(type="integer", value=int(scenario.target_answer))
        annotation_gt = TypedValue(
            type="bbox_set",
            value=[list(bbox) for bbox in rendered.annotation_bboxes],
        )
        matching_panel_ids = list(rendered.render_map["matching_panel_ids"])
        prompt_query_spec = build_prompt_query_spec(
            prompt_artifacts=prompt_artifacts,
            query_id=str(selected_query),
            params={
                "query_id": str(selected_query),
                "target_answer": int(scenario.target_answer),
                "target_current_class": str(scenario.target_current_class),
                "answer_support": list(ANSWER_SUPPORT),
                "panel_count": PANEL_COUNT,
                "query_id_probabilities": dict(branch_probabilities),
                "target_answer_probabilities": dict(scenario.target_answer_probabilities),
            },
        )
        trace_payload = {
            "scene_ir": {
                "scene_kind": "physics_electromagnetic_induction_six_panel_count",
                "entities": [dict(entity) for entity in rendered.scene_entities],
                "relations": {
                    "query_id": str(selected_query),
                    "target_current_class": str(scenario.target_current_class),
                    "target_answer": int(scenario.target_answer),
                    "matching_panel_ids": list(matching_panel_ids),
                },
            },
            "query_spec": dict(prompt_query_spec),
            "render_spec": build_render_spec(rendered),
            "render_map": dict(rendered.render_map),
            "execution_trace": {
                "query_id": str(selected_query),
                "target_current_class": str(scenario.target_current_class),
                "target_answer": int(scenario.target_answer),
                "matching_panel_ids": list(matching_panel_ids),
                "panel_current_classes": panel_current_classes(scenario),
                "panel_flux_changes": panel_flux_changes(scenario),
                "panel_field_orientations": panel_field_orientations(scenario),
                "panel_mechanisms": panel_mechanisms(scenario),
                "annotation_entity_ids": list(matching_panel_ids),
            },
            "sampling": {
                "query_id_probabilities": dict(branch_probabilities),
                "target_answer_probabilities": dict(scenario.target_answer_probabilities),
            },
            "witness_symbolic": {
                "type": "bbox_set",
                "entity_ids": list(matching_panel_ids),
            },
            "projected_annotation": projected_bbox_set(annotation_gt.value),
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
            query_id=str(selected_query),
        )


__all__ = [
    "ANSWER_SUPPORT",
    "PhysicsElectromagneticInductionDirectionCountTask",
    "SUPPORTED_QUERY_IDS",
]
