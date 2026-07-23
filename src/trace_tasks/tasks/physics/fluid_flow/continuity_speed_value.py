"""Compute a missing speed from a steady-flow continuity diagram."""

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

from .shared.annotations import projected_bbox
from .shared.formulas import continuity_product
from .shared.output import build_render_spec, flow_scenario_params
from .shared.prompts import PROMPT_BUNDLE_ID as DEFAULT_PROMPT_BUNDLE_ID
from .shared.prompts import build_fluid_flow_prompt_artifacts
from .shared.rendering import render_fluid_flow
from .shared.sampling import resolve_flow_scenario
from .shared.state import SCENE_ID, SCENE_NAMESPACE


TASK_ID = "task_physics__fluid_flow__continuity_speed_value"
TASK_NAMESPACE = "physics_fluid_flow_continuity_speed_value"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (SINGLE_QUERY_ID,)
TASK_PROMPT_KEY = "continuity_speed_value_query"

_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    "physics",
    SCENE_ID,
    task_id=TASK_ID,
)


@register_task
class PhysicsFluidFlowContinuitySpeedValueTask:
    """Compute a missing steady-flow speed from visible station labels."""

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
        """Generate one fluid-flow diagram and bind answer/annotation."""

        _ = int(max_attempts)
        query_id, query_probabilities, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=dict(params or {}),
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=SINGLE_QUERY_ID,
            task_id=TASK_ID,
            namespace=f"{TASK_NAMESPACE}.query",
        )
        scenario = resolve_flow_scenario(
            instance_seed=int(instance_seed),
            params=task_params,
            generation_defaults=_GEN_DEFAULTS,
            namespace=SCENE_NAMESPACE,
        )
        rendered = render_fluid_flow(
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
        prompt_artifacts = build_fluid_flow_prompt_artifacts(
            domain=self.domain,
            bundle_id=str(prompt_defaults.get("bundle_id", DEFAULT_PROMPT_BUNDLE_ID)),
            task_key=str(prompt_defaults.get("task_key", TASK_PROMPT_KEY)),
            prompt_query_key=str(query_id),
            dynamic_slots={},
            instance_seed=int(instance_seed),
        )
        answer_gt = TypedValue(type="integer", value=int(scenario.target_answer))
        annotation_gt = TypedValue(
            type="bbox",
            value=list(rendered.annotation_bbox_map["missing_speed_label"]),
        )
        sampled_params = flow_scenario_params(scenario)
        prompt_query_spec = build_prompt_query_spec(
            prompt_artifacts=prompt_artifacts,
            query_id=str(query_id),
            params={
                "query_id": str(query_id),
                "query_id_probabilities": dict(query_probabilities),
                **sampled_params,
            },
        )
        render_map = dict(rendered.render_map)
        render_map["query_id"] = str(query_id)
        continuity_lhs = continuity_product(
            area_cm2=int(scenario.area_1_cm2),
            speed_m_s=int(scenario.speed_1_m_s),
        )
        continuity_rhs = continuity_product(
            area_cm2=int(scenario.area_2_cm2),
            speed_m_s=int(scenario.speed_2_m_s),
        )
        trace_payload = {
            "scene_ir": {
                "scene_kind": "physics_fluid_flow_continuity",
                "entities": list(rendered.scene_entities),
                "relations": {
                    "query_id": str(query_id),
                    "missing_station": str(scenario.missing_station),
                    "area_1_cm2": int(scenario.area_1_cm2),
                    "area_2_cm2": int(scenario.area_2_cm2),
                    "speed_1_m_s": int(scenario.speed_1_m_s),
                    "speed_2_m_s": int(scenario.speed_2_m_s),
                    "target_answer": int(scenario.target_answer),
                },
            },
            "query_spec": prompt_query_spec,
            "render_spec": build_render_spec(rendered),
            "render_map": render_map,
            "execution_trace": {
                "query_id": str(query_id),
                "missing_station": str(scenario.missing_station),
                "area_1_cm2": int(scenario.area_1_cm2),
                "area_2_cm2": int(scenario.area_2_cm2),
                "speed_1_m_s": int(scenario.speed_1_m_s),
                "speed_2_m_s": int(scenario.speed_2_m_s),
                "continuity_lhs": int(continuity_lhs),
                "continuity_rhs": int(continuity_rhs),
                "target_answer": int(scenario.target_answer),
                "annotation_entity_ids": ["missing_speed_label"],
                "sampling_probabilities": {
                    "query_id": dict(query_probabilities),
                    "orientation": dict(scenario.orientation_probabilities),
                    "missing_station": dict(scenario.missing_station_probabilities),
                    "target_answer": dict(scenario.target_answer_probabilities),
                },
            },
            "witness_symbolic": {
                "type": "object",
                "id": "missing_speed_label",
            },
            "projected_annotation": projected_bbox(annotation_gt.value),
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
            query_id=str(query_id),
        )


__all__ = ["PhysicsFluidFlowContinuitySpeedValueTask"]
