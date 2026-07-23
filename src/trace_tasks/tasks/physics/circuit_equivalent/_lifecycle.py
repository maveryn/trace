"""Private lifecycle for circuit-equivalent public objectives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.shared.config_defaults import required_group_defaults
from trace_tasks.tasks.shared.font_assets import font_asset_version, get_font_family_record
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_json_example import dump_prompt_json_examples
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec

from .shared.prompts import PROMPT_BUNDLE_ID as DEFAULT_PROMPT_BUNDLE_ID
from .shared.prompts import SCENE_PROMPT_KEY, build_equivalent_circuit_prompt_artifacts
from .shared.rendering import render_equivalent_circuit
from .shared.sampling import resolve_equivalent_circuit_scenario, sample_equivalent_circuit_layout
from .shared.state import SCENE_ID, SCENE_NAMESPACE


@dataclass(frozen=True)
class EquivalentCircuitObjective:
    """Task-owned semantic objective settings passed into the private lifecycle."""

    component_kind: str
    support_key: str
    task_prompt_key: str
    scene_kind_suffix: str
    object_description: str
    quantity_name: str
    component_name_plural: str
    annotation_hint: str
    answer_hint: str
    annotation_example: Sequence[int]


def _prompt_slots(objective: EquivalentCircuitObjective) -> Dict[str, str]:
    json_example, json_example_answer_only = dump_prompt_json_examples(
        annotation=list(objective.annotation_example),
        answer=8,
    )
    return {
        "object_description": str(objective.object_description),
        "quantity_name": str(objective.quantity_name),
        "component_name_plural": str(objective.component_name_plural),
        "annotation_hint": str(objective.annotation_hint),
        "answer_hint": str(objective.answer_hint),
        "json_example": str(json_example),
        "json_example_answer_only": str(json_example_answer_only),
    }


def run_equivalent_circuit_lifecycle(
    *,
    domain: str,
    public_task_id: str,
    lifecycle_namespace: str,
    objective: EquivalentCircuitObjective,
    generation_defaults: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    prompt_defaults: Mapping[str, Any],
    instance_seed: int,
    task_params: Mapping[str, Any],
    selected_branch: str,
    branch_probabilities: Mapping[str, float],
    max_attempts: int,
) -> TaskOutput:
    """Run the shared scene lifecycle for one objective-owned public task."""

    scenario = resolve_equivalent_circuit_scenario(
        instance_seed=int(instance_seed),
        params=task_params,
        defaults=generation_defaults,
        component_kind=str(objective.component_kind),
        support_key=str(objective.support_key),
        namespace=SCENE_NAMESPACE,
    )
    for attempt_index in range(max(1, int(max_attempts))):
        attempt_rng = spawn_rng(int(instance_seed), f"{lifecycle_namespace}.attempt.{int(attempt_index)}")
        try:
            layout = sample_equivalent_circuit_layout(
                attempt_rng,
                scenario=scenario,
                params=task_params,
                defaults=generation_defaults,
            )
        except ValueError:
            continue
        rendered = render_equivalent_circuit(
            instance_seed=int(instance_seed),
            params=task_params,
            scenario=scenario,
            layout=layout,
            render_config=rendering_defaults,
            namespace=SCENE_NAMESPACE,
        )
        resolved_prompt_defaults = required_group_defaults(
            prompt_defaults,
            ("bundle_id", "task_key"),
            context=f"prompt defaults for {public_task_id}",
        )
        prompt_artifacts = build_equivalent_circuit_prompt_artifacts(
            domain=str(domain),
            bundle_id=str(resolved_prompt_defaults.get("bundle_id", DEFAULT_PROMPT_BUNDLE_ID)),
            task_key=str(resolved_prompt_defaults.get("task_key", objective.task_prompt_key)),
            prompt_query_key=str(selected_branch),
            dynamic_slots=_prompt_slots(objective),
            instance_seed=int(instance_seed),
        )
        answer_gt = TypedValue(type="integer", value=int(scenario.target_answer))
        annotation_gt = TypedValue(type="bbox", value=list(rendered.annotation_bbox))
        font_record = get_font_family_record(str(rendered.font_family))
        prompt_query_spec = build_prompt_query_spec(
            prompt_artifacts=prompt_artifacts,
            query_id=str(selected_branch),
            params={
                "scene_variant": str(scenario.scene_variant),
                "component_kind": str(scenario.component_kind),
                "accent_color_name": str(scenario.accent_color_name),
                "target_answer": int(scenario.target_answer),
                "target_answer_support": [int(value) for value in scenario.target_answer_support],
                "query_id_probabilities": dict(branch_probabilities),
                "scene_variant_probabilities": dict(scenario.scene_variant_probabilities),
                "accent_color_name_probabilities": dict(scenario.accent_color_name_probabilities),
                "target_answer_probabilities": dict(scenario.target_answer_probabilities),
            },
        )
        render_spec = dict(rendered.render_spec)
        render_spec["font"] = {
            "font_family": str(rendered.font_family),
            "font_asset_version": font_asset_version(),
            "font_asset": font_record.to_trace(),
            "scope": SCENE_PROMPT_KEY,
            "selection_policy": {
                "pool": "global_approved_font_pool",
                "include_tags": [],
                "exclude_tags": [],
                "exclusion_reason": "",
            },
        }
        trace_payload = {
            "scene_ir": {
                "scene_kind": f"physics_equivalent_circuit_{objective.scene_kind_suffix}_{scenario.scene_variant}",
                "entities": [dict(entity) for entity in rendered.scene_entities],
                "relations": {
                    "scene_variant": str(scenario.scene_variant),
                    "component_kind": str(scenario.component_kind),
                    "target_answer": int(scenario.target_answer),
                    "accent_color_name": str(scenario.accent_color_name),
                    "annotation_entity_id_map": dict(rendered.annotation_entity_id_map),
                    "annotation_entity_id": "equivalent_circuit_network",
                },
            },
            "query_spec": prompt_query_spec,
            "render_spec": render_spec,
            "render_map": dict(rendered.render_map),
            "execution_trace": {
                "query_id": str(selected_branch),
                "scene_variant": str(scenario.scene_variant),
                "component_kind": str(scenario.component_kind),
                "accent_color_name": str(scenario.accent_color_name),
                "target_answer": int(scenario.target_answer),
                "target_answer_support": [int(value) for value in scenario.target_answer_support],
                "equivalent_value": int(layout.equivalent_value),
                "parallel_blocks": [[int(value) for value in block] for block in layout.parallel_blocks],
                "inter_block_series_values": [int(value) for value in layout.inter_block_series_values],
                "outer_series_values": [int(value) for value in layout.outer_series_values],
                "component_specs": [
                    {
                        "component_id": str(spec.component_id),
                        "label": str(spec.label),
                        "kind": str(spec.kind),
                        "value": int(spec.value),
                        "unit": str(spec.unit),
                    }
                    for spec in rendered.component_specs
                ],
                "annotation_entity_id_map": dict(rendered.annotation_entity_id_map),
            },
            "witness_symbolic": {
                "type": "bbox",
                "id": "equivalent_circuit_network",
                "component_entity_ids": [str(value) for value in rendered.annotation_entity_id_map.values()],
            },
            "projected_annotation": {
                "type": "bbox",
                "bbox": list(annotation_gt.value),
                "pixel_bbox": list(annotation_gt.value),
            },
            "background": dict(rendered.render_spec["background_style"]),
            "post_image_noise": dict(rendered.render_spec["post_image_noise"]),
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
    raise RuntimeError(f"{public_task_id} failed to generate a valid scene after {max_attempts} attempts")


__all__ = ["EquivalentCircuitObjective", "run_equivalent_circuit_lifecycle"]
