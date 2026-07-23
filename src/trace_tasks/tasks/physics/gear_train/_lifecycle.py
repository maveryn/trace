"""Private lifecycle helpers for gear-train public objectives."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec

from .shared.output import build_object_witness, build_render_spec
from .shared.prompts import PROMPT_BUNDLE_ID, SCENE_PROMPT_KEY, build_gear_train_prompt_artifacts
from .shared.rendering import render_direction_choice_scene, render_speed_scene
from .shared.sampling import resolve_direction_choice_scenario, resolve_speed_scenario
from .shared.state import DIRECTION_OPTION_LETTERS, GearTrainDefaults, SCENE_ID


def run_direction_lifecycle(
    *,
    domain: str,
    task_prompt_key: str,
    lifecycle_namespace: str,
    generation_defaults: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    prompt_defaults: Mapping[str, Any],
    instance_seed: int,
    task_params: Mapping[str, Any],
    selected_branch: str,
    branch_probabilities: Mapping[str, float],
    max_attempts: int,
) -> TaskOutput:
    """Run direction objective generation while preserving task-owned branch selection."""

    fallback = GearTrainDefaults()
    last_error: Exception | None = None
    for attempt in range(max(1, int(max_attempts))):
        attempt_seed = int(instance_seed) + int(attempt) * 1009
        try:
            scenario = resolve_direction_choice_scenario(
                instance_seed=int(attempt_seed),
                params=task_params,
                defaults=generation_defaults,
                namespace=str(lifecycle_namespace),
            )
            rendered = render_direction_choice_scene(
                instance_seed=int(attempt_seed),
                params=task_params,
                scenario=scenario,
                render_defaults=rendering_defaults,
                fallback=fallback,
                namespace=str(lifecycle_namespace),
            )
            prompt_artifacts = build_gear_train_prompt_artifacts(
                domain=str(domain),
                bundle_id=str(prompt_defaults.get("bundle_id", PROMPT_BUNDLE_ID)),
                task_key=str(prompt_defaults.get("task_key", task_prompt_key)),
                prompt_query_key="output_direction",
                dynamic_slots={
                    "target_direction": str(scenario.target_direction),
                },
                instance_seed=int(attempt_seed),
                scene_key="gear_train_panel_options",
            )
            answer_gt = TypedValue(type="option_letter", value=str(scenario.correct_option_letter))
            annotation_gt = TypedValue(type="bbox", value=list(rendered.annotation_bbox_map["selected_panel"]))
            prompt_query_spec = build_prompt_query_spec(
                prompt_artifacts=prompt_artifacts,
                query_id=str(selected_branch),
                params={
                    "query_id": str(selected_branch),
                    "operation": "output_direction_choice",
                    "target_direction": str(scenario.target_direction),
                    "target_answer": str(scenario.correct_option_letter),
                    "answer_support": list(DIRECTION_OPTION_LETTERS),
                    "query_id_probabilities": dict(branch_probabilities),
                    "scene_variant_probabilities": dict(scenario.scene_variant_probabilities),
                    "gear_count_probabilities": dict(scenario.gear_count_probabilities),
                    "target_direction_probabilities": dict(scenario.target_direction_probabilities),
                    "correct_option_letter_probabilities": dict(scenario.correct_option_letter_probabilities),
                },
            )
            trace_payload = {
                "scene_ir": {
                    "scene_kind": "physics_gear_train_direction_choice",
                    "entities": list(rendered.scene_entities),
                    "relations": {
                        "operation": "output_direction_choice",
                        "target_direction": str(scenario.target_direction),
                        "correct_option_letter": str(scenario.correct_option_letter),
                        "panel_output_directions": dict(rendered.render_map.get("panel_output_directions", {})),
                    },
                },
                "query_spec": dict(prompt_query_spec),
                "render_spec": build_render_spec(rendered=rendered, prompt_scope="gear_train_panel_options"),
                "render_map": dict(rendered.render_map),
                "execution_trace": {
                    "query_id": str(selected_branch),
                    "operation": "output_direction_choice",
                    "target_direction": str(scenario.target_direction),
                    "correct_option_letter": str(scenario.correct_option_letter),
                    "answer_support": list(DIRECTION_OPTION_LETTERS),
                    "panel_output_directions": dict(rendered.render_map.get("panel_output_directions", {})),
                    "panel_input_directions": dict(rendered.render_map.get("panel_input_directions", {})),
                    "panel_gear_counts": dict(rendered.render_map.get("panel_gear_counts", {})),
                    "annotation_entity_ids": ["selected_panel"],
                },
                "sampling": {
                    "query_id_probabilities": dict(branch_probabilities),
                    "scene_variant_probabilities": dict(scenario.scene_variant_probabilities),
                    "gear_count_probabilities": dict(scenario.gear_count_probabilities),
                    "target_direction_probabilities": dict(scenario.target_direction_probabilities),
                    "correct_option_letter_probabilities": dict(scenario.correct_option_letter_probabilities),
                    "target_answer_probabilities": dict(scenario.correct_option_letter_probabilities),
                },
                "witness_symbolic": {
                    "type": "object",
                    "id": "selected_panel",
                },
                "projected_annotation": {
                    "type": "bbox",
                    "bbox": list(annotation_gt.value),
                    "pixel_bbox": list(annotation_gt.value),
                },
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
        except Exception as exc:
            last_error = exc
            continue
    raise RuntimeError(f"failed to generate gear-train direction instance after {max_attempts} attempts: {last_error}")


def run_speed_lifecycle(
    *,
    domain: str,
    task_prompt_key: str,
    lifecycle_namespace: str,
    generation_defaults: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    prompt_defaults: Mapping[str, Any],
    instance_seed: int,
    task_params: Mapping[str, Any],
    selected_branch: str,
    branch_probabilities: Mapping[str, float],
    max_attempts: int,
) -> TaskOutput:
    """Run speed objective generation while preserving exact ratio semantics."""

    fallback = GearTrainDefaults()
    last_error: Exception | None = None
    for attempt in range(max(1, int(max_attempts))):
        attempt_seed = int(instance_seed) + int(attempt) * 1009
        try:
            scenario = resolve_speed_scenario(
                instance_seed=int(attempt_seed),
                params=task_params,
                defaults=generation_defaults,
                namespace=str(lifecycle_namespace),
            )
            rendered = render_speed_scene(
                instance_seed=int(attempt_seed),
                params=task_params,
                scenario=scenario,
                render_defaults=rendering_defaults,
                fallback=fallback,
                namespace=str(lifecycle_namespace),
            )
            prompt_artifacts = build_gear_train_prompt_artifacts(
                domain=str(domain),
                bundle_id=str(prompt_defaults.get("bundle_id", PROMPT_BUNDLE_ID)),
                task_key=str(prompt_defaults.get("task_key", task_prompt_key)),
                prompt_query_key="output_speed",
                dynamic_slots={},
                instance_seed=int(attempt_seed),
            )
            answer_gt = TypedValue(type="integer", value=int(scenario.output_rpm))
            annotation_gt = TypedValue(
                type="bbox_map",
                value={str(key): list(value) for key, value in rendered.annotation_bbox_map.items()},
            )
            prompt_query_spec = build_prompt_query_spec(
                prompt_artifacts=prompt_artifacts,
                query_id=str(selected_branch),
                params={
                    "query_id": str(selected_branch),
                    "operation": "output_speed",
                    "scene_variant": str(scenario.scene_variant),
                    "gear_count": int(scenario.gear_count),
                    "input_teeth": int(scenario.input_teeth),
                    "output_teeth": int(scenario.output_teeth),
                    "idler_teeth": [int(value) for value in scenario.idler_teeth],
                    "input_rpm": int(scenario.input_rpm),
                    "target_answer": int(scenario.output_rpm),
                    "speed_relation": str(scenario.speed_relation),
                    "query_id_probabilities": dict(branch_probabilities),
                    "scene_variant_probabilities": dict(scenario.scene_variant_probabilities),
                    "gear_count_probabilities": dict(scenario.gear_count_probabilities),
                    "speed_relation_probabilities": dict(scenario.speed_relation_probabilities),
                    "input_rpm_probabilities": dict(scenario.input_rpm_probabilities),
                    "target_answer_probabilities": dict(scenario.target_answer_probabilities),
                },
            )
            trace_payload = {
                "scene_ir": {
                    "scene_kind": "physics_gear_train_speed",
                    "entities": list(rendered.scene_entities),
                    "relations": {
                        "operation": "output_speed",
                        "gear_count": int(scenario.gear_count),
                        "input_teeth": int(scenario.input_teeth),
                        "output_teeth": int(scenario.output_teeth),
                        "input_rpm": int(scenario.input_rpm),
                        "output_rpm": int(scenario.output_rpm),
                        "speed_relation": str(scenario.speed_relation),
                        "scene_variant": str(scenario.scene_variant),
                    },
                },
                "query_spec": dict(prompt_query_spec),
                "render_spec": build_render_spec(rendered=rendered, prompt_scope=SCENE_PROMPT_KEY),
                "render_map": dict(rendered.render_map),
                "execution_trace": {
                    "query_id": str(selected_branch),
                    "operation": "output_speed",
                    "gear_count": int(scenario.gear_count),
                    "input_teeth": int(scenario.input_teeth),
                    "output_teeth": int(scenario.output_teeth),
                    "idler_teeth": [int(value) for value in scenario.idler_teeth],
                    "input_rpm": int(scenario.input_rpm),
                    "output_rpm": int(scenario.output_rpm),
                    "ratio_equation": f"{int(scenario.input_rpm)}*{int(scenario.input_teeth)}/{int(scenario.output_teeth)}",
                    "speed_relation": str(scenario.speed_relation),
                    "annotation_entity_ids": sorted(annotation_gt.value.keys()),
                },
                "sampling": {
                    "query_id_probabilities": dict(branch_probabilities),
                    "scene_variant_probabilities": dict(scenario.scene_variant_probabilities),
                    "gear_count_probabilities": dict(scenario.gear_count_probabilities),
                    "speed_relation_probabilities": dict(scenario.speed_relation_probabilities),
                    "input_rpm_probabilities": dict(scenario.input_rpm_probabilities),
                    "target_answer_probabilities": dict(scenario.target_answer_probabilities),
                },
                "witness_symbolic": build_object_witness(ids=sorted(annotation_gt.value.keys())),
                "projected_annotation": {
                    "type": "bbox_map",
                    "bbox_map": dict(annotation_gt.value),
                    "pixel_bbox_map": dict(annotation_gt.value),
                },
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
        except Exception as exc:
            last_error = exc
            continue
    raise RuntimeError(f"failed to generate gear-train speed instance after {max_attempts} attempts: {last_error}")


__all__ = ["run_direction_lifecycle", "run_speed_lifecycle"]
