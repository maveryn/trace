"""Scene-private output assembly for shape-reference tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.shared.config_defaults import split_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import PromptTraceArtifacts, build_prompt_query_spec
from trace_tasks.tasks.geometry.shared.graph_rendering import graph_paper_grid_from_frame

from .shared.construction import TransformationSceneBundle, compose_transformation_scene
from .shared.prompts import relation_prompt_artifacts, transformation_prompt_artifacts
from .shared.relations import SCENE_ID, SimilaritySceneBundle, compose_similarity_scene


@dataclass(frozen=True)
class ShapeReferenceObjectivePlan:
    """Task-owned public objective parameters for one shape-reference instance."""

    scene_family: str
    config_group_key: str
    prompt_branch_key: str
    relation_rule: str | None = None
    transform_rule: str | None = None
    program_scope: str = ""
    trace_values: Mapping[str, Any] | None = None


def _scene_defaults_for_group(group_key: str) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    raw_defaults = get_scene_defaults("geometry", SCENE_ID)
    return split_scene_generation_rendering_prompt_defaults(
        raw_defaults if isinstance(raw_defaults, Mapping) else {},
        task_id=str(group_key),
    )


def _relation_annotation(bundle: SimilaritySceneBundle) -> list[list[float]]:
    rendered = bundle.rendered_scene
    return [list(point) for point in rendered.annotation.get("annotation_value", [])]


def _relation_trace_payload(
    *,
    task_id: str,
    public_query_id: str,
    public_query_probabilities: Mapping[str, float],
    prompt_defaults: Mapping[str, Any],
    prompt_artifacts: PromptTraceArtifacts,
    plan: ShapeReferenceObjectivePlan,
    bundle: SimilaritySceneBundle,
    annotation_points: list[list[float]],
) -> dict[str, Any]:
    """Serialize relation-match trace data without choosing public semantics."""

    _ = prompt_defaults
    resolved = bundle.resolved
    rendered = bundle.rendered_scene
    query_params = {
        "scene_variant": str(resolved.scene_variant),
        "query_id": str(public_query_id),
        "query_id_probabilities": dict(public_query_probabilities),
        "relation_rule": str(resolved.relation_rule),
        "relation_rule_probabilities": dict(resolved.relation_rule_probabilities),
        "scene_variant_probabilities": dict(resolved.scene_variant_probabilities),
        "winner_label_probabilities": dict(resolved.winner_label_probabilities),
        "candidate_label_pool": list(resolved.candidate_label_pool),
        "candidate_count_probabilities": dict(resolved.candidate_count_probabilities),
    }
    trace_values = {
        "task_id": str(task_id),
        "scene_id": SCENE_ID,
        "query_id": str(public_query_id),
        "program_scope": str(plan.program_scope),
        "relation_rule": str(resolved.relation_rule),
        "scene_variant": str(resolved.scene_variant),
        "winner_label": str(rendered.winner_label),
        **dict(plan.trace_values or {}),
    }
    return {
        "scene_ir": {
            "scene_kind": "geometry_shape_reference_relation_match",
            "entities": [dict(entity) for entity in rendered.scene_entities],
            "relations": dict(trace_values),
        },
        "query_spec": build_prompt_query_spec(
            prompt_artifacts=prompt_artifacts,
            query_id=str(public_query_id),
            params=query_params,
        ),
        "render_spec": {
            "canvas_size": int(bundle.context.canvas_size),
            "coord_space": "pixel",
            "background_style": dict(bundle.background_meta),
            "post_image_noise": dict(bundle.post_noise_meta),
            "shape_style": dict(bundle.shape_style_trace),
            "text_style": {
                "font_size_px": int(bundle.label_font_size_px),
                "stroke_width_px": int(bundle.label_stroke_width_px),
            },
            "graph_coordinate_frame": dict(bundle.context.graph_frame),
            "graph_paper_grid": graph_paper_grid_from_frame(bundle.context.graph_frame),
            **dict(bundle.context.graph_layout_metadata),
            "scene_variant": str(resolved.scene_variant),
            "candidate_count": int(len(resolved.candidate_label_pool)),
            "candidate_count_probabilities": dict(resolved.candidate_count_probabilities),
        },
        "render_map": {
            **dict(rendered.render_map),
            "image_id": "img0",
            "winner_label": str(rendered.winner_label),
        },
        "execution_trace": {
            **dict(trace_values),
            "query_id_probabilities": dict(public_query_probabilities),
            "relation_rule_probabilities": dict(resolved.relation_rule_probabilities),
            "scene_variant_probabilities": dict(resolved.scene_variant_probabilities),
            "winner_label_probabilities": dict(resolved.winner_label_probabilities),
            "candidate_count_probabilities": dict(resolved.candidate_count_probabilities),
            "reference_center_graph": list(bundle.rendered_scene.scene_entities[0].get("center_graph", [])),
            "required_annotation_labels": list(rendered.required_annotation_labels),
            "question_format": "label_choice_no_text_options",
        },
        "witness_symbolic": {
            **dict(rendered.annotation["witness_symbolic"]),
            "winner_label": str(rendered.winner_label),
        },
        "projected_annotation": {
            **dict(rendered.annotation["projected_annotation"]),
            "point_set": list(annotation_points),
            "pixel_point_set": list(annotation_points),
        },
    }


def _transformation_trace_payload(
    *,
    task_id: str,
    public_query_id: str,
    public_query_probabilities: Mapping[str, float],
    prompt_defaults: Mapping[str, Any],
    prompt_artifacts: PromptTraceArtifacts,
    plan: ShapeReferenceObjectivePlan,
    bundle: TransformationSceneBundle,
    annotation_points: list[list[float]],
) -> dict[str, Any]:
    """Serialize transform-selection trace data without choosing public semantics."""

    _ = prompt_defaults
    resolved = bundle.resolved
    rendered = bundle.rendered_scene
    query_params: dict[str, Any] = {
        "scene_variant": str(resolved.scene_variant),
        "query_id": str(public_query_id),
        "query_id_probabilities": dict(public_query_probabilities),
        "transform_rule": str(resolved.transform_rule),
        "transform_rule_probabilities": dict(resolved.transform_rule_probabilities),
        "scene_variant_probabilities": dict(resolved.scene_variant_probabilities),
        "winner_label_probabilities": dict(resolved.winner_label_probabilities),
        "candidate_label_pool": list(resolved.candidate_label_pool),
        "candidate_count_probabilities": dict(resolved.candidate_count_probabilities),
    }
    if rendered.translation_vector is not None:
        query_params["translation_vector"] = [int(rendered.translation_vector[0]), int(rendered.translation_vector[1])]
    if rendered.rotation_mode is not None:
        query_params["rotation_mode"] = str(rendered.rotation_mode)
        query_params["rotation_instruction"] = str(rendered.rotation_prompt_label)

    trace_values = {
        "task_id": str(task_id),
        "scene_id": SCENE_ID,
        "query_id": str(public_query_id),
        "program_scope": str(plan.program_scope),
        "transform_rule": str(resolved.transform_rule),
        "scene_variant": str(resolved.scene_variant),
        "winner_label": str(rendered.winner_label),
        "cue_kind": str(rendered.cue_kind),
        **dict(plan.trace_values or {}),
    }
    return {
        "scene_ir": {
            "scene_kind": "geometry_shape_reference_transform_selection",
            "entities": [dict(entity) for entity in rendered.scene_entities],
            "relations": dict(trace_values),
        },
        "query_spec": build_prompt_query_spec(
            prompt_artifacts=prompt_artifacts,
            query_id=str(public_query_id),
            params=query_params,
        ),
        "render_spec": {
            "canvas_size": int(bundle.context.canvas_size),
            "coord_space": "pixel",
            "background_style": dict(bundle.background_meta),
            "post_image_noise": dict(bundle.post_noise_meta),
            "shape_style": dict(bundle.shape_style_trace),
            "text_style": {
                "font_size_px": int(bundle.label_font_size_px),
                "stroke_width_px": int(bundle.label_stroke_width_px),
            },
            "graph_coordinate_frame": dict(bundle.context.graph_frame),
            "graph_paper_grid": graph_paper_grid_from_frame(bundle.context.graph_frame),
            **dict(bundle.context.graph_layout_metadata),
            "scene_variant": str(resolved.scene_variant),
            "candidate_count": int(len(resolved.candidate_label_pool)),
            "candidate_count_probabilities": dict(resolved.candidate_count_probabilities),
        },
        "render_map": {
            **dict(rendered.render_map),
            "image_id": "img0",
            "winner_label": str(rendered.winner_label),
        },
        "execution_trace": {
            **dict(trace_values),
            "query_id_probabilities": dict(public_query_probabilities),
            "transform_rule_probabilities": dict(resolved.transform_rule_probabilities),
            "scene_variant_probabilities": dict(resolved.scene_variant_probabilities),
            "winner_label_probabilities": dict(resolved.winner_label_probabilities),
            "candidate_count_probabilities": dict(resolved.candidate_count_probabilities),
            "required_annotation_labels": list(rendered.required_annotation_labels),
            "question_format": "label_choice_no_text_options",
            "rotation_mode": (str(rendered.rotation_mode) if rendered.rotation_mode is not None else None),
            "rotation_instruction": (str(rendered.rotation_prompt_label) if rendered.rotation_prompt_label is not None else None),
            "translation_vector": (
                [int(rendered.translation_vector[0]), int(rendered.translation_vector[1])]
                if rendered.translation_vector is not None
                else None
            ),
        },
        "witness_symbolic": {
            **dict(rendered.annotation["witness_symbolic"]),
            "winner_label": str(rendered.winner_label),
        },
        "projected_annotation": {
            **dict(rendered.annotation["projected_annotation"]),
            "point_set": list(annotation_points),
            "pixel_point_set": list(annotation_points),
        },
    }


def run_shape_reference_public_entry(
    task: Any,
    instance_seed: int,
    *,
    params: Mapping[str, Any],
    max_attempts: int,
) -> TaskOutput:
    """Run neutral scene plumbing after the public task resolves objective semantics."""

    selected_branch, branch_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=dict(params),
        supported_query_ids=tuple(str(value) for value in task.supported_query_ids),
        default_query_id=str(task.default_query_id),
        task_id=str(task.task_id),
        namespace=f"{task.task_id}.query",
    )
    last_error: Exception | None = None
    for attempt_index in range(max(1, int(max_attempts))):
        attempt_seed = int(instance_seed) + int(attempt_index)
        try:
            plan = task.prepare_objective(
                attempt_seed,
                str(selected_branch),
                dict(branch_probabilities),
                dict(task_params),
            )
            _generation_defaults, _rendering_defaults, prompt_defaults = _scene_defaults_for_group(str(plan.config_group_key))
            if str(plan.scene_family) == "relation_match":
                if plan.relation_rule is None:
                    raise ValueError("relation-match objective requires relation_rule")
                bundle = compose_similarity_scene(
                    attempt_seed,
                    params={**dict(task_params), "_render_attempt": int(attempt_index)},
                    max_attempts=1,
                    relation_rule=str(plan.relation_rule),
                    seed_namespace=str(task.task_id),
                )
                rendered = bundle.rendered_scene
                annotation_points = _relation_annotation(bundle)
                prompt_defaults_used, prompt_artifacts = relation_prompt_artifacts(
                    prompt_defaults=prompt_defaults,
                    prompt_branch_key=str(plan.prompt_branch_key),
                    annotation_value=annotation_points,
                    vertex_count=len(rendered.required_annotation_labels),
                    instance_seed=attempt_seed,
                )
                trace_payload = _relation_trace_payload(
                    task_id=str(task.task_id),
                    public_query_id=str(selected_branch),
                    public_query_probabilities=branch_probabilities,
                    prompt_defaults=prompt_defaults_used,
                    prompt_artifacts=prompt_artifacts,
                    plan=plan,
                    bundle=bundle,
                    annotation_points=annotation_points,
                )
                return TaskOutput(
                    prompt=str(prompt_artifacts.prompt),
                    answer_gt=TypedValue(type="option_letter", value=str(rendered.winner_label)),
                    annotation_gt=TypedValue(type=str(rendered.annotation["annotation_type"]), value=list(annotation_points)),
                    image=bundle.image,
                    image_id="img0",
                    trace_payload=dict(trace_payload),
                    task_versions=default_task_versions(),
                    scene_id=SCENE_ID,
                    query_id=str(selected_branch),
                    prompt_variants=dict(prompt_artifacts.prompt_variants),
                )

            if str(plan.scene_family) == "transform_selection":
                if plan.transform_rule is None:
                    raise ValueError("transform-selection objective requires transform_rule")
                bundle = compose_transformation_scene(
                    attempt_seed,
                    params={**dict(task_params), "_render_attempt": int(attempt_index)},
                    max_attempts=1,
                    transform_rule=str(plan.transform_rule),
                    seed_namespace=str(task.task_id),
                )
                rendered = bundle.rendered_scene
                annotation_points = [list(point) for point in rendered.annotation.get("annotation_value", [])]
                prompt_defaults_used, prompt_artifacts = transformation_prompt_artifacts(
                    prompt_defaults=prompt_defaults,
                    prompt_branch_key=str(plan.prompt_branch_key),
                    annotation_value=annotation_points,
                    vertex_count=len(rendered.required_annotation_labels),
                    rotation_instruction=rendered.rotation_prompt_label,
                    instance_seed=attempt_seed,
                )
                trace_payload = _transformation_trace_payload(
                    task_id=str(task.task_id),
                    public_query_id=str(selected_branch),
                    public_query_probabilities=branch_probabilities,
                    prompt_defaults=prompt_defaults_used,
                    prompt_artifacts=prompt_artifacts,
                    plan=plan,
                    bundle=bundle,
                    annotation_points=annotation_points,
                )
                return TaskOutput(
                    prompt=str(prompt_artifacts.prompt),
                    answer_gt=TypedValue(type="option_letter", value=str(rendered.answer_value)),
                    annotation_gt=TypedValue(type=str(rendered.annotation["annotation_type"]), value=list(annotation_points)),
                    image=bundle.image,
                    image_id="img0",
                    trace_payload=dict(trace_payload),
                    task_versions=default_task_versions(),
                    scene_id=SCENE_ID,
                    query_id=str(selected_branch),
                    prompt_variants=dict(prompt_artifacts.prompt_variants),
                )
            raise ValueError(f"unsupported shape-reference scene family: {plan.scene_family}")
        except Exception as exc:
            if isinstance(exc, ValueError) and (
                "unsupported" in str(exc)
                or "incompatible" in str(exc)
                or "requires" in str(exc)
            ):
                raise
            last_error = exc
            continue
    raise RuntimeError(f"failed to generate {task.task_id}") from last_error


__all__ = ["ShapeReferenceObjectivePlan", "run_shape_reference_public_entry"]
