"""Neutral lifecycle helpers for radial-progress public tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from trace_tasks.core.seed import hash64
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import PromptTraceArtifacts, build_prompt_query_spec

from .shared.annotations import annotation_payload
from .shared.defaults import font_assets_payload
from .shared.prompts import build_prompt_artifacts, dynamic_slots
from .shared.rendering import render_radial_progress_dataset
from .shared.sampling import build_progress_items
from .shared.state import ProgressDataset, ProgressFrame, ProgressItem, ProgressQuestion, RadialProgressRenderResult, SCENE_ID


@dataclass(frozen=True)
class RadialProgressTaskPlan:
    dataset: ProgressDataset
    prompt_key: str
    task_prompt_key: str
    question_format: str
    witness_type: str
    trace_params: dict[str, Any]


def attempt_seed(instance_seed: int, unique_key: str, attempt: int) -> int:
    return int(instance_seed) if int(attempt) == 0 else int(hash64(int(instance_seed), str(unique_key), int(attempt)))


def build_radial_progress_plan(
    *,
    dataset: ProgressDataset,
    prompt_key: str,
    task_prompt_key: str,
    question_format: str,
    witness_type: str,
    trace_params: Mapping[str, Any] | None = None,
) -> RadialProgressTaskPlan:
    return RadialProgressTaskPlan(
        dataset=dataset,
        prompt_key=str(prompt_key),
        task_prompt_key=str(task_prompt_key),
        question_format=str(question_format),
        witness_type=str(witness_type),
        trace_params=dict(trace_params or {}),
    )


def build_progress_dataset_from_components(
    *,
    items: tuple[ProgressItem, ...],
    scene_variant: str,
    branch_id: str,
    branch_probabilities: Mapping[str, float],
    answer: int | str,
    answer_type: str,
    annotation_type: str,
    annotation_item_ids: tuple[str, ...],
    question_params: Mapping[str, Any],
    title: str,
) -> ProgressDataset:
    """Create the scene dataset after a public task has bound its objective."""

    return ProgressDataset(
        items=tuple(items),
        scene_variant=str(scene_variant),
        branch_id=str(branch_id),
        branch_probabilities={str(key): float(value) for key, value in branch_probabilities.items()},
        question=ProgressQuestion(
            branch_id=str(branch_id),
            answer=answer,
            answer_type=str(answer_type),
            annotation_type=str(annotation_type),
            annotation_item_ids=tuple(str(value) for value in annotation_item_ids),
            params=dict(question_params),
        ),
        title=str(title),
    )


def build_progress_dataset_from_frame(
    *,
    frame: ProgressFrame,
    values: list[int],
    branch_id: str,
    branch_probabilities: Mapping[str, float],
    answer: int | str,
    answer_type: str,
    annotation_type: str,
    annotation_item_ids: tuple[str, ...],
    question_params: Mapping[str, Any],
) -> ProgressDataset:
    return build_progress_dataset_from_components(
        items=build_progress_items(labels=frame.labels, values=values, colors=frame.colors),
        scene_variant=str(frame.scene_variant),
        branch_id=str(branch_id),
        branch_probabilities=dict(branch_probabilities),
        answer=answer,
        answer_type=str(answer_type),
        annotation_type=str(annotation_type),
        annotation_item_ids=tuple(annotation_item_ids),
        question_params=dict(question_params),
        title=str(frame.title),
    )


def build_count_dataset_from_frame(
    *,
    frame: ProgressFrame,
    values: list[int],
    branch_id: str,
    branch_probabilities: Mapping[str, float],
    answer_count: int,
    answer_support: list[int],
    answer_probabilities: Mapping[str, float],
    annotation_type: str,
    annotation_item_ids: tuple[str, ...],
    question_params: Mapping[str, Any],
) -> ProgressDataset:
    params = {
        **dict(question_params),
        "answer_count": int(answer_count),
        "answer_count_support": list(answer_support),
        "answer_count_probabilities": dict(answer_probabilities),
        "item_count_probabilities": dict(frame.item_count_probabilities),
    }
    return build_progress_dataset_from_frame(
        frame=frame,
        values=values,
        branch_id=str(branch_id),
        branch_probabilities=dict(branch_probabilities),
        answer=int(answer_count),
        answer_type="integer",
        annotation_type=str(annotation_type),
        annotation_item_ids=tuple(annotation_item_ids),
        question_params=params,
    )


def build_label_dataset_from_frame(
    *,
    frame: ProgressFrame,
    values: list[int],
    branch_id: str,
    branch_probabilities: Mapping[str, float],
    answer: str,
    annotation_type: str,
    annotation_item_ids: tuple[str, ...],
    question_params: Mapping[str, Any],
) -> ProgressDataset:
    params = {
        **dict(question_params),
        "target_label": str(answer),
        "label_support": [str(label) for label in frame.labels],
        "item_count_probabilities": dict(frame.item_count_probabilities),
    }
    return build_progress_dataset_from_frame(
        frame=frame,
        values=values,
        branch_id=str(branch_id),
        branch_probabilities=dict(branch_probabilities),
        answer=str(answer),
        answer_type="string",
        annotation_type=str(annotation_type),
        annotation_item_ids=tuple(annotation_item_ids),
        question_params=params,
    )


def build_count_plan(
    *,
    dataset: ProgressDataset,
    prompt_key: str,
    scene_probabilities: Mapping[str, float],
) -> RadialProgressTaskPlan:
    return build_radial_progress_plan(
        dataset=dataset,
        prompt_key=str(prompt_key),
        task_prompt_key="radial_progress_condition_count_query",
        question_format="radial_progress_condition_count",
        witness_type="radial_progress_condition_count_witness",
        trace_params={"scene_variant_probabilities": dict(scene_probabilities)},
    )


def build_label_plan(
    *,
    dataset: ProgressDataset,
    prompt_key: str,
    scene_probabilities: Mapping[str, float],
) -> RadialProgressTaskPlan:
    return build_radial_progress_plan(
        dataset=dataset,
        prompt_key=str(prompt_key),
        task_prompt_key="radial_progress_remaining_extremum_query",
        question_format="radial_progress_extremum_remaining_label",
        witness_type="radial_progress_extremum_remaining_label_witness",
        trace_params={"scene_variant_probabilities": dict(scene_probabilities)},
    )


def _trace_payload(
    *,
    plan: RadialProgressTaskPlan,
    rendered: RadialProgressRenderResult,
    prompt_artifacts: PromptTraceArtifacts,
    annotation: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble metadata after a public task has selected operands and witnesses."""

    dataset = plan.dataset
    rendered_scene = rendered.rendered_scene
    label_to_value = {str(item.label): int(item.value) for item in dataset.items}
    params = {
        "query_id": str(dataset.branch_id),
        "query_id_probabilities": dict(dataset.branch_probabilities),
        "scene_variant": str(dataset.scene_variant),
        "item_count": int(len(dataset.items)),
        "answer_value": dataset.question.answer,
        "answer_type": str(dataset.question.answer_type),
        "annotation_type": str(annotation["type"]),
        **dict(dataset.question.params),
        **dict(plan.trace_params),
    }
    return {
        "scene_ir": {
            "scene_kind": SCENE_ID,
            "entities": [dict(entity) for entity in rendered_scene.entities],
            "relations": {
                "query_id": str(dataset.branch_id),
                "scene_variant": str(dataset.scene_variant),
                "answer_value": dataset.question.answer,
                "annotation_item_ids": list(annotation["item_ids"]),
            },
        },
        "query_spec": build_prompt_query_spec(
            prompt_artifacts=prompt_artifacts,
            query_id=str(dataset.branch_id),
            params=dict(params),
        ),
        "render_spec": {
            "scene_variant": str(dataset.scene_variant),
            "canvas_width": int(rendered.image.size[0]),
            "canvas_height": int(rendered.image.size[1]),
            "category_labels": [str(item.label) for item in dataset.items],
            "render_meta": dict(rendered_scene.render_meta),
            "font_assets": font_assets_payload(chart_font_family=rendered.chart_font_family),
            "background_style": dict(rendered.background_meta),
            "information_scene_style": dict(rendered.background_meta.get("information_scene_style", {})),
            "post_image_noise": dict(rendered.post_noise_meta),
        },
        "render_map": {
            "plot_bbox_px": list(rendered_scene.plot_bbox_px),
            "item_bboxes_px": dict(rendered_scene.item_bboxes_px),
            "progress_bboxes_px": dict(rendered_scene.progress_bboxes_px),
        },
        "execution_trace": {
            "query_id": str(dataset.branch_id),
            "question_format": str(plan.question_format),
            "scene_variant": str(dataset.scene_variant),
            "item_count": int(len(dataset.items)),
            "items": [dict(entity) for entity in rendered_scene.entities],
            "label_to_value": dict(label_to_value),
            "answer_value": dataset.question.answer,
            "answer_type": str(dataset.question.answer_type),
            "annotation_item_ids": list(annotation["item_ids"]),
            "annotation_labels": list(annotation["labels"]),
            **dict(dataset.question.params),
            **dict(plan.trace_params),
        },
        "witness_symbolic": {
            "type": str(plan.witness_type),
            "answer_value": dataset.question.answer,
            "annotation_item_ids": list(annotation["item_ids"]),
        },
        "projected_annotation": dict(annotation["projected_annotation"]),
        "background": dict(rendered.background_meta),
        "post_image_noise": dict(rendered.post_noise_meta),
    }


def materialize_plan(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    plan: RadialProgressTaskPlan,
) -> TaskOutput:
    rendered = render_radial_progress_dataset(dataset=plan.dataset, params=params, instance_seed=int(instance_seed))
    annotation = annotation_payload(dataset=plan.dataset, rendered=rendered)
    prompt_artifacts = build_prompt_artifacts(
        prompt_query_key=str(plan.prompt_key),
        is_label_answer=str(plan.task_prompt_key) == "radial_progress_remaining_extremum_query",
        dynamic_slot_values=dynamic_slots(dataset=plan.dataset),
        instance_seed=int(instance_seed),
    )
    trace_payload = _trace_payload(
        plan=plan,
        rendered=rendered,
        prompt_artifacts=prompt_artifacts,
        annotation=annotation,
    )
    return TaskOutput(
        prompt=str(prompt_artifacts.prompt),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
        answer_gt=TypedValue(type=str(plan.dataset.question.answer_type), value=plan.dataset.question.answer),
        annotation_gt=TypedValue(type=str(annotation["type"]), value=annotation["value"]),
        image=rendered.image,
        image_id="img0",
        trace_payload=trace_payload,
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
        query_id=str(plan.dataset.branch_id),
    )


def run_radial_progress_task(public_task: Any, instance_seed: int, params: dict[str, Any], max_attempts: int) -> TaskOutput:
    selected, probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=dict(params),
        supported_query_ids=tuple(str(value) for value in public_task.supported_query_ids),
        default_query_id=str(public_task.default_query_id),
        task_id=str(public_task.task_id),
    )
    last_error: Exception | None = None
    for attempt in range(max(1, int(max_attempts))):
        current_seed = attempt_seed(int(instance_seed), str(public_task.task_id), int(attempt))
        try:
            plan = public_task._build_plan(
                dict(task_params),
                int(current_seed),
                str(selected),
                dict(probabilities),
            )
            return materialize_plan(params=dict(task_params), instance_seed=int(current_seed), plan=plan)
        except ValueError as exc:
            last_error = exc
    raise RuntimeError(f"failed to generate {public_task.task_id}: {last_error}")
