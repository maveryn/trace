"""Neutral lifecycle helpers for population-pyramid public tasks."""

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
from .shared.prompts import build_prompt_artifacts, dynamic_slots
from .shared.rendering import PopulationPyramidRenderResult, render_population_pyramid_dataset
from .shared.state import SCENE_ID, PopulationPyramidDataset


@dataclass(frozen=True)
class PopulationPyramidTaskPlan:
    dataset: PopulationPyramidDataset
    prompt_query_key: str
    trace_params: dict[str, Any]


def population_pyramid_attempt_seed(instance_seed: int, unique_key: str, attempt: int) -> int:
    return int(instance_seed) if int(attempt) == 0 else int(hash64(int(instance_seed), str(unique_key), int(attempt)))


def build_population_pyramid_plan(
    *,
    dataset: PopulationPyramidDataset,
    prompt_query_key: str,
    trace_params: Mapping[str, Any] | None = None,
) -> PopulationPyramidTaskPlan:
    return PopulationPyramidTaskPlan(
        dataset=dataset,
        prompt_query_key=str(prompt_query_key),
        trace_params=dict(trace_params or {}),
    )


def _row_records(dataset: PopulationPyramidDataset) -> list[dict[str, Any]]:
    return [
        {
            "row_id": str(row.row_id),
            "label": str(row.label),
            "left_value": int(row.left_value),
            "right_value": int(row.right_value),
            "gap": int(row.gap),
            "total": int(row.total),
        }
        for row in dataset.rows
    ]


def _build_trace_payload(
    *,
    dataset: PopulationPyramidDataset,
    rendered: PopulationPyramidRenderResult,
    prompt_artifacts: PromptTraceArtifacts,
    annotation: Mapping[str, Any],
    trace_params: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble trace metadata after a public task has already bound the query."""

    rows = _row_records(dataset)
    row_labels = [str(row["label"]) for row in rows]
    query_params = {
        "query_id": str(dataset.branch_id),
        "query_id_probabilities": dict(dataset.branch_probabilities),
        "answer_value": dataset.query.answer,
        "annotation_row_ids": [str(row_id) for row_id in dataset.query.annotation_row_ids],
        **dict(dataset.query.params),
        **dict(trace_params),
    }
    return {
        "scene_ir": {
            "scene_kind": "population_pyramid_chart",
            "entities": [dict(entity) for entity in rendered.rendered_scene.entities],
            "relations": {
                "query_id": str(dataset.branch_id),
                "left_series_label": str(dataset.left_series_label),
                "right_series_label": str(dataset.right_series_label),
                "row_labels": list(row_labels),
            },
        },
        "query_spec": build_prompt_query_spec(
            prompt_artifacts=prompt_artifacts,
            query_id=str(dataset.branch_id),
            params=dict(query_params),
        ),
        "render_spec": {
            "canvas_width": int(rendered.image.size[0]),
            "canvas_height": int(rendered.image.size[1]),
            "coord_space": "pixel",
            "left_series_label": str(dataset.left_series_label),
            "right_series_label": str(dataset.right_series_label),
            "left_color_rgb": [int(channel) for channel in dataset.left_color_rgb],
            "right_color_rgb": [int(channel) for channel in dataset.right_color_rgb],
            "plot_bbox_px": list(rendered.rendered_scene.plot_bbox_px),
            "render_meta": dict(rendered.rendered_scene.render_meta),
            "information_scene_style": dict(rendered.rendered_scene.render_meta.get("information_style", {})),
            "post_image_noise": dict(rendered.post_noise_meta),
        },
        "render_map": {
            "image_id": "img0",
            "plot_bbox_px": list(rendered.rendered_scene.plot_bbox_px),
            "panel_bbox_px": list(rendered.rendered_scene.render_meta.get("panel_bbox_px", [])),
            "row_bar_bboxes_px": dict(rendered.rendered_scene.row_bar_bboxes_px),
            "left_bar_bboxes_px": dict(rendered.rendered_scene.left_bar_bboxes_px),
            "right_bar_bboxes_px": dict(rendered.rendered_scene.right_bar_bboxes_px),
            "row_label_bboxes_px": dict(rendered.rendered_scene.row_label_bboxes_px),
        },
        "execution_trace": {
            "query_id": str(dataset.branch_id),
            "question_format": "population_pyramid",
            "left_series_label": str(dataset.left_series_label),
            "right_series_label": str(dataset.right_series_label),
            "row_count": int(len(dataset.rows)),
            "row_labels": list(row_labels),
            "rows": list(rows),
            "answer_value": dataset.query.answer,
            "answer_type": str(dataset.query.answer_type),
            "annotation_type": str(dataset.query.annotation_type),
            "annotation_row_ids": [str(row_id) for row_id in dataset.query.annotation_row_ids],
            **dict(dataset.query.params),
            **dict(trace_params),
        },
        "witness_symbolic": {
            "type": "population_pyramid_witness",
            "answer_value": dataset.query.answer,
            "annotation_type": str(dataset.query.annotation_type),
            "annotation_row_ids": [str(row_id) for row_id in dataset.query.annotation_row_ids],
        },
        "projected_annotation": dict(annotation["projected_annotation"]),
        "post_image_noise": dict(rendered.post_noise_meta),
    }


def materialize_population_pyramid_plan(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    plan: PopulationPyramidTaskPlan,
) -> TaskOutput:
    rendered = render_population_pyramid_dataset(dataset=plan.dataset, params=dict(params), instance_seed=int(instance_seed))
    annotation = annotation_payload(dataset=plan.dataset, rendered=rendered)
    prompt_artifacts = build_prompt_artifacts(
        prompt_query_key=str(plan.prompt_query_key),
        dynamic_slot_values=dynamic_slots(dataset=plan.dataset),
        instance_seed=int(instance_seed),
    )
    trace_payload = _build_trace_payload(
        dataset=plan.dataset,
        rendered=rendered,
        prompt_artifacts=prompt_artifacts,
        annotation=annotation,
        trace_params=dict(plan.trace_params),
    )
    return TaskOutput(
        prompt=str(prompt_artifacts.prompt),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
        answer_gt=TypedValue(type=str(plan.dataset.query.answer_type), value=plan.dataset.query.answer),
        annotation_gt=TypedValue(type=str(annotation["type"]), value=annotation["value"]),
        image=rendered.image,
        image_id="img0",
        trace_payload=trace_payload,
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
        query_id=str(plan.dataset.branch_id),
    )


def run_population_pyramid_task(task: Any, instance_seed: int, params: dict[str, Any], max_attempts: int) -> TaskOutput:
    selected, probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=dict(params),
        supported_query_ids=tuple(str(value) for value in task.supported_query_ids),
        default_query_id=str(task.default_query_id),
        task_id=str(task.task_id),
    )
    last_error: Exception | None = None
    for attempt in range(max(1, int(max_attempts))):
        attempt_seed = population_pyramid_attempt_seed(int(instance_seed), str(task.task_id), int(attempt))
        try:
            plan = task._build_plan(
                dict(task_params),
                int(attempt_seed),
                str(selected),
                dict(probabilities),
            )
            return materialize_population_pyramid_plan(
                params=dict(task_params),
                instance_seed=int(attempt_seed),
                plan=plan,
            )
        except ValueError as exc:
            last_error = exc
    raise RuntimeError(f"failed to generate {task.task_id}: {last_error}")
