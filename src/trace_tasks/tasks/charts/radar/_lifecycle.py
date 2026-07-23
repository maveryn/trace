"""Neutral lifecycle helpers for radar public tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ....core.seed import hash64
from ....core.types import TypedValue
from ...base import TaskOutput
from ...shared.fixed_query import select_task_query_id
from ...shared.output_metadata import default_task_versions
from ...shared.prompt_variants import PromptTraceArtifacts, build_prompt_query_spec

from .shared.annotations import annotation_payload
from .shared.defaults import font_assets_payload
from .shared.prompts import build_prompt_artifacts, dynamic_slots
from .shared.rendering import render_radar_dataset
from .shared.state import RadarDataset, RadarPanel, RadarQuery, RadarRenderResult, SCENE_ID


@dataclass(frozen=True)
class RadarTaskPlan:
    dataset: RadarDataset
    prompt_query_key: str
    trace_params: dict[str, Any]


def radar_attempt_seed(instance_seed: int, unique_key: str, attempt: int) -> int:
    return int(instance_seed) if int(attempt) == 0 else int(hash64(int(instance_seed), str(unique_key), int(attempt)))


def build_radar_plan(
    *,
    dataset: RadarDataset,
    prompt_query_key: str,
    trace_params: Mapping[str, Any] | None = None,
) -> RadarTaskPlan:
    return RadarTaskPlan(
        dataset=dataset,
        prompt_query_key=str(prompt_query_key),
        trace_params=dict(trace_params or {}),
    )


def build_radar_dataset_from_components(
    *,
    metrics: tuple[str, ...],
    panels: tuple[RadarPanel, ...],
    scene_variant: str,
    branch_id: str,
    branch_probabilities: Mapping[str, float],
    highlight_metric_label: str,
    answer: int | str,
    answer_type: str,
    annotation_type: str,
    metric_label: str,
    panel_label: str,
    profile_a_label: str,
    profile_b_label: str,
    threshold_value: int,
    minimum_metric_count: int,
    annotation_point_ids: tuple[str, ...] = (),
    annotation_panel_labels: tuple[str, ...] = (),
    annotation_point_id_pairs: tuple[tuple[str, str], ...] = (),
    params: Mapping[str, Any] | None = None,
) -> RadarDataset:
    """Create the shared dataset object from task-owned answer and annotation fields."""

    return RadarDataset(
        metrics=tuple(str(metric) for metric in metrics),
        panels=tuple(panels),
        scene_variant=str(scene_variant),
        branch_id=str(branch_id),
        branch_probabilities={str(key): float(value) for key, value in branch_probabilities.items()},
        highlight_metric_label=str(highlight_metric_label),
        query=RadarQuery(
            branch_id=str(branch_id),
            answer=answer,
            answer_type=str(answer_type),
            annotation_type=str(annotation_type),
            metric_label=str(metric_label),
            panel_label=str(panel_label),
            profile_a_label=str(profile_a_label),
            profile_b_label=str(profile_b_label),
            threshold_value=int(threshold_value),
            minimum_metric_count=int(minimum_metric_count),
            annotation_point_ids=tuple(str(point_id) for point_id in annotation_point_ids),
            annotation_panel_labels=tuple(str(label) for label in annotation_panel_labels),
            annotation_point_id_pairs=tuple(
                (str(start_id), str(end_id))
                for start_id, end_id in annotation_point_id_pairs
            ),
            params=dict(params or {}),
        ),
    )


def _values_by_panel_profile(dataset: RadarDataset) -> dict[str, dict[str, dict[str, int]]]:
    return {
        str(panel.panel_label): {
            str(profile.profile_label): dict(profile.values)
            for profile in panel.profiles
        }
        for panel in dataset.panels
    }


def _build_trace_payload(
    *,
    dataset: RadarDataset,
    rendered: RadarRenderResult,
    prompt_artifacts: PromptTraceArtifacts,
    annotation: Mapping[str, Any],
    trace_params: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble trace metadata after a public task has bound the objective."""

    rendered_scene = rendered.rendered_scene
    branch_id = str(dataset.branch_id)
    query_params = {
        "query_id": str(branch_id),
        "query_id_probabilities": dict(dataset.branch_probabilities),
        "scene_variant": str(dataset.scene_variant),
        "metric_count": int(len(dataset.metrics)),
        "panel_count": int(len(dataset.panels)),
        "question_format": "radar_profile_query",
        "answer": dataset.query.answer,
        "answer_type": str(dataset.query.answer_type),
        "annotation_type": str(annotation["type"]),
        "annotation_point_ids": list(dataset.query.annotation_point_ids),
        "annotation_panel_labels": list(dataset.query.annotation_panel_labels),
        "annotation_point_id_pairs": [
            [str(start_id), str(end_id)]
            for start_id, end_id in dataset.query.annotation_point_id_pairs
        ],
        **dict(dataset.query.params),
        **dict(trace_params),
    }
    return {
        "scene_ir": {
            "scene_kind": f"chart_radar_{str(dataset.scene_variant)}",
            "entities": [dict(entity) for entity in rendered_scene.entities],
            "relations": {
                "query_id": str(branch_id),
                "scene_variant": str(dataset.scene_variant),
                "answer": dataset.query.answer,
                "annotation_type": str(annotation["type"]),
                "annotation_count": len(annotation["value"]),
            },
        },
        "query_spec": build_prompt_query_spec(
            prompt_artifacts=prompt_artifacts,
            query_id=str(branch_id),
            params=dict(query_params),
        ),
        "render_spec": {
            "canvas_width": int(rendered.image.size[0]),
            "canvas_height": int(rendered.image.size[1]),
            "coord_space": "pixel",
            "scene_variant": str(dataset.scene_variant),
            "plot_bbox_px": list(rendered_scene.plot_bbox_px),
            "font_assets": font_assets_payload(chart_font_family=rendered.chart_font_family),
            **dict(rendered_scene.render_meta),
        },
        "render_map": {
            "image_id": "img0",
            "plot_bbox_px": list(rendered_scene.plot_bbox_px),
            "point_bboxes_px": dict(rendered_scene.point_bboxes),
            "panel_bboxes_px": dict(rendered_scene.panel_bboxes),
            "panel_title_bboxes_px": dict(rendered_scene.panel_title_bboxes),
            "legend_bboxes_px": dict(rendered_scene.legend_bboxes),
        },
        "execution_trace": {
            "query_id": str(branch_id),
            "scene_variant": str(dataset.scene_variant),
            "answer": dataset.query.answer,
            "answer_type": str(dataset.query.answer_type),
            "metrics": list(dataset.metrics),
            "metric_count": int(len(dataset.metrics)),
            "panel_labels": [str(panel.panel_label) for panel in dataset.panels if str(panel.panel_label)],
            "panel_count": int(len(dataset.panels)),
            "values_by_panel_profile": _values_by_panel_profile(dataset),
            "question_format": "radar_profile_query",
            **dict(query_params),
        },
        "witness_symbolic": {
            "type": "radar_chart_witness",
            "answer": dataset.query.answer,
            "annotation_type": str(annotation["type"]),
            "annotation_point_ids": list(dataset.query.annotation_point_ids),
            "annotation_panel_labels": list(dataset.query.annotation_panel_labels),
            "annotation_point_id_pairs": [
                [str(start_id), str(end_id)]
                for start_id, end_id in dataset.query.annotation_point_id_pairs
            ],
        },
        "projected_annotation": dict(annotation["projected_annotation"]),
    }


def materialize_radar_plan(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    plan: RadarTaskPlan,
) -> TaskOutput:
    rendered = render_radar_dataset(dataset=plan.dataset, params=dict(params), instance_seed=int(instance_seed))
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


def run_radar_task(task: Any, instance_seed: int, params: dict[str, Any], max_attempts: int) -> TaskOutput:
    selected, probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=dict(params),
        supported_query_ids=tuple(str(value) for value in task.supported_query_ids),
        default_query_id=str(task.default_query_id),
        task_id=str(task.task_id),
    )
    last_error: Exception | None = None
    for attempt in range(max(1, int(max_attempts))):
        attempt_seed = radar_attempt_seed(int(instance_seed), str(task.task_id), int(attempt))
        try:
            plan = task._build_plan(
                dict(task_params),
                int(attempt_seed),
                str(selected),
                dict(probabilities),
            )
            return materialize_radar_plan(
                params=dict(task_params),
                instance_seed=int(attempt_seed),
                plan=plan,
            )
        except ValueError as exc:
            last_error = exc
    raise RuntimeError(f"failed to generate {task.task_id}: {last_error}")
