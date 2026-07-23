"""Private lifecycle for rendering and packaging style-legend task plans."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from trace_tasks.core.seed import hash64
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.charts.shared.visual_defaults import chart_font_asset_metadata, sample_chart_font_family
from trace_tasks.tasks.charts.style_legend.shared.annotations import (
    center_points_for_markers,
    projected_point_payload,
    projected_point_set_payload,
)
from trace_tasks.tasks.charts.style_legend.shared.prompts import (
    ANSWER_ONLY_EXAMPLES,
    JSON_EXAMPLES,
    POINT_HINT,
    render_prompt_artifacts,
)
from trace_tasks.tasks.charts.style_legend.shared.rendering import render_dataset
from trace_tasks.tasks.charts.style_legend.shared.sampling import style_support_trace
from trace_tasks.tasks.charts.style_legend.shared.state import SCENE_ID, StyleLegendDataset
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec
from trace_tasks.tasks.shared.text_rendering import temporary_default_font_family


@dataclass(frozen=True)
class StyleLegendObjectivePlan:
    """Task-owned objective plan; lifecycle only renders and packages it."""

    dataset: StyleLegendDataset
    params: Mapping[str, Any]
    answer_value: Any
    answer_type: str
    annotation_type: str
    annotation_marker_ids: tuple[str, ...]
    prompt_key: str
    prompt_slots: Mapping[str, Any]
    answer_hint: str
    annotation_hint: str
    json_example: str
    json_example_answer_only: str
    program_code: str
    reasoning_load: float
    objective_trace: Mapping[str, Any]


PlanBuilder = Callable[[Mapping[str, Any], int, str, Mapping[str, float]], StyleLegendObjectivePlan]


def package_style_legend_plan(
    *,
    dataset: StyleLegendDataset,
    params: Mapping[str, Any],
    answer_value: Any,
    answer_type: str,
    annotation_type: str,
    annotation_marker_ids: Sequence[str],
    prompt_key: str,
    prompt_slots: Mapping[str, Any],
    answer_hint: str,
    annotation_hint: str,
    json_example: str,
    json_example_answer_only: str,
    program_code: str,
    reasoning_load: float,
    objective_trace: Mapping[str, Any],
) -> StyleLegendObjectivePlan:
    return StyleLegendObjectivePlan(
        dataset=dataset,
        params=dict(params),
        answer_value=answer_value,
        answer_type=str(answer_type),
        annotation_type=str(annotation_type),
        annotation_marker_ids=tuple(str(value) for value in annotation_marker_ids),
        prompt_key=str(prompt_key),
        prompt_slots=dict(prompt_slots),
        answer_hint=str(answer_hint),
        annotation_hint=str(annotation_hint),
        json_example=str(json_example),
        json_example_answer_only=str(json_example_answer_only),
        program_code=str(program_code),
        reasoning_load=float(reasoning_load),
        objective_trace=dict(objective_trace),
    )


def package_point_label_plan(
    *,
    dataset: StyleLegendDataset,
    params: Mapping[str, Any],
    answer_value: str,
    annotation_marker_id: str,
    prompt_key: str,
    prompt_slots: Mapping[str, Any],
    answer_hint: str,
    json_example_key: str,
    program_code: str,
    reasoning_load: float,
    objective_trace: Mapping[str, Any],
) -> StyleLegendObjectivePlan:
    """Package a scalar string-label objective witnessed by one plotted marker."""

    return package_style_legend_plan(
        dataset=dataset,
        params=params,
        answer_value=str(answer_value),
        answer_type="string",
        annotation_type="point",
        annotation_marker_ids=(str(annotation_marker_id),),
        prompt_key=str(prompt_key),
        prompt_slots=dict(prompt_slots),
        answer_hint=str(answer_hint),
        annotation_hint=POINT_HINT,
        json_example=str(JSON_EXAMPLES[str(json_example_key)]),
        json_example_answer_only=str(ANSWER_ONLY_EXAMPLES[str(json_example_key)]),
        program_code=str(program_code),
        reasoning_load=float(reasoning_load),
        objective_trace=dict(objective_trace),
    )


def _annotation_value_and_projection(plan: StyleLegendObjectivePlan, rendered: Any) -> tuple[Any, dict[str, Any]]:
    points = center_points_for_markers(rendered, plan.annotation_marker_ids)
    if str(plan.annotation_type) == "point":
        if len(points) != 1:
            raise ValueError("scalar point annotation requires exactly one marker")
        return list(points[0]), projected_point_payload(points[0])
    if str(plan.annotation_type) == "point_set":
        return [list(point) for point in points], projected_point_set_payload(points)
    raise ValueError(f"unsupported style-legend annotation type: {plan.annotation_type}")


def run_style_legend_lifecycle(
    *,
    task: Any,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    default_query_id: str,
    build_plan: PlanBuilder,
) -> TaskOutput:
    """Select a public branch, render the task-owned plan, and package output."""

    selected_query_id, probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=dict(params),
        supported_query_ids=tuple(task.supported_query_ids),
        default_query_id=str(default_query_id),
        task_id=str(task.task_id),
    )
    last_error: Exception | None = None
    for attempt_index in range(max(1, int(max_attempts))):
        attempt_seed = int(instance_seed) if attempt_index == 0 else int(hash64(int(instance_seed), str(task.task_id), attempt_index))
        attempt_params = {**dict(task_params), "_attempt_index": int(attempt_index)}
        try:
            plan = build_plan(dict(attempt_params), int(attempt_seed), str(selected_query_id), dict(probabilities))
            chart_font_family = sample_chart_font_family(
                instance_seed=int(attempt_seed),
                namespace=f"{task.task_id}.chart_font",
                params=attempt_params,
            )
            with temporary_default_font_family(str(chart_font_family)):
                rendered = render_dataset(
                    plan.dataset,
                    params={**dict(attempt_params), "_render_style_seed": int(attempt_seed)},
                    instance_seed=int(attempt_seed),
                    chart_font_family=str(chart_font_family),
                )
            annotation_value, projected_annotation = _annotation_value_and_projection(plan, rendered)
            annotation_gt = TypedValue(type=str(plan.annotation_type), value=annotation_value)
            prompt_artifacts = render_prompt_artifacts(
                prompt_key=str(plan.prompt_key),
                answer_hint=str(plan.answer_hint),
                annotation_hint=str(plan.annotation_hint),
                json_example=str(plan.json_example),
                json_example_answer_only=str(plan.json_example_answer_only),
                dynamic_slot_values=plan.prompt_slots,
                instance_seed=int(attempt_seed),
            )
            prompt_query_spec = build_prompt_query_spec(
                prompt_artifacts=prompt_artifacts,
                query_id=str(selected_query_id),
                params={
                    "scene_id": SCENE_ID,
                    "program_code": str(plan.program_code),
                    "reasoning_load": float(plan.reasoning_load),
                    "query_id_probabilities": dict(probabilities),
                    "style_palette_mode": str(plan.dataset.palette_mode),
                    "style_palette_mode_probabilities": dict(plan.dataset.palette_mode_probabilities),
                    "legend_position": str(plan.dataset.legend_position),
                    "legend_position_probabilities": dict(plan.dataset.legend_position_probabilities),
                    **dict(plan.objective_trace),
                },
            )
            execution_trace = {
                "query_id": str(selected_query_id),
                "scene_id": SCENE_ID,
                "question_format": "style_legend",
                "answer_value": plan.answer_value,
                "answer_type": str(plan.answer_type),
                "annotation_type": str(plan.annotation_type),
                "annotation_marker_ids": list(plan.annotation_marker_ids),
                "x_count": int(len(plan.dataset.x_labels)),
                "series_count": int(len(plan.dataset.series)),
                "x_labels": list(plan.dataset.x_labels),
                "x_label_meta": dict(plan.dataset.x_label_meta),
                "series_label_meta": dict(plan.dataset.series_label_meta),
                "series": style_support_trace(plan.dataset.series),
                "target_x_index": int(plan.dataset.target_x_index),
                "target_x_label": str(plan.dataset.x_labels[int(plan.dataset.target_x_index)]),
                "threshold_value": plan.dataset.threshold_value,
                "pair_series_ids": list(plan.dataset.pair_series_ids),
                "style_palette_mode": str(plan.dataset.palette_mode),
                "style_palette_mode_probabilities": dict(plan.dataset.palette_mode_probabilities),
                "legend_position": str(plan.dataset.legend_position),
                "legend_position_probabilities": dict(plan.dataset.legend_position_probabilities),
                **dict(plan.objective_trace),
            }
            trace_payload = {
                "scene_ir": {
                    "scene_kind": "chart_style_legend",
                    "entities": [dict(entity) for entity in rendered.entities],
                    "relations": {
                        "query_id": str(selected_query_id),
                        "scene_id": SCENE_ID,
                        "target_x_index": int(plan.dataset.target_x_index),
                        "annotation_marker_ids": list(plan.annotation_marker_ids),
                    },
                },
                "query_spec": prompt_query_spec,
                "render_spec": {
                    "canvas_width": int(rendered.image.size[0]),
                    "canvas_height": int(rendered.image.size[1]),
                    "coord_space": "pixel",
                    "scene_id": SCENE_ID,
                    "plot_bbox_px": list(rendered.plot_bbox_px),
                    "legend_bbox_px": list(rendered.legend_bbox_px),
                    "font_assets": chart_font_asset_metadata(str(chart_font_family)),
                    **dict(rendered.render_meta),
                },
                "render_map": {
                    "image_id": "img0",
                    "plot_bbox_px": list(rendered.plot_bbox_px),
                    "legend_bbox_px": list(rendered.legend_bbox_px),
                    "legend_item_bboxes_px": dict(rendered.legend_item_bboxes_px),
                    "point_bboxes_px": dict(rendered.point_bboxes_px),
                    "threshold_bbox_px": rendered.threshold_bbox_px,
                    "context_protected_bboxes_px": {
                        "plot": list(rendered.plot_bbox_px),
                        **({"legend": list(rendered.legend_bbox_px)} if rendered.legend_bbox_px else {}),
                    },
                },
                "execution_trace": dict(execution_trace),
                "witness_symbolic": {
                    "type": "style_legend_marker_witness",
                    "marker_ids": list(plan.annotation_marker_ids),
                    "answer": plan.answer_value,
                },
                "projected_annotation": projected_annotation,
            }
            return TaskOutput(
                prompt=str(prompt_artifacts.prompt),
                prompt_variants=dict(prompt_artifacts.prompt_variants),
                answer_gt=TypedValue(type=str(plan.answer_type), value=plan.answer_value),
                annotation_gt=annotation_gt,
                image=rendered.image,
                image_id="img0",
                trace_payload=trace_payload,
                task_versions=default_task_versions(),
                query_id=str(selected_query_id),
            )
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"failed to generate {task.task_id}: {last_error}")
