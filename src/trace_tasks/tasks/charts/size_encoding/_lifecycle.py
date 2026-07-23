"""Private materialization lifecycle for size-encoded chart tasks."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from trace_tasks.core.seed import hash64
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.charts.size_encoding.shared.defaults import (
    SCENE_VARIANT_LOADS,
    resolve_int,
)
from trace_tasks.tasks.charts.size_encoding.shared.prompts import (
    ANNOTATION_HINT_BY_KIND,
    BUNDLE_ID,
    JSON_EXAMPLE_BY_KIND,
    object_description,
    render_prompt_artifacts,
)
from trace_tasks.tasks.charts.size_encoding.shared.rendering import render_size_encoding_scene
from trace_tasks.tasks.charts.size_encoding.shared.state import (
    SCENE_ID,
    SizeEncodingDataset,
    SizeEncodingSelection,
)
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec
from trace_tasks.tasks.shared.text_rendering import temporary_default_font_family
from trace_tasks.tasks.charts.shared.visual_defaults import chart_font_asset_metadata, sample_chart_font_family


@dataclass(frozen=True)
class SizeEncodingObjectivePlan:
    """Task-owned semantic plan consumed by neutral scene materialization."""

    dataset: SizeEncodingDataset
    selection: SizeEncodingSelection
    params: Mapping[str, Any]
    scene_variant: str
    scene_variant_probabilities: Mapping[str, float]
    prompt_key: str
    annotation_kind: str
    question_format: str
    program_code: str
    reasoning_load: float
    answer_type: str = "string"
    answer_hint: str = 'set "answer" to the exact visible item, category, or panel label as a string'


AnnotationBinder = Callable[[SizeEncodingObjectivePlan, Any], tuple[str, Any, dict[str, Any]]]
PlanBuilder = Callable[[Mapping[str, Any], int, str, Mapping[str, float]], SizeEncodingObjectivePlan]


def package_size_encoding_plan(
    *,
    dataset: SizeEncodingDataset,
    selection: SizeEncodingSelection,
    params: Mapping[str, Any],
    scene_variant: str,
    scene_variant_probabilities: Mapping[str, float],
    prompt_key: str,
    annotation_kind: str,
    question_format: str,
    program_code: str,
    reasoning_load: float,
    answer_type: str = "string",
    answer_hint: str = 'set "answer" to the exact visible item, category, or panel label as a string',
) -> SizeEncodingObjectivePlan:
    """Bind task-owned semantic fields to the neutral size-encoding lifecycle."""

    return SizeEncodingObjectivePlan(
        dataset=dataset,
        selection=selection,
        params=dict(params),
        scene_variant=str(scene_variant),
        scene_variant_probabilities=dict(scene_variant_probabilities),
        prompt_key=str(prompt_key),
        annotation_kind=str(annotation_kind),
        question_format=str(question_format),
        program_code=str(program_code),
        reasoning_load=float(reasoning_load),
        answer_type=str(answer_type),
        answer_hint=str(answer_hint),
    )


def _prompt_slots(plan: SizeEncodingObjectivePlan) -> dict[str, Any]:
    extremum_phrase = "largest" if str(plan.selection.direction) == "largest" else "smallest"
    return {
        "object_description": object_description(str(plan.scene_variant)),
        "category_label": str(plan.selection.category_label),
        "panel_label": str(plan.selection.panel_label),
        "reference_label": str(plan.selection.reference_label),
        "extremum_phrase": str(extremum_phrase),
        "json_output_contract": 'Use a valid JSON object with keys "annotation" and "answer" in that order for the final answer.',
        "json_output_contract_answer_only": 'Use a valid JSON object with key "answer" for the final answer.',
        "annotation_hint": str(ANNOTATION_HINT_BY_KIND[str(plan.annotation_kind)]),
        "answer_hint": str(plan.answer_hint),
        "json_example": str(JSON_EXAMPLE_BY_KIND[str(plan.annotation_kind)]),
        "json_example_answer_only": '{"answer":2}' if str(plan.answer_type) == "integer" else '{"answer":"Aero"}',
    }


def run_size_encoding_lifecycle(
    *,
    task: Any,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    default_query_id: str,
    build_plan: PlanBuilder,
    bind_annotation: AnnotationBinder,
) -> TaskOutput:
    """Materialize a task-owned size-encoding plan into one TaskOutput.

    The lifecycle owns deterministic rendering, prompt metadata plumbing, and
    trace packaging only; objective branching and annotation binding come from
    callbacks supplied by the public task file.
    """

    selected_query_id, query_probabilities, task_params = select_task_query_id(
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
            plan = build_plan(dict(attempt_params), int(attempt_seed), str(selected_query_id), dict(query_probabilities))
            chart_font_family = sample_chart_font_family(
                instance_seed=int(attempt_seed),
                namespace=f"{task.task_id}.chart_font",
                params=attempt_params,
            )
            with temporary_default_font_family(str(chart_font_family)):
                rendered = render_size_encoding_scene(
                    plan.dataset,
                    scene_variant=str(plan.scene_variant),
                    params=attempt_params,
                    instance_seed=int(attempt_seed),
                )
            annotation_type, annotation_value, projected_annotation = bind_annotation(plan, rendered)
            if not annotation_value:
                raise RuntimeError("empty size-encoding annotation")
            answer_value: Any = int(plan.selection.answer) if str(plan.answer_type) == "integer" else str(plan.selection.answer)
            prompt_artifacts = render_prompt_artifacts(
                prompt_key=str(plan.prompt_key),
                dynamic_slot_values=_prompt_slots(plan),
                instance_seed=int(attempt_seed),
            )

            items_by_id = {str(item.item_id): item for item in plan.dataset.items}
            values_by_label = {str(item.label): int(item.value) for item in plan.dataset.items}
            category_by_label = {str(item.label): str(item.category) for item in plan.dataset.items}
            panel_by_label = {str(item.label): str(item.panel) for item in plan.dataset.items}
            annotation_labels = [
                str(items_by_id[item_id].label)
                for item_id in plan.selection.annotation_item_ids
                if item_id in items_by_id
            ]
            query_params = {
                "query_id": str(selected_query_id),
                "scene_variant": str(plan.scene_variant),
                "query_id_probabilities": dict(query_probabilities),
                "scene_variant_probabilities": dict(plan.scene_variant_probabilities),
                "item_count": int(len(plan.dataset.items)),
                "category_count": int(len(plan.dataset.categories)),
                "panel_count": int(len(plan.dataset.panels)),
                "category_label": str(plan.selection.category_label),
                "panel_label": str(plan.selection.panel_label),
                "reference_label": str(plan.selection.reference_label),
                "extremum_direction": str(plan.selection.direction),
                "question_format": str(plan.question_format),
                "program_code": str(plan.program_code),
            }
            prompt_query_spec = build_prompt_query_spec(
                prompt_artifacts=prompt_artifacts,
                query_id=str(selected_query_id),
                params=query_params,
            )
            trace_payload = {
                "scene_ir": {
                    "scene_kind": f"chart_size_encoding_{str(plan.scene_variant)}",
                    "entities": [dict(entity) for entity in rendered.entities],
                    "relations": {
                        "query_id": str(selected_query_id),
                        "scene_variant": str(plan.scene_variant),
                        "extremum_direction": str(plan.selection.direction),
                        "answer_label": str(plan.selection.answer),
                        "annotation_labels": list(annotation_labels),
                        "category_label": str(plan.selection.category_label),
                        "panel_label": str(plan.selection.panel_label),
                        "reference_label": str(plan.selection.reference_label),
                    },
                },
                "query_spec": prompt_query_spec,
                "render_spec": {
                    "canvas_width": resolve_int(attempt_params, "canvas_width", 1320),
                    "canvas_height": resolve_int(attempt_params, "canvas_height", 900),
                    "coord_space": "pixel",
                    "scene_variant": str(plan.scene_variant),
                    "plot_bbox_px": list(rendered.plot_bbox_px),
                    "font_assets": chart_font_asset_metadata(str(chart_font_family)),
                    **dict(rendered.render_meta),
                },
                "render_map": {
                    "image_id": "img0",
                    "plot_bbox_px": list(rendered.plot_bbox_px),
                    "item_bboxes_px": dict(rendered.item_bboxes),
                    "panel_title_bboxes_px": dict(rendered.panel_title_bboxes),
                    "category_legend_bboxes_px": dict(rendered.category_legend_bboxes),
                },
                "execution_trace": {
                    "query_id": str(selected_query_id),
                    "scene_variant": str(plan.scene_variant),
                    "extremum_direction": str(plan.selection.direction),
                    "answer_label": str(plan.selection.answer),
                    "answer_value": answer_value,
                    "answer_value_hidden": int(values_by_label[str(plan.selection.answer)])
                    if str(plan.selection.answer) in values_by_label
                    else None,
                    "annotation_labels": list(annotation_labels),
                    "item_count": int(len(plan.dataset.items)),
                    "category_count": int(len(plan.dataset.categories)),
                    "panel_count": int(len(plan.dataset.panels)),
                    "categories": list(plan.dataset.categories),
                    "panels": list(plan.dataset.panels),
                    "values_by_label": dict(values_by_label),
                    "category_by_label": dict(category_by_label),
                    "panel_by_label": dict(panel_by_label),
                    "query_id_probabilities": dict(query_probabilities),
                    "scene_variant_probabilities": dict(plan.scene_variant_probabilities),
                    "question_format": str(plan.question_format),
                    "program_code": str(plan.program_code),
                    "reasoning_load": float(plan.reasoning_load)
                    + float(SCENE_VARIANT_LOADS.get(str(plan.scene_variant), 0.0)),
                    **dict(plan.dataset.trace),
                    **dict(plan.selection.trace),
                },
                "witness_symbolic": {
                    "type": "object_set",
                    "labels": list(annotation_labels),
                    "answer": answer_value,
                },
                "projected_annotation": {
                    **dict(projected_annotation),
                    "annotation_item_ids": list(plan.selection.annotation_item_ids),
                },
            }
            return TaskOutput(
                prompt=str(prompt_artifacts.prompt),
                prompt_variants=dict(prompt_artifacts.prompt_variants),
                answer_gt=TypedValue(type=str(plan.answer_type), value=answer_value),
                annotation_gt=TypedValue(type=str(annotation_type), value=annotation_value),
                image=rendered.image,
                image_id="img0",
                trace_payload=trace_payload,
                task_versions=default_task_versions(),
                scene_id=SCENE_ID,
                query_id=str(selected_query_id),
            )
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"failed to generate {task.task_id}: {last_error}") from last_error
