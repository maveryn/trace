"""Private materialization lifecycle for composition-panel chart tasks."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from trace_tasks.core.seed import hash64
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.charts.shared.visual_defaults import chart_font_asset_metadata, sample_chart_font_family
from trace_tasks.tasks.charts.composition_panels.shared.annotations import (
    bbox_for_panel_role,
    bbox_payload,
    bbox_set_for_panel_roles,
    bbox_set_payload,
    point_map_for_roles,
    point_map_payload,
)
from trace_tasks.tasks.charts.composition_panels.shared.defaults import SCENE_VARIANT_LOADS
from trace_tasks.tasks.charts.composition_panels.shared.prompts import prompt_slots, render_prompt_artifacts
from trace_tasks.tasks.charts.composition_panels.shared.rendering import panel_trace_payload, render_composition_panels
from trace_tasks.tasks.charts.composition_panels.shared.sampling import counts_for_panel
from trace_tasks.tasks.charts.composition_panels.shared.state import (
    SCENE_ID,
    CompositionPanelsDataset,
    CompositionPanelsSelection,
)
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec
from trace_tasks.tasks.shared.text_rendering import temporary_default_font_family


@dataclass(frozen=True)
class CompositionPanelsObjectivePlan:
    """Task-owned plan; lifecycle only renders and packages this fixed contract."""

    dataset: CompositionPanelsDataset
    selection: CompositionPanelsSelection
    params: Mapping[str, Any]
    scene_variant: str
    scene_variant_probabilities: Mapping[str, float]
    prompt_key: str
    annotation_hint_template: str
    json_example: str
    json_example_answer_only: str
    program_code: str
    reasoning_load: float


PlanBuilder = Callable[[Mapping[str, Any], int, str, Mapping[str, float]], CompositionPanelsObjectivePlan]


def _typed_answer(value: Any) -> TypedValue:
    if isinstance(value, bool):
        return TypedValue(type="boolean", value=bool(value))
    if isinstance(value, int):
        return TypedValue(type="integer", value=int(value))
    return TypedValue(type="string", value=str(value))


def package_composition_panels_plan(
    *,
    dataset: CompositionPanelsDataset,
    selection: CompositionPanelsSelection,
    params: Mapping[str, Any],
    scene_variant: str,
    scene_variant_probabilities: Mapping[str, float],
    prompt_key: str,
    annotation_hint_template: str,
    json_example: str,
    json_example_answer_only: str,
    program_code: str,
    reasoning_load: float,
) -> CompositionPanelsObjectivePlan:
    return CompositionPanelsObjectivePlan(
        dataset=dataset,
        selection=selection,
        params=dict(params),
        scene_variant=str(scene_variant),
        scene_variant_probabilities=dict(scene_variant_probabilities),
        prompt_key=str(prompt_key),
        annotation_hint_template=str(annotation_hint_template),
        json_example=str(json_example),
        json_example_answer_only=str(json_example_answer_only),
        program_code=str(program_code),
        reasoning_load=float(reasoning_load),
    )


def run_composition_panels_lifecycle(
    *,
    task: Any,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    default_query_id: str,
    build_plan: PlanBuilder,
) -> TaskOutput:
    """Render and package a task-owned composition-panel plan.

    Key invariant: this lifecycle never chooses the public objective. The task
    file supplies the sampled objective plan, annotation roles, prompt key, and
    program code; this function only applies shared rendering/projection/output
    plumbing.
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
                rendered = render_composition_panels(
                    plan.dataset,
                    scene_variant=str(plan.scene_variant),
                    params=attempt_params,
                    instance_seed=int(attempt_seed),
                )

            annotation_type = str(plan.selection.annotation_type)
            annotation_points: dict[str, list[float]] = {}
            annotation_bbox: list[float] | None = None
            annotation_bboxes: list[list[float]] = []
            if annotation_type == "point_map":
                annotation_points = point_map_for_roles(rendered, plan.selection.annotation_roles)
                projected_annotation = point_map_payload(annotation_points)
                annotation_value: Any = dict(annotation_points)
            elif annotation_type == "bbox":
                if len(plan.selection.annotation_roles) != 1:
                    raise ValueError("bbox annotation requires exactly one composition-panel annotation role")
                annotation_bbox = bbox_for_panel_role(rendered, plan.selection.annotation_roles[0])
                projected_annotation = bbox_payload(annotation_bbox)
                annotation_value = list(annotation_bbox)
            elif annotation_type == "bbox_set":
                annotation_bboxes = bbox_set_for_panel_roles(rendered, plan.selection.annotation_roles)
                projected_annotation = bbox_set_payload(annotation_bboxes)
                annotation_value = [list(bbox) for bbox in annotation_bboxes]
            else:
                raise ValueError(f"unsupported composition-panels annotation type: {annotation_type}")
            annotation_bbox_count = 1 if annotation_bbox is not None else len(annotation_bboxes)
            prompt_artifacts = render_prompt_artifacts(
                prompt_key=str(plan.prompt_key),
                dynamic_slot_values=prompt_slots(
                    prompt_key=str(plan.prompt_key),
                    scene_variant=str(plan.scene_variant),
                    dataset=plan.dataset,
                    selection=plan.selection,
                    annotation_points=annotation_points,
                    annotation_hint_template=str(plan.annotation_hint_template),
                    json_example=str(plan.json_example),
                    json_example_answer_only=str(plan.json_example_answer_only),
                ),
                instance_seed=int(attempt_seed),
            )

            panels_trace = panel_trace_payload(plan.dataset)
            selection_trace = dict(plan.selection.trace)
            dataset_trace = dict(plan.dataset.trace)
            query_params = {
                "query_id": str(selected_query_id),
                "scene_variant": str(plan.scene_variant),
                "query_id_probabilities": dict(query_probabilities),
                "scene_variant_probabilities": dict(plan.scene_variant_probabilities),
                "panel_count": int(dataset_trace["panel_count"]),
                "segment_count": int(dataset_trace["segment_count"]),
                "question_format": str(plan.selection.question_format),
                "program_code": str(plan.program_code),
                **{
                    str(key): value
                    for key, value in selection_trace.items()
                    if str(key)
                    in {
                        "rank_segment",
                        "target_segment",
                        "target_count",
                        "segment_a",
                        "segment_b",
                        "condition_segment",
                        "extremum_direction",
                        "threshold",
                        "top_k",
                        "start_panel",
                        "end_panel",
                    }
                },
            }
            prompt_query_spec = build_prompt_query_spec(
                prompt_artifacts=prompt_artifacts,
                query_id=str(selected_query_id),
                params=query_params,
            )
            annotation_roles_trace = [
                {
                    "role": str(role.role),
                    "panel": str(role.panel),
                    **({} if role.key is None else {"key": str(role.key)}),
                    **({} if role.segment is None else {"segment": str(role.segment)}),
                }
                for role in plan.selection.annotation_roles
            ]
            answer_value = plan.selection.answer_value
            trace_payload = {
                "scene_ir": {
                    "scene_kind": f"chart_{str(plan.scene_variant)}_composition_panels_scene",
                    "entities": [dict(entity) for entity in rendered.entities],
                    "relations": {
                        "query_id": str(selected_query_id),
                        "scene_variant": str(plan.scene_variant),
                        "segment_labels": [str(label) for label in plan.dataset.segment_labels],
                    },
                },
                "query_spec": prompt_query_spec,
                "render_spec": {
                    "canvas_width": int(rendered.image.size[0]),
                    "canvas_height": int(rendered.image.size[1]),
                    "coord_space": "pixel",
                    "scene_variant": str(plan.scene_variant),
                    "font_assets": chart_font_asset_metadata(str(chart_font_family)),
                    "layout_jitter": dict(rendered.layout_jitter_meta),
                    "plot_bbox_px": list(rendered.plot_bbox_px),
                    **dict(rendered.render_meta),
                },
                "render_map": {
                    "image_id": "img0",
                    "plot_bbox_px": list(rendered.plot_bbox_px),
                    "legend_bbox_px": list(rendered.legend_bbox_px),
                    "legend_item_bboxes_px": {
                        str(segment): list(bbox)
                        for segment, bbox in rendered.legend_item_bboxes_px.items()
                    },
                    "context_protected_bboxes_px": {
                        "plot": list(rendered.plot_bbox_px),
                        **({"legend": list(rendered.legend_bbox_px)} if rendered.legend_bbox_px else {}),
                    },
                    "panel_traces": [dict(panel) for panel in rendered.panel_traces],
                    "annotation_bbox_by_key": {
                        f"{panel}:{segment}": list(bbox)
                        for (panel, segment), bbox in rendered.annotation_bbox_by_key.items()
                    },
                    "total_bbox_by_panel": {
                        str(panel): list(bbox)
                        for panel, bbox in rendered.total_bbox_by_panel.items()
                    },
                },
                "execution_trace": {
                    "query_id": str(selected_query_id),
                    "scene_variant": str(plan.scene_variant),
                    "answer_value": answer_value,
                    "annotation_type": str(annotation_type),
                    "annotation_values": [int(value) for value in plan.selection.annotation_values],
                    "annotation_roles": list(annotation_roles_trace),
                    "annotation_point_keys": list(annotation_points.keys()),
                    "annotation_bbox": [] if annotation_bbox is None else list(annotation_bbox),
                    "annotation_bbox_count": int(annotation_bbox_count),
                    "segment_labels": [str(label) for label in plan.dataset.segment_labels],
                    "panels": panels_trace,
                    "question_format": str(plan.selection.question_format),
                    "query_id_probabilities": dict(query_probabilities),
                    "scene_variant_probabilities": dict(plan.scene_variant_probabilities),
                    "program_code": str(plan.program_code),
                    "reasoning_load": float(plan.reasoning_load)
                    + float(SCENE_VARIANT_LOADS.get(str(plan.scene_variant), 0.0)),
                    **dict(dataset_trace),
                    **dict(selection_trace),
                },
                "witness_symbolic": {
                    "type": "composition_panels_aggregate",
                    "query_id": str(selected_query_id),
                    "answer_value": answer_value,
                    "annotation_type": str(annotation_type),
                    "annotation_values": [int(value) for value in plan.selection.annotation_values],
                    "annotation_point_keys": list(annotation_points.keys()),
                    "annotation_bbox": [] if annotation_bbox is None else list(annotation_bbox),
                    "annotation_bbox_count": int(annotation_bbox_count),
                    "calculation": dict(selection_trace),
                },
                "projected_annotation": dict(projected_annotation),
            }
            return TaskOutput(
                prompt=str(prompt_artifacts.prompt),
                prompt_variants=dict(prompt_artifacts.prompt_variants),
                answer_gt=_typed_answer(answer_value),
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


def counts_by_panel_label(dataset: CompositionPanelsDataset) -> dict[str, dict[str, int]]:
    return {str(panel.label): counts_for_panel(panel) for panel in dataset.panels}
