"""Neutral materialization helpers for curve-panel chart tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from PIL import Image

from trace_tasks.core.seed import hash64
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.charts.curve_panels.shared.annotations import (
    bbox_set_from_panel_labels,
    point_map_from_ids,
    point_set_from_ids,
)
from trace_tasks.tasks.charts.curve_panels.shared.defaults import SCENE_ID, SCENE_VARIANT
from trace_tasks.tasks.charts.curve_panels.shared.output import (
    build_trace_scaffold,
    render_dataset,
)
from trace_tasks.tasks.charts.curve_panels.shared.prompts import build_prompt_artifacts
from trace_tasks.tasks.charts.curve_panels.shared.sampling import panels_from_values
from trace_tasks.tasks.charts.curve_panels.shared.state import (
    CurvePanelDataset,
    QuerySelection,
)
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import (
    PromptTraceArtifacts,
    build_prompt_query_spec,
)
from trace_tasks.tasks.shared.fixed_query import select_task_query_id


@dataclass(frozen=True)
class CurvePanelTaskPlan:
    """Task-owned semantic plan consumed by neutral rendering."""

    dataset: CurvePanelDataset
    prompt_artifacts: PromptTraceArtifacts
    relations: Mapping[str, Any]
    annotation_type: str = "point_set"
    allow_empty_annotation: bool = False


@dataclass(frozen=True)
class MaterializedCurvePanelTask:
    """Rendered task payload assembled from a public task's plan."""

    prompt: str
    answer_gt: TypedValue
    annotation_gt: TypedValue
    image: Image.Image
    trace_payload: dict[str, Any]
    query_id: str
    prompt_variants: dict[str, Any]


PlanBuilder = Callable[[int, Mapping[str, Any], str], CurvePanelTaskPlan]


def build_curve_panel_query_record(
    *,
    prompt_key: str,
    answer: Any,
    answer_type: str,
    panel_label: str = "",
    method_label: str = "",
    method_a_label: str = "",
    method_b_label: str = "",
    x_value: int = 0,
    start_x_value: int = 0,
    end_x_value: int = 0,
    threshold_value: int = 0,
    threshold_direction: str = "",
    threshold_panel_labels: tuple[str, ...] = (),
    annotation_panel_labels: tuple[str, ...] = (),
    annotation_point_ids: tuple[str, ...] = (),
    annotation_keyed_point_ids: Mapping[str, str] | None = None,
    annotation_intersection_ids: tuple[str, ...] = (),
    annotation_threshold_crossing_ids: tuple[str, ...] = (),
    trace: Mapping[str, Any] | None = None,
) -> QuerySelection:
    """Build the common query schema from task-owned semantic values."""

    return QuerySelection(
        prompt_key=str(prompt_key),
        scene_variant=SCENE_VARIANT,
        answer=answer,
        answer_type=str(answer_type),
        panel_label=str(panel_label),
        method_label=str(method_label),
        method_a_label=str(method_a_label),
        method_b_label=str(method_b_label),
        x_value=int(x_value),
        start_x_value=int(start_x_value),
        end_x_value=int(end_x_value),
        threshold_value=int(threshold_value),
        threshold_direction=str(threshold_direction),
        threshold_panel_labels=tuple(str(label) for label in threshold_panel_labels),
        annotation_panel_labels=tuple(str(label) for label in annotation_panel_labels),
        annotation_point_ids=tuple(str(item) for item in annotation_point_ids),
        annotation_keyed_point_ids={
            str(key): str(value)
            for key, value in dict(annotation_keyed_point_ids or {}).items()
        },
        annotation_intersection_ids=tuple(
            str(item) for item in annotation_intersection_ids
        ),
        annotation_threshold_crossing_ids=tuple(
            str(item) for item in annotation_threshold_crossing_ids
        ),
        trace=dict(trace or {}),
    )


def curve_panel_attempt_seed(instance_seed: int, attempt: int) -> int:
    """Return the neutral retry seed for curve-panel generation attempts."""

    return (
        int(instance_seed)
        if int(attempt) == 0
        else int(hash64(int(instance_seed), "charts.curve_panels.retry", int(attempt)))
    )


def _annotation_for_plan(plan: CurvePanelTaskPlan, rendered) -> TypedValue:
    annotation_type = str(plan.annotation_type)
    if annotation_type == "bbox_set":
        boxes = bbox_set_from_panel_labels(
            rendered=rendered,
            panel_labels=plan.dataset.query.annotation_panel_labels,
        )
        if not boxes and not bool(plan.allow_empty_annotation):
            raise RuntimeError("curve-panel task produced empty bbox annotation")
        return TypedValue(type=annotation_type, value=[list(box) for box in boxes])
    if annotation_type == "point_map":
        points = point_map_from_ids(
            rendered=rendered,
            keyed_point_ids=plan.dataset.query.annotation_keyed_point_ids,
        )
        if not points and not bool(plan.allow_empty_annotation):
            raise RuntimeError("curve-panel task produced empty keyed annotation")
        return TypedValue(type=annotation_type, value=dict(points))
    if annotation_type not in {"point", "point_set"}:
        raise ValueError(f"unsupported annotation type: {annotation_type}")
    points = point_set_from_ids(
        rendered=rendered,
        point_ids=plan.dataset.query.annotation_point_ids,
        intersection_ids=plan.dataset.query.annotation_intersection_ids,
        crossing_ids=plan.dataset.query.annotation_threshold_crossing_ids,
    )
    if not points and not bool(plan.allow_empty_annotation):
        raise RuntimeError("curve-panel task produced empty annotation")
    if annotation_type == "point":
        if len(points) != 1:
            raise RuntimeError("curve-panel scalar point annotation must contain exactly one point")
        return TypedValue(type=annotation_type, value=list(points[0]))
    return TypedValue(type=annotation_type, value=[list(point) for point in points])


def materialize_curve_panel_plan(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    selected_query_id: str,
    plan: CurvePanelTaskPlan,
) -> MaterializedCurvePanelTask:
    """Render one task-owned plan and build common trace sections."""

    rendered, chart_font_family = render_dataset(
        plan.dataset, params=params, instance_seed=int(instance_seed)
    )
    annotation_gt = _annotation_for_plan(plan, rendered)
    trace_payload = build_trace_scaffold(
        dataset=plan.dataset,
        rendered=rendered,
        annotation_type=str(annotation_gt.type),
        annotation=annotation_gt.value,
        chart_font_family=str(chart_font_family),
        params=params,
        relations=plan.relations,
    )
    trace_payload["scene_ir"]["relations"]["query_id"] = str(selected_query_id)
    trace_payload["query_spec"] = build_prompt_query_spec(
        prompt_artifacts=plan.prompt_artifacts,
        query_id=str(selected_query_id),
        params={"query_id": str(selected_query_id), **dict(plan.relations)},
    )
    trace_payload["execution_trace"]["query_id"] = str(selected_query_id)
    return MaterializedCurvePanelTask(
        prompt=str(plan.prompt_artifacts.prompt),
        answer_gt=TypedValue(
            type=str(plan.dataset.query.answer_type), value=plan.dataset.query.answer
        ),
        annotation_gt=annotation_gt,
        image=rendered.image,
        trace_payload=trace_payload,
        query_id=str(selected_query_id),
        prompt_variants=dict(plan.prompt_artifacts.prompt_variants),
    )


def curve_panel_task_output_fields(
    materialized: MaterializedCurvePanelTask,
) -> dict[str, Any]:
    """Return common TaskOutput kwargs for public curve-panel task files."""

    return {
        "prompt": materialized.prompt,
        "answer_gt": materialized.answer_gt,
        "annotation_gt": materialized.annotation_gt,
        "image": materialized.image,
        "image_id": "img0",
        "trace_payload": materialized.trace_payload,
        "task_versions": default_task_versions(),
        "scene_id": SCENE_ID,
        "query_id": materialized.query_id,
        "prompt_variants": materialized.prompt_variants,
    }


def curve_panel_relation_defaults(dataset: CurvePanelDataset) -> dict[str, Any]:
    """Return renderer-neutral relation fields common to curve-panel tasks."""

    return {
        "scene_variant": str(dataset.scene_variant),
        "panel_count": int(len(dataset.panels)),
        "method_count": int(len(dataset.panels[0].curves)),
        "x_tick_count": int(len(dataset.x_values)),
        "threshold_direction": str(dataset.query.threshold_direction),
    }


def build_curve_panel_plan_from_query(
    *,
    x_values: tuple[int, ...],
    y_min: int,
    y_max: int,
    panel_labels: tuple[str, ...],
    method_labels: tuple[str, ...],
    colors: tuple[tuple[int, int, int], ...],
    values_by_panel_method: Mapping[str, Mapping[str, list[int]]],
    query: QuerySelection,
    dynamic_slots: Mapping[str, str],
    instance_seed: int,
    intersections: tuple[Any, ...] = (),
    threshold_crossings: tuple[Any, ...] = (),
    annotation_type: str = "point_set",
    allow_empty_annotation: bool = False,
    omitted_panel_methods: Mapping[str, tuple[str, ...]] | None = None,
) -> CurvePanelTaskPlan:
    """Assemble neutral dataset, prompt, and relation metadata from task semantics."""

    dataset = CurvePanelDataset(
        scene_variant=query.scene_variant,
        x_values=tuple(int(value) for value in x_values),
        y_min=int(y_min),
        y_max=int(y_max),
        panels=panels_from_values(
            values_by_panel_method=values_by_panel_method,
            panel_labels=panel_labels,
            method_labels=method_labels,
            colors=colors,
            omitted_panel_methods=omitted_panel_methods,
        ),
        query=query,
        intersections=tuple(intersections),
        threshold_crossings=tuple(threshold_crossings),
    )
    prompt_artifacts = build_prompt_artifacts(
        prompt_query_key=str(query.prompt_key),
        dynamic_slots=dict(dynamic_slots),
        instance_seed=int(instance_seed),
    )
    return CurvePanelTaskPlan(
        dataset=dataset,
        prompt_artifacts=prompt_artifacts,
        relations={**curve_panel_relation_defaults(dataset), **dict(query.trace)},
        annotation_type=str(annotation_type),
        allow_empty_annotation=bool(allow_empty_annotation),
    )


def build_curve_panel_task_output(
    materialized: MaterializedCurvePanelTask,
) -> TaskOutput:
    """Build the common TaskOutput shell after a public task binds semantics."""

    return TaskOutput(**curve_panel_task_output_fields(materialized))


def run_curve_panel_public_task(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    selected_query_id: str,
    failure_label: str,
    build_plan: PlanBuilder,
    build_output: Callable[[MaterializedCurvePanelTask], Any],
) -> Any:
    """Retry task-owned plan construction, then materialize the output."""

    last_error: Exception | None = None
    for attempt in range(max(1, int(max_attempts))):
        attempt_seed = curve_panel_attempt_seed(int(instance_seed), int(attempt))
        try:
            plan = build_plan(int(attempt_seed), params, str(selected_query_id))
            materialized = materialize_curve_panel_plan(
                instance_seed=int(attempt_seed),
                params=params,
                selected_query_id=str(selected_query_id),
                plan=plan,
            )
            return build_output(materialized)
        except (ValueError, RuntimeError) as exc:
            last_error = exc
    raise RuntimeError(f"failed to generate {failure_label}: {last_error}")


def run_curve_panel_task_lifecycle(
    *,
    instance_seed: int,
    params: dict[str, Any],
    max_attempts: int,
    supported_query_ids: tuple[str, ...],
    default_query_id: str,
    failure_label: str,
    build_plan: PlanBuilder,
) -> TaskOutput:
    """Run neutral query selection and materialization for a public task hook."""

    selected_query_id, _probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=supported_query_ids,
        default_query_id=str(default_query_id),
        task_id=str(failure_label),
    )
    return run_curve_panel_public_task(
        instance_seed=int(instance_seed),
        params=task_params,
        max_attempts=int(max_attempts),
        selected_query_id=str(selected_query_id),
        failure_label=str(failure_label),
        build_plan=build_plan,
        build_output=build_curve_panel_task_output,
    )


__all__ = [
    "CurvePanelTaskPlan",
    "MaterializedCurvePanelTask",
    "build_curve_panel_task_output",
    "build_curve_panel_plan_from_query",
    "build_curve_panel_query_record",
    "curve_panel_relation_defaults",
    "curve_panel_task_output_fields",
    "materialize_curve_panel_plan",
    "run_curve_panel_public_task",
    "run_curve_panel_task_lifecycle",
]
