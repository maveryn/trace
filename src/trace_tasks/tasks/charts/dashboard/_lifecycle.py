"""Private neutral materialization helpers for dashboard chart tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from PIL import Image

from trace_tasks.core.seed import hash64
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.charts.dashboard.shared.annotations import keyed_point_artifacts, point_artifacts, point_set_artifacts
from trace_tasks.tasks.charts.dashboard.shared.output import build_trace_scaffold, render_dataset
from trace_tasks.tasks.charts.dashboard.shared.prompts import build_prompt_artifacts, build_prompt_slots
from trace_tasks.tasks.charts.dashboard.shared.sampling import DashboardTotalExtremumSample
from trace_tasks.tasks.charts.dashboard.shared.state import AnnotationRef, Category, DashboardDataset, DashboardQuery, SCENE_ID, SCENE_VARIANT
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import PromptTraceArtifacts, build_prompt_query_spec


@dataclass(frozen=True)
class DashboardTaskPlan:
    """Task-owned semantic plan consumed by neutral dashboard rendering."""

    dataset: DashboardDataset
    prompt_artifacts: PromptTraceArtifacts
    relations: Mapping[str, Any]
    answer_gt: TypedValue
    annotation_refs: tuple[AnnotationRef, ...]
    annotation_type: str = "point_set"
    annotation_roles: Mapping[str, AnnotationRef] | None = None


@dataclass(frozen=True)
class MaterializedDashboardTask:
    """Rendered payload assembled from one public task's semantic plan."""

    prompt: str
    answer_gt: TypedValue
    annotation_gt: TypedValue
    image: Image.Image
    trace_payload: dict[str, Any]
    query_id: str
    prompt_variants: dict[str, Any]


def dashboard_attempt_seed(instance_seed: int, attempt: int) -> int:
    """Return the neutral retry seed for dashboard-scene attempts."""

    return int(instance_seed) if int(attempt) == 0 else int(hash64(int(instance_seed), "charts.dashboard.retry", int(attempt)))


def materialize_dashboard_plan(*, instance_seed: int, params: Mapping[str, Any], selected_query_id: str, plan: DashboardTaskPlan) -> MaterializedDashboardTask:
    """Render a task-owned plan and add shared trace sections."""

    rendered, render_meta, sidecar_meta = render_dataset(plan.dataset, params=params, instance_seed=int(instance_seed))
    annotation_type = str(plan.annotation_type)
    if annotation_type == "point":
        if len(plan.annotation_refs) != 1:
            raise RuntimeError("dashboard scalar point annotation must contain exactly one ref")
        annotation_gt, witness_symbolic, projected_annotation, annotation_refs = point_artifacts(
            ref=plan.annotation_refs[0],
            rendered=rendered,
            panels=plan.dataset.panels,
            categories=plan.dataset.categories,
        )
    elif plan.annotation_roles is not None:
        annotation_gt, witness_symbolic, projected_annotation, annotation_refs = keyed_point_artifacts(
            role_to_ref=plan.annotation_roles,
            rendered=rendered,
            panels=plan.dataset.panels,
            categories=plan.dataset.categories,
        )
    elif annotation_type == "point_set":
        annotation_gt, witness_symbolic, projected_annotation = point_set_artifacts(
            refs=plan.annotation_refs,
            rendered=rendered,
            panels=plan.dataset.panels,
            categories=plan.dataset.categories,
        )
        annotation_refs = tuple(plan.annotation_refs)
    else:
        raise ValueError(f"unsupported dashboard annotation type: {annotation_type}")
    trace_payload = build_trace_scaffold(
        dataset=plan.dataset,
        rendered=rendered,
        render_meta=render_meta,
        sidecar_meta=sidecar_meta,
        projected_annotation=projected_annotation,
        annotation_refs=annotation_refs,
        answer_value=plan.answer_gt.value,
        relations=plan.relations,
        witness_symbolic=witness_symbolic,
    )
    relation_params = {"query_id": str(selected_query_id), "scene_id": SCENE_ID, "scene_variant": str(plan.dataset.scene_variant), "question_format": "dashboard_cross_panel_query", **dict(plan.relations)}
    trace_payload["scene_ir"]["relations"]["query_id"] = str(selected_query_id)
    trace_payload["query_spec"] = build_prompt_query_spec(prompt_artifacts=plan.prompt_artifacts, query_id=str(selected_query_id), params=relation_params)
    trace_payload["execution_trace"]["query_id"] = str(selected_query_id)
    return MaterializedDashboardTask(
        prompt=str(plan.prompt_artifacts.prompt),
        answer_gt=plan.answer_gt,
        annotation_gt=annotation_gt,
        image=rendered.image,
        trace_payload=trace_payload,
        query_id=str(selected_query_id),
        prompt_variants=dict(plan.prompt_artifacts.prompt_variants),
    )


def materialize_dashboard_plan_with_retries(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    selected_query_id: str,
    build_plan: Callable[[int, Mapping[str, Any], str], DashboardTaskPlan],
) -> MaterializedDashboardTask:
    """Retry task-owned plan construction, then return rendered payload."""

    attempts = max(1, int(max_attempts))
    last_error: Exception | None = None
    for attempt in range(attempts):
        attempt_seed = dashboard_attempt_seed(int(instance_seed), int(attempt))
        try:
            plan = build_plan(int(attempt_seed), params, str(selected_query_id))
            return materialize_dashboard_plan(instance_seed=int(attempt_seed), params=params, selected_query_id=str(selected_query_id), plan=plan)
        except ValueError as exc:
            last_error = exc
    raise RuntimeError(f"failed to materialize dashboard plan: {last_error}")


def dashboard_task_output_fields(materialized: MaterializedDashboardTask) -> dict[str, Any]:
    """Return neutral final-output fields from an already materialized payload."""

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


def dashboard_total_extremum_plan_from_sample(
    *,
    categories: tuple[Category, ...],
    total_sample: DashboardTotalExtremumSample,
    relations: Mapping[str, Any],
    prompt_query_key: str,
    instance_seed: int,
) -> DashboardTaskPlan:
    """Return a plan for label tasks whose annotation is a total witness set."""

    dataset = DashboardDataset(
        scene_variant=SCENE_VARIANT,
        categories=categories,
        panels=total_sample.panels,
        query=DashboardQuery(
            answer=str(total_sample.answer_label),
            answer_type="string",
            annotation_refs=total_sample.annotation_refs,
            params=dict(relations),
        ),
    )
    prompt_artifacts = build_prompt_artifacts(
        prompt_query_key=str(prompt_query_key),
        dynamic_slots=build_prompt_slots(dataset=dataset),
        instance_seed=int(instance_seed),
    )
    return DashboardTaskPlan(
        dataset=dataset,
        prompt_artifacts=prompt_artifacts,
        relations=dict(relations),
        answer_gt=TypedValue(type="string", value=str(total_sample.answer_label)),
        annotation_refs=total_sample.annotation_refs,
        annotation_type="point_set",
    )


def run_dashboard_public_task(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    selected_query_id: str,
    build_plan: Callable[[int, Mapping[str, Any], str], DashboardTaskPlan],
    build_output: Callable[[MaterializedDashboardTask], Any],
) -> Any:
    """Materialize a public-file-owned plan and call its output factory."""

    materialized = materialize_dashboard_plan_with_retries(
        instance_seed=int(instance_seed),
        params=params,
        max_attempts=int(max_attempts),
        selected_query_id=str(selected_query_id),
        build_plan=build_plan,
    )
    return build_output(materialized)


__all__ = [
    "DashboardTaskPlan",
    "MaterializedDashboardTask",
    "dashboard_task_output_fields",
    "dashboard_total_extremum_plan_from_sample",
    "materialize_dashboard_plan",
    "materialize_dashboard_plan_with_retries",
    "run_dashboard_public_task",
]
