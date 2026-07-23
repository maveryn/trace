"""Private neutral materialization helpers for contour-density chart tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from PIL import Image

from trace_tasks.core.seed import hash64
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.charts.contour_density.shared.annotations import annotation_value
from trace_tasks.tasks.charts.contour_density.shared.defaults import SCENE_ID
from trace_tasks.tasks.charts.contour_density.shared.output import answer_value, build_trace_scaffold
from trace_tasks.tasks.charts.contour_density.shared.rendering import render_dataset
from trace_tasks.tasks.charts.contour_density.shared.state import ContourDataset
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import PromptTraceArtifacts, build_prompt_query_spec


@dataclass(frozen=True)
class ContourTaskPlan:
    """Task-owned semantic plan consumed by neutral contour rendering."""

    dataset: ContourDataset
    prompt_artifacts: PromptTraceArtifacts
    relations: Mapping[str, Any]


@dataclass(frozen=True)
class MaterializedContourTask:
    """Rendered payload assembled from one public task's semantic plan."""

    prompt: str
    answer_gt: TypedValue
    annotation_gt: TypedValue
    image: Image.Image
    trace_payload: dict[str, Any]
    query_id: str
    prompt_variants: dict[str, Any]


PlanBuilder = Callable[[int, Mapping[str, Any], str], ContourTaskPlan]


def contour_attempt_seed(instance_seed: int, attempt: int) -> int:
    """Return the neutral retry seed for contour-scene generation attempts."""

    return (
        int(instance_seed)
        if int(attempt) == 0
        else int(hash64(int(instance_seed), "charts.contour_density.retry", int(attempt)))
    )


def materialize_contour_plan(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    selected_query_id: str,
    plan: ContourTaskPlan,
) -> MaterializedContourTask:
    """Render a task-owned plan and build shared trace sections."""

    rendered, chart_font_family = render_dataset(plan.dataset, params=params, instance_seed=int(instance_seed))
    annotation = annotation_value(plan.dataset, rendered)
    annotation_type = str(plan.dataset.query.annotation_type)
    if annotation_type == "bbox":
        annotation_gt = TypedValue(type=annotation_type, value=list(annotation))
    elif annotation_type == "bbox_set":
        annotation_gt = TypedValue(type=annotation_type, value=[list(value) for value in annotation])
    elif annotation_type in {"bbox_map", "point_map"}:
        annotation_gt = TypedValue(type=annotation_type, value={key: list(value) for key, value in annotation.items()})
    else:
        raise ValueError(f"unsupported annotation type: {annotation_type}")
    resolved_answer = answer_value(plan.dataset)
    trace_payload = build_trace_scaffold(
        dataset=plan.dataset,
        rendered=rendered,
        annotation=annotation,
        chart_font_family=str(chart_font_family),
        relations=plan.relations,
    )
    trace_payload["scene_ir"]["relations"]["query_id"] = str(selected_query_id)
    trace_payload["query_spec"] = build_prompt_query_spec(
        prompt_artifacts=plan.prompt_artifacts,
        query_id=str(selected_query_id),
        params={"query_id": str(selected_query_id), **dict(plan.relations)},
    )
    trace_payload["execution_trace"]["query_id"] = str(selected_query_id)
    return MaterializedContourTask(
        prompt=str(plan.prompt_artifacts.prompt),
        answer_gt=TypedValue(type=str(plan.dataset.query.answer_type), value=resolved_answer),
        annotation_gt=annotation_gt,
        image=rendered.image,
        trace_payload=trace_payload,
        query_id=str(selected_query_id),
        prompt_variants=dict(plan.prompt_artifacts.prompt_variants),
    )


def contour_task_output_fields(materialized: MaterializedContourTask) -> dict[str, Any]:
    """Return common TaskOutput kwargs for public contour task files."""

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


def run_contour_public_task(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    selected_query_id: str,
    failure_label: str,
    build_plan: PlanBuilder,
    build_output: Callable[[MaterializedContourTask], Any],
) -> Any:
    """Retry task-owned plan construction, then materialize the contour output."""

    last_error: Exception | None = None
    for attempt in range(max(1, int(max_attempts))):
        attempt_seed = contour_attempt_seed(int(instance_seed), int(attempt))
        try:
            plan = build_plan(int(attempt_seed), params, str(selected_query_id))
            materialized = materialize_contour_plan(
                instance_seed=int(attempt_seed),
                params=params,
                selected_query_id=str(selected_query_id),
                plan=plan,
            )
            return build_output(materialized)
        except (ValueError, RuntimeError) as exc:
            last_error = exc
    raise RuntimeError(f"failed to generate {failure_label}: {last_error}")


__all__ = [
    "ContourTaskPlan",
    "MaterializedContourTask",
    "contour_task_output_fields",
    "materialize_contour_plan",
    "run_contour_public_task",
]
