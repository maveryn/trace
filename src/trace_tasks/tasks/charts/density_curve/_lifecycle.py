"""Neutral lifecycle helpers for density-curve chart tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from trace_tasks.core.seed import hash64
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.charts.density_curve.shared.annotations import density_curve_annotation_payload
from trace_tasks.tasks.charts.density_curve.shared.defaults import SCENE_ID, SCENE_NAMESPACE, gen_float, resolve_density_curve_render_params
from trace_tasks.tasks.charts.density_curve.shared.metrics import extremum_gap, extremum_label
from trace_tasks.tasks.charts.density_curve.shared.output import build_trace_scaffold, render_dataset
from trace_tasks.tasks.charts.density_curve.shared.prompts import build_prompt_artifacts, dynamic_slots
from trace_tasks.tasks.charts.density_curve.shared.sampling import bind_density_curve_query, sample_density_curve_scene
from trace_tasks.tasks.charts.density_curve.shared.state import DensityCurveDataset, DensityCurveObjectiveSpec, DensityCurveRendered
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec, PromptTraceArtifacts


@dataclass(frozen=True)
class DensityCurveTaskArtifacts:
    """Rendered payload pieces assembled for a public task."""

    dataset: DensityCurveDataset
    rendered: DensityCurveRendered
    render_meta: dict[str, Any]
    prompt_artifacts: PromptTraceArtifacts
    answer_gt: TypedValue
    annotation_gt: TypedValue
    trace_payload: dict[str, Any]


def density_curve_attempt_seed(instance_seed: int, attempt_index: int) -> int:
    """Return deterministic retry seed for density-curve generation."""

    return (
        int(instance_seed)
        if int(attempt_index) == 0
        else int(hash64(int(instance_seed), f"{SCENE_NAMESPACE}.attempt", int(attempt_index)))
    )


def build_density_curve_dataset_for_objective(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    objective: DensityCurveObjectiveSpec,
    prompt_key: str,
    failure_label: str,
) -> DensityCurveDataset:
    """Sample until one scene satisfies a task-owned objective gap."""

    render_params = resolve_density_curve_render_params(params, instance_seed=int(instance_seed))
    last_error: Exception | None = None
    for attempt_index in range(max(1, int(max_attempts))):
        attempt_seed = density_curve_attempt_seed(int(instance_seed), int(attempt_index))
        try:
            sample = sample_density_curve_scene(
                params,
                instance_seed=int(attempt_seed),
                render_params=render_params,
            )
            answer_label = extremum_label(
                sample.curves,
                metric_name=str(objective.metric_name),
                direction=str(objective.direction),
            )
            winner_gap = extremum_gap(
                sample.curves,
                metric_name=str(objective.metric_name),
                direction=str(objective.direction),
            )
            min_gap = gen_float(params, str(objective.min_gap_param), float(objective.min_gap_fallback))
            if float(winner_gap) < float(min_gap):
                raise ValueError("density-curve sampled metrics did not satisfy winner-gap constraints")
            return bind_density_curve_query(
                sample,
                prompt_key=str(prompt_key),
                objective=objective,
                answer_label=str(answer_label),
                winner_gap=float(winner_gap),
            )
        except ValueError as exc:
            last_error = exc
    raise RuntimeError(f"could not generate {failure_label} after {max_attempts} attempts") from last_error


def materialize_density_curve_task(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    selected_query_id: str,
    objective: DensityCurveObjectiveSpec,
    max_attempts: int,
    task_id: str,
) -> DensityCurveTaskArtifacts:
    """Render and trace one public density-curve task objective."""

    dataset = build_density_curve_dataset_for_objective(
        instance_seed=int(instance_seed),
        params=params,
        max_attempts=int(max_attempts),
        objective=objective,
        prompt_key=str(selected_query_id),
        failure_label=str(task_id),
    )
    rendered, render_meta = render_dataset(dataset, params=params, instance_seed=int(instance_seed))
    answer_gt = TypedValue(type="string", value=str(dataset.query.answer_label))
    annotation_type, annotation, projected_annotation = density_curve_annotation_payload(
        dataset=dataset,
        rendered=rendered,
    )
    annotation_gt = TypedValue(type=str(annotation_type), value=annotation)
    prompt_artifacts = build_prompt_artifacts(
        prompt_query_key=str(selected_query_id),
        dynamic_slot_values=dynamic_slots(dataset),
        instance_seed=int(instance_seed),
    )
    trace_payload = build_trace_scaffold(
        dataset=dataset,
        rendered=rendered,
        render_meta=render_meta,
        projected_annotation=projected_annotation,
        answer_label=str(answer_gt.value),
    )
    relation_params = {
        "query_id": str(selected_query_id),
        "scene_id": SCENE_ID,
        "scene_variant": str(dataset.query.trace.get("scene_variant", "density_curve")),
        "curve_count": int(dataset.curve_count),
        "metric_name": str(objective.metric_name),
        "direction": str(objective.direction),
        "visible_role": str(objective.visible_role),
        "interval_start": round(float(dataset.query.interval_start), 3),
        "interval_end": round(float(dataset.query.interval_end), 3),
        "reference_x": round(float(dataset.query.reference_x), 3),
        **dict(dataset.query.trace),
    }
    trace_payload["scene_ir"]["relations"]["query_id"] = str(selected_query_id)
    trace_payload["query_spec"] = build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts,
        query_id=str(selected_query_id),
        params=relation_params,
    )
    trace_payload["execution_trace"]["query_id"] = str(selected_query_id)
    return DensityCurveTaskArtifacts(
        dataset=dataset,
        rendered=rendered,
        render_meta=dict(render_meta),
        prompt_artifacts=prompt_artifacts,
        answer_gt=answer_gt,
        annotation_gt=annotation_gt,
        trace_payload=trace_payload,
    )
