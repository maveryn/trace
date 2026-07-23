"""Private materialization lifecycle for single-series chart tasks."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any

from trace_tasks.core.seed import hash64
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.charts.single_series.shared.annotations import annotation_refs, mark_annotation
from trace_tasks.tasks.charts.single_series.shared.defaults import (
    COUNT_SCENE_VARIANTS,
    COUNTERFACTUAL_SCENE_VARIANTS,
    ORDERED_SCENE_VARIANTS,
    SUMMARY_SCENE_VARIANTS,
    THRESHOLD_SCENE_VARIANTS,
    resolve_scene_variant,
)
from trace_tasks.tasks.charts.single_series.shared.output import (
    execution_record,
    render_map,
    render_spec,
    scene_relations,
    witness_symbolic,
)
from trace_tasks.tasks.charts.single_series.shared.prompts import (
    crossing_slots,
    endpoint_slots,
    object_description,
    ordinal,
    quoted_join,
    render_prompt_artifacts,
    statistic_prompt,
    streak_slots,
    turning_slots,
)
from trace_tasks.tasks.charts.single_series.shared.rendering import render_dataset
from trace_tasks.tasks.charts.single_series.shared.sampling import (
    count_dataset,
    counterfactual_dataset,
    interval_change_dataset,
    summary_dataset,
    threshold_crossing_dataset,
    trend_structure_dataset,
)
from trace_tasks.tasks.charts.single_series.shared.state import SCENE_ID, SingleSeriesDataset
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import PromptTraceArtifacts, build_prompt_query_spec


@dataclass(frozen=True)
class SingleSeriesTaskPlan:
    """Task-owned semantic plan consumed by neutral chart materialization."""

    dataset: SingleSeriesDataset
    params: Mapping[str, Any]
    scene_variant: str
    prompt_bundle_id: str
    prompt_scene_key: str
    prompt_task_key: str
    prompt_key: str
    dynamic_slots: Mapping[str, Any]
    answer_gt: TypedValue
    annotation_kind: str
    annotation_roles: Mapping[str, str]
    relation_params: Mapping[str, Any]
    question_format: str
    program_code: str
    reasoning_load: float


PlanBuilder = Callable[[Mapping[str, Any], int, str, Mapping[str, float]], SingleSeriesTaskPlan]


def _resolve_trace_spec(dataset: SingleSeriesDataset, value: Any) -> Any:
    if not (isinstance(value, str) and value.startswith("trace:")):
        return value
    _prefix, kind, key = value.split(":", 2)
    raw = dataset.trace.get(key, ())
    if kind == "int":
        return int(raw)
    if kind == "str":
        return str(raw)
    if kind == "list":
        return list(raw)
    if kind == "quoted":
        return quoted_join(raw)
    return raw


def _resolve_trace_mapping(dataset: SingleSeriesDataset, mapping: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _resolve_trace_spec(dataset, value) for key, value in mapping.items()}


def package_single_series_plan(
    *,
    dataset: SingleSeriesDataset,
    params: Mapping[str, Any],
    scene_variant: str,
    scene_variant_probabilities: Mapping[str, float],
    prompt_bundle_id: str,
    prompt_scene_key: str,
    prompt_task_key: str,
    prompt_key: str,
    dynamic_slots: Mapping[str, Any],
    answer_type: str,
    annotation_kind: str,
    annotation_roles: Mapping[str, str] | None = None,
    relation_params: Mapping[str, Any] | None = None,
    question_format: str,
    program_code: str,
    reasoning_load: float,
) -> SingleSeriesTaskPlan:
    """Bind task-owned semantic fields to the neutral single-series lifecycle."""

    answer_value = str(dataset.answer_value) if str(answer_type) == "string" else int(dataset.answer_value)
    return SingleSeriesTaskPlan(
        dataset=dataset,
        params=dict(params),
        scene_variant=str(scene_variant),
        prompt_bundle_id=str(prompt_bundle_id),
        prompt_scene_key=str(prompt_scene_key),
        prompt_task_key=str(prompt_task_key),
        prompt_key=str(prompt_key),
        dynamic_slots=dict(dynamic_slots),
        answer_gt=TypedValue(type=str(answer_type), value=answer_value),
        annotation_kind=str(annotation_kind),
        annotation_roles={} if annotation_roles is None else dict(annotation_roles),
        relation_params={
            **({} if relation_params is None else dict(relation_params)),
            "scene_variant_probabilities": dict(scene_variant_probabilities),
        },
        question_format=str(question_format),
        program_code=str(program_code),
        reasoning_load=float(reasoning_load),
    )


def build_count_plan(
    *,
    params: Mapping[str, Any],
    seed: int,
    namespace: str,
    count_variant: str,
    prompt_key: str,
    dynamic_slots: Mapping[str, Any],
    relation_params: Mapping[str, Any],
    program_code: str,
    reasoning_load: float,
) -> SingleSeriesTaskPlan:
    """Build a single-series count task from semantic count arguments."""

    scene_variant, scene_probs = resolve_scene_variant(
        params,
        instance_seed=int(seed),
        supported=COUNT_SCENE_VARIANTS,
        namespace=f"{prompt_key}.scene",
    )
    dataset = count_dataset(
        count_variant=str(count_variant),
        scene_variant=str(scene_variant),
        params=params,
        instance_seed=int(seed),
        namespace=str(namespace),
    )
    slots = {"object_description": object_description(str(scene_variant)), **_resolve_trace_mapping(dataset, dynamic_slots)}
    return package_single_series_plan(
        dataset=dataset,
        params=params,
        scene_variant=str(scene_variant),
        scene_variant_probabilities=scene_probs,
        prompt_bundle_id="charts_counting_v1",
        prompt_scene_key="labeled_chart_counting",
        prompt_task_key="value_count_query",
        prompt_key=str(prompt_key),
        dynamic_slots=slots,
        answer_type="integer",
        annotation_kind="point_set",
        relation_params=_resolve_trace_mapping(dataset, relation_params),
        question_format="numeric_open",
        program_code=str(program_code),
        reasoning_load=float(reasoning_load),
    )


def build_trend_structure_plan(
    *,
    params: Mapping[str, Any],
    seed: int,
    namespace: str,
    trend_variant: str,
    prompt_key: str,
    dynamic_slots: Mapping[str, Any],
    relation_params: Mapping[str, Any],
    program_code: str,
    reasoning_load: float,
) -> SingleSeriesTaskPlan:
    """Build peak/trough or monotone-run tasks from semantic trend arguments."""

    scene_variant, scene_probs = resolve_scene_variant(
        params,
        instance_seed=int(seed),
        supported=ORDERED_SCENE_VARIANTS,
        namespace=f"{prompt_key}.scene",
    )
    dataset = trend_structure_dataset(
        trend_variant=str(trend_variant),
        scene_variant=str(scene_variant),
        params=params,
        instance_seed=int(seed),
        namespace=str(namespace),
    )
    return package_single_series_plan(
        dataset=dataset,
        params=params,
        scene_variant=str(scene_variant),
        scene_variant_probabilities=scene_probs,
        prompt_bundle_id="charts_trend_v1",
        prompt_scene_key="ordered_chart_trend",
        prompt_task_key="trend_value_query",
        prompt_key=str(prompt_key),
        dynamic_slots={"object_description": object_description(str(scene_variant)), **_resolve_trace_mapping(dataset, dynamic_slots)},
        answer_type="integer",
        annotation_kind="point_set",
        relation_params=_resolve_trace_mapping(dataset, relation_params),
        question_format="numeric_open",
        program_code=str(program_code),
        reasoning_load=float(reasoning_load),
    )


def build_interval_change_plan(
    *,
    params: Mapping[str, Any],
    seed: int,
    namespace: str,
    interval_variant: str,
    prompt_key: str,
    dynamic_slots: Mapping[str, Any],
    relation_params: Mapping[str, Any],
    program_code: str,
    reasoning_load: float,
) -> SingleSeriesTaskPlan:
    """Build endpoint-pair numeric tasks with keyed start/end annotations."""

    scene_variant, scene_probs = resolve_scene_variant(
        params,
        instance_seed=int(seed),
        supported=ORDERED_SCENE_VARIANTS,
        namespace=f"{prompt_key}.scene",
    )
    dataset = interval_change_dataset(
        interval_variant=str(interval_variant),
        scene_variant=str(scene_variant),
        params=params,
        instance_seed=int(seed),
        namespace=str(namespace),
    )
    start, end = str(dataset.trace["start_label"]), str(dataset.trace["end_label"])
    return package_single_series_plan(
        dataset=dataset,
        params=params,
        scene_variant=str(scene_variant),
        scene_variant_probabilities=scene_probs,
        prompt_bundle_id="charts_trend_v1",
        prompt_scene_key="ordered_chart_trend",
        prompt_task_key="trend_value_query",
        prompt_key=str(prompt_key),
        dynamic_slots={
            "object_description": object_description(str(scene_variant)),
            "start_label": start,
            "end_label": end,
            **_resolve_trace_mapping(dataset, dynamic_slots),
        },
        answer_type="integer",
        annotation_kind="point_map",
        annotation_roles={"start_mark": start, "end_mark": end},
        relation_params={
            **_resolve_trace_mapping(dataset, relation_params),
            "start_label": start,
            "end_label": end,
            "interval_gap": int(dataset.trace.get("interval_gap", 0)),
        },
        question_format="numeric_open",
        program_code=str(program_code),
        reasoning_load=float(reasoning_load),
    )


def build_crossing_plan(
    *,
    params: Mapping[str, Any],
    seed: int,
    namespace: str,
    crossing_variant: str,
    crossing_mode: str,
    direction: str,
    prompt_key: str,
    projected: bool,
    program_code: str,
    reasoning_load: float,
    annotation_kind: str = "point_set",
) -> SingleSeriesTaskPlan:
    """Build observed/projected threshold crossing label tasks."""

    scene_variant, scene_probs = resolve_scene_variant(
        params,
        instance_seed=int(seed),
        supported=THRESHOLD_SCENE_VARIANTS,
        namespace=f"{prompt_key}.scene",
    )
    dataset = threshold_crossing_dataset(
        crossing_variant=str(crossing_variant),
        scene_variant=str(scene_variant),
        params=params,
        instance_seed=int(seed),
        namespace=str(namespace),
    )
    if str(annotation_kind) == "point":
        answer_label = str(dataset.answer_value)
        dataset = replace(
            dataset,
            annotation_labels=(answer_label,),
            ordered_annotation_labels=(answer_label,),
            trace={
                **dict(dataset.trace),
                "annotation_labels": (answer_label,),
                "ordered_annotation_labels": (answer_label,),
            },
        )
    return package_single_series_plan(
        dataset=dataset,
        params=params,
        scene_variant=str(scene_variant),
        scene_variant_probabilities=scene_probs,
        prompt_bundle_id="charts_trend_v1",
        prompt_scene_key="ordered_chart_trend",
        prompt_task_key="threshold_crossing_label_query",
        prompt_key=str(prompt_key),
        dynamic_slots={
            "object_description": object_description(str(scene_variant)),
            "threshold": str(dataset.trace.get("threshold", "")),
            "unanswerable_instruction": "",
            **crossing_slots(str(direction), projected=bool(projected)),
        },
        answer_type="string",
        annotation_kind=str(annotation_kind),
        relation_params={
            "crossing_variant": str(crossing_variant),
            "crossing_mode": str(crossing_mode),
            "crossing_direction": str(direction),
            "threshold": int(dataset.trace.get("threshold", 0)),
            "comparison": str(dataset.trace.get("comparison", "")),
            "answer_label": str(dataset.answer_value),
            "answer_index": int(dataset.trace.get("answer_index", -1)),
        },
        question_format="label_open",
        program_code=str(program_code),
        reasoning_load=float(reasoning_load),
    )


def build_summary_plan(
    *,
    params: Mapping[str, Any],
    seed: int,
    namespace: str,
    statistic_kind: str,
    prompt_key: str,
    answer_target: str,
    program_code: str,
    reasoning_load: float,
) -> SingleSeriesTaskPlan:
    """Build order-statistic value or label tasks from statistic semantics."""

    scene_variant, scene_probs = resolve_scene_variant(
        params,
        instance_seed=int(seed),
        supported=SUMMARY_SCENE_VARIANTS,
        namespace=f"{prompt_key}.scene",
    )
    dataset = summary_dataset(
        statistic_kind=str(statistic_kind),
        scene_variant=str(scene_variant),
        params=params,
        instance_seed=int(seed),
        namespace=str(namespace),
        answer_target=str(answer_target),
    )
    rank_text = ordinal(int(dataset.trace["rank_n"])) if dataset.trace.get("rank_n") is not None else ""
    return package_single_series_plan(
        dataset=dataset,
        params=params,
        scene_variant=str(scene_variant),
        scene_variant_probabilities=scene_probs,
        prompt_bundle_id="charts_statistics_v1",
        prompt_scene_key="labeled_chart_statistics",
        prompt_task_key=f"summary_{answer_target}_query",
        prompt_key=str(prompt_key),
        dynamic_slots={
            "object_description": object_description(str(scene_variant)),
            "rank_n_ordinal": rank_text,
            "statistic_prompt": statistic_prompt(str(statistic_kind), rank_text=rank_text),
        },
        answer_type="string" if str(answer_target) == "label" else "integer",
        annotation_kind="point",
        relation_params={
            "statistic_kind": str(statistic_kind),
            "answer_target": str(answer_target),
            "rank_n": dataset.trace.get("rank_n"),
        },
        question_format="label_open" if str(answer_target) == "label" else "numeric_open",
        program_code=str(program_code),
        reasoning_load=float(reasoning_load),
    )


def build_counterfactual_plan(
    *,
    params: Mapping[str, Any],
    seed: int,
    namespace: str,
    operation: str,
    prompt_key: str,
    dynamic_slots: Mapping[str, Any],
    relation_params: Mapping[str, Any],
    program_code: str,
    reasoning_load: float,
) -> SingleSeriesTaskPlan:
    """Build counterfactual tasks from neutral operation codes."""

    scene_variant, scene_probs = resolve_scene_variant(
        params,
        instance_seed=int(seed),
        supported=COUNTERFACTUAL_SCENE_VARIANTS,
        namespace=f"{operation}.scene",
    )
    dataset = counterfactual_dataset(
        operation=str(operation),
        scene_variant=str(scene_variant),
        params=params,
        instance_seed=int(seed),
        namespace=str(namespace),
    )
    slots = {"object_description": object_description(str(scene_variant)), **_resolve_trace_mapping(dataset, dynamic_slots)}
    resolved_relations = _resolve_trace_mapping(dataset, relation_params)
    return package_single_series_plan(
        dataset=dataset,
        params=params,
        scene_variant=str(scene_variant),
        scene_variant_probabilities=scene_probs,
        prompt_bundle_id="charts_hypothetical_v1",
        prompt_scene_key="hypothetical_chart_counterfactual",
        prompt_task_key="counterfactual_value_query",
        prompt_key=str(prompt_key),
        dynamic_slots=slots,
        answer_type="integer",
        annotation_kind="point_set",
        relation_params=resolved_relations,
        question_format="numeric_open",
        program_code=str(program_code),
        reasoning_load=float(reasoning_load),
    )


def single_series_attempt_seed(instance_seed: int, public_task_id: str, attempt: int) -> int:
    return int(instance_seed) if int(attempt) == 0 else int(hash64(int(instance_seed), str(public_task_id), int(attempt)))


def materialize_single_series_plan(
    *,
    instance_seed: int,
    selected_query_id: str,
    query_probabilities: Mapping[str, float],
    plan: SingleSeriesTaskPlan,
) -> TaskOutput:
    """Render one public task plan and bind answer, annotation, prompt, and trace."""

    dataset = plan.dataset
    rendered = render_dataset(
        dataset=dataset,
        scene_variant=str(plan.scene_variant),
        params=dict(plan.params),
        instance_seed=int(instance_seed),
        hidden_labels=tuple(str(label) for label in dataset.trace.get("projected_labels", ())),
    )
    annotation_gt, projected_annotation, projected_labels = mark_annotation(
        rendered_scene=rendered.rendered_scene,
        labels=dataset.ordered_annotation_labels,
        annotation_kind=str(plan.annotation_kind),
        roles=dict(plan.annotation_roles),
    )
    prompt_artifacts: PromptTraceArtifacts = render_prompt_artifacts(
        bundle_id=str(plan.prompt_bundle_id),
        scene_key=str(plan.prompt_scene_key),
        task_key=str(plan.prompt_task_key),
        prompt_key=str(plan.prompt_key),
        dynamic_slot_values=dict(plan.dynamic_slots),
        instance_seed=int(instance_seed),
    )
    relation_params = {
        "query_id": str(selected_query_id),
        "scene_id": SCENE_ID,
        "query_id_probabilities": {str(key): float(value) for key, value in query_probabilities.items()},
        "program_code": str(plan.program_code),
        "scene_variant": str(plan.scene_variant),
        **dict(plan.relation_params),
    }
    trace_payload = {
        "scene_ir": {
            "scene_kind": "chart_single_series",
            "entities": [dict(entity) for entity in rendered.rendered_scene.entities],
            "relations": scene_relations(
                dataset=dataset,
                scene_variant=str(plan.scene_variant),
                relation_params=relation_params,
                annotation_labels=list(projected_labels),
            ),
        },
        "query_spec": build_prompt_query_spec(
            prompt_artifacts=prompt_artifacts,
            query_id=str(selected_query_id),
            params=relation_params,
        ),
        "render_spec": render_spec(rendered=rendered, scene_variant=str(plan.scene_variant)),
        "render_map": render_map(rendered),
        "execution_trace": execution_record(
            dataset=dataset,
            scene_variant=str(plan.scene_variant),
            relation_params=relation_params,
            annotation_labels=list(projected_labels),
            question_format=str(plan.question_format),
            annotation_type=str(annotation_gt.type),
            program_code=str(plan.program_code),
            reasoning_load=float(plan.reasoning_load),
            rendered=rendered,
        ),
        "witness_symbolic": witness_symbolic(
            dataset=dataset,
            annotation_labels=list(projected_labels),
            annotation_type=str(annotation_gt.type),
            relation_params=relation_params,
        ),
        "projected_annotation": dict(projected_annotation),
        "annotation_refs": annotation_refs(
            labels=list(projected_labels),
            annotation_value=annotation_gt.value,
            annotation_kind=str(annotation_gt.type),
        ),
        "background": dict(rendered.background_meta),
        "post_image_noise": dict(rendered.post_noise_meta),
    }
    return TaskOutput(
        prompt=str(prompt_artifacts.prompt),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
        answer_gt=plan.answer_gt,
        annotation_gt=annotation_gt,
        image=rendered.image,
        image_id="img0",
        trace_payload=dict(trace_payload),
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
        query_id=str(selected_query_id),
    )


def run_single_series_lifecycle(
    *,
    task: Any,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    default_query_id: str,
    build_plan: PlanBuilder,
) -> TaskOutput:
    selected_query_id, query_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=dict(params),
        supported_query_ids=task.supported_query_ids,
        default_query_id=str(default_query_id),
        task_id=str(task.task_id),
    )
    last_error: Exception | None = None
    for attempt in range(max(1, int(max_attempts))):
        attempt_seed = single_series_attempt_seed(int(instance_seed), str(task.task_id), int(attempt))
        try:
            plan = build_plan(
                {**dict(task_params), "_attempt_index": int(attempt)},
                int(attempt_seed),
                str(selected_query_id),
                dict(query_probabilities),
            )
            return materialize_single_series_plan(
                instance_seed=int(attempt_seed),
                selected_query_id=str(selected_query_id),
                query_probabilities=dict(query_probabilities),
                plan=plan,
            )
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"failed to generate {task.task_id}: {last_error}") from last_error


__all__ = [
    "SingleSeriesTaskPlan",
    "materialize_single_series_plan",
    "package_single_series_plan",
    "run_single_series_lifecycle",
]
