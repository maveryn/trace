"""Private neutral materialization helpers for multiseries chart tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from PIL import Image

from trace_tasks.core.seed import hash64
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.charts.multiseries.shared.annotations import mark_annotation_payload
from trace_tasks.tasks.charts.multiseries.shared.data import (
    build_category_total_extremum_label_dataset,
    build_pair_equality_label_dataset,
    build_ratio_extremum_label_dataset,
    build_series_rank_at_category_label_dataset,
)
from trace_tasks.tasks.charts.multiseries.shared.defaults import DEFAULTS, GEN_DEFAULTS, SCENE_ID
from trace_tasks.tasks.charts.multiseries.shared.output import build_trace_payload, common_trace_fields
from trace_tasks.tasks.charts.multiseries.shared.prompts import (
    build_prompt_artifacts,
    extremum_prompt_slots,
    object_description,
    ratio_measure_prompt_slots,
)
from trace_tasks.tasks.charts.multiseries.shared.rendering import render_multiseries_dataset
from trace_tasks.tasks.charts.multiseries.shared.sampling import (
    balance_answer_label_for_indexed_probe,
    internal_ratio_variant,
    params_for_variant_family,
    resolve_extremum_direction,
    resolve_scene_variant,
    support_params_for_axis_cycle,
)
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import PromptTraceArtifacts, build_prompt_query_spec


@dataclass(frozen=True)
class MultiseriesTaskPlan:
    """Task-owned semantic plan consumed by neutral multiseries rendering."""

    values_by_category: Mapping[str, Mapping[str, int]]
    trace_extras: Mapping[str, Any]
    scene_variant: str
    prompt_artifacts: PromptTraceArtifacts
    answer_gt: TypedValue
    answer_value: Any
    annotation_category_labels: Sequence[str]
    annotation_series_labels: Sequence[str] | Mapping[str, Sequence[str]]
    variant_family: str
    internal_query_id: str
    question_format: str
    relations_extra: Mapping[str, Any]
    query_params_extra: Mapping[str, Any]
    execution_extra: Mapping[str, Any]
    witness_symbolic: Mapping[str, Any]


@dataclass(frozen=True)
class MaterializedMultiseriesTask:
    """Rendered payload assembled from one public task's semantic plan."""

    prompt: str
    answer_gt: TypedValue
    annotation_gt: TypedValue
    image: Image.Image
    trace_payload: dict[str, Any]
    query_id: str
    prompt_variants: dict[str, Any]


def multiseries_attempt_seed(instance_seed: int, attempt: int) -> int:
    """Return the neutral retry seed for multiseries-scene generation attempts."""

    return (
        int(instance_seed)
        if int(attempt) == 0
        else int(hash64(int(instance_seed), "charts.multiseries.retry", int(attempt)))
    )


def materialize_multiseries_plan(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    selected_query_id: str,
    plan: MultiseriesTaskPlan,
) -> MaterializedMultiseriesTask:
    """Render one task-owned semantic plan and build trace payload sections."""

    rendered = render_multiseries_dataset(
        values_by_category=plan.values_by_category,
        trace_extras=plan.trace_extras,
        scene_variant=str(plan.scene_variant),
        params=params,
        instance_seed=int(instance_seed),
    )
    annotation_points, projected_annotation = mark_annotation_payload(
        rendered_scene=rendered.rendered_scene,
        category_labels=plan.annotation_category_labels,
        series_labels=plan.annotation_series_labels,
    )
    relation_payload = {
        "query_id": str(selected_query_id),
        "scene_variant": str(plan.scene_variant),
        "variant_family": str(plan.variant_family),
        "internal_query_id": str(plan.internal_query_id),
        **dict(plan.relations_extra),
    }
    prompt_params = {
        "query_id": str(selected_query_id),
        "scene_variant": str(plan.scene_variant),
        "variant_family": str(plan.variant_family),
        "internal_query_id": str(plan.internal_query_id),
        "series_count": int(plan.trace_extras["series_count"]),
        "series_count_range": list(plan.trace_extras["series_count_range"]),
        "category_count": int(plan.trace_extras["category_count"]),
        "category_count_range": list(plan.trace_extras["category_count_range"]),
        "queried_series_labels": list(plan.trace_extras.get("queried_series_labels", [])),
        **dict(plan.query_params_extra),
    }
    execution_base = {
        "query_id": str(selected_query_id),
        "scene_variant": str(plan.scene_variant),
        "variant_family": str(plan.variant_family),
        "internal_query_id": str(plan.internal_query_id),
    }
    trace_payload = build_trace_payload(
        result=rendered,
        prompt_query_spec=build_prompt_query_spec(
            prompt_artifacts=plan.prompt_artifacts,
            query_id=str(selected_query_id),
            params=prompt_params,
        ),
        relation_payload=relation_payload,
        execution_base=execution_base,
        scene_variant=str(plan.scene_variant),
        question_format=str(plan.question_format),
        trace_extras=plan.trace_extras,
        execution_extra=plan.execution_extra,
        witness_symbolic=plan.witness_symbolic,
        projected_annotation=projected_annotation,
    )
    return MaterializedMultiseriesTask(
        prompt=str(plan.prompt_artifacts.prompt),
        answer_gt=plan.answer_gt,
        annotation_gt=TypedValue(type="point_map", value=dict(annotation_points)),
        image=rendered.image,
        trace_payload=trace_payload,
        query_id=str(selected_query_id),
        prompt_variants=dict(plan.prompt_artifacts.prompt_variants),
    )


def materialize_multiseries_plan_with_retries(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    selected_query_id: str,
    build_plan: Callable[[int, Mapping[str, Any], str], MultiseriesTaskPlan],
) -> MaterializedMultiseriesTask:
    """Retry task-owned plan construction, then return rendered payload."""

    attempts = max(1, int(max_attempts))
    last_error: Exception | None = None
    for attempt in range(attempts):
        attempt_seed = multiseries_attempt_seed(int(instance_seed), int(attempt))
        try:
            plan = build_plan(int(attempt_seed), params, str(selected_query_id))
            return materialize_multiseries_plan(
                instance_seed=int(attempt_seed),
                params=params,
                selected_query_id=str(selected_query_id),
                plan=plan,
            )
        except ValueError as exc:
            last_error = exc
    raise RuntimeError(f"failed to materialize multiseries plan: {last_error}")


def multiseries_task_output_fields(materialized: MaterializedMultiseriesTask) -> dict[str, Any]:
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


def build_label_selection_plan(
    *,
    values_by_category: Mapping[str, Mapping[str, int]],
    trace_extras: Mapping[str, Any],
    scene_variant: str,
    prompt_artifacts: PromptTraceArtifacts,
    answer_label: str,
    annotation_values: Sequence[int],
    annotation_category_labels: Sequence[str],
    annotation_series_labels: Sequence[str] | Mapping[str, Sequence[str]],
    variant_family: str,
    internal_query_id: str,
    optional_trace: Mapping[str, Any],
) -> MultiseriesTaskPlan:
    """Build the common label-answer plan after a task binds its objective."""

    relations, query_params, execution = common_trace_fields(
        answer_label=str(answer_label),
        annotation_values=annotation_values,
        queried_series_labels=trace_extras["queried_series_labels"],
        value_range=trace_extras["value_range"],
        answer_type="string",
        extra=optional_trace,
    )
    return MultiseriesTaskPlan(
        values_by_category=values_by_category,
        trace_extras=trace_extras,
        scene_variant=str(scene_variant),
        prompt_artifacts=prompt_artifacts,
        answer_gt=TypedValue(type="string", value=str(answer_label)),
        answer_value=str(answer_label),
        annotation_category_labels=[str(label) for label in annotation_category_labels],
        annotation_series_labels=annotation_series_labels,
        variant_family=str(variant_family),
        internal_query_id=str(internal_query_id),
        question_format="label_open",
        relations_extra=relations,
        query_params_extra=query_params,
        execution_extra=execution,
        witness_symbolic={
            "type": "numeric_sequence",
            "value": [int(value) for value in annotation_values],
            "answer_label": str(answer_label),
        },
    )


def selected_trace_fields(
    trace_extras: Mapping[str, Any],
    keys: Sequence[str],
    *,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Copy named trace fields and merge task-owned semantic axis metadata."""

    payload = {str(key): trace_extras[str(key)] for key in keys if str(key) in trace_extras}
    if extra:
        payload.update(dict(extra))
    return payload


def build_ranked_ratio_label_plan(
    *,
    ratio_measure: str,
    prompt_query_key: str,
    namespace: str,
    annotation_series_mode: str,
    trace_keys: Sequence[str],
    instance_seed: int,
    params: Mapping[str, Any],
) -> MultiseriesTaskPlan:
    """Build the shared ranked-ratio skeleton for ratio-style label tasks.

    Public task files choose the ratio measure, prompt key, namespace, trace
    fields, and annotation series mode. This helper handles the common ranked
    ratio mechanics without branching on public task id or query id.
    """

    extremum_direction, extremum_probabilities = resolve_extremum_direction(params, instance_seed=int(instance_seed))
    dataset_params = support_params_for_axis_cycle(
        {**dict(params), "ratio_measure": str(ratio_measure)},
        probabilities=extremum_probabilities,
        supported_values=("largest", "smallest"),
        explicit_key="extremum_direction",
        weights_key="extremum_direction_weights",
        balance_flag_key="balanced_extremum_direction_sampling",
    )
    internal_query_id = internal_ratio_variant(
        ratio_measure=str(ratio_measure),
        extremum_direction=str(extremum_direction),
    )
    scene_variant, scene_variant_probabilities = resolve_scene_variant(params, instance_seed=int(instance_seed))
    values_by_category, answer_label, annotation_values, trace_extras = build_ratio_extremum_label_dataset(
        variant_key=str(internal_query_id),
        params=params_for_variant_family(dataset_params, family="ratio"),
        instance_seed=int(instance_seed),
        gen_defaults=GEN_DEFAULTS,
        defaults=DEFAULTS,
        namespace=str(namespace),
    )
    values_by_category, answer_label, trace_extras = balance_answer_label_for_indexed_probe(
        namespace=str(namespace),
        params=dataset_params,
        instance_seed=int(instance_seed),
        values_by_category=values_by_category,
        answer_label=str(answer_label),
        trace_extras=trace_extras,
    )
    answer_rank = int(trace_extras["answer_rank"])
    prompt_artifacts = build_prompt_artifacts(
        prompt_query_key=str(prompt_query_key),
        dynamic_slots={
            "object_description": object_description(str(scene_variant)),
            "rank": answer_rank,
            "target_series": str(trace_extras.get("target_series_label", "")),
            "numerator_series": str(trace_extras.get("numerator_series_label", "")),
            "denominator_series": str(trace_extras.get("denominator_series_label", "")),
            **extremum_prompt_slots(str(extremum_direction), answer_rank=answer_rank),
            **ratio_measure_prompt_slots(
                str(ratio_measure),
                target_series=str(trace_extras.get("target_series_label", "")),
                numerator_series=str(trace_extras.get("numerator_series_label", "")),
                denominator_series=str(trace_extras.get("denominator_series_label", "")),
            ),
        },
        instance_seed=int(instance_seed),
    )
    optional_trace = selected_trace_fields(
        trace_extras,
        trace_keys,
        extra={
            "ratio_measure": str(ratio_measure),
            "extremum_direction": str(extremum_direction),
            "extremum_direction_probabilities": dict(extremum_probabilities),
            "scene_variant_probabilities": dict(scene_variant_probabilities),
        },
    )
    if str(annotation_series_mode) == "all_series":
        annotation_series_labels: Sequence[str] | Mapping[str, Sequence[str]] = trace_extras["series_labels"]
    elif str(annotation_series_mode) == "queried_series":
        annotation_series_labels = trace_extras["queried_series_labels"]
    else:
        raise ValueError(f"unsupported annotation_series_mode: {annotation_series_mode}")
    return build_label_selection_plan(
        values_by_category=values_by_category,
        trace_extras=trace_extras,
        scene_variant=str(scene_variant),
        prompt_artifacts=prompt_artifacts,
        answer_label=str(answer_label),
        annotation_values=annotation_values,
        annotation_category_labels=[str(answer_label)],
        annotation_series_labels=annotation_series_labels,
        variant_family="ratio",
        internal_query_id=str(internal_query_id),
        optional_trace=optional_trace,
    )


def build_ranked_extremum_label_plan(
    *,
    dataset_kind: str,
    prompt_query_key: str,
    namespace: str,
    variant_key: str,
    variant_family: str,
    trace_keys: Sequence[str],
    annotation_category_source: str,
    include_target_category_slot: bool,
    instance_seed: int,
    params: Mapping[str, Any],
) -> MultiseriesTaskPlan:
    """Build a label plan for single-axis ranked extremum tasks."""

    extremum_direction, extremum_probabilities = resolve_extremum_direction(params, instance_seed=int(instance_seed))
    dataset_params = support_params_for_axis_cycle(
        params,
        probabilities=extremum_probabilities,
        supported_values=("largest", "smallest"),
        explicit_key="extremum_direction",
        weights_key="extremum_direction_weights",
        balance_flag_key="balanced_extremum_direction_sampling",
    )
    scene_variant, scene_variant_probabilities = resolve_scene_variant(params, instance_seed=int(instance_seed))
    builder_by_kind = {
        "category_total": build_category_total_extremum_label_dataset,
        "series_rank": build_series_rank_at_category_label_dataset,
    }
    builder = builder_by_kind[str(dataset_kind)]
    values_by_category, answer_label, annotation_values, trace_extras = builder(
        variant_key=str(variant_key),
        extremum_direction=str(extremum_direction),
        params=params_for_variant_family(dataset_params, family=str(variant_family)),
        instance_seed=int(instance_seed),
        gen_defaults=GEN_DEFAULTS,
        defaults=DEFAULTS,
        namespace=str(namespace),
    )
    if str(annotation_category_source) == "answer_label":
        values_by_category, answer_label, trace_extras = balance_answer_label_for_indexed_probe(
            namespace=str(namespace),
            params=dataset_params,
            instance_seed=int(instance_seed),
            values_by_category=values_by_category,
            answer_label=str(answer_label),
            trace_extras=trace_extras,
        )
    answer_rank = int(trace_extras["answer_rank"])
    dynamic_slots = {
        "object_description": object_description(str(scene_variant)),
        "rank": answer_rank,
        **extremum_prompt_slots(str(extremum_direction), answer_rank=answer_rank),
    }
    if include_target_category_slot:
        dynamic_slots["target_category"] = str(trace_extras["target_category_label"])
    prompt_artifacts = build_prompt_artifacts(
        prompt_query_key=str(prompt_query_key),
        dynamic_slots=dynamic_slots,
        instance_seed=int(instance_seed),
    )
    optional_trace = selected_trace_fields(
        trace_extras,
        trace_keys,
        extra={
            "extremum_direction": str(extremum_direction),
            "extremum_direction_probabilities": dict(extremum_probabilities),
            "scene_variant_probabilities": dict(scene_variant_probabilities),
        },
    )
    if str(annotation_category_source) == "answer_label":
        annotation_categories = [str(answer_label)]
    elif str(annotation_category_source) == "target_category_label":
        annotation_categories = [str(trace_extras["target_category_label"])]
    else:
        raise ValueError(f"unsupported annotation_category_source: {annotation_category_source}")
    return build_label_selection_plan(
        values_by_category=values_by_category,
        trace_extras=trace_extras,
        scene_variant=str(scene_variant),
        prompt_artifacts=prompt_artifacts,
        answer_label=str(answer_label),
        annotation_values=annotation_values,
        annotation_category_labels=annotation_categories,
        annotation_series_labels=trace_extras["series_labels"],
        variant_family=str(variant_family),
        internal_query_id=str(variant_key),
        optional_trace=optional_trace,
    )


def build_pair_equality_label_plan(
    *,
    namespace: str,
    prompt_query_key: str,
    instance_seed: int,
    params: Mapping[str, Any],
) -> MultiseriesTaskPlan:
    """Build the exact-equality label plan for two queried series."""

    scene_variant, scene_variant_probabilities = resolve_scene_variant(params, instance_seed=int(instance_seed))
    values_by_category, answer_label, annotation_values, trace_extras = build_pair_equality_label_dataset(
        variant_key="pair_equality",
        params=params_for_variant_family(params, family="equality"),
        instance_seed=int(instance_seed),
        gen_defaults=GEN_DEFAULTS,
        defaults=DEFAULTS,
        namespace=str(namespace),
    )
    values_by_category, answer_label, trace_extras = balance_answer_label_for_indexed_probe(
        namespace=str(namespace),
        params=params,
        instance_seed=int(instance_seed),
        values_by_category=values_by_category,
        answer_label=str(answer_label),
        trace_extras=trace_extras,
    )
    prompt_artifacts = build_prompt_artifacts(
        prompt_query_key=str(prompt_query_key),
        dynamic_slots={
            "object_description": object_description(str(scene_variant)),
            "left_series": str(trace_extras["left_series_label"]),
            "right_series": str(trace_extras["right_series_label"]),
        },
        instance_seed=int(instance_seed),
    )
    return build_label_selection_plan(
        values_by_category=values_by_category,
        trace_extras=trace_extras,
        scene_variant=str(scene_variant),
        prompt_artifacts=prompt_artifacts,
        answer_label=str(answer_label),
        annotation_values=annotation_values,
        annotation_category_labels=[str(answer_label)],
        annotation_series_labels=trace_extras["queried_series_labels"],
        variant_family="equality",
        internal_query_id="pair_equality",
        optional_trace=selected_trace_fields(
            trace_extras,
            (
                "left_series_label",
                "right_series_label",
                "answer_rank",
                "answer_score",
                "answer_equal_value",
                "derived_metric",
                "rank_order",
                "ranked_category_labels",
                "equality_by_category",
                "derived_values_by_category",
            ),
            extra={"scene_variant_probabilities": dict(scene_variant_probabilities)},
        ),
    )


def run_multiseries_public_task(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    selected_query_id: str,
    build_plan: Callable[[int, Mapping[str, Any], str], MultiseriesTaskPlan],
    build_output: Callable[[MaterializedMultiseriesTask], Any],
) -> Any:
    """Materialize a public-file-owned plan and call its output factory."""

    materialized = materialize_multiseries_plan_with_retries(
        instance_seed=int(instance_seed),
        params=params,
        max_attempts=int(max_attempts),
        selected_query_id=str(selected_query_id),
        build_plan=build_plan,
    )
    return build_output(materialized)


def run_configured_multiseries_task(
    task: Any,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
) -> Any:
    """Run a public multiseries task from class metadata and task-owned hooks."""

    selected_query_id, _probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params={**dict(task.task_param_defaults), **dict(params)},
        supported_query_ids=tuple(task.supported_query_ids),
        default_query_id=str(task.default_query_id),
        task_id=str(task.task_id),
    )
    materialized = materialize_multiseries_plan_with_retries(
        instance_seed=int(instance_seed),
        params=task_params,
        max_attempts=int(max_attempts),
        selected_query_id=str(selected_query_id),
        build_plan=task._build_plan,
    )
    return TaskOutput(**multiseries_task_output_fields(materialized))


__all__ = [
    "MaterializedMultiseriesTask",
    "MultiseriesTaskPlan",
    "build_label_selection_plan",
    "build_pair_equality_label_plan",
    "build_ranked_extremum_label_plan",
    "build_ranked_ratio_label_plan",
    "materialize_multiseries_plan",
    "materialize_multiseries_plan_with_retries",
    "multiseries_attempt_seed",
    "multiseries_task_output_fields",
    "run_configured_multiseries_task",
    "run_multiseries_public_task",
    "selected_trace_fields",
]
