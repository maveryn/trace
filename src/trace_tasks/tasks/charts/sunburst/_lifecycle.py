"""Private neutral materialization helpers for sunburst chart tasks."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from PIL import Image

from trace_tasks.core.seed import hash64
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.charts.sunburst.shared.annotations import leaf_value_points
from trace_tasks.tasks.charts.sunburst.shared.rendering import render_scene
from trace_tasks.tasks.charts.sunburst.shared.sampling import hierarchy_rows, nodes_by_id
from trace_tasks.tasks.charts.sunburst.shared.state import SCENE_ID, SCENE_KIND, SCENE_VARIANT, SunburstTree
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.prompt_variants import PromptTraceArtifacts, build_prompt_query_spec


@dataclass(frozen=True)
class SunburstTaskPlan:
    """Task-owned semantic plan consumed by neutral sunburst rendering."""

    tree: SunburstTree
    prompt_artifacts: PromptTraceArtifacts
    answer_gt: TypedValue
    answer_value: Any
    question_format: str
    annotation_leaf_ids: Sequence[str]
    relations: Mapping[str, Any]
    witness_type: str
    witness_calculation: Mapping[str, Any]


@dataclass(frozen=True)
class MaterializedSunburstTask:
    """Rendered payload assembled from one public task's semantic plan."""

    prompt: str
    answer_gt: TypedValue
    annotation_gt: TypedValue
    image: Image.Image
    trace_payload: dict[str, Any]
    query_id: str
    prompt_variants: dict[str, Any]


def sunburst_attempt_seed(instance_seed: int, attempt: int) -> int:
    """Return the neutral retry seed for sunburst-scene generation attempts."""

    return (
        int(instance_seed)
        if int(attempt) == 0
        else int(hash64(int(instance_seed), "charts.sunburst.retry", int(attempt)))
    )


def materialize_sunburst_plan(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    selected_query_id: str,
    query_probabilities: Mapping[str, float],
    plan: SunburstTaskPlan,
) -> MaterializedSunburstTask:
    """Render one task-owned plan and build neutral trace payload sections.

    Public task files own objective sampling, answer binding, annotation-leaf
    selection, prompt slots, and retry policy. This helper only projects that
    completed semantic plan into rendered geometry and common trace envelopes.
    """

    rendered = render_scene(
        plan.tree,
        params=params,
        instance_seed=int(instance_seed),
        font_namespace="charts.sunburst.chart_font",
    )
    annotation_leaf_ids = tuple(str(leaf_id) for leaf_id in plan.annotation_leaf_ids)
    annotation_points = leaf_value_points(rendered, annotation_leaf_ids)
    node_lookup = nodes_by_id(plan.tree)
    relation_fields = {
        "query_id": str(selected_query_id),
        **dict(plan.relations),
        "annotation_node_ids": [str(leaf_id) for leaf_id in annotation_leaf_ids],
        "answer_value": plan.answer_value,
        "query_id_probabilities": dict(query_probabilities),
        "parent_count": int(len(plan.tree.parent_ids)),
        "subgroup_count": int(len(plan.tree.subgroup_ids)),
        "leaf_count": int(len(plan.tree.leaf_ids)),
        "parent_labels": [str(node_lookup[parent].label) for parent in plan.tree.parent_ids],
        **dict(plan.tree.generation_ranges),
    }
    projected_annotation = {
        "type": "point_set",
        "point_set": list(annotation_points),
        "annotation_node_ids": [str(leaf_id) for leaf_id in annotation_leaf_ids],
    }
    trace_payload = {
        "scene_ir": {
            "scene_kind": SCENE_KIND,
            "entities": [dict(entity) for entity in rendered.entities],
            "relations": dict(relation_fields),
        },
        "query_spec": build_prompt_query_spec(
            prompt_artifacts=plan.prompt_artifacts,
            query_id=str(selected_query_id),
            params=relation_fields,
        ),
        "render_spec": {
            "canvas_width": int(rendered.image.size[0]),
            "canvas_height": int(rendered.image.size[1]),
            "coord_space": "pixel",
            "scene_variant": SCENE_VARIANT,
            "background_style": dict(rendered.background_meta),
            "post_image_noise": dict(rendered.post_noise_meta),
            "font_assets": dict(rendered.font_assets),
            "chart_bbox_px": list(rendered.chart_bbox_px),
            **dict(rendered.render_meta),
        },
        "render_map": {
            "image_id": "img0",
            "chart_bbox_px": list(rendered.chart_bbox_px),
            "node_traces": [dict(trace) for trace in rendered.node_traces],
            "leaf_value_bbox_by_node_id": {
                str(node_id): list(bbox)
                for node_id, bbox in rendered.leaf_value_bbox_by_node_id.items()
            },
        },
        "execution_trace": {
            "query_id": str(selected_query_id),
            "answer_value": plan.answer_value,
            "question_format": str(plan.question_format),
            "hierarchy": hierarchy_rows(plan.tree),
            **dict(relation_fields),
        },
        "witness_symbolic": {
            "type": str(plan.witness_type),
            "query_id": str(selected_query_id),
            "answer_value": plan.answer_value,
            "annotation_node_ids": [str(leaf_id) for leaf_id in annotation_leaf_ids],
            "calculation": dict(plan.witness_calculation),
        },
        "projected_annotation": dict(projected_annotation),
    }
    return MaterializedSunburstTask(
        prompt=str(plan.prompt_artifacts.prompt),
        answer_gt=plan.answer_gt,
        annotation_gt=TypedValue(type="point_set", value=list(annotation_points)),
        image=rendered.image,
        trace_payload=trace_payload,
        query_id=str(selected_query_id),
        prompt_variants=dict(plan.prompt_artifacts.prompt_variants),
    )


def materialize_sunburst_plan_with_retries(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    selected_query_id: str,
    query_probabilities: Mapping[str, float],
    build_plan: Callable[[int, Mapping[str, Any], str], SunburstTaskPlan],
) -> MaterializedSunburstTask:
    """Retry task-owned plan construction, then return the rendered payload."""

    attempts = max(1, int(max_attempts))
    last_error: Exception | None = None
    for attempt in range(attempts):
        attempt_seed = sunburst_attempt_seed(int(instance_seed), int(attempt))
        try:
            plan = build_plan(int(attempt_seed), params, str(selected_query_id))
            return materialize_sunburst_plan(
                instance_seed=int(attempt_seed),
                params=params,
                selected_query_id=str(selected_query_id),
                query_probabilities=query_probabilities,
                plan=plan,
            )
        except ValueError as exc:
            last_error = exc
    raise RuntimeError(f"failed to materialize sunburst plan: {last_error}")


def sunburst_task_output_fields(materialized: MaterializedSunburstTask) -> dict[str, Any]:
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


def parent_total_relations(*, program_code: str, parent: Any, leaf_ids: Sequence[str], leaf_values: Sequence[int]) -> dict[str, Any]:
    """Build common trace relation fields for one parent-total computation."""

    return {
        "program_code": str(program_code),
        "parent_id": str(parent.node_id),
        "parent_label": str(parent.label),
        "parent_total": int(parent.value),
        "leaf_ids": [str(leaf_id) for leaf_id in leaf_ids],
        "leaf_values": [int(value) for value in leaf_values],
    }


def parent_extremum_relations(
    *,
    program_code: str,
    direction: str,
    answer_parent: Any,
    parent_totals: Mapping[str, int],
    answer_leaf_ids: Sequence[str],
    annotation_leaf_ids: Sequence[str],
) -> dict[str, Any]:
    """Build common trace relation fields for parent-total extremum tasks."""

    return {
        "program_code": str(program_code),
        "extremum": str(direction),
        "answer_parent_id": str(answer_parent.node_id),
        "answer_parent_label": str(answer_parent.label),
        "parent_totals": dict(parent_totals),
        "answer_leaf_ids": [str(leaf_id) for leaf_id in answer_leaf_ids],
        "leaf_ids": [str(leaf_id) for leaf_id in annotation_leaf_ids],
    }


def leaf_condition_relations(*, program_code: str, case: Mapping[str, Any], fields: Mapping[str, Any]) -> dict[str, Any]:
    """Build common trace relation fields for conditional leaf-count tasks."""

    return {
        "program_code": str(program_code),
        "parent_id": str(case["parent_id"]),
        "parent_label": str(case["parent_label"]),
        **dict(fields),
        "candidate_leaf_ids": [str(leaf_id) for leaf_id in case["leaf_ids"]],
        "candidate_leaf_values": [int(value) for value in case["leaf_values"]],
    }


def run_sunburst_public_task(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    selected_query_id: str,
    query_probabilities: Mapping[str, float],
    build_plan: Callable[[int, Mapping[str, Any], str], SunburstTaskPlan],
    build_output: Callable[[MaterializedSunburstTask], Any],
) -> Any:
    """Materialize a public-file-owned plan and call its output factory."""

    materialized = materialize_sunburst_plan_with_retries(
        instance_seed=int(instance_seed),
        params=params,
        max_attempts=int(max_attempts),
        selected_query_id=str(selected_query_id),
        query_probabilities=query_probabilities,
        build_plan=build_plan,
    )
    return build_output(materialized)


def run_sunburst_task(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    selected_query_id: str,
    query_probabilities: Mapping[str, float],
    build_plan: Callable[[int, Mapping[str, Any], str], SunburstTaskPlan],
) -> TaskOutput:
    """Materialize a public-file-owned plan and return the final task output."""

    materialized = materialize_sunburst_plan_with_retries(
        instance_seed=int(instance_seed),
        params=params,
        max_attempts=int(max_attempts),
        selected_query_id=str(selected_query_id),
        query_probabilities=query_probabilities,
        build_plan=build_plan,
    )
    return TaskOutput(**sunburst_task_output_fields(materialized))


def run_sunburst_task_from_public_class(
    task: Any,
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    build_plan: Callable[[int, Mapping[str, Any], str], SunburstTaskPlan],
) -> TaskOutput:
    """Apply global query selection, then run one public-file-owned plan."""

    selected_query_id, query_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=dict(params),
        supported_query_ids=tuple(str(value) for value in task.supported_query_ids),
        default_query_id=str(task.default_query_id),
        task_id=str(task.task_id),
    )
    return run_sunburst_task(
        instance_seed=int(instance_seed),
        params=task_params,
        max_attempts=int(max_attempts),
        selected_query_id=str(selected_query_id),
        query_probabilities=query_probabilities,
        build_plan=build_plan,
    )


__all__ = [
    "MaterializedSunburstTask",
    "SunburstTaskPlan",
    "leaf_condition_relations",
    "parent_extremum_relations",
    "parent_total_relations",
    "run_sunburst_public_task",
    "run_sunburst_task",
    "run_sunburst_task_from_public_class",
    "sunburst_task_output_fields",
]
