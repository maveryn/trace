"""Private neutral materialization helpers for boxplot chart tasks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from PIL import Image

from trace_tasks.core.seed import hash64
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.charts.boxplot.shared.annotations import keyed_point_artifacts, scalar_bbox_artifacts, scalar_point_artifacts
from trace_tasks.tasks.charts.boxplot.shared.defaults import SCENE_ID
from trace_tasks.tasks.charts.boxplot.shared.rendering import (
    build_trace_scaffold,
    box_bbox_map_for_labels,
    point_map_for_labels,
    render_paired_boxplot_panels,
    render_single_boxplot_scene,
)
from trace_tasks.tasks.charts.shared.chart_scene_types import BoxPlotSpec
from trace_tasks.tasks.shared.prompt_variants import PromptTraceArtifacts, build_prompt_query_spec


@dataclass(frozen=True)
class SingleBoxplotTaskPlan:
    """Task-owned semantic plan for a single-panel boxplot image."""

    boxplots: Sequence[BoxPlotSpec]
    params: Mapping[str, Any]
    mark_style: Mapping[str, Any]
    answer_gt: TypedValue
    answer_value: Any
    question_format: str
    role_to_label: Mapping[str, str]
    relations: Mapping[str, Any]
    prompt_artifacts: PromptTraceArtifacts
    annotation_kind: str = "point_map"


@dataclass(frozen=True)
class PairedBoxplotTaskPlan:
    """Task-owned semantic plan for a before/after boxplot image."""

    before_boxplots: Sequence[BoxPlotSpec]
    after_boxplots: Sequence[BoxPlotSpec]
    params: Mapping[str, Any]
    mark_style: Mapping[str, Any]
    before_title: str
    after_title: str
    answer_gt: TypedValue
    answer_value: Any
    question_format: str
    role_to_label: Mapping[str, str]
    relations: Mapping[str, Any]
    prompt_artifacts: PromptTraceArtifacts


@dataclass(frozen=True)
class MaterializedBoxplotTask:
    """Rendered payload assembled from one public task's semantic plan."""

    prompt: str
    answer_gt: TypedValue
    annotation_gt: TypedValue
    image: Image.Image
    trace_payload: dict[str, Any]
    query_id: str
    prompt_variants: dict[str, Any]


def boxplot_attempt_seed(instance_seed: int, attempt: int) -> int:
    """Return the neutral retry seed for boxplot-scene generation attempts."""

    return (
        int(instance_seed)
        if int(attempt) == 0
        else int(hash64(int(instance_seed), f"{SCENE_ID}.retry", int(attempt)))
    )


def materialize_single_boxplot_plan(
    *,
    instance_seed: int,
    selected_query_id: str,
    plan: SingleBoxplotTaskPlan,
) -> MaterializedBoxplotTask:
    """Render a single-panel plan and project task-bound role annotations."""

    artifacts = render_single_boxplot_scene(
        boxplots=plan.boxplots,
        params=plan.params,
        mark_style=plan.mark_style,
        instance_seed=int(instance_seed),
    )
    role_to_label = {str(role): str(label) for role, label in plan.role_to_label.items()}
    if str(plan.annotation_kind) == "bbox":
        label_to_bbox = box_bbox_map_for_labels(artifacts.rendered_scene, tuple(role_to_label.values()))
        role_to_bbox = {
            str(role): label_to_bbox[str(label)]
            for role, label in role_to_label.items()
        }
        annotation, witness_symbolic = scalar_bbox_artifacts(role_to_bbox, role_to_label)
    else:
        label_to_point = point_map_for_labels(artifacts.rendered_scene, tuple(role_to_label.values()))
        role_to_point = {
            str(role): label_to_point[str(label)]
            for role, label in role_to_label.items()
        }
        if str(plan.annotation_kind) == "point":
            annotation, witness_symbolic = scalar_point_artifacts(role_to_point, role_to_label)
        else:
            annotation, witness_symbolic = keyed_point_artifacts(role_to_point, role_to_label)
    trace_payload = build_trace_scaffold(
        artifacts=artifacts,
        relations=plan.relations,
        question_format=str(plan.question_format),
        witness_symbolic=witness_symbolic,
        projected_annotation=annotation.projected_annotation,
    )
    trace_payload["scene_ir"]["relations"]["query_id"] = str(selected_query_id)
    trace_payload["query_spec"] = build_prompt_query_spec(
        prompt_artifacts=plan.prompt_artifacts,
        query_id=str(selected_query_id),
        params={"query_id": str(selected_query_id), **dict(plan.relations)},
    )
    trace_payload["execution_trace"]["query_id"] = str(selected_query_id)
    return MaterializedBoxplotTask(
        prompt=str(plan.prompt_artifacts.prompt),
        answer_gt=plan.answer_gt,
        annotation_gt=annotation.annotation_gt,
        image=artifacts.rendered_scene.image,
        trace_payload=trace_payload,
        query_id=str(selected_query_id),
        prompt_variants=dict(plan.prompt_artifacts.prompt_variants),
    )


def materialize_paired_boxplot_plan(
    *,
    instance_seed: int,
    selected_query_id: str,
    plan: PairedBoxplotTaskPlan,
) -> MaterializedBoxplotTask:
    """Render a paired-panel plan and project task-bound role annotations."""

    artifacts = render_paired_boxplot_panels(
        before_boxplots=plan.before_boxplots,
        after_boxplots=plan.after_boxplots,
        params=plan.params,
        mark_style=plan.mark_style,
        before_title=str(plan.before_title),
        after_title=str(plan.after_title),
        instance_seed=int(instance_seed),
    )
    role_to_label = {str(role): str(label) for role, label in plan.role_to_label.items()}
    label_to_point = point_map_for_labels(artifacts.rendered_scene, tuple(role_to_label.values()))
    role_to_point = {
        str(role): label_to_point[str(label)]
        for role, label in role_to_label.items()
    }
    annotation, witness_symbolic = keyed_point_artifacts(role_to_point, role_to_label)
    trace_payload = build_trace_scaffold(
        artifacts=artifacts,
        relations=plan.relations,
        question_format=str(plan.question_format),
        witness_symbolic=witness_symbolic,
        projected_annotation=annotation.projected_annotation,
    )
    trace_payload["scene_ir"]["relations"]["query_id"] = str(selected_query_id)
    trace_payload["query_spec"] = build_prompt_query_spec(
        prompt_artifacts=plan.prompt_artifacts,
        query_id=str(selected_query_id),
        params={"query_id": str(selected_query_id), **dict(plan.relations)},
    )
    trace_payload["execution_trace"]["query_id"] = str(selected_query_id)
    return MaterializedBoxplotTask(
        prompt=str(plan.prompt_artifacts.prompt),
        answer_gt=plan.answer_gt,
        annotation_gt=annotation.annotation_gt,
        image=artifacts.rendered_scene.image,
        trace_payload=trace_payload,
        query_id=str(selected_query_id),
        prompt_variants=dict(plan.prompt_artifacts.prompt_variants),
    )


__all__ = [
    "MaterializedBoxplotTask",
    "PairedBoxplotTaskPlan",
    "SingleBoxplotTaskPlan",
    "boxplot_attempt_seed",
    "materialize_paired_boxplot_plan",
    "materialize_single_boxplot_plan",
]
