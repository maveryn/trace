"""Private neutral materialization helpers for area chart tasks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from PIL import Image

from trace_tasks.core.seed import hash64
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.charts.area.shared.annotations import point_set_artifacts, points_for_pairs
from trace_tasks.tasks.charts.area.shared.defaults import SCENE_ID
from trace_tasks.tasks.charts.area.shared.output import build_trace_scaffold, values_by_series
from trace_tasks.tasks.charts.area.shared.rendering import render_area_scene
from trace_tasks.tasks.shared.prompt_variants import PromptTraceArtifacts, build_prompt_query_spec


@dataclass(frozen=True)
class AreaTaskPlan:
    """Task-owned semantic plan consumed by neutral area-scene rendering."""

    x_labels: tuple[str, ...]
    series_labels: tuple[str, ...]
    series_values: Mapping[str, Sequence[int]]
    stacked: bool
    scene_variant: str
    answer_gt: TypedValue
    answer_value: Any
    question_format: str
    annotation_pairs: tuple[tuple[str, str], ...]
    relations: Mapping[str, Any]
    prompt_artifacts: PromptTraceArtifacts


@dataclass(frozen=True)
class MaterializedAreaTask:
    """Rendered payload assembled from one public task's semantic plan."""

    prompt: str
    answer_gt: TypedValue
    annotation_gt: TypedValue
    image: Image.Image
    trace_payload: dict[str, Any]
    query_id: str
    prompt_variants: dict[str, Any]


def area_attempt_seed(instance_seed: int, attempt: int) -> int:
    """Return the neutral retry seed for area-scene generation attempts."""

    return (
        int(instance_seed)
        if int(attempt) == 0
        else int(hash64(int(instance_seed), f"{SCENE_ID}.retry", int(attempt)))
    )


def materialize_area_plan(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    selected_query_id: str,
    plan: AreaTaskPlan,
) -> MaterializedAreaTask:
    """Render one task-owned plan and build neutral trace payload sections.

    Public task files own the objective, answer binding, annotation-pair
    binding, prompt slots, retry loop, and final output construction. This
    helper only projects the already-selected semantic plan into rendered image
    geometry and trace sections.
    """

    rendered = render_area_scene(
        x_labels=plan.x_labels,
        series_labels=plan.series_labels,
        series_values=plan.series_values,
        stacked=bool(plan.stacked),
        highlight_points=plan.annotation_pairs,
        params=params,
        instance_seed=int(instance_seed),
    )
    annotation_points = points_for_pairs(rendered.panel, plan.annotation_pairs)
    annotation_gt, witness_symbolic, projected_annotation = point_set_artifacts(annotation_points)
    series_values_by_label = values_by_series(
        series_labels=plan.series_labels,
        series_values=plan.series_values,
    )
    trace_payload = build_trace_scaffold(
        rendered=rendered,
        scene_variant=str(plan.scene_variant),
        relations=plan.relations,
        answer_value=plan.answer_value,
        question_format=str(plan.question_format),
        series_values_by_label=series_values_by_label,
        annotation_pairs=plan.annotation_pairs,
        witness_symbolic=witness_symbolic,
        projected_annotation=projected_annotation,
    )
    trace_payload["query_spec"] = build_prompt_query_spec(
        prompt_artifacts=plan.prompt_artifacts,
        query_id=str(selected_query_id),
        params=plan.relations,
    )
    return MaterializedAreaTask(
        prompt=str(plan.prompt_artifacts.prompt),
        answer_gt=plan.answer_gt,
        annotation_gt=annotation_gt,
        image=rendered.image,
        trace_payload=trace_payload,
        query_id=str(selected_query_id),
        prompt_variants=dict(plan.prompt_artifacts.prompt_variants),
    )


__all__ = [
    "AreaTaskPlan",
    "MaterializedAreaTask",
    "area_attempt_seed",
    "materialize_area_plan",
]
