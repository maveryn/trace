"""Private neutral materialization lifecycle for treemap chart tasks."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from PIL import Image

from trace_tasks.core.seed import hash64
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.charts.shared.information_style import make_chart_information_background, resolve_chart_information_style
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.charts.treemap.shared.annotations import (
    annotation_value_from_projection,
    bbox_set_projection,
    leaf_cell_boxes,
)
from trace_tasks.tasks.charts.treemap.shared.defaults import (
    POST_IMAGE_NOISE_DEFAULTS,
    chart_font_asset_metadata,
    select_chart_font_family,
)
from trace_tasks.tasks.charts.treemap.shared.rendering import (
    apply_treemap_information_style,
    render_treemap_scene,
    resolve_treemap_render_params,
    treemap_render_style_spec,
)
from trace_tasks.tasks.charts.treemap.shared.state import SCENE_ID, TreemapDataset
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import PromptTraceArtifacts, build_prompt_query_spec
from trace_tasks.tasks.shared.text_rendering import temporary_default_font_family


@dataclass(frozen=True)
class TreemapTaskPlan:
    dataset: TreemapDataset
    answer_gt: TypedValue
    answer_value: Any
    annotation_leaf_ids: Sequence[str]
    prompt_artifacts: PromptTraceArtifacts
    relations: Mapping[str, Any]
    witness_type: str
    witness_calculation: Mapping[str, Any]


@dataclass(frozen=True)
class MaterializedTreemapTask:
    prompt: str
    answer_gt: TypedValue
    annotation_gt: TypedValue
    image: Image.Image
    trace_payload: dict[str, Any]
    selected_branch: str
    prompt_variants: dict[str, Any]


def treemap_attempt_seed(instance_seed: int, attempt: int) -> int:
    return int(instance_seed) if int(attempt) == 0 else int(hash64(int(instance_seed), "charts.treemap.retry", int(attempt)))


def _parent_rows(dataset: TreemapDataset) -> list[dict[str, Any]]:
    return [
        {
            "parent_id": str(parent.parent_id),
            "label": str(parent.label),
            "value": int(parent.value),
            "leaf_ids": [str(leaf_id) for leaf_id in parent.leaf_ids],
        }
        for parent in dataset.parents
    ]


def _leaf_rows(dataset: TreemapDataset) -> list[dict[str, Any]]:
    return [
        {
            "leaf_id": str(leaf.leaf_id),
            "parent_id": str(leaf.parent_id),
            "parent_label": str(leaf.parent_label),
            "label": str(leaf.label),
            "value": int(leaf.value),
        }
        for leaf in dataset.leaves
    ]


def materialize_treemap_plan(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    selected_branch: str,
    branch_probabilities: Mapping[str, float],
    plan: TreemapTaskPlan,
) -> MaterializedTreemapTask:
    """Render one task-owned treemap plan and assemble common trace payload."""

    render_params = resolve_treemap_render_params(params, instance_seed=int(instance_seed))
    protected_colors = tuple(parent.color_rgb for parent in plan.dataset.parents) + tuple(
        leaf.color_rgb for leaf in plan.dataset.leaves
    )
    information_style, information_style_meta = resolve_chart_information_style(
        instance_seed=int(instance_seed),
        params=params,
        scene_id=SCENE_ID,
        protected_colors=protected_colors,
    )
    render_params = apply_treemap_information_style(render_params, information_style)
    background, background_meta = make_chart_information_background(
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
        style=information_style,
        instance_seed=int(instance_seed),
        namespace=f"charts.{SCENE_ID}.information_scene_background",
    )
    background_meta = dict(background_meta)
    background_meta["information_scene_style"] = dict(information_style_meta)
    chart_font_family = select_chart_font_family(instance_seed=int(instance_seed), params=params)
    with temporary_default_font_family(str(chart_font_family)):
        rendered = render_treemap_scene(
            background,
            dataset=plan.dataset,
            params=params,
            instance_seed=int(instance_seed),
            render_params=render_params,
        )
    image, post_noise_meta = apply_post_image_noise(
        rendered.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    annotation_boxes = leaf_cell_boxes(rendered, plan.annotation_leaf_ids)
    if not annotation_boxes:
        raise ValueError("treemap plan produced no projected annotation")
    projected_annotation = bbox_set_projection(annotation_boxes)
    annotation_gt = annotation_value_from_projection(projected_annotation)
    relation_fields = {
        "query_id": str(selected_branch),
        "query_id_probabilities": dict(branch_probabilities),
        "answer_value": plan.answer_value,
        "annotation_leaf_ids": [str(leaf_id) for leaf_id in plan.annotation_leaf_ids],
        "parent_count": int(len(plan.dataset.parents)),
        "leaf_count_per_parent": int(len(plan.dataset.parents[0].leaf_ids)) if plan.dataset.parents else 0,
        "parent_labels": [str(parent.label) for parent in plan.dataset.parents],
        "leaf_labels": sorted({str(leaf.label) for leaf in plan.dataset.leaves}),
        **dict(plan.dataset.generation_ranges),
        **dict(plan.relations),
    }
    trace_payload = {
        "scene_ir": {
            "scene_kind": "chart_treemap_composition",
            "entities": [dict(entity) for entity in rendered.entities],
            "relations": dict(relation_fields),
        },
        "query_spec": build_prompt_query_spec(
            prompt_artifacts=plan.prompt_artifacts,
            query_id=str(selected_branch),
            params=dict(relation_fields),
        ),
        "render_spec": {
            "canvas_width": int(image.size[0]),
            "canvas_height": int(image.size[1]),
            "coord_space": "pixel",
            "scene_variant": "treemap_composition",
            "background_style": dict(background_meta),
            "information_scene_style": dict(information_style_meta),
            "post_image_noise": dict(post_noise_meta),
            "font_assets": chart_font_asset_metadata(str(chart_font_family)),
            "chart_bbox_px": list(rendered.chart_bbox_px),
            "treemap_style": treemap_render_style_spec(render_params),
            **dict(rendered.render_meta),
        },
        "render_map": {
            "image_id": "img0",
            "chart_bbox_px": list(rendered.chart_bbox_px),
            "parent_traces": [dict(trace) for trace in rendered.parent_traces],
            "leaf_traces": [dict(trace) for trace in rendered.leaf_traces],
            "annotation_bbox_by_leaf_id": {
                str(leaf_id): list(bbox)
                for leaf_id, bbox in rendered.annotation_bbox_by_leaf_id.items()
            },
        },
        "execution_trace": {
            "query_id": str(selected_branch),
            "answer_value": plan.answer_value,
            "question_format": "numeric_open" if str(plan.answer_gt.type) == "integer" else "string_label",
            "parents": _parent_rows(plan.dataset),
            "leaves": _leaf_rows(plan.dataset),
            "annotation_leaf_ids": [str(leaf_id) for leaf_id in plan.annotation_leaf_ids],
            **dict(relation_fields),
        },
        "verifier_spec": {
            "answer_type": str(plan.answer_gt.type),
            "annotation_type": str(annotation_gt.type),
        },
        "witness_symbolic": {
            "type": str(plan.witness_type),
            "answer_value": plan.answer_value,
            "annotation_leaf_ids": [str(leaf_id) for leaf_id in plan.annotation_leaf_ids],
            "calculation": dict(plan.witness_calculation),
        },
        "projected_annotation": {
            **dict(projected_annotation),
            "annotation_leaf_ids": [str(leaf_id) for leaf_id in plan.annotation_leaf_ids],
        },
    }
    return MaterializedTreemapTask(
        prompt=str(plan.prompt_artifacts.prompt),
        answer_gt=plan.answer_gt,
        annotation_gt=annotation_gt,
        image=image,
        trace_payload=trace_payload,
        selected_branch=str(selected_branch),
        prompt_variants=dict(plan.prompt_artifacts.prompt_variants),
    )


def materialize_treemap_plan_with_retries(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    selected_branch: str,
    branch_probabilities: Mapping[str, float],
    build_plan: Callable[[int, Mapping[str, Any], str], TreemapTaskPlan],
) -> MaterializedTreemapTask:
    last_error: Exception | None = None
    for attempt in range(max(1, int(max_attempts))):
        attempt_seed = treemap_attempt_seed(int(instance_seed), int(attempt))
        attempt_params = {**dict(params), "_attempt_index": int(attempt)}
        try:
            plan = build_plan(int(attempt_seed), attempt_params, str(selected_branch))
            return materialize_treemap_plan(
                instance_seed=int(attempt_seed),
                params=attempt_params,
                selected_branch=str(selected_branch),
                branch_probabilities=branch_probabilities,
                plan=plan,
            )
        except ValueError as exc:
            last_error = exc
    raise RuntimeError(f"failed to materialize treemap plan: {last_error}")


def treemap_task_output_fields(materialized: MaterializedTreemapTask) -> dict[str, Any]:
    return {
        "prompt": materialized.prompt,
        "answer_gt": materialized.answer_gt,
        "annotation_gt": materialized.annotation_gt,
        "image": materialized.image,
        "image_id": "img0",
        "trace_payload": materialized.trace_payload,
        "task_versions": default_task_versions(),
        "scene_id": SCENE_ID,
        "query_id": materialized.selected_branch,
        "prompt_variants": materialized.prompt_variants,
    }


def run_treemap_task_from_public_class(
    task: Any,
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    build_plan: Callable[[int, Mapping[str, Any], str], TreemapTaskPlan],
) -> TaskOutput:
    selected_branch, branch_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=dict(params),
        supported_query_ids=task.supported_query_ids,
        default_query_id=task.default_query_id,
        task_id=task.task_id,
    )
    materialized = materialize_treemap_plan_with_retries(
        instance_seed=int(instance_seed),
        params=task_params,
        max_attempts=int(max_attempts),
        selected_branch=str(selected_branch),
        branch_probabilities=branch_probabilities,
        build_plan=build_plan,
    )
    return TaskOutput(**treemap_task_output_fields(materialized))


__all__ = [
    "TreemapTaskPlan",
    "run_treemap_task_from_public_class",
]
