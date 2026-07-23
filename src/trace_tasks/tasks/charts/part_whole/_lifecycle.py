"""Private lifecycle helpers for part-whole public tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from trace_tasks.core.seed import hash64
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import PromptTraceArtifacts, build_prompt_query_spec

from .shared.annotations import point_map_annotation
from .shared.defaults import SCENE_ID
from .shared.prompts import build_prompt_artifacts, dynamic_slots
from .shared.rendering import PartWholeRenderResult, render_part_whole_dataset
from .shared.sampling import base_extras, sample_categories, sample_category_count, value_bounds
from .shared.state import PartWholeDataset
from .shared.defaults import resolve_scene_variant, scene_axis_stride, shifted_cursor_params


@dataclass(frozen=True)
class PartWholeTaskPlan:
    """Task-owned semantic plan after sampling and prompt binding."""

    dataset: PartWholeDataset
    scene_variant: str
    scene_variant_probabilities: dict[str, float]
    prompt_artifacts: PromptTraceArtifacts
    trace_params: dict[str, Any]


@dataclass(frozen=True)
class PartWholeBaseSample:
    """Neutral scene/base-category sample shared by part-whole objectives."""

    scene_variant: str
    scene_variant_probabilities: dict[str, float]
    count_params: dict[str, Any]
    categories: tuple[Any, ...]
    category_count_range: tuple[int, int]
    value_min: int
    value_max: int
    base_extras: dict[str, Any]


def part_whole_attempt_seed(instance_seed: int, unique_key: str, attempt: int) -> int:
    """Return the deterministic retry seed for one public task attempt."""

    return int(instance_seed) if int(attempt) == 0 else int(hash64(int(instance_seed), str(unique_key), int(attempt)))


def sample_part_whole_base(
    *,
    params: dict[str, Any],
    instance_seed: int,
    namespace: str,
    min_key: str,
    max_key: str,
    fallback_min: int,
    fallback_max: int,
    even_only: bool = False,
) -> PartWholeBaseSample:
    """Sample scene variant, category count, category shares, and base trace fields.

    The caller owns the objective-specific count bounds and namespace; this
    helper only removes repeated visual-scene setup from public task files.
    """

    scene_variant, scene_probabilities = resolve_scene_variant(params, instance_seed=int(instance_seed))
    count_params = shifted_cursor_params(params, divisor=scene_axis_stride(params))
    category_count, category_range = sample_category_count(
        params=params,
        count_params=count_params,
        instance_seed=int(instance_seed),
        namespace=str(namespace),
        min_key=str(min_key),
        max_key=str(max_key),
        fallback_min=int(fallback_min),
        fallback_max=int(fallback_max),
        even_only=bool(even_only),
    )
    value_min, value_max = value_bounds(params)
    categories = sample_categories(
        category_count=int(category_count),
        value_min=int(value_min),
        value_max=int(value_max),
        instance_seed=int(instance_seed),
    )
    return PartWholeBaseSample(
        scene_variant=str(scene_variant),
        scene_variant_probabilities=dict(scene_probabilities),
        count_params=dict(count_params),
        categories=tuple(categories),
        category_count_range=tuple(int(value) for value in category_range),
        value_min=int(value_min),
        value_max=int(value_max),
        base_extras=base_extras(
            categories,
            category_count_range=category_range,
            value_min=int(value_min),
            value_max=int(value_max),
        ),
    )


def build_part_whole_plan(
    *,
    dataset: PartWholeDataset,
    base: PartWholeBaseSample,
    selected: str,
    instance_seed: int,
    trace_params: dict[str, Any] | None = None,
) -> PartWholeTaskPlan:
    """Attach prompt artifacts to a task-owned dataset and scene sample."""

    prompt_artifacts = build_prompt_artifacts(
        prompt_query_key=str(selected),
        dynamic_slot_values=dynamic_slots(dataset.trace_extras, scene_variant=str(base.scene_variant)),
        instance_seed=int(instance_seed),
    )
    return PartWholeTaskPlan(
        dataset=dataset,
        scene_variant=str(base.scene_variant),
        scene_variant_probabilities=dict(base.scene_variant_probabilities),
        prompt_artifacts=prompt_artifacts,
        trace_params=dict(trace_params or {}),
    )


def finish_part_whole_plan(
    *,
    base: PartWholeBaseSample,
    selected: str,
    instance_seed: int,
    answer_value: int,
    annotation_labels: tuple[str, ...],
    trace_extras: dict[str, Any],
) -> PartWholeTaskPlan:
    """Wrap task-computed answer/annotation bindings into the common plan."""

    dataset = PartWholeDataset(
        categories=tuple(base.categories),
        answer_value=int(answer_value),
        annotation_labels=tuple(str(label) for label in annotation_labels),
        trace_extras=dict(trace_extras),
    )
    return build_part_whole_plan(
        dataset=dataset,
        base=base,
        selected=str(selected),
        instance_seed=int(instance_seed),
    )


def _build_trace_payload(
    *,
    dataset: PartWholeDataset,
    rendered: PartWholeRenderResult,
    prompt_artifacts: PromptTraceArtifacts,
    selected: str,
    probabilities: dict[str, float],
    scene_variant: str,
    scene_variant_probabilities: dict[str, float],
    annotation: dict[str, Any],
    trace_params: dict[str, Any],
) -> dict[str, Any]:
    """Assemble trace sections from already-bound task and render state."""

    extras = dict(dataset.trace_extras)
    rendered_scene = rendered.rendered_scene
    values_by_label = {str(category.label): int(category.value) for category in dataset.categories}
    annotation_keys = list(annotation["keys"])
    prompt_params = {
        "query_id": str(selected),
        "query_id_probabilities": dict(probabilities),
        "scene_variant": str(scene_variant),
        "scene_variant_probabilities": dict(scene_variant_probabilities),
        "category_count": int(extras["category_count"]),
        "annotation_labels": [str(label) for label in dataset.annotation_labels],
        "annotation_keys": [str(key) for key in annotation_keys],
        **dict(trace_params),
    }
    return {
        "scene_ir": {
            "scene_kind": f"chart_{str(scene_variant)}_composition_share_arithmetic",
            "entities": [dict(entity) for entity in rendered_scene.entities],
            "relations": {
                "query_id": str(selected),
                "scene_variant": str(scene_variant),
                "category_labels": [str(category.label) for category in dataset.categories],
                **dict(trace_params),
            },
        },
        "query_spec": build_prompt_query_spec(
            prompt_artifacts=prompt_artifacts,
            query_id=str(selected),
            params=prompt_params,
        ),
        "render_spec": {
            "canvas_width": int(rendered.canvas_width),
            "canvas_height": int(rendered.canvas_height),
            "coord_space": "pixel",
            "scene_id": SCENE_ID,
            "scene_variant": str(scene_variant),
            "background_style": dict(rendered.background_meta),
            "information_scene_style": dict(rendered.background_meta.get("information_scene_style", {})),
            "post_image_noise": dict(rendered.post_noise_meta),
            "font_assets": dict(rendered.font_assets),
            "layout_jitter": dict(rendered_scene.layout_jitter_meta),
            "table_position": str(rendered_scene.layout_jitter_meta.get("table_position", "right")),
            "table_columns": int(rendered_scene.layout_jitter_meta.get("table_columns", 1)),
            "plot_bbox_px": list(rendered_scene.plot_bbox_px),
            "table_bbox_px": list(rendered_scene.table_bbox_px),
        },
        "render_map": {
            "image_id": "img0",
            "plot_bbox_px": list(rendered_scene.plot_bbox_px),
            "table_bbox_px": list(rendered_scene.table_bbox_px),
            "chart_traces": [dict(trace) for trace in rendered_scene.chart_traces],
            "category_traces": [dict(trace) for trace in rendered_scene.category_traces],
            "annotation_bbox_by_label": {
                str(label): list(bbox) for label, bbox in rendered_scene.annotation_bbox_by_label.items()
            },
            "annotation_point_by_label": {
                str(label): list(point) for label, point in rendered_scene.annotation_point_by_label.items()
            },
        },
        "execution_trace": {
            "query_id": str(selected),
            "scene_variant": str(scene_variant),
            "answer_value": int(dataset.answer_value),
            "category_count": int(extras["category_count"]),
            "category_count_range": list(extras["category_count_range"]),
            "category_values": {str(label): int(value) for label, value in values_by_label.items()},
            "categories": [
                {
                    "label": str(category.label),
                    "value": int(category.value),
                    "fill_rgb": [int(channel) for channel in category.color_rgb],
                }
                for category in dataset.categories
            ],
            "annotation_labels": [str(label) for label in dataset.annotation_labels],
            "annotation_keys": [str(key) for key in annotation_keys],
            "annotation_values": [int(value) for value in annotation["values"]],
            "question_format": "numeric_open",
            "query_id_probabilities": dict(probabilities),
            "scene_variant_probabilities": dict(scene_variant_probabilities),
            **dict(extras),
            **dict(trace_params),
        },
        "witness_symbolic": {
            "type": "composition_share_arithmetic",
            "query_id": str(selected),
            "answer_value": int(dataset.answer_value),
            "annotation_values": [int(value) for value in annotation["values"]],
            "calculation": {**dict(extras), **dict(trace_params)},
        },
        "projected_annotation": dict(annotation["projected"]),
    }


def materialize_part_whole_plan(
    *,
    selected: str,
    probabilities: dict[str, float],
    params: dict[str, Any],
    instance_seed: int,
    plan: PartWholeTaskPlan,
) -> TaskOutput:
    """Render one plan and assemble the final task output."""

    rendered = render_part_whole_dataset(
        dataset=plan.dataset,
        scene_variant=str(plan.scene_variant),
        params=dict(params),
        instance_seed=int(instance_seed),
    )
    annotation = point_map_annotation(dataset=plan.dataset, rendered_scene=rendered.rendered_scene)
    trace_payload = _build_trace_payload(
        dataset=plan.dataset,
        rendered=rendered,
        prompt_artifacts=plan.prompt_artifacts,
        selected=str(selected),
        probabilities=dict(probabilities),
        scene_variant=str(plan.scene_variant),
        scene_variant_probabilities=dict(plan.scene_variant_probabilities),
        annotation=annotation,
        trace_params=dict(plan.trace_params),
    )
    return TaskOutput(
        prompt=str(plan.prompt_artifacts.prompt),
        answer_gt=TypedValue(type="integer", value=int(plan.dataset.answer_value)),
        annotation_gt=TypedValue(type="point_map", value=dict(annotation["point_map"])),
        image=rendered.image,
        image_id="img0",
        trace_payload=trace_payload,
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
        query_id=str(selected),
        prompt_variants=dict(plan.prompt_artifacts.prompt_variants),
    )


def run_part_whole_task(task: Any, instance_seed: int, params: dict[str, Any], max_attempts: int) -> TaskOutput:
    """Select branch metadata, retry task-owned sampling, and render output."""

    selected, probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params={**dict(getattr(task, "task_param_defaults", {})), **dict(params)},
        supported_query_ids=tuple(str(value) for value in getattr(task, "supported_query_ids")),
        default_query_id=str(getattr(task, "default_query_id")),
        task_id=str(getattr(task, "task_id")),
    )
    last_error: Exception | None = None
    for attempt in range(max(1, int(max_attempts))):
        attempt_seed = part_whole_attempt_seed(int(instance_seed), str(getattr(task, "task_id")), int(attempt))
        try:
            plan = task._build_plan(dict(task_params), int(attempt_seed), str(selected), dict(probabilities))
            return materialize_part_whole_plan(
                selected=str(selected),
                probabilities=dict(probabilities),
                params=dict(task_params),
                instance_seed=int(attempt_seed),
                plan=plan,
            )
        except ValueError as exc:
            if str(exc).startswith("unsupported scene_variant"):
                raise
            last_error = exc
    raise RuntimeError(f"failed to generate {getattr(task, 'task_id')}: {last_error}") from last_error


__all__ = ["PartWholeTaskPlan", "build_part_whole_plan", "finish_part_whole_plan", "run_part_whole_task", "sample_part_whole_base"]
