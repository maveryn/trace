"""Neutral lifecycle helpers for pictogram public tasks."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

from trace_tasks.core.seed import hash64
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import PromptTraceArtifacts, build_prompt_query_spec

from .shared.annotations import annotation_payload
from .shared.prompts import build_prompt_artifacts, dynamic_slots
from .shared.rendering import PictogramRenderResult, font_assets_payload, render_pictogram_dataset
from .shared.sampling import resolve_categories
from .shared.state import SCENE_ID, PictogramBaseSample, PictogramCategory, PictogramDataset, PictogramQuery


@dataclass(frozen=True)
class PictogramTaskPlan:
    dataset: PictogramDataset
    prompt_task_key: str
    prompt_query_key: str | None
    trace_params: dict[str, Any]


def pictogram_attempt_seed(instance_seed: int, unique_key: str, attempt: int) -> int:
    return int(instance_seed) if int(attempt) == 0 else int(hash64(int(instance_seed), str(unique_key), int(attempt)))


def build_pictogram_plan(
    *,
    dataset: PictogramDataset,
    prompt_task_key: str,
    prompt_query_key: str | None = None,
    trace_params: Mapping[str, Any] | None = None,
) -> PictogramTaskPlan:
    return PictogramTaskPlan(
        dataset=dataset,
        prompt_task_key=str(prompt_task_key),
        prompt_query_key=(str(prompt_query_key) if prompt_query_key is not None else None),
        trace_params=dict(trace_params or {}),
    )


def make_pictogram_query(
    *,
    selected: str,
    answer: Any,
    annotation_type: str,
    annotation_category_ids: tuple[str, ...],
    params: Mapping[str, Any],
    answer_type: str = "integer",
) -> PictogramQuery:
    typed_answer: Any = str(answer) if str(answer_type) == "string" else int(answer)
    return PictogramQuery(
        branch_id=str(selected),
        answer=typed_answer,
        answer_type=str(answer_type),
        annotation_type=str(annotation_type),
        annotation_category_ids=tuple(str(value) for value in annotation_category_ids),
        params=dict(params),
    )


def dataset_from_base(
    *,
    base: PictogramBaseSample,
    categories: tuple[PictogramCategory, ...],
    query: PictogramQuery,
    selected: str,
    probabilities: Mapping[str, float],
) -> PictogramDataset:
    return PictogramDataset(
        categories=tuple(categories),
        unit_scale=int(base.unit_scale),
        unit_scale_probabilities=dict(base.unit_scale_probabilities),
        branch_id=str(selected),
        branch_probabilities=dict(probabilities),
        scene_variant=str(base.scene_variant),
        scene_variant_probabilities=dict(base.scene_variant_probabilities),
        glyph_name=str(base.glyph_name),
        glyph_probabilities=dict(base.glyph_probabilities),
        query=query,
        title=str(base.title),
    )


def plan_from_mark_counts(
    *,
    base: PictogramBaseSample,
    mark_counts: list[int] | tuple[int, ...],
    query: PictogramQuery,
    selected: str,
    probabilities: Mapping[str, float],
    params: Mapping[str, Any],
    instance_seed: int,
    prompt_task_key: str,
    prompt_query_key: str | None = None,
) -> PictogramTaskPlan:
    categories = resolve_categories(
        mark_counts=mark_counts,
        unit_scale=int(base.unit_scale),
        params=params,
        instance_seed=int(instance_seed),
    )
    category_by_id = {str(category.category_id): category for category in categories}
    query_params = dict(query.params)
    if "target_category_id" in query_params:
        query_params["target_category_label"] = str(category_by_id[str(query_params["target_category_id"])].label)
    if "category_id_a" in query_params:
        query_params["category_label_a"] = str(category_by_id[str(query_params["category_id_a"])].label)
        query_params["category_label_b"] = str(category_by_id[str(query_params["category_id_b"])].label)
    query = replace(query, params=query_params)
    dataset = dataset_from_base(
        base=base,
        categories=tuple(categories),
        query=query,
        selected=str(selected),
        probabilities=probabilities,
    )
    return build_pictogram_plan(
        dataset=dataset,
        prompt_task_key=str(prompt_task_key),
        prompt_query_key=prompt_query_key,
    )


def _build_trace_payload(
    *,
    dataset: PictogramDataset,
    rendered: PictogramRenderResult,
    prompt_artifacts: PromptTraceArtifacts,
    annotation: Mapping[str, Any],
    trace_params: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble trace metadata from an already-bound task plan and rendered scene."""

    qparams = dict(dataset.query.params)
    category_id_to_label = dict(annotation["category_id_to_label"])
    totals_by_category = {category.label: int(category.total) for category in dataset.categories}
    mark_counts_by_category = {category.label: int(category.mark_count) for category in dataset.categories}
    answer_value: Any = str(dataset.query.answer) if str(dataset.query.answer_type) == "string" else int(dataset.query.answer)
    query_params = {
        "query_id": str(dataset.branch_id),
        "query_id_probabilities": dict(dataset.branch_probabilities),
        "scene_variant": str(dataset.scene_variant),
        "scene_variant_probabilities": dict(dataset.scene_variant_probabilities),
        "glyph_name": str(dataset.glyph_name),
        "glyph_probabilities": dict(dataset.glyph_probabilities),
        "unit_scale": int(dataset.unit_scale),
        "unit_scale_probabilities": dict(dataset.unit_scale_probabilities),
        "category_count": int(len(dataset.categories)),
        "answer_value": answer_value,
        **dict(qparams),
        **dict(trace_params),
    }
    return {
        "scene_ir": {
            "scene_kind": "pictogram",
            "entities": [dict(entity) for entity in rendered.rendered_scene.entities],
            "relations": {
                "query_id": str(dataset.branch_id),
                "scene_variant": str(dataset.scene_variant),
                "unit_scale": int(dataset.unit_scale),
                "answer_value": answer_value,
                "annotation_category_ids": list(annotation["category_ids"]),
            },
        },
        "query_spec": build_prompt_query_spec(
            prompt_artifacts=prompt_artifacts,
            query_id=str(dataset.branch_id),
            params=dict(query_params),
        ),
        "render_spec": {
            "scene_variant": str(dataset.scene_variant),
            "canvas_width": int(rendered.image.size[0]),
            "canvas_height": int(rendered.image.size[1]),
            "unit_scale": int(dataset.unit_scale),
            "glyph_name": str(dataset.glyph_name),
            "category_labels": [str(category.label) for category in dataset.categories],
            "font_assets": font_assets_payload(chart_font_family=rendered.chart_font_family),
            "background_style": dict(rendered.background_meta),
            "information_scene_style": dict(rendered.background_meta.get("information_scene_style", {})),
            "render_meta": dict(rendered.rendered_scene.render_meta),
            "post_image_noise": dict(rendered.post_noise_meta),
        },
        "render_map": {
            "plot_bbox_px": list(rendered.rendered_scene.plot_bbox_px),
            "legend_bbox_px": list(rendered.rendered_scene.legend_bbox_px),
            "category_bboxes_px": dict(rendered.rendered_scene.category_bboxes_px),
            "mark_bboxes_px": dict(rendered.rendered_scene.mark_bboxes_px),
        },
        "execution_trace": {
            "query_id": str(dataset.branch_id),
            "question_format": "pictogram_quantity",
            "scene_variant": str(dataset.scene_variant),
            "unit_scale": int(dataset.unit_scale),
            "category_count": int(len(dataset.categories)),
            "categories": [str(category.label) for category in dataset.categories],
            "category_id_to_label": dict(category_id_to_label),
            "mark_counts_by_category": dict(mark_counts_by_category),
            "totals_by_category": dict(totals_by_category),
            "answer_value": answer_value,
            "answer_type": str(dataset.query.answer_type),
            "annotation_type": str(dataset.query.annotation_type),
            "annotation_category_ids": list(annotation["category_ids"]),
            "annotation_labels": list(annotation["labels"]),
            **dict(qparams),
            **dict(trace_params),
        },
        "witness_symbolic": {
            "type": "pictogram_quantity_witness",
            "answer_value": answer_value,
            "annotation_type": str(dataset.query.annotation_type),
            "annotation_category_ids": list(annotation["category_ids"]),
        },
        "projected_annotation": dict(annotation["projected_annotation"]),
        "background": dict(rendered.background_meta),
        "post_image_noise": dict(rendered.post_noise_meta),
    }


def materialize_pictogram_plan(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    plan: PictogramTaskPlan,
) -> TaskOutput:
    rendered = render_pictogram_dataset(dataset=plan.dataset, params=dict(params), instance_seed=int(instance_seed))
    annotation = annotation_payload(dataset=plan.dataset, rendered=rendered)
    prompt_artifacts = build_prompt_artifacts(
        prompt_task_key=str(plan.prompt_task_key),
        prompt_query_key=plan.prompt_query_key,
        dynamic_slot_values=dynamic_slots(dataset=plan.dataset, scene_variant=str(plan.dataset.scene_variant)),
        instance_seed=int(instance_seed),
    )
    trace_payload = _build_trace_payload(
        dataset=plan.dataset,
        rendered=rendered,
        prompt_artifacts=prompt_artifacts,
        annotation=annotation,
        trace_params=dict(plan.trace_params),
    )
    return TaskOutput(
        prompt=str(prompt_artifacts.prompt),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
        answer_gt=TypedValue(
            type=str(plan.dataset.query.answer_type),
            value=(str(plan.dataset.query.answer) if str(plan.dataset.query.answer_type) == "string" else int(plan.dataset.query.answer)),
        ),
        annotation_gt=TypedValue(type=str(annotation["type"]), value=annotation["value"]),
        image=rendered.image,
        image_id="img0",
        trace_payload=trace_payload,
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
        query_id=str(plan.dataset.branch_id),
    )


def run_pictogram_task(task: Any, instance_seed: int, params: dict[str, Any], max_attempts: int) -> TaskOutput:
    selected, probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params={**dict(getattr(task, "task_param_defaults", {})), **dict(params)},
        supported_query_ids=tuple(str(value) for value in getattr(task, "supported_query_ids")),
        default_query_id=str(getattr(task, "default_query_id")),
        task_id=str(getattr(task, "task_id")),
    )
    last_error: Exception | None = None
    for attempt in range(max(1, int(max_attempts))):
        attempt_seed = pictogram_attempt_seed(int(instance_seed), str(getattr(task, "task_id")), int(attempt))
        try:
            plan = task._build_plan(dict(task_params), int(attempt_seed), str(selected), dict(probabilities))
            return materialize_pictogram_plan(
                params=dict(task_params),
                instance_seed=int(attempt_seed),
                plan=plan,
            )
        except ValueError as exc:
            if str(exc).startswith("unsupported scene_variant"):
                raise
            last_error = exc
    raise RuntimeError(f"failed to generate {getattr(task, 'task_id')}: {last_error}") from last_error
