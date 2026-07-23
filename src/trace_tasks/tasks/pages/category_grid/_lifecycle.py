"""Scene-private response assembly for category-grid public tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    build_prompt_query_spec,
    build_prompt_trace_artifacts,
    render_task_prompt_variants,
)

from .shared.annotations import make_category_payload
from .shared.defaults import DOMAIN, PROMPT_BUNDLE, PROMPT_SCENE_KEY, PROMPT_TASK_KEY, SCENE
from .shared.rendering import render_category_grid_case
from .shared.state import CategoryGridCase, RenderedCategoryGridBundle


@dataclass(frozen=True)
class CategoryGridPromptBinding:
    """Task-owned prompt key and dynamic prompt slots."""

    prompt_branch_key: str
    dynamic_slots: Mapping[str, Any]


@dataclass(frozen=True)
class CategoryGridAnswerBinding:
    """Task-owned answer, annotation, and trace payload fields."""

    answer_gt: TypedValue
    annotation_gt: TypedValue
    selected_branch: str
    branch_probabilities: Mapping[str, float]
    target_payload: Mapping[str, Any]
    question_format: str


CaseFactory = Callable[..., CategoryGridCase]
BindingFactory = Callable[
    [str, Mapping[str, float], CategoryGridCase, RenderedCategoryGridBundle],
    tuple[CategoryGridPromptBinding, CategoryGridAnswerBinding],
]


def select_public_branch(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    supported: tuple[str, ...],
    default: str,
    public_task: str,
) -> tuple[str, Dict[str, float], Dict[str, Any]]:
    """Resolve the caller-selected public branch through the shared policy."""

    branch, probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=tuple(str(value) for value in supported),
        default_query_id=str(default),
        task_id=str(public_task),
    )
    return str(branch), dict(probabilities), dict(task_params)


def string_binding(
    *,
    annotation_kind: str,
    annotation_value: Any,
    selected_branch: str,
    branch_probabilities: Mapping[str, float],
    answer_value: str,
    target_payload: Mapping[str, Any],
    question_format: str,
) -> CategoryGridAnswerBinding:
    """Build a string-answer binding from task-owned annotation data."""

    return CategoryGridAnswerBinding(
        answer_gt=TypedValue(type="string", value=str(answer_value)),
        annotation_gt=TypedValue(type=str(annotation_kind), value=annotation_value),
        selected_branch=str(selected_branch),
        branch_probabilities=dict(branch_probabilities),
        target_payload=dict(target_payload),
        question_format=str(question_format),
    )


def integer_binding(
    *,
    annotation_kind: str,
    annotation_value: Any,
    selected_branch: str,
    branch_probabilities: Mapping[str, float],
    answer_value: int,
    target_payload: Mapping[str, Any],
    question_format: str,
) -> CategoryGridAnswerBinding:
    """Build an integer-answer binding from task-owned annotation data."""

    return CategoryGridAnswerBinding(
        answer_gt=TypedValue(type="integer", value=int(answer_value)),
        annotation_gt=TypedValue(type=str(annotation_kind), value=annotation_value),
        selected_branch=str(selected_branch),
        branch_probabilities=dict(branch_probabilities),
        target_payload=dict(target_payload),
        question_format=str(question_format),
    )


def common_trace_fields(case: CategoryGridCase, rendered: RenderedCategoryGridBundle) -> Dict[str, Any]:
    """Return shared trace fields independent of objective contract."""

    return {
        "scene_variant": str(case.scene_variant),
        "category_count": int(case.category_count),
        "subcategory_count": int(case.subcategory_count),
        "category_count_support": [int(value) for value in case.category_count_support],
        "subcategory_count_support": [int(value) for value in case.subcategory_count_support],
        "item_count_support": [int(value) for value in case.item_count_support],
        "categories": make_category_payload(rendered.rendered_grid, case.spec),
        "page_text_resources": dict(case.spec.text_resource_metadata),
        "scene_variant_probabilities": dict(case.scene_variant_probabilities),
        "category_count_probabilities": dict(case.category_count_probabilities),
        "subcategory_count_probabilities": dict(case.subcategory_count_probabilities),
    }


def _projected_annotation(answer_binding: CategoryGridAnswerBinding) -> Dict[str, Any]:
    if answer_binding.annotation_gt.type == "bbox_map":
        keyed_bboxes = dict(answer_binding.annotation_gt.value)
        return {
            "type": "bbox_map",
            "bbox_map": dict(keyed_bboxes),
            "pixel_bbox_map": dict(keyed_bboxes),
            "bbox_set": list(keyed_bboxes.values()),
            "target_category_id": str(answer_binding.target_payload["category_id"]),
            "target_subcategory_id": str(answer_binding.target_payload["subcategory_id"]),
        }
    bbox_set = [list(bbox) for bbox in answer_binding.annotation_gt.value]
    return {
        "type": "bbox_set",
        "bbox_set": list(bbox_set),
        "pixel_bbox_set": list(bbox_set),
        "target_category_id": str(answer_binding.target_payload["category_id"]),
        "target_subcategory_id": str(answer_binding.target_payload["subcategory_id"]),
    }


def build_category_grid_response(
    *,
    instance_seed: int,
    case: CategoryGridCase,
    rendered: RenderedCategoryGridBundle,
    prompt_binding: CategoryGridPromptBinding,
    answer_binding: CategoryGridAnswerBinding,
) -> TaskOutput:
    """Assemble one complete category-grid task response."""

    prompt_selection = render_task_prompt_variants(
        domain=DOMAIN,
        scene_id=SCENE,
        bundle_id=PROMPT_BUNDLE,
        scene_key=PROMPT_SCENE_KEY,
        task_key=PROMPT_TASK_KEY,
        query_key=str(prompt_binding.prompt_branch_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots=dict(prompt_binding.dynamic_slots),
        instance_seed=int(instance_seed),
    )
    prompt_artifacts = build_prompt_trace_artifacts(prompt_selection)
    common_fields = common_trace_fields(case, rendered)
    probabilities = {str(key): float(value) for key, value in answer_binding.branch_probabilities.items()}
    params = {
        **common_fields,
        "query_id": str(answer_binding.selected_branch),
        "prompt_query_key": str(prompt_binding.prompt_branch_key),
        "target": dict(answer_binding.target_payload),
        "target_answer": answer_binding.answer_gt.value,
        "query_id_probabilities": dict(probabilities),
    }
    query_spec = build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts,
        query_id=str(answer_binding.selected_branch),
        params=params,
    )
    query_spec["scene_id"] = SCENE

    rendered_grid = rendered.rendered_grid
    trace_payload = {
        "scene_ir": {
            "scene_id": SCENE,
            "scene_kind": "pages_category_grid",
            "entities": [dict(entity) for entity in rendered_grid.entities],
            "relations": {
                "query_id": str(answer_binding.selected_branch),
                "prompt_query_key": str(prompt_binding.prompt_branch_key),
                "scene_variant": str(case.scene_variant),
                "target": dict(answer_binding.target_payload),
                "answer_value": answer_binding.answer_gt.value,
            },
        },
        "query_spec": query_spec,
        "render_spec": {
            "canvas_width": int(rendered.render_params.canvas_width),
            "canvas_height": int(rendered.render_params.canvas_height),
            "coord_space": "pixel",
            "scene_id": SCENE,
            "scene_variant": str(case.scene_variant),
            "background_style": dict(rendered.background_meta),
            "information_scene_style": dict(rendered.style_meta),
            "post_image_noise": dict(rendered.post_noise_meta),
            "panel_bbox_px": list(rendered_grid.panel_bbox_px),
            "layout": dict(rendered_grid.layout_meta),
            "page_text_resources": dict(case.spec.text_resource_metadata),
        },
        "render_map": {
            "image_id": "img0",
            "panel_bbox_px": list(rendered_grid.panel_bbox_px),
            "category_header_bboxes_px": dict(rendered_grid.category_header_bboxes_px),
            "subcategory_header_bboxes_px": dict(rendered_grid.subcategory_header_bboxes_px),
            "item_row_bboxes_px": dict(rendered_grid.item_row_bboxes_px),
            "item_label_bboxes_px": dict(rendered_grid.item_label_bboxes_px),
        },
        "execution_trace": {
            **common_fields,
            "query_id": str(answer_binding.selected_branch),
            "prompt_query_key": str(prompt_binding.prompt_branch_key),
            "question_format": str(answer_binding.question_format),
            "target": dict(answer_binding.target_payload),
            "answer_value": answer_binding.answer_gt.value,
            "query_id_probabilities": dict(probabilities),
        },
        "witness_symbolic": {
            "type": str(answer_binding.question_format),
            "target_category_id": str(answer_binding.target_payload["category_id"]),
            "target_subcategory_id": str(answer_binding.target_payload["subcategory_id"]),
            "answer_value": answer_binding.answer_gt.value,
        },
        "projected_annotation": _projected_annotation(answer_binding),
    }
    return TaskOutput(
        prompt=str(prompt_artifacts.prompt),
        answer_gt=answer_binding.answer_gt,
        annotation_gt=answer_binding.annotation_gt,
        image=rendered.image,
        image_id="img0",
        trace_payload=trace_payload,
        task_versions=default_task_versions(),
        scene_id=SCENE,
        query_id=str(answer_binding.selected_branch),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
    )


def render_bound_category_grid(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    selected_branch: str,
    branch_probabilities: Mapping[str, float],
    case_factory: CaseFactory,
    binding_factory: BindingFactory,
) -> TaskOutput:
    """Resolve scene state, bind task-owned answer data, and assemble response."""

    case = case_factory(int(instance_seed), params=params)
    rendered = render_category_grid_case(
        instance_seed=int(instance_seed),
        params=params,
        case=case,
    )
    prompt_binding, answer_binding = binding_factory(
        str(selected_branch),
        dict(branch_probabilities),
        case,
        rendered,
    )
    return build_category_grid_response(
        instance_seed=int(instance_seed),
        case=case,
        rendered=rendered,
        prompt_binding=prompt_binding,
        answer_binding=answer_binding,
    )
