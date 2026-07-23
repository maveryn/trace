"""Scene-private response assembly for form-section public tasks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, Mapping

from trace_tasks.core.visual.noise import apply_post_image_noise
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

from .shared.annotations import operand_records, operand_role_to_field_id, project_operand_field_bbox_map
from .shared.defaults import (
    DOMAIN,
    GENERATION_DEFAULTS,
    NAMESPACE_ROOT,
    POST_IMAGE_NOISE_DEFAULTS,
    PROMPT_BUNDLE,
    PROMPT_SCENE_KEY,
    PROMPT_TASK_KEY,
    RENDERING_DEFAULTS,
    SCENE,
)
from .shared.forms import resolve_document_render_params
from .shared.rendering import RenderedDocumentScene, render_document_scene
from .shared.sampling import (
    ExpressionPlan,
    RankPlan,
    build_section_expression_case,
    build_section_rank_case,
    resolve_scene_variant,
)
from .shared.styles import prepare_document_information_scene


_SCENE_LOAD_BY_VARIANT = {
    "form_sheet": 0.16,
    "invoice_sheet": 0.22,
    "receipt_sheet": 0.18,
}


@dataclass(frozen=True)
class FormSectionPromptBinding:
    """Task-owned prompt key and dynamic prompt slots."""

    prompt_branch_key: str
    dynamic_slots: Mapping[str, Any]


@dataclass(frozen=True)
class FormSectionAnswerBinding:
    """Task-owned answer, annotation, and trace payload fields."""

    answer_gt: TypedValue
    annotation_gt: TypedValue
    selected_branch: str
    branch_probabilities: Mapping[str, float]
    target_payload: Mapping[str, Any]
    question_format: str
    reasoning_load_base: float


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


def currency_binding(
    *,
    selected_branch: str,
    branch_probabilities: Mapping[str, float],
    answer_value: str,
    annotation_value: Mapping[str, Any],
    target_payload: Mapping[str, Any],
    question_format: str,
    reasoning_load_base: float,
) -> FormSectionAnswerBinding:
    """Build a string-answer binding from task-owned arithmetic data."""

    return FormSectionAnswerBinding(
        answer_gt=TypedValue(type="string", value=str(answer_value)),
        annotation_gt=TypedValue(type="bbox_map", value=dict(annotation_value)),
        selected_branch=str(selected_branch),
        branch_probabilities=dict(branch_probabilities),
        target_payload=dict(target_payload),
        question_format=str(question_format),
        reasoning_load_base=float(reasoning_load_base),
    )


def label_bbox_binding(
    *,
    selected_branch: str,
    branch_probabilities: Mapping[str, float],
    answer_value: str,
    annotation_value: Any,
    target_payload: Mapping[str, Any],
    question_format: str,
    reasoning_load_base: float,
) -> FormSectionAnswerBinding:
    """Build a string-answer binding with one selected field-row bbox."""

    return FormSectionAnswerBinding(
        answer_gt=TypedValue(type="string", value=str(answer_value)),
        annotation_gt=TypedValue(type="bbox", value=list(annotation_value)),
        selected_branch=str(selected_branch),
        branch_probabilities=dict(branch_probabilities),
        target_payload=dict(target_payload),
        question_format=str(question_format),
        reasoning_load_base=float(reasoning_load_base),
    )


def _resolve_branch_value(value: Any, *, selected_branch: str) -> Any:
    """Return a branch-specific lifecycle value when a mapping is supplied."""

    if isinstance(value, Mapping):
        branch = str(selected_branch)
        if branch not in value:
            raise ValueError(f"missing form_section branch value for query_id={branch!r}")
        return value[branch]
    return value


def _clamp_unit_interval(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _normalize_int_with_bounds(value: int, bounds: list[int] | tuple[int, int]) -> float:
    low, high = int(bounds[0]), int(bounds[1])
    if high <= low:
        return 0.0
    return _clamp_unit_interval((int(value) - low) / float(high - low))


def _reasoning_load(case: Mapping[str, Any], answer_binding: FormSectionAnswerBinding) -> float:
    field_scan = _normalize_int_with_bounds(int(case["field_count"]), list(case["field_count_range"]))
    operand_scan = _normalize_int_with_bounds(len(list(case["operand_field_ids"])), (2, 3))
    scene_load = float(_SCENE_LOAD_BY_VARIANT.get(str(case["scene_variant"]), 0.18))
    return _clamp_unit_interval(
        float(answer_binding.reasoning_load_base)
        + (0.10 * float(field_scan))
        + (0.12 * float(operand_scan))
        + float(scene_load)
    )


def build_form_section_response(
    *,
    instance_seed: int,
    case: Mapping[str, Any],
    rendered: RenderedDocumentScene,
    render_params: Any,
    prompt_binding: FormSectionPromptBinding,
    answer_binding: FormSectionAnswerBinding,
    scene_variant_probabilities: Mapping[str, float],
    background_meta: Mapping[str, Any],
    information_style_meta: Mapping[str, Any],
    post_noise_meta: Mapping[str, Any],
) -> TaskOutput:
    """Assemble one complete form-section arithmetic task response."""

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
    probabilities = {str(key): float(value) for key, value in answer_binding.branch_probabilities.items()}
    target_payload = dict(answer_binding.target_payload)
    role_to_field_id = operand_role_to_field_id([str(value) for value in case["operand_field_ids"]])
    annotation_bboxes = {
        str(key): [float(item) for item in value]
        for key, value in dict(answer_binding.annotation_gt.value).items()
    }
    params = {
        "query_id": str(answer_binding.selected_branch),
        "prompt_query_key": str(prompt_binding.prompt_branch_key),
        "target": dict(target_payload),
        "target_answer": answer_binding.answer_gt.value,
        "scene_variant": str(case["scene_variant"]),
        "scene_variant_probabilities": dict(scene_variant_probabilities),
        "field_count": int(case["field_count"]),
        "field_count_range": list(case["field_count_range"]),
        "target_amount_candidate_count": int(case["target_amount_candidate_count"]),
        "operand_count": len(list(case["operand_field_ids"])),
        "operator_sequence": list(case["operator_sequence"]),
    }
    query_spec = build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts,
        query_id=str(answer_binding.selected_branch),
        params=params,
    )
    query_spec["scene_id"] = SCENE

    trace_payload = {
        "scene_ir": {
            "scene_id": SCENE,
            "scene_kind": f"document_{str(case['scene_variant'])}",
            "entities": [dict(entity) for entity in rendered.entities],
            "relations": {
                "query_id": str(answer_binding.selected_branch),
                "prompt_query_key": str(prompt_binding.prompt_branch_key),
                "scene_variant": str(case["scene_variant"]),
                "query_section_id": str(case["query_section_id"]),
                "query_section_label": str(case["query_section_label"]),
                "view_family": str(case["view_family"]),
                "result_value": str(case["result_value"]),
            },
            "frames": {
                "pixel": {"origin": [0.0, 0.0], "x_positive": "right", "y_positive": "down"},
            },
        },
        "query_spec": query_spec,
        "render_spec": {
            "scene_id": SCENE,
            "query_id": str(answer_binding.selected_branch),
            "prompt_query_key": str(prompt_binding.prompt_branch_key),
            "scene_variant": str(case["scene_variant"]),
            "geometry_seed": int(instance_seed),
            "canvas_width": int(render_params.canvas_width),
            "canvas_height": int(render_params.canvas_height),
            "coord_space": "pixel",
            "sheet_page_width_px": int(render_params.sheet_page_width_px),
            "sheet_page_height_px": int(render_params.sheet_page_height_px),
            "receipt_page_width_px": int(render_params.receipt_page_width_px),
            "receipt_page_height_px": int(render_params.receipt_page_height_px),
            "page_shadow_offset_px": int(render_params.page_shadow_offset_px),
            "document_layout_mode": str(render_params.document_layout_mode),
            "layout_jitter": dict(rendered.layout_jitter_meta),
            "information_scene_style": dict(information_style_meta),
            "background_style": dict(background_meta),
            "post_image_noise": dict(post_noise_meta),
            "render_params": asdict(render_params),
        },
        "render_map": {
            "image_id": "img0",
            "page_bbox_px": list(rendered.page_bbox_px),
            "title_bbox_px": list(rendered.title_bbox_px),
            "section_label_bboxes_px": dict(rendered.section_label_bbox_map),
            "section_box_bboxes_px": dict(rendered.section_box_bbox_map),
            "field_label_bboxes_px": dict(rendered.field_label_bbox_map),
            "field_value_bboxes_px": dict(rendered.field_value_bbox_map),
            "field_box_bboxes_px": dict(rendered.field_box_bbox_map),
        },
        "execution_trace": {
            "query_id": str(answer_binding.selected_branch),
            "prompt_query_key": str(prompt_binding.prompt_branch_key),
            "question_format": str(answer_binding.question_format),
            "view_family": str(case["view_family"]),
            "scene_title": str(case["scene_title"]),
            "scene_variant": str(case["scene_variant"]),
            "query_prompt_slots": dict(prompt_binding.dynamic_slots),
            "field_count": int(case["field_count"]),
            "field_specs": [dict(spec) for spec in case["field_specs"]],
            "section_specs": [dict(spec) for spec in case["section_specs"]],
            "query_section_id": str(case["query_section_id"]),
            "query_section_label": str(case["query_section_label"]),
            "target_amount_candidate_count": int(case["target_amount_candidate_count"]),
            "target_amount_candidate_field_ids": list(case["target_amount_candidate_field_ids"]),
            "operand_field_ids": list(case["operand_field_ids"]),
            "operand_field_labels": list(case["operand_field_labels"]),
            "operand_field_values": list(case["operand_field_values"]),
            "operand_value_bbox_ids": list(case["operand_value_bbox_ids"]),
            "operand_field_bbox_ids": list(case["operand_field_ids"]),
            "operator_sequence": list(case["operator_sequence"]),
            "expression_operand_cents": list(case["expression_operand_cents"]),
            "result_cents": int(case["result_cents"]),
            "result_value": str(case["result_value"]),
            "operand_records": operand_records(case, annotation_bboxes),
            "query_id_probabilities": dict(probabilities),
            "scene_variant_probabilities": dict(scene_variant_probabilities),
            "reasoning_load": _reasoning_load(case, answer_binding),
        },
        "witness_symbolic": {
            "type": "bbox_map",
            "operand_roles": list(annotation_bboxes.keys()),
            "operand_role_to_field_id": {
                str(key): str(field_id)
                for key, field_id in zip(annotation_bboxes.keys(), case["operand_field_ids"], strict=True)
            },
            "operand_role_to_bbox_id": dict(role_to_field_id),
            "field_id_sequence": list(case["operand_field_ids"]),
            "bbox_id_sequence": list(case["operand_field_ids"]),
            "value_bbox_id_sequence": list(case["operand_value_bbox_ids"]),
        },
        "projected_annotation": {
            "type": "bbox_map",
            "bbox_map": dict(annotation_bboxes),
            "pixel_bbox_map": dict(annotation_bboxes),
            "bbox_ids": list(role_to_field_id.values()),
        },
        "background": dict(background_meta),
        "post_image_noise": dict(post_noise_meta),
        "answer_gt": answer_binding.answer_gt.to_dict(),
        "annotation_gt": answer_binding.annotation_gt.to_dict(),
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


def build_form_section_ranked_label_response(
    *,
    instance_seed: int,
    case: Mapping[str, Any],
    rendered: RenderedDocumentScene,
    render_params: Any,
    prompt_binding: FormSectionPromptBinding,
    answer_binding: FormSectionAnswerBinding,
    scene_variant_probabilities: Mapping[str, float],
    background_meta: Mapping[str, Any],
    information_style_meta: Mapping[str, Any],
    post_noise_meta: Mapping[str, Any],
) -> TaskOutput:
    """Assemble one complete form-section ranked field-label task response."""

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
    probabilities = {str(key): float(value) for key, value in answer_binding.branch_probabilities.items()}
    annotation_bbox = [float(value) for value in list(answer_binding.annotation_gt.value)]
    params = {
        "query_id": str(answer_binding.selected_branch),
        "prompt_query_key": str(prompt_binding.prompt_branch_key),
        "target": dict(answer_binding.target_payload),
        "target_answer": answer_binding.answer_gt.value,
        "scene_variant": str(case["scene_variant"]),
        "scene_variant_probabilities": dict(scene_variant_probabilities),
        "field_count": int(case["field_count"]),
        "field_count_range": list(case["field_count_range"]),
        "target_amount_candidate_count": int(case["target_amount_candidate_count"]),
        "rank_from": str(case["rank_from"]),
        "rank_position": int(case["rank_position"]),
    }
    query_spec = build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts,
        query_id=str(answer_binding.selected_branch),
        params=params,
    )
    query_spec["scene_id"] = SCENE
    field_scan = _normalize_int_with_bounds(int(case["field_count"]), list(case["field_count_range"]))
    candidate_scan = _normalize_int_with_bounds(int(case["target_amount_candidate_count"]), (4, 10))
    reasoning_load = _clamp_unit_interval(
        float(answer_binding.reasoning_load_base)
        + (0.10 * float(field_scan))
        + (0.10 * float(candidate_scan))
        + float(_SCENE_LOAD_BY_VARIANT.get(str(case["scene_variant"]), 0.18))
    )

    trace_payload = {
        "scene_ir": {
            "scene_id": SCENE,
            "scene_kind": f"document_{str(case['scene_variant'])}",
            "entities": [dict(entity) for entity in rendered.entities],
            "relations": {
                "query_id": str(answer_binding.selected_branch),
                "prompt_query_key": str(prompt_binding.prompt_branch_key),
                "scene_variant": str(case["scene_variant"]),
                "query_section_id": str(case["query_section_id"]),
                "query_section_label": str(case["query_section_label"]),
                "view_family": str(case["view_family"]),
                "selected_field_id": str(case["selected_field_id"]),
                "selected_field_label": str(case["selected_field_label"]),
            },
            "frames": {
                "pixel": {"origin": [0.0, 0.0], "x_positive": "right", "y_positive": "down"},
            },
        },
        "query_spec": query_spec,
        "render_spec": {
            "scene_id": SCENE,
            "query_id": str(answer_binding.selected_branch),
            "prompt_query_key": str(prompt_binding.prompt_branch_key),
            "scene_variant": str(case["scene_variant"]),
            "geometry_seed": int(instance_seed),
            "canvas_width": int(render_params.canvas_width),
            "canvas_height": int(render_params.canvas_height),
            "coord_space": "pixel",
            "sheet_page_width_px": int(render_params.sheet_page_width_px),
            "sheet_page_height_px": int(render_params.sheet_page_height_px),
            "receipt_page_width_px": int(render_params.receipt_page_width_px),
            "receipt_page_height_px": int(render_params.receipt_page_height_px),
            "page_shadow_offset_px": int(render_params.page_shadow_offset_px),
            "document_layout_mode": str(render_params.document_layout_mode),
            "layout_jitter": dict(rendered.layout_jitter_meta),
            "information_scene_style": dict(information_style_meta),
            "background_style": dict(background_meta),
            "post_image_noise": dict(post_noise_meta),
            "render_params": asdict(render_params),
        },
        "render_map": {
            "image_id": "img0",
            "page_bbox_px": list(rendered.page_bbox_px),
            "title_bbox_px": list(rendered.title_bbox_px),
            "section_label_bboxes_px": dict(rendered.section_label_bbox_map),
            "section_box_bboxes_px": dict(rendered.section_box_bbox_map),
            "field_label_bboxes_px": dict(rendered.field_label_bbox_map),
            "field_value_bboxes_px": dict(rendered.field_value_bbox_map),
            "field_box_bboxes_px": dict(rendered.field_box_bbox_map),
        },
        "execution_trace": {
            "query_id": str(answer_binding.selected_branch),
            "prompt_query_key": str(prompt_binding.prompt_branch_key),
            "question_format": str(answer_binding.question_format),
            "view_family": str(case["view_family"]),
            "scene_title": str(case["scene_title"]),
            "scene_variant": str(case["scene_variant"]),
            "query_prompt_slots": dict(prompt_binding.dynamic_slots),
            "field_count": int(case["field_count"]),
            "field_specs": [dict(spec) for spec in case["field_specs"]],
            "section_specs": [dict(spec) for spec in case["section_specs"]],
            "query_section_id": str(case["query_section_id"]),
            "query_section_label": str(case["query_section_label"]),
            "target_amount_candidate_count": int(case["target_amount_candidate_count"]),
            "target_amount_candidate_field_ids": list(case["target_amount_candidate_field_ids"]),
            "target_amount_candidate_field_labels": list(case["target_amount_candidate_field_labels"]),
            "target_amount_candidate_field_values": list(case["target_amount_candidate_field_values"]),
            "rank_from": str(case["rank_from"]),
            "rank_position": int(case["rank_position"]),
            "rank_phrase": str(case["rank_phrase"]),
            "selected_field_id": str(case["selected_field_id"]),
            "selected_field_label": str(case["selected_field_label"]),
            "selected_field_value": str(case["selected_field_value"]),
            "selected_amount_cents": int(case["selected_amount_cents"]),
            "query_id_probabilities": dict(probabilities),
            "scene_variant_probabilities": dict(scene_variant_probabilities),
            "reasoning_load": float(reasoning_load),
        },
        "witness_symbolic": {
            "type": "bbox",
            "selected_field_id": str(case["selected_field_id"]),
            "bbox_id": str(case["selected_field_id"]),
            "field_label": str(case["selected_field_label"]),
            "field_value": str(case["selected_field_value"]),
        },
        "projected_annotation": {
            "type": "bbox",
            "bbox": list(annotation_bbox),
            "pixel_bbox": list(annotation_bbox),
            "bbox_id": str(case["selected_field_id"]),
        },
        "background": dict(background_meta),
        "post_image_noise": dict(post_noise_meta),
        "answer_gt": answer_binding.answer_gt.to_dict(),
        "annotation_gt": answer_binding.annotation_gt.to_dict(),
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


def run_form_section_public_entry(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    public_task: str,
    supported_query_ids: tuple[str, ...],
    expression: ExpressionPlan | Mapping[str, ExpressionPlan],
    prompt_query_key: str | Mapping[str, str],
    question_format: str | Mapping[str, str],
    reasoning_load_base: float | Mapping[str, float],
) -> TaskOutput:
    """Run common form-section rendering after the public file binds its arithmetic objective."""

    del max_attempts
    selected_branch, branch_probabilities, task_params = select_public_branch(
        instance_seed=int(instance_seed),
        params=params,
        supported=supported_query_ids,
        default=str(supported_query_ids[0]),
        public_task=str(public_task),
    )
    resolved_expression = _resolve_branch_value(expression, selected_branch=selected_branch)
    resolved_prompt_query_key = str(_resolve_branch_value(prompt_query_key, selected_branch=selected_branch))
    resolved_question_format = str(_resolve_branch_value(question_format, selected_branch=selected_branch))
    resolved_reasoning_load_base = float(
        _resolve_branch_value(reasoning_load_base, selected_branch=selected_branch)
    )
    sampling_namespace = f"{NAMESPACE_ROOT}.{resolved_prompt_query_key}"
    scene_variant, scene_variant_probabilities = resolve_scene_variant(
        task_params,
        gen_defaults=GENERATION_DEFAULTS,
        instance_seed=int(instance_seed),
        sampling_namespace=sampling_namespace,
    )
    case = build_section_expression_case(
        expression=resolved_expression,
        scene_variant=str(scene_variant),
        instance_seed=int(instance_seed),
        sampling_namespace=sampling_namespace,
    )
    render_params = resolve_document_render_params(
        task_params,
        render_defaults=RENDERING_DEFAULTS,
        instance_seed=int(instance_seed),
    )
    render_params, background, background_meta, information_style_meta = prepare_document_information_scene(
        instance_seed=int(instance_seed),
        params=task_params,
        scene_id=SCENE,
        render_params=render_params,
    )
    rendered_scene = render_document_scene(
        background,
        scene_variant=str(scene_variant),
        geometry_seed=int(instance_seed),
        scene_title=str(case["scene_title"]),
        field_specs=list(case["field_specs"]),
        section_specs=list(case["section_specs"]),
        render_params=render_params,
    )
    image, post_noise_meta = apply_post_image_noise(
        rendered_scene.image,
        instance_seed=int(instance_seed),
        params=task_params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    rendered_scene = replace(rendered_scene, image=image)
    role_to_field_id = operand_role_to_field_id([str(value) for value in case["operand_field_ids"]])
    annotation_map = project_operand_field_bbox_map(rendered_scene.field_box_bbox_map, role_to_field_id)
    prompt_binding = FormSectionPromptBinding(
        prompt_branch_key=str(resolved_prompt_query_key),
        dynamic_slots=dict(case["prompt_slots"]),
    )
    answer_binding = currency_binding(
        selected_branch=str(selected_branch),
        branch_probabilities=branch_probabilities,
        answer_value=str(case["result_value"]),
        annotation_value=annotation_map,
        target_payload={
            "section_label": str(case["query_section_label"]),
            "operand_field_ids": list(case["operand_field_ids"]),
            "operator_sequence": list(case["operator_sequence"]),
        },
        question_format=str(resolved_question_format),
        reasoning_load_base=float(resolved_reasoning_load_base),
    )
    return build_form_section_response(
        instance_seed=int(instance_seed),
        case=case,
        rendered=rendered_scene,
        render_params=render_params,
        prompt_binding=prompt_binding,
        answer_binding=answer_binding,
        scene_variant_probabilities=scene_variant_probabilities,
        background_meta=background_meta,
        information_style_meta=information_style_meta,
        post_noise_meta=post_noise_meta,
    )


def run_form_section_ranked_label_public_entry(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    public_task: str,
    supported_query_ids: tuple[str, ...],
    rank_plan: Mapping[str, RankPlan],
    prompt_query_key: str,
    question_format: str,
    reasoning_load_base: float,
) -> TaskOutput:
    """Run common form-section rendering for ranked amount field-label tasks."""

    del max_attempts
    selected_branch, branch_probabilities, task_params = select_public_branch(
        instance_seed=int(instance_seed),
        params=params,
        supported=supported_query_ids,
        default=str(supported_query_ids[0]),
        public_task=str(public_task),
    )
    resolved_rank_plan = rank_plan[str(selected_branch)]
    sampling_namespace = f"{NAMESPACE_ROOT}.{prompt_query_key}.{selected_branch}"
    scene_variant, scene_variant_probabilities = resolve_scene_variant(
        task_params,
        gen_defaults=GENERATION_DEFAULTS,
        instance_seed=int(instance_seed),
        sampling_namespace=sampling_namespace,
    )
    case = build_section_rank_case(
        rank_plan=resolved_rank_plan,
        scene_variant=str(scene_variant),
        instance_seed=int(instance_seed),
        sampling_namespace=sampling_namespace,
    )
    render_params = resolve_document_render_params(
        task_params,
        render_defaults=RENDERING_DEFAULTS,
        instance_seed=int(instance_seed),
    )
    render_params, background, background_meta, information_style_meta = prepare_document_information_scene(
        instance_seed=int(instance_seed),
        params=task_params,
        scene_id=SCENE,
        render_params=render_params,
    )
    rendered_scene = render_document_scene(
        background,
        scene_variant=str(scene_variant),
        geometry_seed=int(instance_seed),
        scene_title=str(case["scene_title"]),
        field_specs=list(case["field_specs"]),
        section_specs=list(case["section_specs"]),
        render_params=render_params,
    )
    image, post_noise_meta = apply_post_image_noise(
        rendered_scene.image,
        instance_seed=int(instance_seed),
        params=task_params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    rendered_scene = replace(rendered_scene, image=image)
    selected_field_id = str(case["selected_field_id"])
    annotation_bbox = [
        round(float(value), 3)
        for value in rendered_scene.field_box_bbox_map[str(selected_field_id)]
    ]
    prompt_binding = FormSectionPromptBinding(
        prompt_branch_key=str(prompt_query_key),
        dynamic_slots=dict(case["prompt_slots"]),
    )
    answer_binding = label_bbox_binding(
        selected_branch=str(selected_branch),
        branch_probabilities=branch_probabilities,
        answer_value=str(case["answer_value"]),
        annotation_value=annotation_bbox,
        target_payload={
            "section_label": str(case["query_section_label"]),
            "rank_from": str(case["rank_from"]),
            "rank_position": int(case["rank_position"]),
            "selected_field_id": str(case["selected_field_id"]),
        },
        question_format=str(question_format),
        reasoning_load_base=float(reasoning_load_base),
    )
    return build_form_section_ranked_label_response(
        instance_seed=int(instance_seed),
        case=case,
        rendered=rendered_scene,
        render_params=render_params,
        prompt_binding=prompt_binding,
        answer_binding=answer_binding,
        scene_variant_probabilities=scene_variant_probabilities,
        background_meta=background_meta,
        information_style_meta=information_style_meta,
        post_noise_meta=post_noise_meta,
    )


__all__ = [
    "FormSectionAnswerBinding",
    "FormSectionPromptBinding",
    "build_form_section_response",
    "build_form_section_ranked_label_response",
    "currency_binding",
    "label_bbox_binding",
    "project_operand_field_bbox_map",
    "run_form_section_public_entry",
    "run_form_section_ranked_label_public_entry",
    "select_public_branch",
]
