"""Scene-private response assembly for record-table public tasks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Sequence

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

from .shared.annotations import counted_row_bboxes, matching_row_records, row_entities, section_records
from .shared.defaults import DOMAIN, PROMPT_BUNDLE, PROMPT_SCENE_KEY, PROMPT_TASK_KEY, SCENE
from .shared.rendering import (
    RecordTableCase,
    RenderedRecordTableBundle,
    build_record_table_case,
    render_record_table_case,
)


@dataclass(frozen=True)
class RecordTablePromptBinding:
    """Task-owned prompt key and dynamic prompt slots."""

    prompt_branch_key: str
    dynamic_slots: Mapping[str, Any]


@dataclass(frozen=True)
class RecordTableAnswerBinding:
    """Task-owned answer, annotation, and trace payload fields."""

    answer_gt: TypedValue
    annotation_gt: TypedValue
    selected_branch: str
    branch_probabilities: Mapping[str, float]
    target_payload: Mapping[str, Any]
    question_format: str


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


def integer_binding(
    *,
    annotation_value: Sequence[Sequence[float]],
    selected_branch: str,
    branch_probabilities: Mapping[str, float],
    answer_value: int,
    target_payload: Mapping[str, Any],
    question_format: str,
) -> RecordTableAnswerBinding:
    """Build an integer-answer binding from task-owned annotation data."""

    annotation_bboxes = [list(bbox) for bbox in annotation_value]
    return RecordTableAnswerBinding(
        answer_gt=TypedValue(type="integer", value=int(answer_value)),
        annotation_gt=TypedValue(type="bbox_set", value=list(annotation_bboxes)),
        selected_branch=str(selected_branch),
        branch_probabilities=dict(branch_probabilities),
        target_payload=dict(target_payload),
        question_format=str(question_format),
    )


def build_case_and_render(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    filter_key: str,
    row_count_support: Sequence[int],
    answer_count_support: Sequence[int],
) -> tuple[RecordTableCase, RenderedRecordTableBundle]:
    """Materialize and render a task-owned table predicate."""

    case_params = dict(params)
    case_params.setdefault("row_count_support", tuple(int(value) for value in row_count_support))
    case_params.setdefault("answer_count_support", tuple(int(value) for value in answer_count_support))
    case = build_record_table_case(
        int(instance_seed),
        params=case_params,
        filter_key=str(filter_key),
        default_row_count_support=tuple(int(value) for value in row_count_support),
        default_answer_count_support=tuple(int(value) for value in answer_count_support),
    )
    rendered = render_record_table_case(instance_seed=int(instance_seed), params=case_params, case=case)
    return case, rendered


def _common_params(
    *,
    case: RecordTableCase,
    prompt_binding: RecordTablePromptBinding,
    answer_binding: RecordTableAnswerBinding,
) -> Dict[str, Any]:
    return {
        "query_id": str(answer_binding.selected_branch),
        "prompt_query_key": str(prompt_binding.prompt_branch_key),
        "target": dict(answer_binding.target_payload),
        "target_answer": int(answer_binding.answer_gt.value),
        "filter_key": str(case.filter_key),
        "scene_variant": str(case.scene_variant),
        "style_variant": str(case.style_variant),
        "target_status": str(case.target_status),
        "target_type": str(case.target_type),
        "target_action_label": str(case.target_action_label),
        "target_section_name": str(case.target_section_name),
        "target_section_index": int(case.target_section_index),
        "size_threshold_mb": int(case.size_threshold_mb),
        "row_count_support": [int(value) for value in case.row_count_support],
        "answer_count_support": [int(value) for value in case.answer_count_support],
        "section_count_support": [int(value) for value in case.section_count_support],
        "query_id_probabilities": {
            str(key): float(value)
            for key, value in sorted(answer_binding.branch_probabilities.items())
        },
        "filter_key_probabilities": dict(case.filter_key_probabilities),
        "scene_variant_probabilities": dict(case.scene_variant_probabilities),
        "style_variant_probabilities": dict(case.style_variant_probabilities),
    }


def build_record_table_response(
    *,
    instance_seed: int,
    case: RecordTableCase,
    rendered: RenderedRecordTableBundle,
    prompt_binding: RecordTablePromptBinding,
    answer_binding: RecordTableAnswerBinding,
) -> TaskOutput:
    """Assemble one complete record-table task response."""

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
    params = _common_params(
        case=case,
        prompt_binding=prompt_binding,
        answer_binding=answer_binding,
    )
    query_spec = build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts,
        query_id=str(answer_binding.selected_branch),
        params=params,
    )
    query_spec["scene_id"] = SCENE

    table = rendered.rendered_table
    row_records = [dict(record) for record in table.row_records]
    annotation_bboxes = list(answer_binding.annotation_gt.value)
    trace_payload = {
        "scene_ir": {
            "scene_id": SCENE,
            "scene_kind": "gui_table_row_filter",
            "entities": row_entities(rendered),
            "relations": {
                "query_id": str(answer_binding.selected_branch),
                "prompt_query_key": str(prompt_binding.prompt_branch_key),
                "filter_key": str(case.filter_key),
                "scene_variant": str(case.scene_variant),
                "style_variant": str(case.style_variant),
                "target": dict(answer_binding.target_payload),
                "answer_value": int(answer_binding.answer_gt.value),
                "annotation_row_ids": [str(value) for value in case.annotation_row_ids],
            },
            "frames": {
                "pixel": {"origin": [0.0, 0.0], "x_positive": "right", "y_positive": "down"},
            },
        },
        "query_spec": query_spec,
        "render_spec": {
            "scene_id": SCENE,
            "query_id": str(answer_binding.selected_branch),
            "canvas_width": int(rendered.render_params.canvas_width),
            "canvas_height": int(rendered.render_params.canvas_height),
            "coord_space": "pixel",
            "background_style": dict(rendered.background_meta),
            "post_image_noise": dict(rendered.post_noise_meta),
            "scene_variant": str(case.scene_variant),
            "style_variant": str(case.style_variant),
            "window_bbox_px": list(table.window_bbox_px),
            "scene_bbox_px": list(table.scene_bbox_px),
            "render_params": asdict(rendered.render_params),
            "theme": {
                "name": str(table.theme.name),
                "accent_rgb": [int(value) for value in table.theme.accent],
                "accent_alt_rgb": [int(value) for value in table.theme.accent_alt],
                "disabled_fill_rgb": [int(value) for value in table.theme.disabled_fill],
            },
        },
        "render_map": {
            "image_id": "img0",
            "scene_bbox_px": list(table.scene_bbox_px),
            "window_bbox_px": list(table.window_bbox_px),
            "app_profile": asdict(table.profile),
            "section_bboxes_by_name": dict(table.section_bboxes_by_name),
            "row_bboxes_by_id": dict(table.row_bboxes_by_id),
            "cell_bboxes_by_row_id": dict(table.cell_bboxes_by_row_id),
            "annotation_row_ids": [str(value) for value in case.annotation_row_ids],
        },
        "execution_trace": {
            **params,
            "answer_value": int(answer_binding.answer_gt.value),
            "section_records": section_records(case, rendered),
            "rows": list(row_records),
            "matching_row_ids": [str(value) for value in case.annotation_row_ids],
            "matching_rows": matching_row_records(case, rendered),
            "total_row_count": int(len(case.rows)),
            "section_count": int(len(case.section_names)),
            "question_format": str(answer_binding.question_format),
        },
        "witness_symbolic": {
            "type": "bbox_set",
            "annotation_row_ids": [str(value) for value in case.annotation_row_ids],
            "value": list(annotation_bboxes),
        },
        "projected_annotation": {
            "bbox_set": list(annotation_bboxes),
        },
        "background": dict(rendered.background_meta),
        "post_image_noise": dict(rendered.post_noise_meta),
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


def counted_row_annotation(case: RecordTableCase, rendered: RenderedRecordTableBundle) -> list[list[float]]:
    """Expose counted-row boxes to public task binders."""

    return counted_row_bboxes(case, rendered)
