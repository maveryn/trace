"""Scene-private response assembly for control-board public tasks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping

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

from .shared.annotations import control_entities, group_records, matching_control_records
from .shared.defaults import DOMAIN, PROMPT_BUNDLE, PROMPT_SCENE_KEY, PROMPT_TASK_KEY, SCENE
from .shared.state import ControlBoardCase, RenderedControlBoard


@dataclass(frozen=True)
class ControlBoardPromptBinding:
    """Task-owned prompt key and dynamic prompt slots."""

    prompt_branch_key: str
    dynamic_slots: Mapping[str, Any]


@dataclass(frozen=True)
class ControlBoardAnswerBinding:
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
    annotation_kind: str,
    annotation_value: Any,
    selected_branch: str,
    branch_probabilities: Mapping[str, float],
    answer_value: int,
    target_payload: Mapping[str, Any],
    question_format: str,
) -> ControlBoardAnswerBinding:
    """Build an integer-answer binding from task-owned annotation data."""

    return ControlBoardAnswerBinding(
        answer_gt=TypedValue(type="integer", value=int(answer_value)),
        annotation_gt=TypedValue(type=str(annotation_kind), value=annotation_value),
        selected_branch=str(selected_branch),
        branch_probabilities=dict(branch_probabilities),
        target_payload=dict(target_payload),
        question_format=str(question_format),
    )


def string_binding(
    *,
    annotation_kind: str,
    annotation_value: Any,
    selected_branch: str,
    branch_probabilities: Mapping[str, float],
    answer_value: str,
    target_payload: Mapping[str, Any],
    question_format: str,
) -> ControlBoardAnswerBinding:
    """Build a string-answer binding from task-owned annotation data."""

    return ControlBoardAnswerBinding(
        answer_gt=TypedValue(type="string", value=str(answer_value)),
        annotation_gt=TypedValue(type=str(annotation_kind), value=annotation_value),
        selected_branch=str(selected_branch),
        branch_probabilities=dict(branch_probabilities),
        target_payload=dict(target_payload),
        question_format=str(question_format),
    )


def build_control_board_response(
    *,
    instance_seed: int,
    public_task_id: str,
    case: ControlBoardCase,
    rendered: RenderedControlBoard,
    prompt_binding: ControlBoardPromptBinding,
    answer_binding: ControlBoardAnswerBinding,
) -> TaskOutput:
    """Assemble one complete control-board task response."""

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
    control_records = [dict(record) for record in rendered.control_records]
    matched_records = matching_control_records(case, rendered)
    annotation_value = answer_binding.annotation_gt.value
    annotation_type = str(answer_binding.annotation_gt.type)
    information_style_meta = rendered.background_meta.get("style_spec", {})
    if not isinstance(information_style_meta, Mapping):
        information_style_meta = {}
    params = {
        "query_id": str(answer_binding.selected_branch),
        "prompt_query_key": str(prompt_binding.prompt_branch_key),
        "target": dict(target_payload),
        "target_answer": answer_binding.answer_gt.value,
        "target_state_count": int(case.answer_value),
        "count_mode": str(case.count_mode),
        "scene_variant": str(case.scene_variant),
        "target_group_name": str(case.target_group_name),
        "target_group_index": int(case.target_group_index),
        "answer_support": [int(value) for value in case.answer_support],
        "candidate_label_pool": [str(value) for value in case.candidate_label_pool],
        "query_id_probabilities": dict(probabilities),
        "scene_variant_probabilities": dict(case.scene_variant_probabilities),
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
            "scene_kind": "gui_grouped_control_board",
            "entities": control_entities(rendered),
            "relations": {
                "query_id": str(answer_binding.selected_branch),
                "prompt_query_key": str(prompt_binding.prompt_branch_key),
                "count_mode": str(case.count_mode),
                "scene_variant": str(case.scene_variant),
                "target_group_name": str(case.target_group_name),
                "target_group_index": int(case.target_group_index),
                "answer_value": answer_binding.answer_gt.value,
                "target_state_count": int(case.answer_value),
                "annotation_control_ids": [str(value) for value in case.annotation_control_ids],
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
            "information_scene_style": dict(information_style_meta),
            "pages_information_style_policy": {
                "domain_wrapper": "scene_renderer",
                "scene_renderer_recorded_style": True,
                "task_id": str(public_task_id),
                "scene_id": SCENE,
            },
            "post_image_noise": dict(rendered.post_noise_meta),
            "scene_variant": str(case.scene_variant),
            "window_bbox_px": list(rendered.window_bbox_px),
            "scene_bbox_px": list(rendered.scene_bbox_px),
            "render_params": asdict(rendered.render_params),
            "theme": {
                "name": str(rendered.theme.name),
                "accent_rgb": [int(value) for value in rendered.theme.accent],
                "accent_alt_rgb": [int(value) for value in rendered.theme.accent_alt],
                "selected_outline_rgb": [int(value) for value in rendered.theme.selected_outline],
                "disabled_fill_rgb": [int(value) for value in rendered.theme.disabled_fill],
                "badge_fill_rgb": [int(value) for value in rendered.theme.badge_fill],
            },
        },
        "render_map": {
            "image_id": "img0",
            "scene_bbox_px": list(rendered.scene_bbox_px),
            "window_bbox_px": list(rendered.window_bbox_px),
            "app_profile": asdict(rendered.profile),
            "group_bboxes_by_name": dict(rendered.group_bboxes_by_name),
            "control_bboxes_by_id": dict(rendered.control_bboxes_by_id),
            "candidate_label_badge_bboxes_by_id": dict(rendered.badge_bboxes_by_id),
            "annotation_control_ids": [str(value) for value in case.annotation_control_ids],
        },
        "execution_trace": {
            "query_id": str(answer_binding.selected_branch),
            "prompt_query_key": str(prompt_binding.prompt_branch_key),
            "count_mode": str(case.count_mode),
            "scene_variant": str(case.scene_variant),
            "answer_value": answer_binding.answer_gt.value,
            "target_state_count": int(case.answer_value),
            "target_group_name": str(case.target_group_name),
            "target_group_index": int(case.target_group_index),
            "group_records": group_records(case, rendered),
            "controls": list(control_records),
            "matching_control_ids": [str(value) for value in case.annotation_control_ids],
            "matching_controls": list(matched_records),
            "total_control_count": int(len(case.controls)),
            "query_id_probabilities": dict(probabilities),
            "scene_variant_probabilities": dict(case.scene_variant_probabilities),
            "question_format": str(answer_binding.question_format),
        },
        "witness_symbolic": {
            "type": str(annotation_type),
            "annotation_control_ids": [str(value) for value in case.annotation_control_ids],
            "value": list(annotation_value),
        },
        "projected_annotation": (
            {"bbox": list(annotation_value)}
            if str(annotation_type) == "bbox"
            else {"bbox_set": list(annotation_value)}
        ),
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
