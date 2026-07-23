"""Scene-private response assembly for navigation-flow public tasks."""

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

from .shared.annotations import (
    annotation_bbox_map,
    annotation_role_support_ids,
    control_entities,
    support_ids_for_path,
    target_annotation_bbox,
)
from .shared.defaults import DOMAIN, PROMPT_BUNDLE, PROMPT_SCENE_KEY, PROMPT_TASK_KEY, SCENE
from .shared.state import NavigationFlowCase, RenderedNavigationFlow


@dataclass(frozen=True)
class NavigationPromptBinding:
    """Task-owned prompt key and dynamic prompt slots."""

    prompt_branch_key: str
    dynamic_slots: Mapping[str, Any]


@dataclass(frozen=True)
class NavigationAnswerBinding:
    """Task-owned answer, annotation, and trace fields."""

    answer_gt: TypedValue
    annotation_gt: TypedValue
    selected_branch: str
    branch_probabilities: Mapping[str, float]
    prompt_branch_key: str
    question_format: str


def choose_public_branch(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    supported: tuple[str, ...],
    default: str,
    public_task: str,
) -> tuple[str, Dict[str, float], Dict[str, Any]]:
    """Resolve the public branch through the shared selector."""

    branch, probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=tuple(str(value) for value in supported),
        default_query_id=str(default),
        task_id=str(public_task),
    )
    return str(branch), dict(probabilities), dict(task_params)


def option_letter_binding(
    *,
    annotation_value: Any,
    selected_branch: str,
    branch_probabilities: Mapping[str, float],
    answer_value: str,
    prompt_branch_key: str,
    question_format: str,
) -> NavigationAnswerBinding:
    """Build an option-letter answer binding."""

    return NavigationAnswerBinding(
        answer_gt=TypedValue(type="option_letter", value=str(answer_value)),
        annotation_gt=TypedValue(type="bbox", value=list(annotation_value)),
        selected_branch=str(selected_branch),
        branch_probabilities=dict(branch_probabilities),
        prompt_branch_key=str(prompt_branch_key),
        question_format=str(question_format),
    )


def build_navigation_flow_response(
    *,
    instance_seed: int,
    public_task_id: str,
    case: NavigationFlowCase,
    rendered: RenderedNavigationFlow,
    prompt_binding: NavigationPromptBinding,
    answer_binding: NavigationAnswerBinding,
) -> TaskOutput:
    """Assemble one complete navigation-flow task response."""

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
    support_ids = support_ids_for_path(case)
    role_support_ids = annotation_role_support_ids(case)
    support_bbox_map = annotation_bbox_map(case, rendered)
    control_records = [dict(record) for record in rendered.control_records]
    support_records = [dict(record) for record in rendered.support_records]
    target_record = next(record for record in control_records if str(record["control_id"]) == str(case.target_control_id))
    information_style_meta = rendered.background_meta.get("style_spec", {})
    if not isinstance(information_style_meta, Mapping):
        information_style_meta = {}
    support_record_subset = [
        next(record for record in support_records if str(record["support_id"]) == str(support_id))
        for support_id in support_ids
    ]
    common_params = {
        "query_id": str(answer_binding.selected_branch),
        "prompt_query_key": str(prompt_binding.prompt_branch_key),
        "source_query_id": str(answer_binding.prompt_branch_key),
        "navigation_surface": str(case.navigation_surface),
        "scene_variant": str(case.scene_variant),
        "target_control_id": str(case.target_control_id),
        "target_label": str(case.target_label),
        "path_labels": [str(value) for value in case.path_labels],
        "path_display": str(case.path_display),
        "command_label": str(case.command_label),
        "menu_command_count": int(case.menu_command_count),
        "menu_command_count_range": [int(value) for value in case.menu_command_count_range],
        "ribbon_tab_count": int(case.ribbon_tab_count),
        "ribbon_tab_count_range": [int(value) for value in case.ribbon_tab_count_range],
        "ribbon_group_count": int(case.ribbon_group_count),
        "ribbon_group_count_range": [int(value) for value in case.ribbon_group_count_range],
        "ribbon_command_count": int(case.ribbon_command_count),
        "ribbon_command_count_range": [int(value) for value in case.ribbon_command_count_range],
        "candidate_label_pool": [str(value) for value in case.candidate_label_pool],
        "annotation_role_support_ids": dict(role_support_ids),
        "query_id_probabilities": dict(probabilities),
        "navigation_surface_probabilities": dict(case.surface_probabilities),
        "scene_variant_probabilities": dict(case.scene_variant_probabilities),
    }
    spec = build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts,
        query_id=str(answer_binding.selected_branch),
        params=common_params,
    )
    spec["scene_id"] = SCENE

    trace_payload = {
        "scene_ir": {
            "scene_id": SCENE,
            "scene_kind": "gui_navigation_path",
            "entities": control_entities(rendered),
            "relations": {
                "query_id": str(answer_binding.selected_branch),
                "prompt_query_key": str(prompt_binding.prompt_branch_key),
                "source_query_id": str(answer_binding.prompt_branch_key),
                "navigation_surface": str(case.navigation_surface),
                "scene_variant": str(case.scene_variant),
                "target_control_id": str(case.target_control_id),
                "target_label": str(case.target_label),
                "path_labels": [str(value) for value in case.path_labels],
                "path_display": str(case.path_display),
                "command_label": str(case.command_label),
                "annotation_support_ids": [str(value) for value in support_ids],
                "annotation_role_support_ids": dict(role_support_ids),
                "path_support_bbox_map": dict(support_bbox_map),
            },
            "frames": {
                "pixel": {"origin": [0.0, 0.0], "x_positive": "right", "y_positive": "down"},
            },
        },
        "query_spec": spec,
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
                "badge_fill_rgb": [int(value) for value in rendered.theme.badge_fill],
            },
        },
        "render_map": {
            "image_id": "img0",
            "scene_bbox_px": list(rendered.scene_bbox_px),
            "window_bbox_px": list(rendered.window_bbox_px),
            "app_profile": asdict(rendered.profile),
            "control_bboxes_by_id": dict(rendered.control_bboxes_by_id),
            "candidate_label_badge_bboxes_by_id": dict(rendered.badge_bboxes_by_id),
            "support_bboxes_by_id": dict(rendered.support_bboxes_by_id),
            "target_control_id": str(case.target_control_id),
            "annotation_support_ids": [str(value) for value in support_ids],
            "annotation_role_support_ids": dict(role_support_ids),
            "path_support_bbox_map": dict(support_bbox_map),
        },
        "execution_trace": {
            **common_params,
            "annotation_support_ids": [str(value) for value in support_ids],
            "annotation_support_records": [dict(record) for record in support_record_subset],
            "path_support_bbox_map": dict(support_bbox_map),
            "target_control": dict(target_record),
            "controls": list(control_records),
            "support_records": list(support_records),
            "total_control_count": int(len(case.controls)),
            "question_format": str(answer_binding.question_format),
        },
        "witness_symbolic": {
            "type": "bbox",
            "annotation_support_ids": [str(value) for value in support_ids],
            "annotation_role_support_ids": dict(role_support_ids),
            "target_control_id": str(case.target_control_id),
            "path_support_bbox_map": dict(support_bbox_map),
            "value": list(answer_binding.annotation_gt.value),
        },
        "projected_annotation": {
            "type": "bbox",
            "bbox": list(answer_binding.annotation_gt.value),
            "pixel_bbox": list(answer_binding.annotation_gt.value),
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


def bind_navigation_answer(
    *,
    case: NavigationFlowCase,
    rendered: RenderedNavigationFlow,
    selected_branch: str,
    branch_probabilities: Mapping[str, float],
    prompt_branch_key: str,
    question_format: str,
) -> tuple[NavigationPromptBinding, NavigationAnswerBinding]:
    """Bind prompt slots, answer, and annotation from rendered metadata."""

    prompt_binding = NavigationPromptBinding(
        prompt_branch_key=str(prompt_branch_key),
        dynamic_slots={
            "path_display": str(case.path_display),
            "path_parent": str(case.path_labels[0]),
            "path_child": str(case.path_labels[1]),
            "command_label": str(case.command_label),
        },
    )
    answer_binding = option_letter_binding(
        annotation_value=target_annotation_bbox(case, rendered),
        selected_branch=str(selected_branch),
        branch_probabilities=dict(branch_probabilities),
        answer_value=str(case.target_label),
        prompt_branch_key=str(prompt_branch_key),
        question_format=str(question_format),
    )
    return prompt_binding, answer_binding
