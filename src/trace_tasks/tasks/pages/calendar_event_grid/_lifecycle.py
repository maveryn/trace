"""Scene-private response assembly for calendar event-grid public tasks."""

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

from .shared.annotations import event_chip_records, round_box
from .shared.defaults import DOMAIN, EVENT_GRID_TASK_KEY, PROMPT_BUNDLE, PROMPT_SCENE_KEY, SCENE
from .shared.rendering import RenderedEventGridBundle, render_event_grid_case
from .shared.state import EventGridCase


@dataclass(frozen=True)
class EventGridPromptBinding:
    """Task-owned prompt key and dynamic prompt slots."""

    prompt_branch_key: str
    dynamic_slots: Mapping[str, Any]


@dataclass(frozen=True)
class EventGridAnswerBinding:
    """Task-owned answer, annotation, and trace payload fields."""

    answer_gt: TypedValue
    annotation_gt: TypedValue
    selected_branch: str
    branch_probabilities: Mapping[str, float]
    extra_params: Mapping[str, Any]


CaseFactory = Callable[..., EventGridCase]
BindingFactory = Callable[
    [str, Mapping[str, float], EventGridCase, RenderedEventGridBundle],
    tuple[EventGridPromptBinding, EventGridAnswerBinding],
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
    extra_params: Mapping[str, Any],
) -> EventGridAnswerBinding:
    """Build a string-answer binding from task-owned annotation data."""

    return EventGridAnswerBinding(
        answer_gt=TypedValue(type="string", value=str(answer_value)),
        annotation_gt=TypedValue(type=str(annotation_kind), value=annotation_value),
        selected_branch=str(selected_branch),
        branch_probabilities=dict(branch_probabilities),
        extra_params=dict(extra_params),
    )


def integer_binding(
    *,
    annotation_kind: str,
    annotation_value: Any,
    selected_branch: str,
    branch_probabilities: Mapping[str, float],
    answer_value: int,
    extra_params: Mapping[str, Any],
) -> EventGridAnswerBinding:
    """Build an integer-answer binding from task-owned annotation data."""

    return EventGridAnswerBinding(
        answer_gt=TypedValue(type="integer", value=int(answer_value)),
        annotation_gt=TypedValue(type=str(annotation_kind), value=annotation_value),
        selected_branch=str(selected_branch),
        branch_probabilities=dict(branch_probabilities),
        extra_params=dict(extra_params),
    )


def common_trace_fields(case: EventGridCase, rendered: RenderedEventGridBundle) -> Dict[str, Any]:
    """Return shared event-grid trace fields independent of objective contract."""

    records = event_chip_records(rendered=rendered, chips=case.event_chips)
    return {
        "scene_variant": str(case.scene_variant),
        "style_variant": str(case.style_variant),
        "accent_color_name": str(case.accent_color_name),
        "layout_mode": str(case.layout_mode),
        "title_mode": str(case.title_mode),
        "surface_mode": str(case.surface_mode),
        "text_color_mode": str(case.text_color_mode),
        "visible_title_text": str(rendered.rendered_scene.title_text),
        "year": int(case.year),
        "month": int(case.month),
        "month_name": str(case.month_name),
        "days_in_month": int(case.days_in_month),
        "row_count": int(case.row_count),
        "slot_id": str(case.slot_id),
        "slot_label": str(case.slot_label),
        "category_label": str(case.category_label),
        "event_category_labels": [str(value) for value in case.event_category_labels],
        "target_date": int(case.target_date) if case.target_date is not None else None,
        "target_count": int(case.target_count) if case.target_count is not None else None,
        "weekday_index": int(case.weekday_index) if case.weekday_index is not None else None,
        "weekday_label": str(case.weekday_label) if case.weekday_label is not None else None,
        "matching_chip_keys": [str(key) for key in case.matching_chip_keys],
        "event_chip_records": [dict(record) for record in records],
        "category_probabilities": dict(case.category_probabilities),
        "slot_probabilities": dict(case.slot_probabilities),
        "weekday_probabilities": dict(case.weekday_probabilities),
        "target_count_probabilities": dict(case.target_count_probabilities),
        "scene_variant_probabilities": dict(case.scene_variant_probabilities),
        "style_variant_probabilities": dict(case.style_variant_probabilities),
        "accent_color_name_probabilities": dict(case.accent_color_name_probabilities),
        "layout_mode_probabilities": dict(case.layout_mode_probabilities),
        "title_mode_probabilities": dict(case.title_mode_probabilities),
        "surface_mode_probabilities": dict(case.surface_mode_probabilities),
        "text_color_mode_probabilities": dict(case.text_color_mode_probabilities),
    }


def build_event_grid_response(
    *,
    instance_seed: int,
    case: EventGridCase,
    rendered: RenderedEventGridBundle,
    prompt_binding: EventGridPromptBinding,
    answer_binding: EventGridAnswerBinding,
) -> TaskOutput:
    """Assemble one complete calendar event-grid task response."""

    prompt_selection = render_task_prompt_variants(
        domain=DOMAIN,
        scene_id=SCENE,
        bundle_id=PROMPT_BUNDLE,
        scene_key=PROMPT_SCENE_KEY,
        task_key=EVENT_GRID_TASK_KEY,
        query_key=str(prompt_binding.prompt_branch_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots=dict(prompt_binding.dynamic_slots),
        instance_seed=int(instance_seed),
    )
    prompt_artifacts = build_prompt_trace_artifacts(prompt_selection)

    common_fields = common_trace_fields(case, rendered)
    probabilities = {str(key): float(value) for key, value in answer_binding.branch_probabilities.items()}
    task_params = {
        **common_fields,
        **dict(answer_binding.extra_params),
        "query_id": str(answer_binding.selected_branch),
        "prompt_query_key": str(prompt_binding.prompt_branch_key),
        "query_id_probabilities": dict(probabilities),
    }
    query_spec = build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts,
        query_id=str(answer_binding.selected_branch),
        params=task_params,
    )
    query_spec["scene_id"] = SCENE

    rendered_scene = rendered.rendered_scene
    records = event_chip_records(rendered=rendered, chips=case.event_chips)
    trace_payload = {
        "scene_ir": {
            "scene_id": SCENE,
            "scene_kind": "pages_calendar_event_grid",
            "entities": [dict(entity) for entity in rendered_scene.entities],
            "relations": {
                "query_id": str(answer_binding.selected_branch),
                "prompt_query_key": str(prompt_binding.prompt_branch_key),
                "scene_variant": str(case.scene_variant),
                "style_variant": str(case.style_variant),
                "accent_color_name": str(case.accent_color_name),
                "layout_mode": str(case.layout_mode),
                "title_mode": str(case.title_mode),
                "surface_mode": str(case.surface_mode),
                "text_color_mode": str(case.text_color_mode),
                "visible_title_text": str(rendered_scene.title_text),
                "year": int(case.year),
                "month": int(case.month),
                "month_name": str(case.month_name),
                "slot_id": str(case.slot_id),
                "slot_label": str(case.slot_label),
                "category_label": str(case.category_label),
                "target_date": int(case.target_date) if case.target_date is not None else None,
                "target_count": int(case.target_count) if case.target_count is not None else None,
                "weekday_index": int(case.weekday_index) if case.weekday_index is not None else None,
                "weekday_label": str(case.weekday_label) if case.weekday_label is not None else None,
                "matching_chip_keys": [str(key) for key in case.matching_chip_keys],
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
            "post_image_noise": dict(rendered.post_noise_meta),
            "scene_bbox_px": round_box(rendered_scene.scene_bbox_px),
            "calendar_event_grid_style": {
                "accent_color_name": str(case.accent_color_name),
                "style_variant": str(case.style_variant),
                "surface_mode": str(case.surface_mode),
                "text_color_mode": str(case.text_color_mode),
                "layout_mode": str(case.layout_mode),
                "title_mode": str(case.title_mode),
                "title": dict(rendered.title_meta),
                "panel_layout": dict(rendered.panel_layout_meta),
                "row_count": int(case.row_count),
                "slot_order": ["top", "mid", "end"],
                "title_text": str(rendered_scene.title_text),
                "resolved_colors_rgb": {
                    key: [int(value) for value in values]
                    for key, values in rendered.resolved_colors_rgb.items()
                },
            },
        },
        "render_map": {
            "image_id": "img0",
            "scene_bbox_px": round_box(rendered_scene.scene_bbox_px),
            "calendar_panel_bbox_px": round_box(rendered_scene.panel_bbox_px),
            "calendar_event_grid_panel_bbox_px": round_box(rendered_scene.panel_bbox_px),
            "date_cells_by_day": {
                str(day): round_box(bbox)
                for day, bbox in rendered_scene.date_cell_bboxes_by_day.items()
            },
            "event_chips_by_key": {
                str(key): round_box(bbox)
                for key, bbox in rendered_scene.event_chip_bboxes_by_key.items()
            },
            "event_chip_records": [dict(record) for record in records],
            "matching_chip_keys": [str(key) for key in case.matching_chip_keys],
            "target_date": int(case.target_date) if case.target_date is not None else None,
        },
        "execution_trace": {
            **common_fields,
            **dict(answer_binding.extra_params),
            "query_id": str(answer_binding.selected_branch),
            "prompt_query_key": str(prompt_binding.prompt_branch_key),
            "query_id_probabilities": dict(probabilities),
            "answer_value": answer_binding.answer_gt.value,
        },
        "witness_symbolic": {
            "type": str(answer_binding.annotation_gt.type),
            "value": answer_binding.annotation_gt.value,
        },
        "projected_annotation": {
            str(answer_binding.annotation_gt.type): answer_binding.annotation_gt.value,
        },
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


def render_bound_event_grid(
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
    rendered = render_event_grid_case(
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
    return build_event_grid_response(
        instance_seed=int(instance_seed),
        case=case,
        rendered=rendered,
        prompt_binding=prompt_binding,
        answer_binding=answer_binding,
    )
