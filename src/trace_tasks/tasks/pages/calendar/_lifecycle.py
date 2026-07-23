"""Scene-private response assembly for calendar public tasks."""

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

from .shared.defaults import DOMAIN, PROMPT_BUNDLE, PROMPT_SCENE_KEY, SCENE
from .shared.rendering import RenderedCalendarBundle, render_calendar_case
from .shared.state import CalendarCase


@dataclass(frozen=True)
class CalendarPromptBinding:
    """Task-owned prompt keys and runtime operands."""

    task_key: str
    prompt_branch_key: str
    dynamic_slots: Mapping[str, Any]


@dataclass(frozen=True)
class CalendarAnswerBinding:
    """Task-owned answer and annotation payloads."""

    answer_gt: TypedValue
    annotation_gt: TypedValue
    selected_branch: str
    branch_probabilities: Mapping[str, float]
    extra_params: Mapping[str, Any]


CaseFactory = Callable[..., CalendarCase]
BindingFactory = Callable[
    [str, Mapping[str, float], CalendarCase, RenderedCalendarBundle],
    tuple[CalendarPromptBinding, CalendarAnswerBinding],
]


def _round_box(box: Any) -> list[float]:
    return [round(float(value), 3) for value in box]


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


def date_box(rendered: RenderedCalendarBundle, day: int) -> list[float]:
    """Return one rounded date-cell bbox from rendered metadata."""

    return _round_box(rendered.rendered_scene.date_cell_bboxes_by_day[int(day)])


def date_boxes(rendered: RenderedCalendarBundle, days: tuple[int, ...]) -> list[list[float]]:
    """Return rounded date-cell bboxes for the requested date numbers."""

    return [date_box(rendered, int(day)) for day in days]


def integer_binding(
    *,
    annotation_kind: str,
    annotation_value: Any,
    selected_branch: str,
    branch_probabilities: Mapping[str, float],
    answer_value: int,
    extra_params: Mapping[str, Any],
) -> CalendarAnswerBinding:
    """Build a calendar integer-answer binding from task-owned annotation data."""

    return CalendarAnswerBinding(
        answer_gt=TypedValue(type="integer", value=int(answer_value)),
        annotation_gt=TypedValue(type=str(annotation_kind), value=annotation_value),
        selected_branch=str(selected_branch),
        branch_probabilities=dict(branch_probabilities),
        extra_params=dict(extra_params),
    )


def common_trace_fields(case: CalendarCase, rendered: RenderedCalendarBundle) -> Dict[str, Any]:
    """Return shared calendar trace fields independent of objective contract."""

    style_meta = dict(rendered.information_style_meta)
    surface_style = style_meta.get("surface_style", {})
    if not isinstance(surface_style, Mapping):
        surface_style = {}
    return {
        "marked_day_class": case.marked_day_class,
        "scene_variant": str(case.scene_variant),
        "layout_mode": str(case.layout_mode),
        "title_mode": str(case.title_mode),
        "information_scene_treatment": str(style_meta.get("treatment", "")),
        "information_scene_palette_id": str(style_meta.get("palette_id", "")),
        "information_scene_style_pack": str(style_meta.get("style_pack", "")),
        "information_scene_surface_kind": str(surface_style.get("kind", "")),
        "information_scene_chrome_mode": str(surface_style.get("chrome_mode", "")),
        "week_start": str(case.week_start),
        "first_weekday_index": int(case.first_weekday_index),
        "visible_title_text": str(rendered.rendered_scene.title_text),
        "year": int(case.year),
        "month": int(case.month),
        "month_name": str(case.month_name),
        "days_in_month": int(case.days_in_month),
        "start_weekday_index": int(case.start_weekday_index),
        "row_count": int(case.row_count),
        "marked_dates": [int(day) for day in case.marked_dates],
        "annotation_dates": [int(day) for day in case.annotation_dates],
        "answer_value": int(case.answer_value),
        "weekend_weekday_indices": [int(value) for value in case.weekend_weekday_indices],
        "date_occurrence_support": [int(value) for value in case.date_occurrence_support],
        "marked_weekend_count_support": [int(value) for value in case.marked_weekend_count_support],
        "marked_weekday_count_support": [int(value) for value in case.marked_weekday_count_support],
        "marked_weekday_distractor_support": [int(value) for value in case.marked_weekday_distractor_support],
        "marked_weekend_distractor_support": [int(value) for value in case.marked_weekend_distractor_support],
        "workday_offset_support": [int(value) for value in case.workday_offset_support],
        "weekday_index": (int(case.weekday_index) if case.weekday_index is not None else None),
        "occurrence": (int(case.occurrence) if case.occurrence is not None else None),
        "workday_direction": case.workday_direction,
        "workday_offset": (int(case.workday_offset) if case.workday_offset is not None else None),
        "reference_date": (int(case.reference_date) if case.reference_date is not None else None),
        "target_date": (int(case.target_date) if case.target_date is not None else None),
        "marked_day_class_probabilities": dict(case.marked_day_class_probabilities),
        "scene_variant_probabilities": dict(case.scene_variant_probabilities),
        "layout_mode_probabilities": dict(case.layout_mode_probabilities),
        "title_mode_probabilities": dict(case.title_mode_probabilities),
        "week_start_probabilities": dict(case.week_start_probabilities),
    }


def build_calendar_response(
    *,
    instance_seed: int,
    case: CalendarCase,
    rendered: RenderedCalendarBundle,
    prompt_binding: CalendarPromptBinding,
    answer_binding: CalendarAnswerBinding,
) -> TaskOutput:
    """Assemble one complete calendar task response from task-bound semantics."""

    prompt_selection = render_task_prompt_variants(
        domain=DOMAIN,
        scene_id=SCENE,
        bundle_id=PROMPT_BUNDLE,
        scene_key=PROMPT_SCENE_KEY,
        task_key=str(prompt_binding.task_key),
        query_key=str(prompt_binding.prompt_branch_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots=dict(prompt_binding.dynamic_slots),
        instance_seed=int(instance_seed),
    )
    prompt_artifacts = build_prompt_trace_artifacts(prompt_selection)

    common_fields = common_trace_fields(case, rendered)
    probabilities = {str(key): float(value) for key, value in answer_binding.branch_probabilities.items()}
    query_params = {
        **common_fields,
        **dict(answer_binding.extra_params),
        "query_id": str(answer_binding.selected_branch),
        "prompt_query_key": str(prompt_binding.prompt_branch_key),
        "query_id_probabilities": dict(probabilities),
    }
    query_spec = build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts,
        query_id=str(answer_binding.selected_branch),
        params=query_params,
    )
    query_spec["scene_id"] = SCENE

    rendered_scene = rendered.rendered_scene
    trace_payload = {
        "scene_ir": {
            "scene_kind": "pages_month_calendar",
            "entities": [dict(entity) for entity in rendered_scene.entities],
            "relations": {
                "query_id": str(answer_binding.selected_branch),
                "prompt_query_key": str(prompt_binding.prompt_branch_key),
                "marked_day_class": case.marked_day_class,
                "scene_variant": str(case.scene_variant),
                "layout_mode": str(case.layout_mode),
                "title_mode": str(case.title_mode),
                "information_scene_treatment": str(rendered.information_style_meta.get("treatment", "")),
                "information_scene_palette_id": str(rendered.information_style_meta.get("palette_id", "")),
                "information_scene_style_pack": str(rendered.information_style_meta.get("style_pack", "")),
                "week_start": str(case.week_start),
                "first_weekday_index": int(case.first_weekday_index),
                "visible_title_text": str(rendered_scene.title_text),
                "year": int(case.year),
                "month": int(case.month),
                "month_name": str(case.month_name),
                "marked_dates": [int(day) for day in case.marked_dates],
                "workday_direction": case.workday_direction,
                "workday_offset": (int(case.workday_offset) if case.workday_offset is not None else None),
                "reference_date": (int(case.reference_date) if case.reference_date is not None else None),
                "target_date": (int(case.target_date) if case.target_date is not None else None),
            },
        },
        "query_spec": query_spec,
        "render_spec": {
            "canvas_width": int(rendered.render_params.canvas_width),
            "canvas_height": int(rendered.render_params.canvas_height),
            "coord_space": "pixel",
            "scene_variant": str(case.scene_variant),
            "background_style": dict(rendered.background_meta),
            "post_image_noise": dict(rendered.post_noise_meta),
            "scene_bbox_px": _round_box(rendered_scene.scene_bbox_px),
            "calendar_style": {
                "information_scene_style": dict(rendered.information_style_meta),
                "layout_mode": str(case.layout_mode),
                "title_mode": str(case.title_mode),
                "week_start": str(case.week_start),
                "first_weekday_index": int(case.first_weekday_index),
                "title": dict(rendered.title_meta),
                "panel_layout": dict(rendered.panel_layout_meta),
                "row_count": int(case.row_count),
                "title_text": str(rendered_scene.title_text),
                "title_bbox_px": (
                    _round_box(rendered_scene.title_bbox_px)
                    if rendered_scene.title_bbox_px is not None
                    else None
                ),
                "resolved_colors_rgb": {
                    key: [int(value) for value in values]
                    for key, values in rendered.resolved_colors_rgb.items()
                },
                "marker_kind": str(rendered.marker_kind),
            },
        },
        "render_map": {
            "image_id": "img0",
            "scene_bbox_px": _round_box(rendered_scene.scene_bbox_px),
            "calendar_panel_bbox_px": _round_box(rendered_scene.panel_bbox_px),
            "calendar_title_text": str(rendered_scene.title_text),
            "week_start": str(case.week_start),
            "first_weekday_index": int(case.first_weekday_index),
            "calendar_title_bbox_px": (
                _round_box(rendered_scene.title_bbox_px)
                if rendered_scene.title_bbox_px is not None
                else None
            ),
            "date_cells_by_day": {
                str(day): _round_box(bbox)
                for day, bbox in rendered_scene.date_cell_bboxes_by_day.items()
            },
            "annotation_dates": [int(day) for day in case.annotation_dates],
            "marked_dates": [int(day) for day in case.marked_dates],
            "reference_date": (int(case.reference_date) if case.reference_date is not None else None),
            "target_date": (int(case.target_date) if case.target_date is not None else None),
        },
        "execution_trace": {
            **common_fields,
            **dict(answer_binding.extra_params),
            "query_id": str(answer_binding.selected_branch),
            "prompt_query_key": str(prompt_binding.prompt_branch_key),
            "query_id_probabilities": dict(probabilities),
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


def render_bound_calendar(
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
    rendered = render_calendar_case(
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
    return build_calendar_response(
        instance_seed=int(instance_seed),
        case=case,
        rendered=rendered,
        prompt_binding=prompt_binding,
        answer_binding=answer_binding,
    )
