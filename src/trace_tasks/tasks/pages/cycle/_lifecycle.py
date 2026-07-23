"""Scene-private response assembly for cycle public tasks."""

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

from .shared.annotations import stage_records
from .shared.defaults import DOMAIN, PROMPT_BUNDLE, PROMPT_SCENE_KEY, PROMPT_TASK_KEY, SCENE
from .shared.state import CycleCase, RenderedCycleScene


_REASONING_LOAD_BASE_BY_RELATIONSHIP = {
    "after": 0.27,
    "before": 0.33,
}


@dataclass(frozen=True)
class CyclePromptBinding:
    """Task-owned prompt key and dynamic prompt slots."""

    prompt_branch_key: str
    dynamic_slots: Mapping[str, Any]


@dataclass(frozen=True)
class CycleAnswerBinding:
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


def string_binding(
    *,
    annotation_kind: str,
    annotation_value: Any,
    selected_branch: str,
    branch_probabilities: Mapping[str, float],
    answer_value: str,
    target_payload: Mapping[str, Any],
    question_format: str,
) -> CycleAnswerBinding:
    """Build a string-answer binding from task-owned annotation data."""

    return CycleAnswerBinding(
        answer_gt=TypedValue(type="string", value=str(answer_value)),
        annotation_gt=TypedValue(type=str(annotation_kind), value=annotation_value),
        selected_branch=str(selected_branch),
        branch_probabilities=dict(branch_probabilities),
        target_payload=dict(target_payload),
        question_format=str(question_format),
    )


def _clamp_unit_interval(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _normalize_int_with_bounds(value: int, bounds: tuple[int, int]) -> float:
    low, high = int(bounds[0]), int(bounds[1])
    if high <= low:
        return 0.0
    return _clamp_unit_interval((int(value) - low) / float(high - low))


def _reasoning_load(case: CycleCase) -> float:
    stage_scan = _normalize_int_with_bounds(int(case.stage_count), (5, 12))
    step_scan = _normalize_int_with_bounds(int(case.step_count), (2, 11))
    return _clamp_unit_interval(
        float(_REASONING_LOAD_BASE_BY_RELATIONSHIP[str(case.query_relationship)])
        + (0.12 * float(stage_scan))
        + (0.20 * float(step_scan))
    )


def build_cycle_response(
    *,
    instance_seed: int,
    case: CycleCase,
    rendered: RenderedCycleScene,
    prompt_binding: CyclePromptBinding,
    answer_binding: CycleAnswerBinding,
) -> TaskOutput:
    """Assemble one complete cycle task response."""

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
    stage_payload = stage_records(case, rendered)
    annotation_bbox = list(answer_binding.annotation_gt.value)
    params = {
        "query_id": str(answer_binding.selected_branch),
        "prompt_query_key": str(prompt_binding.prompt_branch_key),
        "target": dict(target_payload),
        "target_answer": answer_binding.answer_gt.value,
        "scene_variant": str(case.scene_variant),
        "scene_variant_probabilities": dict(case.scene_variant_probabilities),
        "query_relationship": str(case.query_relationship),
        "query_relationship_probabilities": dict(case.query_relationship_probabilities),
        "cycle_direction": str(case.cycle_direction),
        "cycle_direction_probabilities": dict(case.cycle_direction_probabilities),
        "stage_count": int(case.stage_count),
        "step_count": int(case.step_count),
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
            "scene_kind": f"diagram_cycle_{str(case.scene_variant)}",
            "entities": [dict(entity) for entity in rendered.entities],
            "relations": {
                "query_id": str(answer_binding.selected_branch),
                "prompt_query_key": str(prompt_binding.prompt_branch_key),
                "scene_variant": str(case.scene_variant),
                "query_relationship": str(case.query_relationship),
                "direction": str(case.cycle_direction),
                "answer_stage_id": str(case.answer_stage_id),
                "answer_stage_bbox_id": str(case.answer_stage_bbox_id),
                "view_family": "cycle_diagram",
            },
            "frames": {
                "pixel": {"origin": [0.0, 0.0], "x_positive": "right", "y_positive": "down"},
            },
        },
        "query_spec": query_spec,
        "render_spec": {
            "scene_id": SCENE,
            "query_id": str(answer_binding.selected_branch),
            "scene_variant": str(case.scene_variant),
            "geometry_seed": int(instance_seed),
            "canvas_width": int(rendered.render_params.canvas_width),
            "canvas_height": int(rendered.render_params.canvas_height),
            "coord_space": "pixel",
            "node_width_px": int(rendered.render_params.node_width_px),
            "node_height_px": int(rendered.render_params.node_height_px),
            "ring_radius_x_px": int(rendered.render_params.ring_radius_x_px),
            "ring_radius_y_px": int(rendered.render_params.ring_radius_y_px),
            "edge_width_px": int(rendered.render_params.edge_width_px),
            "layout_jitter": dict(rendered.layout_jitter_meta),
            "background_style": dict(rendered.background_meta),
            "post_image_noise": dict(rendered.post_noise_meta),
            "render_params": asdict(rendered.render_params),
        },
        "render_map": {
            "image_id": "img0",
            "panel_bbox_px": list(rendered.panel_bbox_px),
            "title_bbox_px": list(rendered.title_bbox_px),
            "stage_bboxes_px": dict(rendered.stage_bbox_map),
            "stage_label_bboxes_px": dict(rendered.stage_label_bbox_map),
            "edge_bboxes_px": dict(rendered.edge_bbox_map),
        },
        "execution_trace": {
            "query_id": str(answer_binding.selected_branch),
            "prompt_query_key": str(prompt_binding.prompt_branch_key),
            "question_format": str(answer_binding.question_format),
            "view_family": "cycle_diagram",
            "scene_title": str(case.scene_title),
            "scene_variant": str(case.scene_variant),
            "query_prompt_slots": dict(case.prompt_slots),
            "direction": str(case.cycle_direction),
            "stage_count": int(case.stage_count),
            "step_count": int(case.step_count),
            "query_relationship": str(case.query_relationship),
            "query_stage_id": str(case.query_stage_id),
            "query_stage_label": str(case.query_stage_label),
            "query_stage_index": int(case.query_stage_index),
            "answer_stage_id": str(case.answer_stage_id),
            "answer_stage_label": str(case.answer_stage_label),
            "answer_stage_bbox_id": str(case.answer_stage_bbox_id),
            "answer_stage_index": int(case.answer_stage_index),
            "stage_specs": list(stage_payload),
            "edge_specs": [dict(spec) for spec in case.edge_specs],
            "supporting_stage_bbox_ids": [str(case.answer_stage_bbox_id)],
            "query_id_probabilities": dict(probabilities),
            "scene_variant_probabilities": dict(case.scene_variant_probabilities),
            "query_relationship_probabilities": dict(case.query_relationship_probabilities),
            "cycle_direction_probabilities": dict(case.cycle_direction_probabilities),
            "reasoning_load": _reasoning_load(case),
        },
        "witness_symbolic": {
            "type": "cycle_stage",
            "answer_stage_id": str(case.answer_stage_id),
            "answer_stage_label": str(case.answer_stage_label),
            "answer_stage_bbox_id": str(case.answer_stage_bbox_id),
        },
        "projected_annotation": {
            "type": "bbox",
            "bbox": list(annotation_bbox),
            "pixel_bbox": list(annotation_bbox),
            "bbox_ids": [str(case.answer_stage_bbox_id)],
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
