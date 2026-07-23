"""Scene-private response assembly for process-flow page tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping

from trace_tasks.core.types import TypedValue
from trace_tasks.core.visual.background import make_background_canvas
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    build_prompt_query_spec,
    build_prompt_trace_artifacts,
    render_task_prompt_variants,
)

from .shared.defaults import (
    DOMAIN,
    POST_IMAGE_BACKGROUND_DEFAULTS,
    POST_IMAGE_NOISE_DEFAULTS,
    PROMPT_BUNDLE,
    PROMPT_DEFAULTS,
    PROMPT_SCENE_KEY,
    PROMPT_TASK_KEY,
    SCENE,
)
from .shared.rendering import render_process_flow_scene
from .shared.sampling import build_scene_case
from .shared.state import ProcessFlowSceneCase, RenderedProcessFlow


@dataclass(frozen=True)
class ProcessFlowPromptBinding:
    """Task-owned prompt branch and dynamic prompt slots."""

    prompt_branch_key: str
    dynamic_slots: Mapping[str, Any]


@dataclass(frozen=True)
class ProcessFlowAnswerBinding:
    """Task-owned answer, annotation, and trace payload fields."""

    answer_gt: TypedValue
    annotation_gt: TypedValue
    selected_branch: str
    branch_probabilities: Mapping[str, float]
    target_payload: Mapping[str, Any]
    projected_annotation: Mapping[str, Any]
    annotation_ids: tuple[str, ...]
    annotation_key_to_bbox_id: Mapping[str, str]
    question_format: str


BindingFactory = Callable[
    [int, str, Mapping[str, float], ProcessFlowSceneCase, RenderedProcessFlow],
    tuple[ProcessFlowPromptBinding, ProcessFlowAnswerBinding],
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


def integer_binding(
    *,
    annotation_kind: str,
    annotation_value: Any,
    selected_branch: str,
    branch_probabilities: Mapping[str, float],
    answer_value: int,
    target_payload: Mapping[str, Any],
    projected_annotation: Mapping[str, Any],
    annotation_ids: tuple[str, ...],
    annotation_key_to_bbox_id: Mapping[str, str] | None = None,
    question_format: str,
) -> ProcessFlowAnswerBinding:
    """Build an integer-answer binding from task-owned annotation data."""

    return ProcessFlowAnswerBinding(
        answer_gt=TypedValue(type="integer", value=int(answer_value)),
        annotation_gt=TypedValue(type=str(annotation_kind), value=annotation_value),
        selected_branch=str(selected_branch),
        branch_probabilities=dict(branch_probabilities),
        target_payload=dict(target_payload),
        projected_annotation=dict(projected_annotation),
        annotation_ids=tuple(str(value) for value in annotation_ids),
        annotation_key_to_bbox_id=dict(annotation_key_to_bbox_id or {}),
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
    projected_annotation: Mapping[str, Any],
    annotation_ids: tuple[str, ...],
    annotation_key_to_bbox_id: Mapping[str, str] | None = None,
    question_format: str,
) -> ProcessFlowAnswerBinding:
    """Build a string-answer binding from task-owned annotation data."""

    return ProcessFlowAnswerBinding(
        answer_gt=TypedValue(type="string", value=str(answer_value)),
        annotation_gt=TypedValue(type=str(annotation_kind), value=annotation_value),
        selected_branch=str(selected_branch),
        branch_probabilities=dict(branch_probabilities),
        target_payload=dict(target_payload),
        projected_annotation=dict(projected_annotation),
        annotation_ids=tuple(str(value) for value in annotation_ids),
        annotation_key_to_bbox_id=dict(annotation_key_to_bbox_id or {}),
        question_format=str(question_format),
    )


def common_trace_fields(case: ProcessFlowSceneCase) -> Dict[str, Any]:
    """Return scene fields shared by all process-flow objectives."""

    return {
        "scene_variant": str(case.scene_variant),
        "layout_variant": str(case.layout_variant),
        "style_variant": str(case.style_variant),
        "flow_family": str(case.flow_family),
        "lane_pattern_index": int(case.lane_pattern_index),
        "scene_title": str(case.scene_title),
        "lanes": list(case.lanes),
        "lane_count": int(len(case.lanes)),
        "node_count": int(len(case.nodes)),
        "edge_count": int(len(case.edges)),
        "decision_count": int(sum(1 for node in case.nodes if str(node["role"]) == "decision")),
        "condition_map": dict(case.condition_map),
        "scene_variant_probabilities": dict(case.scene_variant_probabilities),
        "layout_variant_probabilities": dict(case.layout_variant_probabilities),
        "style_variant_probabilities": dict(case.style_variant_probabilities),
    }


def _node_specs(case: ProcessFlowSceneCase) -> list[Dict[str, Any]]:
    """Return compact node specs for verifier trace inspection."""

    specs: list[Dict[str, Any]] = []
    for node in case.nodes:
        specs.append(
            {
                key: value
                for key, value in dict(node).items()
                if key not in {"center", "width", "height"}
            }
        )
    return specs


def _edge_specs(case: ProcessFlowSceneCase) -> list[Dict[str, Any]]:
    """Return compact edge specs for verifier trace inspection."""

    return [dict(edge) for edge in case.edges]


def _witness_kind(answer_binding: ProcessFlowAnswerBinding) -> str:
    """Map annotation schema to a compact symbolic witness type."""

    annotation_type = str(answer_binding.annotation_gt.type)
    if annotation_type == "bbox_map":
        return "keyed_path_support"
    if annotation_type == "segment_set":
        return "segment_id_set"
    return "bbox_id_set"


def build_process_flow_response(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    selected_branch: str,
    branch_probabilities: Mapping[str, float],
    namespace: str,
    binding_factory: BindingFactory,
) -> TaskOutput:
    """Resolve scene state, bind task-owned answer data, and assemble output."""

    case = build_scene_case(
        instance_seed=int(instance_seed),
        params=params,
        namespace=str(namespace),
    )
    background, background_meta = make_background_canvas(
        canvas_width=int(case.render_params.canvas_width),
        canvas_height=int(case.render_params.canvas_height),
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_BACKGROUND_DEFAULTS,
    )
    rendered = render_process_flow_scene(
        background,
        case=case,
        instance_seed=int(instance_seed),
        namespace=str(namespace),
    )
    image, post_noise_meta = apply_post_image_noise(
        rendered.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    rendered = RenderedProcessFlow(image=image, render_map=dict(rendered.render_map))
    prompt_binding, answer_binding = binding_factory(
        int(instance_seed),
        str(selected_branch),
        dict(branch_probabilities),
        case,
        rendered,
    )
    bundle_id = str(PROMPT_DEFAULTS.get("bundle_id", PROMPT_BUNDLE)).strip() or PROMPT_BUNDLE
    prompt_selection = render_task_prompt_variants(
        domain=DOMAIN,
        scene_id=SCENE,
        bundle_id=bundle_id,
        scene_key=PROMPT_SCENE_KEY,
        task_key=PROMPT_TASK_KEY,
        query_key=str(prompt_binding.prompt_branch_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots=dict(prompt_binding.dynamic_slots),
        instance_seed=int(instance_seed),
    )
    prompt_artifacts = build_prompt_trace_artifacts(prompt_selection)
    common_fields = common_trace_fields(case)
    probabilities = {
        str(key): float(value)
        for key, value in answer_binding.branch_probabilities.items()
    }
    params_payload = {
        **common_fields,
        "query_id": str(answer_binding.selected_branch),
        "prompt_query_key": str(prompt_binding.prompt_branch_key),
        "target": dict(answer_binding.target_payload),
        "answer_value": answer_binding.answer_gt.value,
        "query_id_probabilities": dict(probabilities),
    }
    query_spec = build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts,
        query_id=str(answer_binding.selected_branch),
        params=params_payload,
    )
    query_spec["scene_id"] = SCENE

    trace_payload = {
        "scene_ir": {
            "scene_id": SCENE,
            "scene_kind": "pages_process_flow_diagram",
            "entities": [
                {
                    "entity_id": str(node["node_id"]),
                    "entity_type": "process_step",
                    "label": str(node["label"]),
                    "lane": str(node["lane"]),
                    "role": str(node["role"]),
                    "status": str(node["status"]),
                    "shape": str(node["shape"]),
                    "bbox_id": str(node["bbox_id"]),
                }
                for node in case.nodes
            ],
            "relations": {
                "query_id": str(answer_binding.selected_branch),
                "prompt_query_key": str(prompt_binding.prompt_branch_key),
                "scene_variant": str(case.scene_variant),
                "layout_variant": str(case.layout_variant),
                "style_variant": str(case.style_variant),
                "flow_family": str(case.flow_family),
            },
        },
        "query_spec": query_spec,
        "render_spec": {
            "scene_id": SCENE,
            "query_id": str(answer_binding.selected_branch),
            "prompt_query_key": str(prompt_binding.prompt_branch_key),
            "scene_variant": str(case.scene_variant),
            "layout_variant": str(case.layout_variant),
            "style_variant": str(case.style_variant),
            "flow_family": str(case.flow_family),
            "geometry_seed": int(instance_seed),
            "canvas_width": int(case.render_params.canvas_width),
            "canvas_height": int(case.render_params.canvas_height),
            "layout_jitter": dict(case.render_params.layout_jitter_meta),
            "background_style": dict(background_meta),
            "post_image_noise": dict(post_noise_meta),
            "panel_bbox_px": list(rendered.render_map["panel_bbox_px"]),
        },
        "render_map": dict(rendered.render_map),
        "execution_trace": {
            **common_fields,
            "query_id": str(answer_binding.selected_branch),
            "prompt_query_key": str(prompt_binding.prompt_branch_key),
            "question_format": str(answer_binding.question_format),
            "view_family": SCENE,
            "node_specs": _node_specs(case),
            "edge_specs": _edge_specs(case),
            "query": dict(answer_binding.target_payload),
            "answer": answer_binding.answer_gt.to_dict(),
            "annotation_ids": list(answer_binding.annotation_ids),
            "annotation_key_to_bbox_id": dict(answer_binding.annotation_key_to_bbox_id),
            "supporting_bbox_ids": (
                []
                if str(answer_binding.annotation_gt.type) == "segment_set"
                else list(answer_binding.annotation_ids)
            ),
            "supporting_segment_ids": (
                list(answer_binding.annotation_ids)
                if str(answer_binding.annotation_gt.type) == "segment_set"
                else []
            ),
        },
        "witness_symbolic": {
            "type": _witness_kind(answer_binding),
            "ids": list(answer_binding.annotation_ids),
            "keys": list(answer_binding.annotation_key_to_bbox_id.keys()),
        },
        "projected_annotation": dict(answer_binding.projected_annotation),
        "background": dict(background_meta),
        "post_image_noise": dict(post_noise_meta),
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
