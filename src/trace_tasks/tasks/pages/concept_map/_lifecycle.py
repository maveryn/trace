"""Scene-private response assembly for concept-map public tasks."""

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

from .shared.annotations import (
    common_trace_fields,
    node_entities,
    projected_annotation_payload,
    selection_annotation_ids,
    selection_role_node_ids,
)
from .shared.defaults import DOMAIN, PROMPT_BUNDLE, PROMPT_SCENE_KEY, PROMPT_TASK_KEY, SCENE
from .shared.rendering import render_concept_map_case
from .shared.sampling import build_concept_map_case
from .shared.state import ConceptMapCase, RenderedConceptMap


@dataclass(frozen=True)
class ConceptMapPromptBinding:
    """Task-owned prompt key and dynamic prompt slots."""

    prompt_branch_key: str
    dynamic_slots: Mapping[str, Any]


@dataclass(frozen=True)
class ConceptMapAnswerBinding:
    """Task-owned answer, annotation, and trace payload fields."""

    answer_gt: TypedValue
    annotation_gt: TypedValue
    selected_branch: str
    branch_probabilities: Mapping[str, float]
    target_payload: Mapping[str, Any]
    question_format: str


BindingFactory = Callable[
    [str, Mapping[str, float], ConceptMapCase, RenderedConceptMap],
    tuple[ConceptMapPromptBinding, ConceptMapAnswerBinding],
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
    question_format: str,
) -> ConceptMapAnswerBinding:
    """Build an integer-answer binding from task-owned annotation data."""

    return ConceptMapAnswerBinding(
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
) -> ConceptMapAnswerBinding:
    """Build a string-answer binding from task-owned annotation data."""

    return ConceptMapAnswerBinding(
        answer_gt=TypedValue(type="string", value=str(answer_value)),
        annotation_gt=TypedValue(type=str(annotation_kind), value=annotation_value),
        selected_branch=str(selected_branch),
        branch_probabilities=dict(branch_probabilities),
        target_payload=dict(target_payload),
        question_format=str(question_format),
    )


def build_concept_map_response(
    *,
    instance_seed: int,
    case: ConceptMapCase,
    rendered: RenderedConceptMap,
    prompt_binding: ConceptMapPromptBinding,
    answer_binding: ConceptMapAnswerBinding,
) -> TaskOutput:
    """Assemble one complete concept-map task response."""

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
    common_fields = common_trace_fields(case)
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

    annotation_role_node_ids = selection_role_node_ids(case)
    annotation_role_ids = {
        str(role): f"node:{node_id}"
        for role, node_id in sorted(annotation_role_node_ids.items())
    }
    annotation_ids = selection_annotation_ids(case)
    projected_annotation = projected_annotation_payload(
        case,
        rendered,
        annotation_kind=str(answer_binding.annotation_gt.type),
    )
    if str(answer_binding.annotation_gt.type) == "bbox":
        witness_symbolic = {
            "type": "bbox_id",
            "id": str(annotation_ids[0]) if annotation_ids else "",
            "bbox": list(answer_binding.annotation_gt.value),
        }
    else:
        witness_symbolic = {
            "type": "bbox_id_set",
            "ids": list(annotation_ids),
            "annotation_role_ids": dict(annotation_role_ids),
            "value": list(annotation_ids),
        }
    trace_payload = {
        "scene_ir": {
            "scene_id": SCENE,
            "scene_kind": "pages_concept_map_diagram",
            "entities": node_entities(case),
            "relations": {
                "query_id": str(answer_binding.selected_branch),
                "prompt_query_key": str(prompt_binding.prompt_branch_key),
                "scene_variant": str(case.scene["layout_variant"]),
                "layout_variant": str(case.scene["layout_variant"]),
                "style_variant": str(case.scene["style_variant"]),
                "node_shape_profile": str(case.scene["node_shape_profile"]),
                "context_id": str(case.scene["context_id"]),
                "branches": list(common_fields["branches"]),
                "target": dict(answer_binding.target_payload),
                "answer_value": answer_binding.answer_gt.value,
            },
        },
        "query_spec": query_spec,
        "render_spec": {
            "scene_id": SCENE,
            "query_id": str(answer_binding.selected_branch),
            "scene_variant": str(case.scene["layout_variant"]),
            "layout_variant": str(case.scene["layout_variant"]),
            "style_variant": str(case.scene["style_variant"]),
            "node_shape_profile": str(case.scene["node_shape_profile"]),
            "geometry_seed": int(instance_seed),
            "canvas_width": int(case.scene["canvas_width"]),
            "canvas_height": int(case.scene["canvas_height"]),
            "layout_jitter": dict(case.scene["layout_jitter"]),
            "background_style": dict(rendered.background_meta),
            "post_image_noise": dict(rendered.post_noise_meta),
            "page_semantic_assets": dict(common_fields["page_semantic_assets"]),
        },
        "render_map": {
            "image_id": "img0",
            **dict(rendered.render_map),
        },
        "execution_trace": {
            **common_fields,
            "query_id": str(answer_binding.selected_branch),
            "prompt_query_key": str(prompt_binding.prompt_branch_key),
            "question_format": str(answer_binding.question_format),
            "target": dict(answer_binding.target_payload),
            "query": dict(answer_binding.target_payload),
            "answer": answer_binding.answer_gt.to_dict(),
            "answer_value": answer_binding.answer_gt.value,
            "annotation_ids": list(annotation_ids),
            "annotation_role_ids": dict(annotation_role_ids),
            "annotation_role_node_ids": dict(annotation_role_node_ids),
            "supporting_bbox_ids": list(annotation_ids),
            "query_id_probabilities": dict(probabilities),
        },
        "witness_symbolic": dict(witness_symbolic),
        "projected_annotation": dict(projected_annotation),
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


def render_bound_concept_map(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    selected_branch: str,
    branch_probabilities: Mapping[str, float],
    case_kind: str,
    case_defaults: Mapping[str, Any] | None,
    binding_factory: BindingFactory,
) -> TaskOutput:
    """Resolve scene state, bind task-owned answer data, and assemble response."""

    case = build_concept_map_case(
        instance_seed=int(instance_seed),
        params=params,
        case_kind=str(case_kind),
        case_defaults=case_defaults or {},
    )
    rendered = render_concept_map_case(
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
    return build_concept_map_response(
        instance_seed=int(instance_seed),
        case=case,
        rendered=rendered,
        prompt_binding=prompt_binding,
        answer_binding=answer_binding,
    )
