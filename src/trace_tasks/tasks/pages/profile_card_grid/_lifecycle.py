"""Scene-private response assembly for profile-card-grid public tasks."""

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

from .shared.defaults import DOMAIN, PROMPT_BUNDLE, PROMPT_SCENE_KEY, PROMPT_TASK_KEY, SCENE
from .shared.annotations import augment_numeric_candidates_with_boxes, card_bbox, ordering_supporting_bboxes
from .shared.rendering import render_profile_card_case
from .shared.sampling import (
    build_profile_card_case,
    card_by_profile_id,
    numeric_candidates,
    profile_rank_ordinal,
    resolve_profile_numeric_field,
    resolve_profile_rank_position,
    sort_numeric_candidates,
)
from .shared.state import ProfileCardGridCase, RenderedProfileCardGridBundle


@dataclass(frozen=True)
class ProfileCardPromptBinding:
    """Task-owned prompt branch and dynamic prompt slots."""

    prompt_branch_key: str
    dynamic_slots: Mapping[str, Any]


@dataclass(frozen=True)
class ProfileCardAnswerBinding:
    """Task-owned answer, annotation, and trace payload fields."""

    answer_gt: TypedValue
    annotation_gt: TypedValue
    selected_branch: str
    branch_probabilities: Mapping[str, float]
    target_payload: Mapping[str, Any]
    candidate_profiles: tuple[Mapping[str, Any], ...]
    question_format: str
    projected_annotation: Mapping[str, Any]
    supporting_bboxes: Mapping[str, Any]
    extra_trace_fields: Mapping[str, Any]


BindingFactory = Callable[
    [int, Mapping[str, Any], str, Mapping[str, float], ProfileCardGridCase, RenderedProfileCardGridBundle],
    tuple[ProfileCardPromptBinding, ProfileCardAnswerBinding],
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
    annotation_bbox: list[float],
    supporting_bboxes: Mapping[str, Any],
    selected_branch: str,
    branch_probabilities: Mapping[str, float],
    answer_value: str,
    target_payload: Mapping[str, Any],
    candidate_profiles: tuple[Mapping[str, Any], ...] = (),
    question_format: str,
    projected_annotation: Mapping[str, Any] | None = None,
    extra_trace_fields: Mapping[str, Any] | None = None,
) -> ProfileCardAnswerBinding:
    """Build a string-answer binding from task-owned annotation data."""

    bbox = [float(value) for value in annotation_bbox]
    supporting_map = {str(key): [float(coord) for coord in value] for key, value in dict(supporting_bboxes).items()}
    projected = (
        dict(projected_annotation)
        if projected_annotation is not None
        else {
            "type": "bbox",
            "bbox": list(bbox),
            "pixel_bbox": list(bbox),
            "supporting_bboxes": dict(supporting_map),
            "target_profile_id": str(target_payload["profile_id"]),
            "field_label": str(target_payload["field_label"]),
        }
    )
    return ProfileCardAnswerBinding(
        answer_gt=TypedValue(type="string", value=str(answer_value)),
        annotation_gt=TypedValue(type="bbox", value=list(bbox)),
        selected_branch=str(selected_branch),
        branch_probabilities=dict(branch_probabilities),
        target_payload=dict(target_payload),
        candidate_profiles=tuple(dict(candidate) for candidate in candidate_profiles),
        question_format=str(question_format),
        projected_annotation=dict(projected),
        supporting_bboxes=dict(supporting_map),
        extra_trace_fields=dict(extra_trace_fields or {}),
    )


def numeric_ordering_binding(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    selected_branch: str,
    branch_probabilities: Mapping[str, float],
    case: ProfileCardGridCase,
    rendered: RenderedProfileCardGridBundle,
    public_task: str,
    rank_direction: str,
    ranked: bool,
    question_format: str,
) -> tuple[ProfileCardPromptBinding, ProfileCardAnswerBinding]:
    """Bind a numeric-field ordering query to the selected visible profile."""

    target_field = resolve_profile_numeric_field(
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{public_task}.{selected_branch}",
    )
    if bool(ranked):
        rank_position, rank_support, rank_probabilities = resolve_profile_rank_position(
            params=params,
            card_count=int(case.card_count),
            instance_seed=int(instance_seed),
            namespace=f"{public_task}.{selected_branch}",
        )
    else:
        rank_position = 1
        rank_support = (1,)
        rank_probabilities = {1: 1.0}
    rank_ordinal = profile_rank_ordinal(int(rank_position)) if bool(ranked) else ""
    ordered_candidates = sort_numeric_candidates(
        numeric_candidates(cards=case.spec.cards, target_field=str(target_field)),
        rank_direction=str(rank_direction),
    )
    target_candidate = dict(ordered_candidates[int(rank_position) - 1])
    card = card_by_profile_id(case.spec.cards, str(target_candidate["profile_id"]))
    target_value = str(card.fields[str(target_field)])
    target_payload = {
        "profile_id": str(card.profile_id),
        "profile_name": str(card.name),
        "field_label": str(target_field),
        "field_value": str(target_value),
        "field_numeric_value": int(card.numeric_fields[str(target_field)]),
    }
    extra_trace_fields = {
        "extremum_direction": "" if bool(ranked) else str(rank_direction),
        "rank_direction": str(rank_direction),
        "rank_position": int(rank_position),
        "rank_ordinal": str(rank_ordinal),
        "rank_position_support": [int(value) for value in rank_support],
        "rank_position_probabilities": {
            str(int(key)): float(value) for key, value in dict(rank_probabilities).items()
        },
    }
    dynamic_slots = {"field_label": str(target_field)}
    if bool(ranked):
        dynamic_slots["rank_ordinal"] = str(rank_ordinal)
    return (
        ProfileCardPromptBinding(
            prompt_branch_key=str(selected_branch),
            dynamic_slots=dynamic_slots,
        ),
        string_binding(
            annotation_bbox=card_bbox(
                card=card,
                rendered=rendered.rendered_grid,
            ),
            supporting_bboxes=ordering_supporting_bboxes(
                card=card,
                field_label=str(target_field),
                rendered=rendered.rendered_grid,
            ),
            selected_branch=str(selected_branch),
            branch_probabilities=branch_probabilities,
            answer_value=str(card.name),
            target_payload=target_payload,
            candidate_profiles=tuple(
                augment_numeric_candidates_with_boxes(
                    candidates=ordered_candidates,
                    rendered=rendered.rendered_grid,
                    field_label=str(target_field),
                )
            ),
            question_format=str(question_format),
            extra_trace_fields=extra_trace_fields,
        ),
    )


def _common_trace_fields(case: ProfileCardGridCase) -> Dict[str, Any]:
    return {
        "scene_variant": str(case.scene_variant),
        "card_count": int(case.card_count),
        "card_count_support": [int(value) for value in case.card_count_support],
        "card_count_probabilities": dict(case.card_count_probabilities),
        "scene_variant_probabilities": dict(case.scene_variant_probabilities),
        "page_text_resources": dict(case.spec.text_resource_metadata),
    }


def build_profile_card_response(
    *,
    instance_seed: int,
    case: ProfileCardGridCase,
    rendered: RenderedProfileCardGridBundle,
    prompt_binding: ProfileCardPromptBinding,
    answer_binding: ProfileCardAnswerBinding,
) -> TaskOutput:
    """Assemble one complete profile-card-grid task response."""

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
    common_fields = _common_trace_fields(case)
    probabilities = {str(key): float(value) for key, value in answer_binding.branch_probabilities.items()}
    query_params = {
        **common_fields,
        **dict(answer_binding.extra_trace_fields),
        "query_id": str(answer_binding.selected_branch),
        "prompt_query_key": str(prompt_binding.prompt_branch_key),
        "target_profile": dict(answer_binding.target_payload),
        "candidate_profiles": [dict(candidate) for candidate in answer_binding.candidate_profiles],
        "supporting_bboxes": dict(answer_binding.supporting_bboxes),
        "target_answer": answer_binding.answer_gt.value,
        "query_id_probabilities": dict(probabilities),
    }
    query_spec = build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts,
        query_id=str(answer_binding.selected_branch),
        params=dict(query_params),
    )
    query_spec["scene_id"] = SCENE

    rendered_grid = rendered.rendered_grid
    trace_payload = {
        "scene_ir": {
            "scene_id": SCENE,
            "scene_kind": "pages_profile_card_grid",
            "entities": [dict(entity) for entity in rendered_grid.entities],
            "relations": {
                "query_id": str(answer_binding.selected_branch),
                "prompt_query_key": str(prompt_binding.prompt_branch_key),
                "scene_variant": str(case.scene_variant),
                "target_profile": dict(answer_binding.target_payload),
                "candidate_profiles": [dict(candidate) for candidate in answer_binding.candidate_profiles],
                "answer_value": answer_binding.answer_gt.value,
                **dict(answer_binding.extra_trace_fields),
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
            "panel_bbox_px": list(rendered_grid.panel_bbox_px),
            "layout": dict(rendered_grid.layout_meta),
            "page_text_resources": dict(case.spec.text_resource_metadata),
        },
        "render_map": {
            "image_id": "img0",
            "panel_bbox_px": list(rendered_grid.panel_bbox_px),
            "card_bboxes_px": dict(rendered_grid.card_bboxes_px),
            "name_bboxes_px": dict(rendered_grid.name_bboxes_px),
            "field_label_bboxes_px": dict(rendered_grid.field_label_bboxes_px),
            "field_value_bboxes_px": dict(rendered_grid.field_value_bboxes_px),
        },
        "execution_trace": {
            **query_params,
            "question_format": str(answer_binding.question_format),
            "answer_value": answer_binding.answer_gt.value,
            "cards": [dict(trace) for trace in rendered_grid.card_traces],
        },
        "witness_symbolic": {
            "type": str(answer_binding.question_format),
            "target_profile_id": str(answer_binding.target_payload["profile_id"]),
            "field_label": str(answer_binding.target_payload["field_label"]),
            "field_value": str(answer_binding.target_payload["field_value"]),
            "answer_value": answer_binding.answer_gt.value,
            "annotation_role": "target_bbox",
            "supporting_bbox_keys": list(answer_binding.supporting_bboxes.keys()),
            **dict(answer_binding.extra_trace_fields),
        },
        "projected_annotation": dict(answer_binding.projected_annotation),
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


def render_bound_profile_card_grid(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    selected_branch: str,
    branch_probabilities: Mapping[str, float],
    include_numeric_fields: bool,
    include_filter_field: bool = False,
    binding_factory: BindingFactory,
) -> TaskOutput:
    """Resolve scene state, bind task-owned answer data, and assemble response."""

    case = build_profile_card_case(
        int(instance_seed),
        params=params,
        include_numeric_fields=bool(include_numeric_fields),
        include_filter_field=bool(include_filter_field),
    )
    rendered = render_profile_card_case(
        instance_seed=int(instance_seed),
        params=params,
        case=case,
    )
    prompt_binding, answer_binding = binding_factory(
        int(instance_seed),
        params,
        str(selected_branch),
        dict(branch_probabilities),
        case,
        rendered,
    )
    return build_profile_card_response(
        instance_seed=int(instance_seed),
        case=case,
        rendered=rendered,
        prompt_binding=prompt_binding,
        answer_binding=answer_binding,
    )
