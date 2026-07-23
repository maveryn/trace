"""Profile-card-grid task that selects a ranked profile within a visible filter group."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.registry import register_task

from . import _lifecycle
from .shared.annotations import augment_numeric_candidates_with_boxes, card_bbox
from .shared.defaults import DOMAIN
from .shared.sampling import (
    card_by_profile_id,
    numeric_candidates,
    profile_rank_ordinal,
    resolve_profile_numeric_field,
    resolve_profile_rank_position,
    sort_numeric_candidates,
)
from .shared.state import ProfileCard


TASK_ID = "task_pages__profile_card_grid__filtered_ranked_profile_label"
FILTERED_HIGHEST_PROFILE_LABEL_QUERY_ID = "filtered_highest_profile_label"
FILTERED_LOWEST_PROFILE_LABEL_QUERY_ID = "filtered_lowest_profile_label"
FILTERED_NTH_HIGHEST_PROFILE_LABEL_QUERY_ID = "filtered_nth_highest_profile_label"
FILTERED_NTH_LOWEST_PROFILE_LABEL_QUERY_ID = "filtered_nth_lowest_profile_label"
EXTREMUM_QUERY_IDS = (
    FILTERED_HIGHEST_PROFILE_LABEL_QUERY_ID,
    FILTERED_LOWEST_PROFILE_LABEL_QUERY_ID,
)
NTH_RANK_QUERY_IDS = (
    FILTERED_NTH_HIGHEST_PROFILE_LABEL_QUERY_ID,
    FILTERED_NTH_LOWEST_PROFILE_LABEL_QUERY_ID,
)
SUPPORTED_QUERY_IDS = (
    FILTERED_HIGHEST_PROFILE_LABEL_QUERY_ID,
    FILTERED_LOWEST_PROFILE_LABEL_QUERY_ID,
    FILTERED_NTH_HIGHEST_PROFILE_LABEL_QUERY_ID,
    FILTERED_NTH_LOWEST_PROFILE_LABEL_QUERY_ID,
)
RANK_DIRECTION_BY_QUERY_ID = {
    FILTERED_HIGHEST_PROFILE_LABEL_QUERY_ID: "highest",
    FILTERED_LOWEST_PROFILE_LABEL_QUERY_ID: "lowest",
    FILTERED_NTH_HIGHEST_PROFILE_LABEL_QUERY_ID: "highest",
    FILTERED_NTH_LOWEST_PROFILE_LABEL_QUERY_ID: "lowest",
}
IS_NTH_RANK_QUERY_ID = {
    FILTERED_HIGHEST_PROFILE_LABEL_QUERY_ID: False,
    FILTERED_LOWEST_PROFILE_LABEL_QUERY_ID: False,
    FILTERED_NTH_HIGHEST_PROFILE_LABEL_QUERY_ID: True,
    FILTERED_NTH_LOWEST_PROFILE_LABEL_QUERY_ID: True,
}
QUESTION_FORMAT_BY_QUERY_ID = {
    FILTERED_HIGHEST_PROFILE_LABEL_QUERY_ID: "profile_card_grid_filtered_ranked_profile_label",
    FILTERED_LOWEST_PROFILE_LABEL_QUERY_ID: "profile_card_grid_filtered_ranked_profile_label",
    FILTERED_NTH_HIGHEST_PROFILE_LABEL_QUERY_ID: "profile_card_grid_filtered_ranked_profile_label",
    FILTERED_NTH_LOWEST_PROFILE_LABEL_QUERY_ID: "profile_card_grid_filtered_ranked_profile_label",
}


def _group_cards_by_filter_value(cards: Sequence[ProfileCard], *, filter_field_label: str) -> Dict[str, list[ProfileCard]]:
    groups: Dict[str, list[ProfileCard]] = {}
    for card in cards:
        filter_value = str(card.fields[str(filter_field_label)])
        groups.setdefault(filter_value, []).append(card)
    return groups


def _select_filter_value(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    selected_branch: str,
    groups: Mapping[str, Sequence[ProfileCard]],
    rank_position: int,
) -> str:
    explicit_value = params.get("filter_field_value")
    valid_values = sorted(
        str(value)
        for value, grouped_cards in groups.items()
        if len(tuple(grouped_cards)) >= int(rank_position)
    )
    if not valid_values:
        raise ValueError("no filter group has enough profiles for the requested rank")
    if explicit_value is not None:
        value = str(explicit_value)
        if value not in set(valid_values):
            raise ValueError(f"filter_field_value must be one of {valid_values}")
        return value
    rng = spawn_rng(
        int(instance_seed),
        f"pages.profile_card_grid.{TASK_ID}.{selected_branch}.filter_field_value.{int(rank_position)}",
    )
    return str(valid_values[int(rng.randrange(len(valid_values)))])


def _filtered_supporting_bboxes(
    *,
    card: ProfileCard,
    filter_field_label: str,
    rank_field_label: str,
    rendered,
) -> Dict[str, list[float]]:
    profile_id = str(card.profile_id)
    filter_field = str(filter_field_label)
    rank_field = str(rank_field_label)
    grid = rendered.rendered_grid
    return {
        "target_profile": [float(value) for value in grid.name_bboxes_px[profile_id]],
        "filter_field_label": [float(value) for value in grid.field_label_bboxes_px[profile_id][filter_field]],
        "filter_value": [float(value) for value in grid.field_value_bboxes_px[profile_id][filter_field]],
        "rank_field_label": [float(value) for value in grid.field_label_bboxes_px[profile_id][rank_field]],
        "target_rank_value": [float(value) for value in grid.field_value_bboxes_px[profile_id][rank_field]],
    }


def _bind_filtered_ranked_profile_label(
    instance_seed: int,
    params: Mapping[str, Any],
    selected_branch: str,
    branch_probabilities: Mapping[str, float],
    case,
    rendered,
):
    """Bind a filtered numeric ordering query to the selected visible profile."""

    rank_direction = str(RANK_DIRECTION_BY_QUERY_ID[str(selected_branch)])
    ranked = bool(IS_NTH_RANK_QUERY_ID[str(selected_branch)])
    target_field = resolve_profile_numeric_field(
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}.{selected_branch}",
    )
    if bool(ranked):
        rank_position, rank_support, rank_probabilities = resolve_profile_rank_position(
            params=params,
            card_count=int(case.card_count),
            instance_seed=int(instance_seed),
            namespace=f"{TASK_ID}.{selected_branch}",
        )
    else:
        rank_position = 1
        rank_support = (1,)
        rank_probabilities = {1: 1.0}
    rank_ordinal = profile_rank_ordinal(int(rank_position)) if bool(ranked) else ""
    filter_field = str(case.spec.filter_field_label)
    if not filter_field:
        raise ValueError("filtered ranked profile task requires a visible filter field")
    groups = _group_cards_by_filter_value(case.spec.cards, filter_field_label=str(filter_field))
    filter_value = _select_filter_value(
        instance_seed=int(instance_seed),
        params=params,
        selected_branch=str(selected_branch),
        groups=groups,
        rank_position=int(rank_position),
    )
    filtered_cards = tuple(groups[str(filter_value)])
    ordered_candidates = sort_numeric_candidates(
        numeric_candidates(cards=filtered_cards, target_field=str(target_field)),
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
        "filter_field_label": str(filter_field),
        "filter_field_value": str(filter_value),
        "filter_group_size": int(len(filtered_cards)),
    }
    extra_trace_fields = {
        "filter_field_label": str(filter_field),
        "filter_field_value": str(filter_value),
        "filter_group_size": int(len(filtered_cards)),
        "extremum_direction": "" if bool(ranked) else str(rank_direction),
        "rank_direction": str(rank_direction),
        "rank_position": int(rank_position),
        "rank_ordinal": str(rank_ordinal),
        "rank_position_support": [int(value) for value in rank_support],
        "rank_position_probabilities": {
            str(int(key)): float(value) for key, value in dict(rank_probabilities).items()
        },
    }
    dynamic_slots = {
        "filter_field_label": str(filter_field),
        "filter_field_value": str(filter_value),
        "field_label": str(target_field),
    }
    if bool(ranked):
        dynamic_slots["rank_ordinal"] = str(rank_ordinal)
    return (
        _lifecycle.ProfileCardPromptBinding(
            prompt_branch_key=str(selected_branch),
            dynamic_slots=dynamic_slots,
        ),
        _lifecycle.string_binding(
            annotation_bbox=card_bbox(
                card=card,
                rendered=rendered.rendered_grid,
            ),
            supporting_bboxes=_filtered_supporting_bboxes(
                card=card,
                filter_field_label=str(filter_field),
                rank_field_label=str(target_field),
                rendered=rendered,
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
            question_format=str(QUESTION_FORMAT_BY_QUERY_ID[str(selected_branch)]),
            extra_trace_fields=extra_trace_fields,
        ),
    )


@register_task
class PagesProfileCardGridFilteredRankedProfileLabelTask:
    """Find the ranked profile after applying a visible categorical filter."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'ranking')
    domain = DOMAIN
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Mapping[str, Any], max_attempts: int):
        del max_attempts
        selected_branch, branch_probabilities, task_params = _lifecycle.select_public_branch(
            instance_seed=int(instance_seed),
            params=params,
            supported=SUPPORTED_QUERY_IDS,
            default=FILTERED_HIGHEST_PROFILE_LABEL_QUERY_ID,
            public_task=TASK_ID,
        )
        return _lifecycle.render_bound_profile_card_grid(
            instance_seed=int(instance_seed),
            params=task_params,
            selected_branch=selected_branch,
            branch_probabilities=branch_probabilities,
            include_numeric_fields=True,
            include_filter_field=True,
            binding_factory=_bind_filtered_ranked_profile_label,
        )


__all__ = [
    "EXTREMUM_QUERY_IDS",
    "FILTERED_HIGHEST_PROFILE_LABEL_QUERY_ID",
    "FILTERED_LOWEST_PROFILE_LABEL_QUERY_ID",
    "FILTERED_NTH_HIGHEST_PROFILE_LABEL_QUERY_ID",
    "FILTERED_NTH_LOWEST_PROFILE_LABEL_QUERY_ID",
    "IS_NTH_RANK_QUERY_ID",
    "NTH_RANK_QUERY_IDS",
    "QUESTION_FORMAT_BY_QUERY_ID",
    "RANK_DIRECTION_BY_QUERY_ID",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
    "PagesProfileCardGridFilteredRankedProfileLabelTask",
]
