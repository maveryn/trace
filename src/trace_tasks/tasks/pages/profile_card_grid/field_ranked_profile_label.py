"""Profile-card-grid task that selects a profile by numeric field rank."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.registry import register_task

from . import _lifecycle
from .shared.defaults import DOMAIN


TASK_ID = "task_pages__profile_card_grid__field_ranked_profile_label"
HIGHEST_FIELD_PROFILE_LABEL_QUERY_ID = "highest_field_profile_label"
LOWEST_FIELD_PROFILE_LABEL_QUERY_ID = "lowest_field_profile_label"
NTH_HIGHEST_FIELD_PROFILE_LABEL_QUERY_ID = "nth_highest_field_profile_label"
NTH_LOWEST_FIELD_PROFILE_LABEL_QUERY_ID = "nth_lowest_field_profile_label"
EXTREMUM_QUERY_IDS = (
    HIGHEST_FIELD_PROFILE_LABEL_QUERY_ID,
    LOWEST_FIELD_PROFILE_LABEL_QUERY_ID,
)
NTH_RANK_QUERY_IDS = (
    NTH_HIGHEST_FIELD_PROFILE_LABEL_QUERY_ID,
    NTH_LOWEST_FIELD_PROFILE_LABEL_QUERY_ID,
)
SUPPORTED_QUERY_IDS = (
    HIGHEST_FIELD_PROFILE_LABEL_QUERY_ID,
    LOWEST_FIELD_PROFILE_LABEL_QUERY_ID,
    NTH_HIGHEST_FIELD_PROFILE_LABEL_QUERY_ID,
    NTH_LOWEST_FIELD_PROFILE_LABEL_QUERY_ID,
)
RANK_DIRECTION_BY_QUERY_ID = {
    HIGHEST_FIELD_PROFILE_LABEL_QUERY_ID: "highest",
    LOWEST_FIELD_PROFILE_LABEL_QUERY_ID: "lowest",
    NTH_HIGHEST_FIELD_PROFILE_LABEL_QUERY_ID: "highest",
    NTH_LOWEST_FIELD_PROFILE_LABEL_QUERY_ID: "lowest",
}
IS_NTH_RANK_QUERY_ID = {
    HIGHEST_FIELD_PROFILE_LABEL_QUERY_ID: False,
    LOWEST_FIELD_PROFILE_LABEL_QUERY_ID: False,
    NTH_HIGHEST_FIELD_PROFILE_LABEL_QUERY_ID: True,
    NTH_LOWEST_FIELD_PROFILE_LABEL_QUERY_ID: True,
}
QUESTION_FORMAT_BY_QUERY_ID = {
    HIGHEST_FIELD_PROFILE_LABEL_QUERY_ID: "profile_card_grid_field_ranked_profile_label",
    LOWEST_FIELD_PROFILE_LABEL_QUERY_ID: "profile_card_grid_field_ranked_profile_label",
    NTH_HIGHEST_FIELD_PROFILE_LABEL_QUERY_ID: "profile_card_grid_field_ranked_profile_label",
    NTH_LOWEST_FIELD_PROFILE_LABEL_QUERY_ID: "profile_card_grid_field_ranked_profile_label",
}


def _bind_field_ranked_profile_label(
    instance_seed: int,
    params: Mapping[str, Any],
    selected_branch: str,
    branch_probabilities: Mapping[str, float],
    case,
    rendered,
):
    """Bind a numeric-field rank branch to the selected profile answer."""

    return _lifecycle.numeric_ordering_binding(
        instance_seed=int(instance_seed),
        params=params,
        selected_branch=str(selected_branch),
        branch_probabilities=branch_probabilities,
        case=case,
        rendered=rendered,
        public_task=TASK_ID,
        rank_direction=str(RANK_DIRECTION_BY_QUERY_ID[str(selected_branch)]),
        ranked=bool(IS_NTH_RANK_QUERY_ID[str(selected_branch)]),
        question_format=str(QUESTION_FORMAT_BY_QUERY_ID[str(selected_branch)]),
    )


@register_task
class PagesProfileCardGridFieldRankedProfileLabelTask:
    """Find the profile at a requested rank after sorting a numeric field."""

    task_id = TASK_ID
    reasoning_operations = ('ranking',)
    domain = DOMAIN
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Mapping[str, Any], max_attempts: int):
        del max_attempts
        selected_branch, branch_probabilities, task_params = _lifecycle.select_public_branch(
            instance_seed=int(instance_seed),
            params=params,
            supported=SUPPORTED_QUERY_IDS,
            default=HIGHEST_FIELD_PROFILE_LABEL_QUERY_ID,
            public_task=TASK_ID,
        )
        return _lifecycle.render_bound_profile_card_grid(
            instance_seed=int(instance_seed),
            params=task_params,
            selected_branch=selected_branch,
            branch_probabilities=branch_probabilities,
            include_numeric_fields=True,
            binding_factory=_bind_field_ranked_profile_label,
        )


__all__ = [
    "EXTREMUM_QUERY_IDS",
    "HIGHEST_FIELD_PROFILE_LABEL_QUERY_ID",
    "IS_NTH_RANK_QUERY_ID",
    "LOWEST_FIELD_PROFILE_LABEL_QUERY_ID",
    "NTH_RANK_QUERY_IDS",
    "NTH_HIGHEST_FIELD_PROFILE_LABEL_QUERY_ID",
    "NTH_LOWEST_FIELD_PROFILE_LABEL_QUERY_ID",
    "QUESTION_FORMAT_BY_QUERY_ID",
    "RANK_DIRECTION_BY_QUERY_ID",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
    "PagesProfileCardGridFieldRankedProfileLabelTask",
]
