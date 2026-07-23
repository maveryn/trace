"""Control-board task for selecting the group with the most matching controls."""

from __future__ import annotations

from typing import Any, Mapping, Sequence, Tuple

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.deterministic_sampling import resolve_selection_index

from . import _lifecycle
from .shared.annotations import target_group_bbox
from .shared.defaults import DISABLED_MODE, DOMAIN, GENERATION_DEFAULTS, NAMESPACE_ROOT, SCENE_VARIANTS, SELECTED_ENABLED_MODE
from .shared.rendering import render_control_board_case
from .shared.sampling import build_control_board_case_from_group_state_counts


TASK_ID = "task_pages__control_board__state_extremum_group_label"
DISABLED_QUERY_ID = "disabled_extremum_group_label"
SELECTED_ENABLED_QUERY_ID = "selected_enabled_extremum_group_label"
SUPPORTED_QUERY_IDS = (DISABLED_QUERY_ID, SELECTED_ENABLED_QUERY_ID)
DEFAULT_QUERY_ID = DISABLED_QUERY_ID
TARGET_STATE_COUNT_SUPPORT = (3, 4, 5)

_COUNT_MODE_BY_QUERY_ID = {
    DISABLED_QUERY_ID: DISABLED_MODE,
    SELECTED_ENABLED_QUERY_ID: SELECTED_ENABLED_MODE,
}
_QUESTION_FORMAT_BY_QUERY_ID = {
    DISABLED_QUERY_ID: "control_board_disabled_extremum_group_label",
    SELECTED_ENABLED_QUERY_ID: "control_board_selected_enabled_extremum_group_label",
}


def _normalize_int_support(params: Mapping[str, Any], key: str, fallback: Sequence[int]) -> Tuple[int, ...]:
    raw_values = params.get(str(key), group_default(GENERATION_DEFAULTS, str(key), fallback))
    support: list[int] = []
    for raw_value in raw_values:
        value = int(raw_value)
        if value not in support:
            support.append(value)
    if not support:
        raise ValueError(f"{key} must not be empty")
    return tuple(int(value) for value in support)


def _select_target_state_count(*, instance_seed: int, params: Mapping[str, Any], query_id: str) -> tuple[int, tuple[int, ...], dict[str, float]]:
    support = _normalize_int_support(params, "extremum_state_count_support", TARGET_STATE_COUNT_SUPPORT)
    explicit = params.get("target_state_count")
    if explicit is not None:
        selected = int(explicit)
        if int(selected) not in set(support):
            raise ValueError("target_state_count must be in extremum_state_count_support")
        return int(selected), tuple(support), {str(selected): 1.0}
    index = int(
        resolve_selection_index(
            params=params,
            instance_seed=int(instance_seed),
            namespace=f"{NAMESPACE_ROOT}.state_extremum.target_count.{query_id}",
        )
        % len(support)
    )
    selected = int(support[int(index)])
    return int(selected), tuple(support), {str(value): 1.0 / float(len(support)) for value in support}


def _select_target_group_index(*, instance_seed: int, params: Mapping[str, Any], query_id: str, group_count: int) -> tuple[int, dict[str, float]]:
    explicit = params.get("target_group_index")
    if explicit is not None:
        selected = int(explicit)
        if selected < 0 or selected >= int(group_count):
            raise ValueError("target_group_index is out of range")
        return int(selected), {str(selected): 1.0}
    index = int(
        resolve_selection_index(
            params=params,
            instance_seed=int(instance_seed),
            namespace=f"{NAMESPACE_ROOT}.state_extremum.target_group.{query_id}",
        )
        % int(group_count)
    )
    return int(index), {str(value): 1.0 / float(group_count) for value in range(int(group_count))}


def _sample_unique_extremum_counts(
    *,
    instance_seed: int,
    query_id: str,
    target_group_index: int,
    target_state_count: int,
    group_count: int,
) -> tuple[int, ...]:
    """Return per-group state counts with one unique maximum group."""

    if int(target_state_count) <= 0:
        raise ValueError("target_state_count must be positive for an extremum group")
    counts = [0 for _ in range(int(group_count))]
    counts[int(target_group_index)] = int(target_state_count)
    other_groups = [idx for idx in range(int(group_count)) if int(idx) != int(target_group_index)]
    rng = spawn_rng(
        int(instance_seed),
        f"{NAMESPACE_ROOT}.state_extremum.distractor_counts.{query_id}.{target_group_index}.{target_state_count}",
    )
    close_group = int(other_groups[int(rng.randrange(len(other_groups)))])
    for group_index in other_groups:
        if int(group_index) == int(close_group):
            counts[int(group_index)] = max(0, int(target_state_count) - 1)
            continue
        low = 0 if int(target_state_count) <= 3 else 1
        high = max(0, int(target_state_count) - 1)
        counts[int(group_index)] = int(rng.randint(int(low), int(high)))
    return tuple(int(value) for value in counts)


def _bind_state_extremum_group(selected_branch, branch_probabilities, case, rendered, target_count_probabilities, target_group_probabilities, group_state_counts):
    prompt_binding = _lifecycle.ControlBoardPromptBinding(
        prompt_branch_key=str(selected_branch),
        dynamic_slots={},
    )
    answer_binding = _lifecycle.string_binding(
        annotation_kind="bbox",
        annotation_value=target_group_bbox(case, rendered),
        selected_branch=str(selected_branch),
        branch_probabilities=dict(branch_probabilities),
        answer_value=str(case.target_group_name),
        target_payload={
            "count_mode": str(case.count_mode),
            "target_group_name": str(case.target_group_name),
            "target_group_index": int(case.target_group_index),
            "target_state_count": int(case.answer_value),
            "group_state_counts": [int(value) for value in group_state_counts],
            "target_state_count_probabilities": dict(target_count_probabilities),
            "target_group_probabilities": dict(target_group_probabilities),
        },
        question_format=str(_QUESTION_FORMAT_BY_QUERY_ID[str(selected_branch)]),
    )
    return prompt_binding, answer_binding


@register_task
class PagesControlBoardStateExtremumGroupLabelTask:
    """Select the visible control group with the most matching state controls."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'ranking', 'logical_composition')
    domain = DOMAIN
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed, *, params, max_attempts):
        """Sample a unique argmax group, render the board, and bind the winning panel bbox."""

        del max_attempts
        selected_branch, branch_probabilities, task_params = _lifecycle.select_public_branch(
            instance_seed=int(instance_seed),
            params=params,
            supported=SUPPORTED_QUERY_IDS,
            default=DEFAULT_QUERY_ID,
            public_task=TASK_ID,
        )
        target_state_count, count_support, target_count_probabilities = _select_target_state_count(
            instance_seed=int(instance_seed),
            params=task_params,
            query_id=str(selected_branch),
        )
        target_group_index, target_group_probabilities = _select_target_group_index(
            instance_seed=int(instance_seed),
            params=task_params,
            query_id=str(selected_branch),
            group_count=4,
        )
        group_state_counts = _sample_unique_extremum_counts(
            instance_seed=int(instance_seed),
            query_id=str(selected_branch),
            target_group_index=int(target_group_index),
            target_state_count=int(target_state_count),
            group_count=4,
        )
        case = build_control_board_case_from_group_state_counts(
            instance_seed=int(instance_seed),
            params=task_params,
            count_mode=str(_COUNT_MODE_BY_QUERY_ID[str(selected_branch)]),
            target_group_index=int(target_group_index),
            group_state_counts=tuple(int(value) for value in group_state_counts),
            answer_support=count_support,
        )
        rendered = render_control_board_case(instance_seed=int(instance_seed), params=task_params, case=case)
        prompt_binding, answer_binding = _bind_state_extremum_group(
            str(selected_branch),
            branch_probabilities,
            case,
            rendered,
            target_count_probabilities,
            target_group_probabilities,
            group_state_counts,
        )
        return _lifecycle.build_control_board_response(
            instance_seed=int(instance_seed),
            public_task_id=TASK_ID,
            case=case,
            rendered=rendered,
            prompt_binding=prompt_binding,
            answer_binding=answer_binding,
        )


__all__ = [
    "DEFAULT_QUERY_ID",
    "DISABLED_QUERY_ID",
    "SCENE_VARIANTS",
    "SELECTED_ENABLED_QUERY_ID",
    "SUPPORTED_QUERY_IDS",
    "TARGET_STATE_COUNT_SUPPORT",
    "TASK_ID",
    "PagesControlBoardStateExtremumGroupLabelTask",
]
