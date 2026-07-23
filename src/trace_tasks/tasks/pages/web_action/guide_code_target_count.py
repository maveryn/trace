"""Web-action task counting controls that share a visible guide code."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.core.seed import hash64
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.fixed_query import select_task_query_id

from . import _lifecycle


TASK_ID = "task_pages__web_action__guide_code_target_count"
SUPPORTED_QUERY_IDS = _lifecycle.SUPPORTED_GUIDE_CODE_COUNT_QUERY_IDS
DEFAULT_QUERY_ID = SUPPORTED_QUERY_IDS[0]


@register_task
class PagesWebActionGuideCodeTargetCountTask:
    """Count candidate controls that use the requested guide code."""

    task_id = TASK_ID
    reasoning_operations = ('counting',)
    domain = "pages"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Mapping[str, Any], max_attempts: int):
        """Select a count branch and pin its row/group count for stable answer distribution."""

        del max_attempts
        selected_branch, branch_probabilities, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=params,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=DEFAULT_QUERY_ID,
            task_id=TASK_ID,
            namespace=f"{TASK_ID}.query",
        )
        task_params = dict(task_params)
        sample_cursor = task_params.get("_sample_cursor")
        if sample_cursor is not None:
            desired_count = 3 + (abs(int(sample_cursor)) % 4)
        else:
            desired_count = 3 + (hash64(int(instance_seed), f"{TASK_ID}.{selected_branch}.answer_count") % 4)
        if str(selected_branch) == "click_guide_code_target_count":
            task_params.setdefault("web_click_item_count_min", int(desired_count))
            task_params.setdefault("web_click_item_count_max", int(desired_count))
        else:
            task_params.setdefault("web_click_item_count_min", 3)
            task_params.setdefault("web_click_item_count_max", 6)
        if str(selected_branch) == "type_field_guide_code_target_count":
            task_params.setdefault("web_type_section_count_min", int(desired_count))
            task_params.setdefault("web_type_section_count_max", int(desired_count))
        else:
            task_params.setdefault("web_type_section_count_min", 3)
            task_params.setdefault("web_type_section_count_max", 6)
        task_params.setdefault("web_type_field_count_min", 3)
        task_params.setdefault("web_type_field_count_max", 3)
        task_params.setdefault(
            "web_section_pool",
            ("Account", "Traveler", "Billing", "Delivery", "Notifications", "Preferences"),
        )
        if str(selected_branch) == "select_option_guide_code_target_count":
            task_params.setdefault("web_select_group_count_min", int(desired_count))
            task_params.setdefault("web_select_group_count_max", int(desired_count))
        else:
            task_params.setdefault("web_select_group_count_min", 3)
            task_params.setdefault("web_select_group_count_max", 6)
        task_params.setdefault("web_select_option_count_min", 3)
        task_params.setdefault("web_select_option_count_max", 3)
        task_params.setdefault(
            "web_option_group_pool",
            ("Delivery speed", "Plan type", "Seat zone", "Alert channel", "Payment method", "Notification channel"),
        )
        return _lifecycle.build_web_action_response(
            instance_seed=int(instance_seed),
            params=task_params,
            task_id=TASK_ID,
            prompt_query_key=str(selected_branch),
            public_query_id=str(selected_branch),
            query_id_probabilities=dict(branch_probabilities),
            control_family_key=str(_lifecycle.GUIDE_CODE_COUNT_CONTROL_FAMILY_BY_QUERY_ID[str(selected_branch)]),
            answer_mode="guide_code_count",
            question_format="web_action_guide_code_target_count",
        )


__all__ = [
    "DEFAULT_QUERY_ID",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
    "PagesWebActionGuideCodeTargetCountTask",
]
