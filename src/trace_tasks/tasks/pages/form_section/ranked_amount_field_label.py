"""Form-section task for selecting a ranked currency field label."""

from __future__ import annotations

from trace_tasks.tasks.registry import register_task

from ._lifecycle import run_form_section_ranked_label_public_entry
from .shared.sampling import RankPlan, SCENE_VARIANTS


TASK_ID = "task_pages__form_section__ranked_amount_field_label"
SECOND_HIGHEST_QUERY_ID = "second_highest_amount_field_label"
SECOND_LOWEST_QUERY_ID = "second_lowest_amount_field_label"
SUPPORTED_QUERY_IDS = (
    SECOND_HIGHEST_QUERY_ID,
    SECOND_LOWEST_QUERY_ID,
)
PROMPT_QUERY_KEY = "ranked_amount_field_label"
QUESTION_FORMAT = "form_section_ranked_amount_field_label"
REASONING_LOAD_BASE = 0.34


def _build_rank_plans() -> dict[str, RankPlan]:
    """Bind query branches to section-local amount ranks."""

    return {
        SECOND_HIGHEST_QUERY_ID: RankPlan(
            operation_name="ranked_amount_field_label",
            rank_from="highest",
            rank_position=2,
            rank_phrase="second-highest",
        ),
        SECOND_LOWEST_QUERY_ID: RankPlan(
            operation_name="ranked_amount_field_label",
            rank_from="lowest",
            rank_position=2,
            rank_phrase="second-lowest",
        ),
    }


@register_task
class PagesFormSectionRankedAmountFieldLabelTask:
    """Select the field label for a ranked amount inside one named section."""

    task_id = TASK_ID
    reasoning_operations = ('ranking',)
    domain = "pages"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed, *, params, max_attempts):
        return run_form_section_ranked_label_public_entry(
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            public_task=TASK_ID,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            rank_plan=_build_rank_plans(),
            prompt_query_key=PROMPT_QUERY_KEY,
            question_format=QUESTION_FORMAT,
            reasoning_load_base=REASONING_LOAD_BASE,
        )


__all__ = [
    "PROMPT_QUERY_KEY",
    "QUESTION_FORMAT",
    "REASONING_LOAD_BASE",
    "SCENE_VARIANTS",
    "SECOND_HIGHEST_QUERY_ID",
    "SECOND_LOWEST_QUERY_ID",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
    "PagesFormSectionRankedAmountFieldLabelTask",
]
