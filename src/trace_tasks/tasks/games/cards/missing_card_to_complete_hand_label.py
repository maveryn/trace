"""Choose the candidate card that completes one requested hand pattern."""

from __future__ import annotations

from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults

from ._lifecycle import (
    build_cards_rule_objective_plan,
    resolve_cards_integer_axis,
    run_cards_rule_lifecycle,
)
from .shared.sampling import (
    sample_missing_card_to_complete_hand,
    target_candidate_index,
)
from .shared.state import SCENE_ID


TASK_ID = "task_games__cards__missing_card_to_complete_hand_label"
MISSING_CARD_COMPLETION_KINDS = (
    "missing_flush_card_label",
    "missing_straight_card_label",
    "missing_full_house_card_label",
    "missing_three_of_kind_card_label",
)
_INTERNAL_PATTERN_BY_QUERY_ID = {
    "missing_flush_card_label": "missing_flush",
    "missing_straight_card_label": "missing_straight",
    "missing_full_house_card_label": "missing_full_house",
    "missing_three_of_kind_card_label": "missing_three_kind",
}
SUPPORTED_QUERY_IDS = MISSING_CARD_COMPLETION_KINDS
MISSING_CARD_CANDIDATE_COUNT_SUPPORT = (4, 6)
_GEN_DEFAULTS, _RENDER_DEFAULTS_UNUSED, _PROMPT_DEFAULTS_UNUSED = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
    task_id=TASK_ID,
)


def _prepare_missing_card_objective(instance_seed, task_params, query_id, query_probabilities):
    """Resolve missing-card candidate count and bind the selected pattern sampler."""

    candidate_count = resolve_cards_integer_axis(
        instance_seed=int(instance_seed),
        params=task_params,
        gen_defaults=_GEN_DEFAULTS,
        support_key="missing_card_candidate_count_support",
        explicit_key="option_count",
        fallback_support=MISSING_CARD_CANDIDATE_COUNT_SUPPORT,
        namespace=f"{TASK_ID}.candidate_count",
    )

    def construct_attempt(rng):
        return sample_missing_card_to_complete_hand(
            rng,
            candidate_count=int(candidate_count.value),
            pattern_kind=str(_INTERNAL_PATTERN_BY_QUERY_ID[str(query_id)]),
            target_index=target_candidate_index(
                rng=rng,
                params=task_params,
                candidate_count=int(candidate_count.value),
                cycle_divisor=len(SUPPORTED_QUERY_IDS),
            ),
            candidate_count_support=candidate_count.support,
            candidate_count_probabilities=candidate_count.probabilities,
            pattern_kind_probabilities=query_probabilities,
        )

    return build_cards_rule_objective_plan(
        attempt_namespace="games.cards.missing_card_to_complete_hand_label",
        prompt_query_key=str(query_id),
        construct_attempt=construct_attempt,
        scalar_annotation=True,
    )


@register_task
class GamesCardsMissingCardToCompleteHandLabelTask:
    """Choose the candidate card that completes a requested card pattern."""

    task_id = TASK_ID
    reasoning_operations = ('matching',)
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        """Generate a missing-card scene by binding the selected pattern locally."""

        return run_cards_rule_lifecycle(
            task_id=self.task_id,
            domain=self.domain,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=SUPPORTED_QUERY_IDS[0],
            gen_defaults=_GEN_DEFAULTS,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            prepare_objective=_prepare_missing_card_objective,
        )


__all__ = ["GamesCardsMissingCardToCompleteHandLabelTask"]
