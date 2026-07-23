"""Choose the blackjack hand with the highest value not exceeding 21."""

from __future__ import annotations

from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.fixed_query import DEFAULT_QUERY_ID

from ._lifecycle import (
    build_cards_rule_objective_plan,
    resolve_cards_integer_axis,
    run_cards_rule_lifecycle,
)
from .shared.sampling import sample_blackjack_best_hand
from .shared.state import SCENE_ID


TASK_ID = "task_games__cards__blackjack_best_hand_label"
QUERY_ID = DEFAULT_QUERY_ID
PROMPT_QUERY_KEY = "blackjack_best_hand_label"
SUPPORTED_QUERY_IDS = (QUERY_ID,)
BLACKJACK_HAND_COUNT_SUPPORT = (4, 6)
BLACKJACK_CARDS_PER_HAND_SUPPORT = (3, 4)
_GEN_DEFAULTS, _RENDER_DEFAULTS_UNUSED, _PROMPT_DEFAULTS_UNUSED = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
    task_id=TASK_ID,
)


def _prepare_blackjack_objective(instance_seed, task_params, _query_id, _query_probabilities):
    """Resolve blackjack option/card axes and bind the unique-best-hand sampler."""

    hand_count = resolve_cards_integer_axis(
        instance_seed=int(instance_seed),
        params=task_params,
        gen_defaults=_GEN_DEFAULTS,
        support_key="blackjack_hand_count_support",
        explicit_key="option_count",
        fallback_support=BLACKJACK_HAND_COUNT_SUPPORT,
        namespace=f"{TASK_ID}.hand_count",
    )
    cards_per_hand = resolve_cards_integer_axis(
        instance_seed=int(instance_seed),
        params=task_params,
        gen_defaults=_GEN_DEFAULTS,
        support_key="blackjack_cards_per_hand_support",
        explicit_key="cards_per_hand",
        fallback_support=BLACKJACK_CARDS_PER_HAND_SUPPORT,
        namespace=f"{TASK_ID}.cards_per_hand",
        balanced_flag_key="balanced_cards_per_hand_sampling",
    )

    def construct_attempt(rng):
        return sample_blackjack_best_hand(
            rng,
            hand_count=int(hand_count.value),
            cards_per_hand=int(cards_per_hand.value),
            hand_count_support=hand_count.support,
            hand_count_probabilities=hand_count.probabilities,
            cards_per_hand_support=cards_per_hand.support,
            cards_per_hand_probabilities=cards_per_hand.probabilities,
        )

    return build_cards_rule_objective_plan(
        attempt_namespace="games.cards.blackjack_best_hand_label",
        prompt_query_key=PROMPT_QUERY_KEY,
        construct_attempt=construct_attempt,
    )


@register_task
class GamesCardsBlackjackBestHandLabelTask:
    """Generate the blackjack best-hand label task."""

    task_id = TASK_ID
    reasoning_operations = ('ranking', 'formula_evaluation')
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        """Generate a blackjack scene by binding option/card axes locally."""

        return run_cards_rule_lifecycle(
            task_id=self.task_id,
            domain=self.domain,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=QUERY_ID,
            gen_defaults=_GEN_DEFAULTS,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            prepare_objective=_prepare_blackjack_objective,
        )


__all__ = ["GamesCardsBlackjackBestHandLabelTask"]
