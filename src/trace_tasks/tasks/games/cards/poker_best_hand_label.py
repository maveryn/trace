"""Choose the poker hand with the strongest category."""

from __future__ import annotations

from trace_tasks.tasks.games.shared.sampling import resolve_games_named_axis
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.fixed_query import DEFAULT_QUERY_ID

from ._lifecycle import (
    build_cards_rule_objective_plan,
    resolve_cards_integer_axis,
    run_cards_rule_lifecycle,
)
from .shared.sampling import SUPPORTED_POKER_WINNING_CATEGORIES, sample_poker_best_hand
from .shared.state import SCENE_ID


TASK_ID = "task_games__cards__poker_best_hand_label"
QUERY_ID = DEFAULT_QUERY_ID
PROMPT_QUERY_KEY = "poker_best_hand_label"
SUPPORTED_QUERY_IDS = (QUERY_ID,)
POKER_HAND_COUNT_SUPPORT = (4, 6)
_GEN_DEFAULTS, _RENDER_DEFAULTS_UNUSED, _PROMPT_DEFAULTS_UNUSED = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
    task_id=TASK_ID,
)


def _prepare_poker_best_objective(instance_seed, task_params, _query_id, _query_probabilities):
    """Resolve poker hand-count/category axes and bind the best-hand sampler."""

    hand_count = resolve_cards_integer_axis(
        instance_seed=int(instance_seed),
        params=task_params,
        gen_defaults=_GEN_DEFAULTS,
        support_key="poker_hand_count_support",
        explicit_key="option_count",
        fallback_support=POKER_HAND_COUNT_SUPPORT,
        namespace=f"{TASK_ID}.hand_count",
    )
    winning_category, winning_category_probs = resolve_games_named_axis(
        instance_seed=int(instance_seed),
        params=task_params,
        gen_defaults=_GEN_DEFAULTS,
        namespace="poker_winning_category",
        explicit_key="poker_winning_category",
        weights_key="poker_winning_category_weights",
        balance_flag_key="balanced_poker_winning_category_sampling",
        supported_variants=SUPPORTED_POKER_WINNING_CATEGORIES,
    )

    def construct_attempt(rng):
        return sample_poker_best_hand(
            rng,
            hand_count=int(hand_count.value),
            winning_category_key=str(winning_category),
            hand_count_support=hand_count.support,
            hand_count_probabilities=hand_count.probabilities,
            winning_category_probabilities=winning_category_probs,
        )

    return build_cards_rule_objective_plan(
        attempt_namespace="games.cards.poker_best_hand_label",
        prompt_query_key=PROMPT_QUERY_KEY,
        construct_attempt=construct_attempt,
    )


@register_task
class GamesCardsPokerBestHandLabelTask:
    """Generate the poker best-hand label task."""

    task_id = TASK_ID
    reasoning_operations = ('ranking',)
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        """Generate a poker best-hand scene by binding category and option axes locally."""

        return run_cards_rule_lifecycle(
            task_id=self.task_id,
            domain=self.domain,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=QUERY_ID,
            gen_defaults=_GEN_DEFAULTS,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            prepare_objective=_prepare_poker_best_objective,
        )


__all__ = ["GamesCardsPokerBestHandLabelTask"]
