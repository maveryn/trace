"""Choose the draw card that completes the requested poker category."""

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
from .shared.sampling import SUPPORTED_POKER_DRAW_TARGET_CATEGORIES, sample_poker_draw_card, target_candidate_index
from .shared.state import SCENE_ID


TASK_ID = "task_games__cards__poker_draw_card_label"
QUERY_ID = DEFAULT_QUERY_ID
PROMPT_QUERY_KEY = "poker_draw_card_label"
SUPPORTED_QUERY_IDS = (QUERY_ID,)
POKER_DRAW_CANDIDATE_COUNT_SUPPORT = (4, 6)
_GEN_DEFAULTS, _RENDER_DEFAULTS_UNUSED, _PROMPT_DEFAULTS_UNUSED = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
    task_id=TASK_ID,
)


def _prepare_poker_draw_objective(instance_seed, task_params, _query_id, _query_probabilities):
    """Resolve draw candidate/category axes and bind the completion-card sampler."""

    candidate_count = resolve_cards_integer_axis(
        instance_seed=int(instance_seed),
        params=task_params,
        gen_defaults=_GEN_DEFAULTS,
        support_key="poker_draw_candidate_count_support",
        explicit_key="option_count",
        fallback_support=POKER_DRAW_CANDIDATE_COUNT_SUPPORT,
        namespace=f"{TASK_ID}.candidate_count",
    )
    target_category, target_category_probs = resolve_games_named_axis(
        instance_seed=int(instance_seed),
        params=task_params,
        gen_defaults=_GEN_DEFAULTS,
        namespace="poker_draw_target_category",
        explicit_key="poker_draw_target_category",
        weights_key="poker_draw_target_category_weights",
        balance_flag_key="balanced_poker_draw_target_category_sampling",
        supported_variants=SUPPORTED_POKER_DRAW_TARGET_CATEGORIES,
    )

    def construct_attempt(rng):
        return sample_poker_draw_card(
            rng,
            candidate_count=int(candidate_count.value),
            target_category_key=str(target_category),
            target_index=target_candidate_index(rng=rng, params=task_params, candidate_count=int(candidate_count.value)),
            candidate_count_support=candidate_count.support,
            candidate_count_probabilities=candidate_count.probabilities,
            target_category_probabilities=target_category_probs,
        )

    return build_cards_rule_objective_plan(
        attempt_namespace="games.cards.poker_draw_card_label",
        prompt_query_key=PROMPT_QUERY_KEY,
        construct_attempt=construct_attempt,
        scalar_annotation=True,
    )


@register_task
class GamesCardsPokerDrawCardLabelTask:
    """Generate the poker draw-card label task."""

    task_id = TASK_ID
    reasoning_operations = ('ranking', 'matching')
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        """Generate a poker draw scene by binding candidate/category axes locally."""

        return run_cards_rule_lifecycle(
            task_id=self.task_id,
            domain=self.domain,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=QUERY_ID,
            gen_defaults=_GEN_DEFAULTS,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            prepare_objective=_prepare_poker_draw_objective,
        )


__all__ = ["GamesCardsPokerDrawCardLabelTask"]
