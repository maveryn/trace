from __future__ import annotations

from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.fixed_query import DEFAULT_QUERY_ID

from ._lifecycle import (
    bbox_set_for_cards,
    build_cards_hand_count_objective_plan,
    resolve_cards_boolean_flag,
    resolve_cards_hand_count_axes,
    run_cards_hand_count_lifecycle,
)
from .shared.sampling import sample_higher_rank_hand
from .shared.state import SCENE_ID


TASK_ID = "task_games__cards__higher_than_reference_count"
PROMPT_QUERY_KEY = "higher_than_reference_count"
SUPPORTED_QUERY_IDS = (DEFAULT_QUERY_ID,)
TARGET_ANSWER_SUPPORT = (0, 1, 2, 3, 4, 5)
CARD_COUNT_SUPPORT = (10, 11, 12, 13, 14, 15)
_GEN_DEFAULTS, _RENDER_DEFAULTS_UNUSED, _PROMPT_DEFAULTS_UNUSED = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
    task_id=TASK_ID,
)


def _prepare_higher_rank_objective(instance_seed, task_params, _query_id, _query_probabilities):
    axes = resolve_cards_hand_count_axes(
        instance_seed=int(instance_seed),
        params=task_params,
        gen_defaults=_GEN_DEFAULTS,
        hand_kind=PROMPT_QUERY_KEY,
        target_support_key="higher_rank_target_answer_support",
        target_fallback_support=TARGET_ANSWER_SUPPORT,
        card_count_support_key="higher_than_reference_count_card_count_support",
        card_count_fallback_support=CARD_COUNT_SUPPORT,
        target_namespace=f"{TASK_ID}.target_answer",
        card_count_namespace=f"{TASK_ID}.card_count",
    )
    order_by_rank = resolve_cards_boolean_flag(
        params=task_params,
        gen_defaults=_GEN_DEFAULTS,
        key="higher_rank_order_by_rank",
        fallback=False,
    )

    def construct_attempt(rng):
        return sample_higher_rank_hand(
            rng,
            card_count=int(axes.card_count.value),
            target_answer=int(axes.target.value),
            order_by_rank=order_by_rank,
            reference_anchor_index=0,
        )

    return build_cards_hand_count_objective_plan(
        attempt_namespace="games.cards.higher_than_reference_count",
        prompt_query_key=PROMPT_QUERY_KEY,
        axes=axes,
        card_ordering="rank_grouped" if order_by_rank else "sampled",
        construct_attempt=construct_attempt,
        build_annotation=bbox_set_for_cards,
        center_label_key="higher_rank_center_label_mode",
        params=task_params,
        gen_defaults=_GEN_DEFAULTS,
    )


@register_task
class GamesCardsHigherThanReferenceCountTask:
    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'comparison', 'ranking')
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        return run_cards_hand_count_lifecycle(
            task_id=self.task_id,
            domain=self.domain,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=DEFAULT_QUERY_ID,
            gen_defaults=_GEN_DEFAULTS,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            prepare_objective=_prepare_higher_rank_objective,
        )
