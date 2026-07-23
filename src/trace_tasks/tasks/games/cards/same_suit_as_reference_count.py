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
from .shared.sampling import sample_same_suit_hand
from .shared.state import SCENE_ID


TASK_ID = "task_games__cards__same_suit_as_reference_count"
PROMPT_QUERY_KEY = "same_suit_as_reference_count"
SUPPORTED_QUERY_IDS = (DEFAULT_QUERY_ID,)
TARGET_ANSWER_SUPPORT = (0, 1, 2, 3, 4, 5)
CARD_COUNT_SUPPORT = (16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26)
_GEN_DEFAULTS, _RENDER_DEFAULTS_UNUSED, _PROMPT_DEFAULTS_UNUSED = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
    task_id=TASK_ID,
)


def _prepare_same_suit_objective(instance_seed, task_params, _query_id, _query_probabilities):
    axes = resolve_cards_hand_count_axes(
        instance_seed=int(instance_seed),
        params=task_params,
        gen_defaults=_GEN_DEFAULTS,
        hand_kind=PROMPT_QUERY_KEY,
        target_support_key="same_suit_target_answer_support",
        target_fallback_support=TARGET_ANSWER_SUPPORT,
        card_count_support_key="same_suit_as_reference_count_card_count_support",
        card_count_fallback_support=CARD_COUNT_SUPPORT,
        target_namespace=f"{TASK_ID}.target_answer",
        card_count_namespace=f"{TASK_ID}.card_count",
    )
    order_by_suit = resolve_cards_boolean_flag(
        params=task_params,
        gen_defaults=_GEN_DEFAULTS,
        key="same_suit_order_by_suit",
        fallback=False,
    )

    def construct_attempt(rng):
        return sample_same_suit_hand(
            rng,
            card_count=int(axes.card_count.value),
            target_answer=int(axes.target.value),
            order_by_suit=order_by_suit,
            reference_anchor_index=0,
        )

    return build_cards_hand_count_objective_plan(
        attempt_namespace="games.cards.same_suit_as_reference_count",
        prompt_query_key=PROMPT_QUERY_KEY,
        axes=axes,
        card_ordering="suit_grouped" if order_by_suit else "sampled",
        construct_attempt=construct_attempt,
        build_annotation=bbox_set_for_cards,
        center_label_key="same_suit_center_label_mode",
        params=task_params,
        gen_defaults=_GEN_DEFAULTS,
    )


@register_task
class GamesCardsSameSuitAsReferenceCountTask:
    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting')
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
            prepare_objective=_prepare_same_suit_objective,
        )
