"""Count ranks that appear exactly three times."""

from __future__ import annotations

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.fixed_query import DEFAULT_QUERY_ID

from ._lifecycle import (
    build_cards_hand_count_objective_plan,
    resolve_cards_boolean_flag,
    resolve_cards_hand_count_axes,
    run_cards_hand_count_lifecycle,
)
from .shared.annotations import keyed_card_bbox_set_map
from .shared.sampling import sample_exact_triple_count_hand
from .shared.state import SCENE_ID


TASK_ID = "task_games__cards__exact_triple_count"
QUERY_ID = DEFAULT_QUERY_ID
PROMPT_QUERY_KEY = "exact_triple_count"
SUPPORTED_QUERY_IDS = (QUERY_ID,)
TARGET_ANSWER_SUPPORT = (0, 1, 2, 3, 4)
CARD_COUNT_SUPPORT = (12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22)
_GEN_DEFAULTS, _RENDER_DEFAULTS_UNUSED, _PROMPT_DEFAULTS_UNUSED = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
    task_id=TASK_ID,
)


def _exact_triple_annotation(rendered_context, sampled_hand):
    """Bind each exact-triple rank key to the three visible card boxes."""

    annotation_bbox_map = keyed_card_bbox_set_map(
        rendered_context.rendered_scene.render_map,
        sampled_hand.keyed_annotation_card_ids,
    )
    return TypedValue(type="bbox_set_map", value=dict(annotation_bbox_map)), {
        "bbox_set_map": dict(annotation_bbox_map),
        "pixel_bbox_set_map": dict(annotation_bbox_map),
    }


def _prepare_exact_triple_objective(instance_seed, task_params, _query_id, _query_probabilities):
    """Resolve exact-triple target/count axes and bind the rank-group sampler."""

    axes = resolve_cards_hand_count_axes(
        instance_seed=int(instance_seed),
        params=task_params,
        gen_defaults=_GEN_DEFAULTS,
        hand_kind="exact_triple",
        target_support_key="exact_triple_count_support",
        target_fallback_support=TARGET_ANSWER_SUPPORT,
        card_count_support_key="exact_triple_count_card_count_support",
        card_count_fallback_support=CARD_COUNT_SUPPORT,
        target_namespace=f"{TASK_ID}.target_answer",
        card_count_namespace=f"{TASK_ID}.card_count",
    )
    order_by_rank = resolve_cards_boolean_flag(
        params=task_params,
        gen_defaults=_GEN_DEFAULTS,
        key="exact_triple_count_order_by_rank",
        fallback=True,
    )

    def construct_attempt(rng):
        return sample_exact_triple_count_hand(
            rng,
            card_count=int(axes.card_count.value),
            target_answer=int(axes.target.value),
            order_by_rank=bool(order_by_rank),
        )

    return build_cards_hand_count_objective_plan(
        attempt_namespace="games.cards.exact_triple_count",
        prompt_query_key=PROMPT_QUERY_KEY,
        axes=axes,
        card_ordering="rank_grouped" if order_by_rank else "sampled",
        construct_attempt=construct_attempt,
        build_annotation=_exact_triple_annotation,
        center_label_key="exact_triple_count_center_label_mode",
        params=task_params,
        gen_defaults=_GEN_DEFAULTS,
    )


@register_task
class GamesCardsExactTripleCountTask:
    """Count ranks that appear exactly three times."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting')
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        """Generate an exact-triple count scene by binding rank-group witnesses locally."""

        return run_cards_hand_count_lifecycle(
            task_id=self.task_id,
            domain=self.domain,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=QUERY_ID,
            gen_defaults=_GEN_DEFAULTS,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            prepare_objective=_prepare_exact_triple_objective,
        )


__all__ = ["GamesCardsExactTripleCountTask"]
