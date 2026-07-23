"""Choose which candidate card would win the current trick."""

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
from .shared.sampling import SUPPORTED_TRICK_PLAY_TRUMP_MODES, sample_trick_winning_play, target_candidate_index
from .shared.state import SCENE_ID


TASK_ID = "task_games__cards__trick_winning_play_label"
QUERY_ID = DEFAULT_QUERY_ID
PROMPT_QUERY_KEY = "trick_winning_play_label"
SUPPORTED_QUERY_IDS = (QUERY_ID,)
TRICK_PLAY_CANDIDATE_COUNT_SUPPORT = (4, 6)
TRICK_PLAY_PLAYED_COUNT_SUPPORT = (3, 4)
_GEN_DEFAULTS, _RENDER_DEFAULTS_UNUSED, _PROMPT_DEFAULTS_UNUSED = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
    task_id=TASK_ID,
)


def _prepare_trick_play_objective(instance_seed, task_params, _query_id, _query_probabilities):
    """Resolve trick-play axes and bind the winning candidate-card sampler."""

    candidate_count = resolve_cards_integer_axis(
        instance_seed=int(instance_seed),
        params=task_params,
        gen_defaults=_GEN_DEFAULTS,
        support_key="trick_play_candidate_count_support",
        explicit_key="option_count",
        fallback_support=TRICK_PLAY_CANDIDATE_COUNT_SUPPORT,
        namespace=f"{TASK_ID}.candidate_count",
    )
    played_count = resolve_cards_integer_axis(
        instance_seed=int(instance_seed),
        params=task_params,
        gen_defaults=_GEN_DEFAULTS,
        support_key="trick_play_played_count_support",
        explicit_key="played_count",
        fallback_support=TRICK_PLAY_PLAYED_COUNT_SUPPORT,
        namespace=f"{TASK_ID}.played_count",
        balanced_flag_key="balanced_trick_played_count_sampling",
    )
    trump_mode, trump_mode_probs = resolve_games_named_axis(
        instance_seed=int(instance_seed),
        params=task_params,
        gen_defaults=_GEN_DEFAULTS,
        namespace="trick_play_trump_mode",
        explicit_key="trick_play_trump_mode",
        weights_key="trick_play_trump_mode_weights",
        balance_flag_key="balanced_trick_play_trump_mode_sampling",
        supported_variants=SUPPORTED_TRICK_PLAY_TRUMP_MODES,
    )

    def construct_attempt(rng):
        return sample_trick_winning_play(
            rng,
            candidate_count=int(candidate_count.value),
            played_count=int(played_count.value),
            trump_mode=str(trump_mode),
            target_index=target_candidate_index(rng=rng, params=task_params, candidate_count=int(candidate_count.value)),
            candidate_count_support=candidate_count.support,
            candidate_count_probabilities=candidate_count.probabilities,
            played_count_support=played_count.support,
            played_count_probabilities=played_count.probabilities,
            trump_mode_probabilities=trump_mode_probs,
        )

    return build_cards_rule_objective_plan(
        attempt_namespace="games.cards.trick_winning_play_label",
        prompt_query_key=PROMPT_QUERY_KEY,
        construct_attempt=construct_attempt,
        scalar_annotation=True,
    )


@register_task
class GamesCardsTrickWinningPlayLabelTask:
    """Generate the trick-winning play label task."""

    task_id = TASK_ID
    reasoning_operations = ('state_update',)
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        """Generate a trick-play scene by binding play axes locally."""

        return run_cards_rule_lifecycle(
            task_id=self.task_id,
            domain=self.domain,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=QUERY_ID,
            gen_defaults=_GEN_DEFAULTS,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            prepare_objective=_prepare_trick_play_objective,
        )


__all__ = ["GamesCardsTrickWinningPlayLabelTask"]
