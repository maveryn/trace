"""Choose the winning player in a visible trick-taking round."""

from __future__ import annotations

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.fixed_query import DEFAULT_QUERY_ID
from trace_tasks.tasks.shared.support_sampling import resolve_integer_choice, resolve_integer_support

from ._lifecycle import build_cards_rule_objective_plan, run_cards_rule_lifecycle
from .shared.sampling import sample_trick_taking_winner
from .shared.state import SCENE_ID


TASK_ID = "task_games__cards__trick_taking_winner_label"
QUERY_ID = DEFAULT_QUERY_ID
PROMPT_QUERY_KEY = "trick_taking_winner_label"
SUPPORTED_QUERY_IDS = (QUERY_ID,)
TRICK_PLAYER_COUNT_SUPPORT = (4, 6)
_GEN_DEFAULTS, _RENDER_DEFAULTS_UNUSED, _PROMPT_DEFAULTS_UNUSED = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
    task_id=TASK_ID,
)


def _target_winner_index(instance_seed, task_params, player_support):
    """Resolve a target winner index that fits at least one supported player count."""

    raw_target_index = task_params.get("target_winner_index")
    raw_player_count = task_params.get("option_count")
    if raw_player_count is not None:
        target_upper = int(raw_player_count)
        if target_upper not in {int(value) for value in player_support}:
            raise ValueError(f"unsupported option_count: {target_upper}")
    else:
        target_upper = max(int(value) for value in player_support)
    if raw_target_index is not None:
        target_winner_index = int(raw_target_index)
        if target_winner_index < 0 or target_winner_index >= target_upper:
            raise ValueError(f"unsupported target_winner_index: {target_winner_index}")
        return target_winner_index
    if task_params.get("_sample_cursor") is not None:
        return abs(int(task_params["_sample_cursor"])) % target_upper
    return int(spawn_rng(int(instance_seed), f"{TASK_ID}.target_winner_index").randrange(target_upper))


def _prepare_trick_winner_objective(instance_seed, task_params, _query_id, _query_probabilities):
    """Resolve player count and winner-index axes before sampling one trick."""

    player_support = resolve_integer_support(
        task_params,
        gen_defaults=_GEN_DEFAULTS,
        key="trick_player_count_support",
        fallback=TRICK_PLAYER_COUNT_SUPPORT,
    )
    target_index = _target_winner_index(int(instance_seed), task_params, tuple(int(value) for value in player_support))
    feasible_support = tuple(int(value) for value in player_support if int(value) > int(target_index))
    if not feasible_support:
        raise ValueError("no feasible trick player count can contain target winner")
    player_count, player_count_probs = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params={**dict(task_params), "option_count_support": [int(value) for value in feasible_support]},
        gen_defaults=_GEN_DEFAULTS,
        support_key="option_count_support",
        explicit_key="option_count",
        fallback_support=feasible_support,
        namespace=f"{TASK_ID}.player_count",
        balanced_flag_key="balanced_option_count_sampling",
        namespace_support_permutation=True,
    )

    def construct_attempt(rng):
        return sample_trick_taking_winner(
            rng,
            player_count=int(player_count),
            target_winner_index=int(target_index),
            player_count_support=feasible_support,
            player_count_probabilities=player_count_probs,
        )

    return build_cards_rule_objective_plan(
        attempt_namespace="games.cards.trick_taking_winner_label",
        prompt_query_key=PROMPT_QUERY_KEY,
        construct_attempt=construct_attempt,
        scalar_annotation=True,
    )


@register_task
class GamesCardsTrickTakingWinnerLabelTask:
    """Generate the trick-taking winner label task."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'comparison', 'ranking')
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        """Generate a trick winner scene by binding winner index and player count locally."""

        return run_cards_rule_lifecycle(
            task_id=self.task_id,
            domain=self.domain,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=QUERY_ID,
            gen_defaults=_GEN_DEFAULTS,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            prepare_objective=_prepare_trick_winner_objective,
        )


__all__ = ["GamesCardsTrickTakingWinnerLabelTask"]
