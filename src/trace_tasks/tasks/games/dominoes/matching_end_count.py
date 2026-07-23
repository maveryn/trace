"""Count loose dominoes matching the open chain end."""

from __future__ import annotations

from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults

from ._lifecycle import prepare_domino_recipe_count_objective, run_domino_lifecycle
from .shared.defaults import DEFAULTS, SCENE_ID
from .shared.rules import PIP_VALUES
from .shared.sampling import CountedCandidateRecipe, sample_chain_with_end


TASK_ID = "task_games__dominoes__matching_end_count"
QUERY_ID = "matching_end_count"
SUPPORTED_QUERY_IDS = (QUERY_ID,)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS_UNUSED = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
    task_id=TASK_ID,
)


def _build_matching_end_recipe(recipe_rng):
    """Choose a visible open end and predicate for playable candidates."""

    open_end_value = int(PIP_VALUES[int(recipe_rng.randrange(len(PIP_VALUES)))])
    connector_options = [value for value in PIP_VALUES if int(value) != int(open_end_value)]
    connector_value = int(connector_options[int(recipe_rng.randrange(len(connector_options)))])
    oriented_chain = sample_chain_with_end(
        recipe_rng,
        end_tile=(int(connector_value), int(open_end_value)),
        avoid_prefix_values=(int(open_end_value),),
    )
    return CountedCandidateRecipe(
        oriented_chain=oriented_chain,
        is_annotation_tile=lambda tile: int(open_end_value) in {int(tile[0]), int(tile[1])},
        reference_role="reference_end",
        highlight_open_end=True,
        open_end_value=int(open_end_value),
        reference_sum=int(oriented_chain[-1][0] + oriented_chain[-1][1]),
    )


def _prepare_matching_end_objective(
    instance_seed,
    task_params,
    _query_id,
    _query_probabilities,
    axes,
):
    """Resolve answer/candidate axes and bind open-end matching semantics."""

    return prepare_domino_recipe_count_objective(
        instance_seed=int(instance_seed),
        task_params=task_params,
        gen_defaults=_GEN_DEFAULTS,
        axes=axes,
        prompt_query_key=QUERY_ID,
        attempt_namespace="games.dominoes.matching_end",
        target_support_key="matching_end_target_answer_support",
        target_fallback_support=DEFAULTS.matching_end_target_answer_support,
        target_namespace="matching_end.target_answer",
        minimum_candidate_count=lambda target: max(7, int(target)),
        candidate_namespace="matching_end",
        build_recipe=_build_matching_end_recipe,
        recipe_attempts=256,
        example_answer=2,
    )


@register_task
class GamesDominoesMatchingEndCountTask:
    """Count loose dominoes that can connect to the marked reference end."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'matching')
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed, *, params, max_attempts):
        return run_domino_lifecycle(
            task_id=TASK_ID,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            gen_defaults=_GEN_DEFAULTS,
            render_defaults=_RENDER_DEFAULTS,
            prepare_objective=_prepare_matching_end_objective,
        )


__all__ = ["GamesDominoesMatchingEndCountTask"]
