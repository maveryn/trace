"""Read the score of the only visible dart on a simplified dartboard."""

from __future__ import annotations

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults

from ._lifecycle import DartsObjectivePlan, dart_point_attempt, run_darts_lifecycle
from .shared.defaults import SCENE_ID
from .shared.prompts import darts_output_slots, darts_single_point_json_examples
from .shared.sampling import (
    resolve_darts_score_axis,
    sample_darts_for_score_value,
)


TASK_ID = "task_games__darts__dart_score_value"
QUERY_ID = "single"
PROMPT_QUERY_KEY = "dart_score_value"
SUPPORTED_QUERY_IDS = (QUERY_ID,)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS_UNUSED = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
    task_id=TASK_ID,
)


def _prepare_score_value_objective(
    instance_seed,
    task_params,
    _query_id,
    _query_probabilities,
    render_params,
):
    """Resolve the visible dart score and bind the score-value objective."""

    score_axis = resolve_darts_score_axis(
        instance_seed=int(instance_seed),
        params=task_params,
        gen_defaults=_GEN_DEFAULTS,
    )
    json_example, json_example_answer_only = darts_single_point_json_examples()
    prompt_dynamic_slots = darts_output_slots(
        prompt_query_key=PROMPT_QUERY_KEY,
        json_example=json_example,
        json_example_answer_only=json_example_answer_only,
    )
    query_params = {
        "target_score": int(score_axis.value),
        "score_value_support": [int(value) for value in score_axis.support],
        "score_value_probabilities": dict(score_axis.probabilities),
    }

    def construct_attempt(rng, _axes):
        sample = sample_darts_for_score_value(
            rng,
            target_score=int(score_axis.value),
            render_params=render_params,
        )
        return dart_point_attempt(
            sample=sample,
            answer_gt=TypedValue(type="integer", value=int(score_axis.value)),
            execution_extra={"target_score": int(score_axis.value)},
        )

    return DartsObjectivePlan(
        attempt_namespace="games.darts.dart_score_value",
        prompt_query_key=PROMPT_QUERY_KEY,
        query_params=query_params,
        prompt_dynamic_slots=prompt_dynamic_slots,
        construct_attempt=construct_attempt,
    )


@register_task
class GamesDartsDartScoreValueTask:
    """Read the score of the only visible dart."""

    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed, *, params, max_attempts):
        return run_darts_lifecycle(
            task_id=TASK_ID,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            gen_defaults=_GEN_DEFAULTS,
            render_defaults=_RENDER_DEFAULTS,
            prepare_objective=_prepare_score_value_objective,
        )


__all__ = ["GamesDartsDartScoreValueTask"]
