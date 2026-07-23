"""Count darts inside or outside the simplified dartboard bullseye."""

from __future__ import annotations

from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults

from ._lifecycle import prepare_darts_exact_count_objective, run_darts_lifecycle
from .shared.defaults import SCENE_ID
from .shared.prompts import darts_integer_json_examples, darts_output_slots
from .shared.rules import BULLSEYE_SLOT, SECTOR_SLOTS


TASK_ID = "task_games__darts__bullseye_membership_count"
SUPPORTED_QUERY_IDS = ("inside_bullseye_count", "outside_bullseye_count")
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS_UNUSED = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
    task_id=TASK_ID,
)


def _prepare_bullseye_membership_objective(
    instance_seed,
    task_params,
    query_id,
    _query_probabilities,
    render_params,
):
    """Bind inside/outside bullseye membership to an exact-count objective."""

    selected_query = str(query_id)
    inside_query = selected_query == "inside_bullseye_count"
    qualifying_slots = (BULLSEYE_SLOT,) if inside_query else tuple(SECTOR_SLOTS)
    nonqualifying_slots = tuple(SECTOR_SLOTS) if inside_query else (BULLSEYE_SLOT,)
    json_example, json_example_answer_only = darts_integer_json_examples()
    prompt_dynamic_slots = darts_output_slots(
        prompt_query_key=selected_query,
        json_example=json_example,
        json_example_answer_only=json_example_answer_only,
    )
    return prepare_darts_exact_count_objective(
        task_params=task_params,
        gen_defaults=_GEN_DEFAULTS,
        render_params=render_params,
        instance_seed=int(instance_seed),
        target_namespace=selected_query,
        attempt_namespace=f"games.darts.{selected_query}",
        prompt_query_key=selected_query,
        prompt_dynamic_slots=prompt_dynamic_slots,
        qualifying_slots=tuple(qualifying_slots),
        nonqualifying_slots=tuple(nonqualifying_slots),
        extra_query_params={"bullseye_membership": "inside" if inside_query else "outside"},
        extra_execution_params={"bullseye_membership": "inside" if inside_query else "outside"},
    )


@register_task
class GamesDartsBullseyeMembershipCountTask:
    """Count darts inside or outside the single bullseye."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'spatial_relations')
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
            prepare_objective=_prepare_bullseye_membership_objective,
        )


__all__ = ["GamesDartsBullseyeMembershipCountTask"]
