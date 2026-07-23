"""Count visible cube stacks satisfying a height condition."""

from __future__ import annotations

from trace_tasks.tasks.registry import register_task

from ._lifecycle import MinecraftObjectivePlan, minecraft_integer_attempt, run_minecraft_registered_task
from .shared.defaults import HEIGHT_CONDITION_AT_LEAST, HEIGHT_CONDITION_EXACT
from .shared.sampling import resolve_height_filter_axes, sample_height_filter_scene


TASK_ID = "task_games__minecraft__stack_height_condition_count"
EXACT_HEIGHT_QUERY_ID = "exact_height_count"
AT_LEAST_HEIGHT_QUERY_ID = "at_least_height_count"
SUPPORTED_QUERY_IDS = (EXACT_HEIGHT_QUERY_ID, AT_LEAST_HEIGHT_QUERY_ID)
HEIGHT_BRANCHES = {
    EXACT_HEIGHT_QUERY_ID: HEIGHT_CONDITION_EXACT,
    AT_LEAST_HEIGHT_QUERY_ID: HEIGHT_CONDITION_AT_LEAST,
}


def _prepare_height_filter_objective(
    instance_seed,
    task_params,
    selected_branch,
    _branch_probabilities,
    gen_defaults,
) -> MinecraftObjectivePlan:
    """Resolve the selected height predicate and bind exact-count construction."""

    height_condition = HEIGHT_BRANCHES[str(selected_branch)]
    axes = resolve_height_filter_axes(
        int(instance_seed),
        gen_defaults=gen_defaults,
        namespace=f"games.minecraft.height_filter.{height_condition}",
        params=task_params,
        height_condition=str(height_condition),
    )

    def construct_attempt(rng, resolved_axes):
        sample = sample_height_filter_scene(
            rng=rng,
            axes=resolved_axes,
            gen_defaults=gen_defaults,
            params=task_params,
            height_condition=str(height_condition),
        )
        return minecraft_integer_attempt(
            sample=sample,
            prompt_key=str(selected_branch),
            object_description_key="object_description_block_world",
            answer_hint_key=f"answer_hint_{str(selected_branch)}",
            annotation_hint_key=f"annotation_hint_{str(selected_branch)}",
            example_annotation=[[265, 218, 315, 274], [398, 174, 448, 230], [517, 250, 567, 306]],
            example_answer=3,
            target_stack_height=int(sample.target_stack_height),
            extra_query_params={
                "prompt_query_key": str(selected_branch),
                "target_stack_height": int(sample.target_stack_height),
                "target_stack_height_probabilities": dict(resolved_axes.target_stack_height_probabilities),
                "stack_height_condition": str(sample.stack_height_condition),
            },
        )

    return MinecraftObjectivePlan(
        axes=axes,
        attempt_namespace=f"games.minecraft.height_filter.{height_condition}",
        construct_attempt=construct_attempt,
    )


@register_task
class GamesMinecraftStackHeightConditionCountTask:
    """Count visible cube stacks satisfying a sampled height condition."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'comparison')
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS
    _default_branch = EXACT_HEIGHT_QUERY_ID
    _namespace = "games.minecraft.height_filter"
    _prepare_objective = staticmethod(_prepare_height_filter_objective)

    def generate(
        self,
        instance_seed,
        *,
        params=None,
        max_attempts=100,
    ):
        """Generate a stack height condition count task instance."""

        return run_minecraft_registered_task(
            self,
            int(instance_seed),
            params=params or {},
            max_attempts=int(max_attempts),
        )


__all__ = [
    "AT_LEAST_HEIGHT_QUERY_ID",
    "EXACT_HEIGHT_QUERY_ID",
    "GamesMinecraftStackHeightConditionCountTask",
]
