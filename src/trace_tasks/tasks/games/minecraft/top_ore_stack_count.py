"""Count visible stacks whose top cube is a requested ore type."""

from __future__ import annotations

from trace_tasks.tasks.registry import register_task

from ._lifecycle import MinecraftObjectivePlan, minecraft_integer_attempt, run_minecraft_registered_task
from .shared.defaults import DEFAULT_BRANCH_ID
from .shared.sampling import resolve_top_resource_axes, sample_top_resource_scene


TASK_ID = "task_games__minecraft__top_ore_stack_count"
QUERY_ID = DEFAULT_BRANCH_ID
PROMPT_QUERY_KEY = "top_ore_stack_count"
SUPPORTED_QUERY_IDS = (QUERY_ID,)


def _prepare_top_resource_objective(
    instance_seed,
    task_params,
    _selected_branch,
    _branch_probabilities,
    gen_defaults,
) -> MinecraftObjectivePlan:
    """Resolve top-resource axes and bind exact-count sample construction."""

    axes = resolve_top_resource_axes(
        int(instance_seed),
        gen_defaults=gen_defaults,
        namespace="games.minecraft.top_resource",
        params=task_params,
    )

    def construct_attempt(rng, resolved_axes):
        sample = sample_top_resource_scene(
            rng=rng,
            axes=resolved_axes,
            gen_defaults=gen_defaults,
            params=task_params,
        )
        return minecraft_integer_attempt(
            sample=sample,
            prompt_key=PROMPT_QUERY_KEY,
            object_description_key="object_description_block_world",
            answer_hint_key=f"answer_hint_{PROMPT_QUERY_KEY}",
            annotation_hint_key=f"annotation_hint_{PROMPT_QUERY_KEY}",
            example_annotation=[
                [314, 278, 364, 334],
                [372, 249, 422, 305],
                [430, 220, 480, 276],
                [459, 191, 509, 247],
                [343, 162, 393, 218],
            ],
            example_answer=5,
            counted_resource_kind=str(sample.counted_resource_kind),
            extra_query_params={
                "prompt_query_key": PROMPT_QUERY_KEY,
                "target_resource_kind": str(sample.target_resource_kind),
                "counted_resource_kind": str(sample.counted_resource_kind),
            },
        )

    return MinecraftObjectivePlan(
        axes=axes,
        attempt_namespace="games.minecraft.top_resource",
        construct_attempt=construct_attempt,
    )


@register_task
class GamesMinecraftTopOreStackCountTask:
    """Count visible stacks whose top cube is a requested Minecraft-like ore type."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting')
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS
    _default_branch = QUERY_ID
    _namespace = "games.minecraft.top_resource"
    _prepare_objective = staticmethod(_prepare_top_resource_objective)

    def generate(
        self,
        instance_seed,
        *,
        params=None,
        max_attempts=100,
    ):
        """Generate a top-resource stack count task instance."""

        return run_minecraft_registered_task(
            self,
            int(instance_seed),
            params=params or {},
            max_attempts=int(max_attempts),
        )


__all__ = ["GamesMinecraftTopOreStackCountTask"]
