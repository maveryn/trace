"""Count raised blocks along one visible Minecraft-like track."""

from __future__ import annotations

from trace_tasks.tasks.registry import register_task

from ._lifecycle import MinecraftObjectivePlan, minecraft_integer_attempt, run_minecraft_registered_task
from .shared.defaults import DEFAULT_BRANCH_ID
from .shared.sampling import resolve_route_cost_axes, sample_route_cost_scene


TASK_ID = "task_games__minecraft__resource_route_cost"
QUERY_ID = DEFAULT_BRANCH_ID
PROMPT_QUERY_KEY = "resource_route_cost"
SUPPORTED_QUERY_IDS = (QUERY_ID,)


def _prepare_route_cost_objective(
    instance_seed,
    task_params,
    _selected_branch,
    _branch_probabilities,
    gen_defaults,
) -> MinecraftObjectivePlan:
    """Resolve route axes and bind single-track cost construction."""

    axes = resolve_route_cost_axes(
        int(instance_seed),
        gen_defaults=gen_defaults,
        namespace="games.minecraft.route_cost",
        params=task_params,
    )

    def construct_attempt(rng, resolved_axes):
        sample = sample_route_cost_scene(rng=rng, axes=resolved_axes)
        return minecraft_integer_attempt(
            sample=sample,
            prompt_key=PROMPT_QUERY_KEY,
            object_description_key="object_description_block_world",
            answer_hint_key=f"answer_hint_{PROMPT_QUERY_KEY}",
            annotation_hint_key=f"annotation_hint_{PROMPT_QUERY_KEY}",
            example_annotation=[[290, 286, 340, 342], [319, 257, 369, 313], [498, 182, 548, 238], [527, 210, 577, 266]],
            example_answer=4,
            extra_query_params={
                "prompt_query_key": PROMPT_QUERY_KEY,
                "track_cells": [list(cell) for cell in sample.track_cells],
                "track_raised_block_count": int(sample.answer),
            },
        )

    return MinecraftObjectivePlan(
        axes=axes,
        attempt_namespace="games.minecraft.route_cost",
        construct_attempt=construct_attempt,
    )


@register_task
class GamesMinecraftResourceRouteCostTask:
    """Count raised stone or dirt blocks along one Minecraft-like track."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'topology')
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS
    _default_branch = QUERY_ID
    _namespace = "games.minecraft.route_cost"
    _prepare_objective = staticmethod(_prepare_route_cost_objective)

    def generate(
        self,
        instance_seed,
        *,
        params=None,
        max_attempts=100,
    ):
        """Generate a single-track route cost task instance."""

        return run_minecraft_registered_task(
            self,
            int(instance_seed),
            params=params or {},
            max_attempts=int(max_attempts),
        )


__all__ = ["GamesMinecraftResourceRouteCostTask"]
