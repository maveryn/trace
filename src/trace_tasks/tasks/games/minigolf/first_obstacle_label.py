"""Identify the first obstacle hit by a shown Mini-golf cue."""

from __future__ import annotations

from trace_tasks.tasks.registry import register_task

from ._lifecycle import MinigolfObjectivePlan, minigolf_string_answer_attempt, run_minigolf_registered_task
from .shared.annotations import minigolf_obstacle_point_annotation
from .shared.defaults import DEFAULT_BRANCH_ID, OBSTACLE_LABELS
from .shared.sampling import resolve_minigolf_axes, resolve_minigolf_label_choice, sample_first_obstacle_scene


TASK_ID = "task_games__minigolf__first_obstacle_label"
PROMPT_QUERY_KEY = "first_obstacle_label"
SUPPORTED_QUERY_IDS = (DEFAULT_BRANCH_ID,)


def _prepare_first_obstacle_objective(
    instance_seed,
    task_params,
    _selected_branch,
    branch_probabilities,
    gen_defaults,
) -> MinigolfObjectivePlan:
    """Resolve target-label axes and bind first-obstacle construction."""

    axes = resolve_minigolf_axes(
        int(instance_seed),
        gen_defaults=gen_defaults,
        namespace="games.minigolf.first_obstacle",
        params=task_params,
    )
    visible_label_support = tuple(str(label) for label in OBSTACLE_LABELS[: int(axes.obstacle_count)])
    target_label, target_label_probabilities = resolve_minigolf_label_choice(
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        params=task_params,
        support_key="target_obstacle_label_support",
        explicit_key="target_obstacle_label",
        fallback_support=visible_label_support,
        namespace="games.minigolf.first_obstacle.target_label",
        balanced_flag_key="balanced_target_obstacle_label_sampling",
    )

    def construct_attempt(rng, resolved_axes):
        sample = sample_first_obstacle_scene(
            rng=rng,
            axes=resolved_axes,
            target_label=str(target_label),
        )
        target_id = str(sample.target_obstacle_id)
        return minigolf_string_answer_attempt(
            sample=sample,
            prompt_key=PROMPT_QUERY_KEY,
            object_description_key=f"object_description_{str(resolved_axes.scene_variant)}",
            answer_hint_key=f"answer_hint_{PROMPT_QUERY_KEY}",
            annotation_hint_key=f"annotation_hint_{PROMPT_QUERY_KEY}",
            example_annotation=[486, 258],
            example_answer="D",
            bind_annotation=lambda rendered: minigolf_obstacle_point_annotation(
                rendered=rendered,
                obstacle_id=target_id,
            ),
            annotation_entity_ids=(target_id,),
            extra_query_params={
                "prompt_query_key": PROMPT_QUERY_KEY,
                "target_obstacle_label": str(target_label),
                "target_obstacle_label_probabilities": dict(target_label_probabilities),
            },
        )

    return MinigolfObjectivePlan(
        axes=axes,
        attempt_namespace="games.minigolf.first_obstacle",
        construct_attempt=construct_attempt,
    )


@register_task
class GamesMinigolfFirstObstacleLabelTask:
    """Identify the first labeled obstacle hit by the cue."""

    task_id = TASK_ID
    reasoning_operations = ('ranking', 'spatial_relations', 'topology', 'state_update')
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS
    _default_branch = DEFAULT_BRANCH_ID
    _namespace = "games.minigolf.first_obstacle"
    _prepare_objective = staticmethod(_prepare_first_obstacle_objective)

    def generate(self, instance_seed, *, params=None, max_attempts=100):
        return run_minigolf_registered_task(
            self,
            int(instance_seed),
            params=params or {},
            max_attempts=int(max_attempts),
        )


__all__ = ["GamesMinigolfFirstObstacleLabelTask"]
