"""Identify the Mini-golf shot path that reaches the hole."""

from __future__ import annotations

from trace_tasks.tasks.registry import register_task

from ._lifecycle import MinigolfObjectivePlan, minigolf_string_answer_attempt, run_minigolf_registered_task
from .shared.annotations import minigolf_path_segment_annotation
from .shared.defaults import DEFAULT_BRANCH_ID
from .shared.sampling import resolve_minigolf_axes, resolve_minigolf_integer_choice, sample_shot_options_scene


TASK_ID = "task_games__minigolf__shot_path_label"
PROMPT_QUERY_KEY = "shot_path_label"
SUPPORTED_QUERY_IDS = (DEFAULT_BRANCH_ID,)
PATH_OPTION_COUNT_SUPPORT_KEY = "path_option_count_support"
PATH_OPTION_COUNT_FALLBACK_SUPPORT = (4, 5, 6)
TARGET_PATH_INDEX_SUPPORT_KEY = "target_path_index_support"
TARGET_PATH_INDEX_FALLBACK_SUPPORT = (0, 1, 2, 3, 4, 5)


def _prepare_shot_path_objective(
    instance_seed,
    task_params,
    _selected_branch,
    branch_probabilities,
    gen_defaults,
) -> MinigolfObjectivePlan:
    """Resolve option axes and bind unique hole-reaching shot construction."""

    axes = resolve_minigolf_axes(
        int(instance_seed),
        gen_defaults=gen_defaults,
        namespace="games.minigolf.shot_path",
        params=task_params,
    )
    option_count, option_count_probabilities = resolve_minigolf_integer_choice(
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        params=task_params,
        support_key=PATH_OPTION_COUNT_SUPPORT_KEY,
        explicit_key="path_option_count",
        fallback_support=PATH_OPTION_COUNT_FALLBACK_SUPPORT,
        namespace="games.minigolf.shot_path.option_count",
        balanced_flag_key="balanced_path_option_count_sampling",
    )
    target_index, target_index_probabilities = resolve_minigolf_integer_choice(
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        params=task_params,
        support_key=TARGET_PATH_INDEX_SUPPORT_KEY,
        explicit_key="target_path_index",
        fallback_support=TARGET_PATH_INDEX_FALLBACK_SUPPORT,
        namespace="games.minigolf.shot_path.target_index",
        balanced_flag_key="balanced_target_path_sampling",
    )
    option_count = max(int(option_count), int(target_index) + 1)

    def construct_attempt(rng, resolved_axes):
        sample = sample_shot_options_scene(
            rng=rng,
            axes=resolved_axes,
            option_count=int(option_count),
            target_index=int(target_index),
        )
        target_id = str(sample.target_path_id)
        return minigolf_string_answer_attempt(
            sample=sample,
            prompt_key=PROMPT_QUERY_KEY,
            object_description_key=f"object_description_{str(resolved_axes.scene_variant)}",
            answer_hint_key=f"answer_hint_{PROMPT_QUERY_KEY}",
            annotation_hint_key=f"annotation_hint_{PROMPT_QUERY_KEY}",
            example_annotation=[[486, 562], [592, 520]],
            example_answer="3",
            bind_annotation=lambda rendered: minigolf_path_segment_annotation(
                rendered=rendered,
                path_id=target_id,
            ),
            annotation_entity_ids=(target_id,),
            extra_query_params={
                "prompt_query_key": PROMPT_QUERY_KEY,
                "path_option_count": int(option_count),
                "target_path_index": int(target_index),
                "path_option_count_probabilities": dict(option_count_probabilities),
                "target_path_index_probabilities": dict(target_index_probabilities),
            },
            execution_extra={"target_path_index": int(target_index)},
        )

    return MinigolfObjectivePlan(
        axes=axes,
        attempt_namespace="games.minigolf.shot_path",
        construct_attempt=construct_attempt,
    )


@register_task
class GamesMinigolfShotPathLabelTask:
    """Identify the numbered shot cue that reaches the hole."""

    task_id = TASK_ID
    reasoning_operations = ('spatial_relations', 'topology', 'state_update')
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS
    _default_branch = DEFAULT_BRANCH_ID
    _namespace = "games.minigolf.shot_path"
    _prepare_objective = staticmethod(_prepare_shot_path_objective)

    def generate(self, instance_seed, *, params=None, max_attempts=100):
        return run_minigolf_registered_task(
            self,
            int(instance_seed),
            params=params or {},
            max_attempts=int(max_attempts),
        )


__all__ = ["GamesMinigolfShotPathLabelTask"]
