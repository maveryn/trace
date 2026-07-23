"""Select the Bowling spare path that reaches all standing pins."""

from __future__ import annotations

from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults

from ._lifecycle import (
    BowlingObjectivePlan,
    bowling_integer_axis_spec,
    path_segment_label_attempt,
    resolve_bowling_integer_axis_specs,
    run_bowling_lifecycle,
)
from .shared.defaults import SCENE_ID
from .shared.rules import sample_spare_path_scene
from .shared.sampling import ResolvedBowlingSceneAxes


TASK_ID = "task_games__bowling__spare_path_label"
QUERY_ID = "spare_path_label"
SUPPORTED_QUERY_IDS = (QUERY_ID,)
SPARE_PATH_AXIS_SPECS = (
    bowling_integer_axis_spec(
        "path_option_count",
        (4, 5, 6),
        balanced_flag_key="balanced_path_option_count_sampling",
    ),
    bowling_integer_axis_spec(
        "target_path_index",
        tuple(range(6)),
        balanced_flag_key="balanced_target_path_sampling",
    ),
)
_GEN_DEFAULTS = load_scene_generation_rendering_prompt_defaults("games", SCENE_ID, task_id=TASK_ID)[0]


def _prepare_spare_path_objective(
    instance_seed,
    task_params,
    _query_id,
    _query_probabilities,
):
    """Resolve path-option axes and bind the spare-path constructor."""

    axis_values, query_params = resolve_bowling_integer_axis_specs(
        instance_seed=int(instance_seed),
        task_params=task_params,
        gen_defaults=_GEN_DEFAULTS,
        task_id=TASK_ID,
        axis_specs=SPARE_PATH_AXIS_SPECS,
    )
    option_axis = axis_values["path_option_count"]
    target_axis = axis_values["target_path_index"]
    path_option_count = max(int(option_axis.value), int(target_axis.value) + 1)
    query_params = {**query_params, "path_option_count": int(path_option_count)}

    def construct_attempt(rng, axes: ResolvedBowlingSceneAxes):
        sample = sample_spare_path_scene(
            rng=rng,
            scene_variant=str(axes.scene_variant),
            style_variant=str(axes.style_variant),
            path_option_count=int(path_option_count),
            target_path_index=int(target_axis.value),
        )
        return path_segment_label_attempt(
            sample=sample,
            answer_value=str(sample.target_path_label),
            execution_extra={
                "target_path_index": int(target_axis.value),
            },
        )

    return BowlingObjectivePlan(
        attempt_namespace="games.bowling.spare_path_label",
        prompt_query_key=QUERY_ID,
        render_mode="path_options",
        query_params=query_params,
        construct_attempt=construct_attempt,
    )


@register_task
class GamesBowlingSparePathLabelTask:
    """Identify the numbered path that covers the remaining standing pins."""

    task_id = TASK_ID
    reasoning_operations = ('spatial_relations', 'topology', 'state_update')
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed, *, params, max_attempts):
        return run_bowling_lifecycle(
            task_id=TASK_ID,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            prepare_objective=_prepare_spare_path_objective,
        )


__all__ = ["GamesBowlingSparePathLabelTask"]
