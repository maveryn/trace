"""Count seeds in a target Mancala pit after one sowing move."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task

from ._lifecycle import MancalaAttemptResult, MancalaObjectivePlan, MancalaSingleQueryTaskBase, build_mancala_attempt_result, run_mancala_registered_task
from .shared.annotations import keyed_pit_bbox_annotation
from .shared.prompts import make_mancala_prompt_slots
from .shared.rules import pit_label, sow_counts, sowing_path
from .shared.sampling import random_initial_counts, resolve_mancala_integer_axis
from .shared.state import DEFAULTS, LABELS, SCENE_ID, MancalaSample, MancalaSceneAxes


TASK_ID = "task_games__mancala_pit_board__post_sow_pit_count_value"
PROMPT_SLOTS = make_mancala_prompt_slots(
    prompt_query_key="post_sow_pit_count_value",
    answer_hint_key="answer_hint_post_sow_pit_count_value",
    annotation_hint_key="annotation_hint_post_sow_pit_count_value",
    example_annotation={"source_pit": [80, 140, 180, 204], "target_pit": [420, 140, 520, 204]},
    example_answer=5,
)


def _shuffled_non_source_targets(*, rng: Any, source_index: int) -> tuple[int, ...]:
    """Return target candidates for the count task, excluding the source pit."""

    candidates = [index for index in range(len(LABELS)) if index != int(source_index)]
    rng.shuffle(candidates)
    return tuple(int(index) for index in candidates)


def _pre_sow_target_count(*, desired_final_count: int, receives_seed: bool, max_seed_count: int) -> int | None:
    """Infer whether a target pit can start with a count that reaches the final value."""

    initial_count = int(desired_final_count) - (1 if bool(receives_seed) else 0)
    if initial_count < 0 or initial_count > int(max_seed_count):
        return None
    return int(initial_count)


def _construct_post_sow_count_sample(*, rng: Any, axes: MancalaSceneAxes, target_count: int) -> MancalaSample:
    """Construct a source/target pair where the target has the requested final count."""

    target_answer = int(target_count)
    min_source = int(axes.min_source_seed_count)
    max_source = int(axes.max_source_seed_count)
    max_seed = int(axes.max_seed_count_per_pit)
    if target_answer < 0 or target_answer > max_seed:
        raise ValueError("Mancala target count must fit seed support")
    for _attempt in range(200):
        source_index = int(rng.randrange(len(LABELS)))
        source_seed_count = int(rng.randint(max(1, min_source), max_source))
        path = sowing_path(source_index, source_seed_count)
        path_members = set(path)
        for target_index in _shuffled_non_source_targets(rng=rng, source_index=source_index):
            initial_target_count = _pre_sow_target_count(
                desired_final_count=target_answer,
                receives_seed=int(target_index) in path_members,
                max_seed_count=max_seed,
            )
            if initial_target_count is None:
                continue
            counts = random_initial_counts(rng=rng, axes=axes)
            counts[int(source_index)] = int(source_seed_count)
            counts[int(target_index)] = int(initial_target_count)
            final_counts, computed_path = sow_counts(counts, int(source_index))
            if int(final_counts[int(target_index)]) != int(target_answer):
                continue
            return MancalaSample(
                initial_counts=tuple(int(value) for value in counts),
                final_counts=tuple(int(value) for value in final_counts),
                source_index=int(source_index),
                sowing_path_indices=tuple(int(value) for value in computed_path),
                landing_index=int(computed_path[-1]),
                target_index=int(target_index),
                construction_mode="target_conditioned_post_sow_target_pit_count",
            )
    raise ValueError("failed to construct Mancala post-sow count sample")


def _prepare_post_sow_count_objective(
    instance_seed: int,
    task_params: Mapping[str, Any],
    selected_query_id: str,
    _branch_probabilities: Mapping[str, float],
    _axes: MancalaSceneAxes,
    gen_defaults: Mapping[str, Any],
) -> MancalaObjectivePlan:
    """Resolve the target final count and bind count-task construction."""

    target_axis = resolve_mancala_integer_axis(
        instance_seed=int(instance_seed),
        params=task_params,
        gen_defaults=gen_defaults,
        support_key="target_count_support",
        explicit_key="target_count",
        fallback_support=DEFAULTS.target_count_support,
        namespace=f"{SCENE_ID}.post_sow_count.target",
        balanced_flag_key="balanced_target_count_sampling",
    )

    def construct_attempt(rng: Any, axes: MancalaSceneAxes) -> MancalaAttemptResult:
        """Bind the target pit final count, annotation, and trace fields."""

        sample = _construct_post_sow_count_sample(
            rng=rng,
            axes=axes,
            target_count=int(target_axis.value),
        )
        assert sample.target_index is not None
        source_label = pit_label(sample.source_index)
        target_label = pit_label(int(sample.target_index))
        source_pit_id = f"pit_{source_label}"
        target_pit_id = f"pit_{target_label}"
        answer = int(sample.final_counts[int(sample.target_index)])
        return build_mancala_attempt_result(
            answer_gt=TypedValue(type="integer", value=int(answer)),
            sample=sample,
            prompt_slots=PROMPT_SLOTS,
            build_annotation=lambda rendered: keyed_pit_bbox_annotation(
                rendered=rendered,
                role_pit_ids={"source_pit": source_pit_id, "target_pit": target_pit_id},
            ),
            selected_query_id=str(selected_query_id),
            annotation_entity_ids={"source_pit": source_pit_id, "target_pit": target_pit_id},
            extra_execution_fields={
                "scene_variant": str(axes.scene_variant),
                "style_variant": str(axes.style_variant),
                "target_count": int(target_axis.value),
            },
            extra_query_params={
                "target_count": int(target_axis.value),
                "target_count_support": [int(value) for value in target_axis.support],
                "target_count_probabilities": dict(target_axis.probabilities),
            },
            relations_extra={
                "source_pit": str(source_label),
                "target_pit": str(target_label),
                "landing_pit": str(pit_label(sample.landing_index)),
            },
        )

    return MancalaObjectivePlan(
        attempt_namespace=f"{SCENE_ID}.post_sow_count",
        construct_attempt=construct_attempt,
    )


@register_task
class GamesMancalaPitBoardPostSowPitCountTask(MancalaSingleQueryTaskBase):
    """Count seeds in a target Mancala pit after one sowing move."""

    task_id = TASK_ID
    reasoning_operations = ('counting', 'state_update')
    _namespace = f"{SCENE_ID}.post_sow_count"
    _prepare_objective = staticmethod(_prepare_post_sow_count_objective)

    def generate(self, instance_seed: int, *, params: dict[str, Any] | None = None, max_attempts: int = 100) -> TaskOutput:
        return run_mancala_registered_task(self, instance_seed, params=params, max_attempts=max_attempts)


__all__ = ["GamesMancalaPitBoardPostSowPitCountTask", "TASK_ID"]
