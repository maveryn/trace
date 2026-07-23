"""Select the marked Mancala option pit with the largest post-sow count."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task

from ._lifecycle import MancalaAttemptResult, MancalaObjectivePlan, MancalaSingleQueryTaskBase, build_mancala_attempt_result, run_mancala_registered_task
from .shared.annotations import pit_bbox_annotation
from .shared.prompts import make_mancala_prompt_slots
from .shared.rules import pit_label, sow_counts
from .shared.sampling import random_initial_counts, resolve_mancala_label_axis
from .shared.state import DEFAULTS, LABELS, OPTION_LABELS, SCENE_ID, MancalaSample, MancalaSceneAxes


TASK_ID = "task_games__mancala_pit_board__max_post_sow_option_label"
PROMPT_SLOTS = make_mancala_prompt_slots(
    prompt_query_key="max_post_sow_option_label",
    answer_hint_key="answer_hint_max_post_sow_option_label",
    annotation_hint_key="annotation_hint_max_post_sow_option_label",
    example_annotation=[420, 140, 520, 204],
    example_answer="B",
)


def _construct_max_post_sow_option_sample(
    *,
    rng: Any,
    axes: MancalaSceneAxes,
    answer_option_label: str,
) -> MancalaSample:
    """Construct four options where the requested label uniquely has the most seeds."""

    if str(answer_option_label) not in OPTION_LABELS:
        raise ValueError("answer option label must be one of the visible Mancala option labels")
    correct_slot = int(OPTION_LABELS.index(str(answer_option_label)))
    min_source = int(axes.min_source_seed_count)
    max_source = int(axes.max_source_seed_count)

    for _attempt in range(300):
        source_index = int(rng.randrange(len(LABELS)))
        source_seed_count = int(rng.randint(max(1, min_source), max_source))
        counts = random_initial_counts(rng=rng, axes=axes)
        counts[int(source_index)] = int(source_seed_count)
        final_counts, path = sow_counts(counts, int(source_index))

        candidate_indices = [index for index in range(len(LABELS)) if int(index) != int(source_index)]
        rng.shuffle(candidate_indices)
        winner_candidates = []
        for index in candidate_indices:
            lower_candidates = [
                other
                for other in candidate_indices
                if int(other) != int(index) and int(final_counts[int(other)]) < int(final_counts[int(index)])
            ]
            if len(lower_candidates) >= len(OPTION_LABELS) - 1:
                winner_candidates.append((int(index), tuple(int(value) for value in lower_candidates)))
        if not winner_candidates:
            continue

        correct_index, lower_candidates = winner_candidates[int(rng.randrange(len(winner_candidates)))]
        distractors = list(lower_candidates)
        rng.shuffle(distractors)
        option_indices: list[int | None] = [None] * len(OPTION_LABELS)
        option_indices[correct_slot] = int(correct_index)
        distractor_iter = iter(distractors[: len(OPTION_LABELS) - 1])
        for slot_index in range(len(option_indices)):
            if option_indices[slot_index] is None:
                option_indices[slot_index] = int(next(distractor_iter))

        return MancalaSample(
            initial_counts=tuple(int(value) for value in counts),
            final_counts=tuple(int(value) for value in final_counts),
            source_index=int(source_index),
            sowing_path_indices=tuple(int(value) for value in path),
            landing_index=int(path[-1]),
            target_index=None,
            construction_mode="unique_max_post_sow_option_count",
            option_pit_indices=tuple(int(index) for index in option_indices if index is not None),
            option_labels=OPTION_LABELS,
        )
    raise ValueError("failed to construct Mancala max post-sow option sample")


def _prepare_max_post_sow_option_objective(
    instance_seed: int,
    task_params: Mapping[str, Any],
    selected_query_id: str,
    _branch_probabilities: Mapping[str, float],
    _axes: MancalaSceneAxes,
    gen_defaults: Mapping[str, Any],
) -> MancalaObjectivePlan:
    """Resolve the requested winning option label and bind construction."""

    answer_option_axis = resolve_mancala_label_axis(
        instance_seed=int(instance_seed),
        params=task_params,
        gen_defaults=gen_defaults,
        namespace=f"{SCENE_ID}.max_post_sow_option.answer_option_label",
        support_key="answer_option_label_support",
        explicit_key="answer_option_label",
        weights_key="answer_option_label_weights",
        balance_flag_key="balanced_answer_option_label_sampling",
        fallback_support=DEFAULTS.landing_option_label_support,
    )

    def construct_attempt(rng: Any, axes: MancalaSceneAxes) -> MancalaAttemptResult:
        """Bind the unique maximum option, annotation, and trace fields."""

        sample = _construct_max_post_sow_option_sample(
            rng=rng,
            axes=axes,
            answer_option_label=str(answer_option_axis.value),
        )
        option_pits_by_label = {
            str(option_label): str(pit_label(int(option_index)))
            for option_label, option_index in zip(sample.option_labels, sample.option_pit_indices)
        }
        option_final_counts_by_label = {
            str(option_label): int(sample.final_counts[int(option_index)])
            for option_label, option_index in zip(sample.option_labels, sample.option_pit_indices)
        }
        winning_pit_label = str(option_pits_by_label[str(answer_option_axis.value)])
        winning_pit_id = f"pit_{winning_pit_label}"
        return build_mancala_attempt_result(
            answer_gt=TypedValue(type="option_letter", value=str(answer_option_axis.value)),
            sample=sample,
            prompt_slots=PROMPT_SLOTS,
            build_annotation=lambda rendered: pit_bbox_annotation(
                rendered=rendered,
                pit_id=winning_pit_id,
                role_name="winning_option_pit",
            ),
            selected_query_id=str(selected_query_id),
            annotation_entity_ids={"winning_option_pit": winning_pit_id},
            extra_execution_fields={
                "scene_variant": str(axes.scene_variant),
                "style_variant": str(axes.style_variant),
                "answer_option_label": str(answer_option_axis.value),
                "option_pits_by_label": dict(option_pits_by_label),
                "option_final_counts_by_label": dict(option_final_counts_by_label),
                "winning_option_pit": str(winning_pit_label),
            },
            extra_query_params={
                "answer_option_label": str(answer_option_axis.value),
                "answer_option_label_support": list(answer_option_axis.support),
                "answer_option_label_probabilities": dict(answer_option_axis.probabilities),
            },
            relations_extra={
                "source_pit": str(pit_label(sample.source_index)),
                "target_pit": None,
                "landing_pit": str(pit_label(sample.landing_index)),
                "answer_option_label": str(answer_option_axis.value),
                "winning_option_pit": str(winning_pit_label),
            },
        )

    return MancalaObjectivePlan(
        attempt_namespace=f"{SCENE_ID}.max_post_sow_option",
        construct_attempt=construct_attempt,
    )


@register_task
class GamesMancalaPitBoardMaxPostSowOptionLabelTask(MancalaSingleQueryTaskBase):
    """Select the marked option pit with the largest count after sowing."""

    task_id = TASK_ID
    reasoning_operations = ('ranking', 'state_update')
    _namespace = f"{SCENE_ID}.max_post_sow_option"
    _prepare_objective = staticmethod(_prepare_max_post_sow_option_objective)

    def generate(self, instance_seed: int, *, params: dict[str, Any] | None = None, max_attempts: int = 100) -> TaskOutput:
        return run_mancala_registered_task(self, instance_seed, params=params, max_attempts=max_attempts)


__all__ = ["GamesMancalaPitBoardMaxPostSowOptionLabelTask", "TASK_ID"]
