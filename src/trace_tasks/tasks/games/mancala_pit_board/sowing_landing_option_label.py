"""Select the option marking the final Mancala landing pit after one sowing move."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task

from ._lifecycle import MancalaAttemptResult, MancalaObjectivePlan, MancalaSingleQueryTaskBase, build_mancala_attempt_result, run_mancala_registered_task
from .shared.annotations import pit_bbox_annotation
from .shared.prompts import make_mancala_prompt_slots
from .shared.rules import pit_index, pit_label, sow_counts
from .shared.sampling import random_initial_counts, resolve_mancala_label_axis
from .shared.state import DEFAULTS, LABELS, OPTION_LABELS, SCENE_ID, MancalaSample, MancalaSceneAxes


TASK_ID = "task_games__mancala_pit_board__sowing_landing_option_label"
PROMPT_SLOTS = make_mancala_prompt_slots(
    prompt_query_key="sowing_landing_option_label",
    answer_hint_key="answer_hint_sowing_landing_option_label",
    annotation_hint_key="annotation_hint_sowing_landing_option_label",
    example_annotation=[420, 140, 520, 204],
    example_answer="B",
)


def _construct_landing_sample(*, rng: Any, axes: MancalaSceneAxes, target_label: str) -> MancalaSample:
    """Place a source pit so the final sown seed lands in the target label."""

    target_index = pit_index(str(target_label))
    min_source = int(axes.min_source_seed_count)
    max_source = int(axes.max_source_seed_count)
    viable_seed_counts = [
        seed_count
        for seed_count in range(max(1, min_source), max_source + 1)
        if seed_count <= len(LABELS) - 1
    ]
    rng.shuffle(viable_seed_counts)
    if not viable_seed_counts:
        raise ValueError("no viable Mancala source seed counts")
    source_seed_count = int(viable_seed_counts[0])
    source_index = (int(target_index) - int(source_seed_count)) % len(LABELS)
    counts = random_initial_counts(rng=rng, axes=axes)
    counts[int(source_index)] = int(source_seed_count)
    final_counts, path = sow_counts(counts, int(source_index))
    if not path or int(path[-1]) != int(target_index):
        raise ValueError("constructed Mancala landing mismatch")
    return MancalaSample(
        initial_counts=tuple(int(value) for value in counts),
        final_counts=tuple(int(value) for value in final_counts),
        source_index=int(source_index),
        sowing_path_indices=tuple(int(value) for value in path),
        landing_index=int(path[-1]),
        target_index=None,
        construction_mode="target_conditioned_last_seed_landing_option",
    )


def _option_pit_indices(
    *,
    rng: Any,
    correct_index: int,
    source_index: int,
    answer_option_label: str,
) -> tuple[int, ...]:
    """Bind four visible option markers with the correct one at the requested label."""

    if str(answer_option_label) not in OPTION_LABELS:
        raise ValueError("answer option label must be one of the visible Mancala option labels")
    correct_slot = int(OPTION_LABELS.index(str(answer_option_label)))
    distractor_candidates = [
        index
        for index in range(len(LABELS))
        if int(index) not in {int(correct_index), int(source_index)}
    ]
    rng.shuffle(distractor_candidates)
    if len(distractor_candidates) < len(OPTION_LABELS) - 1:
        raise ValueError("not enough candidate pits for Mancala landing options")
    option_indices: list[int | None] = [None] * len(OPTION_LABELS)
    option_indices[correct_slot] = int(correct_index)
    distractor_iter = iter(distractor_candidates)
    for slot_index in range(len(option_indices)):
        if option_indices[slot_index] is None:
            option_indices[slot_index] = int(next(distractor_iter))
    return tuple(int(index) for index in option_indices if index is not None)


def _prepare_landing_objective(
    instance_seed: int,
    task_params: Mapping[str, Any],
    selected_query_id: str,
    _branch_probabilities: Mapping[str, float],
    _axes: MancalaSceneAxes,
    gen_defaults: Mapping[str, Any],
) -> MancalaObjectivePlan:
    """Resolve landing target and answer option axes, then bind construction."""

    target_axis = resolve_mancala_label_axis(
        instance_seed=int(instance_seed),
        params=task_params,
        gen_defaults=gen_defaults,
        namespace=f"{SCENE_ID}.landing.target_label",
        support_key="target_landing_label_support",
        explicit_key="target_landing_label",
        weights_key="target_landing_label_weights",
        balance_flag_key="balanced_target_landing_label_sampling",
        fallback_support=DEFAULTS.target_landing_label_support,
    )
    answer_option_axis = resolve_mancala_label_axis(
        instance_seed=int(instance_seed),
        params=task_params,
        gen_defaults=gen_defaults,
        namespace=f"{SCENE_ID}.landing.answer_option_label",
        support_key="answer_option_label_support",
        explicit_key="answer_option_label",
        weights_key="answer_option_label_weights",
        balance_flag_key="balanced_answer_option_label_sampling",
        fallback_support=DEFAULTS.landing_option_label_support,
    )

    def construct_attempt(rng: Any, axes: MancalaSceneAxes) -> MancalaAttemptResult:
        """Bind final landing pit, visible option markers, annotation, and trace fields."""

        sample = _construct_landing_sample(
            rng=rng,
            axes=axes,
            target_label=str(target_axis.value),
        )
        option_indices = _option_pit_indices(
            rng=rng,
            correct_index=int(sample.landing_index),
            source_index=int(sample.source_index),
            answer_option_label=str(answer_option_axis.value),
        )
        sample = MancalaSample(
            initial_counts=sample.initial_counts,
            final_counts=sample.final_counts,
            source_index=int(sample.source_index),
            sowing_path_indices=sample.sowing_path_indices,
            landing_index=int(sample.landing_index),
            target_index=None,
            construction_mode=str(sample.construction_mode),
            option_pit_indices=tuple(option_indices),
            option_labels=OPTION_LABELS,
        )
        landing_label = pit_label(sample.landing_index)
        landing_pit_id = f"pit_{landing_label}"
        option_pits_by_label = {
            str(option_label): str(pit_label(int(option_index)))
            for option_label, option_index in zip(sample.option_labels, sample.option_pit_indices)
        }
        return build_mancala_attempt_result(
            answer_gt=TypedValue(type="option_letter", value=str(answer_option_axis.value)),
            sample=sample,
            prompt_slots=PROMPT_SLOTS,
            build_annotation=lambda rendered: pit_bbox_annotation(
                rendered=rendered,
                pit_id=landing_pit_id,
                role_name="landing_pit",
            ),
            selected_query_id=str(selected_query_id),
            annotation_entity_ids={"landing_pit": landing_pit_id},
            extra_execution_fields={
                "scene_variant": str(axes.scene_variant),
                "style_variant": str(axes.style_variant),
                "target_landing_label": str(target_axis.value),
                "answer_option_label": str(answer_option_axis.value),
                "option_pits_by_label": dict(option_pits_by_label),
            },
            extra_query_params={
                "target_landing_label": str(target_axis.value),
                "target_landing_label_support": list(target_axis.support),
                "target_landing_label_probabilities": dict(target_axis.probabilities),
                "answer_option_label": str(answer_option_axis.value),
                "answer_option_label_support": list(answer_option_axis.support),
                "answer_option_label_probabilities": dict(answer_option_axis.probabilities),
            },
            relations_extra={
                "source_pit": str(pit_label(sample.source_index)),
                "target_pit": None,
                "landing_pit": str(landing_label),
                "answer_option_label": str(answer_option_axis.value),
            },
        )

    return MancalaObjectivePlan(
        attempt_namespace=f"{SCENE_ID}.landing",
        construct_attempt=construct_attempt,
    )


@register_task
class GamesMancalaPitBoardSowingLandingOptionLabelTask(MancalaSingleQueryTaskBase):
    """Select the option marking the final Mancala landing pit after one sowing move."""

    task_id = TASK_ID
    reasoning_operations = ('ranking', 'state_update')
    _namespace = f"{SCENE_ID}.landing"
    _prepare_objective = staticmethod(_prepare_landing_objective)

    def generate(self, instance_seed: int, *, params: dict[str, Any] | None = None, max_attempts: int = 100) -> TaskOutput:
        return run_mancala_registered_task(self, instance_seed, params=params, max_attempts=max_attempts)


__all__ = ["GamesMancalaPitBoardSowingLandingOptionLabelTask", "TASK_ID"]
