"""Select the labeled dart with the highest simplified score."""

from __future__ import annotations

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults

from ._lifecycle import DartsObjectivePlan, dart_point_attempt, run_darts_lifecycle
from .shared.defaults import SCENE_ID
from .shared.prompts import darts_option_letter_json_examples, darts_output_slots
from .shared.sampling import (
    resolve_darts_integer_axis,
    sample_darts_for_highest_scoring_label,
)


TASK_ID = "task_games__darts__highest_scoring_dart_label"
QUERY_ID = "single"
PROMPT_QUERY_KEY = "highest_scoring_dart_label"
SUPPORTED_QUERY_IDS = (QUERY_ID,)
OPTION_LABELS = ("A", "B", "C", "D")
TARGET_SUPPORT_KEY = "highest_scoring_dart_label_support"
TARGET_FALLBACK_SUPPORT = (0, 1, 2, 3)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS_UNUSED = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
    task_id=TASK_ID,
)


def _prepare_highest_scoring_dart_label_objective(
    instance_seed,
    task_params,
    _query_id,
    _query_probabilities,
    render_params,
):
    """Resolve the answer-label axis and bind the highest-score objective."""

    label_axis = resolve_darts_integer_axis(
        instance_seed=int(instance_seed),
        params=task_params,
        gen_defaults=_GEN_DEFAULTS,
        support_key=TARGET_SUPPORT_KEY,
        explicit_key="target_answer",
        fallback_support=TARGET_FALLBACK_SUPPORT,
        namespace="highest_scoring_dart_label.target_answer",
        balanced_flag_key="balanced_target_answer_sampling",
    )
    json_example, json_example_answer_only = darts_option_letter_json_examples()
    prompt_dynamic_slots = darts_output_slots(
        prompt_query_key=PROMPT_QUERY_KEY,
        json_example=json_example,
        json_example_answer_only=json_example_answer_only,
    )
    query_params = {
        "target_answer": int(label_axis.value),
        "target_answer_support": [int(value) for value in label_axis.support],
        "target_answer_probabilities": dict(label_axis.probabilities),
        "option_labels": [str(label) for label in OPTION_LABELS],
    }

    def construct_attempt(rng, _axes):
        target_index = int(label_axis.value)
        sample = sample_darts_for_highest_scoring_label(
            rng,
            target_label_index=int(target_index),
            option_labels=OPTION_LABELS,
            render_params=render_params,
        )
        correct_label = str(OPTION_LABELS[target_index])
        winning_dart = next(dart for dart in sample.darts if str(dart.label) == correct_label)
        scores_by_label = {
            str(dart.label): int(dart.score)
            for dart in sample.darts
            if dart.label is not None
        }
        return dart_point_attempt(
            sample=sample,
            answer_gt=TypedValue(type="option_letter", value=correct_label),
            annotation_entity_id=str(winning_dart.dart_id),
            execution_extra={
                "target_answer": int(target_index),
                "option_labels": [str(label) for label in OPTION_LABELS],
                "correct_option_label": correct_label,
                "correct_dart_id": str(winning_dart.dart_id),
                "winning_score": int(winning_dart.score),
                "scores_by_label": dict(scores_by_label),
            },
        )

    return DartsObjectivePlan(
        attempt_namespace="games.darts.highest_scoring_dart_label",
        prompt_query_key=PROMPT_QUERY_KEY,
        query_params=query_params,
        prompt_dynamic_slots=prompt_dynamic_slots,
        construct_attempt=construct_attempt,
    )


@register_task
class GamesDartsHighestScoringDartLabelTask:
    """Choose the labeled dart with the highest simplified score."""

    task_id = TASK_ID
    reasoning_operations = ('ranking', 'formula_evaluation')
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
            prepare_objective=_prepare_highest_scoring_dart_label_objective,
        )


__all__ = ["GamesDartsHighestScoringDartLabelTask"]
