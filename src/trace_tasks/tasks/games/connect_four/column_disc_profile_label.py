"""Choose the labeled Connect Four column with a requested disc-color profile."""

from __future__ import annotations

from typing import Any, Mapping, Tuple

from trace_tasks.core.sampling import uniform_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.games.connect_four._lifecycle import ConnectFourObjectivePlan, run_connect_four_lifecycle
from trace_tasks.tasks.games.connect_four.shared.defaults import SCENE_ID
from trace_tasks.tasks.games.connect_four.shared.prompts import (
    connect_four_object_description,
    connect_four_output_slots,
    json_examples_for_label_answer,
)
from trace_tasks.tasks.games.connect_four.shared.sampling import (
    resolve_connect_four_scene_axes,
    sample_column_disc_profile_label_scene,
)
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.support_sampling import resolve_integer_support


TASK_ID = "task_games__connect_four__column_disc_profile_label"
QUERY_ID = "column_disc_profile_label"
PROMPT_QUERY_KEY = QUERY_ID
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (QUERY_ID,)
COLUMN_DISC_PROFILE_TOTAL_SUPPORT: Tuple[int, ...] = (2, 3, 4, 5)
_GEN_DEFAULTS, _RENDER_DEFAULTS_UNUSED, _PROMPT_DEFAULTS_UNUSED = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
    task_id=TASK_ID,
)


def _resolve_target_profile_counts(
    *,
    instance_seed: int,
    task_params: Mapping[str, Any],
) -> tuple[int, int, tuple[int, ...], dict[str, float]]:
    """Select target red/yellow counts for the unique column-profile label task."""

    total_support = resolve_integer_support(
        task_params,
        gen_defaults=_GEN_DEFAULTS,
        key="column_disc_profile_total_support",
        fallback=COLUMN_DISC_PROFILE_TOTAL_SUPPORT,
    )
    explicit_red = task_params.get("target_red_count")
    explicit_yellow = task_params.get("target_yellow_count")
    if explicit_red is not None or explicit_yellow is not None:
        if explicit_red is None or explicit_yellow is None:
            raise ValueError("target_red_count and target_yellow_count must be provided together")
        red_count = int(explicit_red)
        yellow_count = int(explicit_yellow)
        total = int(red_count + yellow_count)
        if int(red_count) <= 0 or int(yellow_count) <= 0 or int(total) not in set(total_support):
            raise ValueError("unsupported Connect Four column disc profile counts")
        return int(red_count), int(yellow_count), tuple(int(value) for value in total_support), {
            str(value): 1.0 / float(len(total_support)) for value in total_support
        }

    rng = spawn_rng(int(instance_seed), f"{TASK_ID}.target_profile")
    total = int(uniform_choice(rng, total_support))
    red_count = int(uniform_choice(rng, tuple(range(1, int(total)))))
    yellow_count = int(total - red_count)
    return int(red_count), int(yellow_count), tuple(int(value) for value in total_support), {
        str(value): 1.0 / float(len(total_support)) for value in total_support
    }


def _prepare_column_disc_profile_label_objective(
    instance_seed: int,
    task_params: Mapping[str, Any],
    selected_query_id: str,
    _query_probabilities: Mapping[str, float],
) -> ConnectFourObjectivePlan:
    """Bind unique column red/yellow count-profile label semantics."""

    axes = resolve_connect_four_scene_axes(
        int(instance_seed),
        params=task_params,
        gen_defaults=_GEN_DEFAULTS,
        namespace_suffix=str(selected_query_id),
    )
    target_red_count, target_yellow_count, total_support, total_probabilities = _resolve_target_profile_counts(
        instance_seed=int(instance_seed),
        task_params=task_params,
    )

    def construct_attempt(rng):
        return sample_column_disc_profile_label_scene(
            rng=rng,
            axes=axes,
            params=task_params,
            target_red_count=int(target_red_count),
            target_yellow_count=int(target_yellow_count),
        )

    def prompt_slots(sample) -> dict[str, Any]:
        json_example, json_example_answer_only = json_examples_for_label_answer()
        return {
            "object_description": connect_four_object_description(str(sample.scene_variant)),
            "target_red_count": int(target_red_count),
            "target_yellow_count": int(target_yellow_count),
            **connect_four_output_slots(
                prompt_query_key=PROMPT_QUERY_KEY,
                json_example=json_example,
                json_example_answer_only=json_example_answer_only,
            ),
        }

    def query_spec_params(sample) -> dict[str, Any]:
        return {
            "answer_label": str(sample.answer_label),
            "answer_column": int(sample.answer_column),
            "answer_support": [str(label) for label in sample.column_labels],
            "target_red_count": int(target_red_count),
            "target_yellow_count": int(target_yellow_count),
            "column_disc_profile_total_support": [int(value) for value in total_support],
            "column_disc_profile_total_probabilities": dict(total_probabilities),
        }

    def execution_updates(sample) -> dict[str, Any]:
        return {
            "answer_label": str(sample.answer_label),
            "answer_column": int(sample.answer_column),
            "answer_support": [str(label) for label in sample.column_labels],
            "column_labels": [str(label) for label in sample.column_labels],
            "target_red_count": int(target_red_count),
            "target_yellow_count": int(target_yellow_count),
        }

    return ConnectFourObjectivePlan(
        axes=axes,
        attempt_namespace=TASK_ID,
        construct_attempt=construct_attempt,
        prompt_query_key=PROMPT_QUERY_KEY,
        prompt_dynamic_slots=prompt_slots,
        answer_gt=lambda sample: TypedValue(type="string", value=str(sample.answer_label)),
        annotation_coords=lambda sample: sample.evaluation.annotation_coords,
        annotation_type="bbox_set",
        render_marked_square=lambda _sample: None,
        render_column_labels=lambda sample: sample.column_labels,
        query_spec_params=query_spec_params,
        execution_updates=execution_updates,
    )


@register_task
class GamesConnectFourColumnDiscProfileLabelTask:
    """Return the visible column label with the requested red/yellow disc counts."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'logical_composition')
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one board with a unique matching column profile."""

        return run_connect_four_lifecycle(
            task_id=TASK_ID,
            domain=self.domain,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=QUERY_ID,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            prepare_objective=_prepare_column_disc_profile_label_objective,
        )


__all__ = ["GamesConnectFourColumnDiscProfileLabelTask"]
