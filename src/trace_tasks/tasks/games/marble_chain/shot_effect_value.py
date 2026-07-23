"""Compute the numeric effect of the marked marble-chain shot."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task

from ._lifecycle import MarbleObjectivePlan, MarbleSingleQueryTaskBase, marble_bbox_set_attempt, run_marble_registered_task
from .shared.defaults import SCENE_ID
from .shared.prompts import make_marble_prompt_slots
from .shared.rules import popped_marble_annotation_ids
from .shared.sampling import (
    interior_slots,
    resolve_chain_length_axis,
    resolve_color_count_axis,
    resolve_target_pop_axis,
    sample_chain_state,
)
from .shared.state import MarbleSample, MarbleSceneAxes


TASK_ID = "task_games__marble_chain__shot_effect_value"
PROMPT_SLOTS = make_marble_prompt_slots(
    prompt_query_key="pop_count_after_marked_shot",
    answer_hint_key="answer_hint_pop_count_after_marked_shot",
    annotation_hint_key="annotation_hint_pop_count_after_marked_shot",
    example_annotation=[[430, 206, 466, 242], [484, 224, 520, 260], [533, 258, 569, 294], [572, 304, 608, 340]],
    example_answer=4,
)


def _sample_shot_effect_scene(
    rng: Any,
    *,
    axes: MarbleSceneAxes,
    chain_length: int,
    color_count: int,
    target_answer: int,
    metadata: Mapping[str, Any],
) -> MarbleSample:
    """Construct a marked shot with the requested immediate pop count."""

    desired_pop_count = int(target_answer)
    for _attempt in range(800):
        chain_colors, shooter_color, outcomes = sample_chain_state(
            rng,
            chain_length=int(chain_length),
            color_count=int(color_count),
        )
        candidate_slots = [
            slot
            for slot, outcome in outcomes.items()
            if int(outcome.pop_count) == int(desired_pop_count)
        ]
        candidate_slots = interior_slots(candidate_slots, chain_length=len(chain_colors))
        if not candidate_slots:
            continue
        marked_slot = int(candidate_slots[int(rng.randrange(len(candidate_slots)))])
        marked_outcome = outcomes[int(marked_slot)]
        return MarbleSample(
            scene_variant=str(axes.scene_variant),
            chain_colors=tuple(chain_colors),
            shooter_color=str(shooter_color),
            answer=int(marked_outcome.pop_count),
            answer_type="integer",
            option_specs=(),
            marked_slot_index=int(marked_slot),
            marked_outcome=marked_outcome,
            target_pop_count=int(desired_pop_count),
            annotation_entity_ids=popped_marble_annotation_ids(marked_outcome),
            metadata=dict(metadata),
        )
    raise ValueError("failed to sample marked marble-chain shot effect")


def _prepare_shot_effect_objective(
    instance_seed: int,
    task_params: Mapping[str, Any],
    _selected_query_id: str,
    _branch_probabilities: Mapping[str, float],
    _axes: MarbleSceneAxes,
    gen_defaults: Mapping[str, Any],
) -> MarbleObjectivePlan:
    """Resolve axes and bind marked-shot numeric pop-count semantics."""

    namespace = f"{SCENE_ID}.shot_effect"
    chain_axis = resolve_chain_length_axis(
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        params=task_params,
        namespace=namespace,
    )
    color_axis = resolve_color_count_axis(
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        params=task_params,
        namespace=namespace,
    )
    target_axis = resolve_target_pop_axis(
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        params=task_params,
        namespace=namespace,
        explicit_key="target_answer",
    )
    query_params = {
        "chain_length": int(chain_axis.value),
        "chain_length_support": [int(value) for value in chain_axis.support],
        "chain_length_probabilities": dict(chain_axis.probabilities),
        "color_count": int(color_axis.value),
        "color_count_support": [int(value) for value in color_axis.support],
        "color_count_probabilities": dict(color_axis.probabilities),
        "target_answer": int(target_axis.value),
        "target_answer_support": [int(value) for value in target_axis.support],
        "target_answer_probabilities": dict(target_axis.probabilities),
        "target_pop_count": int(target_axis.value),
        "desired_pop_count": int(target_axis.value),
    }

    def construct_attempt(rng: Any, axes: MarbleSceneAxes):
        """Bind the marked arrow, popped-marble annotation, and integer answer."""

        sample = _sample_shot_effect_scene(
            rng,
            axes=axes,
            chain_length=int(chain_axis.value),
            color_count=int(color_axis.value),
            target_answer=int(target_axis.value),
            metadata=query_params,
        )
        return marble_bbox_set_attempt(
            answer_gt=TypedValue(type="integer", value=int(sample.answer)),
            sample=sample,
            prompt_slots=PROMPT_SLOTS,
        )

    return MarbleObjectivePlan(
        attempt_namespace=namespace,
        construct_attempt=construct_attempt,
    )


@register_task
class GamesMarbleChainShotEffectValueTask(MarbleSingleQueryTaskBase):
    """Compute the numeric effect of the marked marble-chain shot."""

    task_id = TASK_ID
    reasoning_operations = ('counting', 'topology', 'state_update')
    _namespace = f"{SCENE_ID}.shot_effect"
    _prepare_objective = staticmethod(_prepare_shot_effect_objective)

    def generate(self, instance_seed: int, *, params: dict[str, Any] | None = None, max_attempts: int = 100) -> TaskOutput:
        return run_marble_registered_task(self, instance_seed, params=params, max_attempts=max_attempts)


__all__ = ["GamesMarbleChainShotEffectValueTask", "TASK_ID"]
