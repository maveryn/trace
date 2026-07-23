"""Select the matchstick number reachable by adding or removing one stick."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults

from ._lifecycle import (
    bind_number_transform_output,
    run_matchstick_public_task,
)
from .shared.rendering import render_number_scene
from .shared.sampling import build_number_dataset
from .shared.state import DOMAIN, NumberDataset, SCENE_ID


TASK_ID = "task_puzzles__matchstick__matchstick_number_transform_label"
SUPPORTED_QUERY_IDS = ("add_one_stick", "remove_one_stick")
TASK_PROMPT_KEY = "matchstick_number_transform_query"
STICK_DELTA_BY_QUERY = {"add_one_stick": 1, "remove_one_stick": -1}

_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS_UNUSED = (
    load_scene_generation_rendering_prompt_defaults(DOMAIN, SCENE_ID, task_id=TASK_ID)
)


def _build_dataset_for_query(
    *,
    query_id: str,
    scene_variant: str,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    namespace: str,
    instance_seed: int,
) -> NumberDataset:
    """Translate the task query into an add/remove-one-stick sampling program."""

    return build_number_dataset(
        stick_delta=int(STICK_DELTA_BY_QUERY[str(query_id)]),
        scene_variant=str(scene_variant),
        params=params,
        generation_defaults=generation_defaults,
        namespace=str(namespace),
        instance_seed=int(instance_seed),
    )


def _bind_output(
    *,
    dataset: NumberDataset,
    context: Any,
    query_id: str,
    query_probabilities: Mapping[str, float],
    scene_variant_probabilities: Mapping[str, float],
) -> Any:
    """Bind number-transform output using the task-owned stick delta."""

    return bind_number_transform_output(
        dataset=dataset,
        context=context,
        query_id=str(query_id),
        query_probabilities=query_probabilities,
        scene_variant_probabilities=scene_variant_probabilities,
        stick_delta=int(STICK_DELTA_BY_QUERY[str(query_id)]),
    )


@register_task
class PuzzlesMatchstickNumberTransformLabelTask:
    """Choose the candidate number reachable from Source by one stick edit."""

    task_id = TASK_ID
    reasoning_operations = ('transformation', 'state_update')
    domain = DOMAIN
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(
        self,
        instance_seed: int,
        *,
        params: Dict[str, Any],
        max_attempts: int,
    ) -> TaskOutput:
        """Run the task-owned number-transform callbacks through scene plumbing."""

        return run_matchstick_public_task(
            task_id=TASK_ID,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            task_prompt_key=TASK_PROMPT_KEY,
            params=params,
            generation_defaults=_GEN_DEFAULTS,
            rendering_defaults=_RENDER_DEFAULTS,
            instance_seed=int(instance_seed),
            max_attempts=int(max_attempts),
            dataset_builder=_build_dataset_for_query,
            render_scene=render_number_scene,
            output_binder=_bind_output,
        )


__all__ = [
    "PuzzlesMatchstickNumberTransformLabelTask",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
]
