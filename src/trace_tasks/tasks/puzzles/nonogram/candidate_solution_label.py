"""Public nonogram task for selecting the full grid satisfying all clues."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults

from ._lifecycle import (
    build_nonogram_option_label_case,
    retry_nonogram_generation,
)
from .shared.sampling import sample_candidate_solution_dataset
from .shared.state import DOMAIN, NonogramDataset, SCENE_ID

TASK_ID = "task_puzzles__nonogram__candidate_solution_label"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
PROMPT_TASK_KEY = "candidate_solution_label_query"
PROMPT_QUERY_KEY = "candidate_solution_label"
_NAMESPACE_BASE = f"{DOMAIN}.{SCENE_ID}.candidate_solution_label"
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS_UNUSED = (
    load_scene_generation_rendering_prompt_defaults(DOMAIN, SCENE_ID, task_id=TASK_ID)
)


@register_task
class PuzzlesNonogramCandidateSolutionLabelTask:
    """Select the filled-grid option satisfying all visible clues."""

    task_id = TASK_ID
    reasoning_operations = ('matching',)
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
        """Generate one solution-selection task with scalar bbox annotation."""

        return retry_nonogram_generation(
            build_case=_build_candidate_solution_case,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
        )


def _build_candidate_solution_case(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
) -> TaskOutput:
    """Sample clue-consistent options, bind the answer, and annotate it."""

    return build_nonogram_option_label_case(
        task_id=TASK_ID,
        namespace=_NAMESPACE_BASE,
        instance_seed=int(instance_seed),
        params=params,
        generation_defaults=_GEN_DEFAULTS,
        rendering_defaults=_RENDER_DEFAULTS,
        prompt_task_key=PROMPT_TASK_KEY,
        prompt_query_key=PROMPT_QUERY_KEY,
        dataset_builder=sample_candidate_solution_dataset,
        prompt_dynamic_slots=_candidate_prompt_slots,
        validate_dataset=_validate_candidate_solution_dataset,
    )


def _candidate_prompt_slots(_dataset: NonogramDataset) -> Mapping[str, Any]:
    """Return task-local dynamic prompt slots for full-grid selection."""

    return {}


def _validate_candidate_solution_dataset(dataset: NonogramDataset) -> None:
    """Validate the answer label matches exactly one clue-consistent option."""

    correct = [
        str(option["option_label"])
        for option in dataset.option_specs
        if bool(option.get("is_correct", False))
    ]
    if correct != [str(dataset.answer_value)]:
        raise ValueError("nonogram candidate-solution answer drifted from option specs")


__all__ = [
    "PROMPT_QUERY_KEY",
    "PuzzlesNonogramCandidateSolutionLabelTask",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
]
