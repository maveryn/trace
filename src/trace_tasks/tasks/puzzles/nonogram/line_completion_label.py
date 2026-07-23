"""Public nonogram task for selecting the row strip completing a marked row."""

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
from .shared.sampling import sample_line_completion_dataset
from .shared.state import DOMAIN, NonogramDataset, SCENE_ID

TASK_ID = "task_puzzles__nonogram__line_completion_label"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
PROMPT_TASK_KEY = "line_completion_label_query"
PROMPT_QUERY_KEY = "line_completion_label"
_NAMESPACE_BASE = f"{DOMAIN}.{SCENE_ID}.line_completion_label"
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS_UNUSED = (
    load_scene_generation_rendering_prompt_defaults(DOMAIN, SCENE_ID, task_id=TASK_ID)
)


@register_task
class PuzzlesNonogramLineCompletionLabelTask:
    """Select the visual strip option that completes the marked row."""

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
        """Generate one row-completion task with scalar bbox annotation."""

        return retry_nonogram_generation(
            build_case=_build_line_completion_case,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
        )


def _build_line_completion_case(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
) -> TaskOutput:
    """Sample a marked row, bind the correct option label, and annotate it."""

    return build_nonogram_option_label_case(
        task_id=TASK_ID,
        namespace=_NAMESPACE_BASE,
        instance_seed=int(instance_seed),
        params=params,
        generation_defaults=_GEN_DEFAULTS,
        rendering_defaults=_RENDER_DEFAULTS,
        prompt_task_key=PROMPT_TASK_KEY,
        prompt_query_key=PROMPT_QUERY_KEY,
        dataset_builder=sample_line_completion_dataset,
        prompt_dynamic_slots=_line_prompt_slots,
        validate_dataset=_validate_line_completion_dataset,
        relation_fields=_line_relation_fields,
    )


def _line_prompt_slots(dataset: NonogramDataset) -> Mapping[str, Any]:
    """Bind the marked row phrase used by the task prompt template."""

    return {"line_label": f"row {int(dataset.marked_index or 0) + 1}"}


def _line_relation_fields(dataset: NonogramDataset) -> Mapping[str, Any]:
    """Record the marked row relation without exposing task routing state."""

    return {
        "marked_axis": "row",
        "marked_index": int(dataset.marked_index or 0),
    }


def _validate_line_completion_dataset(dataset: NonogramDataset) -> None:
    """Validate the answer label matches exactly one marked correct option."""

    correct = [
        str(option["option_label"])
        for option in dataset.option_specs
        if bool(option.get("is_correct", False))
    ]
    if correct != [str(dataset.answer_value)]:
        raise ValueError("nonogram line-completion answer drifted from option specs")


__all__ = [
    "PROMPT_QUERY_KEY",
    "PuzzlesNonogramLineCompletionLabelTask",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
]
