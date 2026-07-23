
from __future__ import annotations

from typing import Any, Dict, Mapping

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import (
    load_scene_generation_rendering_prompt_defaults,
)

from ._lifecycle import build_overlay_union_result_label_case, retry_overlay_generation
from .shared.rules import sample_overlay_dataset
from .shared.constraints import single_correct_option
from .shared.state import DOMAIN, OverlayDataset, SCENE_ID

TASK_ID = "task_puzzles__sheet_transform__overlay_union_result_label"
_NAMESPACE_BASE = f"{DOMAIN}.{SCENE_ID}.overlay_union_result_label"
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS_UNUSED = (
    load_scene_generation_rendering_prompt_defaults(DOMAIN, SCENE_ID, task_id=TASK_ID)
)


@register_task
class PuzzlesSheetTransformOverlayUnionResultLabelTask:

    task_id = TASK_ID
    reasoning_operations = ('logical_composition', 'transformation')
    domain = DOMAIN
    default_dataset_enabled = True
    supported_query_ids = (SINGLE_QUERY_ID,)

    def generate(
        self,
        instance_seed: int,
        *,
        params: Dict[str, Any],
        max_attempts: int,
    ) -> TaskOutput:

        return retry_overlay_generation(
            build_case=_build_overlay_result_case,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
        )


def _build_overlay_result_case(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
) -> TaskOutput:

    return build_overlay_union_result_label_case(
        namespace=_NAMESPACE_BASE,
        instance_seed=int(instance_seed),
        params=params,
        generation_defaults=_GEN_DEFAULTS,
        rendering_defaults=_RENDER_DEFAULTS,
        prompt_task_key="overlay_union_result_label_query",
        prompt_query_key="overlay_union_result_label",
        dataset_builder=sample_overlay_dataset,
        validate_dataset=_validate_overlay_dataset,
    )


def _validate_overlay_dataset(dataset: OverlayDataset) -> None:

    correct_option = single_correct_option(
        dataset.option_specs,
        expected_label=str(dataset.answer_option_label),
        expected_choice_id=str(dataset.correct_option_choice_id),
        context="overlay-union",
    )
    union = sorted(tuple(int(value) for value in cell) for cell in dataset.union_cells)
    for option in dataset.option_specs:
        option_cells = sorted(
            tuple(int(value) for value in cell)
            for cell in option.get("cells", ())
        )
        if option is correct_option and option_cells != union:
            raise ValueError("overlay-union correct option does not equal source union")
        if (not bool(option.get("is_correct", False))) and option_cells == union:
            raise ValueError("overlay-union distractor duplicates the source union")
