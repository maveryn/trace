
from __future__ import annotations

from typing import Any, Dict, Mapping

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import (
    load_scene_generation_rendering_prompt_defaults,
)

from ._lifecycle import build_fold_projection_result_label_case, retry_paper_fold_generation
from .shared.constraints import single_correct_option
from .shared.state import DOMAIN, PaperFoldDataset, SCENE_ID

TASK_ID = "task_puzzles__sheet_transform__fold_projection_result_label"
_NAMESPACE_BASE = f"{DOMAIN}.{SCENE_ID}.fold_projection_result_label"
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS_UNUSED = (
    load_scene_generation_rendering_prompt_defaults(DOMAIN, SCENE_ID, task_id=TASK_ID)
)


@register_task
class PuzzlesSheetTransformFoldProjectionResultLabelTask:

    task_id = TASK_ID
    reasoning_operations = ('transformation',)
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

        return retry_paper_fold_generation(
            build_case=_build_paper_fold_result_case,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
        )


def _build_paper_fold_result_case(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
) -> TaskOutput:

    return build_fold_projection_result_label_case(
        namespace=_NAMESPACE_BASE,
        instance_seed=int(instance_seed),
        params=params,
        generation_defaults=_GEN_DEFAULTS,
        rendering_defaults=_RENDER_DEFAULTS,
        prompt_task_key="fold_projection_result_label_query",
        prompt_query_key="fold_projection_result_label",
        validate_dataset=_validate_paper_fold_dataset,
    )


def _validate_paper_fold_dataset(dataset: PaperFoldDataset) -> None:

    correct_option = single_correct_option(
        dataset.option_specs,
        expected_label=str(dataset.answer_option_label),
        expected_choice_id=str(dataset.correct_option_choice_id),
        context="fold-projection",
    )
    expected_signature = sorted(
        (
            str(mark["object_type"]),
            int(mark["cell"][0]),
            int(mark["cell"][1]),
        )
        for mark in dataset.folded_result_mark_specs
    )
    correct_signature = sorted(
        (
            str(mark["object_type"]),
            int(mark["cell"][0]),
            int(mark["cell"][1]),
        )
        for mark in correct_option.get("mark_specs", ())
    )
    if correct_signature != expected_signature:
        raise ValueError("fold-projection correct option does not match folded result")
