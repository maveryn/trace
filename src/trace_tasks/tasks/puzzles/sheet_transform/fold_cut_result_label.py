
from __future__ import annotations

from typing import Any, Dict, Mapping

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import (
    load_scene_generation_rendering_prompt_defaults,
)

from ._lifecycle import (
    build_fold_cut_result_label_case,
    retry_paper_fold_cut_generation,
)
from .shared.constraints import single_correct_option
from .shared.state import DOMAIN, SCENE_ID, PaperFoldCutDataset

TASK_ID = "task_puzzles__sheet_transform__fold_cut_result_label"
_NAMESPACE_BASE = f"{DOMAIN}.{SCENE_ID}.fold_cut_result_label"
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS_UNUSED = (
    load_scene_generation_rendering_prompt_defaults(DOMAIN, SCENE_ID, task_id=TASK_ID)
)


@register_task
class PuzzlesSheetTransformFoldCutResultLabelTask:

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

        return retry_paper_fold_cut_generation(
            build_case=_build_paper_fold_cut_result_case,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
        )


def _build_paper_fold_cut_result_case(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
) -> TaskOutput:

    return build_fold_cut_result_label_case(
        namespace=_NAMESPACE_BASE,
        instance_seed=int(instance_seed),
        params=params,
        generation_defaults=_GEN_DEFAULTS,
        rendering_defaults=_RENDER_DEFAULTS,
        prompt_task_key="fold_cut_result_label_query",
        prompt_query_key="fold_cut_result_label",
        validate_dataset=_validate_paper_fold_cut_dataset,
    )


def _validate_paper_fold_cut_dataset(dataset: PaperFoldCutDataset) -> None:

    winning_option = single_correct_option(
        dataset.option_specs,
        expected_label=str(dataset.answer_option_label),
        expected_choice_id=str(dataset.correct_option_choice_id),
        context="fold-cut",
    )
    expected_cells = sorted((int(x), int(y)) for x, y in dataset.unfolded_hole_cells)
    actual_cells = sorted(
        (int(cell[0]), int(cell[1])) for cell in winning_option.get("cells", ())
    )
    if actual_cells != expected_cells:
        raise ValueError("fold-cut correct option does not match unfolded holes")
    option_signatures = {
        tuple(sorted((int(cell[0]), int(cell[1])) for cell in option["cells"]))
        for option in dataset.option_specs
    }
    if len(option_signatures) != len(dataset.option_specs):
        raise ValueError("fold-cut options are not unique")
