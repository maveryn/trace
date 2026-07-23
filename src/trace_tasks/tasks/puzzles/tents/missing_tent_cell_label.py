"""Public Tents task for selecting the missing tent candidate cell."""

from __future__ import annotations

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.registry import register_task

from ._lifecycle import (
    TentsObjectiveBinding,
    TentsSceneTask,
)
from .shared.annotations import candidate_bbox_annotation
from .shared.sampling import sample_single_legal_cell_board
from .shared.state import DOMAIN, SCENE_ID, TentsSample

TASK_ID = "task_puzzles__tents__missing_tent_cell_label"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
PROMPT_TASK_KEY = "missing_tent_cell_label_query"
PROMPT_QUERY_KEY = "missing_tent_cell_label"
_NAMESPACE_BASE = f"{DOMAIN}.{SCENE_ID}.missing_tent_cell_label"


def _sample_missing_tent_cell(params, instance_seed, generation_defaults, rng):
    """Construct the objective sample with exactly one legal candidate cell."""

    return sample_single_legal_cell_board(
        params=params,
        instance_seed=int(instance_seed),
        generation_defaults=generation_defaults,
        rng=rng,
    )


def _bind_missing_tent_cell_output(
    sample: TentsSample,
    visual,
    selected_query,
    branch_probabilities,
):
    """Validate the unique label answer and bind its scalar bbox annotation."""

    _ = (str(selected_query), dict(branch_probabilities))
    correct_specs = [
        spec
        for spec in sample.candidate_specs
        if bool(spec.is_correct) and bool(spec.is_legal)
    ]
    if len(correct_specs) != 1:
        raise ValueError("Tents missing-cell sample must expose exactly one answer")
    answer_label = str(correct_specs[0].label)
    answer_cell = tuple(correct_specs[0].cell)
    annotation_gt, projected_annotation, witness_symbolic = candidate_bbox_annotation(
        visual["rendered_scene"].item_bbox_map,
        answer_label,
    )
    return TentsObjectiveBinding(
        answer_gt=TypedValue(type="option_letter", value=str(answer_label)),
        annotation_gt=annotation_gt,
        projected_annotation=projected_annotation,
        witness_symbolic=witness_symbolic,
        semantic_params={
            "correct_candidate_label": str(answer_label),
            "answer_schema": "option_letter",
        },
        execution_fields={
            "correct_candidate_label": str(answer_label),
            "correct_cell": [int(value) for value in answer_cell],
            "annotation_policy": "scalar_bbox_selected_candidate_cell",
            "supporting_item_ids": [f"candidate_{answer_label}"],
        },
    )


@register_task
class PuzzlesTentsMissingTentCellLabelTask(TentsSceneTask):
    """Choose the only candidate cell that can hold the marked tree's missing tent."""

    task_id = TASK_ID
    reasoning_operations = ('topology',)
    supported_query_ids = SUPPORTED_QUERY_IDS
    prompt_task_key = PROMPT_TASK_KEY
    prompt_query_key = PROMPT_QUERY_KEY
    namespace = _NAMESPACE_BASE
    sample_builder = staticmethod(_sample_missing_tent_cell)
    output_binder = staticmethod(_bind_missing_tent_cell_output)

    def generate(self, instance_seed, *, params, max_attempts):
        """Generate one missing-cell selection case."""

        output = super().generate(
            instance_seed,
            params=params,
            max_attempts=max_attempts,
        )
        return output
