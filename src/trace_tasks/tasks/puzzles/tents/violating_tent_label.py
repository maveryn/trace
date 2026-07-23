"""Public Tents task for selecting the labeled tent that breaks a rule."""

from __future__ import annotations

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.registry import register_task

from ._lifecycle import (
    TentsObjectiveBinding,
    TentsSceneTask,
)
from .shared.annotations import labeled_tent_bbox_annotation
from .shared.sampling import sample_violating_tent_board
from .shared.state import DOMAIN, SCENE_ID, TentsSample

TASK_ID = "task_puzzles__tents__violating_tent_label"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
PROMPT_TASK_KEY = "violating_tent_label_query"
PROMPT_QUERY_KEY = "violating_tent_label"
_NAMESPACE_BASE = f"{DOMAIN}.{SCENE_ID}.violating_tent_label"


def _sample_violating_tent(params, instance_seed, generation_defaults, rng):
    """Construct a board with exactly one labeled tent lacking an adjacent tree."""

    return sample_violating_tent_board(
        params=params,
        instance_seed=int(instance_seed),
        generation_defaults=generation_defaults,
        rng=rng,
    )


def _bind_violating_tent_output(
    sample: TentsSample,
    visual,
    selected_query,
    branch_probabilities,
):
    """Validate the unique violating tent and bind its scalar bbox annotation."""

    _ = (str(selected_query), dict(branch_probabilities))
    correct_specs = [
        spec for spec in sample.labeled_tent_specs if bool(spec.is_correct)
    ]
    if len(correct_specs) != 1:
        raise ValueError("Tents violation sample must expose exactly one answer")
    answer_label = str(correct_specs[0].label)
    answer_cell = tuple(correct_specs[0].cell)
    annotation_gt, projected_annotation, witness_symbolic = (
        labeled_tent_bbox_annotation(
            visual["rendered_scene"].item_bbox_map,
            answer_label,
        )
    )
    return TentsObjectiveBinding(
        answer_gt=TypedValue(type="option_letter", value=str(answer_label)),
        annotation_gt=annotation_gt,
        projected_annotation=projected_annotation,
        witness_symbolic=witness_symbolic,
        semantic_params={
            "correct_tent_label": str(answer_label),
            "violation_type": str(correct_specs[0].violation_type),
            "answer_schema": "option_letter",
        },
        execution_fields={
            "correct_tent_label": str(answer_label),
            "correct_tent_cell": [int(value) for value in answer_cell],
            "violation_type": str(correct_specs[0].violation_type),
            "annotation_policy": "scalar_bbox_selected_labeled_tent_cell",
            "supporting_item_ids": [f"labeled_tent_{answer_label}"],
        },
    )


@register_task
class PuzzlesTentsViolatingTentLabelTask(TentsSceneTask):
    """Choose the labeled tent that is not orthogonally adjacent to any tree."""

    task_id = TASK_ID
    reasoning_operations = ('topology',)
    supported_query_ids = SUPPORTED_QUERY_IDS
    prompt_task_key = PROMPT_TASK_KEY
    prompt_query_key = PROMPT_QUERY_KEY
    namespace = _NAMESPACE_BASE
    sample_builder = staticmethod(_sample_violating_tent)
    output_binder = staticmethod(_bind_violating_tent_output)

    def generate(self, instance_seed, *, params, max_attempts):
        """Generate one labeled-tent violation case."""

        output = super().generate(
            instance_seed,
            params=params,
            max_attempts=max_attempts,
        )
        return output
