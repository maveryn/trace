"""Public toggle-grid task for selecting the result grid."""

from __future__ import annotations

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.registry import register_task

from ._lifecycle import ToggleBinding, ToggleGridSceneTask
from .shared.annotations import option_panel_bbox_annotation
from .shared.rendering import render_toggle_result_scene
from .shared.sampling import sample_result_dataset
from .shared.state import DOMAIN, SCENE_ID, ToggleDataset

TASK_ID = "task_puzzles__toggle_grid__toggle_result_label"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
PROMPT_TASK_KEY = "toggle_result_label_query"
PROMPT_QUERY_KEY = "toggle_result_label"
_NAMESPACE_BASE = f"{DOMAIN}.{SCENE_ID}.toggle_result_label"


def _sample_result(params, generation_defaults, rng):
    """Construct one red-marked single-press result-option dataset."""

    return sample_result_dataset(
        params=params,
        generation_defaults=generation_defaults,
        rng=rng,
    )


def _bind_result_output(
    dataset: ToggleDataset,
    visual,
    selected_query,
    branch_probabilities,
):
    """Bind the correct option label and selected option-panel bbox."""

    _ = (str(selected_query), dict(branch_probabilities))
    correct = [
        option for option in dataset.result_options if bool(option.is_correct)
    ]
    if len(correct) != 1:
        raise ValueError("toggle result task requires exactly one correct option")
    answer_label = str(correct[0].option_label)
    if answer_label != str(dataset.correct_option_label):
        raise ValueError("toggle result answer drifted from dataset label")
    annotation_gt, projected_annotation, witness_symbolic = (
        option_panel_bbox_annotation(
            visual["rendered_scene"].option_panel_bbox_map,
            answer_label,
        )
    )
    return ToggleBinding(
        answer_gt=TypedValue(type="option_letter", value=str(answer_label)),
        annotation_gt=annotation_gt,
        projected_annotation=projected_annotation,
        witness_symbolic=witness_symbolic,
        semantic_params={"answer_schema": "option_letter"},
        execution_fields={
            "annotation_policy": "scalar_bbox_selected_result_option_panel",
            "supporting_item_ids": [f"option_{answer_label}"],
        },
    )


@register_task
class PuzzlesToggleGridToggleResultLabelTask(ToggleGridSceneTask):
    """Choose the result grid after pressing the red marked toggle switch."""

    task_id = TASK_ID
    reasoning_operations = ('state_update',)
    supported_query_ids = SUPPORTED_QUERY_IDS
    prompt_task_key = PROMPT_TASK_KEY
    prompt_query_key = PROMPT_QUERY_KEY
    namespace = _NAMESPACE_BASE
    sample_builder = staticmethod(_sample_result)
    render_builder = staticmethod(render_toggle_result_scene)
    output_binder = staticmethod(_bind_result_output)

    def generate(self, instance_seed, *, params, max_attempts):
        """Generate one toggle-result option-label case."""

        output = super().generate(
            instance_seed,
            params=params,
            max_attempts=max_attempts,
        )
        return output


__all__ = [
    "PROMPT_QUERY_KEY",
    "PuzzlesToggleGridToggleResultLabelTask",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
]
