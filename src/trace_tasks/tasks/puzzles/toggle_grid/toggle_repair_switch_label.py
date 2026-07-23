"""Public toggle-grid task for selecting the one repair switch."""

from __future__ import annotations

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.registry import register_task

from ._lifecycle import ToggleBinding, ToggleGridSceneTask
from .shared.annotations import switch_cell_bbox_annotation
from .shared.rendering import render_toggle_repair_scene
from .shared.sampling import sample_repair_dataset
from .shared.state import DOMAIN, SCENE_ID, ToggleDataset

TASK_ID = "task_puzzles__toggle_grid__toggle_repair_switch_label"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
PROMPT_TASK_KEY = "toggle_repair_switch_label_query"
PROMPT_QUERY_KEY = "toggle_repair_switch_label"
_NAMESPACE_BASE = f"{DOMAIN}.{SCENE_ID}.toggle_repair_switch_label"


def _sample_repair(params, generation_defaults, rng):
    """Construct one single-switch repair dataset."""

    return sample_repair_dataset(
        params=params,
        generation_defaults=generation_defaults,
        rng=rng,
    )


def _bind_repair_output(
    dataset: ToggleDataset,
    visual,
    selected_query,
    branch_probabilities,
):
    """Bind the correct switch label and selected cell bbox."""

    _ = (str(selected_query), dict(branch_probabilities))
    correct = [option for option in dataset.switch_options if bool(option.is_correct)]
    if len(correct) != 1:
        raise ValueError("toggle repair task requires exactly one correct switch")
    selected = correct[0]
    answer_label = str(selected.option_label)
    if answer_label != str(dataset.correct_option_label):
        raise ValueError("toggle repair answer drifted from dataset label")
    annotation_gt, projected_annotation, witness_symbolic = (
        switch_cell_bbox_annotation(
            visual["rendered_scene"].start_cell_bbox_map,
            row=int(selected.row),
            col=int(selected.col),
            label=answer_label,
        )
    )
    return ToggleBinding(
        answer_gt=TypedValue(type="option_letter", value=str(answer_label)),
        annotation_gt=annotation_gt,
        projected_annotation=projected_annotation,
        witness_symbolic=witness_symbolic,
        semantic_params={"answer_schema": "option_letter"},
        execution_fields={
            "selected_switch_cell": [int(selected.row), int(selected.col)],
            "annotation_policy": "scalar_bbox_selected_switch_cell",
            "supporting_item_ids": [f"cell_{int(selected.row)}_{int(selected.col)}"],
        },
    )


@register_task
class PuzzlesToggleGridToggleRepairSwitchLabelTask(ToggleGridSceneTask):
    """Choose the switch press that transforms the start grid into the target grid."""

    task_id = TASK_ID
    reasoning_operations = ('state_update',)
    supported_query_ids = SUPPORTED_QUERY_IDS
    prompt_task_key = PROMPT_TASK_KEY
    prompt_query_key = PROMPT_QUERY_KEY
    namespace = _NAMESPACE_BASE
    sample_builder = staticmethod(_sample_repair)
    render_builder = staticmethod(render_toggle_repair_scene)
    output_binder = staticmethod(_bind_repair_output)

    def generate(self, instance_seed, *, params, max_attempts):
        """Generate one toggle-repair switch-label case."""

        output = super().generate(
            instance_seed,
            params=params,
            max_attempts=max_attempts,
        )
        return output


__all__ = [
    "PROMPT_QUERY_KEY",
    "PuzzlesToggleGridToggleRepairSwitchLabelTask",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
]
