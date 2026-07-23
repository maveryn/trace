"""Select the labeled matchstick that repairs an equation when removed."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults

from ._lifecycle import (
    BoundMatchstickOutput,
    MatchstickRenderContext,
    build_bound_output,
    run_matchstick_public_task,
)
from .shared.annotations import segment_artifacts
from .shared.rendering import render_equation_repair_scene
from .shared.rules import equation_text
from .shared.sampling import build_equation_repair_dataset
from .shared.state import DOMAIN, EquationRepairDataset, SCENE_ID


TASK_ID = "task_puzzles__matchstick__equation_repair_stick_label"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
TASK_PROMPT_KEY = "equation_repair_stick_label_query"

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
) -> EquationRepairDataset:
    """Build the false-equation instance for the single repair-stick query."""

    if str(query_id) != SINGLE_QUERY_ID:
        raise ValueError(f"unsupported matchstick equation query: {query_id}")
    return build_equation_repair_dataset(
        scene_variant=str(scene_variant),
        params=params,
        generation_defaults=generation_defaults,
        namespace=str(namespace),
        instance_seed=int(instance_seed),
    )


def _bind_output(
    *,
    dataset: EquationRepairDataset,
    context: MatchstickRenderContext,
    query_id: str,
    query_probabilities: Mapping[str, float],
    scene_variant_probabilities: Mapping[str, float],
) -> BoundMatchstickOutput:
    """Bind the selected labeled stick as one scalar segment annotation."""

    selected_item_id = f"stick_{dataset.answer_label}"
    annotation = segment_artifacts(
        context.rendered_scene.item_segment_map,
        selected_item_id,
    )
    source_equation = equation_text(
        tuple(int(value) for value in dataset.source_digits),
        str(dataset.operator),
    )
    repaired_equation = equation_text(
        tuple(int(value) for value in dataset.repaired_digits),
        str(dataset.operator),
    )
    option_specs = [
        {
            "option_label": str(option.label),
            "source_stick_id": str(option.value),
            "is_correct": bool(option.is_correct),
        }
        for option in dataset.option_specs
    ]
    return build_bound_output(
        dataset=dataset,
        query_id=str(query_id),
        query_probabilities=query_probabilities,
        scene_variant_probabilities=scene_variant_probabilities,
        prompt_query_key=SINGLE_QUERY_ID,
        answer_gt=TypedValue(type="option_letter", value=str(dataset.answer_label)),
        annotation_artifacts=annotation,
        annotation_source="item_segments_px",
        scene_extra={
            "source_equation": source_equation,
            "repaired_equation": repaired_equation,
            "operator": str(dataset.operator),
        },
        execution_extra={
            "source_equation": source_equation,
            "repaired_equation": repaired_equation,
            "operator": str(dataset.operator),
            "source_digits": [int(value) for value in dataset.source_digits],
            "repaired_digits": [int(value) for value in dataset.repaired_digits],
            "repair_digit_index": int(dataset.repair_digit_index),
            "repair_segment_key": str(dataset.repair_segment_key),
            "repair_stick_id": str(dataset.repair_stick_id),
            "supporting_item_ids": [selected_item_id],
            "selected_item_id": selected_item_id,
            "option_specs": option_specs,
            "all_removal_outcomes": [dict(row) for row in dataset.all_removal_outcomes],
        },
        witness_symbolic={
            "type": "segment",
            "value": [list(point) for point in annotation.value],
        },
    )


@register_task
class PuzzlesMatchstickEquationRepairStickLabelTask:
    """Choose the labeled stick whose removal makes the equation true."""

    task_id = TASK_ID
    reasoning_operations = ('state_update', 'formula_evaluation', 'matching')
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
        """Run the equation-repair callbacks through shared scene plumbing."""

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
            render_scene=render_equation_repair_scene,
            output_binder=_bind_output,
        )


__all__ = [
    "PuzzlesMatchstickEquationRepairStickLabelTask",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
]
