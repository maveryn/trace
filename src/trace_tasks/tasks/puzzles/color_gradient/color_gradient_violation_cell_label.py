"""Public color-gradient task for identifying the violating swatch cell."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task

from ._lifecycle import (
    build_color_gradient_label_task_output,
    run_color_gradient_case,
)
from .shared.rendering import render_violation_scene
from .shared.sampling import build_violation_dataset
from .shared.state import DOMAIN, SCENE_ID

TASK_ID = "task_puzzles__color_gradient__color_gradient_violation_cell_label"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
PROMPT_TASK_KEY = "color_gradient_violation_query"
PROMPT_QUERY_KEY = "color_gradient_violation_cell_label"
_NAMESPACE_BASE = f"{DOMAIN}.{SCENE_ID}.color_gradient_violation_cell_label"


@register_task
class PuzzlesColorGradientViolationCellLabelTask:
    """Identify the labeled swatch that breaks a smooth color progression."""

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
        """Generate one scalar-bbox swatch-violation task instance."""

        return run_color_gradient_case(
            task_id=TASK_ID,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            namespace=_NAMESPACE_BASE,
            params=params,
            instance_seed=int(instance_seed),
            max_attempts=int(max_attempts),
            build_output=_build_violation_output,
        )


def _build_violation_output(
    *,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    prompt_defaults: Mapping[str, Any],
    instance_seed: int,
    public_branch: str,
    branch_probabilities: Mapping[str, float],
    attempt_limit: int,
) -> TaskOutput:
    """Sample, render, and bind one violating-swatch output."""

    return build_color_gradient_label_task_output(
        params=params,
        generation_defaults=generation_defaults,
        rendering_defaults=rendering_defaults,
        prompt_defaults=prompt_defaults,
        instance_seed=int(instance_seed),
        public_branch=str(public_branch),
        branch_probabilities=branch_probabilities,
        attempt_limit=int(attempt_limit),
        namespace=_NAMESPACE_BASE,
        prompt_task_key=PROMPT_TASK_KEY,
        prompt_query_key=PROMPT_QUERY_KEY,
        completion_prompt=False,
        label_scope="color_gradient_swatch_labels",
        sample_dataset=build_violation_dataset,
        render_scene=render_violation_scene,
        answer_label_field="answer_label",
        annotation_item_field="violation_cell_id",
        query_field_names=(
            "grid_size_variant",
            "grid_size_variant_probabilities",
            "rule_variant",
            "rule_variant_probabilities",
            "answer_label_probabilities",
            "rows",
            "cols",
            "answer_label",
        ),
        relation_field_names=(
            "grid_size_variant",
            "rule_variant",
            "answer_label",
        ),
        execution_field_names=(
            "cells",
            "rule_params",
            "violation_cell_id",
            "violation_index",
            "borrowed_from_label",
        ),
        support_item_field="violation_cell_id",
    )


__all__ = [
    "PROMPT_QUERY_KEY",
    "PROMPT_TASK_KEY",
    "PuzzlesColorGradientViolationCellLabelTask",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
]
