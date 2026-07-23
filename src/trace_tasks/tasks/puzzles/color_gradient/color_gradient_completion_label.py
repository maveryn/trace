"""Public color-gradient task for choosing the missing swatch option."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task

from ._lifecycle import (
    build_color_gradient_label_task_output,
    run_color_gradient_case,
)
from .shared.rendering import render_completion_scene
from .shared.sampling import build_completion_dataset
from .shared.state import DOMAIN, SCENE_ID

TASK_ID = "task_puzzles__color_gradient__color_gradient_completion_label"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
PROMPT_TASK_KEY = "color_gradient_completion_query"
PROMPT_QUERY_KEY = "linear_gradient_completion_label"
_NAMESPACE_BASE = f"{DOMAIN}.{SCENE_ID}.color_gradient_completion_label"


@register_task
class PuzzlesColorGradientCompletionLabelTask:
    """Choose the labeled option that completes a one-dimensional gradient."""

    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation', 'matching')
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
        """Generate one scalar-bbox color-gradient completion task."""

        return run_color_gradient_case(
            task_id=TASK_ID,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            namespace=_NAMESPACE_BASE,
            params=params,
            instance_seed=int(instance_seed),
            max_attempts=int(max_attempts),
            build_output=_build_completion_output,
        )


def _build_completion_output(
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
    """Sample, render, and bind one missing-swatch output."""

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
        completion_prompt=True,
        label_scope="color_gradient_sequence_and_option_labels",
        sample_dataset=build_completion_dataset,
        render_scene=render_completion_scene,
        answer_label_field="answer_label",
        annotation_item_field="correct_option_id",
        query_field_names=(
            "sequence_length_variant",
            "sequence_length_variant_probabilities",
            "option_count_variant",
            "option_count_variant_probabilities",
            "rule_variant",
            "rule_variant_probabilities",
            "missing_index_probabilities",
            "answer_label_probabilities",
            "sequence_length",
            "option_count",
            "missing_index",
            "answer_label",
        ),
        relation_field_names=(
            "sequence_length_variant",
            "option_count_variant",
            "rule_variant",
            "answer_label",
        ),
        execution_field_names=(
            "cells",
            "options",
            "rule_params",
            "missing_cell_id",
            "correct_option_id",
        ),
        support_item_field="correct_option_id",
        context_item_field="missing_cell_id",
    )


__all__ = [
    "PROMPT_QUERY_KEY",
    "PROMPT_TASK_KEY",
    "PuzzlesColorGradientCompletionLabelTask",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
]
