"""Prompt rendering helpers for graph-paper scene tasks."""

from __future__ import annotations

from typing import Mapping

from trace_tasks.tasks.shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)

from .state import PromptPlan, SCENE_ID


def render_prompt_artifacts(*, plan: PromptPlan, instance_seed: int):
    """Render one v1 prompt bundle using task-owned dynamic slot values."""

    dynamic_slots = {
        "answer_hint": str(plan.answer_hint),
        "annotation_hint": str(plan.annotation_hint),
        "json_example": str(plan.json_example),
        "json_example_answer_only": str(plan.json_example_answer_only),
        "shape_text": str(plan.shape_text),
        "metric_text": str(plan.metric_text),
        "target_text": str(plan.target_text),
    }
    selection = render_scene_prompt_variants(
        domain="geometry",
        scene_id=SCENE_ID,
        bundle_id=str(plan.bundle_id),
        scene_key=str(plan.scene_key),
        task_key=str(plan.task_key),
        query_key=str(plan.prompt_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots=dynamic_slots,
        instance_seed=int(instance_seed),
    )
    return build_prompt_trace_artifacts(selection)


def prompt_defaults(prompt_defaults: Mapping[str, object]) -> tuple[str, str, str]:
    """Return the configured graph-paper prompt bundle keys."""

    return (
        str(prompt_defaults.get("bundle_id", "geometry_graph_paper_v1")),
        str(prompt_defaults.get("scene_key", "graph_paper_scene")),
        str(prompt_defaults.get("task_key", "graph_paper_task")),
    )
