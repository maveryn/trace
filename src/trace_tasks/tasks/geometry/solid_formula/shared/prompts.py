"""Prompt helpers for solid-formula tasks."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.shared.config_defaults import required_group_defaults
from trace_tasks.tasks.shared.prompt_json_example import dump_prompt_json_examples
from trace_tasks.tasks.shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)

from .defaults import DOMAIN, SCENE_ID


def solid_formula_prompt_artifacts(
    *,
    prompt_defaults: Mapping[str, Any],
    prompt_key: str,
    answer: float,
    instance_seed: int,
):
    """Render prompt variants from external prompt assets."""

    defaults = required_group_defaults(
        prompt_defaults,
        (
            "bundle_id",
            "scene_key",
            "task_key",
        ),
        context="prompt defaults for solid_formula",
    )
    annotation_hint = (
        "set \"annotation\" to the pixel bounding box [x0,y0,x1,y1] around the "
        "compound solid shape, excluding numeric and dimension labels"
    )
    json_example, json_example_answer_only = dump_prompt_json_examples(
        annotation=[150, 70, 610, 500],
        answer=float(answer),
    )
    prompt_selection = render_scene_prompt_variants(
        domain=DOMAIN,
        scene_id=SCENE_ID,
        bundle_id=str(defaults["bundle_id"]),
        scene_key=str(defaults["scene_key"]),
        task_key=str(defaults["task_key"]),
        query_key=str(prompt_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots={
            "annotation_hint": str(annotation_hint),
            "json_example": str(json_example),
            "json_example_answer_only": str(json_example_answer_only),
        },
        instance_seed=int(instance_seed),
    )
    return dict(defaults), build_prompt_trace_artifacts(prompt_selection)


__all__ = ["solid_formula_prompt_artifacts"]
