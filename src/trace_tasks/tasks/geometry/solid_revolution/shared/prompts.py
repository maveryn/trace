"""Prompt helpers for solid-revolution tasks."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from trace_tasks.tasks.shared.config_defaults import required_group_defaults
from trace_tasks.tasks.shared.prompt_json_example import dump_prompt_json_examples
from trace_tasks.tasks.shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)

from .defaults import DOMAIN, SCENE_ID


def _example_bbox_for_key(key: str) -> list[int]:
    examples = {
        "generating_shape": [80, 132, 365, 440],
        "rotation_axis": [184, 98, 208, 472],
        "solid_preview": [520, 145, 756, 442],
        "source_diagram_bbox": [70, 70, 390, 500],
        "resulting_solid_bbox": [510, 120, 760, 455],
    }
    return list(examples.get(str(key), [120, 120, 180, 160]))


def solid_revolution_prompt_artifacts(
    *,
    prompt_defaults: Mapping[str, Any],
    prompt_key: str,
    annotation_keys: Sequence[str],
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
        context="prompt defaults for solid_revolution",
    )
    annotation_names = ", ".join(f'"{key}"' for key in annotation_keys)
    annotation_hint = (
        "set \"annotation\" to a JSON object whose values are pixel bounding boxes "
        f"[x0,y0,x1,y1] for the required visible witnesses. Required keys: {annotation_names}"
    )
    annotation = {str(key): _example_bbox_for_key(str(key)) for key in annotation_keys}
    json_example, json_example_answer_only = dump_prompt_json_examples(
        annotation=annotation,
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


__all__ = ["solid_revolution_prompt_artifacts"]
