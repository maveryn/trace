"""Prompt assembly helpers for dots-and-boxes scene tasks."""

from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Tuple

from trace_tasks.tasks.shared.config_defaults import (
    load_scene_generation_rendering_prompt_defaults,
    required_group_defaults,
)
from trace_tasks.tasks.shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)

from .defaults import SCENE_ID

_GEN_DEFAULTS_UNUSED, _RENDER_DEFAULTS_UNUSED, _PROMPT_DEFAULTS = (
    load_scene_generation_rendering_prompt_defaults(
        "games",
        SCENE_ID,
    )
)


def build_dots_and_boxes_prompt_json_examples(
    *,
    annotation_example_shape: str,
    answer_example: int | str = 2,
) -> Tuple[str, str]:
    """Return JSON examples matching the active dots-and-boxes annotation shape."""

    if str(annotation_example_shape) == "segment_set":
        annotation_value = [
            [[180, 220], [300, 220]],
            [[310, 340], [430, 340]],
        ]
    elif str(annotation_example_shape) == "bbox":
        annotation_value = [180, 220, 300, 340]
    else:
        annotation_value = [
            [180, 220, 300, 340],
            [310, 220, 430, 340],
        ]
    return (
        json.dumps(
            {"annotation": annotation_value, "answer": answer_example},
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ),
        json.dumps(
            {"answer": answer_example},
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ),
    )


def build_dots_and_boxes_prompt_artifacts(
    *,
    domain: str,
    scene_variant: str,
    prompt_query_key: str,
    annotation_example_shape: str,
    answer_example: int | str = 2,
    dynamic_slots: Mapping[str, Any] | None = None,
    instance_seed: int,
) -> tuple[Dict[str, Any], Any]:
    """Build prompt artifacts for one objective-owned dots-and-boxes task file."""

    prompt_defaults = required_group_defaults(
        _PROMPT_DEFAULTS,
        (
            "bundle_id",
            "scene_key",
            "task_key",
            "json_output_contract",
            "json_output_contract_answer_only",
            "object_description_single_board",
            f"answer_hint_{str(prompt_query_key)}",
            f"annotation_hint_{str(prompt_query_key)}",
        ),
        context="dots-and-boxes prompt wiring defaults",
    )
    json_example, json_example_answer_only = build_dots_and_boxes_prompt_json_examples(
        annotation_example_shape=str(annotation_example_shape),
        answer_example=answer_example,
    )
    format_slots = {str(key): value for key, value in dict(dynamic_slots or {}).items()}
    answer_hint = str(
        prompt_defaults[f"answer_hint_{str(prompt_query_key)}"]
    ).format_map(format_slots)
    annotation_hint = str(
        prompt_defaults[f"annotation_hint_{str(prompt_query_key)}"]
    ).format_map(format_slots)
    prompt_slots = {
        "object_description": str(
            prompt_defaults[f"object_description_{str(scene_variant)}"]
        ),
        "json_output_contract": str(prompt_defaults["json_output_contract"]),
        "json_output_contract_answer_only": str(
            prompt_defaults["json_output_contract_answer_only"]
        ),
        "answer_hint": str(answer_hint),
        "annotation_hint": str(annotation_hint),
        "json_example": str(json_example),
        "json_example_answer_only": str(json_example_answer_only),
        **format_slots,
    }
    prompt_selection = render_scene_prompt_variants(
        domain=str(domain),
        scene_id=SCENE_ID,
        bundle_id=str(prompt_defaults["bundle_id"]),
        scene_key=str(prompt_defaults["scene_key"]),
        task_key=str(prompt_defaults["task_key"]),
        query_key=str(prompt_query_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots=prompt_slots,
        instance_seed=int(instance_seed),
    )
    return dict(prompt_defaults), build_prompt_trace_artifacts(prompt_selection)


__all__ = [
    "build_dots_and_boxes_prompt_artifacts",
    "build_dots_and_boxes_prompt_json_examples",
]
