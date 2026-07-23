"""Prompt assembly helpers for Rhythm scene-package tasks."""

from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Tuple

from trace_tasks.tasks.shared.config_defaults import required_group_defaults
from trace_tasks.tasks.shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)

from .state import SCENE_ID


_PROMPT_WIRING_KEYS = (
    "bundle_id",
    "scene_key",
    "task_key",
    "json_output_contract",
    "json_output_contract_answer_only",
)


def rhythm_json_examples(
    *,
    annotation_type: str,
    example_annotation: Any | None = None,
    example_answer: int | None = None,
) -> Tuple[str, str]:
    """Return format-only JSON examples aligned to the task annotation shape."""

    if example_annotation is not None and example_answer is not None:
        annotation = example_annotation
        answer = int(example_answer)
    elif str(annotation_type) == "bbox":
        annotation: Any = [260, 432, 341, 479]
        answer = 4
    else:
        annotation = [[260, 498, 341, 545], [260, 432, 341, 479], [260, 302, 341, 349]]
        answer = 3
    return (
        json.dumps({"annotation": annotation, "answer": answer}, separators=(",", ":"), ensure_ascii=True),
        json.dumps({"answer": answer}, separators=(",", ":"), ensure_ascii=True),
    )


def build_rhythm_prompt_artifacts(
    *,
    domain: str,
    scene_variant: str,
    prompt_query_key: str,
    annotation_type: str,
    prompt_defaults: Mapping[str, Any],
    selected_lane_label: str,
    target_color: str,
    score_values_by_color: Mapping[str, int] | None,
    prompt_rule_keys: Tuple[str, ...],
    json_example_annotation: Any | None,
    json_example_answer: int | None,
    beat_window: int,
    instance_seed: int,
) -> tuple[Dict[str, Any], Any]:
    """Build Rhythm prompt artifacts from external prompt templates."""

    object_description_key = f"object_description_{str(scene_variant)}"
    answer_hint_key = f"answer_hint_{str(prompt_query_key)}"
    annotation_hint_key = f"annotation_hint_{str(prompt_query_key)}"
    rule_keys = tuple(str(key) for key in prompt_rule_keys)
    defaults = required_group_defaults(
        prompt_defaults,
        (
            *_PROMPT_WIRING_KEYS,
            object_description_key,
            *rule_keys,
            answer_hint_key,
            annotation_hint_key,
        ),
        context="rhythm prompt wiring defaults",
    )
    json_example, json_example_answer_only = rhythm_json_examples(
        annotation_type=str(annotation_type),
        example_annotation=json_example_annotation,
        example_answer=json_example_answer,
    )
    prompt_selection = render_scene_prompt_variants(
        domain=str(domain),
        scene_id=SCENE_ID,
        bundle_id=str(defaults["bundle_id"]),
        scene_key=str(defaults["scene_key"]),
        task_key=str(defaults["task_key"]),
        query_key=str(prompt_query_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots={
            "object_description": str(defaults[object_description_key]),
            "rhythm_motion_rule_text": str(defaults.get("rhythm_motion_rule_text", "")),
            "note_object_rule_text": str(defaults.get("note_object_rule_text", "")),
            "score_palette_rule_text": str(defaults.get("score_palette_rule_text", "")),
            "beat_window": str(int(beat_window)),
            "selected_lane_label": str(selected_lane_label),
            "target_color": str(target_color),
            "score_values_by_color": ""
            if score_values_by_color is None
            else ", ".join(
                f"{str(color)}={int(score_values_by_color[str(color)])}"
                for color in sorted(score_values_by_color)
            ),
            "json_output_contract": str(defaults["json_output_contract"]),
            "json_output_contract_answer_only": str(defaults["json_output_contract_answer_only"]),
            "answer_hint": str(defaults[answer_hint_key]),
            "annotation_hint": str(defaults[annotation_hint_key]),
            "json_example": str(json_example),
            "json_example_answer_only": str(json_example_answer_only),
        },
        instance_seed=int(instance_seed),
    )
    return dict(defaults), build_prompt_trace_artifacts(prompt_selection)


__all__ = ["build_rhythm_prompt_artifacts", "rhythm_json_examples"]
