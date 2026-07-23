"""Prompt assembly for survey-traverse tasks."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from trace_tasks.tasks.shared.prompt_json_example import build_keyed_point_prompt_json_examples, dump_prompt_json_examples
from trace_tasks.tasks.shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)

from .defaults import PROMPT_BUNDLE_ID, SCENE_PROMPT_KEY
from .state import DOMAIN, SCENE_ID


def _bbox_prompt_json_examples(annotation_keys: Sequence[str], answer: Any) -> tuple[str, str]:
    """Build compact examples for role-bound bbox annotation contracts."""

    annotation = {}
    for index, key in enumerate(annotation_keys):
        row = int(index) // 3
        col = int(index) % 3
        x0 = int(80 + col * 140)
        y0 = int(90 + row * 90)
        annotation[str(key)] = [x0, y0, x0 + 80, y0 + 54]
    return dump_prompt_json_examples(annotation=annotation, answer=answer)


def build_survey_traverse_prompt_artifacts(
    *,
    prompt_defaults: Mapping[str, Any],
    task_prompt_key: str,
    prompt_branch_key: str,
    annotation_roles: Sequence[str],
    annotation_kind: str,
    answer_value: int,
    instance_seed: int,
):
    """Render v1 prompt variants after the public task selects its semantic branch."""

    annotation_keys = tuple(str(role) for role in annotation_roles)
    annotation_key_list = ", ".join(f'"{key}"' for key in annotation_keys)
    if str(annotation_kind) == "bbox_map":
        json_example, json_example_answer_only = _bbox_prompt_json_examples(annotation_keys, int(answer_value))
        annotation_instruction = (
            "set \"annotation\" to a JSON object with exactly these visible region keys: "
            f"{annotation_key_list}; each value must be the pixel bounding box [x0,y0,x1,y1] around that region"
        )
    else:
        json_example, json_example_answer_only = build_keyed_point_prompt_json_examples(
            annotation_keys=annotation_keys,
            answer=int(answer_value),
        )
        annotation_instruction = (
            "set \"annotation\" to a JSON object with exactly these visible point keys: "
            f"{annotation_key_list}; each value must be the pixel point [x,y] at that station, reference, line midpoint, or note center"
        )
    prompt_selection = render_scene_prompt_variants(
        domain=DOMAIN,
        scene_id=SCENE_ID,
        bundle_id=str(prompt_defaults.get("bundle_id", PROMPT_BUNDLE_ID)),
        scene_key=str(prompt_defaults.get("scene_key", SCENE_PROMPT_KEY)),
        task_key=str(task_prompt_key),
        query_key=str(prompt_branch_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots={
            "annotation_instruction": str(annotation_instruction),
            "annotation_key_list": str(annotation_key_list),
            "json_example": str(json_example),
            "json_example_answer_only": str(json_example_answer_only),
        },
        instance_seed=int(instance_seed),
    )
    return build_prompt_trace_artifacts(prompt_selection)


__all__ = ["build_survey_traverse_prompt_artifacts"]
