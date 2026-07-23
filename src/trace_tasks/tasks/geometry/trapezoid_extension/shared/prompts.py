"""Prompt assembly for trapezoid-extension tasks."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from trace_tasks.tasks.shared.prompt_json_example import dump_prompt_json_examples
from trace_tasks.tasks.shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)

from .state import DOMAIN, PROMPT_BUNDLE_ID, SCENE_ID, SCENE_PROMPT_KEY


def _bbox_map_prompt_json_examples(annotation_keys: Sequence[str], answer: Any) -> tuple[str, str]:
    annotation: dict[str, list[int]] = {}
    for index, key in enumerate(annotation_keys):
        row = int(index) // 2
        col = int(index) % 2
        x0 = 92 + col * 170
        y0 = 112 + row * 92
        annotation[str(key)] = [x0, y0, x0 + 112, y0 + 58]
    return dump_prompt_json_examples(annotation=annotation, answer=answer)


def _scalar_prompt_json_examples(annotation_type: str, answer: Any) -> tuple[str, str]:
    if str(annotation_type) == "segment":
        return dump_prompt_json_examples(annotation=[[320, 180], [520, 180]], answer=answer)
    if str(annotation_type) == "bbox":
        return dump_prompt_json_examples(annotation=[120, 180, 620, 430], answer=answer)
    raise ValueError(f"unsupported trapezoid-extension annotation type: {annotation_type}")


def build_trapezoid_extension_prompt_artifacts(
    *,
    prompt_defaults: Mapping[str, Any],
    task_prompt_key: str,
    prompt_branch_key: str,
    annotation_roles: Sequence[str],
    annotation_type: str,
    answer_value: float,
    instance_seed: int,
):
    """Render v1 prompt variants for one trapezoid-extension objective."""

    annotation_keys = tuple(str(role) for role in annotation_roles)
    annotation_key_list = ", ".join(f'"{key}"' for key in annotation_keys)
    if str(annotation_type) in {"bbox", "segment"}:
        json_example, json_example_answer_only = _scalar_prompt_json_examples(
            str(annotation_type),
            round(float(answer_value), 1),
        )
        if str(annotation_type) == "segment":
            annotation_instruction = 'set "annotation" to [[x0,y0],[x1,y1]] for segment BE'
        else:
            annotation_instruction = 'set "annotation" to [x0,y0,x1,y1] around the original trapezoid'
    else:
        json_example, json_example_answer_only = _bbox_map_prompt_json_examples(
            annotation_keys,
            round(float(answer_value), 1),
        )
        annotation_instruction = (
            "set \"annotation\" to a JSON object with exactly these visible region keys: "
            f"{annotation_key_list}; each value must be the pixel bounding box [x0,y0,x1,y1] around that region"
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


__all__ = ["build_trapezoid_extension_prompt_artifacts"]
