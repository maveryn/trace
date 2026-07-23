"""Identity-free prompt slot helpers for special-quadrilateral tasks."""

from __future__ import annotations

from typing import Any, Sequence

from trace_tasks.tasks.shared.prompt_json_example import build_keyed_point_prompt_json_examples


def special_quadrilateral_prompt_slots(
    *,
    target_name: str,
    annotation_roles: Sequence[str],
    answer_value: int,
) -> dict[str, Any]:
    """Build reusable dynamic prompt slots for point-map annotation."""

    annotation_keys = tuple(str(role) for role in annotation_roles)
    json_example, json_example_answer_only = build_keyed_point_prompt_json_examples(
        annotation_keys=annotation_keys,
        answer=int(answer_value),
    )
    annotation_key_list = ", ".join(f'"{key}"' for key in annotation_keys)
    annotation_hint = (
        "set \"annotation\" to a JSON object with exactly these visible point-label keys: "
        f"{annotation_key_list}; each value must be the pixel point [x,y] at that labeled point"
    )
    return {
        "target_name": str(target_name),
        "annotation_hint": str(annotation_hint),
        "json_example": str(json_example),
        "json_example_answer_only": str(json_example_answer_only),
    }


__all__ = ["special_quadrilateral_prompt_slots"]
