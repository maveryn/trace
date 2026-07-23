"""Private scene-output assembly for spring public tasks."""

from __future__ import annotations

from typing import Any

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.shared.output_metadata import default_task_versions

from .shared.mechanics import SpringTaskParts


def assemble_spring_result(
    *,
    parts: SpringTaskParts,
    scene_name: str,
    annotation_value: Any,
) -> TaskOutput:
    """Bind public answer/annotation values and return the final task output."""

    fields = parts.base_fields()
    fields["answer_gt"] = TypedValue(type="integer", value=int(parts.answer_value))
    fields["annotation_gt"] = TypedValue(
        type=str(parts.annotation_type),
        value=annotation_value,
    )
    fields["task_versions"] = default_task_versions()
    fields["query_id"] = str(parts.public_branch)
    fields["scene_id"] = str(scene_name)
    return TaskOutput(**fields)


__all__ = ["assemble_spring_result"]
