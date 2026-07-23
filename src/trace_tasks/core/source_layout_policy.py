"""Public task-id parsing and source-layout detection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


_V0_TASK_ID_PATTERN = re.compile(
    r"^task_(?P<domain>[a-z0-9_]+)__(?P<scene_id>[a-z0-9_]+)__(?P<objective_contract>[a-z0-9_]+)$"
)
_TASKS_ROOT = Path(__file__).resolve().parents[1] / "tasks"


@dataclass(frozen=True)
class PublicTaskIdParts:
    """Parsed taxonomy-v0 public task-id components."""

    domain: str
    scene_id: str
    objective_contract: str


def parse_public_task_id(task_id: str) -> PublicTaskIdParts:
    """Parse ``task_<domain>__<scene_id>__<objective_contract>``."""

    match = _V0_TASK_ID_PATTERN.fullmatch(str(task_id))
    if match is None:
        raise ValueError(
            "task_id must follow taxonomy-v0 public form "
            "'task_<domain>__<scene_id>__<objective_contract>' "
            f"(got: {task_id})"
        )
    return PublicTaskIdParts(
        domain=str(match.group("domain")),
        scene_id=str(match.group("scene_id")),
        objective_contract=str(match.group("objective_contract")),
    )


def uses_current_source_layout(task_id: str, *, domain: str | None = None) -> bool:
    """Return whether a task has a direct taxonomy-v0 module in this package."""

    try:
        parts = parse_public_task_id(str(task_id))
    except ValueError:
        return False
    if domain is not None and str(domain) != parts.domain:
        return False

    module_path = (
        _TASKS_ROOT
        / parts.domain
        / parts.scene_id
        / f"{parts.objective_contract}.py"
    )
    return module_path.is_file()


__all__ = ["PublicTaskIdParts", "parse_public_task_id", "uses_current_source_layout"]
