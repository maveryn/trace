"""Contracts for the public taxonomy-v0 source layout."""

from __future__ import annotations

import inspect
from pathlib import Path

from trace_tasks.core.source_layout_policy import (
    parse_public_task_id,
    uses_current_source_layout,
)
from trace_tasks.core.taxonomy import resolve_task_taxonomy
from trace_tasks.tasks.registry import TASK_REGISTRY, ensure_all_tasks_registered


TASKS_ROOT = Path(__file__).resolve().parents[1] / "src" / "trace_tasks" / "tasks"


def _registered_tasks() -> list[tuple[str, type]]:
    ensure_all_tasks_registered()
    return sorted(
        ((str(task_id), task_cls) for task_id, task_cls in dict.items(TASK_REGISTRY)),
        key=lambda item: item[0],
    )


def test_public_tasks_use_direct_taxonomy_v0_modules() -> None:
    tasks = _registered_tasks()
    assert tasks

    failures: list[str] = []
    for task_id, task_cls in tasks:
        parts = parse_public_task_id(task_id)
        expected_module = (
            f"trace_tasks.tasks.{parts.domain}.{parts.scene_id}."
            f"{parts.objective_contract}"
        )
        expected_path = (
            TASKS_ROOT
            / parts.domain
            / parts.scene_id
            / f"{parts.objective_contract}.py"
        ).resolve()
        actual_path = Path(inspect.getsourcefile(task_cls) or "").resolve()
        if task_cls.__module__ != expected_module or actual_path != expected_path:
            failures.append(
                f"{task_id}: module={task_cls.__module__}, source={actual_path}"
            )
        if not uses_current_source_layout(task_id, domain=parts.domain):
            failures.append(f"{task_id}: current source layout was not detected")

    assert failures == []


def test_public_task_classes_match_id_taxonomy() -> None:
    failures: list[str] = []
    for task_id, task_cls in _registered_tasks():
        parts = parse_public_task_id(task_id)
        taxonomy = resolve_task_taxonomy(task_id)
        if str(getattr(task_cls, "domain", "")) != parts.domain:
            failures.append(f"{task_id}: class domain does not match task id")
        if taxonomy.domain != parts.domain or taxonomy.scene_id != parts.scene_id:
            failures.append(f"{task_id}: resolved taxonomy does not match task id")
        if getattr(task_cls, "scene_id", None) is not None:
            failures.append(f"{task_id}: class must not duplicate scene_id")

    assert failures == []
