"""Tests for canonical public task-id and module naming conventions."""

from __future__ import annotations

import re

from trace_tasks.core.taxonomy import lookup_task_taxonomy
from trace_tasks.tasks import TASK_REGISTRY


_TASK_ID_PATTERN = re.compile(
    r"^task_(?P<domain>[a-z0-9_]+)__(?P<scene>[a-z0-9_]+)__(?P<objective>[a-z0-9_]+)$"
)


def test_registered_task_ids_follow_public_taxonomy_v0_pattern() -> None:
    for task_id, task_cls in TASK_REGISTRY.items():
        task_id_text = str(task_id)
        match = _TASK_ID_PATTERN.match(task_id_text)
        assert match is not None, task_id

        domain = str(getattr(task_cls, "domain"))
        assert str(match.group("domain")) == domain, task_id
        assert str(match.group("scene")).strip(), task_id
        assert str(match.group("objective")).strip(), task_id
        assert lookup_task_taxonomy(task_id_text) is not None, task_id


def test_task_modules_live_under_registered_source_domain() -> None:
    for task_id, task_cls in TASK_REGISTRY.items():
        module_name = str(getattr(task_cls, "__module__", ""))
        if not module_name.startswith("trace_tasks.tasks."):
            continue
        domain = str(getattr(task_cls, "domain"))
        assert module_name.startswith(f"trace_tasks.tasks.{domain}."), task_id
