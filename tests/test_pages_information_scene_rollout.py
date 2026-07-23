"""Pages-domain shared information-scene style rollout tests."""

from __future__ import annotations

from pathlib import Path

from trace_tasks.tasks.pages.hierarchy.subtree_descendant_count import (
    PagesHierarchySubtreeDescendantCountTask,
)
from trace_tasks.tasks.pages.timeline.interval_membership_count import (
    PagesTimelineIntervalMembershipCountTask,
)
from trace_tasks.tasks.registry import TASK_REGISTRY, ensure_task_registered


def _active_pages_task_ids() -> list[str]:
    return [path.stem for path in sorted(Path("docs/tasks/pages").glob("*/*.md"))]


def test_all_active_pages_tasks_honor_forced_information_scene_treatment() -> None:
    task_ids = _active_pages_task_ids()

    assert task_ids
    for index, task_id in enumerate(task_ids):
        ensure_task_registered(str(task_id))
        task = TASK_REGISTRY[str(task_id)]()
        output = task.generate(
            882000 + int(index),
            params={
                "information_scene_treatments": ["dark_report_card"],
                "pages_context_text_enabled": False,
            },
            max_attempts=40,
        )
        render_spec = output.trace_payload["render_spec"]
        execution = output.trace_payload["execution_trace"]

        assert render_spec["information_scene_style"]["kind"] == "information_scene_style"
        assert render_spec["information_scene_style"]["treatment"] == "dark_report_card"
        assert execution["information_scene_treatment"] == "dark_report_card"


def test_timeline_hierarchy_report_paragraph_context_uses_information_scene_colors() -> None:
    tasks = (
        PagesTimelineIntervalMembershipCountTask(),
        PagesHierarchySubtreeDescendantCountTask(),
    )

    for index, task in enumerate(tasks):
        output = task.generate(
            883000 + int(index),
            params={
                "information_scene_treatments": ["dark_report_card"],
                "pages_context_mode": "paragraph_box",
                "pages_context_paragraph_box_count": 1,
            },
            max_attempts=40,
        )
        render_spec = output.trace_payload["render_spec"]
        context_layer = render_spec["context_text_layer"]
        layout_spec = context_layer["layout_spec"]

        assert render_spec["information_scene_style"]["treatment"] == "dark_report_card"
        assert layout_spec["context_profile"] == "report_paragraph"
        assert layout_spec["mode"] == "paragraph_box"
        assert layout_spec["paragraph_box_count"] >= 1
        assert layout_spec["context_color_source"] == "information_scene_style"
