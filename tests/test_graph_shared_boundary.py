"""Regression checks for graph-domain shared helper boundaries."""

from __future__ import annotations

from pathlib import Path


def test_graph_scenes_do_not_import_graph_sampling_facade() -> None:
    """Graph scenes should use scene-local or narrow shared modules, not the old facade."""

    graph_root = Path("src/trace_tasks/tasks/graph")
    facade_import_patterns = (
        "trace_tasks.tasks.graph.shared.graph_sampling",
        "from ..shared.graph_sampling",
        "from ...shared.graph_sampling",
        "from .graph_sampling",
    )
    offenders: list[str] = []
    for path in graph_root.rglob("*.py"):
        relative = path.relative_to(graph_root)
        text = path.read_text(encoding="utf-8")
        if any(pattern in text for pattern in facade_import_patterns):
            offenders.append(str(path))

    assert offenders == []
