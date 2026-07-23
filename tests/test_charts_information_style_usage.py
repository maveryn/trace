"""Regression checks for chart information-scene styling."""

from __future__ import annotations

from pathlib import Path


def test_chart_scene_renderers_do_not_use_background_only_helper() -> None:
    """Chart scene renderers should theme internal chrome, not only wrappers."""

    chart_root = Path("src/trace_tasks/tasks/charts")
    adapter_path = chart_root / "shared" / "information_style.py"
    offenders: list[str] = []
    for path in chart_root.rglob("*.py"):
        if path == adapter_path:
            continue
        text = path.read_text(encoding="utf-8")
        if "make_chart_background_canvas(" in text:
            offenders.append(str(path))

    assert offenders == []
