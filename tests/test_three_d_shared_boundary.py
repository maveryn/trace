"""Focused three_d shared-boundary checks."""

from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _function_names(source_path: Path) -> set[str]:
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    return {str(node.name) for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}


def test_object_scene_room_renderer_is_scene_local() -> None:
    shared_rendering = REPO_ROOT / "trace" / "tasks" / "three_d" / "shared" / "object_scene_rendering.py"
    scene_rendering = REPO_ROOT / "trace" / "tasks" / "three_d" / "object_scene" / "shared" / "rendering.py"

    assert "_draw_room" not in _function_names(shared_rendering)
    assert "draw_object_scene_room" in _function_names(scene_rendering)
