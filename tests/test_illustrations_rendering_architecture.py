"""Architecture guard for illustration scene/rendering separation."""

from __future__ import annotations

import ast
from pathlib import Path


SCENE_FILES = tuple(sorted(Path("src/trace_tasks/tasks/illustrations/shared").glob("*_scene.py")))
FORBIDDEN_PIL_IMPORTS = {"Image", "ImageDraw", "ImageFont"}
FORBIDDEN_DRAWING_CALLS = {"rectangle", "rounded_rectangle", "ellipse", "polygon", "line", "arc", "pieslice"}


def test_illustration_scene_modules_are_drawing_free_interfaces() -> None:
    assert SCENE_FILES
    for path in SCENE_FILES:
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                assert not node.name.startswith("_draw"), f"{path} defines scene-local drawing helper {node.name}"
                assert not node.name.startswith("draw_"), f"{path} defines scene-local drawing helper {node.name}"
            if isinstance(node, ast.ImportFrom) and node.module == "PIL":
                imported = {alias.name for alias in node.names}
                assert not imported.intersection(FORBIDDEN_PIL_IMPORTS), f"{path} imports PIL drawing primitives"
            if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_DRAWING_CALLS:
                assert False, f"{path} calls drawing primitive {node.attr}; move it to a rendering module"
            if isinstance(node, ast.Attribute) and node.attr in {"new", "Draw"}:
                base_name = node.value.id if isinstance(node.value, ast.Name) else ""
                assert base_name not in {"Image", "ImageDraw"}, f"{path} creates images/draw contexts directly"
