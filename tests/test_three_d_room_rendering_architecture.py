import ast
from pathlib import Path


ROOM_TASK_OR_COMMON_FILES = (
    Path("src/trace_tasks/tasks/three_d/room/wall_object_camera_distance_label.py"),
    Path("src/trace_tasks/tasks/three_d/room/wall_object_same_wall_reference_label.py"),
    Path("src/trace_tasks/tasks/three_d/room/wall_object_side_relation_label.py"),
    Path("src/trace_tasks/tasks/three_d/room/_lifecycle.py"),
    Path("src/trace_tasks/tasks/three_d/room/shared/metrics.py"),
    Path("src/trace_tasks/tasks/three_d/room/shared/relations.py"),
    Path("src/trace_tasks/tasks/three_d/room/shared/spatial_primitives.py"),
    Path("src/trace_tasks/tasks/three_d/room/shared/state.py"),
)

ROOM_RENDERER_FILES = (
    Path("src/trace_tasks/tasks/three_d/room/shared/rendering.py"),
    Path("src/trace_tasks/tasks/three_d/shared/room_wall_object_rendering.py"),
    Path("src/trace_tasks/tasks/three_d/shared/room_floor_object_rendering.py"),
)


def _imported_names(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.extend(alias.name for alias in node.names)
    return names


def test_room_public_task_modules_are_drawing_free() -> None:
    forbidden_snippets = (
        "from PIL import",
        "import PIL",
        "ImageDraw",
        "Image.new",
        "Image.alpha_composite",
        "def _draw_",
        "draw.",
    )

    for path in ROOM_TASK_OR_COMMON_FILES:
        source = path.read_text()
        for snippet in forbidden_snippets:
            assert snippet not in source, f"{path} should not contain renderer code: {snippet}"
        assert not any(name.startswith("_draw_") for name in _imported_names(path)), str(path)


def test_room_renderer_modules_are_explicit_scene_boundaries() -> None:
    for path in ROOM_RENDERER_FILES:
        assert path.exists()

    renderer_source = ROOM_RENDERER_FILES[0].read_text()
    assert "class _RenderedRoomScene" in renderer_source
    assert "render_room_scene_3d" in renderer_source


def test_room_legacy_root_helpers_are_removed() -> None:
    assert not Path("src/trace_tasks/tasks/three_d/room/wall_mounted_common.py").exists()
    assert not Path("src/trace_tasks/tasks/three_d/room/wall_mounted_dataset.py").exists()
    assert not Path("src/trace_tasks/tasks/three_d/room/wall_mounted_rendering.py").exists()
    assert not Path("src/trace_tasks/tasks/three_d/room/wall_object_camera_distance.py").exists()
    assert not Path("src/trace_tasks/tasks/three_d/room/wall_object_same_wall_reference.py").exists()
    assert not Path("src/trace_tasks/tasks/three_d/room/wall_object_side_relation.py").exists()
