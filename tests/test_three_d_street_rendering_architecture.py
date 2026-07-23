import ast
from pathlib import Path


STREET_TASK_OR_COMMON_FILES = (
    Path("src/trace_tasks/tasks/three_d/street/shared/state.py"),
    Path("src/trace_tasks/tasks/three_d/street/intersection_nearest_label.py"),
    Path("src/trace_tasks/tasks/three_d/street/lane_ahead_object_label.py"),
    Path("src/trace_tasks/tasks/three_d/street/same_road_arm_reference_label.py"),
)

STREET_RENDERER_FILES = (
    Path("src/trace_tasks/tasks/three_d/street/shared/rendering.py"),
    Path("src/trace_tasks/tasks/three_d/street/shared/components.py"),
    Path("src/trace_tasks/tasks/three_d/street/shared/objects.py"),
    Path("src/trace_tasks/tasks/three_d/shared/street_object_rendering.py"),
    Path("src/trace_tasks/tasks/three_d/shared/street_vehicle_object_rendering.py"),
    Path("src/trace_tasks/tasks/three_d/shared/street_fixture_object_rendering.py"),
    Path("src/trace_tasks/tasks/three_d/shared/street_pedestrian_object_rendering.py"),
    Path("src/trace_tasks/tasks/three_d/shared/street_landscape_object_rendering.py"),
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


def test_street_public_task_modules_are_drawing_free() -> None:
    forbidden_snippets = (
        "from PIL import",
        "import PIL",
        "ImageDraw",
        "Image.new",
        "Image.alpha_composite",
        "def _draw_",
        "draw.",
    )

    for path in STREET_TASK_OR_COMMON_FILES:
        source = path.read_text()
        for snippet in forbidden_snippets:
            assert snippet not in source, f"{path} should not contain renderer code: {snippet}"
        assert not any(name.startswith("_draw_") for name in _imported_names(path)), str(path)


def test_street_renderer_modules_are_explicit_scene_boundaries() -> None:
    for path in STREET_RENDERER_FILES:
        assert path.exists()

    renderer_source = STREET_RENDERER_FILES[0].read_text()
    assert "class _RenderedStreetScene" in renderer_source
    assert "render_street_intersection_scene_3d" in renderer_source
