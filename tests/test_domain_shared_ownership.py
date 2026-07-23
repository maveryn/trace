"""Regression checks for domain-owned shared helper placement."""

from __future__ import annotations

from pathlib import Path
import re


_DOMAIN_ROOTS = {
    "games": Path("src/trace_tasks/tasks/games"),
    "puzzles": Path("src/trace_tasks/tasks/puzzles"),
    "symbolic": Path("src/trace_tasks/tasks/symbolic"),
}
_CROSS_DOMAIN_IMPORT_RE = re.compile(
    r"^\s*(?:from\s+(?:trace\.tasks\.)?(games|puzzles|symbolic)(?:\.|\s+import)|"
    r"from\s+\.+(games|puzzles|symbolic)(?:\.|\s+import)|"
    r"import\s+trace\.tasks\.(games|puzzles|symbolic)(?:\.|$))",
    re.MULTILINE,
)


def test_games_puzzles_and_misc_do_not_cross_import_domain_helpers() -> None:
    """Games, puzzles, and symbolic source must not import each other's helpers."""

    offenders: list[str] = []
    for source_domain, root in _DOMAIN_ROOTS.items():
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for match in _CROSS_DOMAIN_IMPORT_RE.finditer(text):
                target_domain = next(group for group in match.groups() if group is not None)
                if str(target_domain) != str(source_domain):
                    offenders.append(f"{path}:{target_domain}")
    assert offenders == []


def test_moved_scene_helpers_are_not_kept_under_puzzles_shared() -> None:
    """Moved game/symbolic scene helpers should not leave stale puzzle copies."""

    stale_paths = [
        Path("src/trace_tasks/tasks/puzzles/shared/clock_scene.py"),
        Path("src/trace_tasks/tasks/puzzles/shared/dice_scene.py"),
        Path("src/trace_tasks/tasks/puzzles/shared/music_notation_scene.py"),
        Path("src/trace_tasks/tasks/puzzles/shared/organic_structure_scene.py"),
        Path("src/trace_tasks/tasks/puzzles/shared/spinner_scene.py"),
        Path("src/trace_tasks/tasks/puzzles/shared/sokoban_scene.py"),
    ]
    assert [str(path) for path in stale_paths if path.exists()] == []


def test_sokoban_scene_helper_is_games_owned() -> None:
    """Sokoban shared rendering is owned by the games domain."""

    from trace_tasks.tasks.games.shared import sokoban_scene

    assert hasattr(sokoban_scene, "render_sokoban_scene")
