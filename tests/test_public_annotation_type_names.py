"""Public annotation type-name guardrails."""

from __future__ import annotations

from pathlib import Path

from trace_tasks.core.type_registry import load_type_registry


TEXT_SUFFIXES = {".py", ".md", ".json", ".yaml", ".yml", ".toml", ".txt", ".html", ".js", ".css"}
SCAN_ROOTS = (
    Path("src/trace_tasks"),
    Path("tests"),
    Path("docs"),
    Path("scripts"),
)
RETIRED_PUBLIC_ANNOTATION_TYPES = tuple(
    f"keyed_{geometry}{suffix}"
    for geometry in ("point", "bbox")
    for suffix in ("_map", "_set_map")
)


def _iter_existing_text_files() -> list[Path]:
    files: list[Path] = []
    for root in SCAN_ROOTS:
        if not root.exists():
            continue
        files.extend(path for path in root.rglob("*") if path.is_file() and path.suffix in TEXT_SUFFIXES)
    return sorted(files)


def test_type_registry_uses_map_annotation_names() -> None:
    registry = load_type_registry()

    for retired_type in RETIRED_PUBLIC_ANNOTATION_TYPES:
        assert not registry.validate_annotation_type(retired_type)
    for current_type in ("point_map", "point_set_map", "bbox_map", "bbox_set_map"):
        assert registry.validate_annotation_type(current_type)


def test_retired_keyed_map_annotation_names_do_not_reappear() -> None:
    offenders: list[str] = []
    for path in _iter_existing_text_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        present = [name for name in RETIRED_PUBLIC_ANNOTATION_TYPES if name in text]
        if present:
            offenders.append(f"{path}: {', '.join(present)}")

    assert offenders == []
