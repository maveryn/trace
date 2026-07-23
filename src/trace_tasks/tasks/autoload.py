"""Task-module autoload helpers.

These helpers deliberately separate scene-scoped registration from full
repository registration. Targeted callers can import one scene while global
inventory and build commands still import every task.
"""

from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec
from pathlib import Path

from trace_tasks.core.source_layout_policy import parse_public_task_id


_TASKS_ROOT = Path(__file__).resolve().parent
_SKIPPED_DIR_NAMES = frozenset({"__pycache__", ".ipynb_checkpoints", "shared"})
_SKIPPED_FILE_NAMES = frozenset({"__init__.py", "shared.py"})


def module_name_for_public_task_id(task_id: str) -> str:
    """Return the taxonomy-v0 module path for one public task id."""

    parts = parse_public_task_id(str(task_id))
    return f"trace_tasks.tasks.{parts.domain}.{parts.scene_id}.{parts.objective_contract}"


def task_module_exists(task_id: str) -> bool:
    """Return whether the direct taxonomy-v0 task module can be imported."""

    try:
        module_name = module_name_for_public_task_id(str(task_id))
    except ValueError:
        return False
    try:
        return find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False


def register_task_module(task_id: str) -> None:
    """Import the direct taxonomy-v0 module for ``task_id`` if it exists."""

    if not task_module_exists(str(task_id)):
        return
    import_module(module_name_for_public_task_id(str(task_id)))


def _iter_public_module_paths(root: Path) -> list[Path]:
    """Return candidate public task files under ``root`` in deterministic order."""

    if not root.exists():
        return []
    paths: list[Path] = []
    for path in sorted(root.rglob("*.py")):
        if path.name in _SKIPPED_FILE_NAMES or path.name.startswith("_"):
            continue
        if any(part in _SKIPPED_DIR_NAMES for part in path.relative_to(_TASKS_ROOT).parts[:-1]):
            continue
        paths.append(path)
    return paths


def _module_name_from_path(path: Path) -> str:
    relative = path.relative_to(_TASKS_ROOT).with_suffix("")
    return "trace_tasks.tasks." + ".".join(relative.parts)


def register_scene_task_modules(domain: str, scene_id: str) -> None:
    """Import public task modules for one scene only."""

    scene_root = _TASKS_ROOT / str(domain) / str(scene_id)
    for module_path in _iter_public_module_paths(scene_root):
        import_module(_module_name_from_path(module_path))


def register_all_task_modules() -> None:
    """Import all discoverable public task modules."""

    for module_path in _iter_public_module_paths(_TASKS_ROOT):
        import_module(_module_name_from_path(module_path))


__all__ = [
    "module_name_for_public_task_id",
    "register_all_task_modules",
    "register_scene_task_modules",
    "register_task_module",
    "task_module_exists",
]
