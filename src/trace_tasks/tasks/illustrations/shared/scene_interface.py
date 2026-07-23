"""Helpers for drawing-free public illustration scene interfaces."""

from __future__ import annotations

from types import ModuleType
from typing import Any, MutableMapping


_BLOCKED_EXPORTS = {
    "Any",
    "Callable",
    "Dict",
    "Image",
    "ImageDraw",
    "ImageFont",
    "Iterable",
    "List",
    "Mapping",
    "MutableMapping",
    "Path",
    "Sequence",
    "Tuple",
    "annotations",
    "dataclass",
    "field",
    "math",
}


def export_scene_interface(module_globals: MutableMapping[str, Any], rendering_module: ModuleType) -> None:
    """Populate a public scene module from its implementation module."""

    exports = {
        name: getattr(rendering_module, name)
        for name in dir(rendering_module)
        if not name.startswith("_") and name not in _BLOCKED_EXPORTS
    }
    module_globals.update(exports)
    module_globals["__all__"] = tuple(sorted(exports))


__all__ = ["export_scene_interface"]
