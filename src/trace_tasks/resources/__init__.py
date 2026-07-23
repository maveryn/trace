"""Access resources bundled with the installed Trace package."""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Union


_PathInput = Union[str, PathLike[str]]
_RESOURCE_ROOT = Path(__file__).resolve().parent


def resource_root() -> Path:
    """Return the filesystem root containing packaged Trace resources."""

    return _RESOURCE_ROOT


def safe_resource_join(root: _PathInput, *parts: _PathInput) -> Path:
    """Join relative resource components without permitting root traversal."""

    base = Path(root).expanduser().resolve()
    relative = Path(*parts)
    if not parts or relative.is_absolute() or not relative.parts:
        raise ValueError(f"invalid Trace resource path: {parts!r}")
    if any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"invalid Trace resource path: {parts!r}")
    candidate = base.joinpath(relative)
    try:
        candidate.resolve().relative_to(base)
    except ValueError as exc:
        raise ValueError(f"Trace resource path escapes its root: {parts!r}") from exc
    return candidate


def resource_path(*parts: _PathInput) -> Path:
    """Resolve a traversal-safe path below the packaged resource root."""

    return safe_resource_join(_RESOURCE_ROOT, *parts)


__all__ = ["resource_path", "resource_root", "safe_resource_join"]
