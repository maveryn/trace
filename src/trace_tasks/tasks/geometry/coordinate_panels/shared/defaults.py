"""Configuration resolvers for coordinate-panel scenes."""

from __future__ import annotations

from typing import Any, Mapping, Sequence, Tuple

from trace_tasks.tasks.shared.config_defaults import group_default


def resolve_int_param(params: Mapping[str, Any], defaults: Mapping[str, Any], key: str, fallback: int) -> int:
    """Resolve an integer parameter from explicit params, task defaults, or fallback."""

    return int(params.get(str(key), group_default(defaults, str(key), int(fallback))))


def resolve_label_pool(
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    key: str,
    fallback: Sequence[str],
) -> Tuple[str, ...]:
    """Resolve and validate a visible panel-label pool."""

    raw = params.get(str(key), group_default(defaults, str(key), list(fallback)))
    if isinstance(raw, str):
        labels = tuple(part.strip() for part in raw.split(",") if part.strip())
    elif isinstance(raw, Sequence):
        labels = tuple(str(item).strip() for item in raw if str(item).strip())
    else:
        labels = tuple(str(item) for item in fallback)
    if not labels:
        raise ValueError(f"{key} must contain at least one label")
    if len(set(labels)) != len(labels):
        raise ValueError(f"{key} must not contain duplicates")
    return labels


__all__ = ["resolve_int_param", "resolve_label_pool"]
