"""Neutral sampling helpers for named-grid icons tasks."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from ....shared.config_defaults import group_default
from ...shared.procedural_named_icons import (
    PROCEDURAL_NAMED_ICON_FILL_STYLES,
    PROCEDURAL_NAMED_ICON_SHAPES,
    validate_procedural_named_icon_fill_style_support,
)

from .defaults import DEFAULT_GRID_SIZE_SUPPORT, NamedGridDefaults


_DEFAULTS = NamedGridDefaults()


def string_probability_map(values: Sequence[str], *, selected: str | None = None) -> Dict[str, float]:
    support = tuple(str(value) for value in values)
    if not support:
        return {}
    if selected is not None:
        return {str(value): (1.0 if str(value) == str(selected) else 0.0) for value in support}
    probability = 1.0 / float(len(support))
    return {str(value): float(probability) for value in support}


def int_bounds(params: Mapping[str, Any], defaults: Mapping[str, Any], low_key: str, high_key: str, fallback_low: int, fallback_high: int) -> Tuple[int, int]:
    low = int(params.get(low_key, group_default(defaults, low_key, fallback_low)))
    high = int(params.get(high_key, group_default(defaults, high_key, fallback_high)))
    if low < 0 or high < low:
        raise ValueError(f"invalid {low_key}/{high_key} bounds")
    return int(low), int(high)


def shape_support(params: Mapping[str, Any], defaults: Mapping[str, Any], *, min_count: int = 8) -> Tuple[str, ...]:
    raw = params.get("shape_id_support", group_default(defaults, "shape_id_support", PROCEDURAL_NAMED_ICON_SHAPES))
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("shape_id_support must be a sequence")
    values = tuple(dict.fromkeys(str(value).strip() for value in raw if str(value).strip()))
    unsupported = sorted(set(values) - set(PROCEDURAL_NAMED_ICON_SHAPES))
    if unsupported:
        raise ValueError(f"unsupported procedural named icon shapes: {unsupported}")
    if len(values) < int(min_count):
        raise ValueError(f"named-grid task needs at least {int(min_count)} supported named shapes")
    return values


def fill_style_support(params: Mapping[str, Any], defaults: Mapping[str, Any]) -> Tuple[str, ...]:
    raw = params.get(
        "named_icon_fill_style_support",
        group_default(defaults, "named_icon_fill_style_support", _DEFAULTS.named_icon_fill_style_support),
    )
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raw = PROCEDURAL_NAMED_ICON_FILL_STYLES
    return validate_procedural_named_icon_fill_style_support(tuple(str(value) for value in raw))


def fill_style_probability_map(params: Mapping[str, Any], defaults: Mapping[str, Any], support: Sequence[str]) -> Dict[str, float]:
    raw = params.get(
        "named_icon_fill_style_weights",
        group_default(defaults, "named_icon_fill_style_weights", None),
    )
    if not isinstance(raw, Mapping):
        probability = 1.0 / float(len(tuple(support)))
        return {str(value): float(probability) for value in support}
    weights = {str(value): max(0.0, float(raw.get(str(value), 0.0))) for value in support}
    total = sum(float(value) for value in weights.values())
    if total <= 0.0:
        probability = 1.0 / float(len(tuple(support)))
        return {str(value): float(probability) for value in support}
    return {str(value): float(weights[str(value)]) / float(total) for value in support}


def grid_size_support(params: Mapping[str, Any], defaults: Mapping[str, Any]) -> Tuple[Tuple[int, int], ...]:
    raw = params.get("grid_size_support", group_default(defaults, "grid_size_support", DEFAULT_GRID_SIZE_SUPPORT))
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("grid_size_support must be a sequence")
    values: List[Tuple[int, int]] = []
    for item in raw:
        if isinstance(item, str):
            parts = str(item).lower().split("x")
            if len(parts) != 2:
                raise ValueError(f"unsupported grid size string: {item}")
            rows, cols = int(parts[0]), int(parts[1])
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)) and len(item) >= 2:
            rows, cols = int(item[0]), int(item[1])
        else:
            raise ValueError(f"unsupported grid size entry: {item}")
        if rows < 2 or cols < 2:
            raise ValueError("named-grid sizes must be at least 2x2")
        values.append((int(rows), int(cols)))
    support = tuple(dict.fromkeys(values))
    if not support:
        raise ValueError("grid_size_support resolved no grid sizes")
    return support


def grid_size_label(size: Tuple[int, int]) -> str:
    return f"{int(size[0])}x{int(size[1])}"


def resolve_target_shape(rng, *, params: Mapping[str, Any], defaults: Mapping[str, Any], support: Sequence[str]) -> Tuple[str, Dict[str, float]]:
    explicit_shape = params.get("shape_id", params.get("target_shape_id"))
    if explicit_shape is not None:
        target_shape_id = str(explicit_shape)
        if target_shape_id not in set(support):
            raise ValueError(f"target shape must be one of {support}")
        return str(target_shape_id), string_probability_map(tuple(str(value) for value in support), selected=str(target_shape_id))
    target_shape_id = str(rng.choice(tuple(str(value) for value in support)))
    return str(target_shape_id), string_probability_map(tuple(str(value) for value in support))
