"""Annotation projection helpers for paper-fold result puzzles."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from trace_tasks.core.types import TypedValue


def _round_bbox(bbox: Sequence[float]) -> list[float]:
    """Normalize one image-pixel bbox into stable float coordinates."""

    return [round(float(value), 3) for value in bbox]


def option_choice_bbox(
    option_choice_bbox_map: Mapping[str, Sequence[float]],
    option_choice_id: str,
) -> tuple[TypedValue, dict[str, Any], dict[str, Any]]:
    """Return scalar bbox annotation artifacts for the selected option panel."""

    option_id = str(option_choice_id)
    if option_id not in option_choice_bbox_map:
        raise RuntimeError(f"missing paper-fold option bbox for {option_id!r}")
    bbox = _round_bbox(option_choice_bbox_map[option_id])
    projected = {
        "bbox": list(bbox),
        "pixel_bbox": list(bbox),
        "value": list(bbox),
    }
    witness = {
        "type": "bbox",
        "value": list(bbox),
        "ordered_item_ids": [option_id],
    }
    return TypedValue(type="bbox", value=list(bbox)), projected, witness


__all__ = ["option_choice_bbox"]
