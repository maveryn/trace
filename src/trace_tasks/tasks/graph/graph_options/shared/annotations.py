"""Annotation projection helpers for graph option-panel scenes."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence


def project_option_panel_bbox(
    bbox_map: Mapping[str, Sequence[float]],
    panel_id: str,
) -> Dict[str, Any]:
    """Project one selected option panel to a scalar pixel-space bbox."""

    if str(panel_id) not in bbox_map:
        raise ValueError(f"missing bbox for option panel {panel_id}")
    bbox = [round(float(value), 3) for value in bbox_map[str(panel_id)]]
    return {"type": "bbox", "bbox": list(bbox), "pixel_bbox": list(bbox)}


__all__ = ["project_option_panel_bbox"]
