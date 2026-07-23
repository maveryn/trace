"""Annotation helpers for refraction-layers bbox-set witnesses."""

from __future__ import annotations

from typing import List, Mapping, Sequence


def normalize_refraction_annotation_bbox_set(
    annotation_bbox_map: Mapping[str, Sequence[float]],
) -> List[List[float]]:
    """Return the two bend witness boxes in JSON-stable order."""

    required = ("interface_1_bend", "interface_2_bend")
    return [
        [float(value) for value in annotation_bbox_map[key]]
        for key in required
    ]
