"""Annotation helpers for shadow-cause bbox witnesses."""

from __future__ import annotations

from typing import List, Mapping, Sequence


def normalize_shadow_annotation_bbox(
    candidate_light_sources: Mapping[str, Mapping[str, Sequence[float]]],
    *,
    selected_label: str,
) -> List[float]:
    """Return the selected labeled lamp option box in JSON-stable form."""

    return [float(value) for value in candidate_light_sources[str(selected_label)]["option_bbox_px"]]
