"""Annotation projection helpers for uncertainty-band charts."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .state import Dataset, Rendered


def overlap_center_points(
    *,
    dataset: Dataset,
    rendered: Rendered,
    x_indices: Sequence[int],
) -> list[list[float]]:
    """Return center points inside each counted overlap region."""

    return [
        list(rendered.overlap_points_px[str(dataset.x_labels[int(index)])])
        for index in x_indices
    ]


def band_width_segment(
    *,
    rendered: Rendered,
    target_series_id: str,
    x_label: str,
) -> list[list[float]]:
    """Return the lower-to-upper visual segment for one band width."""

    series_points: dict[str, Any] = rendered.point_map_px[str(target_series_id)][str(x_label)]
    return [
        list(series_points["lower_bound"]),
        list(series_points["upper_bound"]),
    ]


__all__ = ["band_width_segment", "overlap_center_points"]
