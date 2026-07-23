"""Annotation projection for hexbin-density charts."""

from __future__ import annotations

from typing import List

from trace_tasks.tasks.charts.hexbin_density.shared.state import HexbinDataset, RenderedHexbinScene


def annotation_point_set(dataset: HexbinDataset, rendered: RenderedHexbinScene) -> List[List[float]]:
    annotation: List[List[float]] = []
    for bin_id in dataset.query.annotation_bin_ids:
        point = rendered.bin_centers_px.get(str(bin_id))
        if point is None:
            raise RuntimeError(f"missing annotation point for bin: {bin_id}")
        annotation.append(list(point))
    return annotation


def projected_annotation_payload(dataset: HexbinDataset, annotation: list[list[float]]) -> dict[str, object]:
    return {
        "type": "point_set",
        "point_set": [list(value) for value in annotation],
        "pixel_point_set": [list(value) for value in annotation],
        "annotation_bin_ids": list(dataset.query.annotation_bin_ids),
    }


__all__ = [
    "annotation_point_set",
    "projected_annotation_payload",
]
