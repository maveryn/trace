"""Annotation projection helpers for density-curve chart scenes."""

from __future__ import annotations

from typing import Any

from trace_tasks.tasks.charts.density_curve.shared.state import DensityCurveDataset, DensityCurveRendered


def _bbox_center(bbox: list[float]) -> list[float]:
    """Return the center point of a bbox."""

    return [
        round((float(bbox[0]) + float(bbox[2])) / 2.0, 3),
        round((float(bbox[1]) + float(bbox[3])) / 2.0, 3),
    ]


def density_curve_annotation_payload(
    *,
    dataset: DensityCurveDataset,
    rendered: DensityCurveRendered,
) -> tuple[str, list[float], dict[str, Any]]:
    """Return annotation payloads for the answer curve witness."""

    answer_label = str(dataset.query.answer_label)
    annotation_key = str(dataset.query.annotation_key)
    if annotation_key == "answer_density_at_x":
        annotation_type = "point"
        annotation = list(rendered.density_at_x_points_px[str(answer_label)])
    elif annotation_key == "answer_mode_marker":
        annotation_type = "point"
        annotation = _bbox_center(list(rendered.mode_marker_bboxes_px[str(answer_label)]))
    elif annotation_key == "answer_mean_marker":
        annotation_type = "point"
        annotation = _bbox_center(list(rendered.mean_marker_bboxes_px[str(answer_label)]))
    elif annotation_key == "answer_interval_mass":
        annotation_type = "point"
        annotation = list(rendered.interval_mass_points_px[str(answer_label)])
    else:
        raise ValueError(f"unsupported density-curve annotation key: {annotation_key}")
    projected: dict[str, Any] = {
        "type": str(annotation_type),
        "answer_label": str(answer_label),
        "annotation_key": str(annotation_key),
    }
    if str(annotation_type) == "point":
        projected["point"] = list(annotation)
        projected["pixel_point"] = list(annotation)
    else:
        projected["bbox"] = list(annotation)
        projected["pixel_bbox"] = list(annotation)
    return str(annotation_type), list(annotation), dict(projected)
