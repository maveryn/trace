"""Annotation projection helpers for error-bar series scenes."""

from __future__ import annotations

from typing import Any

from trace_tasks.tasks.charts.errorbar_series.shared.state import ErrorbarDataset, ErrorbarRendered


def _bbox_center(bbox: list[float]) -> list[float]:
    """Return the midpoint of one projected bbox."""

    x0, y0, x1, y1 = [float(value) for value in bbox]
    return [round((x0 + x1) / 2.0, 3), round((y0 + y1) / 2.0, 3)]


def _interval_segment_for_key(*, key: str, rendered: ErrorbarRendered) -> tuple[str, str, list[list[float]]]:
    """Return the lower-to-upper endpoint segment for one rendered error-bar key."""

    series_label, x_label = str(key).split(":", 1)
    point_record = rendered.point_map_px[str(series_label)][str(x_label)]
    segment = [list(point_record["lower_bound"]), list(point_record["upper_bound"])]
    return str(series_label), str(x_label), segment


def annotation_payload(
    *,
    dataset: ErrorbarDataset,
    rendered: ErrorbarRendered,
) -> tuple[str, Any, dict[str, Any], list[dict[str, Any]]]:
    """Project task-bound symbolic mark keys into pixel annotation payloads."""

    if str(dataset.query.annotation_kind) == "bbox_set":
        boxes = []
        refs = []
        for key in dataset.query.annotation_item_keys:
            if str(key) not in rendered.errorbar_bboxes_px:
                raise RuntimeError(f"missing errorbar bbox for {key}")
            boxes.append(list(rendered.errorbar_bboxes_px[str(key)]))
            refs.append({"key": str(key), "bbox_px": list(rendered.errorbar_bboxes_px[str(key)])})
        projected = {"type": "bbox_set", "bbox_set": list(boxes), "pixel_bbox_set": list(boxes), "annotation_refs": list(refs)}
        return "bbox_set", list(boxes), dict(projected), [dict(ref) for ref in refs]

    if str(dataset.query.annotation_kind) == "point_set":
        points = []
        refs = []
        for key in dataset.query.annotation_item_keys:
            if str(key) not in rendered.errorbar_bboxes_px:
                raise RuntimeError(f"missing errorbar bbox for {key}")
            series_label, x_label = str(key).split(":", 1)
            point = _bbox_center(list(rendered.errorbar_bboxes_px[str(key)]))
            points.append(list(point))
            refs.append(
                {
                    "key": str(key),
                    "series_label": str(series_label),
                    "x_label": str(x_label),
                    "point_xy": list(point),
                }
            )
        projected = {
            "type": "point_set",
            "point_set": list(points),
            "pixel_point_set": list(points),
            "annotation_refs": list(refs),
        }
        return "point_set", list(points), dict(projected), [dict(ref) for ref in refs]

    if str(dataset.query.annotation_kind) == "segment_set":
        segments = []
        refs = []
        for key in dataset.query.annotation_item_keys:
            series_label, x_label, segment = _interval_segment_for_key(key=str(key), rendered=rendered)
            segments.append([list(point) for point in segment])
            refs.append(
                {
                    "key": str(key),
                    "series_label": str(series_label),
                    "x_label": str(x_label),
                    "segment_px": [list(point) for point in segment],
                }
            )
        projected = {
            "type": "segment_set",
            "segment_set": list(segments),
            "pixel_segment_set": list(segments),
            "annotation_refs": list(refs),
        }
        return "segment_set", list(segments), dict(projected), [dict(ref) for ref in refs]

    if str(dataset.query.annotation_kind) == "point":
        key = str(dataset.query.annotation_item_keys[0])
        series_label, x_label = key.split(":", 1)
        bound_kind = str(dataset.query.params["bound_kind"])
        point = list(rendered.point_map_px[str(series_label)][str(x_label)][f"{bound_kind}_bound"])
        refs = [{"key": str(key), "series_label": str(series_label), "x_label": str(x_label), "point_xy": list(point)}]
        projected = {
            "type": "point",
            "point": list(point),
            "pixel_point": list(point),
            "annotation_refs": list(refs),
        }
        return "point", list(point), dict(projected), [dict(ref) for ref in refs]

    raise ValueError(f"unsupported annotation kind: {dataset.query.annotation_kind}")


__all__ = ["annotation_payload"]
