"""Identity-free trace serialization helpers for coordinate-panel scenes."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def panel_bboxes_by_label(panels_by_label: Mapping[str, Any]) -> dict[str, list[int]]:
    """Return panel boxes keyed by visible option label."""

    return {str(label): list(spec.panel_bbox) for label, spec in panels_by_label.items()}


def plot_bboxes_by_label(panels_by_label: Mapping[str, Any]) -> dict[str, list[int]]:
    """Return plot boxes keyed by visible option label."""

    return {str(label): list(spec.plot_bbox) for label, spec in panels_by_label.items()}


def sorted_panel_entities(panels_trace: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return deterministic entity records for panel traces."""

    return [dict(panels_trace[str(label)]) for label in sorted(panels_trace)]


def normalize_point_set(points: Sequence[Sequence[float]]) -> list[list[float]]:
    """Return a JSON-ready pixel point set."""

    return [[float(point[0]), float(point[1])] for point in points]


def normalize_segment_set(segments: Sequence[Sequence[Sequence[float]]]) -> list[list[list[float]]]:
    """Return a JSON-ready pixel segment set."""

    return [
        [[float(point[0]), float(point[1])] for point in segment]
        for segment in segments
    ]


def quadrilateral_panels_trace(rendered: Any, query: Any) -> dict[str, dict[str, Any]]:
    """Serialize quadrilateral panel points and classifications."""

    return {
        str(label): {
            "label": str(label),
            "points_graph": [[int(value) for value in point] for point in spec.points],
            "points_px": [[float(value) for value in point] for point in spec.points_px],
            "classified_kind": str(spec.classified_kind),
            "panel_bbox": list(spec.panel_bbox),
            "plot_bbox": list(spec.plot_bbox),
            "is_answer": str(label) == str(query.winner_label),
        }
        for label, spec in rendered.panels_by_label.items()
    }


def quadrilateral_annotation_value(rendered: Any, query: Any) -> list[list[float]]:
    """Return selected-panel quadrilateral point witnesses."""

    return normalize_point_set(rendered.panels_by_label[str(query.winner_label)].points_px)


def quadrilateral_render_map_extra(rendered: Any) -> dict[str, Any]:
    """Return quadrilateral point maps keyed by panel label."""

    return {
        "points_graph_by_label": {
            str(label): [[int(value) for value in point] for point in spec.points]
            for label, spec in rendered.panels_by_label.items()
        },
        "points_px_by_label": {
            str(label): [[float(value) for value in point] for point in spec.points_px]
            for label, spec in rendered.panels_by_label.items()
        },
    }


def quadrilateral_projected_annotation(rendered: Any, annotation_value: Any) -> dict[str, Any]:
    """Return projected point-set annotation for quadrilateral panel selection."""

    return {
        "type": "point_set",
        "point_set": [list(point) for point in annotation_value],
        "pixel_point_set": [list(point) for point in annotation_value],
        "points_px_by_label": {
            str(label): [[float(value) for value in point] for point in spec.points_px]
            for label, spec in rendered.panels_by_label.items()
        },
        "panel_bbox_by_label": panel_bboxes_by_label(rendered.panels_by_label),
    }


def segment_panels_trace(rendered: Any, query: Any) -> dict[str, dict[str, Any]]:
    """Serialize segment panel endpoints and relation flags."""

    return {
        str(label): {
            "label": str(label),
            "segments_graph": [[[int(point[0]), int(point[1])] for point in segment] for segment in spec.segments_graph],
            "segments_px": [[[float(point[0]), float(point[1])] for point in segment] for segment in spec.segments_px],
            "relation_flags": dict(spec.relation_flags),
            "panel_bbox": list(spec.panel_bbox),
            "plot_bbox": list(spec.plot_bbox),
            "is_answer": str(label) == str(query.winner_label),
        }
        for label, spec in rendered.panels_by_label.items()
    }


def segment_annotation_value(rendered: Any, query: Any) -> list[list[list[float]]]:
    """Return selected-panel segment witnesses."""

    return normalize_segment_set(rendered.panels_by_label[str(query.winner_label)].segments_px)


def segment_render_map_extra(rendered: Any) -> dict[str, Any]:
    """Return segment maps keyed by panel label."""

    return {
        "segments_graph_by_label": {
            str(label): [[[int(point[0]), int(point[1])] for point in segment] for segment in spec.segments_graph]
            for label, spec in rendered.panels_by_label.items()
        },
        "segments_px_by_label": {
            str(label): [[[float(point[0]), float(point[1])] for point in segment] for segment in spec.segments_px]
            for label, spec in rendered.panels_by_label.items()
        },
    }


def segment_projected_annotation(rendered: Any, annotation_value: Any) -> dict[str, Any]:
    """Return projected segment-set annotation for segment panel selection."""

    return {
        "type": "segment_set",
        "segment_set": [list(segment) for segment in annotation_value],
        "pixel_segment_set": [list(segment) for segment in annotation_value],
        "panel_bbox_by_label": panel_bboxes_by_label(rendered.panels_by_label),
    }


def transform_panels_trace(rendered: Any, query: Any) -> dict[str, dict[str, Any]]:
    """Serialize source/candidate point sets and transform flags."""

    return {
        str(label): {
            "label": str(label),
            "source_points_graph": [[int(value) for value in point] for point in spec.source_points_graph],
            "candidate_points_graph": [[int(value) for value in point] for point in spec.candidate_points_graph],
            "source_points_px": [[float(value) for value in point] for point in spec.source_points_px],
            "candidate_points_px": [[float(value) for value in point] for point in spec.candidate_points_px],
            "transform_flags": dict(spec.transform_flags),
            "panel_bbox": list(spec.panel_bbox),
            "plot_bbox": list(spec.plot_bbox),
            "is_answer": str(label) == str(query.winner_label),
        }
        for label, spec in rendered.panels_by_label.items()
    }


def transform_annotation_value(rendered: Any, query: Any) -> list[list[float]]:
    """Return selected-panel source and candidate point witnesses."""

    selected = rendered.panels_by_label[str(query.winner_label)]
    return normalize_point_set([*selected.source_points_px, *selected.candidate_points_px])


def transform_render_map_extra(rendered: Any) -> dict[str, Any]:
    """Return transform point maps keyed by panel label."""

    return {
        "source_points_px_by_label": {
            str(label): [[float(value) for value in point] for point in spec.source_points_px]
            for label, spec in rendered.panels_by_label.items()
        },
        "candidate_points_px_by_label": {
            str(label): [[float(value) for value in point] for point in spec.candidate_points_px]
            for label, spec in rendered.panels_by_label.items()
        },
    }


def transform_projected_annotation(rendered: Any, annotation_value: Any) -> dict[str, Any]:
    """Return projected point-set annotation for transform panel selection."""

    return {
        "type": "point_set",
        "point_set": [list(point) for point in annotation_value],
        "pixel_point_set": [list(point) for point in annotation_value],
        "panel_bbox_by_label": panel_bboxes_by_label(rendered.panels_by_label),
    }


__all__ = [
    "normalize_point_set",
    "normalize_segment_set",
    "panel_bboxes_by_label",
    "plot_bboxes_by_label",
    "quadrilateral_annotation_value",
    "quadrilateral_panels_trace",
    "quadrilateral_projected_annotation",
    "quadrilateral_render_map_extra",
    "segment_annotation_value",
    "segment_panels_trace",
    "segment_projected_annotation",
    "segment_render_map_extra",
    "sorted_panel_entities",
    "transform_annotation_value",
    "transform_panels_trace",
    "transform_projected_annotation",
    "transform_render_map_extra",
]
