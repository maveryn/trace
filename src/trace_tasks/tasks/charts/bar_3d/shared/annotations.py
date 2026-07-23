"""Annotation projection helpers for the 3D bar scene."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.shared.annotation_artifacts import (
    AnnotationArtifacts,
    segment_set_annotation_artifacts,
    point_set_annotation_artifacts,
)

from .state import _RenderedBarGrid, _Selection


def _round_point(point: Sequence[float]) -> list[float]:
    return [round(float(point[0]), 3), round(float(point[1]), 3)]


def _trace_by_bar_id(rendered: _RenderedBarGrid) -> dict[str, Mapping[str, Any]]:
    return {str(trace["bar_id"]): trace for trace in rendered.bar_traces}


def annotation_points_for_bars(
    *,
    rendered: _RenderedBarGrid,
    bar_ids: Sequence[str],
) -> list[list[float]]:
    """Project selected bar ids to top-center point annotations."""

    trace_by_id = _trace_by_bar_id(rendered)
    return [
        _round_point(trace_by_id[str(bar_id)]["top_center_px"])
        for bar_id in bar_ids
        if str(bar_id) in trace_by_id
    ]


def annotation_point_pairs_for_bar_pairs(
    *,
    rendered: _RenderedBarGrid,
    bar_id_pairs: Sequence[Sequence[str]],
) -> list[list[list[float]]]:
    """Project selected bar-id pairs to top-center segment annotations."""

    trace_by_id = _trace_by_bar_id(rendered)
    pairs: list[list[list[float]]] = []
    for pair in bar_id_pairs:
        if len(pair) < 2:
            continue
        first_id = str(pair[0])
        second_id = str(pair[1])
        if first_id not in trace_by_id or second_id not in trace_by_id:
            continue
        pairs.append(
            [
                _round_point(trace_by_id[first_id]["top_center_px"]),
                _round_point(trace_by_id[second_id]["top_center_px"]),
            ]
        )
    return pairs


def point_set_map_annotation_artifacts(
    keyed_points: Mapping[str, Sequence[Sequence[float]]],
) -> AnnotationArtifacts:
    """Build public keyed point-set map artifacts for role-bound bar groups."""

    value = {
        str(key): [_round_point(point) for point in points]
        for key, points in keyed_points.items()
    }
    projected_annotation = {
        "type": "point_set_map",
        "point_set_map": {key: [list(point) for point in points] for key, points in value.items()},
        "pixel_point_set_map": {key: [list(point) for point in points] for key, points in value.items()},
    }
    return AnnotationArtifacts(
        annotation_type="point_set_map",
        value={key: [list(point) for point in points] for key, points in value.items()},
        annotation_gt=TypedValue(
            type="point_set_map",
            value={key: [list(point) for point in points] for key, points in value.items()},
        ),
        projected_annotation=projected_annotation,
    )


def point_map_annotation_artifacts(
    keyed_points: Mapping[str, Sequence[float]],
) -> AnnotationArtifacts:
    """Build public keyed point-map artifacts for role-bound bar points."""

    value = {str(key): _round_point(point) for key, point in keyed_points.items()}
    projected_annotation = {
        "type": "point_map",
        "point_map": {key: list(point) for key, point in value.items()},
        "pixel_point_map": {key: list(point) for key, point in value.items()},
    }
    return AnnotationArtifacts(
        annotation_type="point_map",
        value={key: list(point) for key, point in value.items()},
        annotation_gt=TypedValue(
            type="point_map",
            value={key: list(point) for key, point in value.items()},
        ),
        projected_annotation=projected_annotation,
    )


def annotation_artifacts_for_selection(
    *,
    rendered: _RenderedBarGrid,
    selection: _Selection,
) -> tuple[AnnotationArtifacts, dict[str, Any]]:
    """Project one task-owned selection into its declared annotation contract."""

    if str(selection.annotation_kind) == "segment_set":
        point_pairs = annotation_point_pairs_for_bar_pairs(
            rendered=rendered,
            bar_id_pairs=selection.annotation_bar_id_pairs,
        )
        annotation = segment_set_annotation_artifacts(point_pairs)
        return annotation, {
            "type": "segment_set",
            "annotation_bar_id_pairs": [
                [str(first), str(second)]
                for first, second in selection.annotation_bar_id_pairs
            ],
        }

    if str(selection.annotation_kind) == "point_map":
        trace_by_id = _trace_by_bar_id(rendered)
        keyed_points: dict[str, list[float]] = {}
        for key, bar_ids in (selection.annotation_bar_id_groups or {}).items():
            if not bar_ids:
                continue
            bar_id = str(bar_ids[0])
            if bar_id not in trace_by_id:
                continue
            keyed_points[str(key)] = _round_point(trace_by_id[bar_id]["top_center_px"])
        annotation = point_map_annotation_artifacts(keyed_points)
        return annotation, {
            "type": "point_map",
            "keys": {
                str(key): str(bar_ids[0])
                for key, bar_ids in (selection.annotation_bar_id_groups or {}).items()
                if bar_ids
            },
        }

    if str(selection.annotation_kind) == "point_set_map":
        trace_by_id = _trace_by_bar_id(rendered)
        keyed_points = {
            str(key): [
                _round_point(trace_by_id[str(bar_id)]["top_center_px"])
                for bar_id in bar_ids
                if str(bar_id) in trace_by_id
            ]
            for key, bar_ids in (selection.annotation_bar_id_groups or {}).items()
        }
        annotation = point_set_map_annotation_artifacts(keyed_points)
        return annotation, {
            "type": "point_set_map",
            "keys": {
                str(key): [str(bar_id) for bar_id in bar_ids]
                for key, bar_ids in (selection.annotation_bar_id_groups or {}).items()
            },
        }

    annotation_points = annotation_points_for_bars(
        rendered=rendered,
        bar_ids=selection.annotation_bar_ids,
    )
    annotation = point_set_annotation_artifacts(annotation_points)
    return annotation, {
        "type": "point_set",
        "annotation_bar_ids": [str(bar_id) for bar_id in selection.annotation_bar_ids],
    }


__all__ = [
    "annotation_artifacts_for_selection",
    "annotation_point_pairs_for_bar_pairs",
    "annotation_points_for_bars",
    "point_map_annotation_artifacts",
    "point_set_map_annotation_artifacts",
]
