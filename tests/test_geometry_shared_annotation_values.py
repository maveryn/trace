from __future__ import annotations

import pytest

from trace_tasks.tasks.geometry.shared.annotation_values import (
    bbox_set_annotation_artifacts,
    build_role_value_annotation,
    keyed_bbox_annotation_artifacts,
    keyed_point_annotation_artifacts,
    point_set_annotation_artifacts,
)


def test_keyed_point_annotation_artifacts_round_and_project() -> None:
    artifacts = keyed_point_annotation_artifacts(
        {"target": (10.12345, 20.98765), "source": (30.0, 40.0)},
        roles=("source", "target"),
    )

    assert artifacts.annotation_type == "point_map"
    assert artifacts.value == {"source": [30.0, 40.0], "target": [10.123, 20.988]}
    assert artifacts.projected_annotation == {
        "type": "point_map",
        "point_map": {"source": [30.0, 40.0], "target": [10.123, 20.988]},
        "pixel_point_map": {"source": [30.0, 40.0], "target": [10.123, 20.988]},
    }


def test_keyed_bbox_annotation_artifacts_can_preserve_bbox_only_projection() -> None:
    artifacts = keyed_bbox_annotation_artifacts(
        {"region": (1.23456, 2.0, 11.0, 12.98765)},
        roles=("region",),
        include_point_centers=False,
    )

    assert artifacts.annotation_type == "bbox_map"
    assert artifacts.value == {"region": [1.235, 2.0, 11.0, 12.988]}
    assert artifacts.projected_annotation == {
        "type": "bbox_map",
        "bbox_map": {"region": [1.235, 2.0, 11.0, 12.988]},
        "pixel_bbox_map": {"region": [1.235, 2.0, 11.0, 12.988]},
    }


def test_bbox_set_annotation_artifacts_include_center_projection() -> None:
    artifacts = bbox_set_annotation_artifacts([(0.0, 10.0, 20.0, 30.0)])

    assert artifacts.annotation_type == "bbox_set"
    assert artifacts.value == [[0.0, 10.0, 20.0, 30.0]]
    assert artifacts.projected_annotation["point_set"] == [[10.0, 20.0]]
    assert artifacts.projected_annotation["pixel_point_set"] == [[10.0, 20.0]]


def test_point_set_annotation_artifacts_round_and_project() -> None:
    artifacts = point_set_annotation_artifacts([(1.1111, 2.2222), (3.3333, 4.4444)])

    assert artifacts.annotation_type == "point_set"
    assert artifacts.value == [[1.111, 2.222], [3.333, 4.444]]
    assert artifacts.projected_annotation == {
        "type": "point_set",
        "point_set": [[1.111, 2.222], [3.333, 4.444]],
        "pixel_point_set": [[1.111, 2.222], [3.333, 4.444]],
    }


def test_build_role_value_annotation_allows_duplicate_matching_tokens() -> None:
    annotation = build_role_value_annotation(
        roles=("a", "b"),
        role_to_annotation={"a": "x", "b": "x"},
        role_to_value={"a": 7, "b": 7.0},
    )

    assert annotation == {"x": 7}


def test_build_role_value_annotation_rejects_conflicting_duplicate_tokens() -> None:
    with pytest.raises(ValueError, match="conflicting values"):
        build_role_value_annotation(
            roles=("a", "b"),
            role_to_annotation={"a": "x", "b": "x"},
            role_to_value={"a": 7, "b": 8},
        )
