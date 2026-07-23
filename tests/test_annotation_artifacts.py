from __future__ import annotations

from trace_tasks.tasks.shared.annotation_artifacts import (
    bbox_annotation_artifacts,
    point_annotation_artifacts,
)


def test_bbox_annotation_artifacts_build_scalar_bbox_payload() -> None:
    artifacts = bbox_annotation_artifacts([1.23456, 2.34567, 30.0004, 40.9999])

    assert artifacts.annotation_type == "bbox"
    assert artifacts.value == [1.235, 2.346, 30.0, 41.0]
    assert artifacts.annotation_gt.to_dict() == {
        "type": "bbox",
        "value": [1.235, 2.346, 30.0, 41.0],
    }
    assert artifacts.projected_annotation == {
        "type": "bbox",
        "bbox": [1.235, 2.346, 30.0, 41.0],
        "pixel_bbox": [1.235, 2.346, 30.0, 41.0],
    }


def test_point_annotation_artifacts_build_scalar_point_payload() -> None:
    artifacts = point_annotation_artifacts([5.4321, 6.7891])

    assert artifacts.annotation_type == "point"
    assert artifacts.value == [5.432, 6.789]
    assert artifacts.annotation_gt.to_dict() == {
        "type": "point",
        "value": [5.432, 6.789],
    }
    assert artifacts.projected_annotation == {
        "type": "point",
        "point": [5.432, 6.789],
        "pixel_point": [5.432, 6.789],
    }
