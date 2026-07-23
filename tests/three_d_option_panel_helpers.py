"""Shared assertions for 3D below-scene option panels."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from tests.three_d_canvas_helpers import assert_three_d_canvas_contract


def _bbox_center(bbox: Sequence[float]) -> tuple[float, float]:
    return (
        (float(bbox[0]) + float(bbox[2])) * 0.5,
        (float(bbox[1]) + float(bbox[3])) * 0.5,
    )


def _bbox_min_side(bbox: Sequence[float]) -> float:
    return min(float(bbox[2]) - float(bbox[0]), float(bbox[3]) - float(bbox[1]))


def _bbox_contains_point(bbox: Sequence[float], point: Sequence[float]) -> bool:
    return (
        float(bbox[0]) <= float(point[0]) <= float(bbox[2])
        and float(bbox[1]) <= float(point[1]) <= float(bbox[3])
    )


def assert_option_panel_matches_candidates(
    output: Any,
    candidate_specs: Sequence[Mapping[str, Any]],
    *,
    answer_label: str,
    answer_object_id: str,
    expected_image_size: tuple[int, int] | None = None,
) -> None:
    """Assert option-panel metadata maps letters to scene objects.

    Annotation should point to the selected object in the scene, not to option
    text in the panel.
    """

    trace = output.trace_payload["execution_trace"]
    render_map = output.trace_payload["render_map"]
    labels = sorted(str(spec["point_label"]) for spec in candidate_specs)
    choices = [dict(choice) for choice in render_map["option_choices"]]
    choice_by_label = {str(choice["label"]): choice for choice in choices}
    option_bboxes = dict(render_map["option_choice_bboxes_px"])
    panel_bbox = [float(value) for value in render_map["option_panel_bbox_px"]]
    if output.annotation_gt.type == "bbox":
        annotation_bboxes = [list(output.annotation_gt.value)]
    else:
        annotation_bboxes = [list(bbox) for bbox in output.annotation_gt.value]
    annotation_bbox = [float(value) for value in annotation_bboxes[0]]
    image_width, image_height = output.image.size

    assert_three_d_canvas_contract(output)
    assert int(render_map["option_panel_height_px"]) == int(image_height - panel_bbox[1])
    assert panel_bbox == [0.0, panel_bbox[1], float(image_width), float(image_height)]
    assert panel_bbox[1] > 0.0
    assert sorted(option_bboxes) == labels
    assert sorted(choice_by_label) == labels
    assert [str(choice["label"]) for choice in choices] == labels
    assert trace["option_choices"] == choices
    assert trace["option_descriptor_by_label"] == {
        str(choice["label"]): str(choice["descriptor"]) for choice in choices
    }
    text_records = list(
        output.trace_payload.get("render_spec", {})
        .get("drawn_text", {})
        .get("text_legibility", {})
        .get("records", [])
    )
    option_label_records = {
        str(record.get("option_label")): dict(record)
        for record in text_records
        if str(record.get("role")) == "three_d_option_label" and record.get("option_label") is not None
    }
    assert sorted(option_label_records) == labels

    for spec in candidate_specs:
        label = str(spec["point_label"])
        object_id = str(spec["object_id"])
        assert str(choice_by_label[label]["object_id"]) == object_id
        assert str(choice_by_label[label]["descriptor"]).strip()
        assert str(choice_by_label[label]["object_name"]).strip()
        label_record = option_label_records[label]
        text_center = _bbox_center(label_record["bbox_px"])
        badge_center = _bbox_center(label_record["badge_bbox_px"])
        assert abs(float(text_center[0]) - float(badge_center[0])) <= 1.25
        assert abs(float(text_center[1]) - float(badge_center[1])) <= 1.25

    raw_answer_bbox = [round(float(value), 3) for value in render_map["object_bboxes_px"][str(answer_object_id)]]
    if "annotation_bboxes_px" in render_map:
        normalized_answer_bbox = [round(float(value), 3) for value in render_map["annotation_bboxes_px"][0]]
        assert [[round(float(value), 3) for value in bbox] for bbox in render_map["annotation_raw_bboxes_px"]] == [
            raw_answer_bbox
        ]
    else:
        normalized_answer_bbox = list(raw_answer_bbox)
    assert annotation_bboxes == [normalized_answer_bbox]
    assert _bbox_min_side(normalized_answer_bbox) >= 24.0
    assert _bbox_contains_point(normalized_answer_bbox, _bbox_center(raw_answer_bbox))
    if output.annotation_gt.type == "bbox":
        assert output.trace_payload["projected_annotation"]["bbox"] == normalized_answer_bbox
    assert str(answer_label) in option_bboxes
    assert annotation_bbox[3] <= panel_bbox[1]
    assert annotation_bboxes[0] != option_bboxes[str(answer_label)]
