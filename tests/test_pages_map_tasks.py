"""Behavior tests for pages map-navigation tasks."""

from __future__ import annotations

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.pages.map.destination_after_directions_label import PagesMapDestinationAfterDirectionsLabelTask
from trace_tasks.tasks.pages.map.landmark_after_route_step_label import PagesMapLandmarkAfterRouteStepLabelTask
from tests.helpers import extract_prompt_json_example


def _task_cases():
    return (
        (
            "destination_after_directions",
            "map_destination_after_directions_label",
            PagesMapDestinationAfterDirectionsLabelTask(),
        ),
        (
            "landmark_after_route_step",
            "map_landmark_after_route_step_label",
            PagesMapLandmarkAfterRouteStepLabelTask(),
        ),
    )


def test_pages_map_navigation_label_contract_matches_annotation_bboxes() -> None:
    for query_id_index, (prompt_query_key, question_format, task) in enumerate(_task_cases()):
        out = task.generate(
            62400 + query_id_index,
            params={"scene_variant": "campus_map"},
            max_attempts=10,
        )
        trace = out.trace_payload
        execution = trace["execution_trace"]
        render = trace["render_spec"]
        render_map = trace["render_map"]
        query_params = trace["query_spec"]["params"]
        annotation_bboxes = [[float(value) for value in bbox] for bbox in out.annotation_gt.value]
        annotation_bbox_ids = [str(bbox_id) for bbox_id in execution["annotation_bbox_ids"]]
        bbox_source = {
            **render_map["landmark_bboxes_px"],
            **render_map["zone_label_bboxes_px"],
        }

        assert out.answer_gt.type == "string"
        assert out.annotation_gt.type == "bbox_sequence"
        assert sorted(out.prompt_variants.keys()) == ["answer_and_annotation", "answer_only"]
        assert not hasattr(task, "scene_id")
        assert str(render["font_assets"]["scene_id"]) == "map"
        assert str(out.query_id) == SINGLE_QUERY_ID
        assert str(execution["query_id"]) == SINGLE_QUERY_ID
        assert str(query_params["query_id"]) == SINGLE_QUERY_ID
        assert str(execution["prompt_query_key"]) == str(prompt_query_key)
        assert str(execution["source_query_id"]) == str(prompt_query_key)
        assert str(query_params["prompt_query_key"]) == str(prompt_query_key)
        assert str(query_params["source_query_id"]) == str(prompt_query_key)
        assert str(execution["scene_variant"]) == "campus_map"
        assert str(execution["question_format"]) == str(question_format)
        assert str(execution["view_family"]) == "printed_campus_map"
        assert out.image.size == (int(render["canvas_width"]), int(render["canvas_height"]))
        assert str(out.answer_gt.value) == str(execution["answer_label"])
        assert 8 <= int(execution["landmark_count"]) <= 12
        assert len(execution["landmark_specs"]) == int(execution["landmark_count"])
        assert len(render_map["landmark_bboxes_px"]) == int(execution["landmark_count"])
        assert len(render_map["zone_label_bboxes_px"]) == 4
        assert trace["projected_annotation"]["bbox_sequence"] == annotation_bboxes

        expected_bboxes = [[float(value) for value in bbox_source[bbox_id]] for bbox_id in annotation_bbox_ids]
        assert annotation_bboxes == expected_bboxes
        assert [str(item) for item in execution["supporting_bbox_ids"]] == annotation_bbox_ids

        if str(prompt_query_key) == "destination_after_directions":
            assert len(execution["route_landmark_ids"]) >= 3
            assert str(execution["annotation_semantics"]) == "route_landmarks_ordered"
            assert len(execution["annotation_landmark_bbox_ids"]) == len(execution["route_landmark_ids"])
            assert not execution["highlighted_route_landmark_ids"]
        elif str(prompt_query_key) == "landmark_after_route_step":
            assert len(execution["highlighted_route_landmark_ids"]) >= 4
            assert render_map["highlighted_route_bboxes_px"]
            assert str(execution["annotation_semantics"]) == "highlighted_route_landmarks_ordered_to_answer"


def test_pages_map_navigation_label_prompt_examples_match_string_contract() -> None:
    expected = {
        "destination_after_directions": (PagesMapDestinationAfterDirectionsLabelTask(), "Clinic"),
        "landmark_after_route_step": (PagesMapLandmarkAfterRouteStepLabelTask(), "Gallery"),
    }

    for index, (_prompt_query_key, (task, expected_answer)) in enumerate(expected.items(), start=62460):
        out = task.generate(index, params={"scene_variant": "campus_map"}, max_attempts=10)
        answer_and_annotation = extract_prompt_json_example(out.prompt_variants["answer_and_annotation"])
        answer_only = extract_prompt_json_example(out.prompt_variants["answer_only"])
        assert str(answer_and_annotation["answer"]) == str(expected_answer)
        assert str(answer_only["answer"]) == str(expected_answer)
        assert isinstance(answer_and_annotation["annotation"], list)


def test_pages_map_navigation_label_is_deterministic() -> None:
    task = PagesMapLandmarkAfterRouteStepLabelTask()
    params = {"scene_variant": "campus_map"}
    out_a = task.generate(62510, params=params, max_attempts=10)
    out_b = task.generate(62510, params=params, max_attempts=10)

    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.trace_payload["query_spec"]["prompt_variant"] == out_b.trace_payload["query_spec"]["prompt_variant"]
    assert out_a.prompt == out_b.prompt
    assert out_a.image.tobytes() == out_b.image.tobytes()


def test_pages_map_public_tasks_use_single_query_with_prompt_branch_metadata() -> None:
    for index, (prompt_query_key, _question_format, task) in enumerate(_task_cases(), start=62540):
        out = task.generate(index, params={}, max_attempts=10)
        execution = out.trace_payload["execution_trace"]
        query_params = out.trace_payload["query_spec"]["params"]

        assert str(out.query_id) == SINGLE_QUERY_ID
        assert str(execution["query_id"]) == SINGLE_QUERY_ID
        assert str(query_params["query_id"]) == SINGLE_QUERY_ID
        assert str(execution["prompt_query_key"]) == str(prompt_query_key)
        assert str(query_params["prompt_query_key"]) == str(prompt_query_key)
        assert str(execution["scene_variant"]) == "campus_map"
        assert query_params["query_id_probabilities"] == {SINGLE_QUERY_ID: 1.0}
