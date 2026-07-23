"""Trace core reward-scoring tests."""

from __future__ import annotations

import json

import numpy as np

from trace_tasks.core.reward_scoring import score_trace_response


def _reward_contract(annotation_id: str, annotation_type: str, answer_type: str = "integer") -> dict[str, object]:
    return {
        "reward_contract_version": "v0",
        "answer": {"id": "answer_exact_match_v0", "type": answer_type},
        "annotation": {"id": annotation_id, "type": annotation_type},
    }


def test_core_reward_scoring_uses_trace_v0_contract_ids() -> None:
    score = score_trace_response(
        response='{"answer":2,"annotation":[[10,10,20,20]]}',
        answer_gt={"type": "integer", "value": 2},
        annotation_gt={"type": "bbox_set", "value": [[10, 10, 20, 20]]},
        reward_contract=_reward_contract("bbox_set_soft_iou_v0", "bbox_set"),
    )

    assert score["overall"] == 1.0
    assert score["answer_reward"] == 1.0
    assert score["annotation_reward"] == 1.0


def test_core_reward_scoring_scores_scalar_point_exactly() -> None:
    score = score_trace_response(
        response='{"answer":2,"annotation":[10,10]}',
        answer_gt={"type": "integer", "value": 2},
        annotation_gt={"type": "point", "value": [10, 10]},
        reward_contract=_reward_contract("point_soft_distance_v0", "point"),
        point_half_life_px=32.0,
    )

    assert score["overall"] == 1.0
    assert score["annotation_reward"] == 1.0
    assert score["annotation_parse_ok"] == 1.0
    assert score["annotation_type_point"] == 1.0


def test_core_reward_scoring_scores_scalar_point_soft_distance() -> None:
    score = score_trace_response(
        response='{"answer":2,"annotation":[42,10]}',
        answer_gt={"type": "integer", "value": 2},
        annotation_gt={"type": "point", "value": [10, 10]},
        reward_contract=_reward_contract("point_soft_distance_v0", "point"),
        point_half_life_px=32.0,
    )

    assert np.isclose(score["annotation_reward"], 0.5)
    assert score["annotation_parse_ok"] == 1.0


def test_core_reward_scoring_rejects_point_set_shape_for_scalar_point() -> None:
    score = score_trace_response(
        response='{"answer":2,"annotation":[[10,10]]}',
        answer_gt={"type": "integer", "value": 2},
        annotation_gt={"type": "point", "value": [10, 10]},
        reward_contract=_reward_contract("point_soft_distance_v0", "point"),
    )

    assert score["annotation_reward"] == 0.0
    assert score["annotation_parse_ok"] == 0.0


def test_core_reward_scoring_scores_scalar_segment_exactly_with_reversed_endpoints() -> None:
    score = score_trace_response(
        response='{"answer":2,"annotation":[[30,40],[10,20]]}',
        answer_gt={"type": "integer", "value": 2},
        annotation_gt={"type": "segment", "value": [[10, 20], [30, 40]]},
        reward_contract=_reward_contract("segment_soft_distance_v0", "segment"),
        point_half_life_px=32.0,
    )

    assert score["overall"] == 1.0
    assert score["annotation_reward"] == 1.0
    assert score["annotation_parse_ok"] == 1.0
    assert score["annotation_type_segment"] == 1.0


def test_core_reward_scoring_rejects_segment_set_shape_for_scalar_segment() -> None:
    score = score_trace_response(
        response='{"answer":2,"annotation":[[[10,20],[30,40]]]}',
        answer_gt={"type": "integer", "value": 2},
        annotation_gt={"type": "segment", "value": [[10, 20], [30, 40]]},
        reward_contract=_reward_contract("segment_soft_distance_v0", "segment"),
    )

    assert score["annotation_reward"] == 0.0
    assert score["annotation_parse_ok"] == 0.0


def test_core_reward_scoring_scores_scalar_bbox_exactly() -> None:
    score = score_trace_response(
        response='{"answer":2,"annotation":[10,10,20,20]}',
        answer_gt={"type": "integer", "value": 2},
        annotation_gt={"type": "bbox", "value": [10, 10, 20, 20]},
        reward_contract=_reward_contract("bbox_soft_iou_v0", "bbox"),
    )

    assert score["overall"] == 1.0
    assert score["annotation_reward"] == 1.0
    assert score["annotation_parse_ok"] == 1.0
    assert score["annotation_type_bbox"] == 1.0


def test_core_reward_scoring_scores_scalar_bbox_iou() -> None:
    score = score_trace_response(
        response='{"answer":2,"annotation":[5,0,15,10]}',
        answer_gt={"type": "integer", "value": 2},
        annotation_gt={"type": "bbox", "value": [0, 0, 10, 10]},
        reward_contract=_reward_contract("bbox_soft_iou_v0", "bbox"),
    )

    assert np.isclose(score["annotation_reward"], 1.0 / 3.0)
    assert score["annotation_parse_ok"] == 1.0


def test_core_reward_scoring_rejects_bbox_set_shape_for_scalar_bbox() -> None:
    score = score_trace_response(
        response='{"answer":2,"annotation":[[10,10,20,20]]}',
        answer_gt={"type": "integer", "value": 2},
        annotation_gt={"type": "bbox", "value": [10, 10, 20, 20]},
        reward_contract=_reward_contract("bbox_soft_iou_v0", "bbox"),
    )

    assert score["annotation_reward"] == 0.0
    assert score["annotation_parse_ok"] == 0.0


def test_core_reward_scoring_scores_point_map_by_shared_keys() -> None:
    response = json.dumps({"answer": 2, "annotation": {"A": [132, 200], "extra": [0, 0]}})
    score = score_trace_response(
        response=response,
        answer_gt={"type": "integer", "value": 2},
        annotation_gt={"type": "point_map", "value": {"A": [100, 200], "B": [320, 420]}},
        reward_contract=_reward_contract("point_map_soft_distance_v0", "point_map"),
        point_half_life_px=32.0,
    )

    assert np.isclose(score["annotation_reward"], 0.5 / 3.0)
    assert score["annotation_parse_ok"] == 1.0
    assert score["annotation_shared_key_count"] == 1.0
    assert score["annotation_missing_key_count"] == 1.0
    assert score["annotation_extra_key_count"] == 1.0


def test_core_reward_scoring_scores_bbox_map_by_shared_keys() -> None:
    response = json.dumps(
        {
            "answer": 2,
            "annotation": {
                "source": [10, 10, 20, 20],
                "extra": [0, 0, 5, 5],
            },
        }
    )
    score = score_trace_response(
        response=response,
        answer_gt={"type": "integer", "value": 2},
        annotation_gt={
            "type": "bbox_map",
            "value": {
                "source": [10, 10, 20, 20],
                "target": [30, 30, 40, 40],
            },
        },
        reward_contract=_reward_contract("bbox_map_soft_iou_v0", "bbox_map"),
    )

    assert np.isclose(score["annotation_reward"], 1.0 / 3.0)
    assert score["annotation_parse_ok"] == 1.0
    assert score["annotation_shared_key_count"] == 1.0
    assert score["annotation_missing_key_count"] == 1.0
    assert score["annotation_extra_key_count"] == 1.0


def test_core_reward_scoring_scores_point_set_map_by_shared_keys_and_unordered_points() -> None:
    response = json.dumps(
        {
            "answer": 2,
            "annotation": {
                "A": [[132, 200]],
                "extra": [[0, 0]],
            },
        }
    )
    score = score_trace_response(
        response=response,
        answer_gt={"type": "integer", "value": 2},
        annotation_gt={
            "type": "point_set_map",
            "value": {
                "A": [[100, 200], [200, 200]],
                "B": [[320, 420]],
            },
        },
        reward_contract=_reward_contract("point_set_map_soft_distance_v0", "point_set_map"),
        point_half_life_px=32.0,
    )

    assert np.isclose(score["annotation_reward"], 0.25 / 3.0)
    assert score["annotation_parse_ok"] == 1.0
    assert score["annotation_shared_key_count"] == 1.0
    assert score["annotation_missing_key_count"] == 1.0
    assert score["annotation_extra_key_count"] == 1.0
    assert score["annotation_assigned_count"] == 1.0


def test_core_reward_scoring_scores_bbox_set_map_by_shared_keys_and_unordered_boxes() -> None:
    response = json.dumps(
        {
            "answer": 2,
            "annotation": {
                "source": [[10, 10, 20, 20]],
                "extra": [[0, 0, 5, 5]],
            },
        }
    )
    score = score_trace_response(
        response=response,
        answer_gt={"type": "integer", "value": 2},
        annotation_gt={
            "type": "bbox_set_map",
            "value": {
                "source": [[30, 30, 40, 40], [10, 10, 20, 20]],
                "target": [[50, 50, 60, 60]],
            },
        },
        reward_contract=_reward_contract("bbox_set_map_soft_iou_v0", "bbox_set_map"),
    )

    assert np.isclose(score["annotation_reward"], 0.5 / 3.0)
    assert score["annotation_parse_ok"] == 1.0
    assert score["annotation_shared_key_count"] == 1.0
    assert score["annotation_missing_key_count"] == 1.0
    assert score["annotation_extra_key_count"] == 1.0
    assert score["annotation_assigned_count"] == 1.0


def test_core_reward_scoring_rejects_non_current_annotation_contract_id() -> None:
    score = score_trace_response(
        response='{"answer":2,"annotation":[[10,10,20,20]]}',
        answer_gt={"type": "integer", "value": 2},
        annotation_gt={"type": "bbox_set", "value": [[10, 10, 20, 20]]},
        reward_contract={
            "reward_contract_version": "v0",
            "answer": {"id": "answer_exact_match_v0", "type": "integer"},
            "annotation": {"id": "bbox_set_soft_iou_v1", "type": "bbox_set"},
        },
    )

    assert score["overall"] == 0.5
    assert score["answer_reward"] == 1.0
    assert score["annotation_reward"] == 0.0
    assert score["annotation_parse_ok"] == 0.0
