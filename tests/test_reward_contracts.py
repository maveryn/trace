"""Reward-contract resolution and validation tests."""

from __future__ import annotations

from trace_tasks.core.reward_contracts import (
    ANSWER_REWARD_CONTRACT_ID,
    resolve_reward_contract,
    validate_reward_contract_payload,
)
from trace_tasks.core.type_registry import load_type_registry


def test_resolve_reward_contract_for_supported_annotation_types() -> None:
    cases = [
        ("bbox", "bbox_soft_iou_v0"),
        ("bbox_sequence", "bbox_sequence_soft_iou_v0"),
        ("bbox_set", "bbox_set_soft_iou_v0"),
        ("bbox_map", "bbox_map_soft_iou_v0"),
        ("bbox_set_map", "bbox_set_map_soft_iou_v0"),
        ("point", "point_soft_distance_v0"),
        ("point_map", "point_map_soft_distance_v0"),
        ("point_set_map", "point_set_map_soft_distance_v0"),
        ("point_sequence", "point_sequence_soft_distance_v0"),
        ("segment", "segment_soft_distance_v0"),
        ("segment_set", "segment_set_soft_distance_v0"),
        ("point_set", "point_set_soft_distance_v0"),
    ]

    for annotation_type, expected_contract_id in cases:
        contract = resolve_reward_contract(answer_type="integer", annotation_type=annotation_type)
        assert contract.answer.id == ANSWER_REWARD_CONTRACT_ID
        assert contract.answer.type == "integer"
        assert contract.annotation.id == expected_contract_id
        assert contract.annotation.type == annotation_type
        assert validate_reward_contract_payload(
            contract.to_dict(),
            answer_type="integer",
            annotation_type=annotation_type,
        ) is None


def test_resolve_reward_contract_rejects_unsupported_annotation_types() -> None:
    unsupported_types = [
        "integer",
        "integer_list",
        "label_list",
        "polygon_set",
        "line_set",
        "mask",
        "heatmap",
    ]

    for annotation_type in unsupported_types:
        try:
            resolve_reward_contract(answer_type="integer", annotation_type=annotation_type)
        except ValueError as exc:
            assert "unsupported annotation type" in str(exc)
        else:
            raise AssertionError(f"unsupported annotation type unexpectedly resolved: {annotation_type}")


def test_reward_contract_validation_rejects_mismatched_annotation_contract() -> None:
    payload = resolve_reward_contract(answer_type="integer", annotation_type="bbox_set").to_dict()
    payload["annotation"]["id"] = "point_set_soft_distance_v0"
    error = validate_reward_contract_payload(
        payload,
        answer_type="integer",
        annotation_type="bbox_set",
    )
    assert error is not None
    assert "must match the resolved contract" in error


def test_scalar_annotation_types_are_registered() -> None:
    registry = load_type_registry()

    assert registry.validate_annotation_type("point") is True
    assert registry.validate_annotation_type("bbox") is True
    assert registry.validate_annotation_type("segment") is True
