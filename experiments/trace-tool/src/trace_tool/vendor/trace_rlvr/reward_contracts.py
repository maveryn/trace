"""Public RLVR reward-contract metadata for Trace instances."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping


REWARD_CONTRACT_VERSION = "v0"
ANSWER_REWARD_CONTRACT_ID = "answer_exact_match_v0"

ANNOTATION_REWARD_CONTRACT_IDS = frozenset(
    {
        "bbox_soft_iou_v0",
        "bbox_sequence_soft_iou_v0",
        "bbox_set_soft_iou_v0",
        "bbox_map_soft_iou_v0",
        "bbox_set_map_soft_iou_v0",
        "point_map_soft_distance_v0",
        "point_set_map_soft_distance_v0",
        "point_soft_distance_v0",
        "segment_soft_distance_v0",
        "segment_set_soft_distance_v0",
        "point_sequence_soft_distance_v0",
        "point_set_soft_distance_v0",
    }
)

_ANNOTATION_REWARD_BY_TYPE = {
    "bbox": "bbox_soft_iou_v0",
    "bbox_sequence": "bbox_sequence_soft_iou_v0",
    "bbox_set": "bbox_set_soft_iou_v0",
    "bbox_map": "bbox_map_soft_iou_v0",
    "bbox_set_map": "bbox_set_map_soft_iou_v0",
    "point_map": "point_map_soft_distance_v0",
    "point_set_map": "point_set_map_soft_distance_v0",
    "point": "point_soft_distance_v0",
    "segment": "segment_soft_distance_v0",
    "segment_set": "segment_set_soft_distance_v0",
    "point_sequence": "point_sequence_soft_distance_v0",
    "point_set": "point_set_soft_distance_v0",
}


@dataclass(frozen=True)
class RewardMatcherSpec:
    """One side of a public reward contract."""

    id: str
    type: str

    def to_dict(self) -> Dict[str, str]:
        return {"id": self.id, "type": self.type}


@dataclass(frozen=True)
class RewardContract:
    """Portable reward metadata stored with Trace instances."""

    reward_contract_version: str
    answer: RewardMatcherSpec
    annotation: RewardMatcherSpec

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reward_contract_version": self.reward_contract_version,
            "answer": self.answer.to_dict(),
            "annotation": self.annotation.to_dict(),
        }


def resolve_annotation_reward_contract_id(annotation_type: str) -> str:
    """Resolve the public annotation reward id for one annotation type."""

    normalized = str(annotation_type).strip()
    if not normalized:
        raise ValueError("reward_contract requires a non-empty annotation type")
    resolved = _ANNOTATION_REWARD_BY_TYPE.get(normalized)
    if resolved is None:
        raise ValueError(f"unsupported annotation type for reward_contract: {normalized}")
    return resolved


def resolve_reward_contract(*, answer_type: str, annotation_type: str) -> RewardContract:
    """Resolve the public reward contract for one Trace instance."""

    normalized_answer_type = str(answer_type).strip()
    normalized_annotation_type = str(annotation_type).strip()
    if not normalized_answer_type:
        raise ValueError("reward_contract requires a non-empty answer type")
    annotation_contract_id = resolve_annotation_reward_contract_id(normalized_annotation_type)
    return RewardContract(
        reward_contract_version=REWARD_CONTRACT_VERSION,
        answer=RewardMatcherSpec(id=ANSWER_REWARD_CONTRACT_ID, type=normalized_answer_type),
        annotation=RewardMatcherSpec(id=annotation_contract_id, type=normalized_annotation_type),
    )


def validate_reward_contract_payload(
    payload: Mapping[str, Any],
    *,
    answer_type: str | None = None,
    annotation_type: str | None = None,
) -> str | None:
    """Return a human-readable validation message when a payload is invalid."""

    if not isinstance(payload, Mapping):
        return "reward_contract must be an object"

    version = payload.get("reward_contract_version")
    if version != REWARD_CONTRACT_VERSION:
        return f"reward_contract_version must be {REWARD_CONTRACT_VERSION!r}"

    answer = payload.get("answer")
    if not isinstance(answer, Mapping):
        return "reward_contract.answer must be an object with keys {id, type}"
    answer_id = answer.get("id")
    answer_payload_type = answer.get("type")
    if not isinstance(answer_id, str) or not answer_id:
        return "reward_contract.answer.id must be a non-empty string"
    if not isinstance(answer_payload_type, str) or not answer_payload_type:
        return "reward_contract.answer.type must be a non-empty string"
    if answer_id != ANSWER_REWARD_CONTRACT_ID:
        return f"reward_contract.answer.id must be {ANSWER_REWARD_CONTRACT_ID!r}"

    annotation = payload.get("annotation")
    if not isinstance(annotation, Mapping):
        return "reward_contract.annotation must be an object with keys {id, type}"
    annotation_id = annotation.get("id")
    annotation_payload_type = annotation.get("type")
    if not isinstance(annotation_id, str) or not annotation_id:
        return "reward_contract.annotation.id must be a non-empty string"
    if not isinstance(annotation_payload_type, str) or not annotation_payload_type:
        return "reward_contract.annotation.type must be a non-empty string"
    if annotation_id not in ANNOTATION_REWARD_CONTRACT_IDS:
        allowed = ", ".join(sorted(ANNOTATION_REWARD_CONTRACT_IDS))
        return f"reward_contract.annotation.id must be one of {{{allowed}}}"

    if answer_type is not None and str(answer_type).strip() != answer_payload_type:
        return "reward_contract.answer.type must match answer_gt.type"
    if annotation_type is not None and str(annotation_type).strip() != annotation_payload_type:
        return "reward_contract.annotation.type must match annotation_gt.type"

    if annotation_type is not None:
        try:
            expected_annotation_id = resolve_annotation_reward_contract_id(str(annotation_type))
        except ValueError as exc:
            return str(exc)
        if annotation_id != expected_annotation_id:
            return (
                "reward_contract.annotation.id must match the resolved contract for "
                f"annotation_gt.type ({expected_annotation_id})"
            )

    return None
