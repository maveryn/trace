"""TRACE reward adapter for the EasyR1 training runtime."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from trace_tasks.core.reward_scoring import score_trace_response


REWARD_NAME = "trace_answer_only"
REWARD_TYPE = "batch"


def _decode_mapping(value: Any, *, field: str) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{field} must contain a JSON object") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return dict(value)


def _decode_reward_input(reward_input: Mapping[str, Any]) -> tuple[str, dict[str, Any], dict[str, Any], dict[str, Any]]:
    response = str(reward_input.get("response", "") or "")
    answer_gt = _decode_mapping(reward_input.get("ground_truth"), field="ground_truth")
    annotation_gt = _decode_mapping(reward_input.get("annotation_gt"), field="annotation_gt")
    reward_contract = _decode_mapping(reward_input.get("reward_contract"), field="reward_contract")

    if set(("type", "value")).difference(answer_gt):
        raise ValueError("ground_truth must contain type and value")
    if set(("type", "value")).difference(annotation_gt):
        raise ValueError("annotation_gt must contain type and value")
    answer_contract = reward_contract.get("answer")
    annotation_contract = reward_contract.get("annotation")
    if not isinstance(answer_contract, Mapping):
        raise ValueError("reward_contract.answer must be an object")
    if not isinstance(annotation_contract, Mapping):
        raise ValueError("reward_contract.annotation must be an object")
    if reward_contract.get("reward_contract_version") != "v0":
        raise ValueError("reward_contract_version must be v0")
    if answer_contract.get("id") != "answer_exact_match_v0":
        raise ValueError("reward_contract.answer must use answer_exact_match_v0")
    if answer_contract.get("type") != answer_gt.get("type"):
        raise ValueError("reward_contract.answer type must match ground_truth")
    if not annotation_contract.get("id"):
        raise ValueError("reward_contract.annotation must contain an id")
    if annotation_contract.get("type") != annotation_gt.get("type"):
        raise ValueError("reward_contract.annotation type must match annotation_gt")
    return response, answer_gt, annotation_gt, reward_contract


def _score_one(reward_input: Mapping[str, Any]) -> dict[str, float]:
    response, answer_gt, annotation_gt, reward_contract = _decode_reward_input(reward_input)
    result = score_trace_response(
        response=response,
        answer_gt=answer_gt,
        annotation_gt=annotation_gt,
        reward_contract=reward_contract,
        trace_reward_mode="answer",
        trace_answer_scoring="exact_json",
        answer_weight=1.0,
        annotation_weight=0.0,
        format_weight=0.05,
    )
    overall = float(result["overall"])
    return {
        "overall": overall,
        "score": overall,
        "accuracy": float(result["accuracy"]),
        "answer_reward": float(result["answer_reward"]),
        "format": float(result["format"]),
        "zero_reward": float(result["zero_reward"]),
        "trace_reward": 1.0,
    }


def compute_score(reward_inputs: list[dict[str, Any]]) -> list[dict[str, float]]:
    """Score a batch in input order using the configured reward contract."""

    return [_score_one(reward_input) for reward_input in reward_inputs]
