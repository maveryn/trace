"""Row-level scoring with TRACE's own code.

Two scores per response, both on the *final* assistant message:

- ``strict_*``: TRACE's GRPO reward (``trace_rlvr.py``): ``score_trace_response``
  in answer-only / exact_json mode with ``format_weight=0.05``. ``strict_correct``
  is the RL accuracy term; ``strict_reward`` = 0.95 * correct + 0.05 * format.
- ``official_*``: the TRACE validation protocol: ground-truth-blind deterministic
  extraction, canonical exact match, and (optionally) a Qwen3-32B extraction
  pass for rows that are ``missing``/``ambiguous``; unresolved rows score 0.
"""

from __future__ import annotations

from typing import Any

from .vendor.trace_rlvr.answer_extraction import extract_answer
from .vendor.trace_rlvr.reward_scoring import _canonical_jsonable, score_trace_response

RL_FORMAT_WEIGHT = 0.05


def semantic_correct(candidate: Any, answer_gt: dict[str, Any]) -> bool:
    return _canonical_jsonable(candidate) == _canonical_jsonable(answer_gt["value"])


def strict_score(response: str, answer_gt: dict[str, Any], annotation_gt: dict[str, Any], reward_contract: dict[str, Any]) -> dict[str, float]:
    result = score_trace_response(
        response=response,
        answer_gt=answer_gt,
        annotation_gt=annotation_gt,
        reward_contract=reward_contract,
        trace_reward_mode="answer",
        trace_answer_scoring="exact_json",
        answer_weight=1.0,
        annotation_weight=0.0,
        format_weight=RL_FORMAT_WEIGHT,
    )
    return {
        "strict_correct": float(result["accuracy"]),
        "strict_format": float(result["format"]),
        "strict_reward": float(result["overall"]),
    }


def official_deterministic(response: str, answer_gt: dict[str, Any]) -> dict[str, Any]:
    extraction = extract_answer(response, answer_type=str(answer_gt["type"]))
    found = extraction.status == "found"
    return {
        "extraction_status": extraction.status,
        "extraction_route": extraction.route,
        "extracted_answer": extraction.typed_candidate,
        "deterministic_correct": float(found and semantic_correct(extraction.typed_candidate, answer_gt)),
        "judge_needed": not found,
    }


def score_row(response: str, answer_gt: dict[str, Any], annotation_gt: dict[str, Any], reward_contract: dict[str, Any]) -> dict[str, Any]:
    response = response or ""
    out: dict[str, Any] = {}
    out.update(strict_score(response, answer_gt, annotation_gt, reward_contract))
    out.update(official_deterministic(response, answer_gt))
    return out


def apply_judge(scores: dict[str, Any], judge: dict[str, Any] | None, answer_gt: dict[str, Any]) -> dict[str, Any]:
    """Combine deterministic extraction with a judge result (if any) into ``official_correct``."""
    scores = dict(scores)
    if not scores["judge_needed"]:
        scores["official_correct"] = scores["deterministic_correct"]
        scores["official_source"] = "deterministic"
    elif judge is not None and judge.get("status") == "ok":
        scores["official_correct"] = float(semantic_correct(judge["answer"], answer_gt))
        scores["official_source"] = "judge"
    else:
        scores["official_correct"] = 0.0
        scores["official_source"] = "unresolved" if judge is not None else "judge_pending"
    return scores
