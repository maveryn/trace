"""Shared support for controlled-unanswerable task queries."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from trace_tasks.core.seed import spawn_rng


UNANSWERABLE_ANSWER = "unanswerable"


def should_use_unanswerable_branch(
    params: Mapping[str, Any],
    *,
    instance_seed: int,
    namespace: str,
    enabled: bool,
    default_probability: float = 0.15,
) -> bool:
    """Return true when this sampled instance should be controlled-unanswerable.

    Finite-support sampling may pass ``_sample_cursor``; use a small
    deterministic cycle in that case to keep the answer distribution stable.
    Direct generation stays answerable unless explicitly forced.
    """

    if not bool(enabled):
        return False
    forced = params.get("force_unanswerable")
    if forced is not None:
        return bool(forced)

    probability = float(params.get("unanswerable_probability", default_probability))
    probability = max(0.0, min(1.0, float(probability)))
    if probability <= 0.0:
        return False

    cursor = params.get("_sample_cursor")
    if cursor is None:
        return False

    period = max(1, int(params.get("unanswerable_cycle_period", 20)))
    positives = int(round(float(probability) * float(period)))
    positives = max(0, min(int(period), int(positives)))
    if positives <= 0:
        return False

    offset_rng = spawn_rng(int(instance_seed), f"{namespace}.unanswerable.cycle_offset")
    offset = int(offset_rng.randrange(int(period)))
    return ((abs(int(cursor)) + int(offset)) % int(period)) < int(positives)


def choose_missing_label(
    *,
    visible_labels: Sequence[str],
    candidate_labels: Sequence[str],
    fallback_prefix: str,
    instance_seed: int,
    namespace: str,
) -> str:
    """Choose one plausible label that is not visible in the rendered scene."""

    visible = {str(label) for label in visible_labels}
    candidates = [str(label) for label in candidate_labels if str(label) not in visible]
    if not candidates:
        base_index = len(visible) + 1
        candidates = [f"{fallback_prefix}{base_index + offset}" for offset in range(8)]
    rng = spawn_rng(int(instance_seed), f"{namespace}.missing_label")
    return str(candidates[int(rng.randrange(len(candidates)))])


def absence_proof(
    *,
    requested_item: str,
    visible_candidates: Sequence[str],
    checked_scope: str,
    absence_reason: str,
) -> Dict[str, Any]:
    """Build a proof-of-absence payload for trace metadata."""

    return {
        "answerability": "unanswerable",
        "requested_item": str(requested_item),
        "visible_candidate_set": [str(value) for value in visible_candidates],
        "checked_scope": str(checked_scope),
        "absence_reason": str(absence_reason),
    }


__all__ = [
    "UNANSWERABLE_ANSWER",
    "absence_proof",
    "choose_missing_label",
    "should_use_unanswerable_branch",
]
