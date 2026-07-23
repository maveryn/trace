"""Shared marked-point overlay helpers for 3D spatial tasks."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

from ...shared.object_scene import POINT_LABELS


def bbox_union(*bboxes: Sequence[float]) -> List[float]:
    return [
        round(float(min(float(bbox[0]) for bbox in bboxes)), 3),
        round(float(min(float(bbox[1]) for bbox in bboxes)), 3),
        round(float(max(float(bbox[2]) for bbox in bboxes)), 3),
        round(float(max(float(bbox[3]) for bbox in bboxes)), 3),
    ]


def assign_answer_label(
    *,
    records: Sequence[Mapping[str, Any]],
    answer_marker_id: str,
    point_count: int,
    answer_label_index: int,
    rng,
) -> List[Dict[str, Any]]:
    answer_label = str(POINT_LABELS[abs(int(answer_label_index)) % int(point_count)])
    remaining_labels = [str(label) for label in POINT_LABELS[: int(point_count)] if str(label) != answer_label]
    rng.shuffle(remaining_labels)
    finalized: List[Dict[str, Any]] = []
    for record in records:
        updated = dict(record)
        label = answer_label if str(record["marker_id"]) == str(answer_marker_id) else str(remaining_labels.pop())
        updated["point_label"] = str(label)
        updated["point_id"] = f"marked_point_{label}"
        finalized.append(updated)
    return sorted(finalized, key=lambda item: str(item["point_label"]))


__all__ = ["assign_answer_label", "bbox_union"]
