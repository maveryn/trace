"""Pure helpers for object-scene marked-point ordering tasks."""

from __future__ import annotations

import math
from itertools import permutations
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple


ORDER_POINT_LABELS: Tuple[str, ...] = ("P", "Q", "R")
ORDER_OPTION_LABELS: Tuple[str, ...] = tuple("ABCDEF")


def label_order_descriptor(labels: Sequence[str]) -> str:
    return " < ".join(str(label) for label in labels)


def order_option_choices(*, answer_order: Sequence[str]) -> Tuple[List[Dict[str, Any]], str]:
    """Return the six permutation options and the answer option label."""

    answer_descriptor = label_order_descriptor(answer_order)
    choices: List[Dict[str, Any]] = []
    answer_label = ""
    for option_label, labels in zip(ORDER_OPTION_LABELS, permutations(ORDER_POINT_LABELS)):
        descriptor = label_order_descriptor(labels)
        if descriptor == answer_descriptor:
            answer_label = str(option_label)
        choices.append(
            {
                "label": str(option_label),
                "option_id": f"order_option_{option_label}",
                "object_id": f"order_option_{option_label}",
                "descriptor": str(descriptor),
                "object_name": "point order",
                "color_name": None,
            }
        )
    if not answer_label:
        raise ValueError(f"answer order is not a permutation of {ORDER_POINT_LABELS}: {answer_order}")
    return choices, str(answer_label)


def demote_context_spec(spec: Mapping[str, Any], *, index: int, prefix: str) -> Dict[str, Any]:
    """Turn a sampled small candidate object into unlettered context."""

    updated = dict(spec)
    shape_type = str(updated["shape_type"])
    updated["object_id"] = f"{prefix}_context_{int(index):02d}_{shape_type}"
    updated["object_role"] = "small_context"
    updated["is_answer_candidate"] = False
    for key in ("point_id", "point_label", "object_label"):
        updated.pop(key, None)
    return updated


def finalize_object_specs(
    specs: Sequence[Mapping[str, Any]],
    *,
    camera,
    frame,
    project_screen_fn: Callable[..., Sequence[float]],
) -> List[Dict[str, Any]]:
    finalized_specs: List[Dict[str, Any]] = []
    for spec in specs:
        screen = project_screen_fn(spec["world_xyz"], camera, frame)
        finalized = dict(spec)
        finalized.update(
            {
                "screen_xy": [round(float(screen[0]), 3), round(float(screen[1]), 3)],
                "camera_xyz": [
                    round(float(screen[5]), 4),
                    round(float(screen[6]), 4),
                    round(float(screen[4]), 4),
                ],
                "camera_distance": round(float(screen[7]), 4),
            }
        )
        finalized_specs.append(finalized)
    return finalized_specs


def relabel_order_points(records: Sequence[Mapping[str, Any]], *, rng) -> List[Dict[str, Any]]:
    labels = list(ORDER_POINT_LABELS)
    rng.shuffle(labels)
    relabeled: List[Dict[str, Any]] = []
    for record, label in zip(records, labels):
        updated = dict(record)
        updated["point_label"] = str(label)
        updated["point_id"] = f"marked_point_{label}"
        relabeled.append(updated)
    return sorted(relabeled, key=lambda item: str(item["point_label"]))


def point_map_by_label(
    *,
    marker_render_map: Mapping[str, Any],
    labels: Sequence[str] = ORDER_POINT_LABELS,
) -> Dict[str, List[float]]:
    centers = marker_render_map["marked_point_centers_px"]
    return {
        str(label): [round(float(value), 3) for value in centers[str(label)]]
        for label in labels
    }


def bbox_union(*bboxes: Sequence[float]) -> List[float]:
    clean = [list(bbox) for bbox in bboxes if bbox]
    return [
        round(float(min(bbox[0] for bbox in clean)), 3),
        round(float(min(bbox[1] for bbox in clean)), 3),
        round(float(max(bbox[2] for bbox in clean)), 3),
        round(float(max(bbox[3] for bbox in clean)), 3),
    ]


def min_pairwise(values: Sequence[float]) -> float:
    if len(values) < 2:
        return float("inf")
    return min(abs(float(a) - float(b)) for index, a in enumerate(values) for b in values[index + 1 :])


def screen_separation_ok(points: Sequence[Mapping[str, Any]], *, min_px: float) -> bool:
    centers = [(float(point["screen_xy"][0]), float(point["screen_xy"][1])) for point in points]
    return all(
        math.hypot(float(a[0] - b[0]), float(a[1] - b[1])) >= float(min_px)
        for index, a in enumerate(centers)
        for b in centers[index + 1 :]
    )


__all__ = [
    "ORDER_OPTION_LABELS",
    "ORDER_POINT_LABELS",
    "bbox_union",
    "demote_context_spec",
    "finalize_object_specs",
    "label_order_descriptor",
    "min_pairwise",
    "order_option_choices",
    "point_map_by_label",
    "relabel_order_points",
    "screen_separation_ok",
]
