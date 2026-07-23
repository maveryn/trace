"""Annotation binding helpers for profile-card-grid page scenes."""

from __future__ import annotations

from typing import Dict, List, Mapping

from .state import ProfileCard, RenderedProfileCardGrid


def card_bbox(
    *,
    card: ProfileCard,
    rendered: RenderedProfileCardGrid,
) -> List[float]:
    """Return the selected profile card bbox."""

    return [float(value) for value in rendered.card_bboxes_px[str(card.profile_id)]]


def field_value_bbox(
    *,
    card: ProfileCard,
    field_label: str,
    rendered: RenderedProfileCardGrid,
) -> List[float]:
    """Return the selected field-value bbox."""

    return [float(value) for value in rendered.field_value_bboxes_px[str(card.profile_id)][str(field_label)]]


def lookup_supporting_bboxes(
    *,
    card: ProfileCard,
    field_label: str,
    rendered: RenderedProfileCardGrid,
) -> Dict[str, List[float]]:
    """Return trace-only role boxes for direct profile-field lookup tasks."""

    profile_id = str(card.profile_id)
    field = str(field_label)
    return {
        "profile_name": [float(value) for value in rendered.name_bboxes_px[profile_id]],
        "field_label": [float(value) for value in rendered.field_label_bboxes_px[profile_id][field]],
        "field_value": [float(value) for value in rendered.field_value_bboxes_px[profile_id][field]],
    }


def ordering_supporting_bboxes(
    *,
    card: ProfileCard,
    field_label: str,
    rendered: RenderedProfileCardGrid,
) -> Dict[str, List[float]]:
    """Return trace-only role boxes for numeric ordering tasks."""

    profile_id = str(card.profile_id)
    field = str(field_label)
    return {
        "target_profile": [float(value) for value in rendered.name_bboxes_px[profile_id]],
        "field_label": [float(value) for value in rendered.field_label_bboxes_px[profile_id][field]],
        "target_value": [float(value) for value in rendered.field_value_bboxes_px[profile_id][field]],
    }


def augment_numeric_candidates_with_boxes(
    *,
    candidates: list[Mapping[str, object]],
    rendered: RenderedProfileCardGrid,
    field_label: str,
) -> list[dict]:
    """Attach projected name/field/value boxes to ordered numeric candidates."""

    field = str(field_label)
    resolved: list[dict] = []
    for candidate in candidates:
        profile_id = str(candidate["profile_id"])
        resolved.append(
            {
                **dict(candidate),
                "field_label_bbox_px": [float(value) for value in rendered.field_label_bboxes_px[profile_id][field]],
                "field_value_bbox_px": [float(value) for value in rendered.field_value_bboxes_px[profile_id][field]],
                "profile_name_bbox_px": [float(value) for value in rendered.name_bboxes_px[profile_id]],
            }
        )
    return resolved
