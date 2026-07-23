"""Annotation and trace projection helpers for the library scene."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from ...shared.object_rendering import make_vector_scene_object_record
from .state import RenderedLibraryScene


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return round(float(value), 3)
    return value


def book_bbox_map(scene: RenderedLibraryScene) -> Dict[str, List[float]]:
    """Return book bboxes keyed by book id."""

    return {str(book.book_id): [round(float(v), 3) for v in book.bbox_xyxy] for book in scene.books}


def book_point_map(scene: RenderedLibraryScene) -> Dict[str, List[float]]:
    """Return book center points keyed by book id."""

    points: Dict[str, List[float]] = {}
    for book in scene.books:
        x0, y0, x1, y1 = [float(value) for value in book.bbox_xyxy]
        points[str(book.book_id)] = [
            round(0.5 * (x0 + x1), 3),
            round(0.5 * (y0 + y1), 3),
        ]
    return points


def section_bbox_map(scene: RenderedLibraryScene) -> Dict[str, List[float]]:
    """Return section bboxes keyed by section id."""

    return {str(section.section_id): [round(float(v), 3) for v in section.bbox_xyxy] for section in scene.sections}


def sort_library_bboxes(bbox_map: Mapping[str, Sequence[float]], ids: Iterable[str]) -> List[List[float]]:
    """Return bboxes sorted top-to-bottom then left-to-right for stable annotation."""

    boxes = [list(float(v) for v in bbox_map[str(item_id)]) for item_id in ids]
    boxes.sort(key=lambda box: (round(float(box[1]), 3), round(float(box[0]), 3), round(float(box[3]), 3), round(float(box[2]), 3)))
    return [[round(float(v), 3) for v in box] for box in boxes]


def sort_library_points(bbox_map: Mapping[str, Sequence[float]], ids: Iterable[str]) -> List[List[float]]:
    """Return bbox center points sorted top-to-bottom then left-to-right."""

    boxes = [list(float(v) for v in bbox_map[str(item_id)]) for item_id in ids]
    boxes.sort(key=lambda box: (round(float(box[1]), 3), round(float(box[0]), 3), round(float(box[3]), 3), round(float(box[2]), 3)))
    return [
        [
            round(0.5 * (float(box[0]) + float(box[2])), 3),
            round(0.5 * (float(box[1]) + float(box[3])), 3),
        ]
        for box in boxes
    ]


def library_scene_entities(scene: RenderedLibraryScene) -> List[Dict[str, Any]]:
    """Return generic entity records for the scene trace."""

    entities: List[Dict[str, Any]] = []
    for section in scene.sections:
        object_record = make_vector_scene_object_record(
            object_id=str(section.section_id),
            object_type="library_section",
            bbox_xyxy=section.bbox_xyxy,
            semantic_attributes={
                "section_key": str(section.section_key),
                "section_name": str(section.section_name),
                "label": str(section.label),
                "book_ids": list(section.book_ids),
                **dict(section.attributes),
            },
            role=str(section.role),
            source_entity_type="library_section",
            render_scale=int(scene.render_scale),
            style_id=str(scene.style_id),
        )
        entities.append(
            {
                "entity_id": str(section.section_id),
                "entity_type": "library_section",
                "section_key": str(section.section_key),
                "section_name": str(section.section_name),
                "label": str(section.label),
                "bbox": [round(float(v), 3) for v in section.bbox_xyxy],
                "label_bbox": [round(float(v), 3) for v in section.label_bbox_xyxy],
                "book_ids": list(section.book_ids),
                "attributes": _json_safe(section.attributes),
                "object_record": object_record,
            }
        )
    for book in scene.books:
        object_record = make_vector_scene_object_record(
            object_id=str(book.book_id),
            object_type="book",
            bbox_xyxy=book.bbox_xyxy,
            semantic_attributes={
                "section_id": str(book.section_id),
                "section_key": str(book.section_key),
                "section_name": str(book.section_name),
                "color_name": str(book.color_name),
                "color_label": str(book.color_label),
                "orientation": str(book.orientation),
                **dict(book.attributes),
            },
            visual_attributes={"color_rgb": [int(v) for v in book.color_rgb]},
            role=str(book.role),
            source_entity_type="library_book",
            render_scale=int(scene.render_scale),
            style_id=str(scene.style_id),
        )
        entities.append(
            {
                "entity_id": str(book.book_id),
                "entity_type": "library_book",
                "section_id": str(book.section_id),
                "section_key": str(book.section_key),
                "section_name": str(book.section_name),
                "color_name": str(book.color_name),
                "color_rgb": [int(v) for v in book.color_rgb],
                "color_label": str(book.color_label),
                "orientation": str(book.orientation),
                "bbox": [round(float(v), 3) for v in book.bbox_xyxy],
                "role": str(book.role),
                "attributes": _json_safe(book.attributes),
                "object_record": object_record,
            }
        )
    for item in scene.decor:
        decor_attributes = dict(item.attributes)
        object_record = (
            dict(item.object_record)
            if item.object_record is not None
            else make_vector_scene_object_record(
                object_id=str(item.decor_id),
                object_type=str(item.decor_type),
                bbox_xyxy=item.bbox_xyxy,
                semantic_attributes={
                    "decor_type": str(item.decor_type),
                    **{key: value for key, value in decor_attributes.items() if key != "gender_id"},
                },
                visual_attributes={
                    key: decor_attributes[key]
                    for key in ("gender_id",)
                    if key in decor_attributes
                },
                role=str(decor_attributes.get("role", "distractor")),
                source_entity_type="library_decor",
                render_scale=int(scene.render_scale),
                style_id=str(scene.style_id),
            )
        )
        entities.append(
            {
                "entity_id": str(item.decor_id),
                "entity_type": "library_decor",
                "decor_type": str(item.decor_type),
                "bbox": [round(float(v), 3) for v in item.bbox_xyxy],
                "attributes": _json_safe(item.attributes),
                "object_record": object_record,
            }
        )
    return entities


def serialize_library_scene(scene: RenderedLibraryScene) -> Tuple[List[Dict[str, Any]], Dict[str, List[float]], Dict[str, List[float]]]:
    """Serialize scene records and bbox maps for trace payloads."""

    section_records = [
        {
            "section_id": str(section.section_id),
            "section_key": str(section.section_key),
            "section_name": str(section.section_name),
            "label": str(section.label),
            "bbox": [round(float(v), 3) for v in section.bbox_xyxy],
            "label_bbox": [round(float(v), 3) for v in section.label_bbox_xyxy],
            "shelf_bboxes": [[round(float(v), 3) for v in shelf] for shelf in section.shelf_bboxes_xyxy],
            "book_ids": list(section.book_ids),
            "role": str(section.role),
            "attributes": _json_safe(section.attributes),
        }
        for section in scene.sections
    ]
    book_records = [
        {
            "book_id": str(book.book_id),
            "section_id": str(book.section_id),
            "section_key": str(book.section_key),
            "section_name": str(book.section_name),
            "color_name": str(book.color_name),
            "color_rgb": [int(v) for v in book.color_rgb],
            "color_label": str(book.color_label),
            "orientation": str(book.orientation),
            "bbox": [round(float(v), 3) for v in book.bbox_xyxy],
            "row_index": int(book.row_index),
            "slot_index": int(book.slot_index),
            "role": str(book.role),
            "attributes": _json_safe(book.attributes),
        }
        for book in scene.books
    ]
    decor_records = [
        {
            "decor_id": str(item.decor_id),
            "decor_type": str(item.decor_type),
            "bbox": [round(float(v), 3) for v in item.bbox_xyxy],
            "attributes": _json_safe(item.attributes),
        }
        for item in scene.decor
    ]
    return (
        [
            {
                "setting_id": str(scene.setting_id),
                "sections": section_records,
                "books": book_records,
                "decor": decor_records,
            }
        ],
        book_bbox_map(scene),
        section_bbox_map(scene),
    )


__all__ = [
    "book_bbox_map",
    "book_point_map",
    "library_scene_entities",
    "section_bbox_map",
    "serialize_library_scene",
    "sort_library_bboxes",
    "sort_library_points",
]
