"""Annotation and trace payload helpers for category-grid scenes."""

from __future__ import annotations

from typing import Any, Dict, List

from .sampling import ordinal_label
from .state import CategoryGridCase, CategoryGridSpec, RenderedCategoryGrid


def make_category_payload(rendered: RenderedCategoryGrid, spec: CategoryGridSpec) -> List[Dict[str, Any]]:
    """Return category/subcategory/item trace records with projected boxes."""

    payload: List[Dict[str, Any]] = []
    for category in spec.categories:
        category_payload = {
            "category_id": str(category.category_id),
            "category_label": str(category.label),
            "category_header_bbox_px": [
                float(value) for value in rendered.category_header_bboxes_px[str(category.category_id)]
            ],
            "subcategories": [],
        }
        for subcategory in category.subcategories:
            subcategory_payload = {
                "subcategory_id": str(subcategory.subcategory_id),
                "subcategory_label": str(subcategory.label),
                "subcategory_header_bbox_px": [
                    float(value)
                    for value in rendered.subcategory_header_bboxes_px[str(category.category_id)][str(subcategory.subcategory_id)]
                ],
                "item_count": int(len(subcategory.items)),
                "items": [],
            }
            for item_index, item in enumerate(subcategory.items):
                subcategory_payload["items"].append(
                    {
                        "item_id": str(item.item_id),
                        "item_label": str(item.label),
                        "slot_index": int(item_index) + 1,
                        "slot_ordinal": ordinal_label(int(item_index) + 1),
                        "item_row_bbox_px": [
                            float(value)
                            for value in rendered.item_row_bboxes_px[str(category.category_id)][str(subcategory.subcategory_id)][str(item.item_id)]
                        ],
                        "item_label_bbox_px": [
                            float(value)
                            for value in rendered.item_label_bboxes_px[str(category.category_id)][str(subcategory.subcategory_id)][str(item.item_id)]
                        ],
                    }
                )
            category_payload["subcategories"].append(subcategory_payload)
        payload.append(category_payload)
    return payload


def target_payload_for_slot(case: CategoryGridCase) -> Dict[str, Any]:
    """Return task target metadata for an ordinal item lookup."""

    if case.target_item is None or case.target_slot_index is None:
        raise ValueError("slot item case requires a selected item")
    return {
        "category_id": str(case.target_category.category_id),
        "category_label": str(case.target_category.label),
        "subcategory_id": str(case.target_subcategory.subcategory_id),
        "subcategory_label": str(case.target_subcategory.label),
        "slot_index": int(case.target_slot_index) + 1,
        "slot_ordinal": ordinal_label(int(case.target_slot_index) + 1),
        "item_id": str(case.target_item.item_id),
        "item_label": str(case.target_item.label),
    }


def target_payload_for_count(case: CategoryGridCase) -> Dict[str, Any]:
    """Return task target metadata for a category/subcategory count."""

    return {
        "category_id": str(case.target_category.category_id),
        "category_label": str(case.target_category.label),
        "subcategory_id": str(case.target_subcategory.subcategory_id),
        "subcategory_label": str(case.target_subcategory.label),
        "item_count": int(len(case.target_subcategory.items)),
        "item_ids": [str(item.item_id) for item in case.target_subcategory.items],
    }


def slot_item_annotation(case: CategoryGridCase, rendered: RenderedCategoryGrid) -> Dict[str, List[float]]:
    """Return keyed boxes for a slot item lookup."""

    if case.target_item is None:
        raise ValueError("slot item annotation requires a selected item")
    category_id = str(case.target_category.category_id)
    subcategory_id = str(case.target_subcategory.subcategory_id)
    item_id = str(case.target_item.item_id)
    return {
        "category_header": [float(value) for value in rendered.category_header_bboxes_px[category_id]],
        "subcategory_header": [
            float(value) for value in rendered.subcategory_header_bboxes_px[category_id][subcategory_id]
        ],
        "target_item": [
            float(value)
            for value in rendered.item_row_bboxes_px[category_id][subcategory_id][item_id]
        ],
    }


def item_count_annotation(case: CategoryGridCase, rendered: RenderedCategoryGrid) -> List[List[float]]:
    """Return item row boxes for the target category/subcategory block."""

    category_id = str(case.target_category.category_id)
    subcategory_id = str(case.target_subcategory.subcategory_id)
    return [
        [
            float(value)
            for value in rendered.item_row_bboxes_px[category_id][subcategory_id][str(item.item_id)]
        ]
        for item in case.target_subcategory.items
    ]
