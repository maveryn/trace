"""Trace payload assembly helpers for counterfactual-board games."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from .state import CountedElement, CounterfactualBoardCase, RenderedCounterfactualBoard, SCENE_ID


def json_ready(value: Any) -> Any:
    """Convert nested tuples and mappings into JSON-friendly containers."""

    if isinstance(value, Mapping):
        return {str(key): json_ready(inner) for key, inner in value.items()}
    if isinstance(value, tuple):
        return [json_ready(inner) for inner in value]
    if isinstance(value, list):
        return [json_ready(inner) for inner in value]
    return value


def _segment_record(element: CountedElement) -> list[list[float]]:
    """Return a rounded JSON-ready segment for one line witness."""

    if element.segment is None:
        raise ValueError(f"counted element {element.element_id!r} has no segment")
    return [
        [
            round(float(element.segment[0][0]), 3),
            round(float(element.segment[0][1]), 3),
        ],
        [
            round(float(element.segment[1][0]), 3),
            round(float(element.segment[1][1]), 3),
        ],
    ]


def build_counterfactual_board_trace_payload(
    *,
    case: CounterfactualBoardCase,
    rendered: RenderedCounterfactualBoard,
    counted_elements: Sequence[CountedElement],
    annotation_artifacts: Any,
    prompt_defaults: Mapping[str, Any],
    prompt_artifacts: Any,
    query_spec: Mapping[str, Any],
) -> Dict[str, Any]:
    """Assemble trace payload after task-owned answer and annotation binding."""

    annotation_source = (
        "counted_element_segments_px"
        if str(annotation_artifacts.annotation_type) == "segment_set"
        else "counted_element_bboxes_px"
    )
    element_records = [
        {
            "entity_id": str(element.element_id),
            "entity_type": str(element.element_kind),
            "bbox": [round(float(value), 3) for value in element.bbox],
            "segment": (
                _segment_record(element)
                if element.segment is not None
                else None
            ),
            "reading_order_index": int(index),
        }
        for index, element in enumerate(counted_elements)
    ]
    execution = {
        "scene_id": SCENE_ID,
        "board_style": str(case.style),
        "board_kind": str(case.board_kind),
        "visible_rows": int(case.rows),
        "visible_columns": int(case.cols),
        "counted_axis": str(case.counted_axis),
        "prompt_query_key": str(case.prompt_query_key),
        "answer_value": int(case.answer_value),
        "canonical_bias_answer": int(case.canonical_bias_answer),
        "counterfactual_delta": int(case.answer_value) - int(case.canonical_bias_answer),
        "counted_element_ids": [str(element.element_id) for element in counted_elements],
        "counted_element_bboxes_px": [
            [round(float(value), 3) for value in element.bbox]
            for element in counted_elements
        ],
        "counted_element_segments_px": [
            _segment_record(element)
            for element in counted_elements
            if element.segment is not None
        ],
        "supporting_item_ids": [str(element.element_id) for element in counted_elements],
        **json_ready(case.execution_trace),
    }
    return {
        "scene_ir": {
            "scene_kind": SCENE_ID,
            "entities": [dict(entity) for entity in rendered.entities] + element_records,
            "relations": {
                "scene_id": SCENE_ID,
                "board_style": str(case.style),
                "board_kind": str(case.board_kind),
                "visible_rows": int(case.rows),
                "visible_columns": int(case.cols),
                "target_answer": int(case.answer_value),
            },
        },
        "query_spec": dict(query_spec),
        "render_spec": dict(rendered.render_meta),
        "render_map": {
            "image_id": "img0",
            "board_bbox_px": list(rendered.layout.board_bbox_px),
            "counted_element_bboxes_px": {
                str(element.element_id): [
                    round(float(value), 3) for value in element.bbox
                ]
                for element in counted_elements
            },
            "counted_element_segments_px": {
                str(element.element_id): _segment_record(element)
                for element in counted_elements
                if element.segment is not None
            },
            "counted_element_ids": [
                str(element.element_id) for element in counted_elements
            ],
            "annotation_source": annotation_source,
        },
        "execution_trace": execution,
        "witness_symbolic": {
            "type": str(annotation_artifacts.annotation_type),
            "value": json_ready(annotation_artifacts.value),
            "counted_element_ids": [
                str(element.element_id) for element in counted_elements
            ],
            "visible_element_count": int(case.answer_value),
            "canonical_bias_answer": int(case.canonical_bias_answer),
        },
        "projected_annotation": dict(annotation_artifacts.projected_annotation),
        "prompt_spec": {
            "defaults": dict(prompt_defaults),
            "active": dict(prompt_artifacts.prompt_variant),
        },
        "background": dict(rendered.background_meta),
        "post_image_noise": dict(rendered.post_noise_meta),
    }


__all__ = ["build_counterfactual_board_trace_payload", "json_ready"]
