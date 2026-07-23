"""Trace payload assembly helpers for cell-board puzzles."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from .state import CellBoardCase, RenderedCellBoard, SCENE_ID
from .topology import cell_id


def json_ready(value: Any) -> Any:
    """Convert nested tuples and mappings into JSON-friendly values."""

    if isinstance(value, Mapping):
        return {str(key): json_ready(inner) for key, inner in value.items()}
    if isinstance(value, tuple):
        return [json_ready(inner) for inner in value]
    if isinstance(value, list):
        return [json_ready(inner) for inner in value]
    return value


def build_cell_board_trace_payload(
    *,
    case: CellBoardCase,
    rendered: RenderedCellBoard,
    annotation_artifacts: Any,
    prompt_defaults: Mapping[str, Any],
    prompt_artifacts: Any,
    query_spec: Mapping[str, Any],
    execution_extra: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Assemble trace payload after task-owned answer and annotation binding."""

    color_grid = [
        [
            str(case.board_colors[(row, col)][0])
            for col in range(int(case.cols))
        ]
        for row in range(int(case.rows))
    ]
    execution = {
        "scene_id": SCENE_ID,
        "rows": int(case.rows),
        "cols": int(case.cols),
        "answer_value": int(case.answer_value),
        "annotation_kind": str(case.annotation_kind),
        "prompt_task_key": str(case.prompt_task_key),
        "prompt_query_key": str(case.prompt_query_key),
        "color_grid": color_grid,
        "annotation_coords": [
            [int(row), int(col)] for row, col in case.annotation_coords
        ],
        "annotation_path": [
            [int(row), int(col)] for row, col in case.annotation_path
        ],
        "annotation_coord_pairs": [
            [[int(row), int(col)] for row, col in pair]
            for pair in case.annotation_coord_pairs
        ],
        **json_ready(case.execution_trace),
    }
    if execution_extra:
        execution.update(json_ready(dict(execution_extra)))

    return {
        "scene_ir": {
            "scene_kind": SCENE_ID,
            "entities": [dict(entity) for entity in rendered.entities],
            "relations": {
                "scene_id": SCENE_ID,
                "rows": int(case.rows),
                "cols": int(case.cols),
            },
        },
        "query_spec": dict(query_spec),
        "render_spec": dict(rendered.render_meta),
        "render_map": {
            "cell_bboxes_px": {
                cell_id((row, col)): [
                    round(float(value), 3)
                    for value in rendered.bbox_map[cell_id((row, col))]
                ]
                for row in range(int(case.rows))
                for col in range(int(case.cols))
            },
            "annotation_source": str(annotation_artifacts.annotation_type),
        },
        "execution_trace": execution,
        "witness_symbolic": {
            "type": str(annotation_artifacts.annotation_type),
            "value": json_ready(annotation_artifacts.value),
        },
        "projected_annotation": dict(annotation_artifacts.projected_annotation),
        "prompt_spec": {
            "defaults": dict(prompt_defaults),
            "active": dict(prompt_artifacts.prompt_variant),
        },
        "background": dict(rendered.background_meta),
        "post_image_noise": dict(rendered.post_noise_meta),
    }


__all__ = [
    "build_cell_board_trace_payload",
    "json_ready",
]
