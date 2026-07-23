"""Trace payload helpers for Sudoku task outputs."""

from __future__ import annotations

from typing import Any, Mapping

from PIL import Image

from trace_tasks.tasks.shared.font_assets import get_font_family_record
from trace_tasks.tasks.shared.prompt_variants import PromptTraceArtifacts

from .rendering import RenderedSudokuScene
from .rules import candidate_digits, coord_to_cell_id, unit_coords
from .state import SCENE_ID, SudokuSample


def _answer_value(answer: int | str) -> int | str:
    """Return a JSON-stable Sudoku answer value."""

    return int(answer) if isinstance(answer, int) else str(answer)


def text_style_metadata(font_family: str) -> dict[str, Any]:
    """Return trace metadata for the active Sudoku digit font."""

    return {
        "font_family": str(font_family),
        "font_asset": get_font_family_record(str(font_family)).to_trace(),
    }


def projected_annotation_payload(
    annotation_type: str,
    annotation_value: Any,
) -> dict[str, Any]:
    """Build the normalized projected-annotation trace payload."""

    kind = str(annotation_type)
    payload = {"type": kind}
    payload[kind] = annotation_value
    payload[f"pixel_{kind}"] = annotation_value
    return payload


def build_sudoku_trace_payload(
    *,
    sample: SudokuSample,
    rendered_scene: RenderedSudokuScene,
    image: Image.Image,
    prompt_artifacts: PromptTraceArtifacts,
    prompt_spec_payload: Mapping[str, Any],
    execution_fields: Mapping[str, Any],
    annotation_type: str,
    annotation_value: Any,
    annotation_entity_ids: Any,
    scene_variant: str,
    style_variant: str,
    panel_style_meta: Mapping[str, Any],
    text_style_meta: Mapping[str, Any],
    background_meta: Mapping[str, Any],
    post_noise_meta: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble Sudoku scene trace metadata from task-owned bindings."""

    highlighted_coords = (
        unit_coords(sample.highlighted_unit_type, int(sample.highlighted_unit_index))
        if sample.highlighted_unit_type is not None
        and sample.highlighted_unit_index is not None
        else ()
    )
    candidate_values = (
        candidate_digits(sample.board, sample.marked_cell)
        if sample.marked_cell is not None
        else ()
    )
    answer_value = _answer_value(sample.answer)
    return {
        "scene_ir": {
            "scene_kind": f"puzzle_sudoku_grid_{str(scene_variant)}",
            "entities": [dict(entity) for entity in rendered_scene.scene_entities],
            "relations": {
                "scene_variant": str(scene_variant),
                "unit_type": sample.highlighted_unit_type,
                "unit_index": sample.highlighted_unit_index,
                "style_variant": str(style_variant),
                "target_answer": answer_value,
                "visible_count": int(sample.visible_count),
                "annotation_entity_ids": annotation_entity_ids,
                "correct_option_label": sample.correct_option_label,
            },
        },
        "query_spec": dict(prompt_spec_payload),
        "render_spec": {
            "scene_variant": str(scene_variant),
            "style_variant": str(style_variant),
            "canvas_width": int(image.size[0]),
            "canvas_height": int(image.size[1]),
            "layout_jitter": dict(rendered_scene.render_map.get("layout_jitter", {})),
            "panel_scene_style": dict(panel_style_meta),
            "text_style": dict(text_style_meta),
        },
        "render_map": dict(rendered_scene.render_map),
        "execution_trace": {
            "scene_variant": str(scene_variant),
            "style_variant": str(style_variant),
            "target_answer": answer_value,
            "visible_count": int(sample.visible_count),
            "construction_mode": str(sample.construction_mode),
            "board_rows": [[int(cell) for cell in row] for row in sample.board],
            "solution_rows": [[int(cell) for cell in row] for row in sample.solution],
            "marked_cell": (
                list(sample.marked_cell) if sample.marked_cell is not None else None
            ),
            "candidate_digit_values": [int(value) for value in candidate_values],
            "highlighted_unit_type": sample.highlighted_unit_type,
            "highlighted_unit_index": sample.highlighted_unit_index,
            "highlighted_unit_coords": [
                [int(row), int(col)] for row, col in highlighted_coords
            ],
            "missing_digit_values": [
                int(value) for value in sample.missing_digit_values
            ],
            "repeated_digit_values": [
                int(value) for value in sample.repeated_digit_values
            ],
            "target_digit": sample.target_digit,
            "option_specs": [dict(spec) for spec in sample.option_specs],
            "correct_option_label": sample.correct_option_label,
            "annotation_coords": [
                [int(row), int(col)] for row, col in sample.annotation_coords
            ],
            "annotation_entity_ids": annotation_entity_ids,
            "marked_cell_id": (
                coord_to_cell_id(sample.marked_cell)
                if sample.marked_cell is not None
                else None
            ),
            **dict(execution_fields),
        },
        "witness_symbolic": {
            "type": "sudoku_cell_annotation",
            "ids": annotation_entity_ids,
        },
        "projected_annotation": projected_annotation_payload(
            str(annotation_type),
            annotation_value,
        ),
        "background": background_meta,
        "post_image_noise": post_noise_meta,
    }


__all__ = [
    "build_sudoku_trace_payload",
    "projected_annotation_payload",
    "text_style_metadata",
]
