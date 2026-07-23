"""Prompt assembly for triangle-relations scene-package tasks."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.shared.prompt_json_example import dump_prompt_json_examples
from trace_tasks.tasks.shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)

from .state import DOMAIN, PROMPT_BUNDLE_ID, SCENE_ID, SCENE_PROMPT_KEY


_CONSTRUCTION_DESCRIPTION_BY_TASK_KEY = {
    "angle_bisector_segment_value_query": "In the triangle, AD bisects angle BAC.",
    "angle_bisector_variable_value_query": "In the triangle, AD bisects angle BAC.",
    "altitude_to_hypotenuse_value_query": "A right triangle has an altitude drawn to the hypotenuse.",
    "centroid_median_segment_value_query": "In the triangle, G is the centroid and D is the midpoint of BC.",
    "parallel_section_segment_value_query": "In the triangle, DE is parallel to BC.",
    "pythagorean_length_value_chained_rectangle_diagonal_length_query": "A rectangle is split by a vertical segment; AE, EB, the left diagonal DE, and the whole diagonal DB are labeled.",
    "pythagorean_length_value_rectangle_triangle_shared_height_length_query": "A rectangle and a right triangle share the same height.",
    "leg_projection_length_value_query": "A right triangle has an altitude drawn to the hypotenuse.",
    "right_triangle_missing_side_value_query": "A right triangle has one marked angle and labeled side lengths.",
    "similar_triangles_side_length_query": "In the triangle, DE is parallel to BC.",
    "split_triangle_angle_value_query": "A split triangle has marked angles.",
    "split_triangle_trig_side_length_value_query": "A split triangle has standard marked angles and labeled segments.",
}


def _construction_description(task_prompt_key: str) -> str:
    try:
        return _CONSTRUCTION_DESCRIPTION_BY_TASK_KEY[str(task_prompt_key)]
    except KeyError as exc:
        raise ValueError(f"missing construction description for triangle-relations prompt key: {task_prompt_key}") from exc


def _examples(annotation_mode: str, answer: Any, annotation_roles: tuple[str, ...]) -> tuple[str, str]:
    if annotation_mode == "segment":
        annotation: Any = [[120, 260], [280, 260]]
    elif annotation_mode == "point":
        annotation = [180, 320]
    else:
        roles = tuple(annotation_roles) or ("A", "B", "C", "D")
        annotation = {
            str(role): [160 + 28 * index, 160 + 20 * (index % 3)]
            for index, role in enumerate(roles)
        }
    return dump_prompt_json_examples(annotation=annotation, answer=answer, ensure_ascii=False)


def _annotation_instruction(annotation_mode: str, annotation_roles: tuple[str, ...]) -> str:
    if annotation_mode == "segment":
        return 'set "annotation" to the target visual segment as [[x0,y0],[x1,y1]]'
    if annotation_mode == "point":
        return 'set "annotation" to the target angle vertex as [x,y]'
    keys = ", ".join(f'"{key}"' for key in annotation_roles)
    return f'set "annotation" to a JSON object with exactly these point keys: {keys}; each value is [x,y]'


def build_triangle_relations_prompt_artifacts(
    *,
    prompt_defaults: Mapping[str, Any],
    task_prompt_key: str,
    prompt_branch_key: str,
    annotation_mode: str,
    annotation_roles: tuple[str, ...],
    answer_value: Any,
    target_name: str,
    instance_seed: int,
):
    """Render v1 prompt variants for one triangle-relations objective."""

    json_example, json_example_answer_only = _examples(str(annotation_mode), answer_value, tuple(annotation_roles))
    prompt_selection = render_scene_prompt_variants(
        domain=DOMAIN,
        scene_id=SCENE_ID,
        bundle_id=str(prompt_defaults.get("bundle_id", PROMPT_BUNDLE_ID)),
        scene_key=str(prompt_defaults.get("scene_key", SCENE_PROMPT_KEY)),
        task_key=str(task_prompt_key),
        query_key=str(prompt_branch_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots={
            "annotation_instruction": _annotation_instruction(str(annotation_mode), tuple(annotation_roles)),
            "construction_description": _construction_description(str(task_prompt_key)),
            "json_example": str(json_example),
            "json_example_answer_only": str(json_example_answer_only),
            "target_name": str(target_name),
        },
        instance_seed=int(instance_seed),
    )
    return build_prompt_trace_artifacts(prompt_selection)


__all__ = ["build_triangle_relations_prompt_artifacts"]
