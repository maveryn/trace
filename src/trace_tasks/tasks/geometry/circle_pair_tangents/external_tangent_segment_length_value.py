"""Solve either segment length in an external common tangent diagram."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import split_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions

from ._lifecycle import (
    PairTangentValueProblem,
    build_pair_tangent_value_problem,
    pair_tangent_trace_payload,
    prepare_pair_tangent_scene,
)
from .shared.construction import TANGENT_CASES, select_tangent_layout
from .shared.state import ANNOTATION_KEYS, LARGER_CIRCLE_SIDES, SCENE_ID, TANGENT_SIDES

TASK_ID_EXTERNAL_TANGENT_SEGMENT_LENGTH = (
    "task_geometry__circle_pair_tangents__external_tangent_segment_length_value"
)
TASK_ID = TASK_ID_EXTERNAL_TANGENT_SEGMENT_LENGTH
QUERY_ID_TANGENT_SEGMENT_LENGTH_FROM_CENTER_DISTANCE = "tangent_segment_length_from_center_distance"
QUERY_ID_CENTER_DISTANCE_FROM_TANGENT_SEGMENT_LENGTH = "center_distance_from_tangent_segment_length"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (
    QUERY_ID_TANGENT_SEGMENT_LENGTH_FROM_CENTER_DISTANCE,
    QUERY_ID_CENTER_DISTANCE_FROM_TANGENT_SEGMENT_LENGTH,
)
_SCENE_DEFAULTS = get_scene_defaults("geometry", SCENE_ID)
_GEN_DEFAULTS_UNUSED, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    _SCENE_DEFAULTS,
    task_id=TASK_ID,
)


def _build_tangent_segment_problem(*, instance_seed: int, params: Mapping[str, Any]) -> PairTangentValueProblem:
    """Bind tangent segment AB as the requested integer answer."""

    layout = select_tangent_layout(
        instance_seed=int(instance_seed),
        params=params,
        sampling_namespace=f"{TASK_ID}.{QUERY_ID_TANGENT_SEGMENT_LENGTH_FROM_CENTER_DISTANCE}",
    )
    return build_pair_tangent_value_problem(
        layout=layout,
        answer=int(layout.case.tangent_length),
        center_segment_label=f"CD={int(layout.case.center_distance)}",
        tangent_segment_label="AB=?",
        formula_family="external_common_tangent_right_triangle",
        formula="AB^2 = CD^2 - (r_D-r_C)^2",
        unknown_role="tangent_length",
        answer_metric="tangent_length",
    )


def _build_center_distance_problem(*, instance_seed: int, params: Mapping[str, Any]) -> PairTangentValueProblem:
    """Bind center segment CD as the requested integer answer."""

    layout = select_tangent_layout(
        instance_seed=int(instance_seed),
        params=params,
        sampling_namespace=f"{TASK_ID}.{QUERY_ID_CENTER_DISTANCE_FROM_TANGENT_SEGMENT_LENGTH}",
    )
    return build_pair_tangent_value_problem(
        layout=layout,
        answer=int(layout.case.center_distance),
        center_segment_label="CD=?",
        tangent_segment_label=f"AB={int(layout.case.tangent_length)}",
        formula_family="external_common_tangent_right_triangle",
        formula="CD^2 = AB^2 + (r_D-r_C)^2",
        unknown_role="center_distance",
        answer_metric="center_distance",
    )


def _build_problem(*, selected_query: str, instance_seed: int, params: Mapping[str, Any]) -> PairTangentValueProblem:
    if str(selected_query) == QUERY_ID_TANGENT_SEGMENT_LENGTH_FROM_CENTER_DISTANCE:
        return _build_tangent_segment_problem(instance_seed=int(instance_seed), params=params)
    if str(selected_query) == QUERY_ID_CENTER_DISTANCE_FROM_TANGENT_SEGMENT_LENGTH:
        return _build_center_distance_problem(instance_seed=int(instance_seed), params=params)
    raise ValueError(f"unsupported query_id for {TASK_ID}: {selected_query}")


@register_task
class GeometryCirclePairTangentsExternalTangentSegmentLengthValueTask:
    """Solve AB or CD from the external common tangent right-triangle relation."""

    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    domain = "geometry"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Select the requested unknown segment and construct answer plus annotation."""

        selected_query, query_probabilities, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=params,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=QUERY_ID_TANGENT_SEGMENT_LENGTH_FROM_CENTER_DISTANCE,
            task_id=TASK_ID,
        )
        problem = _build_problem(
            selected_query=str(selected_query),
            instance_seed=int(instance_seed),
            params=task_params,
        )
        prepared = prepare_pair_tangent_scene(
            diagram_spec=problem.diagram_spec,
            prompt_defaults=_PROMPT_DEFAULTS,
            prompt_key=str(selected_query),
            answer_value=int(problem.answer),
            instance_seed=int(instance_seed),
            params=task_params,
            render_defaults=_RENDER_DEFAULTS,
            max_attempts=int(max_attempts),
            style_namespace=f"{TASK_ID}.{selected_query}.render.scene",
        )
        trace_payload = pair_tangent_trace_payload(
            prepared=prepared,
            branch_name=str(selected_query),
            branch_probabilities=query_probabilities,
            problem=problem,
        )
        return TaskOutput(
            prompt=str(prepared.prompt_artifacts.prompt),
            answer_gt=TypedValue(type="integer", value=int(problem.answer)),
            annotation_gt=TypedValue(type="point_map", value=dict(prepared.annotation_value)),
            image=prepared.image,
            image_id="img0",
            trace_payload=trace_payload,
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=str(selected_query),
            prompt_variants=dict(prepared.prompt_artifacts.prompt_variants),
        )


_TANGENT_CASES = TANGENT_CASES

__all__ = [
    "ANNOTATION_KEYS",
    "GeometryCirclePairTangentsExternalTangentSegmentLengthValueTask",
    "LARGER_CIRCLE_SIDES",
    "QUERY_ID_CENTER_DISTANCE_FROM_TANGENT_SEGMENT_LENGTH",
    "QUERY_ID_TANGENT_SEGMENT_LENGTH_FROM_CENTER_DISTANCE",
    "SCENE_ID",
    "SUPPORTED_QUERY_IDS",
    "TANGENT_SIDES",
    "TASK_ID",
    "TASK_ID_EXTERNAL_TANGENT_SEGMENT_LENGTH",
    "_TANGENT_CASES",
]
