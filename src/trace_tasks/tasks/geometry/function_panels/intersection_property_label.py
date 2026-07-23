"""Select the panel with the requested line/circle intersection property."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions

from ._lifecycle import build_intersection_components, intersection_plan, property_defaults
from .shared.state import (
    INTERSECTION_CIRCLE_CIRCLE,
    INTERSECTION_LINE_CIRCLE,
    INTERSECTION_TANGENT,
    INTERSECTION_TWO_POINTS,
    SCENE_ID,
)


TASK_ID = "task_geometry__function_panels__intersection_property_label"
QUERY_ID_LINE_CIRCLE_TANGENT = "line_circle_tangent_label"
QUERY_ID_LINE_CIRCLE_TWO = "line_circle_two_intersections_label"
QUERY_ID_CIRCLE_CIRCLE_TWO = "circle_circle_two_intersections_label"
SUPPORTED_QUERY_IDS = (
    QUERY_ID_LINE_CIRCLE_TANGENT,
    QUERY_ID_LINE_CIRCLE_TWO,
    QUERY_ID_CIRCLE_CIRCLE_TWO,
)
_DEFAULTS = property_defaults(TASK_ID)


def _prepare_intersection(branch_name: str, _params: Mapping[str, Any]):
    """Bind primitive-pair semantics for the selected intersection prompt."""

    if str(branch_name) == QUERY_ID_LINE_CIRCLE_TANGENT:
        pair_kind, relation_class = INTERSECTION_LINE_CIRCLE, INTERSECTION_TANGENT
    elif str(branch_name) == QUERY_ID_LINE_CIRCLE_TWO:
        pair_kind, relation_class = INTERSECTION_LINE_CIRCLE, INTERSECTION_TWO_POINTS
    else:
        pair_kind, relation_class = INTERSECTION_CIRCLE_CIRCLE, INTERSECTION_TWO_POINTS
    return intersection_plan(
        prompt_key=str(branch_name),
        pair_kind=pair_kind,
        relation_class=relation_class,
        defaults=_DEFAULTS,
    )


@register_task
class GeometryFunctionPanelsIntersectionPropertyLabelTask:
    """Choose the only panel matching the requested intersection condition."""

    task_id = TASK_ID
    reasoning_operations = ('spatial_relations',)
    domain = "geometry"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int) -> TaskOutput:
        del max_attempts
        branch_name, branch_probabilities, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=params,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=QUERY_ID_LINE_CIRCLE_TANGENT,
            task_id=TASK_ID,
        )
        components = build_intersection_components(
            instance_seed=int(instance_seed),
            params=task_params,
            branch_name=str(branch_name),
            branch_probabilities=branch_probabilities,
            namespace=TASK_ID,
            plan=_prepare_intersection(str(branch_name), task_params),
        )
        return TaskOutput(
            prompt=str(components.prompt),
            answer_gt=TypedValue(type="option_letter", value=str(components.trace_payload["execution_trace"]["answer_label"])),
            annotation_gt=components.annotation.annotation_gt,
            image=components.image,
            image_id="img_0",
            trace_payload=dict(components.trace_payload),
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=str(branch_name),
            prompt_variants=dict(components.prompt_variants),
        )


__all__ = ["GeometryFunctionPanelsIntersectionPropertyLabelTask"]
