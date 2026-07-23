"""Select the marked point satisfying a composite coordinate-region predicate."""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Tuple

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task

from .shared.defaults import load_coordinate_composite_default_sets, split_coordinate_composite_defaults
from .shared.output import candidate_point_output_fields, compose_resolved_candidate_point_artifacts
from .shared.relations import (
    circle_object,
    first_circle_object,
    first_line_object,
    first_polygon_object,
    line_object,
    point_above_line,
    point_below_line,
    point_inside_circle,
    point_inside_polygon,
    polygon_object,
)
from .shared.sampling import CandidatePointCase, ResolvedCandidatePointProblem, resolve_candidate_point_selection
from .shared.state import GraphPoint, SceneObject

DOMAIN = "geometry"
SCENE_ID = "coordinate_composite"
TASK_ID = "task_geometry__coordinate_composite__region_membership_label"
PROMPT_BUNDLE_ID = "geometry_coordinate_composite_v0"

QUERY_ID_INSIDE_CIRCLE_OUTSIDE_POLYGON = "inside_circle_outside_polygon"
QUERY_ID_INSIDE_POLYGON_OUTSIDE_CIRCLE = "inside_polygon_outside_circle"
QUERY_ID_INSIDE_CIRCLE_ABOVE_LINE = "inside_circle_above_line"
QUERY_ID_INSIDE_POLYGON_BELOW_LINE = "inside_polygon_below_line"
QUERY_IDS: Tuple[str, ...] = (
    QUERY_ID_INSIDE_CIRCLE_OUTSIDE_POLYGON,
    QUERY_ID_INSIDE_POLYGON_OUTSIDE_CIRCLE,
    QUERY_ID_INSIDE_CIRCLE_ABOVE_LINE,
    QUERY_ID_INSIDE_POLYGON_BELOW_LINE,
)
LABELS: Tuple[str, ...] = ("A", "B", "C", "D")
REGION_TRANSFORMS: Tuple[str, ...] = ("identity", "reflect_x", "reflect_y", "rotate180")
LINE_TRANSFORMS: Tuple[str, ...] = ("identity", "reflect_y")

_SCENE_DEFAULTS, _BACKGROUND_DEFAULTS, _NOISE_DEFAULTS = load_coordinate_composite_default_sets(domain=DOMAIN, scene_id=SCENE_ID)


Predicate = Callable[[GraphPoint, Tuple[SceneObject, ...]], bool]


def _base_objects() -> Tuple[SceneObject, ...]:
    """Return a stable circle/polygon/line composite for candidate-point predicates."""

    return (
        circle_object("circle_a", (0.0, 0.0), 4.0),
        polygon_object("polygon_b", ((-5.0, -2.0), (1.0, -2.0), (1.0, 3.0), (-5.0, 3.0))),
        line_object("line_c", (-6.0, 1.0), (6.0, 1.0)),
    )


def _case_pool_by_query() -> Dict[str, Tuple[CandidatePointCase, ...]]:
    return {
        QUERY_ID_INSIDE_CIRCLE_OUTSIDE_POLYGON: (
            CandidatePointCase(
                case_id="circle_not_polygon_default",
                objects=_base_objects(),
                candidate_points=((3.0, 0.0), (-3.0, 0.0), (0.0, 0.0), (-5.0, 4.0)),
                transforms=REGION_TRANSFORMS,
            ),
        ),
        QUERY_ID_INSIDE_POLYGON_OUTSIDE_CIRCLE: (
            CandidatePointCase(
                case_id="polygon_not_circle_default",
                objects=_base_objects(),
                candidate_points=((-4.0, 2.0), (-3.0, 0.0), (3.0, 0.0), (5.0, 0.0)),
                transforms=REGION_TRANSFORMS,
            ),
        ),
        QUERY_ID_INSIDE_CIRCLE_ABOVE_LINE: (
            CandidatePointCase(
                case_id="circle_above_line_default",
                objects=_base_objects(),
                candidate_points=((2.0, 2.0), (-3.0, 0.0), (-5.0, 2.0), (0.0, -3.0)),
                transforms=LINE_TRANSFORMS,
            ),
        ),
        QUERY_ID_INSIDE_POLYGON_BELOW_LINE: (
            CandidatePointCase(
                case_id="polygon_below_line_default",
                objects=_base_objects(),
                candidate_points=((-3.0, 0.0), (-3.0, 2.0), (3.0, 0.0), (-4.0, -3.0)),
                transforms=LINE_TRANSFORMS,
            ),
        ),
    }


_CASES_BY_QUERY = _case_pool_by_query()


def _split_defaults_for_task() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    return split_coordinate_composite_defaults(_SCENE_DEFAULTS, public_identifier=TASK_ID)


def _predicate_for_query(query_id: str) -> Predicate:
    if query_id == QUERY_ID_INSIDE_CIRCLE_OUTSIDE_POLYGON:
        return lambda point, objects: point_inside_circle(point, first_circle_object(objects)) and not point_inside_polygon(point, first_polygon_object(objects))
    if query_id == QUERY_ID_INSIDE_POLYGON_OUTSIDE_CIRCLE:
        return lambda point, objects: point_inside_polygon(point, first_polygon_object(objects)) and not point_inside_circle(point, first_circle_object(objects))
    if query_id == QUERY_ID_INSIDE_CIRCLE_ABOVE_LINE:
        return lambda point, objects: point_inside_circle(point, first_circle_object(objects)) and point_above_line(point, first_line_object(objects))
    if query_id == QUERY_ID_INSIDE_POLYGON_BELOW_LINE:
        return lambda point, objects: point_inside_polygon(point, first_polygon_object(objects)) and point_below_line(point, first_line_object(objects))
    raise ValueError(f"unsupported region-membership query id: {query_id!r}")


def _select_problem(*, instance_seed: int, params: Mapping[str, Any]) -> ResolvedCandidatePointProblem:
    """Resolve the public predicate query and bind randomized candidate labels."""

    return resolve_candidate_point_selection(
        instance_seed=int(instance_seed),
        params=params,
        supported_keys=QUERY_IDS,
        default_key=QUERY_ID_INSIDE_CIRCLE_OUTSIDE_POLYGON,
        public_identifier=TASK_ID,
        cases_by_selection=_CASES_BY_QUERY,
        predicate_for_selection=_predicate_for_query,
        option_labels=LABELS,
    )


@register_task
class GeometryCoordinateCompositeRegionMembershipLabelTask:
    """Select the marked candidate point satisfying a non-numeric region predicate."""

    task_id = TASK_ID
    reasoning_operations = ('spatial_relations',)
    domain = DOMAIN
    default_dataset_enabled = True
    supported_query_ids = QUERY_IDS

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one candidate-point region-membership selection instance."""

        _ = int(max_attempts)
        generation_defaults, render_defaults, prompt_defaults = _split_defaults_for_task()
        _ = generation_defaults
        problem = _select_problem(instance_seed=int(instance_seed), params=params)
        artifacts = compose_resolved_candidate_point_artifacts(
            domain=DOMAIN,
            scene_id=SCENE_ID,
            family_code="coordinate_composite_region_membership",
            resolved=problem,
            params=params,
            render_defaults=render_defaults,
            prompt_defaults=prompt_defaults,
            background_defaults=_BACKGROUND_DEFAULTS,
            noise_defaults=_NOISE_DEFAULTS,
            random_namespace=f"{TASK_ID}.render",
            instance_seed=int(instance_seed),
            prompt_bundle_id=PROMPT_BUNDLE_ID,
        )
        return TaskOutput(**candidate_point_output_fields(
            artifacts=artifacts,
            answer_gt=TypedValue(type="option_letter", value=str(problem.selected_label)),
            annotation_gt=TypedValue(type="point", value=list(artifacts.annotation_value)),
            image_id=f"{TASK_ID}:{int(instance_seed)}",
            scene_id=SCENE_ID,
            selection_key=str(problem.selection_key),
        ))


__all__ = [
    "GeometryCoordinateCompositeRegionMembershipLabelTask",
    "QUERY_IDS",
    "SCENE_ID",
    "TASK_ID",
]
