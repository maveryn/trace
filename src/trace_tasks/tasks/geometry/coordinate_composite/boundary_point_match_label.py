"""Select the marked point lying on two requested object boundaries."""

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
    point_on_circle_boundary,
    point_on_line_segment,
    point_on_polygon_boundary,
    polygon_object,
)
from .shared.sampling import CandidatePointCase, ResolvedCandidatePointProblem, resolve_candidate_point_selection
from .shared.state import GraphPoint, SceneObject

DOMAIN = "geometry"
SCENE_ID = "coordinate_composite"
TASK_ID = "task_geometry__coordinate_composite__boundary_point_match_label"
PROMPT_BUNDLE_ID = "geometry_coordinate_composite_v0"

QUERY_ID_LINE_CIRCLE_BOUNDARY_POINT = "line_circle_boundary_point"
QUERY_ID_CIRCLE_POLYGON_BOUNDARY_POINT = "circle_polygon_boundary_point"
QUERY_ID_LINE_POLYGON_BOUNDARY_POINT = "line_polygon_boundary_point"
QUERY_IDS: Tuple[str, ...] = (
    QUERY_ID_LINE_CIRCLE_BOUNDARY_POINT,
    QUERY_ID_CIRCLE_POLYGON_BOUNDARY_POINT,
    QUERY_ID_LINE_POLYGON_BOUNDARY_POINT,
)
LABELS: Tuple[str, ...] = ("A", "B", "C", "D")
TRANSFORMS: Tuple[str, ...] = ("identity", "reflect_x", "reflect_y", "rotate180")

_SCENE_DEFAULTS, _BACKGROUND_DEFAULTS, _NOISE_DEFAULTS = load_coordinate_composite_default_sets(domain=DOMAIN, scene_id=SCENE_ID)


BoundaryPointPredicate = Callable[[GraphPoint, Tuple[SceneObject, ...]], bool]


def _case_pool_by_query() -> Dict[str, Tuple[CandidatePointCase, ...]]:
    return {
        QUERY_ID_LINE_CIRCLE_BOUNDARY_POINT: (
            CandidatePointCase(
                case_id="line_circle_grid_intersection",
                objects=(
                    line_object("line_a", (-6.0, 3.0), (6.0, 3.0)),
                    circle_object("circle_b", (0.0, 0.0), 5.0),
                    polygon_object("polygon_c", ((-5.0, -3.0), (-1.0, -3.0), (-1.0, 0.0), (-5.0, 0.0))),
                ),
                candidate_points=((4.0, 3.0), (0.0, 3.0), (5.0, 0.0), (0.0, 0.0)),
                transforms=TRANSFORMS,
            ),
        ),
        QUERY_ID_CIRCLE_POLYGON_BOUNDARY_POINT: (
            CandidatePointCase(
                case_id="circle_rectangle_corner_intersection",
                objects=(
                    circle_object("circle_a", (0.0, 0.0), 5.0),
                    polygon_object("polygon_b", ((-4.0, -3.0), (4.0, -3.0), (4.0, 3.0), (-4.0, 3.0))),
                    line_object("line_c", (-6.0, -4.0), (6.0, -4.0)),
                ),
                candidate_points=((4.0, 3.0), (5.0, 0.0), (4.0, 0.0), (0.0, 0.0)),
                transforms=TRANSFORMS,
            ),
        ),
        QUERY_ID_LINE_POLYGON_BOUNDARY_POINT: (
            CandidatePointCase(
                case_id="line_rectangle_side_intersection",
                objects=(
                    line_object("line_a", (-6.0, 0.0), (6.0, 0.0)),
                    polygon_object("polygon_b", ((-4.0, -2.0), (4.0, -2.0), (4.0, 2.0), (-4.0, 2.0))),
                    circle_object("circle_c", (0.0, 4.0), 1.5),
                ),
                candidate_points=((4.0, 0.0), (0.0, 0.0), (4.0, 2.0), (6.0, 0.0)),
                transforms=TRANSFORMS,
            ),
        ),
    }


_CASES_BY_QUERY = _case_pool_by_query()


def _split_boundary_defaults() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    return split_coordinate_composite_defaults(_SCENE_DEFAULTS, public_identifier=TASK_ID)


def _boundary_predicate(query_id: str) -> BoundaryPointPredicate:
    if query_id == QUERY_ID_LINE_CIRCLE_BOUNDARY_POINT:
        return lambda point, objects: point_on_line_segment(point, first_line_object(objects)) and point_on_circle_boundary(point, first_circle_object(objects))
    if query_id == QUERY_ID_CIRCLE_POLYGON_BOUNDARY_POINT:
        return lambda point, objects: point_on_circle_boundary(point, first_circle_object(objects)) and point_on_polygon_boundary(point, first_polygon_object(objects))
    if query_id == QUERY_ID_LINE_POLYGON_BOUNDARY_POINT:
        return lambda point, objects: point_on_line_segment(point, first_line_object(objects)) and point_on_polygon_boundary(point, first_polygon_object(objects))
    raise ValueError(f"unsupported boundary-point query id: {query_id!r}")


def _select_boundary_problem(*, instance_seed: int, params: Mapping[str, Any]) -> ResolvedCandidatePointProblem:
    """Resolve the public boundary relation and bind randomized candidate labels."""

    return resolve_candidate_point_selection(
        instance_seed=int(instance_seed),
        params=params,
        supported_keys=QUERY_IDS,
        default_key=QUERY_ID_LINE_CIRCLE_BOUNDARY_POINT,
        public_identifier=TASK_ID,
        cases_by_selection=_CASES_BY_QUERY,
        predicate_for_selection=_boundary_predicate,
        option_labels=LABELS,
    )


@register_task
class GeometryCoordinateCompositeBoundaryPointMatchLabelTask:
    domain = DOMAIN
    task_id = TASK_ID
    reasoning_operations = ('logical_composition', 'spatial_relations', 'matching')
    supported_query_ids = QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        attempts_seen = int(max_attempts)
        _ = attempts_seen
        generation_defaults, visual_defaults, prose_defaults = _split_boundary_defaults()
        _ = generation_defaults
        boundary_problem = _select_boundary_problem(instance_seed=int(instance_seed), params=params)
        bundle = compose_resolved_candidate_point_artifacts(
            domain=DOMAIN,
            scene_id=SCENE_ID,
            family_code="coordinate_composite_boundary_point_match",
            resolved=boundary_problem,
            params=params,
            render_defaults=visual_defaults,
            prompt_defaults=prose_defaults,
            background_defaults=_BACKGROUND_DEFAULTS,
            noise_defaults=_NOISE_DEFAULTS,
            random_namespace=f"{TASK_ID}.render",
            instance_seed=int(instance_seed),
            prompt_bundle_id=PROMPT_BUNDLE_ID,
        )
        return TaskOutput(**candidate_point_output_fields(
            artifacts=bundle,
            answer_gt=TypedValue(type="option_letter", value=str(boundary_problem.selected_label)),
            annotation_gt=TypedValue(type="point", value=list(bundle.annotation_value)),
            image_id=f"{TASK_ID}:{int(instance_seed)}",
            scene_id=SCENE_ID,
            selection_key=str(boundary_problem.selection_key),
        ))


__all__ = [
    "GeometryCoordinateCompositeBoundaryPointMatchLabelTask",
    "QUERY_IDS",
    "SCENE_ID",
    "TASK_ID",
]
