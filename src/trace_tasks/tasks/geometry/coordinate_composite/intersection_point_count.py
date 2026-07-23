"""Count intersection points in coordinate-composite diagrams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.core.sampling import uniform_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import split_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.deterministic_sampling import resolve_selection_index
from trace_tasks.tasks.shared.fixed_query import geometry_selected_probability_map, select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions

from ..shared.background_defaults import load_geometry_background_defaults
from ..shared.noise_defaults import load_geometry_noise_defaults
from ..shared.vector2d import point_to_list
from .shared.relations import circle_object, filtered_intersections, line_object, polygon_object, transform_object
from .shared.prompts import build_coordinate_composite_prompt_artifacts
from .shared.rendering import render_coordinate_composite_scene
from .shared.state import PairFilter, SceneObject

DOMAIN = "geometry"
SCENE_ID = "coordinate_composite"
TASK_ID = "task_geometry__coordinate_composite__intersection_point_count"
PROMPT_BUNDLE_ID = "geometry_coordinate_composite_v0"

QUERY_ID_LINE_CIRCLE = "line_circle_intersection_count"
QUERY_ID_CIRCLE_CIRCLE = "circle_circle_intersection_count"
QUERY_ID_LINE_POLYGON = "line_polygon_intersection_count"
QUERY_ID_CIRCLE_POLYGON = "circle_polygon_intersection_count"
QUERY_ID_MIXED_OBJECT = "mixed_object_intersection_count"
QUERY_IDS: Tuple[str, ...] = (
    QUERY_ID_LINE_CIRCLE,
    QUERY_ID_CIRCLE_CIRCLE,
    QUERY_ID_LINE_POLYGON,
    QUERY_ID_CIRCLE_POLYGON,
    QUERY_ID_MIXED_OBJECT,
)
TRANSFORMS: Tuple[str, ...] = ("identity", "reflect_x", "reflect_y", "rotate90", "rotate180")
PAIR_FILTER_BY_QUERY = {
    QUERY_ID_LINE_CIRCLE: PairFilter.LINE_CIRCLE,
    QUERY_ID_CIRCLE_CIRCLE: PairFilter.CIRCLE_CIRCLE,
    QUERY_ID_LINE_POLYGON: PairFilter.LINE_POLYGON,
    QUERY_ID_CIRCLE_POLYGON: PairFilter.CIRCLE_POLYGON,
    QUERY_ID_MIXED_OBJECT: PairFilter.ALL,
}

_SCENE_DEFAULTS = get_scene_defaults(DOMAIN, SCENE_ID)
_BACKGROUND_DEFAULTS = load_geometry_background_defaults(scene_id=SCENE_ID)
_NOISE_DEFAULTS = load_geometry_noise_defaults(scene_id=SCENE_ID)


@dataclass(frozen=True)
class _CompositeCase:
    """One task-owned object case with a known intersection count."""

    case_id: str
    objects: Tuple[SceneObject, ...]
    expected_count: int


@dataclass(frozen=True)
class _ResolvedProblem:
    """Task-selected case, transform, and review probability metadata."""

    query_id: str
    pair_filter: PairFilter
    case: _CompositeCase
    transform: str
    query_id_probabilities: Dict[str, float]
    target_count_probabilities: Dict[str, float]
    case_probabilities: Dict[str, float]
    transform_probabilities: Dict[str, float]


def _split_defaults_for_task() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    return split_scene_generation_rendering_prompt_defaults(
        _SCENE_DEFAULTS if isinstance(_SCENE_DEFAULTS, Mapping) else {},
        task_id=TASK_ID,
    )


def _case_pool_by_query() -> Dict[str, Tuple[_CompositeCase, ...]]:
    """Build task-owned case pools for each public query branch."""

    rect = ((-3.0, -2.0), (3.0, -2.0), (3.0, 2.0), (-3.0, 2.0))
    wide_rect = ((-4.0, -3.0), (4.0, -3.0), (4.0, 3.0), (-4.0, 3.0))
    return {
        QUERY_ID_LINE_CIRCLE: (
            _CompositeCase("line_circle_zero_points", (line_object("line_a", (-7.0, 5.0), (7.0, 5.0)), circle_object("circle_b", (0.0, 0.0), 2.0)), 0),
            _CompositeCase("line_circle_one_tangent", (line_object("line_a", (-7.0, 2.0), (7.0, 2.0)), circle_object("circle_b", (0.0, 0.0), 2.0)), 1),
            _CompositeCase("line_circle_two_secant", (line_object("line_a", (-7.0, 1.0), (7.0, 1.0)), circle_object("circle_b", (0.0, 0.0), 2.0)), 2),
            _CompositeCase(
                "line_circle_three_two_plus_tangent",
                (
                    line_object("line_a", (-7.0, 1.0), (7.0, 1.0)),
                    circle_object("circle_b", (-3.0, 0.0), 1.0),
                    circle_object("circle_c", (3.0, 1.0), 1.0),
                ),
                3,
            ),
            _CompositeCase(
                "line_circle_four_two_secants",
                (
                    line_object("line_a", (-7.0, 1.0), (7.0, 1.0)),
                    circle_object("circle_b", (-3.0, 1.0), 1.0),
                    circle_object("circle_c", (3.0, 1.0), 1.0),
                ),
                4,
            ),
        ),
        QUERY_ID_CIRCLE_CIRCLE: (
            _CompositeCase("circle_circle_zero_separate", (circle_object("circle_a", (-5.0, 0.0), 1.5), circle_object("circle_b", (1.0, 0.0), 1.5)), 0),
            _CompositeCase("circle_circle_one_tangent", (circle_object("circle_a", (-2.0, 0.0), 2.0), circle_object("circle_b", (2.0, 0.0), 2.0)), 1),
            _CompositeCase("circle_circle_two_overlap", (circle_object("circle_a", (-2.0, 0.0), 3.0), circle_object("circle_b", (2.0, 0.0), 3.0)), 2),
            _CompositeCase(
                "circle_circle_three_mixed",
                (circle_object("circle_a", (0.0, 0.0), 2.0), circle_object("circle_b", (4.0, 0.0), 2.0), circle_object("circle_c", (0.0, 4.0), 3.0)),
                3,
            ),
            _CompositeCase(
                "circle_circle_four_two_pairs",
                (
                    circle_object("circle_a", (-5.0, 0.0), 1.6),
                    circle_object("circle_b", (-2.5, 0.0), 1.6),
                    circle_object("circle_c", (2.5, 0.0), 1.6),
                    circle_object("circle_d", (5.0, 0.0), 1.6),
                ),
                4,
            ),
        ),
        QUERY_ID_LINE_POLYGON: (
            _CompositeCase("line_polygon_zero_external", (line_object("line_a", (-7.0, 4.0), (7.0, 4.0)), polygon_object("polygon_b", rect)), 0),
            _CompositeCase("line_polygon_one_corner_touch", (line_object("line_a", (-7.0, 2.0), (-3.0, 2.0)), polygon_object("polygon_b", rect)), 1),
            _CompositeCase("line_polygon_two_crossing", (line_object("line_a", (-7.0, 1.0), (7.0, 1.0)), polygon_object("polygon_b", rect)), 2),
            _CompositeCase(
                "line_polygon_three_cross_and_touch",
                (line_object("line_a", (-7.0, 1.0), (7.0, 1.0)), line_object("line_b", (-7.0, 2.0), (-3.0, 2.0)), polygon_object("polygon_c", rect)),
                3,
            ),
            _CompositeCase(
                "line_polygon_four_two_crossings",
                (line_object("line_a", (-7.0, 1.0), (7.0, 1.0)), line_object("line_b", (1.0, -6.0), (1.0, 6.0)), polygon_object("polygon_c", rect)),
                4,
            ),
        ),
        QUERY_ID_CIRCLE_POLYGON: (
            _CompositeCase("circle_polygon_zero_inside", (circle_object("circle_a", (0.0, 0.0), 1.0), polygon_object("polygon_b", wide_rect)), 0),
            _CompositeCase(
                "circle_polygon_one_tangent",
                (circle_object("circle_a", (0.0, 0.0), 2.0), polygon_object("polygon_b", ((-4.0, 2.0), (4.0, 2.0), (4.0, 5.0), (-4.0, 5.0)))),
                1,
            ),
            _CompositeCase(
                "circle_polygon_two_tangencies",
                (circle_object("circle_a", (0.0, 0.0), 2.0), polygon_object("polygon_b", ((-2.0, -3.0), (2.0, -3.0), (2.0, 3.0), (-2.0, 3.0)))),
                2,
            ),
            _CompositeCase(
                "circle_polygon_three_tangent_plus_two",
                (
                    circle_object("circle_a", (0.0, 0.0), 2.0),
                    polygon_object("polygon_b", ((-2.0, -3.0), (2.0, -3.0), (2.0, 3.0), (-2.0, 3.0))),
                    polygon_object("polygon_c", ((-4.0, 2.0), (4.0, 2.0), (4.0, 5.0), (-4.0, 5.0))),
                ),
                3,
            ),
            _CompositeCase(
                "circle_polygon_four_crossings",
                (circle_object("circle_a", (0.0, 0.0), 3.0), polygon_object("polygon_b", ((-4.0, -2.0), (4.0, -2.0), (4.0, 2.0), (-4.0, 2.0)))),
                4,
            ),
        ),
        QUERY_ID_MIXED_OBJECT: (
            _CompositeCase(
                "mixed_zero_separate",
                (
                    line_object("line_a", (-7.0, -5.0), (7.0, -5.0)),
                    circle_object("circle_b", (0.0, 0.0), 2.0),
                    polygon_object("polygon_c", ((4.0, -2.0), (7.0, -2.0), (7.0, 2.0), (4.0, 2.0))),
                ),
                0,
            ),
            _CompositeCase(
                "mixed_one_tangent",
                (
                    line_object("line_a", (-7.0, 2.0), (2.0, 2.0)),
                    circle_object("circle_b", (0.0, 0.0), 2.0),
                    polygon_object("polygon_c", ((4.0, -2.0), (7.0, -2.0), (7.0, 2.0), (4.0, 2.0))),
                ),
                1,
            ),
            _CompositeCase(
                "mixed_two_secant",
                (
                    line_object("line_a", (-7.0, 1.0), (2.0, 1.0)),
                    circle_object("circle_b", (0.0, 0.0), 2.0),
                    polygon_object("polygon_c", ((4.0, -2.0), (7.0, -2.0), (7.0, 2.0), (4.0, 2.0))),
                ),
                2,
            ),
            _CompositeCase(
                "mixed_three_line_circle_rectangle",
                (
                    line_object("line_a", (-7.0, 1.0), (7.0, 1.0)),
                    circle_object("circle_b", (0.0, 4.0), 3.0),
                    polygon_object("polygon_c", ((4.0, -2.0), (7.0, -2.0), (7.0, 2.0), (4.0, 2.0))),
                ),
                3,
            ),
            _CompositeCase(
                "mixed_four_line_circle_rectangle",
                (
                    line_object("line_a", (-7.0, 1.0), (7.0, 1.0)),
                    circle_object("circle_b", (-2.0, 1.0), 2.0),
                    polygon_object("polygon_c", ((3.0, -1.0), (6.0, -1.0), (6.0, 3.0), (3.0, 3.0))),
                ),
                4,
            ),
        ),
    }


_CASES_BY_QUERY: Dict[str, Tuple[_CompositeCase, ...]] = _case_pool_by_query()


def _select_problem(*, instance_seed: int, params: Mapping[str, Any]) -> _ResolvedProblem:
    """Resolve public query, target count, concrete case, and transform."""

    selected_query, query_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=QUERY_IDS,
        default_query_id=QUERY_ID_LINE_CIRCLE,
        task_id=TASK_ID,
    )
    cases = _CASES_BY_QUERY[str(selected_query)]
    target_counts = tuple(sorted({int(case.expected_count) for case in cases}))
    explicit_target = task_params.get("target_count")
    eligible_cases = tuple(cases)
    if explicit_target is not None:
        target_count = int(explicit_target)
        if target_count not in set(target_counts):
            raise ValueError(f"target_count={target_count} is not supported for {selected_query}")
        eligible_cases = tuple(case for case in cases if int(case.expected_count) == int(target_count))
        target_count_probabilities = geometry_selected_probability_map(target_counts, selected=target_count)
    else:
        target_count_probabilities = geometry_selected_probability_map(target_counts)

    explicit_case = task_params.get("case_id")
    if explicit_case is not None:
        case_id = str(explicit_case)
        matching = tuple(case for case in eligible_cases if str(case.case_id) == case_id)
        if not matching:
            raise ValueError(f"case_id={case_id!r} is not valid for {selected_query}")
        case = matching[0]
        case_probabilities = geometry_selected_probability_map((case.case_id for case in eligible_cases), selected=case.case_id)
    else:
        rng = spawn_rng(int(instance_seed), f"{TASK_ID}.{selected_query}.case")
        case = uniform_choice(rng, eligible_cases)
        case_probabilities = geometry_selected_probability_map(tuple(case.case_id for case in eligible_cases))

    explicit_transform = task_params.get("transform")
    if explicit_transform is not None:
        transform = str(explicit_transform)
        if transform not in TRANSFORMS:
            raise ValueError(f"transform={transform!r} is not valid for {TASK_ID}")
        transform_probabilities = geometry_selected_probability_map(TRANSFORMS, selected=transform)
    else:
        rng = spawn_rng(int(instance_seed), f"{TASK_ID}.transform")
        transform = str(uniform_choice(rng, TRANSFORMS))
        transform_probabilities = geometry_selected_probability_map(TRANSFORMS)

    return _ResolvedProblem(
        query_id=str(selected_query),
        pair_filter=PAIR_FILTER_BY_QUERY[str(selected_query)],
        case=case,
        transform=str(transform),
        query_id_probabilities=dict(query_probabilities),
        target_count_probabilities=dict(target_count_probabilities),
        case_probabilities=dict(case_probabilities),
        transform_probabilities=dict(transform_probabilities),
    )


def _prompt_artifacts(
    *,
    prompt_defaults: Mapping[str, Any],
    problem: _ResolvedProblem,
    params: Mapping[str, Any],
    instance_seed: int,
) -> Any:
    """Render the external prompt bundle after the public query is selected."""

    return build_coordinate_composite_prompt_artifacts(
        domain=DOMAIN,
        scene_id=SCENE_ID,
        prompt_defaults=prompt_defaults,
        params=params,
        instance_seed=int(instance_seed),
        query_key=str(problem.query_id),
        prompt_bundle_id=PROMPT_BUNDLE_ID,
        object_description_key="object_description",
        annotation_hint_key="annotation_hint",
        answer_hint_key="answer_hint_integer",
        json_example_key="json_example",
        json_example_answer_only_key="json_example_answer_only",
        context=f"prompt defaults for {TASK_ID}",
    )


def _trace_payload(
    *,
    problem: _ResolvedProblem,
    rendered: Any,
    prompt_artifacts: Any,
    answer_value: int,
    annotation_value: list[list[float]],
) -> Dict[str, Any]:
    """Build verifier payload from the same rendered trace used for answer binding."""

    return {
        "scene_id": SCENE_ID,
        "query_id": str(problem.query_id),
        "scene_ir": {
            "scene_kind": "geometry_coordinate_composite",
            "scene_id": SCENE_ID,
            "objects": [dict(obj) for obj in rendered.object_specs],
        },
        "query_spec": {
            "query_id": str(problem.query_id),
            "query_id_probabilities": dict(problem.query_id_probabilities),
            "target_count": int(answer_value),
            "target_count_probabilities": dict(problem.target_count_probabilities),
            "case_id": str(problem.case.case_id),
            "case_probabilities": dict(problem.case_probabilities),
            "transform": str(problem.transform),
            "transform_probabilities": dict(problem.transform_probabilities),
            "params": {
                "query_id": str(problem.query_id),
                "target_count": int(answer_value),
                "case_id": str(problem.case.case_id),
                "transform": str(problem.transform),
            },
        },
        "render_spec": {
            "canvas_width": int(rendered.image.width),
            "canvas_height": int(rendered.image.height),
            "background": dict(rendered.background_meta),
            "post_image_noise": dict(rendered.post_noise_meta),
            **dict(rendered.render_spec_extra),
        },
        "render_map": dict(rendered.render_map),
        "execution_trace": {
            "formula_family": "coordinate_composite_intersection_count",
            "query_id": str(problem.query_id),
            "case_id": str(problem.case.case_id),
            "answer": int(answer_value),
            "intersection_points_graph": [point_to_list(point) for point in rendered.intersection_points_graph],
            "intersection_points_px": [point_to_list(point) for point in rendered.intersection_points_px],
        },
        "witness_symbolic": {
            "formula_family": "coordinate_composite_intersection_count",
            "objects_graph": [dict(obj) for obj in rendered.object_specs],
            "intersection_points_graph": [point_to_list(point) for point in rendered.intersection_points_graph],
        },
        "projected_annotation": {
            "type": "point_set",
            "point_set": list(annotation_value),
            "pixel_point_set": list(annotation_value),
            "source": "intersection_points_graph_projected_to_pixels",
        },
        "prompt": {
            "prompt_variant": dict(prompt_artifacts.prompt_variant),
            "prompt_variant_active_key": str(prompt_artifacts.prompt_variant_active_key),
            "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
        },
    }


@register_task
class GeometryCoordinateCompositeIntersectionPointCountTask:
    """Count visible intersection points in a composite coordinate scene."""

    task_id = TASK_ID
    reasoning_operations = ('counting', 'spatial_relations')
    domain = DOMAIN
    default_dataset_enabled = True
    supported_query_ids = QUERY_IDS

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one task instance while owning answer and annotation binding."""

        _ = int(max_attempts)
        generation_defaults, render_defaults, prompt_defaults = _split_defaults_for_task()
        _ = generation_defaults
        problem = _select_problem(instance_seed=int(instance_seed), params=params)
        objects = tuple(transform_object(obj, problem.transform) for obj in problem.case.objects)
        rendered = render_coordinate_composite_scene(
            instance_seed=int(instance_seed),
            objects=objects,
            pair_filter=problem.pair_filter,
            transform=problem.transform,
            expected_count=int(problem.case.expected_count),
            params=params,
            render_defaults=render_defaults,
            background_defaults=_BACKGROUND_DEFAULTS,
            noise_defaults=_NOISE_DEFAULTS,
            random_namespace=f"{TASK_ID}.render",
        )
        # The public task binds answer and annotation from the same rendered trace.
        answer_value = int(len(filtered_intersections(objects, problem.pair_filter)))
        annotation_value = [point_to_list(point) for point in rendered.intersection_points_px]
        prompt_artifacts = _prompt_artifacts(
            prompt_defaults=prompt_defaults,
            problem=problem,
            params=params,
            instance_seed=int(instance_seed),
        )
        return TaskOutput(
            prompt=str(prompt_artifacts.prompt),
            answer_gt=TypedValue(type="integer", value=int(answer_value)),
            annotation_gt=TypedValue(type="point_set", value=list(annotation_value)),
            image=rendered.image,
            image_id=f"{TASK_ID}:{int(instance_seed)}",
            trace_payload=_trace_payload(
                problem=problem,
                rendered=rendered,
                prompt_artifacts=prompt_artifacts,
                answer_value=int(answer_value),
                annotation_value=list(annotation_value),
            ),
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=str(problem.query_id),
            prompt_variants=dict(prompt_artifacts.prompt_variants),
        )


__all__ = [
    "GeometryCoordinateCompositeIntersectionPointCountTask",
    "QUERY_IDS",
    "SCENE_ID",
    "TASK_ID",
]
