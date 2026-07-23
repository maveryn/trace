"""Scene-private response assembly for printed map public tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Sequence

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.deterministic_sampling import resolve_selection_index
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    build_prompt_query_spec,
    build_prompt_trace_artifacts,
    render_task_prompt_variants,
)

from .shared.annotations import bbox_sequence_for_ids, projected_annotation
from .shared.defaults import DOMAIN, PROMPT_BUNDLE, PROMPT_SCENE_KEY, PROMPT_TASK_KEY, SCENE
from .shared.rendering import render_map_case
from .shared.routes import (
    direction_between,
    format_direction_steps,
    landmark_bbox_ids_for_ids,
    landmark_ids_for_route,
    landmark_labels_for_ids,
    ordinal_label,
    sample_route,
)
from .shared.sampling import build_map_scene_case
from .shared.sampling import resolve_direction_step_bounds, resolve_highlighted_route_step_bounds
from .shared.state import MapSceneCase, RenderedMapBundle


@dataclass(frozen=True)
class MapPromptBinding:
    """Task-owned prompt key and dynamic prompt slots."""

    prompt_branch_key: str
    dynamic_slots: Mapping[str, Any]


@dataclass(frozen=True)
class MapRoutePlan:
    """Task-owned route answer data before final bbox projection."""

    prompt_binding: MapPromptBinding
    selected_branch: str
    branch_probabilities: Mapping[str, float]
    answer_label: str
    route_landmark_ids: Sequence[str]
    highlighted_route_landmark_ids: Sequence[str]
    annotation_bbox_ids: Sequence[str]
    annotation_landmark_bbox_ids: Sequence[str]
    annotation_zone_label_bbox_ids: Sequence[str]
    annotation_semantics: str
    question_format: str


RoutePlanFactory = Callable[
    [int, Mapping[str, Any], str, Mapping[str, float], MapSceneCase],
    MapRoutePlan,
]


def select_public_branch(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    supported: tuple[str, ...],
    default: str,
    public_task: str,
) -> tuple[str, Dict[str, float], Dict[str, Any]]:
    """Resolve the caller-selected public branch through the shared policy."""

    branch, probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=tuple(str(value) for value in supported),
        default_query_id=str(default),
        task_id=str(public_task),
    )
    return str(branch), dict(probabilities), dict(task_params)


def common_trace_fields(case: MapSceneCase) -> Dict[str, Any]:
    """Return trace fields independent of objective contract."""

    return {
        "scene_variant": str(case.scene_variant),
        "scene_variant_probabilities": dict(case.scene_variant_probabilities),
        "view_family": "printed_campus_map",
        "scene_title": str(case.scene_title),
        "grid_cols": int(case.grid_cols),
        "grid_rows": int(case.grid_rows),
        "landmark_count": int(case.landmark_count),
        "zone_specs": [dict(spec) for spec in case.zone_specs],
        "landmark_specs": [dict(spec) for spec in case.landmark_specs],
        "path_specs": [dict(spec) for spec in case.path_specs],
    }


def destination_route_plan(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    selected_branch: str,
    branch_probabilities: Mapping[str, float],
    case: MapSceneCase,
    route_namespace: str,
    prompt_query_key: str,
) -> MapRoutePlan:
    """Build the route plan for a destination-after-directions objective."""

    step_min, step_max = resolve_direction_step_bounds(params)
    rng = spawn_rng(int(instance_seed), str(route_namespace))
    route = sample_route(
        rng=rng,
        cells=case.cells,
        adjacency=case.adjacency,
        min_edges=int(step_min),
        max_edges=int(step_max),
    )
    route_landmark_ids = landmark_ids_for_route(case, route)
    route_labels = landmark_labels_for_ids(case, route_landmark_ids)
    directions = [direction_between(source, target) for source, target in zip(route, route[1:])]
    annotation_landmark_bbox_ids = landmark_bbox_ids_for_ids(case, route_landmark_ids)
    return MapRoutePlan(
        prompt_binding=MapPromptBinding(
            prompt_branch_key=str(prompt_query_key),
            dynamic_slots={
                "start_label": str(route_labels[0]),
                "direction_text": str(format_direction_steps(directions)),
            },
        ),
        selected_branch=str(selected_branch),
        branch_probabilities=dict(branch_probabilities),
        answer_label=str(route_labels[-1]),
        route_landmark_ids=tuple(route_landmark_ids),
        highlighted_route_landmark_ids=tuple(),
        annotation_bbox_ids=tuple(annotation_landmark_bbox_ids),
        annotation_landmark_bbox_ids=tuple(annotation_landmark_bbox_ids),
        annotation_zone_label_bbox_ids=tuple(),
        annotation_semantics="route_landmarks_ordered",
        question_format="map_destination_after_directions_label",
    )


def highlighted_route_step_plan(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    selected_branch: str,
    branch_probabilities: Mapping[str, float],
    case: MapSceneCase,
    route_namespace: str,
    step_namespace: str,
    prompt_query_key: str,
) -> MapRoutePlan:
    """Build the route plan for a highlighted-route step objective."""

    step_min, step_max = resolve_highlighted_route_step_bounds(params)
    rng = spawn_rng(int(instance_seed), str(route_namespace))
    route = sample_route(
        rng=rng,
        cells=case.cells,
        adjacency=case.adjacency,
        min_edges=max(int(step_min), 3),
        max_edges=int(step_max) + 1,
    )
    route_landmark_ids = landmark_ids_for_route(case, route)
    route_labels = landmark_labels_for_ids(case, route_landmark_ids)
    max_step = min(int(step_max), len(route_labels) - 1)
    step_index = int(
        int(step_min)
        + (
            resolve_selection_index(
                params=params,
                instance_seed=int(instance_seed),
                namespace=str(step_namespace),
            )
            % max(1, int(max_step) - int(step_min) + 1)
        )
    )
    annotation_route_ids = route_landmark_ids[: int(step_index) + 1]
    annotation_landmark_bbox_ids = landmark_bbox_ids_for_ids(case, annotation_route_ids)
    return MapRoutePlan(
        prompt_binding=MapPromptBinding(
            prompt_branch_key=str(prompt_query_key),
            dynamic_slots={
                "route_start_label": str(route_labels[0]),
                "route_end_label": str(route_labels[-1]),
                "step_ordinal": str(ordinal_label(step_index)),
            },
        ),
        selected_branch=str(selected_branch),
        branch_probabilities=dict(branch_probabilities),
        answer_label=str(route_labels[step_index]),
        route_landmark_ids=tuple(route_landmark_ids),
        highlighted_route_landmark_ids=tuple(route_landmark_ids),
        annotation_bbox_ids=tuple(annotation_landmark_bbox_ids),
        annotation_landmark_bbox_ids=tuple(annotation_landmark_bbox_ids),
        annotation_zone_label_bbox_ids=tuple(),
        annotation_semantics="highlighted_route_landmarks_ordered_to_answer",
        question_format="map_landmark_after_route_step_label",
    )


def _render_map_payload(rendered: RenderedMapBundle) -> Dict[str, Any]:
    scene = rendered.rendered_scene
    return {
        "image_id": "img0",
        "panel_bbox_px": list(scene.panel_bbox_px),
        "title_bbox_px": list(scene.title_bbox_px),
        "map_bbox_px": list(scene.map_bbox_px),
        "landmark_bboxes_px": dict(scene.landmark_bbox_map),
        "landmark_label_bboxes_px": dict(scene.landmark_label_bbox_map),
        "zone_label_bboxes_px": dict(scene.zone_label_bbox_map),
        "path_bboxes_px": dict(scene.path_bbox_map),
        "highlighted_route_bboxes_px": dict(scene.highlighted_route_bbox_map),
    }


def build_map_response(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    case: MapSceneCase,
    rendered: RenderedMapBundle,
    route_plan: MapRoutePlan,
) -> TaskOutput:
    """Assemble one complete printed map task response."""

    prompt_selection = render_task_prompt_variants(
        domain=DOMAIN,
        scene_id=SCENE,
        bundle_id=PROMPT_BUNDLE,
        scene_key=PROMPT_SCENE_KEY,
        task_key=PROMPT_TASK_KEY,
        query_key=str(route_plan.prompt_binding.prompt_branch_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots=dict(route_plan.prompt_binding.dynamic_slots),
        instance_seed=int(instance_seed),
    )
    prompt_artifacts = build_prompt_trace_artifacts(prompt_selection)
    annotation_bboxes = bbox_sequence_for_ids(
        rendered.rendered_scene,
        [str(item) for item in route_plan.annotation_bbox_ids],
    )
    answer_gt = TypedValue(type="string", value=str(route_plan.answer_label))
    annotation_gt = TypedValue(type="bbox_sequence", value=list(annotation_bboxes))
    common_fields = common_trace_fields(case)
    probabilities = {str(key): float(value) for key, value in route_plan.branch_probabilities.items()}
    query_params = {
        **common_fields,
        "query_id": str(route_plan.selected_branch),
        "prompt_query_key": str(route_plan.prompt_binding.prompt_branch_key),
        "source_query_id": str(route_plan.prompt_binding.prompt_branch_key),
        "answer_label": str(route_plan.answer_label),
        "route_landmark_count": int(len(route_plan.route_landmark_ids)),
        "annotation_bbox_count": int(len(route_plan.annotation_bbox_ids)),
        "query_id_probabilities": dict(probabilities),
    }
    query_spec = build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts,
        query_id=str(route_plan.selected_branch),
        params=query_params,
    )
    query_spec["scene_id"] = SCENE

    render_params = rendered.render_params
    trace_payload = {
        "scene_ir": {
            "scene_id": SCENE,
            "scene_kind": f"document_map_{case.scene_variant}",
            "entities": [dict(entity) for entity in rendered.rendered_scene.entities],
            "relations": {
                "query_id": str(route_plan.selected_branch),
                "prompt_query_key": str(route_plan.prompt_binding.prompt_branch_key),
                "source_query_id": str(route_plan.prompt_binding.prompt_branch_key),
                "scene_variant": str(case.scene_variant),
                "answer_label": str(route_plan.answer_label),
                "route_landmark_ids": [str(item) for item in route_plan.route_landmark_ids],
                "view_family": "printed_campus_map",
            },
        },
        "query_spec": query_spec,
        "render_spec": {
            "scene_id": SCENE,
            "scene_variant": str(case.scene_variant),
            "geometry_seed": int(instance_seed),
            "canvas_width": int(render_params.canvas_width),
            "canvas_height": int(render_params.canvas_height),
            "landmark_width_px": int(render_params.landmark_width_px),
            "landmark_height_px": int(render_params.landmark_height_px),
            "path_width_px": int(render_params.path_width_px),
            "highlighted_path_width_px": int(render_params.highlighted_path_width_px),
            "layout_jitter": dict(rendered.rendered_scene.layout_jitter_meta),
            "background_style": dict(rendered.background_meta),
            "post_image_noise": dict(rendered.post_noise_meta),
        },
        "render_map": _render_map_payload(rendered),
        "execution_trace": {
            **common_fields,
            "query_id": str(route_plan.selected_branch),
            "prompt_query_key": str(route_plan.prompt_binding.prompt_branch_key),
            "source_query_id": str(route_plan.prompt_binding.prompt_branch_key),
            "question_format": str(route_plan.question_format),
            "answer_label": str(route_plan.answer_label),
            "route_landmark_ids": [str(item) for item in route_plan.route_landmark_ids],
            "highlighted_route_landmark_ids": [
                str(item) for item in route_plan.highlighted_route_landmark_ids
            ],
            "annotation_bbox_ids": [str(item) for item in route_plan.annotation_bbox_ids],
            "annotation_landmark_bbox_ids": [
                str(item) for item in route_plan.annotation_landmark_bbox_ids
            ],
            "annotation_zone_label_bbox_ids": [
                str(item) for item in route_plan.annotation_zone_label_bbox_ids
            ],
            "supporting_bbox_ids": [str(item) for item in route_plan.annotation_bbox_ids],
            "annotation_semantics": str(route_plan.annotation_semantics),
            "query_id_probabilities": dict(probabilities),
        },
        "witness_symbolic": {
            "type": "ordered_id_path",
            "ids": [str(item) for item in route_plan.route_landmark_ids],
        },
        "projected_annotation": projected_annotation(
            rendered.rendered_scene,
            [str(item) for item in route_plan.annotation_bbox_ids],
        ),
    }
    return TaskOutput(
        prompt=str(prompt_artifacts.prompt),
        answer_gt=answer_gt,
        annotation_gt=annotation_gt,
        image=rendered.image,
        image_id="img0",
        trace_payload=trace_payload,
        task_versions=default_task_versions(),
        scene_id=SCENE,
        query_id=str(route_plan.selected_branch),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
    )


def render_bound_map(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    selected_branch: str,
    branch_probabilities: Mapping[str, float],
    plan_factory: RoutePlanFactory,
) -> TaskOutput:
    """Sample the map scene, bind a task-owned route plan, and render output."""

    case = build_map_scene_case(instance_seed=int(instance_seed), params=params)
    route_plan = plan_factory(
        int(instance_seed),
        params,
        str(selected_branch),
        dict(branch_probabilities),
        case,
    )
    rendered = render_map_case(
        instance_seed=int(instance_seed),
        params=params,
        case=case,
        highlighted_route_landmark_ids=[
            str(item) for item in route_plan.highlighted_route_landmark_ids
        ],
    )
    return build_map_response(
        instance_seed=int(instance_seed),
        params=params,
        case=case,
        rendered=rendered,
        route_plan=route_plan,
    )


__all__ = [
    "DOMAIN",
    "MapPromptBinding",
    "MapRoutePlan",
    "RoutePlanFactory",
    "build_map_response",
    "destination_route_plan",
    "highlighted_route_step_plan",
    "render_bound_map",
    "select_public_branch",
]
