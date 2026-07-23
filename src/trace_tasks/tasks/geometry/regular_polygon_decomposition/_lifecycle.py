"""Scene-private lifecycle plumbing for regular-polygon public tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from trace_tasks.core.types import TypedValue
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.shared.config_defaults import split_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec

from .shared.annotations import point_map_for_roles
from .shared.defaults import DOMAIN, POST_IMAGE_NOISE_DEFAULTS, SCENE_ID, raw_scene_defaults
from .shared.prompts import regular_polygon_prompt_artifacts
from .shared.rendering import create_render_context, render_regular_polygon_scene
from .shared.state import RegularPolygonProblem, RenderedRegularPolygonScene


@dataclass(frozen=True)
class RegularPolygonObjectivePlan:
    """Task-owned objective binding prepared after branch selection."""

    prompt_task_key: str
    prompt_branch_key: str
    problem: RegularPolygonProblem
    answer_gt: TypedValue
    annotation_roles: tuple[str, ...]
    query_params: Mapping[str, Any]
    trace_values: Mapping[str, Any]


def _measurement_payload(problem: RegularPolygonProblem, rendered: RenderedRegularPolygonScene) -> dict[str, Any]:
    return {
        "n_sides": int(problem.n_sides),
        "wedge_count": int(problem.wedge_count),
        "start_index": int(problem.start_index),
        "selected_wedge_indices": [int(index) for index in rendered.geometry.selected_wedge_indices],
        "central_angle_degrees": int(problem.central_angle_degrees),
        "angle_span_degrees": round(float(rendered.geometry.angle_span_degrees), 3),
        "total_area": None if problem.total_area is None else float(problem.total_area),
        "wedge_area": None if problem.wedge_area is None else float(problem.wedge_area),
        "side_length": None if problem.side_length is None else float(problem.side_length),
        "apothem": None if problem.apothem is None else float(problem.apothem),
        "perimeter": None if problem.perimeter is None else float(problem.perimeter),
        "target_name": str(problem.target_name),
        "relation": str(problem.relation),
    }


def _trace_payload(
    *,
    selected_branch: str,
    branch_probabilities: Mapping[str, float],
    prompt_artifacts: Any,
    plan: RegularPolygonObjectivePlan,
    rendered: RenderedRegularPolygonScene,
    image_size: tuple[int, int],
    annotation_value: Mapping[str, Any],
    noise_meta: Mapping[str, Any],
    style_meta: Mapping[str, Any],
) -> dict[str, Any]:
    """Serialize task-owned objective values without deciding task/query behavior."""

    answer_value = plan.answer_gt.value
    measurements = _measurement_payload(plan.problem, rendered)
    query_params = {
        "scene_id": SCENE_ID,
        "query_id": str(selected_branch),
        "query_id_probabilities": dict(branch_probabilities),
        **dict(plan.query_params),
    }
    trace_values = {
        "answer": answer_value,
        **measurements,
        "annotation_roles": list(plan.annotation_roles),
        **dict(plan.trace_values),
    }
    return {
        "scene_ir": {
            "domain": DOMAIN,
            "scene_id": SCENE_ID,
            "entities": [
                {
                    "type": "regular_polygon_cut_into_center_wedges",
                    "n_sides": int(plan.problem.n_sides),
                    "center": list(rendered.render_map["center"]),
                    "vertices": list(rendered.render_map["vertices"]),
                    "selected_wedge_indices": [int(index) for index in rendered.geometry.selected_wedge_indices],
                }
            ],
            "relations": {"type": str(plan.problem.relation), **dict(trace_values)},
        },
        "query_spec": build_prompt_query_spec(
            prompt_artifacts=prompt_artifacts,
            query_id=str(selected_branch),
            params=query_params,
        ),
        "render_spec": {
            "canvas": {"width": int(image_size[0]), "height": int(image_size[1])},
            "coord_space": "pixel",
            "style": dict(style_meta),
            "post_image_noise": dict(noise_meta),
        },
        "render_map": {"coord_space": "pixel", **dict(rendered.render_map)},
        "execution_trace": {
            "scene_id": SCENE_ID,
            "query_id": str(selected_branch),
            "query_id_probabilities": dict(branch_probabilities),
            **dict(trace_values),
        },
        "witness_symbolic": {
            "scene_id": SCENE_ID,
            "query_id": str(selected_branch),
            **dict(trace_values),
        },
        "projected_annotation": {
            "type": "point_map",
            "point_map": dict(annotation_value),
            "pixel_point_map": dict(annotation_value),
        },
    }


def run_regular_polygon_public_entry(
    task: Any,
    instance_seed: int,
    *,
    params: Mapping[str, Any],
    max_attempts: int,
) -> TaskOutput:
    """Run common scene plumbing after a public task prepares the objective."""

    selected_branch, branch_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=dict(params),
        supported_query_ids=tuple(str(value) for value in task.supported_query_ids),
        default_query_id=str(task.default_query_id),
        task_id=str(task.task_id),
    )
    plan = task.prepare_objective(int(instance_seed), task_params, str(selected_branch), branch_probabilities)
    _generation_defaults, rendering_defaults, prompt_defaults = split_scene_generation_rendering_prompt_defaults(
        raw_scene_defaults(),
        task_id=str(task.task_id),
    )
    last_error: Exception | None = None
    rendered = None
    ctx = None
    for attempt_index in range(max(1, int(max_attempts))):
        try:
            ctx = create_render_context(
                instance_seed=int(instance_seed) + int(attempt_index),
                params={**dict(task_params), "_render_attempt": int(attempt_index)},
                rendering_defaults=rendering_defaults,
            )
            rendered = render_regular_polygon_scene(ctx, plan.problem)
            break
        except Exception as exc:
            last_error = exc
            continue
    if rendered is None or ctx is None:
        raise RuntimeError(f"failed to generate {task.task_id}") from last_error

    image, noise_meta = apply_post_image_noise(
        rendered.image,
        instance_seed=int(instance_seed),
        params=task_params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    annotation_value = point_map_for_roles(rendered, plan.annotation_roles)
    annotation_gt = TypedValue(type="point_map", value=dict(annotation_value))
    _prompt_defaults, prompt_artifacts = regular_polygon_prompt_artifacts(
        prompt_defaults=prompt_defaults,
        prompt_task_key=str(plan.prompt_task_key),
        prompt_branch_key=str(plan.prompt_branch_key),
        annotation_roles=tuple(plan.annotation_roles),
        target_name=str(plan.problem.target_name),
        answer=plan.answer_gt.value,
        instance_seed=int(instance_seed),
    )
    trace_payload = _trace_payload(
        selected_branch=str(selected_branch),
        branch_probabilities=branch_probabilities,
        prompt_artifacts=prompt_artifacts,
        plan=plan,
        rendered=rendered,
        image_size=image.size,
        annotation_value=annotation_value,
        noise_meta=noise_meta,
        style_meta={
            "technical_diagram": dict(ctx.diagram_style_meta),
            "background": dict(ctx.background_meta),
            "single_object_scene_rotation": dict(ctx.scene_transform.metadata()),
        },
    )
    return TaskOutput(
        prompt=str(prompt_artifacts.prompt),
        answer_gt=plan.answer_gt,
        annotation_gt=annotation_gt,
        image=image,
        image_id="img0",
        trace_payload=trace_payload,
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
        query_id=str(selected_branch),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
    )


__all__ = ["RegularPolygonObjectivePlan", "run_regular_polygon_public_entry"]
