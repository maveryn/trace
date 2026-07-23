"""Neutral render/prompt/trace plumbing for circle-theorem scene tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec
from trace_tasks.tasks.shared.support_sampling import resolve_integer_choice

from .shared.annotations import circle_theorem_keyed_point_annotation
from .shared.defaults import GEN_DEFAULTS, RENDER_DEFAULTS
from .shared.prompts import build_circle_theorem_prompt_artifacts
from .shared.rendering import _render_base_scene
from .shared.state import CircleTheoremProblem, RenderedCircleTheoremScene, SCENE_ID


@dataclass(frozen=True)
class CircleTheoremRuntime:
    """Rendered scene and prompt/annotation artifacts after task binding."""

    rendered_scene: RenderedCircleTheoremScene
    annotation_artifacts: Any
    prompt_artifacts: Any
    prompt_defaults: Dict[str, Any]
    scene_payload: Dict[str, Any]


def render_circle_theorem_runtime(
    *,
    instance_seed: int,
    task_params: Mapping[str, Any],
    max_attempts: int,
    selected_query: str,
    problem: CircleTheoremProblem,
    build_scene_payload: Callable[[Any, CircleTheoremProblem], Dict[str, Any]],
    answer_hint_key: str,
    answer_example: int | float,
    annotation_hint_key: str = "annotation_hint_circle_points",
    object_description_key: str = "object_description",
) -> CircleTheoremRuntime:
    """Retry stochastic layout for one already-selected theorem construction."""

    rng = spawn_rng(int(instance_seed), f"geometry.{SCENE_ID}.{selected_query}.scene")
    last_error: Exception | None = None
    for _attempt in range(max(1, int(max_attempts))):
        try:
            scene_payload = build_scene_payload(rng, problem)
            rendered_scene = _render_base_scene(
                rng=rng,
                instance_seed=int(instance_seed),
                params=task_params,
                render_defaults=RENDER_DEFAULTS,
                point_model=scene_payload["point_model"],
                circle_center=scene_payload["circle_center"],
                circle_radius=float(scene_payload["circle_radius"]),
                segments=scene_payload["segments"],
                measurement_specs=scene_payload["measurement_specs"],
                support_measurement_tokens=scene_payload["support_measurement_tokens"],
                annotation_point_labels=scene_payload["annotation_point_labels"],
                annotation_values=scene_payload["annotation_values"],
                theorem_trace=scene_payload["theorem_trace"],
                angle_marker_specs=scene_payload.get("angle_marker_specs"),
                right_angle_marker_specs=scene_payload.get("right_angle_marker_specs"),
                circle_arc_specs=scene_payload.get("circle_arc_specs"),
            )
            annotation_artifacts = circle_theorem_keyed_point_annotation(rendered_scene)
            prompt_defaults, prompt_artifacts = build_circle_theorem_prompt_artifacts(
                prompt_query_key=str(selected_query),
                prompt_slots=dict(scene_payload.get("prompt_slots", {})),
                annotation_keys=rendered_scene.annotation_point_labels,
                answer_hint_key=str(answer_hint_key),
                answer_example=answer_example,
                annotation_hint_key=str(annotation_hint_key),
                object_description_key=str(object_description_key),
                instance_seed=int(instance_seed),
            )
            return CircleTheoremRuntime(
                rendered_scene=rendered_scene,
                annotation_artifacts=annotation_artifacts,
                prompt_artifacts=prompt_artifacts,
                prompt_defaults=dict(prompt_defaults),
                scene_payload=dict(scene_payload),
            )
        except Exception as exc:
            last_error = exc
            continue
    raise RuntimeError("failed to render circle theorem instance") from last_error


def circle_theorem_trace_payload(
    *,
    task_id: str,
    selected_query: str,
    query_probabilities: Mapping[str, float],
    problem: CircleTheoremProblem,
    runtime: CircleTheoremRuntime,
    answer_type: str,
    extra_query_params: Mapping[str, Any] | None = None,
    extra_execution_fields: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Assemble trace metadata from task-owned query and answer facts."""

    rendered = runtime.rendered_scene
    prompt_artifacts = runtime.prompt_artifacts
    selected_scene_payload = runtime.scene_payload
    annotation_value = dict(runtime.annotation_artifacts.value)
    query_params: Dict[str, Any] = {
        "query_id": str(selected_query),
        "query_id_probabilities": dict(query_probabilities),
        "target_answer": int(problem.target_answer),
        "target_answer_probabilities": dict(problem.target_answer_probabilities),
        **dict(extra_query_params or {}),
    }
    prompt_query_spec = build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts,
        query_id=str(selected_query),
        params=query_params,
    )
    prompt_query_spec["scene_id"] = SCENE_ID
    prompt_query_spec["task_id"] = str(task_id)
    theorem_trace = dict(rendered.theorem_trace)
    return {
        "scene_ir": {
            "scene_id": SCENE_ID,
            "task_id": str(task_id),
            "query_id": str(selected_query),
            "scene_kind": "geometry_circle_theorem",
            "entities": list(rendered.scene_entities),
            "relations": {
                "answer_segment": str(theorem_trace["answer_segment"]),
                "answer_value": rendered.answer_value,
                "theorem": str(theorem_trace["theorem"]),
            },
        },
        "query_spec": prompt_query_spec,
        "render_spec": {
            "scene_id": SCENE_ID,
            "task_id": str(task_id),
            "query_id": str(selected_query),
            "canvas_size": int(rendered.image.size[0]),
            "coord_space": "pixel",
            "background_style": dict(rendered.background_meta),
            "post_image_noise": dict(rendered.post_noise_meta),
            "shape_style": dict(rendered.shape_style),
            "prompt": {
                "prompt_variant": dict(prompt_artifacts.prompt_variant),
                "prompt_variant_active_key": str(prompt_artifacts.prompt_variant_active_key),
                "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
            },
            **dict(rendered.render_params),
        },
        "render_map": {
            "point_pixels": dict(rendered.point_pixels),
            "point_label_bboxes": dict(rendered.point_label_bboxes),
            "point_model": dict(rendered.point_model),
            "segment_pixels": dict(rendered.segment_pixels),
            "circle_center_pixel": list(rendered.circle_center_pixel),
            "circle_center_model": list(rendered.circle_center_model),
            "circle_radius_px": float(rendered.circle_radius_px),
            "circle_radius_model": float(rendered.circle_radius_model),
            "measurement_token_bboxes": dict(rendered.token_bboxes),
            "coord_space": "pixel",
        },
        "execution_trace": {
            "scene_id": SCENE_ID,
            "task_id": str(task_id),
            "query_id": str(selected_query),
            "query_id_probabilities": dict(query_probabilities),
            "target_answer": int(problem.target_answer),
            "target_answer_probabilities": dict(problem.target_answer_probabilities),
            "answer_type": str(answer_type),
            "answer_value": rendered.answer_value,
            "annotation_point_labels": list(rendered.annotation_point_labels),
            "support_measurement_tokens": list(rendered.support_measurement_tokens),
            "annotation_values": dict(rendered.annotation_values),
            **theorem_trace,
            **dict(extra_execution_fields or {}),
        },
        "witness_symbolic": {
            "task_id": str(task_id),
            "scene_id": SCENE_ID,
            "query_id": str(selected_query),
            "answer_segment": str(theorem_trace["answer_segment"]),
            "answer_value": rendered.answer_value,
            "annotation_point_labels": list(rendered.annotation_point_labels),
            "support_measurement_tokens": list(rendered.support_measurement_tokens),
            "source_witness_type": str(runtime.annotation_artifacts.annotation_type),
            "original_annotation_value": list(rendered.annotation_point_labels),
            "annotation_values": dict(rendered.annotation_values),
        },
        "projected_annotation": {
            **dict(runtime.annotation_artifacts.projected_annotation),
            "point_set": list(annotation_value.values()),
            "pixel_point_set": list(annotation_value.values()),
        },
    }


def run_circle_theorem_lifecycle(
    *,
    task_id: str,
    instance_seed: int,
    task_params: Mapping[str, Any],
    max_attempts: int,
    selected_query: str,
    query_probabilities: Mapping[str, float],
    problem: CircleTheoremProblem,
    build_scene_payload: Callable[[Any, CircleTheoremProblem], Dict[str, Any]],
    answer_type: str,
    answer_hint_key: str,
    answer_example: int | float,
    extra_query_params: Mapping[str, Any] | None = None,
    annotation_hint_key: str = "annotation_hint_circle_points",
    object_description_key: str = "object_description",
) -> TaskOutput:
    """Build the final task output after a public file binds objective facts."""

    runtime = render_circle_theorem_runtime(
        instance_seed=int(instance_seed),
        task_params=task_params,
        max_attempts=int(max_attempts),
        selected_query=str(selected_query),
        problem=problem,
        build_scene_payload=build_scene_payload,
        answer_hint_key=str(answer_hint_key),
        answer_example=answer_example,
        annotation_hint_key=str(annotation_hint_key),
        object_description_key=str(object_description_key),
    )
    if str(answer_type) == "integer":
        answer_value: int | float = int(runtime.rendered_scene.answer_value)
    else:
        answer_value = float(runtime.rendered_scene.answer_value)
    annotation_gt = TypedValue(
        type=str(runtime.annotation_artifacts.annotation_type),
        value=runtime.annotation_artifacts.value,
    )
    trace_payload = circle_theorem_trace_payload(
        task_id=str(task_id),
        selected_query=str(selected_query),
        query_probabilities=query_probabilities,
        problem=problem,
        runtime=runtime,
        answer_type=str(answer_type),
        extra_query_params=extra_query_params,
    )
    return TaskOutput(
        prompt=str(runtime.prompt_artifacts.prompt),
        answer_gt=TypedValue(type=str(answer_type), value=answer_value),
        annotation_gt=annotation_gt,
        image=runtime.rendered_scene.image,
        image_id="img0",
        trace_payload=trace_payload,
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
        query_id=str(selected_query),
        prompt_variants=dict(runtime.prompt_artifacts.prompt_variants),
    )


def run_integer_circle_theorem_task(
    *,
    task_id: str,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    supported_query_ids: tuple[str, ...],
    answer_support: tuple[int, ...] | Mapping[str, tuple[int, ...]],
    bind_problem: Callable[
        [int, Mapping[str, Any], str, int, Mapping[str, float]],
        tuple[CircleTheoremProblem, Mapping[str, Any], Callable[[Any, CircleTheoremProblem], Dict[str, Any]]],
    ],
) -> TaskOutput:
    """Run the common integer-valued theorem path after task-local binding."""

    selected_query, query_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=supported_query_ids,
        default_query_id=supported_query_ids[0],
        task_id=str(task_id),
    )
    support = (
        tuple(answer_support[str(selected_query)])
        if isinstance(answer_support, Mapping)
        else tuple(answer_support)
    )
    target_answer, target_probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=task_params,
        gen_defaults=GEN_DEFAULTS,
        support_key=f"{selected_query}_answer_support",
        explicit_key="target_answer",
        fallback_support=support,
        namespace=f"{task_id}.{selected_query}.target_answer",
        balanced_flag_key="balanced_answer_sampling",
        namespace_support_permutation=True,
    )
    problem, extra_query_params, scene_builder = bind_problem(
        int(instance_seed),
        task_params,
        str(selected_query),
        int(target_answer),
        dict(target_probabilities),
    )
    return run_circle_theorem_lifecycle(
        task_id=str(task_id),
        instance_seed=int(instance_seed),
        task_params=task_params,
        max_attempts=int(max_attempts),
        selected_query=str(selected_query),
        query_probabilities=dict(query_probabilities),
        problem=problem,
        build_scene_payload=scene_builder,
        answer_type="integer",
        answer_hint_key="answer_hint_integer",
        answer_example=int(problem.target_answer),
        extra_query_params=extra_query_params,
    )


def run_label_keyed_number_circle_theorem_task(
    *,
    task_id: str,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    query_id: str,
    answer_value: float,
    query_params: Mapping[str, Any],
    build_scene_payload: Callable[[Any], Dict[str, Any]],
    render_defaults: Mapping[str, Any],
    scene_kind: str,
    witness_type: str,
    object_description_key: str,
    answer_hint_key: str,
    annotation_hint_key: str,
) -> TaskOutput:
    """Render and serialize a number-answer task with visible-label annotation."""

    rng = spawn_rng(int(instance_seed), f"{task_id}.scene")
    rendered_scene: RenderedCircleTheoremScene | None = None
    scene_payload: Dict[str, Any] | None = None
    last_error: Exception | None = None
    for _attempt in range(max(1, int(max_attempts))):
        try:
            candidate_payload = build_scene_payload(rng)
            rendered_scene = _render_base_scene(
                rng=rng,
                instance_seed=int(instance_seed),
                params=params,
                render_defaults=render_defaults,
                point_model=candidate_payload["point_model"],
                circle_center=candidate_payload["circle_center"],
                circle_radius=float(candidate_payload["circle_radius"]),
                segments=candidate_payload["segments"],
                measurement_specs=candidate_payload["measurement_specs"],
                support_measurement_tokens=candidate_payload["support_measurement_tokens"],
                annotation_point_labels=candidate_payload["annotation_point_labels"],
                annotation_values=candidate_payload["annotation_values"],
                theorem_trace=candidate_payload["theorem_trace"],
                angle_marker_specs=candidate_payload.get("angle_marker_specs"),
                right_angle_marker_specs=candidate_payload.get("right_angle_marker_specs"),
                circle_arc_specs=candidate_payload.get("circle_arc_specs"),
            )
            scene_payload = dict(candidate_payload)
            break
        except Exception as exc:
            last_error = exc
            continue
    if rendered_scene is None or scene_payload is None:
        raise RuntimeError(f"failed to generate {task_id}") from last_error

    annotation_labels = tuple(
        str(label) for label in scene_payload["annotation_point_labels"]
    )
    annotation_keyed_points = {
        str(label): [
            round(float(rendered_scene.point_pixels[str(label)][0]), 3),
            round(float(rendered_scene.point_pixels[str(label)][1]), 3),
        ]
        for label in annotation_labels
    }
    annotation_keys = tuple(annotation_keyed_points)
    prompt_defaults, prompt_artifacts = build_circle_theorem_prompt_artifacts(
        prompt_query_key=str(query_id),
        prompt_slots=dict(scene_payload.get("prompt_slots", {})),
        annotation_keys=annotation_keys,
        answer_hint_key=str(answer_hint_key),
        answer_example=float(answer_value),
        annotation_hint_key=str(annotation_hint_key),
        object_description_key=str(object_description_key),
        instance_seed=int(instance_seed),
    )
    full_query_params = {
        "scene_id": SCENE_ID,
        "query_id": str(query_id),
        **dict(query_params),
        "annotation_point_labels": list(annotation_keys),
    }
    prompt_query_spec = build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts,
        query_id=str(query_id),
        params=full_query_params,
    )
    prompt_query_spec["scene_id"] = SCENE_ID
    annotation_points = [list(point) for point in annotation_keyed_points.values()]
    trace_payload: Dict[str, Any] = {
        "scene_ir": {
            "scene_kind": str(scene_kind),
            "scene_id": SCENE_ID,
            "entities": list(rendered_scene.scene_entities),
            "relations": {
                "query_id": str(query_id),
                "answer_segment": str(rendered_scene.theorem_trace["answer_segment"]),
                "answer_value": float(answer_value),
                "theorem": str(rendered_scene.theorem_trace["theorem"]),
                "annotation_point_labels": list(annotation_keys),
            },
        },
        "query_spec": prompt_query_spec,
        "render_spec": {
            "canvas_size": int(rendered_scene.image.size[0]),
            "coord_space": "pixel",
            "background_style": dict(rendered_scene.background_meta),
            "post_image_noise": dict(rendered_scene.post_noise_meta),
            "shape_style": dict(rendered_scene.shape_style),
            **dict(rendered_scene.render_params),
        },
        "render_map": {
            "point_pixels": dict(rendered_scene.point_pixels),
            "point_label_bboxes": dict(rendered_scene.point_label_bboxes),
            "point_model": dict(rendered_scene.point_model),
            "segment_pixels": dict(rendered_scene.segment_pixels),
            "circle_center_pixel": list(rendered_scene.circle_center_pixel),
            "circle_center_model": list(rendered_scene.circle_center_model),
            "circle_radius_px": float(rendered_scene.circle_radius_px),
            "circle_radius_model": float(rendered_scene.circle_radius_model),
            "measurement_token_bboxes": dict(rendered_scene.token_bboxes),
            "coord_space": "pixel",
        },
        "execution_trace": {
            "scene_id": SCENE_ID,
            "query_id": str(query_id),
            "answer_type": "number",
            "answer_value": float(answer_value),
            "answer_rounding": "one_decimal",
            "support_measurement_tokens": list(rendered_scene.support_measurement_tokens),
            "annotation_values": dict(rendered_scene.annotation_values),
            **dict(query_params),
            **dict(rendered_scene.theorem_trace),
            "annotation_point_labels": list(annotation_keys),
        },
        "witness_symbolic": {
            "type": str(witness_type),
            "scene_id": SCENE_ID,
            "query_id": str(query_id),
            "answer_segment": str(rendered_scene.theorem_trace["answer_segment"]),
            "answer_value": float(answer_value),
            "source_witness_type": "point_map",
            "original_annotation_value": dict(annotation_keyed_points),
            "annotation_point_labels": list(annotation_keys),
            "support_measurement_tokens": list(rendered_scene.support_measurement_tokens),
        },
        "projected_annotation": {
            "type": "point_map",
            "point_map": dict(annotation_keyed_points),
            "pixel_point_map": dict(annotation_keyed_points),
            "point_set": list(annotation_points),
            "pixel_point_set": list(annotation_points),
        },
    }
    return TaskOutput(
        prompt=str(prompt_artifacts.prompt),
        answer_gt=TypedValue(type="number", value=float(answer_value)),
        annotation_gt=TypedValue(type="point_map", value=dict(annotation_keyed_points)),
        image=rendered_scene.image,
        image_id="img0",
        trace_payload=trace_payload,
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
        query_id=str(query_id),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
    )


__all__ = [
    "CircleTheoremRuntime",
    "circle_theorem_trace_payload",
    "render_circle_theorem_runtime",
    "run_integer_circle_theorem_task",
    "run_label_keyed_number_circle_theorem_task",
    "run_circle_theorem_lifecycle",
]
