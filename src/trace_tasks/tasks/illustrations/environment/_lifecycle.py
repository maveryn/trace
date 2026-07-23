"""Neutral lifecycle plumbing for environment illustration count tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Tuple

from ...base import TaskOutput
from ...shared.annotation_artifacts import bbox_set_annotation_artifacts
from ...shared.output_metadata import default_task_versions
from ....core.seed import spawn_rng
from ....core.types import TypedValue

from .shared.annotations import feature_bbox_map, feature_path_map
from .shared.output import serialize_environment_objects
from .shared.prompts import render_environment_prompt
from .shared.rendering import ENVIRONMENT_THEME_IDS, environment_scene_entities, render_environment_object_scene
from .shared.sampling import capped_object_count_probabilities, environment_render_params, style_weights
from .shared.state import BoundCountResult, EnvironmentChoice


SCENE_ID = "environment"


@dataclass(frozen=True)
class EnvironmentCountPlan:
    """Public-owned hooks for one environment count objective."""

    public_id: str
    local_query_id: str
    seed_namespace: str
    prompt_query_key: str
    resolve_choice: Callable[[int, Mapping[str, Any], Mapping[str, Any]], EnvironmentChoice]
    object_count_sampler: Callable[[Mapping[str, Any], int, EnvironmentChoice, Mapping[str, Any]], Tuple[int, Dict[str, float]]]
    target_count_sampler: Callable[[Mapping[str, Any], int, EnvironmentChoice, Mapping[str, Any]], Tuple[int, Dict[str, float]]]
    render_overrides: Callable[[Mapping[str, Any], EnvironmentChoice, int, int], Dict[str, Any]]
    bind_result: Callable[
        [Any, EnvironmentChoice, Mapping[str, list[float]], Mapping[str, list[float]], int],
        BoundCountResult,
    ]
    prompt_slots: Callable[[Mapping[str, Any], EnvironmentChoice, BoundCountResult, Any], Dict[str, Any]]
    render_fallback: Mapping[str, Any]


def run_environment_count_lifecycle(
    *,
    plan: EnvironmentCountPlan,
    domain: str,
    generation_defaults: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    prompt_defaults: Mapping[str, Any],
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
) -> TaskOutput:
    """Render one scene, bind task-owned witnesses, and package `TaskOutput`."""

    choice = plan.resolve_choice(int(instance_seed), dict(params), generation_defaults)
    requested_object_count, requested_object_count_probabilities = plan.object_count_sampler(
        dict(params),
        int(instance_seed),
        choice,
        generation_defaults,
    )
    target_count, target_count_probabilities = plan.target_count_sampler(
        dict(params),
        int(instance_seed),
        choice,
        generation_defaults,
    )
    object_count_probabilities = capped_object_count_probabilities(
        requested_object_count_probabilities,
        choice.theme_probabilities,
    )
    render_params = environment_render_params(
        dict(params),
        rendering_defaults,
        fallback=plan.render_fallback,
        instance_seed=int(instance_seed),
        namespace=f"{plan.seed_namespace}:canvas_profile",
    )

    scene = None
    bound_result: BoundCountResult | None = None
    last_error: Exception | None = None
    for attempt in range(max(1, int(max_attempts))):
        try:
            scene_rng = spawn_rng(int(instance_seed), str(plan.seed_namespace), int(attempt))
            overrides = plan.render_overrides(dict(params), choice, int(requested_object_count), int(target_count))
            scene = render_environment_object_scene(
                rng=scene_rng,
                canvas_width=int(render_params["canvas_width"]),
                canvas_height=int(render_params["canvas_height"]),
                object_count=int(requested_object_count),
                render_scale=int(render_params["render_scale"]),
                theme_weights={theme: (1.0 if theme == choice.theme_id else 0.0) for theme in ENVIRONMENT_THEME_IDS},
                style_weights=style_weights(dict(params), rendering_defaults),
                object_size_min_px=int(render_params["object_size_min_px"]),
                object_size_max_px=int(render_params["object_size_max_px"]),
                min_gap_px=int(render_params["min_gap_px"]),
                max_overlap_fraction=float(render_params["max_overlap_fraction"]),
                placement_max_attempts=int(render_params["placement_max_attempts"]),
                skyline_building_min=int(render_params["skyline_building_min"]),
                skyline_building_max=int(render_params["skyline_building_max"]),
                **overrides,
            )
            _serialized_objects, object_bboxes, part_bboxes = serialize_environment_objects(scene)
            feature_bboxes = feature_bbox_map(scene)
            bound_result = plan.bind_result(scene, choice, object_bboxes, feature_bboxes, int(target_count))
            break
        except Exception as exc:  # pragma: no cover - random placement feasibility is retry based
            last_error = exc
            scene = None
            bound_result = None
    if scene is None or bound_result is None:
        raise RuntimeError(f"could not generate environment task instance: {last_error}") from last_error

    serialized_objects, object_bboxes, part_bboxes = serialize_environment_objects(scene)
    feature_bboxes = feature_bbox_map(scene)
    feature_paths = feature_path_map(scene)
    dynamic_slots = plan.prompt_slots(prompt_defaults, choice, bound_result, scene)
    annotation_artifacts = bbox_set_annotation_artifacts(bound_result.annotation_value)
    prompt_artifacts = render_environment_prompt(
        domain=str(domain),
        scene_id=SCENE_ID,
        prompt_defaults=prompt_defaults,
        dynamic_slots=dynamic_slots,
        instance_seed=int(instance_seed),
        preferred_mode="answer_and_annotation",
    )
    answer = int(bound_result.answer)
    trace_payload = {
        "scene_ir": {
            "domain": str(domain),
            "scene_id": SCENE_ID,
            "entities": environment_scene_entities(scene),
            "relations": {
                "query_id": str(plan.local_query_id),
                **dict(bound_result.scene_relations),
            },
        },
        "query_spec": {
            "task_id": str(plan.public_id),
            "query_id": str(plan.local_query_id),
            "prompt_query_key": str(plan.prompt_query_key),
            "prompt_variant_active_key": prompt_artifacts.prompt_variant_active_key,
            "prompt_variant": dict(prompt_artifacts.prompt_variant),
            "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
            "params": {
                "theme": str(choice.theme_id),
                "theme_id": str(choice.theme_id),
                "target_count": int(answer),
                "target_count_probabilities": dict(target_count_probabilities),
                "object_count": int(len(scene.placements)),
                "requested_object_count": int(requested_object_count),
                "theme_probabilities": dict(choice.theme_probabilities),
                "object_count_probabilities": dict(object_count_probabilities),
                **dict(bound_result.operand_params),
            },
        },
        "render_spec": {
            "canvas_size": [int(scene.canvas_width), int(scene.canvas_height)],
            "coord_space": "pixel",
            "scene_id": SCENE_ID,
            "style": {
                "theme_id": str(scene.theme_id),
                "layout": dict(scene.layout),
                "style_id": str(scene.style_id),
                "render_scale": int(scene.render_scale),
            },
        },
        "render_map": {
            "image_id": "img0",
            "object_bboxes_px": object_bboxes,
            "part_bboxes_px": part_bboxes,
            "feature_bboxes_px": feature_bboxes,
            "feature_paths_px": feature_paths,
            **dict(bound_result.render_map_extra),
        },
        "execution_trace": {
            "query_id": str(plan.local_query_id),
            "prompt_query_key": str(plan.prompt_query_key),
            "scene_id": SCENE_ID,
            "theme_id": str(choice.theme_id),
            "target_count": int(answer),
            "object_count": int(len(scene.placements)),
            "requested_object_count": int(requested_object_count),
            **dict(bound_result.execution_extra),
        },
        "witness_symbolic": dict(bound_result.witness_symbolic),
        "projected_annotation": dict(annotation_artifacts.projected_annotation),
    }
    if serialized_objects:
        trace_payload["execution_trace"]["objects"] = serialized_objects
    return TaskOutput(
        prompt=str(prompt_artifacts.prompt),
        prompt_variants={str(key): str(value) for key, value in prompt_artifacts.prompt_variants.items()},
        answer_gt=TypedValue(type="integer", value=int(answer)),
        annotation_gt=annotation_artifacts.annotation_gt,
        image=scene.image,
        image_id="img0",
        trace_payload=trace_payload,
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
        query_id=str(plan.local_query_id),
    )


__all__ = [
    "BoundCountResult",
    "EnvironmentChoice",
    "EnvironmentCountPlan",
    "run_environment_count_lifecycle",
]
