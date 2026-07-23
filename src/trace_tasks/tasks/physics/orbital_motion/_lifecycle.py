"""Private scene assembly for orbital-motion option-label objectives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Sequence

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.shared.config_defaults import (
    load_scene_generation_rendering_prompt_defaults,
    required_group_defaults,
)
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_json_example import build_prompt_json_examples
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec

from .shared.annotations import build_point_annotation_payload
from .shared.output import build_render_spec
from .shared.prompts import (
    PROMPT_BUNDLE_ID as DEFAULT_PROMPT_BUNDLE_ID,
    build_orbital_motion_prompt_artifacts,
)
from .shared.rendering import render_orbit_scene
from .shared.state import SCENE_ID, SCENE_NAMESPACE, OrbitRenderDefaults, OrbitSpec


AnnotationBuilder = Callable[[OrbitSpec], Sequence[float]]
SpecBuilder = Callable[[int, Mapping[str, Any], Mapping[str, Any]], OrbitSpec]


@dataclass(frozen=True)
class OrbitLifecyclePlan:
    """Task-owned semantic bindings consumed by the common orbit render/output lifecycle."""

    task_identifier: str
    namespace: str
    prompt_task_key: str
    prompt_query_key: str
    public_query_id: str
    spec_builder: SpecBuilder
    annotation_builder: AnnotationBuilder
    query_probabilities: Mapping[str, float]
    execution_params: Mapping[str, Any]


def run_orbit_lifecycle(
    *,
    domain: str,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    plan: OrbitLifecyclePlan,
) -> TaskOutput:
    """Render one orbit scene and assemble the task-owned output payload."""

    _ = int(max_attempts)
    task_params = dict(params or {})
    _gen_defaults, render_defaults, prompt_defaults_group = load_scene_generation_rendering_prompt_defaults(
        "physics",
        SCENE_ID,
        task_id=str(plan.task_identifier),
    )
    spec = plan.spec_builder(int(instance_seed), task_params, render_defaults)
    rendered = render_orbit_scene(
        instance_seed=int(instance_seed),
        params=task_params,
        spec=spec,
        render_defaults=render_defaults,
        fallback=OrbitRenderDefaults(),
        namespace=SCENE_NAMESPACE,
    )
    answer_gt = TypedValue(type="option_letter", value=str(spec.selected_label))
    annotation_gt = TypedValue(type="point", value=list(plan.annotation_builder(spec)))
    json_example, json_example_answer_only = build_prompt_json_examples(
        annotation_value=annotation_gt.value,
        answer_type=str(answer_gt.type),
    )
    prompt_defaults = required_group_defaults(
        prompt_defaults_group,
        ("bundle_id", "task_key"),
        context=f"prompt defaults for {plan.task_identifier}",
    )
    prompt_artifacts = build_orbital_motion_prompt_artifacts(
        domain=str(domain),
        bundle_id=str(prompt_defaults.get("bundle_id", DEFAULT_PROMPT_BUNDLE_ID)),
        task_key=str(prompt_defaults.get("task_key", plan.prompt_task_key)),
        prompt_query_key=str(plan.prompt_query_key),
        dynamic_slots={
            "json_example": str(json_example),
            "json_example_answer_only": str(json_example_answer_only),
        },
        instance_seed=int(instance_seed),
    )
    query_params = {
        "query_id": str(plan.public_query_id),
        "prompt_query_key": str(plan.prompt_query_key),
        "target_answer": str(spec.selected_label),
        **dict(plan.execution_params),
    }
    if plan.query_probabilities:
        query_params["query_id_probabilities"] = dict(plan.query_probabilities)
    query_spec = build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts,
        query_id=str(plan.public_query_id),
        params=query_params,
    )
    annotation_payload = build_point_annotation_payload(annotation_gt.value)
    trace_payload = {
        "scene_ir": {
            "scene_kind": "physics_orbital_motion_ellipse",
            "entities": [dict(entity) for entity in rendered.scene_entities],
            "relations": {
                "query_id": str(plan.public_query_id),
                "prompt_query_key": str(plan.prompt_query_key),
                "selected_label": str(spec.selected_label),
                "selected_point": list(annotation_gt.value),
                "eccentricity": round(float(spec.eccentricity), 4),
                **dict(plan.execution_params),
            },
        },
        "query_spec": query_spec,
        "render_spec": build_render_spec(rendered, scope="orbital_motion_diagram"),
        "render_map": dict(rendered.render_map),
        "execution_trace": {
            "query_id": str(plan.public_query_id),
            "prompt_query_key": str(plan.prompt_query_key),
            "target_answer": str(spec.selected_label),
            "candidate_points": dict(rendered.render_map["candidate_points"]),
            "focus_side": int(spec.focus_side),
            **dict(plan.execution_params),
        },
        "sampling": {
            "query_id_probabilities": dict(plan.query_probabilities),
        },
        "background": dict(rendered.render_map.get("background_style", {})),
        "post_image_noise": dict(rendered.render_map.get("post_image_noise", {})),
        **annotation_payload,
    }
    return TaskOutput(
        prompt=str(prompt_artifacts.prompt),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
        answer_gt=answer_gt,
        annotation_gt=annotation_gt,
        image=rendered.image,
        image_id="img0",
        trace_payload=trace_payload,
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
        query_id=str(plan.public_query_id),
    )


__all__ = ["OrbitLifecyclePlan", "run_orbit_lifecycle"]
