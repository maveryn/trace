"""Neutral render lifecycle helpers for park/playground tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence, Tuple

from ....core.seed import spawn_rng
from ....core.types import TypedValue
from ...base import TaskOutput
from ...shared.annotation_artifacts import bbox_set_annotation_artifacts
from ...shared.config_defaults import required_group_defaults
from ...shared.output_metadata import default_task_versions
from .shared.output import park_render_spec, park_scene_ir
from .shared.prompts import build_park_prompt_artifacts
from .shared.rendering import render_park_playground_scene
from .shared.sampling import render_params, setting_weights, style_weights
from .shared.state import ParkCountBinding, ParkEquipmentSpec, ParkPersonSpec, RenderedParkPlaygroundScene


@dataclass(frozen=True)
class ParkCountPlan:
    """Public-owned hooks for one park/playground count objective."""

    public_id: str
    prompt_query_key: str
    sample_spec: Callable[[int, Mapping[str, Any], int], Any]
    person_specs: Callable[[Any], Sequence[ParkPersonSpec]]
    equipment_specs: Callable[[Any], Sequence[ParkEquipmentSpec] | None]
    required_zones: Callable[[Any], Sequence[str]]
    bind_result: Callable[[RenderedParkPlaygroundScene, Any, Mapping[str, Any]], ParkCountBinding]
    fallback_width: int
    fallback_height: int
    fallback_scale: int


def render_scene_with_retries(
    *,
    namespace: str,
    instance_seed: int,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    fallback_width: int,
    fallback_height: int,
    fallback_scale: int,
    max_attempts: int,
    person_specs: Sequence[ParkPersonSpec],
    equipment_specs: Sequence[ParkEquipmentSpec] | None = None,
    required_zones: Sequence[str] = (),
) -> RenderedParkPlaygroundScene:
    """Render a scene from already-sampled semantic specs, retrying layout only."""

    last_error: Exception | None = None
    for attempt in range(max(1, int(max_attempts))):
        try:
            scene_rng = spawn_rng(int(instance_seed), f"{namespace}:scene", int(attempt))
            rp = render_params(
                params,
                render_defaults,
                fallback_width=int(fallback_width),
                fallback_height=int(fallback_height),
                fallback_scale=int(fallback_scale),
                instance_seed=int(instance_seed),
                namespace=f"{namespace}:canvas_profile",
            )
            return render_park_playground_scene(
                rng=scene_rng,
                person_specs=tuple(person_specs),
                equipment_specs=None if equipment_specs is None else tuple(equipment_specs),
                required_zones=tuple(str(value) for value in required_zones),
                canvas_width=int(rp["canvas_width"]),
                canvas_height=int(rp["canvas_height"]),
                render_scale=int(rp["render_scale"]),
                setting_weights=setting_weights(params, render_defaults),
                style_weights=style_weights(params, render_defaults),
            )
        except Exception as exc:  # pragma: no cover - retry surface is seed/layout dependent.
            last_error = exc
    raise RuntimeError(f"could not render park/playground scene: {last_error}") from last_error


def compose_count_result(
    *,
    task: Any,
    scene: RenderedParkPlaygroundScene,
    prompt_defaults: Mapping[str, Any],
    prompt_required_keys: Sequence[str],
    prompt_query_key: str,
    slots: Mapping[str, Any],
    instance_seed: int,
    answer: int,
    annotation_value: Sequence[Sequence[float]],
    render_map: Mapping[str, Any],
    scene_relations: Mapping[str, Any],
    branch_params: Mapping[str, Any],
    execution_trace: Mapping[str, Any],
    witness_symbolic: Mapping[str, Any],
    scene_entities: Sequence[Mapping[str, Any]],
) -> TaskOutput:
    """Assemble the common count-task prompt, trace payload, and return value."""

    required_defaults = required_group_defaults(
        prompt_defaults,
        list(prompt_required_keys),
        context=f"prompt defaults for {task.task_id}",
    )
    prompt_artifacts = build_park_prompt_artifacts(
        domain=str(task.domain),
        scene_id="park_playground",
        prompt_defaults=required_defaults,
        prompt_query_key=str(prompt_query_key),
        slots=dict(slots),
        instance_seed=int(instance_seed),
    )
    annotation_artifacts = bbox_set_annotation_artifacts(annotation_value)
    branch_metadata = dict(branch_params)
    public_query_id = str(branch_metadata.pop("branch_id", "single"))
    public_scene_relations = dict(scene_relations)
    public_scene_relations.pop("branch_id", None)
    public_scene_relations["query_id"] = public_query_id
    public_query_params = {"query_id": public_query_id, **branch_metadata}
    public_execution_trace = dict(execution_trace)
    public_execution_trace.pop("branch_id", None)
    public_execution_trace["query_id"] = public_query_id
    trace_payload = {
        "scene_ir": park_scene_ir(
            domain=str(task.domain),
            scene_id="park_playground",
            entities=[dict(entity) for entity in scene_entities],
            relations=public_scene_relations,
        ),
        "query_spec": {
            "task_id": str(task.task_id),
            "query_id": public_query_id,
            "prompt_variant_active_key": prompt_artifacts.prompt_variant_active_key,
            "prompt_variant": dict(prompt_artifacts.prompt_variant),
            "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
            "params": public_query_params,
        },
        "render_spec": park_render_spec(scene),
        "render_map": dict(render_map),
        "execution_trace": public_execution_trace,
        "witness_symbolic": dict(witness_symbolic),
        "projected_annotation": dict(annotation_artifacts.projected_annotation),
    }
    return TaskOutput(
        prompt=str(prompt_artifacts.prompt),
        prompt_variants={str(key): str(value) for key, value in prompt_artifacts.prompt_variants.items()},
        answer_gt=TypedValue(type="integer", value=int(answer)),
        annotation_gt=annotation_artifacts.annotation_gt,
        image=scene.image,
        image_id="img0",
        trace_payload=trace_payload,
        task_versions=default_task_versions(),
        scene_id="park_playground",
        query_id=public_query_id,
    )


def run_park_count_lifecycle(
    *,
    task: Any,
    plan: ParkCountPlan,
    rendering_defaults: Mapping[str, Any],
    prompt_defaults: Mapping[str, Any],
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
) -> TaskOutput:
    """Run neutral render/compose plumbing around task-owned sampling and binding hooks."""

    sample = plan.sample_spec(instance_seed=int(instance_seed), params=dict(params), attempt_index=0)
    scene = render_scene_with_retries(
        namespace=str(plan.public_id),
        instance_seed=int(instance_seed),
        params=params,
        render_defaults=rendering_defaults,
        fallback_width=int(plan.fallback_width),
        fallback_height=int(plan.fallback_height),
        fallback_scale=int(plan.fallback_scale),
        max_attempts=int(max_attempts),
        person_specs=tuple(plan.person_specs(sample)),
        equipment_specs=plan.equipment_specs(sample),
        required_zones=tuple(plan.required_zones(sample)),
    )
    bound = plan.bind_result(scene, sample, prompt_defaults)
    return compose_count_result(
        task=task,
        scene=scene,
        prompt_defaults=bound.prompt_defaults,
        prompt_required_keys=tuple(bound.prompt_defaults.keys()),
        prompt_query_key=str(plan.prompt_query_key),
        slots=bound.slots,
        instance_seed=int(instance_seed),
        answer=int(bound.answer),
        annotation_value=bound.annotation_value,
        render_map=bound.render_map,
        scene_relations=bound.scene_relations,
        branch_params=bound.branch_params,
        execution_trace=bound.execution_trace,
        witness_symbolic=bound.witness_symbolic,
        scene_entities=bound.scene_entities,
    )


__all__ = ["ParkCountPlan", "compose_count_result", "render_scene_with_retries", "run_park_count_lifecycle"]
