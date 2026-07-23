"""Scene-private artifact assembly for circular-sector formula tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from trace_tasks.core.types import TypedValue
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.shared.config_defaults import split_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import PromptTraceArtifacts, build_prompt_query_spec

from .shared.defaults import DOMAIN, POST_IMAGE_NOISE_DEFAULTS, SCENE_ID, raw_scene_defaults
from .shared.prompts import sector_prompt_artifacts
from .shared.rendering import create_render_context, render_sector_scene
from .shared.state import RenderedSectorScene, SectorObjectivePlan
from trace_tasks.tasks.geometry.shared.annotation_values import bbox_annotation_artifacts


@dataclass(frozen=True)
class SectorArtifact:
    """Generated scene artifact before the public task wraps it as TaskOutput."""

    prompt_artifacts: PromptTraceArtifacts
    image: Any
    annotation_value: Any
    projected_annotation: Mapping[str, Any]
    trace_payload: Mapping[str, Any]
    task_versions: Mapping[str, Any]
    rendered_scene: RenderedSectorScene


def scene_defaults_for_task(task_id: str) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    """Return split scene defaults for one public sector task id."""

    return split_scene_generation_rendering_prompt_defaults(raw_scene_defaults(), task_id=str(task_id))


def _trace_payload(
    *,
    task_id: str,
    selected_branch: str,
    branch_probabilities: Mapping[str, float],
    prompt_artifacts: PromptTraceArtifacts,
    plan: SectorObjectivePlan,
    rendered: RenderedSectorScene,
    image_size: tuple[int, int],
    annotation_value: Any,
    projected_annotation: Mapping[str, Any],
    noise_meta: Mapping[str, Any],
    style_meta: Mapping[str, Any],
) -> dict[str, Any]:
    """Serialize trace metadata while leaving objective semantics task-owned."""

    query_params = {
        "scene_id": SCENE_ID,
        "query_id": str(selected_branch),
        "query_id_probabilities": dict(branch_probabilities),
        **dict(plan.replay_params),
    }
    trace_values = {
        "answer": float(plan.problem.answer),
        "answer_type": str(plan.answer_type),
        "answer_rounding": "one_decimal",
        "annotation_roles": list(plan.annotation_roles),
        "target_kind": str(plan.problem.target_kind),
        "visible_measure_kind": str(plan.problem.visible_measure_kind),
        "visual_case": str(plan.problem.visual_case),
        **dict(rendered.witness),
        **dict(plan.trace_values),
    }
    return {
        "scene_ir": {
            "domain": DOMAIN,
            "scene_id": SCENE_ID,
            "task_id": str(task_id),
            "entities": [dict(entity) for entity in rendered.scene_entities],
            "relations": {
                "type": "sector_formula",
                "query_id": str(selected_branch),
                **dict(trace_values),
            },
        },
        "query_spec": build_prompt_query_spec(
            prompt_artifacts=prompt_artifacts,
            query_id=str(selected_branch),
            params=query_params,
        ),
        "render_spec": {
            "task_id": str(task_id),
            "scene_id": SCENE_ID,
            "query_id": str(selected_branch),
            "canvas": {"width": int(image_size[0]), "height": int(image_size[1])},
            "coord_space": "pixel",
            "style": dict(style_meta),
            "prompt": {
                "prompt_variant": dict(prompt_artifacts.prompt_variant),
                "prompt_variant_active_key": str(prompt_artifacts.prompt_variant_active_key),
                "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
            },
        },
        "render_map": {"coord_space": "pixel", **dict(rendered.render_map)},
        "execution_trace": {
            "task_id": str(task_id),
            "scene_id": SCENE_ID,
            "query_id": str(selected_branch),
            "query_id_probabilities": dict(branch_probabilities),
            "reasoning_steps": int(rendered.reasoning_steps),
            **dict(trace_values),
        },
        "witness_symbolic": {
            "task_id": str(task_id),
            "scene_id": SCENE_ID,
            "query_id": str(selected_branch),
            "source_witness_type": "bbox",
            "original_annotation_value": list(annotation_value),
            **dict(trace_values),
        },
        "projected_annotation": dict(projected_annotation),
    }


def build_sector_artifact(
    *,
    task_id: str,
    instance_seed: int,
    selected_branch: str,
    branch_probabilities: Mapping[str, float],
    task_params: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    prompt_defaults: Mapping[str, Any],
    plan: SectorObjectivePlan,
) -> SectorArtifact:
    """Build a rendered sector artifact from one task-owned objective plan."""

    ctx = create_render_context(
        instance_seed=int(instance_seed),
        params=dict(task_params),
        rendering_defaults=rendering_defaults,
    )
    rendered = render_sector_scene(ctx, plan.problem)
    image, noise_meta = apply_post_image_noise(
        rendered.image,
        instance_seed=int(instance_seed),
        params=task_params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    if len(tuple(plan.annotation_roles)) != 1:
        raise ValueError("sector tasks require exactly one scalar bbox annotation role")
    annotation_role = str(tuple(plan.annotation_roles)[0])
    if annotation_role not in rendered.annotation_bboxes:
        raise ValueError(f"sector render did not produce annotation role: {annotation_role}")
    annotation_artifacts = bbox_annotation_artifacts(rendered.annotation_bboxes[annotation_role])
    annotation_value = list(annotation_artifacts.value)
    _prompt_defaults, prompt_artifacts = sector_prompt_artifacts(
        prompt_defaults=prompt_defaults,
        prompt_task_key=str(plan.prompt_task_key),
        prompt_branch_key=str(plan.prompt_branch_key),
        answer=float(plan.problem.answer),
        instance_seed=int(instance_seed),
    )
    trace_payload = _trace_payload(
        task_id=str(task_id),
        selected_branch=str(selected_branch),
        branch_probabilities=branch_probabilities,
        prompt_artifacts=prompt_artifacts,
        plan=plan,
        rendered=rendered,
        image_size=image.size,
        annotation_value=annotation_value,
        projected_annotation=annotation_artifacts.projected_annotation,
        noise_meta=noise_meta,
        style_meta={
            "technical_diagram": dict(ctx.diagram_style_meta),
            "background": dict(ctx.background_meta),
            "sector_fill": dict(ctx.fill_style_meta),
            "post_image_noise": dict(noise_meta),
            "single_object_scene_rotation": dict(ctx.scene_transform.metadata()),
        },
    )
    return SectorArtifact(
        prompt_artifacts=prompt_artifacts,
        image=image,
        annotation_value=list(annotation_value),
        projected_annotation=dict(annotation_artifacts.projected_annotation),
        trace_payload=trace_payload,
        task_versions=default_task_versions(),
        rendered_scene=rendered,
    )


def run_sector_public_entry(
    task: Any,
    instance_seed: int,
    *,
    params: Mapping[str, Any],
    max_attempts: int,
) -> TaskOutput:
    """Run neutral scene plumbing after the public task resolves objective semantics."""

    selected_branch, branch_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=dict(params),
        supported_query_ids=tuple(str(value) for value in task.supported_query_ids),
        default_query_id=str(task.default_query_id),
        task_id=str(task.task_id),
        namespace=f"{task.task_id}.query",
    )
    _generation_defaults, rendering_defaults, prompt_defaults = scene_defaults_for_task(str(task.task_id))
    last_error: Exception | None = None
    artifact = None
    plan = None
    for attempt_index in range(max(1, int(max_attempts))):
        attempt_seed = int(instance_seed) + int(attempt_index)
        try:
            plan = task.prepare_objective(
                attempt_seed,
                str(selected_branch),
                dict(branch_probabilities),
                dict(task_params),
            )
            artifact = build_sector_artifact(
                task_id=str(task.task_id),
                instance_seed=attempt_seed,
                selected_branch=str(selected_branch),
                branch_probabilities=branch_probabilities,
                task_params={**dict(task_params), "_render_attempt": int(attempt_index)},
                rendering_defaults=rendering_defaults,
                prompt_defaults=prompt_defaults,
                plan=plan,
            )
            break
        except Exception as exc:
            last_error = exc
            continue
    if artifact is None or plan is None:
        raise RuntimeError(f"failed to generate {task.task_id}") from last_error
    return TaskOutput(
        prompt=str(artifact.prompt_artifacts.prompt),
        answer_gt=TypedValue(type=str(plan.answer_type), value=float(plan.problem.answer)),
        annotation_gt=TypedValue(type="bbox", value=list(artifact.annotation_value)),
        image=artifact.image,
        image_id="img0",
        trace_payload=dict(artifact.trace_payload),
        task_versions=dict(artifact.task_versions),
        scene_id=SCENE_ID,
        query_id=str(selected_branch),
        prompt_variants=dict(artifact.prompt_artifacts.prompt_variants),
    )


__all__ = [
    "SectorArtifact",
    "SectorObjectivePlan",
    "build_sector_artifact",
    "run_sector_public_entry",
    "scene_defaults_for_task",
]
