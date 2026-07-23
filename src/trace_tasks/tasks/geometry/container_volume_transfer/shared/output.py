"""Output primitives for container volume-transfer scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from PIL import Image

from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.shared.output_metadata import default_task_versions

from .annotations import annotation_bbox_map
from .defaults import DOMAIN, POST_IMAGE_NOISE_DEFAULTS, SCENE_ID, SCENE_KIND
from .prompts import container_volume_prompt_artifacts
from .relations import ContainerVolumeTaskBinding
from .rendering import render_container_volume_transfer_with_retries
from .state import RenderedScene, ResolvedProblem


@dataclass(frozen=True)
class ContainerVolumeArtifacts:
    """Rendered/prompted scene artifacts before public TaskOutput assembly."""

    prompt: str
    prompt_variants: dict[str, str]
    prompt_artifacts: Any
    image: Image.Image
    rendered: RenderedScene
    annotation_value: dict[str, list[float]]
    render_meta: dict[str, Any]
    noise_meta: dict[str, Any]
    task_versions: dict[str, str]
    scene_id: str


def prepare_container_volume_artifacts(
    *,
    prompt_query_key: str,
    problem: ResolvedProblem,
    binding: ContainerVolumeTaskBinding,
    render_defaults: Mapping[str, Any],
    prompt_defaults: Mapping[str, Any],
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    random_namespace: str,
) -> ContainerVolumeArtifacts:
    """Render one resolved problem and build prompt artifacts."""

    rendered, render_meta = render_container_volume_transfer_with_retries(
        problem=problem,
        instance_seed=int(instance_seed),
        params=params,
        render_defaults=render_defaults,
        max_attempts=int(max_attempts),
        random_namespace=str(random_namespace),
    )
    image, noise_meta = apply_post_image_noise(
        rendered.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    annotation_value = annotation_bbox_map(rendered, binding.annotation_keys)
    prompt_artifacts = container_volume_prompt_artifacts(
        prompt_defaults=prompt_defaults,
        task_key=str(binding.prompt_task_key),
        query_key=str(prompt_query_key),
        annotation_keys=binding.annotation_keys,
        answer_hint_key=str(binding.answer_hint_key),
        answer=problem.answer,
        instance_seed=int(instance_seed),
    )
    return ContainerVolumeArtifacts(
        prompt=str(prompt_artifacts.prompt),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
        prompt_artifacts=prompt_artifacts,
        image=image,
        rendered=rendered,
        annotation_value=dict(annotation_value),
        render_meta=dict(render_meta),
        noise_meta=dict(noise_meta),
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
    )


def container_volume_trace_payload(
    *,
    artifacts: ContainerVolumeArtifacts,
    query_spec: Mapping[str, Any],
    relations: Mapping[str, Any],
    execution_trace: Mapping[str, Any],
    witness_symbolic: Mapping[str, Any],
) -> dict[str, Any]:
    """Compose scene-neutral trace sections from public task-bound fields."""

    projected_annotation = {
        "type": "bbox_map",
        "bbox_map": dict(artifacts.annotation_value),
        "pixel_bbox_map": dict(artifacts.annotation_value),
    }
    return {
        "scene_ir": {
            "domain": DOMAIN,
            "scene_kind": SCENE_KIND,
            "scene_id": SCENE_ID,
            "entities": [dict(entity) for entity in artifacts.rendered.scene_entities],
            "relations": dict(relations),
        },
        "query_spec": dict(query_spec),
        "render_spec": {
            "canvas_size": [int(artifacts.image.size[0]), int(artifacts.image.size[1])],
            "coord_space": "pixel",
            "post_image_noise": dict(artifacts.noise_meta),
            "prompt": {
                "prompt_variant": dict(artifacts.prompt_artifacts.prompt_variant),
                "prompt_variant_active_key": str(artifacts.prompt_artifacts.prompt_variant_active_key),
                "prompt_variants": dict(artifacts.prompt_artifacts.prompt_variants_for_trace),
            },
            **dict(artifacts.render_meta),
        },
        "render_map": dict(artifacts.rendered.render_map),
        "execution_trace": {
            "scene_id": SCENE_ID,
            **dict(execution_trace),
        },
        "witness_symbolic": {
            "type": "container_volume_transfer",
            "source_witness_type": "bbox_map",
            "original_annotation_value": dict(artifacts.annotation_value),
            **dict(witness_symbolic),
        },
        "projected_annotation": projected_annotation,
    }


__all__ = [
    "ContainerVolumeArtifacts",
    "container_volume_trace_payload",
    "prepare_container_volume_artifacts",
]
