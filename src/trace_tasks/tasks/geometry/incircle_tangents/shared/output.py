"""Trace payload assembly for incircle-tangent task outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from PIL import Image

from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.geometry.shared.annotation_values import PixelAnnotationArtifacts
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import PromptTraceArtifacts

from .annotations import incircle_point_annotation
from .defaults import POST_IMAGE_NOISE_DEFAULTS
from .prompts import incircle_prompt_artifacts
from .rendering import render_incircle_scene_with_retries
from .state import IncircleDiagramSpec, RenderedIncircleScene


@dataclass(frozen=True)
class IncircleTaskParts:
    """Non-TaskOutput fields prepared after public objective binding."""

    prompt: str
    prompt_variants: dict[str, str]
    image: Image.Image
    prompt_artifacts: PromptTraceArtifacts
    rendered: RenderedIncircleScene
    annotation_artifacts: PixelAnnotationArtifacts
    render_meta: dict[str, Any]
    noise_meta: dict[str, Any]
    task_versions: dict[str, str]


def prepare_incircle_task_parts(
    *,
    random_namespace: str,
    prompt_key: str,
    spec: IncircleDiagramSpec,
    render_defaults: Mapping[str, Any],
    prompt_defaults: Mapping[str, Any],
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
) -> IncircleTaskParts:
    """Prepare shared artifacts without constructing the public TaskOutput."""

    rendered, render_meta = render_incircle_scene_with_retries(
        random_namespace=str(random_namespace),
        spec=spec,
        instance_seed=int(instance_seed),
        params=params,
        render_defaults=render_defaults,
        max_attempts=int(max_attempts),
    )
    image, noise_meta = apply_post_image_noise(
        rendered.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    _prompt_defaults, prompt_artifacts = incircle_prompt_artifacts(
        prompt_defaults=prompt_defaults,
        prompt_query_key=str(prompt_key),
        instance_seed=int(instance_seed),
    )
    annotation_artifacts = incircle_point_annotation(rendered)
    return IncircleTaskParts(
        prompt=str(prompt_artifacts.prompt),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
        image=image,
        prompt_artifacts=prompt_artifacts,
        rendered=rendered,
        annotation_artifacts=annotation_artifacts,
        render_meta=dict(render_meta),
        noise_meta=dict(noise_meta),
        task_versions=default_task_versions(),
    )


__all__ = ["IncircleTaskParts", "prepare_incircle_task_parts"]
