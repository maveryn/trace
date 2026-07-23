"""Identity-free trace helpers for volume-equivalence conversion scenes."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from PIL import Image

from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.shared.prompt_variants import PromptTraceArtifacts

from .annotations import annotation_bbox, annotation_bbox_map
from .construction import solid_volume
from .defaults import DOMAIN, POST_IMAGE_NOISE_DEFAULTS, SCENE_ID
from .prompts import volume_equivalence_prompt_artifacts
from .rendering import render_volume_equivalence_with_retries
from .state import RenderedScene, ResolvedProblem


@dataclass(frozen=True)
class PreparedVolumeEquivalenceArtifacts:
    """Rendered diagram artifacts after a public task chooses objective data."""

    image: Image.Image
    rendered: RenderedScene
    annotation_value: Any
    prompt_artifacts: PromptTraceArtifacts
    noise_meta: dict[str, Any]


def prepare_volume_equivalence_artifacts(
    *,
    problem: ResolvedProblem,
    render_scene: Callable[..., RenderedScene],
    annotation_keys: Sequence[str],
    annotation_schema: str = "bbox_map",
    prompt_task_key: str,
    prompt_branch_key: str,
    answer: int | str,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    render_defaults: Mapping[str, Any],
    prompt_defaults: Mapping[str, Any],
    random_namespace: str,
) -> PreparedVolumeEquivalenceArtifacts:
    """Render, annotate, and compose prompts without choosing task semantics."""

    rendered = render_volume_equivalence_with_retries(
        problem=problem,
        render_scene=render_scene,
        instance_seed=int(instance_seed),
        params=params,
        max_attempts=int(max_attempts),
        render_defaults=render_defaults,
        random_namespace=str(random_namespace),
    )
    image, noise_meta = apply_post_image_noise(
        rendered.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    if str(annotation_schema) == "bbox":
        if len(annotation_keys) != 1:
            raise ValueError("bbox annotations require exactly one annotation key")
        annotation_value = annotation_bbox(rendered, str(annotation_keys[0]))
    else:
        annotation_value = annotation_bbox_map(rendered, annotation_keys)
    _prompt_defaults, prompt_artifacts = volume_equivalence_prompt_artifacts(
        prompt_defaults=prompt_defaults,
        prompt_task_key=str(prompt_task_key),
        prompt_branch_key=str(prompt_branch_key),
        annotation_keys=annotation_keys,
        annotation_schema=str(annotation_schema),
        answer=answer,
        instance_seed=int(instance_seed),
    )
    return PreparedVolumeEquivalenceArtifacts(
        image=image,
        rendered=rendered,
        annotation_value=annotation_value,
        prompt_artifacts=prompt_artifacts,
        noise_meta=dict(noise_meta),
    )


def common_trace_sections(
    *,
    problem: ResolvedProblem,
    rendered: RenderedScene,
    annotation_keys: Sequence[str],
    noise_meta: Mapping[str, Any],
    image_size: tuple[int, int],
    option_count: int,
) -> dict[str, Any]:
    """Build scene-level trace sections before a public task binds identity."""

    render_style = dict(rendered.render_meta.get("style", {}))
    if render_style:
        render_style["post_image_noise"] = dict(noise_meta)
    trace = {
        "scene_ir": {
            "domain": DOMAIN,
            "scene_id": SCENE_ID,
            "entities": [dict(entity) for entity in rendered.scene_entities],
            "relations": {
                "type": str(problem.formula_family),
                "source_volume": int(solid_volume(problem.source)),
                "target_volume": int(solid_volume(problem.target)),
                "annotation_roles": [str(key) for key in annotation_keys],
            },
        },
        "render_spec": {
            "scene_id": SCENE_ID,
            "canvas": {"width": int(image_size[0]), "height": int(image_size[1])},
            "option_count": int(option_count),
            "style": render_style,
        },
        "render_map": dict(rendered.render_map),
        "execution_trace": {
            "scene_id": SCENE_ID,
            "formula_family": str(problem.formula_family),
            "formula": str(problem.formula),
            "source_shape": problem.source.shape,
            "target_shape": problem.target.shape,
            "source_volume": int(solid_volume(problem.source)),
            "target_volume": int(solid_volume(problem.target)),
            "target_unknown_role": str(problem.target_unknown_role),
            "annotation_roles": [str(key) for key in annotation_keys],
        },
    }
    if problem.option_count_probabilities:
        trace["render_spec"]["option_count_probabilities"] = dict(problem.option_count_probabilities)
        trace["execution_trace"]["option_count_probabilities"] = dict(problem.option_count_probabilities)
    return trace


def prompt_render_spec(prompt_artifacts: PromptTraceArtifacts) -> dict[str, Any]:
    """Serialize prompt variant metadata for a render spec."""

    return {
        "prompt_variant": dict(prompt_artifacts.prompt_variant),
        "prompt_variant_active_key": str(prompt_artifacts.prompt_variant_active_key),
        "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
    }


__all__ = [
    "PreparedVolumeEquivalenceArtifacts",
    "common_trace_sections",
    "prepare_volume_equivalence_artifacts",
    "prompt_render_spec",
]
