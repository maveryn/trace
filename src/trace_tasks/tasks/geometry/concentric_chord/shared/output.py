"""Identity-free output sections for concentric-chord diagrams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from PIL import Image

from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.geometry.shared.annotation_values import PixelAnnotationArtifacts

from .annotations import concentric_chord_annotation
from .defaults import POST_IMAGE_NOISE_DEFAULTS, SCENE_ID, SCENE_KIND, SCENE_VARIANT
from .prompts import concentric_chord_prompt_artifacts
from .rendering import render_concentric_chord_with_retries
from .state import ConcentricChordDiagramSpec, RenderedConcentricChordScene


@dataclass(frozen=True)
class ConcentricChordArtifacts:
    """Rendered artifacts after the public task has selected the objective case."""

    rendered: RenderedConcentricChordScene
    image: Image.Image
    render_meta: dict[str, Any]
    noise_meta: dict[str, Any]
    prompt_artifacts: Any
    annotation_artifacts: PixelAnnotationArtifacts


def prepare_concentric_chord_artifacts(
    *,
    spec: ConcentricChordDiagramSpec,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    render_defaults: Mapping[str, Any],
    prompt_defaults: Mapping[str, Any],
    prompt_query_key: str,
    random_namespace: str,
) -> ConcentricChordArtifacts:
    """Render the scene and prompt without selecting answer or public identity."""

    rendered, render_meta = render_concentric_chord_with_retries(
        spec=spec,
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
    _prompt_defaults, prompt_artifacts = concentric_chord_prompt_artifacts(
        prompt_defaults=prompt_defaults,
        prompt_query_key=str(prompt_query_key),
        instance_seed=int(instance_seed),
    )
    return ConcentricChordArtifacts(
        rendered=rendered,
        image=image,
        render_meta=dict(render_meta),
        noise_meta=dict(noise_meta),
        prompt_artifacts=prompt_artifacts,
        annotation_artifacts=concentric_chord_annotation(rendered),
    )


def concentric_measurement_fields(spec: ConcentricChordDiagramSpec) -> dict[str, Any]:
    """Serialize formula inputs from a task-bound concentric-chord spec."""

    return {
        "formula_family": "concentric_circle_tangent_chord",
        "unknown_measure": str(spec.unknown_measure),
        "outer_radius": int(spec.outer_radius),
        "inner_radius": int(spec.inner_radius),
        "half_chord": int(spec.half_chord),
        "chord_length": int(spec.chord_length),
        "pythagorean_relation": "R^2 = r^2 + (c/2)^2",
        "answer_value": int(spec.answer),
    }


def concentric_trace_base(
    *,
    rendered: RenderedConcentricChordScene,
    annotation_artifacts: PixelAnnotationArtifacts,
    spec: ConcentricChordDiagramSpec,
    case_index: int,
    answer_probabilities: Mapping[str, float],
    render_meta: Mapping[str, Any],
    noise_meta: Mapping[str, Any],
    image_size: tuple[int, int],
) -> dict[str, Any]:
    """Return common trace facts that do not know public task/query identity."""

    measurement_fields = concentric_measurement_fields(spec)
    annotation_roles = [str(role) for role in rendered.annotation_roles]
    return {
        "measurement_fields": measurement_fields,
        "query_params": {
            "scene_id": SCENE_ID,
            "case_index": int(case_index),
            "target_support_probabilities": dict(answer_probabilities),
            **measurement_fields,
        },
        "scene_ir_base": {
            "scene_kind": SCENE_KIND,
            "scene_id": SCENE_ID,
            "entities": [dict(entity) for entity in rendered.scene_entities],
        },
        "relation_base": {
            "scene_variant": SCENE_VARIANT,
            "answer_value": int(spec.answer),
            "annotation_roles": annotation_roles,
        },
        "render_spec": {
            "canvas_size": [int(image_size[0]), int(image_size[1])],
            "coord_space": "pixel",
            "post_image_noise": dict(noise_meta),
            **dict(render_meta),
        },
        "render_map": {"coord_space": "pixel", **dict(rendered.render_map)},
        "execution_base": {
            "scene_id": SCENE_ID,
            "scene_variant": SCENE_VARIANT,
            "answer_type": "integer",
            "annotation_roles": annotation_roles,
            "reasoning_steps": 1,
            **measurement_fields,
        },
        "witness_base": {
            "type": "concentric_circle_chord_formula",
            "scene_id": SCENE_ID,
            "source_witness_type": annotation_artifacts.annotation_type,
            "original_annotation_value": annotation_artifacts.value,
            **measurement_fields,
        },
        "projected_annotation": dict(annotation_artifacts.projected_annotation),
    }


__all__ = [
    "ConcentricChordArtifacts",
    "concentric_measurement_fields",
    "concentric_trace_base",
    "prepare_concentric_chord_artifacts",
]
