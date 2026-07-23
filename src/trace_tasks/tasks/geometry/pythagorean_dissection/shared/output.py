"""Trace-section helpers for Pythagorean dissection tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from trace_tasks.tasks.geometry.shared.annotation_values import PixelAnnotationArtifacts

from .defaults import SCENE_KIND, SCENE_VARIANT
from .state import PythagoreanDissectionPlan, RenderedPythagoreanDissectionScene


@dataclass(frozen=True)
class PythagoreanTraceSections:
    """Task-neutral trace fragments assembled with public task fields later."""

    scene_kind: str
    query_params_base: dict[str, Any]
    scene_relations: dict[str, Any]
    render_spec_base: dict[str, Any]
    execution_common: dict[str, Any]
    witness_common: dict[str, Any]


def build_pythagorean_trace_sections(
    *,
    plan: PythagoreanDissectionPlan,
    rendered: RenderedPythagoreanDissectionScene,
    annotation_artifacts: PixelAnnotationArtifacts,
    answer_support_probabilities: Mapping[str, float],
    render_meta: Mapping[str, Any],
    noise_meta: Mapping[str, Any],
    image_size: tuple[int, int],
) -> PythagoreanTraceSections:
    """Format scene/verifier fragments without public task identity."""

    annotation_roles = [str(role) for role in rendered.annotation_roles]
    query_params_base = {
        "scene_variant": SCENE_VARIANT,
        "target_support_probabilities": dict(answer_support_probabilities),
        "answer_support_size": int(len(plan.answer_support)),
        "case_index": int(plan.case_index),
        **dict(plan.witness),
    }
    scene_relations = {
        "scene_variant": SCENE_VARIANT,
        "answer_value": int(plan.answer),
        "annotation_roles": list(annotation_roles),
        "formula_family": "central_square_area_from_triangle_legs",
    }
    render_spec_base = {
        "canvas_size": [int(image_size[0]), int(image_size[1])],
        "coord_space": "pixel",
        "post_image_noise": dict(noise_meta),
        **dict(render_meta),
    }
    execution_common = {
        "scene_variant": SCENE_VARIANT,
        "answer_type": "integer",
        "answer_value": int(plan.answer),
        "annotation_roles": list(annotation_roles),
        "annotation_type": str(annotation_artifacts.annotation_type),
        "reasoning_steps": 1,
        **dict(rendered.witness),
    }
    witness_common = {
        "scene_variant": SCENE_VARIANT,
        "answer_value": int(plan.answer),
        "source_witness_type": str(annotation_artifacts.annotation_type),
        "original_annotation_value": dict(annotation_artifacts.value),
        **dict(plan.witness),
    }
    return PythagoreanTraceSections(
        scene_kind=SCENE_KIND,
        query_params_base=query_params_base,
        scene_relations=scene_relations,
        render_spec_base=render_spec_base,
        execution_common=execution_common,
        witness_common=witness_common,
    )


__all__ = ["PythagoreanTraceSections", "build_pythagorean_trace_sections"]
