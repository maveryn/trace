"""Trace-section helpers for Pythagorean tree tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .defaults import SCENE_KIND, SCENE_VARIANT
from .state import PythagoreanTreePlan, RenderedPythagoreanTreeScene


@dataclass(frozen=True)
class PythagoreanTreeTraceSections:
    """Task-neutral trace fragments assembled with public task fields later."""

    scene_kind: str
    query_params_base: dict[str, Any]
    scene_relations: dict[str, Any]
    render_spec_base: dict[str, Any]
    execution_common: dict[str, Any]
    witness_common: dict[str, Any]


def build_pythagorean_tree_trace_sections(
    *,
    plan: PythagoreanTreePlan,
    rendered: RenderedPythagoreanTreeScene,
    annotation_artifacts: Any,
    annotation_roles: Sequence[str],
    triple_probabilities: Mapping[str, float],
    target_role_probabilities: Mapping[str, float],
    render_meta: Mapping[str, Any],
    noise_meta: Mapping[str, Any],
    image_size: tuple[int, int],
) -> PythagoreanTreeTraceSections:
    """Format scene/verifier fragments without public task identity."""

    selected_annotation_roles = [str(role) for role in annotation_roles]
    triple = plan.triple
    query_params_base = {
        "scene_variant": SCENE_VARIANT,
        "target_role": str(plan.target_role),
        "target_role_probabilities": dict(target_role_probabilities),
        "triple_probabilities": dict(triple_probabilities),
        **dict(plan.witness),
    }
    scene_relations = {
        "scene_variant": SCENE_VARIANT,
        "target_role": str(plan.target_role),
        "answer_value": int(plan.answer),
        "annotation_roles": list(selected_annotation_roles),
        "formula_family": "pythagorean_attached_square_area",
    }
    render_spec_base = {
        "canvas_size": [int(image_size[0]), int(image_size[1])],
        "coord_space": "pixel",
        "post_image_noise": dict(noise_meta),
        **dict(render_meta),
    }
    execution_common = {
        "scene_variant": SCENE_VARIANT,
        "target_role": str(plan.target_role),
        "answer_type": "integer",
        "answer_value": int(plan.answer),
        "annotation_roles": list(selected_annotation_roles),
        "annotation_type": str(annotation_artifacts.annotation_type),
        "leg_a": int(triple.leg_a),
        "leg_b": int(triple.leg_b),
        "hypotenuse": int(triple.hypotenuse),
        **dict(plan.witness),
        **dict(rendered.witness),
    }
    witness_common = {
        "scene_variant": SCENE_VARIANT,
        "target_role": str(plan.target_role),
        "answer_value": int(plan.answer),
        "source_witness_type": str(annotation_artifacts.annotation_type),
        "original_annotation_value": annotation_artifacts.value,
        **dict(plan.witness),
    }
    return PythagoreanTreeTraceSections(
        scene_kind=SCENE_KIND,
        query_params_base=query_params_base,
        scene_relations=scene_relations,
        render_spec_base=render_spec_base,
        execution_common=execution_common,
        witness_common=witness_common,
    )


__all__ = ["PythagoreanTreeTraceSections", "build_pythagorean_tree_trace_sections"]
