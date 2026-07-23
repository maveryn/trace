"""Identity-free prompt and trace serialization for special quadrilaterals."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)

from .._lifecycle import SpecialQuadrilateralRenderParts
from .defaults import DOMAIN, PROMPT_BUNDLE_ID, SCENE_ID, SCENE_KIND, SCENE_PROMPT_KEY
from .prompts import special_quadrilateral_prompt_slots
from .state import SpecialQuadrilateralProblem


def prompt_artifacts_for_bound_case(
    *,
    prompt_defaults: Mapping[str, Any],
    task_prompt_key: str,
    branch_prompt_key: str,
    target_name: str,
    annotation_roles: tuple[str, ...],
    answer_value: int,
    instance_seed: int,
):
    """Render prompt variants after a public task binds semantic prompt keys."""

    prompt_selection = render_scene_prompt_variants(
        domain=DOMAIN,
        scene_id=SCENE_ID,
        bundle_id=str(prompt_defaults.get("bundle_id", PROMPT_BUNDLE_ID)),
        scene_key=str(prompt_defaults.get("scene_key", SCENE_PROMPT_KEY)),
        task_key=str(task_prompt_key),
        query_key=str(branch_prompt_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots=special_quadrilateral_prompt_slots(
            target_name=str(target_name),
            annotation_roles=tuple(str(role) for role in annotation_roles),
            answer_value=int(answer_value),
        ),
        instance_seed=int(instance_seed),
    )
    return build_prompt_trace_artifacts(prompt_selection)


def common_trace_sections(
    *,
    branch_probabilities: Mapping[str, float],
    answer_probabilities: Mapping[str, float],
    prompt_artifacts: Any,
    problem: SpecialQuadrilateralProblem,
    parts: SpecialQuadrilateralRenderParts,
    extra_case_values: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Serialize common trace sections from a resolved construction case."""

    case = problem.case
    rendered = parts.rendered
    extra_values = dict(extra_case_values or {})
    annotation_roles = [str(role) for role in rendered.annotation_points.keys()]
    trace_values = {
        "scene_id": SCENE_ID,
        "query_id_probabilities": dict(branch_probabilities),
        "answer_support_probabilities": dict(answer_probabilities),
        "shape_kind": str(case.shape_kind),
        "target_name": str(case.target_name),
        "target_label": str(case.target_label),
        "support_label": str(case.support_label),
        "theorem": str(case.theorem),
        "answer_type": "integer",
        "answer": int(case.answer),
        "answer_value": int(case.answer),
        "annotation_roles": list(annotation_roles),
        **dict(extra_values),
    }
    image_width, image_height = parts.image.size
    vertices = {
        key: [round(float(point[0]), 3), round(float(point[1]), 3)]
        for key, point in rendered.vertices.items()
    }
    return {
        "scene_ir": {
            "domain": DOMAIN,
            "scene_kind": SCENE_KIND,
            "scene_id": SCENE_ID,
            "entities": [
                {
                    "type": str(case.shape_kind),
                    "labels": ["A", "B", "C", "D"],
                    "vertices": dict(vertices),
                }
            ],
            "relations": {
                "theorem": str(case.theorem),
                "answer_value": int(case.answer),
            },
        },
        "render_spec": {
            "scene_id": SCENE_ID,
            "canvas": {"width": int(image_width), "height": int(image_height)},
            "single_object_scene_rotation": rendered.render_map.get("single_object_scene_rotation", {}),
            "style": {"post_image_noise": dict(parts.noise_meta)},
            "prompt": {
                "prompt_variant": dict(prompt_artifacts.prompt_variant),
                "prompt_variant_active_key": str(prompt_artifacts.prompt_variant_active_key),
                "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
            },
        },
        "render_map": dict(rendered.render_map),
        "execution_trace": dict(trace_values),
        "projected_annotation": dict(parts.annotation_artifacts.projected_annotation),
    }


__all__ = ["common_trace_sections", "prompt_artifacts_for_bound_case"]
