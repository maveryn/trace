"""Compute a missing side length in a tangential quadrilateral."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.core.types import TypedValue
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import split_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec
from trace_tasks.tasks.geometry.shared.noise_defaults import POST_IMAGE_NOISE_DEFAULTS

from .shared.annotations import keyed_point_annotation, projected_keyed_point_payload
from .shared.construction import (
    select_missing_side,
    select_tangent_case,
    side_lengths_from_vertex_tangents,
    vertex_tangents_from_case,
)
from .shared.prompts import tangential_prompt_artifacts
from .shared.rendering import create_circle_polygon_render_context, render_tangential_scene, render_with_layout_retry
from .shared.state import SCENE_ID, RenderedTangentialScene, TangentialDiagramSpec


TASK_ID = "task_geometry__circle_polygon_composite__tangential_quadrilateral_side_length_value"
QUERY_ID = "missing_side_from_tangent_quadrilateral"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (QUERY_ID,)

_SCENE_DEFAULTS = get_scene_defaults("geometry", SCENE_ID)
_GEN_DEFAULTS_UNUSED, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    _SCENE_DEFAULTS,
    task_id=TASK_ID,
)


@dataclass(frozen=True)
class _TangentialProblem:
    """Task-owned answer, prompt, and trace facts for one missing-side instance."""

    diagram_spec: TangentialDiagramSpec
    missing_side: str
    visible_sides: Tuple[str, ...]
    answer: int
    missing_side_probabilities: dict[str, float]
    tangent_case_probabilities: dict[str, float]
    answer_probabilities: dict[str, float]


def _pitot_missing_side_answer(side_lengths: Mapping[str, int], missing_side: str) -> int:
    """Return one missing side from Pitot's theorem."""

    side = str(missing_side)
    if side == "AB":
        return int(side_lengths["BC"] + side_lengths["DA"] - side_lengths["CD"])
    if side == "BC":
        return int(side_lengths["AB"] + side_lengths["CD"] - side_lengths["DA"])
    if side == "CD":
        return int(side_lengths["BC"] + side_lengths["DA"] - side_lengths["AB"])
    if side == "DA":
        return int(side_lengths["AB"] + side_lengths["CD"] - side_lengths["BC"])
    raise ValueError(f"unsupported missing side: {side}")


def _bind_side_length_problem(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    selected_query: str,
) -> _TangentialProblem:
    """Bind the hidden side and answer before rendering."""

    missing_side, missing_side_probabilities = select_missing_side(
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}.{selected_query}.missing_side",
    )
    tangent_case, tangent_case_probabilities, answer_probabilities = select_tangent_case(
        missing_side=str(missing_side),
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}.{selected_query}.tangent_case",
    )
    vertex_tangents = vertex_tangents_from_case(tangent_case)
    side_lengths = side_lengths_from_vertex_tangents(tangent_case)
    visible_sides = tuple(side for side in ("AB", "BC", "CD", "DA") if side != missing_side)
    answer = _pitot_missing_side_answer(side_lengths, missing_side)
    return _TangentialProblem(
        diagram_spec=TangentialDiagramSpec(
            vertex_tangents=dict(vertex_tangents),
            side_lengths=dict(side_lengths),
            unknown_sides=(str(missing_side),),
        ),
        missing_side=str(missing_side),
        visible_sides=tuple(str(side) for side in visible_sides),
        answer=int(answer),
        missing_side_probabilities=dict(missing_side_probabilities),
        tangent_case_probabilities=dict(tangent_case_probabilities),
        answer_probabilities=dict(answer_probabilities),
    )


def _draw_tangential_diagram_with_retry(
    *,
    instance_seed: int,
    task_params: Mapping[str, Any],
    selected_query: str,
    problem: _TangentialProblem,
    max_attempts: int,
) -> tuple[Any, RenderedTangentialScene]:
    """Retry only the stochastic layout while preserving the task-bound answer."""

    def build_context(attempt_seed: int, attempt_params: Mapping[str, Any]) -> Any:
        return create_circle_polygon_render_context(
            instance_seed=int(attempt_seed),
            params=attempt_params,
            rendering_defaults=_RENDER_DEFAULTS,
            fill_namespace=f"{TASK_ID}.{selected_query}.fill_palette",
        )

    def draw_scene(render_context: Any, attempt_seed: int) -> RenderedTangentialScene:
        return render_tangential_scene(
            render_context,
            problem.diagram_spec,
            instance_seed=int(attempt_seed),
            render_namespace=f"{TASK_ID}.{selected_query}.render.scene",
        )

    return render_with_layout_retry(
        instance_seed=int(instance_seed),
        task_params=task_params,
        max_attempts=int(max_attempts),
        build_context=build_context,
        draw_scene=draw_scene,
    )


def _build_side_length_trace_payload(
    *,
    rendered: RenderedTangentialScene,
    image_size: tuple[int, int],
    render_context: Any,
    noise_meta: Mapping[str, Any],
    selected_query: str,
    query_probabilities: Mapping[str, float],
    prompt_artifacts: Any,
    problem: _TangentialProblem,
    annotation_value: Mapping[str, list[float]],
) -> dict[str, Any]:
    """Serialize task-owned missing-side facts into trace metadata."""

    side_lengths = problem.diagram_spec.side_lengths
    opposite_sum_ab_cd = int(side_lengths["AB"] + side_lengths["CD"])
    opposite_sum_bc_da = int(side_lengths["BC"] + side_lengths["DA"])
    query_spec = build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts,
        query_id=str(selected_query),
        params={
            "query_id": str(selected_query),
            "query_id_probabilities": dict(query_probabilities),
            "missing_side": str(problem.missing_side),
            "missing_side_probabilities": dict(problem.missing_side_probabilities),
            "tangent_case_probabilities": dict(problem.tangent_case_probabilities),
            "target_answer_support_probabilities": dict(problem.answer_probabilities),
        },
    )
    query_spec["scene_id"] = SCENE_ID
    query_spec["task_id"] = TASK_ID

    return {
        "scene_ir": {
            "domain": "geometry",
            "scene_id": SCENE_ID,
            "task_id": TASK_ID,
            "query_id": str(selected_query),
            "entities": {
                "vertices": dict(rendered.render_map["vertices"]),
                "tangency_points": dict(rendered.render_map["tangency_points"]),
                "incircle_center": list(rendered.render_map["incircle_center"]),
            },
            "relations": {
                "type": "tangential_quadrilateral_missing_side",
                "missing_side": str(problem.missing_side),
                "visible_sides": list(problem.visible_sides),
            },
        },
        "query_spec": query_spec,
        "render_spec": {
            "task_id": TASK_ID,
            "scene_id": SCENE_ID,
            "query_id": str(selected_query),
            "canvas": {
                "width": int(image_size[0]),
                "height": int(image_size[1]),
            },
            "single_object_scene_rotation": render_context.scene_transform.metadata(),
            "style": {
                "technical_diagram": dict(render_context.diagram_style_meta),
                "background": dict(render_context.background_meta),
                "post_image_noise": dict(noise_meta),
            },
            "prompt": {
                "prompt_variant": dict(prompt_artifacts.prompt_variant),
                "prompt_variant_active_key": str(prompt_artifacts.prompt_variant_active_key),
                "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
            },
        },
        "render_map": dict(rendered.render_map),
        "execution_trace": {
            "task_id": TASK_ID,
            "scene_id": SCENE_ID,
            "query_id": str(selected_query),
            "missing_side": str(problem.missing_side),
            "visible_sides": list(problem.visible_sides),
            "vertex_tangents": dict(problem.diagram_spec.vertex_tangents),
            "side_lengths": dict(side_lengths),
            "target_answer_support_probabilities": dict(problem.answer_probabilities),
            "opposite_sum_AB_CD": int(opposite_sum_ab_cd),
            "opposite_sum_BC_DA": int(opposite_sum_bc_da),
            "answer": int(problem.answer),
            "annotation_roles": list(rendered.annotation_roles),
        },
        "witness_symbolic": {
            "task_id": TASK_ID,
            "scene_id": SCENE_ID,
            "query_id": str(selected_query),
            "formula_family": "pitot_theorem_tangential_quadrilateral",
            "missing_side": str(problem.missing_side),
            "visible_sides": list(problem.visible_sides),
            "vertex_tangents": dict(problem.diagram_spec.vertex_tangents),
            "side_lengths": dict(side_lengths),
            "target_answer_support_probabilities": dict(problem.answer_probabilities),
            "opposite_sum_AB_CD": int(opposite_sum_ab_cd),
            "opposite_sum_BC_DA": int(opposite_sum_bc_da),
            "answer_value": int(problem.answer),
        },
        "projected_annotation": projected_keyed_point_payload(annotation_value),
    }


@register_task
class GeometryCirclePolygonCompositeTangentialQuadrilateralSideLengthValueTask:
    """Compute a missing side length in a tangential quadrilateral."""

    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    domain = "geometry"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Select a side-length query and bind answer/annotation in this public task."""

        selected_query, query_probabilities, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=params,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=QUERY_ID,
            task_id=TASK_ID,
        )
        problem = _bind_side_length_problem(
            instance_seed=int(instance_seed),
            params=task_params,
            selected_query=str(selected_query),
        )
        render_context, rendered = _draw_tangential_diagram_with_retry(
            instance_seed=int(instance_seed),
            task_params=task_params,
            selected_query=str(selected_query),
            problem=problem,
            max_attempts=int(max_attempts),
        )
        image, noise_meta = apply_post_image_noise(
            rendered.image,
            instance_seed=int(instance_seed),
            params=task_params,
            default_config=POST_IMAGE_NOISE_DEFAULTS,
        )
        _prompt_defaults, prompt_artifacts = tangential_prompt_artifacts(
            prompt_defaults=_PROMPT_DEFAULTS,
            prompt_query_key=str(selected_query),
            target_side=str(problem.missing_side),
            visible_sides=", ".join(problem.visible_sides),
            answer_value=int(problem.answer),
            annotation_keys=rendered.annotation_roles,
            instance_seed=int(instance_seed),
        )
        annotation_value = keyed_point_annotation(rendered)
        trace_payload = _build_side_length_trace_payload(
            rendered=rendered,
            image_size=image.size,
            render_context=render_context,
            noise_meta=dict(noise_meta),
            selected_query=str(selected_query),
            query_probabilities=query_probabilities,
            prompt_artifacts=prompt_artifacts,
            problem=problem,
            annotation_value=annotation_value,
        )
        return TaskOutput(
            prompt=str(prompt_artifacts.prompt),
            answer_gt=TypedValue(type="integer", value=int(problem.answer)),
            annotation_gt=TypedValue(type="point_map", value=dict(annotation_value)),
            image=image,
            image_id="img0",
            trace_payload=trace_payload,
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=str(selected_query),
            prompt_variants=dict(prompt_artifacts.prompt_variants),
        )


__all__ = [
    "GeometryCirclePolygonCompositeTangentialQuadrilateralSideLengthValueTask",
    "QUERY_ID",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
]
