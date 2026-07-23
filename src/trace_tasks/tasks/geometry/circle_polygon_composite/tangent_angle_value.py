"""Compute a tangent angle in a circle-polygon construction."""

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
from .shared.construction import select_angle_degrees, select_construction_kind, select_side_sign
from .shared.prompts import tangent_angle_prompt_artifacts
from .shared.rendering import create_circle_polygon_render_context, render_angle_scene, render_with_layout_retry
from .shared.state import ANGLE_ANNOTATION_KEYS, SCENE_ID, AngleDiagramSpec, RenderedAngleScene


TASK_ID = "task_geometry__circle_polygon_composite__tangent_angle_value"
QUERY_ID_TANGENT_ANGLE = "tangent_angle_from_radius_perpendicular"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (QUERY_ID_TANGENT_ANGLE,)
_PROMPT_DESCRIPTION_BY_CONSTRUCTION = {
    "incircle": (
        "It includes a square with an incircle, a tangent line touching the circle at T, "
        "radius OT, one known angle, and one target angle marked ?."
    ),
    "semicircle": (
        "It includes a rectangle with a semicircle, a tangent line touching the semicircle at T, "
        "radius OT, one known angle, and one target angle marked ?."
    ),
}
_ROUND_SHAPE_BY_CONSTRUCTION = {
    "incircle": "incircle",
    "semicircle": "semicircle",
}
_SCENE_DEFAULTS = get_scene_defaults("geometry", SCENE_ID)
_GEN_DEFAULTS_UNUSED, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    _SCENE_DEFAULTS,
    task_id=TASK_ID,
)


@dataclass(frozen=True)
class _AngleTransferProblem:
    """Task-owned answer, prompt, and trace facts for one tangent-angle instance."""

    diagram_spec: AngleDiagramSpec
    answer: int
    angle_probabilities: dict[str, float]
    construction_probabilities: dict[str, float]
    side_probabilities: dict[str, float]


@register_task
class GeometryCirclePolygonCompositeTangentAngleValueTask:
    """Compute an angle from a tangent-radius construction."""

    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    domain = "geometry"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    @staticmethod
    def _prepare_tangent_angle_problem(
        *,
        instance_seed: int,
        params: Mapping[str, Any],
        selected_query: str,
    ) -> _AngleTransferProblem:
        """Resolve construction, answer, and trace facts for one tangent-angle instance."""

        construction_kind, construction_probabilities = select_construction_kind(
            instance_seed=int(instance_seed),
            params=params,
            namespace=f"{TASK_ID}.{selected_query}.construction_kind",
        )
        angle_degrees, angle_probabilities = select_angle_degrees(
            instance_seed=int(instance_seed),
            params=params,
            namespace=f"{TASK_ID}.{selected_query}.target_angle",
        )
        side_sign, side_probabilities = select_side_sign(
            instance_seed=int(instance_seed),
            params=params,
            namespace=f"{TASK_ID}.{selected_query}.side_sign",
        )
        return _AngleTransferProblem(
            diagram_spec=AngleDiagramSpec(
                construction_kind=str(construction_kind),
                angle_degrees=int(angle_degrees),
                side_sign=int(side_sign),
            ),
            answer=int(angle_degrees),
            angle_probabilities=dict(angle_probabilities),
            construction_probabilities=dict(construction_probabilities),
            side_probabilities=dict(side_probabilities),
        )

    @staticmethod
    def _draw_tangent_angle_diagram_with_retry(
        *,
        instance_seed: int,
        task_params: Mapping[str, Any],
        selected_query: str,
        problem: _AngleTransferProblem,
        max_attempts: int,
    ) -> tuple[Any, RenderedAngleScene]:
        """Retry only stochastic layout while preserving the selected angle answer."""

        return render_with_layout_retry(
            instance_seed=int(instance_seed),
            task_params=task_params,
            max_attempts=int(max_attempts),
            build_context=lambda attempt_seed, attempt_params: create_circle_polygon_render_context(
                instance_seed=int(attempt_seed),
                params=attempt_params,
                rendering_defaults=_RENDER_DEFAULTS,
                fill_namespace=f"{TASK_ID}.{selected_query}.fill_palette",
            ),
            draw_scene=lambda render_context, attempt_seed: render_angle_scene(
                render_context,
                problem.diagram_spec,
                instance_seed=int(attempt_seed),
                render_namespace=f"{TASK_ID}.{selected_query}.render.scene",
            ),
        )

    @staticmethod
    def _serialize_tangent_angle_trace_payload(
        *,
        rendered: RenderedAngleScene,
        image_size: tuple[int, int],
        render_context: Any,
        noise_meta: Mapping[str, Any],
        selected_query: str,
        query_probabilities: Mapping[str, float],
        prompt_artifacts: Any,
        problem: _AngleTransferProblem,
        annotation_value: Mapping[str, list[float]],
    ) -> dict[str, Any]:
        """Serialize task-owned tangent-angle facts into trace metadata."""

        spec = problem.diagram_spec
        query = str(selected_query)
        construction = str(spec.construction_kind)
        answer_degrees = int(problem.answer)
        query_spec = build_prompt_query_spec(
            prompt_artifacts=prompt_artifacts,
            query_id=query,
            params={
                "query_id": query,
                "query_id_probabilities": dict(query_probabilities),
                "answer_support_probabilities": dict(problem.angle_probabilities),
                "construction_kind_probabilities": dict(problem.construction_probabilities),
                "side_probabilities": dict(problem.side_probabilities),
                "construction_kind": construction,
            },
        )
        query_spec["scene_id"] = SCENE_ID
        query_spec["task_id"] = TASK_ID

        scene_ir = {
            "domain": "geometry",
            "scene_id": SCENE_ID,
            "task_id": TASK_ID,
            "query_id": query,
            "entities": {
                "shape_corners": dict(rendered.render_map["shape_corners"]),
                "circle_center": list(rendered.render_map["circle_center"]),
                "tangent_point": list(rendered.render_map["tangent_point"]),
            },
            "relations": {
                "type": "circle_polygon_tangent_angle_transfer",
                "construction_kind": construction,
                "known_angle_degrees": answer_degrees,
                "target_angle_degrees": answer_degrees,
            },
        }
        render_spec = {
            "task_id": TASK_ID,
            "scene_id": SCENE_ID,
            "query_id": query,
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
        }
        execution_trace = {
            "task_id": TASK_ID,
            "scene_id": SCENE_ID,
            "query_id": query,
            "answer": answer_degrees,
            "known_angle_degrees": answer_degrees,
            "target_angle_degrees": answer_degrees,
            "side_sign": int(spec.side_sign),
            "construction_kind": construction,
            "annotation_roles": list(rendered.annotation_roles),
        }
        payload: dict[str, Any] = {
            "scene_ir": scene_ir,
            "query_spec": query_spec,
            "render_spec": render_spec,
            "render_map": dict(rendered.render_map),
            "execution_trace": execution_trace,
        }
        payload["witness_symbolic"] = dict(
            task_id=TASK_ID,
            scene_id=SCENE_ID,
            query_id=query,
            formula_family="tangent_radius_perpendicular_angle_transfer",
            construction_kind=construction,
            known_angle_degrees=answer_degrees,
            target_angle_degrees=answer_degrees,
            answer_value=answer_degrees,
        )
        payload["projected_annotation"] = projected_keyed_point_payload(annotation_value)
        return payload

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Select a tangent-angle query and bind answer/annotation in this public task."""

        selected_query, query_probabilities, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=params,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=QUERY_ID_TANGENT_ANGLE,
            task_id=TASK_ID,
        )
        problem = self._prepare_tangent_angle_problem(
            instance_seed=int(instance_seed),
            params=task_params,
            selected_query=str(selected_query),
        )
        render_context, rendered = self._draw_tangent_angle_diagram_with_retry(
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
        _prompt_defaults, prompt_artifacts = tangent_angle_prompt_artifacts(
            prompt_defaults=_PROMPT_DEFAULTS,
            prompt_query_key=str(selected_query),
            angle_object_description=_PROMPT_DESCRIPTION_BY_CONSTRUCTION[
                str(problem.diagram_spec.construction_kind)
            ],
            round_shape=_ROUND_SHAPE_BY_CONSTRUCTION[str(problem.diagram_spec.construction_kind)],
            answer_value=int(problem.answer),
            annotation_keys=ANGLE_ANNOTATION_KEYS,
            instance_seed=int(instance_seed),
        )
        annotation_value = keyed_point_annotation(rendered)
        trace_payload = self._serialize_tangent_angle_trace_payload(
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
        output_fields = dict(
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
        return TaskOutput(**output_fields)


__all__ = [
    "GeometryCirclePolygonCompositeTangentAngleValueTask",
    "QUERY_ID_TANGENT_ANGLE",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
]
