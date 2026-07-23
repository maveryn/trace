"""Tangent length from radius and exterior distance circle-theorem task."""

from __future__ import annotations

from typing import Any, Dict

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import (
    split_scene_generation_rendering_prompt_defaults,
)
from trace_tasks.tasks.geometry.circle_theorem._lifecycle import (
    run_label_keyed_number_circle_theorem_task,
)

from .shared.construction import (
    SCENE_ID,
    build_tangent_radius_payload,
    resolve_tangent_length_from_radius_and_external_distance,
)

TASK_ID = "task_geometry__circle_theorem__tangent_length_from_radius_and_external_distance_value"
INTERNAL_QUERY_ID = "tangent_length_from_radius_and_external_distance"
SUPPORTED_QUERY_IDS = (INTERNAL_QUERY_ID,)


def _defaults() -> tuple[dict[str, Any], dict[str, Any]]:
    scene_defaults = get_scene_defaults("geometry", SCENE_ID)
    generation_defaults, render_defaults, _prompt_defaults = (
        split_scene_generation_rendering_prompt_defaults(
            scene_defaults,
            task_id=TASK_ID,
        )
    )
    return dict(generation_defaults), dict(render_defaults)


@register_task
class GeometryCircleTangentLengthFromRadiusAndExternalDistanceValueTask:
    """Compute the tangent length using the radius and exterior distance."""

    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    domain = "geometry"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(
        self,
        instance_seed: int,
        *,
        params: Dict[str, Any],
        max_attempts: int,
    ) -> TaskOutput:
        """Bind the tangent-length formula objective, then render the shared tangent-radius diagram."""
        generation_defaults, render_defaults = _defaults()
        query = resolve_tangent_length_from_radius_and_external_distance(
            int(instance_seed),
            rng_namespace=TASK_ID,
            params=params,
            generation_defaults=generation_defaults,
        )
        query_params = {
            "query_id_probabilities": {"single": 1.0},
            "internal_query_id": INTERNAL_QUERY_ID,
            "internal_query_id_probabilities": {INTERNAL_QUERY_ID: 1.0},
            "radius_value": float(query.radius_value),
            "tangent_length": float(query.tangent_length),
            "external_distance": float(query.external_distance),
            "angle_degrees": None,
            "support_probabilities": dict(query.support_probabilities),
            "answer_rounding": "one_decimal",
        }
        return run_label_keyed_number_circle_theorem_task(
            task_id=TASK_ID,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            query_id=INTERNAL_QUERY_ID,
            answer_value=float(query.answer_value),
            query_params=query_params,
            build_scene_payload=lambda rng: build_tangent_radius_payload(rng, query=query),
            render_defaults=render_defaults,
            scene_kind="geometry_circle_theorem_tangent_radius_right_triangle",
            witness_type="circle_theorem_tangent_radius_right_triangle",
            object_description_key="object_description_tangent_radius",
            answer_hint_key="answer_hint_tangent_radius_number",
            annotation_hint_key="annotation_hint_tangent_radius_points",
        )


__all__ = [
    "GeometryCircleTangentLengthFromRadiusAndExternalDistanceValueTask",
    "INTERNAL_QUERY_ID",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
]
