"""Compute a coordinate difference between two polar graph points."""

from __future__ import annotations

import random
from typing import Any

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import Task, TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec

from .shared.output import (
    point_map_annotation_from_render,
    point_map_witness,
    projected_point_map_annotation,
)
from .shared.prompts import polar_graph_prompt_artifacts
from .shared.rendering import render_polar_graph_paper_difference_scene
from .shared.sampling import select_polar_difference_case
from .shared.state import ReadoutComponent

TASK_ID = "task_geometry__polar_graph_paper__coordinate_difference_value"
SCENE_ID = "polar_graph_paper"
QUERY_IDS = ("radius_difference_value", "angle_difference_value")


def _component_for_query(query_id: str) -> ReadoutComponent:
    return "radius" if query_id == "radius_difference_value" else "angle_degrees"


@register_task
class PolarGraphPaperCoordinateDifferenceValueTask(Task):
    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    domain = "geometry"
    default_dataset_enabled = True
    supported_query_ids = QUERY_IDS

    def generate(
        self,
        instance_seed: int,
        *,
        params: dict[str, Any],
        max_attempts: int = 1,
    ) -> TaskOutput:
        """Generate one two-point polar coordinate difference sample."""

        del max_attempts
        params = dict(params or {})
        query_id, query_probs, task_params = select_task_query_id(
            instance_seed=instance_seed,
            params=params,
            supported_query_ids=QUERY_IDS,
            default_query_id=QUERY_IDS[0],
            task_id=TASK_ID,
        )
        component = _component_for_query(str(query_id))
        gen_defaults, render_defaults, prompt_defaults = load_scene_generation_rendering_prompt_defaults(
            "geometry",
            SCENE_ID,
            task_id=TASK_ID,
        )
        case = select_polar_difference_case(
            rng=random.Random(f"geometry:{SCENE_ID}:{TASK_ID}:{query_id}:{instance_seed}"),
            component=component,
            params=task_params,
            generation_defaults=gen_defaults,
        )
        rendered = render_polar_graph_paper_difference_scene(
            instance_seed=instance_seed,
            params=task_params,
            scene_id=SCENE_ID,
            case=case,
            rendering_defaults=render_defaults,
        )
        annotation_value = point_map_annotation_from_render(rendered.render_map, ("P", "Q"))
        prompt_artifacts = polar_graph_prompt_artifacts(
            scene_id=SCENE_ID,
            prompt_defaults_all=prompt_defaults,
            prompt_query_key=str(query_id),
            annotation_value=annotation_value,
            instance_seed=instance_seed,
            annotation_hint_key="annotation_hint_points_pq",
            answer_hint_key="answer_hint_integer",
            answer_type="integer",
        )
        point_trace = {
            "point_p": {
                "radius": case.point_p.radius,
                "theta_degrees": case.point_p.theta_degrees,
            },
            "point_q": {
                "radius": case.point_q.radius,
                "theta_degrees": case.point_q.theta_degrees,
            },
        }
        query_spec = build_prompt_query_spec(
            prompt_artifacts=prompt_artifacts,
            query_id=str(query_id),
            params={
                "difference_component": component,
                **point_trace,
                "correct_value": case.correct_value,
            },
        )

        return TaskOutput(
            prompt=prompt_artifacts.prompt,
            answer_gt=TypedValue(type="integer", value=int(case.correct_value)),
            annotation_gt=TypedValue(type="point_map", value=dict(annotation_value)),
            image=rendered.image,
            image_id="img0",
            trace_payload={
                "scene_ir": {
                    "scene_id": SCENE_ID,
                    "task_id": TASK_ID,
                    "object_description": "polar graph paper diagram with labeled point markers",
                },
                "query_spec": {
                    **query_spec,
                    "query_id_probabilities": query_probs,
                    "supported_query_ids": list(QUERY_IDS),
                },
                "render_spec": {**rendered.render_spec, "style": rendered.style_metadata},
                "render_map": rendered.render_map,
                "execution_trace": {
                    "component": component,
                    **point_trace,
                    "correct_value": case.correct_value,
                },
                "witness_symbolic": point_map_witness(("P", "Q")),
                "projected_annotation": projected_point_map_annotation(annotation_value),
            },
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=query_id,
            prompt_variants=dict(prompt_artifacts.prompt_variants),
        )
