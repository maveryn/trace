"""Scene-private lifecycle helpers for polar graph paper tasks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import random
from typing import Any

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_json_example import dump_prompt_json_examples
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec

from .shared.output import (
    point_set_annotation_from_render,
    point_set_witness,
    projected_point_set_annotation,
)
from .shared.prompts import polar_graph_prompt_artifacts
from .shared.rendering import render_polar_graph_paper_coordinate_count_scene
from .shared.sampling import select_polar_coordinate_count_case
from .shared.state import ReadoutComponent

SCENE_ID = "polar_graph_paper"


@dataclass(frozen=True)
class PolarCoordinateValuePointCountPlan:
    """Task-owned coordinate-value count semantics for the polar graph scene."""

    public_identifier: str
    query_ids: tuple[str, ...]
    component_by_query_id: Mapping[str, ReadoutComponent]


def run_polar_coordinate_value_point_count(
    *,
    plan: PolarCoordinateValuePointCountPlan,
    instance_seed: int,
    params: dict[str, Any],
    max_attempts: int,
) -> TaskOutput:
    """Generate a coordinate-value point-count instance from a public task plan."""

    del max_attempts
    params = dict(params or {})
    query_id, query_probs, task_params = select_task_query_id(
        instance_seed=instance_seed,
        params=params,
        supported_query_ids=plan.query_ids,
        default_query_id=plan.query_ids[0],
        task_id=plan.public_identifier,
    )
    component = plan.component_by_query_id[str(query_id)]
    gen_defaults, render_defaults, prompt_defaults = load_scene_generation_rendering_prompt_defaults(
        "geometry",
        SCENE_ID,
        task_id=plan.public_identifier,
    )
    case = select_polar_coordinate_count_case(
        rng=random.Random(f"geometry:{SCENE_ID}:{plan.public_identifier}:{query_id}:{instance_seed}"),
        component=component,
        params=task_params,
        generation_defaults=gen_defaults,
    )
    rendered = render_polar_graph_paper_coordinate_count_scene(
        instance_seed=instance_seed,
        params=task_params,
        scene_id=SCENE_ID,
        case=case,
        rendering_defaults=render_defaults,
    )
    annotation_value = point_set_annotation_from_render(
        rendered.render_map,
        tuple(case.matching_labels),
    )
    prompt_artifacts = polar_graph_prompt_artifacts(
        scene_id=SCENE_ID,
        prompt_defaults_all=prompt_defaults,
        prompt_query_key=str(query_id),
        annotation_value=annotation_value,
        instance_seed=instance_seed,
        annotation_hint_key="annotation_hint_matching_point_set",
        answer_hint_key="answer_hint_integer",
        answer_type="integer",
        extra_dynamic_slots={"target_value": str(case.target_value)},
        json_examples=dump_prompt_json_examples(
            annotation=[[120, 140], [180, 140], [150, 190]],
            answer=3,
        ),
    )
    query_spec = build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts,
        query_id=str(query_id),
        params={
            "coordinate_component": component,
            "target_value": case.target_value,
            "correct_value": case.correct_value,
            "total_point_count": len(case.points),
            "matching_labels": list(case.matching_labels),
        },
    )
    execution_trace = {
        "component": component,
        "target_value": int(case.target_value),
        "total_point_count": len(case.points),
        "matching_labels": list(case.matching_labels),
        "points": {
            str(point.label): {
                "radius": int(point.radius),
                "theta_degrees": int(point.theta_degrees),
            }
            for point in case.points
        },
        "correct_value": int(case.correct_value),
    }

    return TaskOutput(
        prompt=prompt_artifacts.prompt,
        answer_gt=TypedValue(type="integer", value=int(case.correct_value)),
        annotation_gt=TypedValue(type="point_set", value=list(annotation_value)),
        image=rendered.image,
        image_id="img0",
        trace_payload={
            "scene_ir": {
                "scene_id": SCENE_ID,
                "task_id": plan.public_identifier,
                "object_description": "polar graph paper diagram with multiple marked point markers",
            },
            "query_spec": {
                **query_spec,
                "query_id_probabilities": query_probs,
                "supported_query_ids": list(plan.query_ids),
            },
            "render_spec": {**rendered.render_spec, "style": rendered.style_metadata},
            "render_map": rendered.render_map,
            "execution_trace": execution_trace,
            "witness_symbolic": point_set_witness(tuple(case.matching_labels)),
            "projected_annotation": projected_point_set_annotation(annotation_value),
        },
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
        query_id=str(query_id),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
    )
