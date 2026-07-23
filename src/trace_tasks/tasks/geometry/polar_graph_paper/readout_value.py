"""Read a plotted polar-coordinate value from polar graph paper."""

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

from .shared.output import point_annotation_from_render, point_witness, projected_point_annotation
from .shared.prompts import polar_graph_prompt_artifacts
from .shared.rendering import render_polar_graph_paper_scene
from .shared.sampling import select_polar_readout_case
from .shared.state import ReadoutComponent

TASK_ID = "task_geometry__polar_graph_paper__readout_value"
SCENE_ID = "polar_graph_paper"
QUERY_IDS = ("radius_readout_value", "angle_readout_value")

QUERY_COMPONENTS: dict[str, ReadoutComponent] = {
    "radius_readout_value": "radius",
    "angle_readout_value": "angle_degrees",
}
PROMPT_QUERY_KEYS = {
    "radius_readout_value": "radius_readout_value",
    "angle_readout_value": "angle_readout_value",
}


def _rng(instance_seed: int, query_id: str) -> random.Random:
    return random.Random(f"geometry:{SCENE_ID}:{query_id}:{instance_seed}")


def _trace_payload(
    *,
    query_id: str,
    query_probabilities: dict[str, float],
    component: ReadoutComponent,
    case: Any,
    rendered: Any,
    annotation_value: list[float],
    prompt_artifacts: Any,
) -> dict[str, Any]:
    """Assemble replay metadata after answer and annotation have been bound."""

    return {
        "scene_ir": {
            "scene_id": SCENE_ID,
            "task_id": TASK_ID,
            "object_description": "point P plotted on polar graph paper",
        },
        "query_spec": {
            **build_prompt_query_spec(
                prompt_artifacts=prompt_artifacts,
                query_id=query_id,
                params={
                    "readout_component": component,
                    "radius": case.radius,
                    "theta_degrees": case.theta_degrees,
                    "correct_value": case.correct_value,
                },
            ),
            "query_id_probabilities": query_probabilities,
            "supported_query_ids": list(QUERY_IDS),
        },
        "render_spec": {
            **rendered.render_spec,
            "style": rendered.style_metadata,
        },
        "render_map": rendered.render_map,
        "execution_trace": {
            "component": component,
            "radius": case.radius,
            "theta_degrees": case.theta_degrees,
            "correct_value": case.correct_value,
        },
        "witness_symbolic": point_witness("P"),
        "projected_annotation": projected_point_annotation(annotation_value),
    }


@register_task
class PolarGraphPaperReadoutValueTask(Task):
    task_id = TASK_ID
    reasoning_operations = ('direct_retrieval',)
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
        """Generate one direct polar readout sample."""

        del max_attempts
        params = dict(params or {})
        query_id, query_probabilities, task_params = select_task_query_id(
            instance_seed=instance_seed,
            params=params,
            supported_query_ids=QUERY_IDS,
            default_query_id=QUERY_IDS[0],
            task_id=TASK_ID,
        )
        component = QUERY_COMPONENTS[query_id]
        generation_defaults, rendering_defaults, prompt_defaults_all = load_scene_generation_rendering_prompt_defaults(
            "geometry",
            SCENE_ID,
            task_id=TASK_ID,
        )
        rng = _rng(instance_seed, query_id)
        case = select_polar_readout_case(
            rng=rng,
            component=component,
            params=task_params,
            generation_defaults=generation_defaults,
        )
        rendered = render_polar_graph_paper_scene(
            instance_seed=instance_seed,
            params=task_params,
            scene_id=SCENE_ID,
            case=case,
            rendering_defaults=rendering_defaults,
        )
        annotation_value = point_annotation_from_render(rendered.render_map)
        prompt_artifacts = polar_graph_prompt_artifacts(
            scene_id=SCENE_ID,
            prompt_defaults_all=prompt_defaults_all,
            prompt_query_key=PROMPT_QUERY_KEYS[query_id],
            annotation_value=annotation_value,
            instance_seed=instance_seed,
        )

        return TaskOutput(
            prompt=prompt_artifacts.prompt,
            answer_gt=TypedValue(type="integer", value=int(case.correct_value)),
            annotation_gt=TypedValue(type="point", value=annotation_value),
            image=rendered.image,
            image_id="img0",
            trace_payload=_trace_payload(
                query_id=query_id,
                query_probabilities=query_probabilities,
                component=component,
                case=case,
                rendered=rendered,
                annotation_value=annotation_value,
                prompt_artifacts=prompt_artifacts,
            ),
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=query_id,
            prompt_variants=dict(prompt_artifacts.prompt_variants),
        )
