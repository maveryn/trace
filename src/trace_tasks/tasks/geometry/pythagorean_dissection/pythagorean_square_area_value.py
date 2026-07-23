"""Compute square EFGH's area in a Pythagorean dissection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from trace_tasks.core.types import TypedValue
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import split_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.fixed_query import geometry_selected_probability_map, select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec

from .shared.defaults import DOMAIN, POST_IMAGE_NOISE_DEFAULTS, SCENE_DEFAULTS, SCENE_ID
from .shared.output import build_pythagorean_trace_sections
from .shared.prompts import pythagorean_dissection_prompt_artifacts
from .shared.rendering import make_render_context, render_pythagorean_dissection_scene
from .shared.sampling import (
    SQUARE_AREA_ANSWER_SUPPORT,
    build_square_area_plan,
    select_square_area_answer,
)
from .shared.state import PythagoreanDissectionPlan, RenderedPythagoreanDissectionScene
from ..shared.annotation_values import keyed_point_annotation_artifacts

TASK_ID = "task_geometry__pythagorean_dissection__pythagorean_square_area_value"
SUPPORTED_QUERY_IDS: tuple[str, ...] = ("single",)
PROMPT_TASK_KEY = "pythagorean_square_area_value"


@dataclass(frozen=True)
class _SquareAreaRequest:
    """Task-owned public query and answer binding before pixel rendering."""

    selected_query: str
    query_probabilities: dict[str, float]
    params: dict[str, Any]
    plan: PythagoreanDissectionPlan
    answer_support_probabilities: dict[str, float]


def _resolve_square_area_request(*, instance_seed: int, params: Mapping[str, Any]) -> _SquareAreaRequest:
    """Resolve the public no-branch query, then select an answer-balanced case."""

    selected_query, query_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=dict(params),
        supported_query_ids=SUPPORTED_QUERY_IDS,
        default_query_id="single",
        task_id=TASK_ID,
        namespace=f"{TASK_ID}.query",
    )
    answer = select_square_area_answer(
        params=task_params,
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}.answer",
    )
    plan = build_square_area_plan(
        answer=int(answer),
        params=task_params,
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}.case",
    )
    return _SquareAreaRequest(
        selected_query=str(selected_query),
        query_probabilities=dict(query_probabilities),
        params=dict(task_params),
        plan=plan,
        answer_support_probabilities=geometry_selected_probability_map(
            SQUARE_AREA_ANSWER_SUPPORT,
            int(plan.answer),
            key_fn=lambda value: str(int(value)),
            is_selected=lambda value, selected: int(value) == int(selected),
        ),
    )


def _render_square_area(
    *,
    request: _SquareAreaRequest,
    instance_seed: int,
    max_attempts: int,
    rendering_defaults: Mapping[str, Any],
) -> tuple[RenderedPythagoreanDissectionScene, dict[str, Any]]:
    """Render the task-bound dissection, retrying only layout/style failures."""

    rendered: RenderedPythagoreanDissectionScene | None = None
    render_meta: dict[str, Any] | None = None
    last_error: Exception | None = None
    for attempt in range(max(1, int(max_attempts))):
        attempt_seed = int(instance_seed) + int(attempt)
        try:
            ctx, attempt_meta = make_render_context(
                instance_seed=attempt_seed,
                params=request.params,
                rendering_defaults=rendering_defaults,
            )
            rendered = render_pythagorean_dissection_scene(ctx, request.plan)
            render_meta = dict(attempt_meta)
            render_meta["single_object_scene_rotation"] = ctx.scene_transform.metadata()
            break
        except Exception as exc:
            last_error = exc
            continue
    if rendered is None or render_meta is None:
        raise RuntimeError(f"failed to render {TASK_ID}") from last_error
    return rendered, render_meta


@register_task
class GeometryPythagoreanSquareAreaValueTask:
    """Compute square EFGH's area from the two visible segment labels."""

    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    domain = DOMAIN
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int) -> TaskOutput:
        """Own query selection, answer binding, annotation binding, and final output."""

        _generation_defaults, rendering_defaults, prompt_defaults = split_scene_generation_rendering_prompt_defaults(
            SCENE_DEFAULTS,
            task_id=TASK_ID,
        )
        request = _resolve_square_area_request(
            instance_seed=int(instance_seed),
            params=params,
        )
        rendered, render_meta = _render_square_area(
            request=request,
            instance_seed=int(instance_seed),
            max_attempts=int(max_attempts),
            rendering_defaults=rendering_defaults,
        )
        image, noise_meta = apply_post_image_noise(
            rendered.image,
            instance_seed=int(instance_seed),
            params=request.params,
            default_config=POST_IMAGE_NOISE_DEFAULTS,
        )
        annotation_artifacts = keyed_point_annotation_artifacts(
            rendered.annotation_keyed_points,
            roles=rendered.annotation_roles,
        )
        _prompt_defaults, prompt_artifacts = pythagorean_dissection_prompt_artifacts(
            prompt_defaults=prompt_defaults,
            prompt_task_key=PROMPT_TASK_KEY,
            annotation_keys=tuple(annotation_artifacts.value.keys()),
            answer=int(request.plan.answer),
            instance_seed=int(instance_seed),
        )
        sections = build_pythagorean_trace_sections(
            plan=request.plan,
            rendered=rendered,
            annotation_artifacts=annotation_artifacts,
            answer_support_probabilities=request.answer_support_probabilities,
            render_meta=render_meta,
            noise_meta=noise_meta,
            image_size=(int(image.size[0]), int(image.size[1])),
        )
        query_spec = build_prompt_query_spec(
            prompt_artifacts=prompt_artifacts,
            query_id=request.selected_query,
            params={
                "scene_id": SCENE_ID,
                "query_id": request.selected_query,
                "query_id_probabilities": dict(request.query_probabilities),
                **sections.query_params_base,
            },
        )
        query_spec["scene_id"] = SCENE_ID
        scene_ir = {
            "scene_kind": sections.scene_kind,
            "scene_id": SCENE_ID,
            "task_id": TASK_ID,
            "entities": [dict(entity) for entity in rendered.scene_entities],
            "relations": {
                "query_id": request.selected_query,
                **sections.scene_relations,
            },
        }
        trace_payload = {
            "scene_ir": scene_ir,
            "query_spec": query_spec,
            "render_spec": {
                "task_id": TASK_ID,
                "scene_id": SCENE_ID,
                "query_id": request.selected_query,
                **sections.render_spec_base,
            },
            "render_map": dict(rendered.render_map),
            "execution_trace": {
                "task_id": TASK_ID,
                "scene_id": SCENE_ID,
                "query_id": request.selected_query,
                "query_id_probabilities": dict(request.query_probabilities),
                "objective": "pythagorean_square_area_value",
                **sections.execution_common,
            },
            "witness_symbolic": {
                "type": "pythagorean_square_dissection_formula",
                "task_id": TASK_ID,
                "query_id": request.selected_query,
                **sections.witness_common,
            },
            "projected_annotation": dict(annotation_artifacts.projected_annotation),
        }
        return TaskOutput(
            prompt=str(prompt_artifacts.prompt),
            answer_gt=TypedValue(type="integer", value=int(request.plan.answer)),
            annotation_gt=TypedValue(
                type=annotation_artifacts.annotation_type,
                value=annotation_artifacts.value,
            ),
            image=image,
            image_id="img0",
            trace_payload=trace_payload,
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=request.selected_query,
            prompt_variants=dict(prompt_artifacts.prompt_variants),
        )


__all__ = [
    "GeometryPythagoreanSquareAreaValueTask",
    "SCENE_ID",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
]
