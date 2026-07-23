"""Compute the folded cone base radius from a sector net."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.fixed_query import geometry_selected_probability_map, select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec

from .shared.annotations import cone_net_annotation
from .shared.defaults import DOMAIN, POST_IMAGE_NOISE_DEFAULTS, SCENE_ID, load_cone_net_task_defaults
from .shared.measurements import (
    base_radius_from_sector,
    base_radius_support_values,
    cone_net_diagram_spec,
)
from .shared.output import cone_net_trace_payload
from .shared.prompts import cone_net_prompt_artifacts
from .shared.rendering import render_cone_net_with_retries
from .shared.sampling import CONE_NET_CASES, resolve_cone_net_case

TASK_ID = "task_geometry__cone_net__base_radius_from_sector_angle"
INTERNAL_QUERY_ID = "base_radius_from_sector_angle"
SUPPORTED_QUERY_IDS = ("single",)
BASE_RADIUS_ANNOTATION_ROLES = ("S", "P", "Q", "C", "R")
BASE_RADIUS_TARGET_LABEL = "r=?"
BASE_RADIUS_LABEL_ANCHOR = "radius_segment"


@dataclass(frozen=True)
class _BaseRadiusRequest:
    """Task-owned radius answer binding before shared rendering."""

    selected_query: str
    query_probabilities: Mapping[str, float]
    params: Mapping[str, Any]
    case_index: int
    answer_value: float
    answer_support: tuple[float, ...]
    diagram_spec: Any


def _base_radius_prompt_and_trace(
    *,
    request: _BaseRadiusRequest,
    instance_seed: int,
    max_attempts: int,
) -> tuple[str, dict[str, str], Any, Any, dict[str, Any], dict[str, str], str]:
    """Render the diagram and bind prompt, annotation, and trace fields."""

    render_defaults, prompt_defaults = load_cone_net_task_defaults(TASK_ID)
    rendered, render_meta = render_cone_net_with_retries(
        spec=request.diagram_spec,
        instance_seed=int(instance_seed),
        params=request.params,
        render_defaults=render_defaults,
        max_attempts=int(max_attempts),
        random_namespace=f"{TASK_ID}.{INTERNAL_QUERY_ID}.render",
    )
    image, noise_meta = apply_post_image_noise(
        rendered.image,
        instance_seed=int(instance_seed),
        params=request.params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    annotation_artifacts = cone_net_annotation(rendered)
    _prompt_defaults, prompt_artifacts = cone_net_prompt_artifacts(
        prompt_defaults=prompt_defaults,
        prompt_key=INTERNAL_QUERY_ID,
        annotation_keys=tuple(annotation_artifacts.value.keys()),
        answer_value=float(request.answer_value),
        instance_seed=int(instance_seed),
    )
    measurement_fields = dict(rendered.measurements)
    annotation_roles = [str(role) for role in rendered.annotation_roles]
    support_probabilities = geometry_selected_probability_map(
        request.answer_support,
        float(request.answer_value),
        is_selected=lambda value, selected: float(value) == float(selected),
    )
    query_params = {
        "scene_id": SCENE_ID,
        "query_id": request.selected_query,
        "internal_query_id": INTERNAL_QUERY_ID,
        "query_id_probabilities": dict(request.query_probabilities),
        "case_index": int(request.case_index),
        "target_support_probabilities": dict(support_probabilities),
        **measurement_fields,
    }
    query_spec = build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts,
        query_id=request.selected_query,
        params=query_params,
    )
    query_spec["scene_id"] = SCENE_ID
    trace_payload = cone_net_trace_payload(
        rendered=rendered,
        annotation_artifacts=annotation_artifacts,
        query_spec=query_spec,
        render_meta=render_meta,
        noise_meta=noise_meta,
        image_size=(int(image.size[0]), int(image.size[1])),
        relations={
            "query_id": request.selected_query,
            "internal_query_id": INTERNAL_QUERY_ID,
            "answer_value": float(request.answer_value),
            "annotation_roles": list(annotation_roles),
        },
        execution_trace={
            "query_id": request.selected_query,
            "internal_query_id": INTERNAL_QUERY_ID,
            "query_id_probabilities": dict(request.query_probabilities),
            "answer_type": "number",
            "answer_value": float(request.answer_value),
            "answer_rounding": "one_decimal",
            "annotation_roles": list(annotation_roles),
            "reasoning_steps": int(measurement_fields.get("reasoning_steps", 1)),
            **measurement_fields,
        },
        witness_symbolic={
            "query_id": request.selected_query,
            "internal_query_id": INTERNAL_QUERY_ID,
            "answer_value": float(request.answer_value),
            **measurement_fields,
        },
    )
    return (
        str(prompt_artifacts.prompt),
        dict(prompt_artifacts.prompt_variants),
        image,
        annotation_artifacts,
        trace_payload,
        default_task_versions(),
        SCENE_ID,
    )


def _radius_public_branch(*, instance_seed: int, params: Mapping[str, Any]) -> tuple[str, Mapping[str, float], Mapping[str, Any]]:
    """Resolve the public no-branch query while preserving review overrides."""

    selected_query, query_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=SUPPORTED_QUERY_IDS,
        default_query_id="single",
        task_id=TASK_ID,
    )
    return str(selected_query), dict(query_probabilities), dict(task_params)


def _base_radius_request(*, instance_seed: int, params: Mapping[str, Any]) -> _BaseRadiusRequest:
    """Select a radius answer first, then bind one compatible cone net."""

    selected_query, query_probabilities, task_params = _radius_public_branch(
        instance_seed=int(instance_seed),
        params=params,
    )
    case, case_index = resolve_cone_net_case(
        target_measure="base_radius",
        instance_seed=int(instance_seed),
        params=task_params,
        namespace=f"{TASK_ID}.{INTERNAL_QUERY_ID}.case",
    )
    radius_value = float(base_radius_from_sector(case))
    diagram_spec = cone_net_diagram_spec(
        case,
        answer=radius_value,
        target_measure="base_radius",
        target_label=BASE_RADIUS_TARGET_LABEL,
        target_label_anchor=BASE_RADIUS_LABEL_ANCHOR,
        annotation_roles=BASE_RADIUS_ANNOTATION_ROLES,
        formula_family=INTERNAL_QUERY_ID,
        reasoning_steps=1,
    )
    return _BaseRadiusRequest(
        selected_query=selected_query,
        query_probabilities=query_probabilities,
        params=task_params,
        case_index=int(case_index),
        answer_value=radius_value,
        answer_support=base_radius_support_values(CONE_NET_CASES),
        diagram_spec=diagram_spec,
    )


@register_task
class GeometryConeNetBaseRadiusFromSectorAngleTask:
    """Return the cone base radius implied by the arc length of the sector net."""

    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    domain = DOMAIN
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Build the radius-specific answer and annotation from one rendered trace."""

        request = _base_radius_request(
            instance_seed=int(instance_seed),
            params=params,
        )
        prompt, prompt_variants, image, annotation_artifacts, trace_payload, task_versions, scene_id = _base_radius_prompt_and_trace(
            request=request,
            instance_seed=int(instance_seed),
            max_attempts=int(max_attempts),
        )
        radius_answer = TypedValue(type="number", value=request.answer_value)
        radius_annotation = TypedValue(
            type=annotation_artifacts.annotation_type,
            value=annotation_artifacts.value,
        )
        return TaskOutput(
            prompt=prompt,
            answer_gt=radius_answer,
            annotation_gt=radius_annotation,
            image=image,
            image_id="img0",
            trace_payload=trace_payload,
            task_versions=task_versions,
            scene_id=scene_id,
            query_id=request.selected_query,
            prompt_variants=prompt_variants,
        )
