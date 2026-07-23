"""Compute the folded cone height from a sector net."""

from __future__ import annotations

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
    cone_net_diagram_spec,
    height_from_sector,
    height_support_values,
)
from .shared.output import cone_net_trace_payload
from .shared.prompts import cone_net_prompt_artifacts
from .shared.rendering import render_cone_net_with_retries
from .shared.sampling import CONE_NET_CASES, resolve_cone_net_case

TASK_ID = "task_geometry__cone_net__height_from_sector_angle"
INTERNAL_QUERY_ID = "height_from_sector_angle"
SUPPORTED_QUERY_IDS = ("single",)
HEIGHT_TARGET_MEASURE = "height"
HEIGHT_ANNOTATION_ROLES = ("S", "P", "Q", "C", "A")
HEIGHT_PROMPT_LABEL = "h=?"
HEIGHT_PROMPT_ANCHOR = "height_segment"


def _public_query_state(*, instance_seed: int, params: Mapping[str, Any]) -> tuple[str, dict[str, float], dict[str, Any]]:
    """Resolve the public single-query branch for the height objective."""

    selected_query, query_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=SUPPORTED_QUERY_IDS,
        default_query_id="single",
        task_id=TASK_ID,
    )
    return str(selected_query), dict(query_probabilities), dict(task_params)


def _height_diagram_spec(*, instance_seed: int, params: Mapping[str, Any]) -> tuple[Any, int, float]:
    """Select a height answer first and encode the matching sector-net diagram."""

    case, case_index = resolve_cone_net_case(
        target_measure=HEIGHT_TARGET_MEASURE,
        instance_seed=int(instance_seed),
        params=params,
        namespace=f"{TASK_ID}.{INTERNAL_QUERY_ID}.case",
    )
    height_value = float(height_from_sector(case))
    if height_value <= 0:
        raise ValueError("cone-net height objective requires a positive folded height")
    diagram_spec = cone_net_diagram_spec(
        case,
        answer=height_value,
        target_measure=HEIGHT_TARGET_MEASURE,
        target_label=HEIGHT_PROMPT_LABEL,
        target_label_anchor=HEIGHT_PROMPT_ANCHOR,
        annotation_roles=HEIGHT_ANNOTATION_ROLES,
        formula_family=INTERNAL_QUERY_ID,
        reasoning_steps=2,
    )
    return diagram_spec, int(case_index), height_value


def _height_prompt_and_trace(
    *,
    selected_query: str,
    query_probabilities: Mapping[str, float],
    task_params: Mapping[str, Any],
    diagram_spec: Any,
    case_index: int,
    height_value: float,
    instance_seed: int,
    max_attempts: int,
) -> tuple[str, dict[str, str], Any, Any, dict[str, Any], dict[str, str], str]:
    """Render the diagram and bind height-specific prompt and trace fields."""

    render_defaults, prompt_defaults = load_cone_net_task_defaults(TASK_ID)
    rendered, render_meta = render_cone_net_with_retries(
        spec=diagram_spec,
        instance_seed=int(instance_seed),
        params=task_params,
        render_defaults=render_defaults,
        max_attempts=int(max_attempts),
        random_namespace=f"{TASK_ID}.{INTERNAL_QUERY_ID}.render",
    )
    image, noise_meta = apply_post_image_noise(
        rendered.image,
        instance_seed=int(instance_seed),
        params=task_params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    annotation_artifacts = cone_net_annotation(rendered)
    _prompt_defaults, prompt_artifacts = cone_net_prompt_artifacts(
        prompt_defaults=prompt_defaults,
        prompt_key=INTERNAL_QUERY_ID,
        annotation_keys=tuple(annotation_artifacts.value.keys()),
        answer_value=float(height_value),
        instance_seed=int(instance_seed),
    )
    measurement_fields = dict(rendered.measurements)
    annotation_roles = [str(role) for role in rendered.annotation_roles]
    support_probabilities = geometry_selected_probability_map(
        height_support_values(CONE_NET_CASES),
        float(height_value),
        is_selected=lambda value, selected: float(value) == float(selected),
    )
    query_params = {
        "scene_id": SCENE_ID,
        "query_id": str(selected_query),
        "internal_query_id": INTERNAL_QUERY_ID,
        "query_id_probabilities": dict(query_probabilities),
        "case_index": int(case_index),
        "target_support_probabilities": dict(support_probabilities),
        **measurement_fields,
    }
    query_spec = build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts,
        query_id=str(selected_query),
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
            "query_id": str(selected_query),
            "internal_query_id": INTERNAL_QUERY_ID,
            "answer_value": float(height_value),
            "annotation_roles": list(annotation_roles),
        },
        execution_trace={
            "query_id": str(selected_query),
            "internal_query_id": INTERNAL_QUERY_ID,
            "query_id_probabilities": dict(query_probabilities),
            "answer_type": "number",
            "answer_value": float(height_value),
            "answer_rounding": "one_decimal",
            "annotation_roles": list(annotation_roles),
            "reasoning_steps": int(measurement_fields.get("reasoning_steps", 1)),
            **measurement_fields,
        },
        witness_symbolic={
            "query_id": str(selected_query),
            "internal_query_id": INTERNAL_QUERY_ID,
            "answer_value": float(height_value),
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


@register_task
class GeometryConeNetHeightFromSectorAngleTask:
    """Return the cone height after deriving radius from the sector arc."""

    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    domain = DOMAIN
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Bind the two-step height computation and its rendered annotation."""

        selected_query, query_probabilities, task_params = _public_query_state(
            instance_seed=int(instance_seed),
            params=params,
        )
        diagram_spec, case_index, height_value = _height_diagram_spec(
            instance_seed=int(instance_seed),
            params=task_params,
        )
        prompt, prompt_variants, image, annotation_artifacts, trace_payload, task_versions, scene_id = _height_prompt_and_trace(
            selected_query=str(selected_query),
            query_probabilities=query_probabilities,
            task_params=task_params,
            diagram_spec=diagram_spec,
            case_index=int(case_index),
            height_value=float(height_value),
            instance_seed=instance_seed,
            max_attempts=max_attempts,
        )
        return TaskOutput(
            prompt,
            TypedValue(type="number", value=height_value),
            TypedValue(type=annotation_artifacts.annotation_type, value=annotation_artifacts.value),
            image,
            "img0",
            trace_payload,
            task_versions,
            scene_id,
            str(selected_query),
            prompt_variants,
        )
