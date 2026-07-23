"""Scene-private lifecycle for incircle-tangent public tasks."""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Sequence

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.shared.config_defaults import split_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.prompt_variants import PromptTraceArtifacts, build_prompt_query_spec

from .shared.defaults import SCENE_DEFAULTS, SCENE_ID, SCENE_KIND, SCENE_VARIANT
from .shared.output import IncircleTaskParts, prepare_incircle_task_parts
from .shared.state import IncircleDiagramSpec, RenderedIncircleScene

SpecBuilder = Callable[[int, Mapping[str, Any]], tuple[IncircleDiagramSpec, int, dict[str, float]]]


def incircle_trace_payload(
    *,
    selected_query: str,
    internal_query_id: str,
    query_probabilities: Mapping[str, float],
    spec: IncircleDiagramSpec,
    rendered: RenderedIncircleScene,
    annotation_artifacts: Any,
    prompt_artifacts: PromptTraceArtifacts,
    case_index: int,
    target_support_probabilities: Mapping[str, float],
    render_meta: Mapping[str, Any],
    noise_meta: Mapping[str, Any],
    image_size: tuple[int, int],
    answer_value: int | float,
    reasoning_steps: int,
) -> dict[str, Any]:
    """Build public-query trace metadata after objective-specific binding."""

    measurement_fields = spec.measurement_fields()
    query_params = {
        "scene_id": SCENE_ID,
        "query_id": str(selected_query),
        "internal_query_id": str(internal_query_id),
        "query_id_probabilities": dict(query_probabilities),
        "case_index": int(case_index),
        "target_support_probabilities": dict(target_support_probabilities),
        **dict(measurement_fields),
    }
    query_spec = build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts,
        query_id=str(selected_query),
        params=query_params,
    )
    query_spec["scene_id"] = SCENE_ID
    return {
        "scene_ir": {
            "scene_kind": SCENE_KIND,
            "scene_id": SCENE_ID,
            "entities": [dict(entity) for entity in rendered.scene_entities],
            "relations": {
                "query_id": str(selected_query),
                "internal_query_id": str(internal_query_id),
                "scene_variant": SCENE_VARIANT,
                "answer_value": answer_value,
                "annotation_roles": list(rendered.annotation_roles),
            },
        },
        "query_spec": query_spec,
        "render_spec": {
            "canvas_size": [int(image_size[0]), int(image_size[1])],
            "coord_space": "pixel",
            "post_image_noise": dict(noise_meta),
            **dict(render_meta),
        },
        "render_map": {"coord_space": "pixel", **dict(rendered.render_map)},
        "execution_trace": {
            "scene_id": SCENE_ID,
            "scene_variant": SCENE_VARIANT,
            "query_id": str(selected_query),
            "internal_query_id": str(internal_query_id),
            "query_id_probabilities": dict(query_probabilities),
            "answer_type": str(spec.answer_type),
            "answer_value": answer_value,
            "answer_rounding": str(spec.answer_rounding),
            "annotation_roles": list(rendered.annotation_roles),
            "reasoning_steps": int(reasoning_steps),
            **dict(measurement_fields),
        },
        "witness_symbolic": {
            "type": "tangent_polygon_incircle_formula",
            "scene_id": SCENE_ID,
            "query_id": str(selected_query),
            "internal_query_id": str(internal_query_id),
            "answer_value": answer_value,
            "source_witness_type": annotation_artifacts.annotation_type,
            "original_annotation_value": annotation_artifacts.value,
            **dict(measurement_fields),
        },
        "projected_annotation": dict(annotation_artifacts.projected_annotation),
    }


def run_incircle_lifecycle(
    *,
    task_id: str,
    internal_query_id: str,
    supported_query_ids: Sequence[str],
    answer_type: str,
    reasoning_steps: int,
    build_spec: SpecBuilder,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
) -> TaskOutput:
    """Run neutral query/render/prompt plumbing after task-owned spec binding."""

    selected_query, query_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=tuple(str(value) for value in supported_query_ids),
        default_query_id="single",
        task_id=str(task_id),
        namespace=f"{task_id}.query",
    )
    _generation_defaults, render_defaults, prompt_defaults = split_scene_generation_rendering_prompt_defaults(
        SCENE_DEFAULTS,
        task_id=str(task_id),
    )
    spec, case_index, answer_probabilities = build_spec(
        instance_seed=int(instance_seed),
        params=task_params,
    )
    answer_value = int(spec.answer) if str(answer_type) == "integer" else float(spec.answer)
    parts = prepare_incircle_task_parts(
        random_namespace=str(task_id),
        prompt_key=str(internal_query_id),
        spec=spec,
        render_defaults=render_defaults,
        prompt_defaults=prompt_defaults,
        instance_seed=int(instance_seed),
        params=task_params,
        max_attempts=int(max_attempts),
    )
    trace_payload = incircle_trace_payload(
        selected_query=str(selected_query),
        internal_query_id=str(internal_query_id),
        query_probabilities=query_probabilities,
        spec=spec,
        rendered=parts.rendered,
        annotation_artifacts=parts.annotation_artifacts,
        prompt_artifacts=parts.prompt_artifacts,
        case_index=int(case_index),
        target_support_probabilities=answer_probabilities,
        render_meta=parts.render_meta,
        noise_meta=parts.noise_meta,
        image_size=(int(parts.image.size[0]), int(parts.image.size[1])),
        answer_value=answer_value,
        reasoning_steps=int(reasoning_steps),
    )
    return TaskOutput(
        prompt=str(parts.prompt),
        answer_gt=TypedValue(type=str(answer_type), value=answer_value),
        annotation_gt=TypedValue(
            type=parts.annotation_artifacts.annotation_type,
            value=parts.annotation_artifacts.value,
        ),
        image=parts.image,
        image_id="img0",
        trace_payload=trace_payload,
        task_versions=parts.task_versions,
        scene_id=SCENE_ID,
        query_id=str(selected_query),
        prompt_variants=dict(parts.prompt_variants),
    )


__all__ = ["run_incircle_lifecycle"]
