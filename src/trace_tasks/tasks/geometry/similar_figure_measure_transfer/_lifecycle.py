"""Private assembly helpers for similar-figure public tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.core.types import TypedValue
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.shared.config_defaults import split_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec
from trace_tasks.tasks.geometry.shared.metadata_serialization import geometry_json_ready
from trace_tasks.tasks.geometry.shared.noise_defaults import POST_IMAGE_NOISE_DEFAULTS

from .shared.annotations import point_map_for_labels
from .shared.prompts import similar_prompt_artifacts
from .shared.rendering import build_render_context, render_equation_scene, render_measure_scene
from .shared.state import SCENE_ID, RenderedSimilarScene, SimilarEquationCase, SimilarMeasureCase


@dataclass(frozen=True)
class SimilarFigureObjectivePlan:
    """Task-owned case and prompt metadata for one public objective."""

    case: SimilarMeasureCase | SimilarEquationCase
    config_group_key: str
    prompt_branch_key: str
    answer_type: str
    answer_hint_key: str
    program_scope: str
    public_branch: str
    branch_probabilities: Mapping[str, float]


def run_similar_figure_public_entry(task: Any, instance_seed: int, *, params: Mapping[str, Any], max_attempts: int) -> TaskOutput:
    """Select the public branch, call the task hook, and assemble output."""

    selected_branch, branch_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=tuple(str(value) for value in getattr(task, "supported_query_ids")),
        default_query_id=str(getattr(task, "default_query_id")),
        task_id=str(getattr(task, "task_id")),
        namespace=f"{getattr(task, 'task_id')}.query",
    )
    plan = task.prepare_objective(
        int(instance_seed),
        str(selected_branch),
        dict(branch_probabilities),
        dict(task_params),
    )
    return build_similar_figure_task_output(
        task_id=str(getattr(task, "task_id")),
        instance_seed=int(instance_seed),
        params=task_params,
        max_attempts=int(max_attempts),
        plan=plan,
    )


def forced_or_sampled_family(
    *,
    context_label: str,
    options: tuple[str, ...],
    instance_seed: int,
    params: Mapping[str, Any],
    namespace: str,
) -> str:
    """Resolve replay metadata for one construction-family axis."""

    forced = params.get("construction_family")
    if forced is not None:
        family = str(forced)
        if family not in options:
            raise ValueError(f"unsupported construction_family for {context_label}: {family}")
        return family
    from trace_tasks.core.seed import spawn_rng

    rng = spawn_rng(int(instance_seed), str(namespace))
    return str(rng.choice(options))


def build_similar_figure_task_output(
    *,
    task_id: str,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    plan: SimilarFigureObjectivePlan,
) -> TaskOutput:
    """Retry rendering and serialize one public similar-figure task output."""

    generation_defaults, rendering_defaults, prompt_defaults = split_scene_generation_rendering_prompt_defaults(
        get_scene_defaults("geometry", SCENE_ID),
        task_id=str(plan.config_group_key),
    )
    _ = generation_defaults
    last_error: Exception | None = None
    rendered: RenderedSimilarScene | None = None
    runtime_params: dict[str, Any] = {}
    attempt_seed = int(instance_seed)
    for attempt in range(max(1, int(max_attempts))):
        attempt_seed = int(instance_seed) + int(attempt)
        try:
            runtime_params = dict(params)
            context = build_render_context(
                instance_seed=attempt_seed,
                params=runtime_params,
                rendering_defaults=rendering_defaults,
            )
            if isinstance(plan.case, SimilarMeasureCase):
                rendered = render_measure_scene(plan.case, context=context, instance_seed=attempt_seed)
            else:
                rendered = render_equation_scene(plan.case, context=context, instance_seed=attempt_seed)
            break
        except Exception as exc:
            last_error = exc
            continue
    if rendered is None:
        raise RuntimeError(f"failed to generate {task_id}") from last_error

    annotation_value = point_map_for_labels(rendered.figure_geometry, tuple(plan.case.annotation_labels))
    prompt_defaults, prompt_artifacts = similar_prompt_artifacts(
        prompt_defaults=prompt_defaults,
        prompt_branch_key=str(plan.prompt_branch_key),
        target_name=str(plan.case.target_name),
        annotation_keys=tuple(annotation_value.keys()),
        answer_value=_answer_value(plan.case.answer, answer_type=str(plan.answer_type)),
        answer_hint_key=str(plan.answer_hint_key),
        instance_seed=int(instance_seed),
    )
    image, noise_meta = apply_post_image_noise(
        rendered.image,
        instance_seed=int(instance_seed),
        params=runtime_params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    answer_value = _answer_value(plan.case.answer, answer_type=str(plan.answer_type))
    trace_payload = _trace_payload(
        task_id=str(task_id),
        plan=plan,
        rendered=rendered,
        prompt_artifacts=prompt_artifacts,
        annotation_value=annotation_value,
        noise_meta=noise_meta,
        attempt_seed=int(attempt_seed),
    )
    return TaskOutput(
        prompt=str(prompt_artifacts.prompt),
        answer_gt=TypedValue(type=str(plan.answer_type), value=answer_value),
        annotation_gt=TypedValue(type="point_map", value=dict(annotation_value)),
        image=image,
        image_id="img0",
        trace_payload=trace_payload,
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
        query_id=str(plan.public_branch),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
    )


def _trace_payload(
    *,
    task_id: str,
    plan: SimilarFigureObjectivePlan,
    rendered: RenderedSimilarScene,
    prompt_artifacts: Any,
    annotation_value: Mapping[str, Any],
    noise_meta: Mapping[str, Any],
    attempt_seed: int,
) -> dict[str, Any]:
    """Serialize task trace fields without selecting public objective behavior."""

    case = plan.case
    figure_payload = {
        "source_vertices": {
            label: list(point)
            for label, point in zip(rendered.figure_geometry.source_labels, rendered.figure_geometry.source_vertices)
        },
        "target_vertices": {
            label: list(point)
            for label, point in zip(rendered.figure_geometry.target_labels, rendered.figure_geometry.target_vertices)
        },
    }
    measure_values = _case_measure_values(case)
    branch_params = {
        "query_id": str(plan.public_branch),
        "query_id_probabilities": dict(plan.branch_probabilities),
        "construction_family": str(case.construction_family),
        "shape_kind": str(case.shape_kind),
        "attempt_seed": int(attempt_seed),
    }
    return {
        "scene_ir": {
            "domain": "geometry",
            "scene_id": SCENE_ID,
            "task_id": str(task_id),
            "entities": [
                {
                    "type": "similar_figure_pair",
                    "shape_kind": str(case.shape_kind),
                    **dict(figure_payload),
                },
            ],
            "relations": {
                "type": str(case.relation),
                "scale_factor": float(case.scale_factor),
                "construction_family": str(case.construction_family),
            },
        },
        "query_spec": build_prompt_query_spec(
            prompt_artifacts=prompt_artifacts,
            query_id=str(plan.public_branch),
            params=branch_params,
        ),
        "render_spec": {
            "scene_id": SCENE_ID,
            "task_id": str(task_id),
            "query_id": str(plan.public_branch),
            "canvas": {
                "width": int(rendered.image.size[0]),
                "height": int(rendered.image.size[1]),
            },
            "style": {
                "technical_diagram": dict(rendered.style_metadata),
                "background": dict(rendered.background_metadata),
                "post_image_noise": dict(noise_meta),
                "scene_transform": dict(rendered.figure_geometry.transform_metadata),
            },
            "prompt": {
                "prompt_variant": dict(prompt_artifacts.prompt_variant),
                "prompt_variant_active_key": str(prompt_artifacts.prompt_variant_active_key),
                "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
            },
        },
        "render_map": geometry_json_ready(rendered.render_map),
        "execution_trace": {
            "task_id": str(task_id),
            "scene_id": SCENE_ID,
            "query_id": str(plan.public_branch),
            "program_scope": str(plan.program_scope),
            "construction_family": str(case.construction_family),
            "shape_kind": str(case.shape_kind),
            "target_name": str(case.target_name),
            "relation": str(case.relation),
            "scale_factor": float(case.scale_factor),
            "answer": _answer_value(case.answer, answer_type=str(plan.answer_type)),
            **({"answer_rounding": "one_decimal"} if str(plan.answer_type) == "number" else {}),
            "annotation_roles": list(annotation_value.keys()),
            **measure_values,
        },
        "witness_symbolic": {
            "task_id": str(task_id),
            "scene_id": SCENE_ID,
            "query_id": str(plan.public_branch),
            "target_name": str(case.target_name),
            "relation": str(case.relation),
            "answer": _answer_value(case.answer, answer_type=str(plan.answer_type)),
            **({"answer_rounding": "one_decimal"} if str(plan.answer_type) == "number" else {}),
            **measure_values,
        },
        "projected_annotation": {
            "type": "point_map",
            "point_map": dict(annotation_value),
            "pixel_point_map": dict(annotation_value),
        },
    }


def _case_measure_values(case: SimilarMeasureCase | SimilarEquationCase) -> dict[str, Any]:
    if isinstance(case, SimilarMeasureCase):
        return {
            "source_target_side_value": case.source_target_side_value,
            "target_target_side_value": case.target_target_side_value,
            "support_source_side_value": case.support_source_side_value,
            "support_target_side_value": case.support_target_side_value,
            "source_perimeter": case.source_perimeter,
            "target_perimeter": case.target_perimeter,
            "source_area": case.source_area,
            "target_area": case.target_area,
            "area_ratio_label": case.area_ratio_label,
        }
    return {
        "source_target_side_value": _number(case.source_target_value),
        "target_target_side_value": _number(case.target_target_value),
        "support_source_side_value": _number(float(case.support_source_label)),
        "support_target_side_value": _number(float(case.support_target_label)),
        "variable_name": str(case.variable_name),
        "source_target_label": str(case.source_target_label),
        "target_target_label": str(case.target_target_label),
    }


def _answer_value(value: float | int, *, answer_type: str) -> int | float:
    if str(answer_type) == "integer":
        return int(round(float(value)))
    return _number(float(value))


def _number(value: float) -> int | float:
    return float(round(float(value), 1))
