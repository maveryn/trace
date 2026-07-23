"""Scene-private lifecycle plumbing for rectangular-solid public tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from trace_tasks.core.types import TypedValue
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec

from .shared.defaults import DOMAIN, POST_IMAGE_NOISE_DEFAULTS, SCENE_ID, load_rectangular_solid_defaults
from .shared.prompts import rectangular_solid_prompt_artifacts
from .shared.rendering import create_render_context
from .shared.state import RenderContext, RenderedRectangularSolidScene

RenderBuilder = Callable[..., RenderedRectangularSolidScene]
AnnotationBuilder = Callable[[RenderedRectangularSolidScene], tuple[TypedValue, Mapping[str, Any]]]
ObjectivePreparer = Callable[[int, Mapping[str, Any], str, Mapping[str, float]], "RectangularSolidObjectivePlan"]


@dataclass(frozen=True)
class RectangularSolidObjectivePlan:
    """Task-owned objective binding prepared for one selected branch."""

    prompt_task_key: str
    prompt_branch_key: str
    problem: Any
    render_scene: RenderBuilder
    bind_annotation: AnnotationBuilder
    answer_gt: TypedValue
    query_params: Mapping[str, Any]
    trace_values: Mapping[str, Any]


def _common_trace_payload(
    *,
    selected_branch: str,
    branch_probabilities: Mapping[str, float],
    prompt_artifacts: Any,
    image_size: tuple[int, int],
    plan: RectangularSolidObjectivePlan,
    rendered: RenderedRectangularSolidScene,
    projected_annotation: Mapping[str, Any],
    noise_meta: Mapping[str, Any],
    style_meta: Mapping[str, Any],
) -> dict[str, Any]:
    """Build shared trace structure from task-owned objective values."""

    formula_family = str(getattr(plan.problem, "formula_family"))
    formula = str(getattr(plan.problem, "formula"))
    answer_value = int(plan.answer_gt.value)
    query_params = {
        "scene_id": SCENE_ID,
        "query_id": str(selected_branch),
        "query_id_probabilities": dict(branch_probabilities),
        **dict(plan.query_params),
    }
    trace_values = {
        "formula_family": formula_family,
        "formula": formula,
        "answer": int(answer_value),
        "answer_value": int(answer_value),
        "annotation_roles": list(rendered.annotation_roles),
        **dict(plan.trace_values),
    }
    return {
        "scene_ir": {
            "domain": DOMAIN,
            "scene_id": SCENE_ID,
            "entities": [dict(entity) for entity in rendered.scene_entities],
            "relations": {
                "type": formula_family,
                **dict(trace_values),
            },
        },
        "query_spec": build_prompt_query_spec(
            prompt_artifacts=prompt_artifacts,
            query_id=str(selected_branch),
            params=query_params,
        ),
        "render_spec": {
            "canvas": {"width": int(image_size[0]), "height": int(image_size[1])},
            "coord_space": "pixel",
            "style": dict(style_meta),
            "post_image_noise": dict(noise_meta),
        },
        "render_map": {"coord_space": "pixel", **dict(rendered.render_map)},
        "execution_trace": {
            "scene_id": SCENE_ID,
            "query_id": str(selected_branch),
            "query_id_probabilities": dict(branch_probabilities),
            **dict(trace_values),
        },
        "witness_symbolic": {
            "type": formula_family,
            "scene_id": SCENE_ID,
            "query_id": str(selected_branch),
            "formula": formula,
            **dict(plan.trace_values),
            "answer_value": int(answer_value),
        },
        "projected_annotation": dict(projected_annotation),
    }


def run_rectangular_solid_public_entry(
    task: Any,
    instance_seed: int,
    *,
    params: Mapping[str, Any],
    max_attempts: int,
) -> TaskOutput:
    """Run the common lifecycle after a public task prepares its objective plan."""

    selected_branch, branch_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=dict(params),
        supported_query_ids=tuple(str(value) for value in task.supported_query_ids),
        default_query_id=str(task.default_query_id),
        task_id=str(task.task_id),
    )
    plan = task.prepare_objective(
        int(instance_seed),
        task_params,
        str(selected_branch),
        branch_probabilities,
    )
    _generation_defaults, rendering_defaults, prompt_defaults = load_rectangular_solid_defaults(
        public_identifier=str(task.task_id),
    )
    last_error: Exception | None = None
    rendered = None
    ctx = None
    for attempt_index in range(max(1, int(max_attempts))):
        try:
            ctx = create_render_context(
                instance_seed=int(instance_seed) + int(attempt_index),
                params={**dict(task_params), "_render_attempt": int(attempt_index)},
                rendering_defaults=rendering_defaults,
            )
            rendered = plan.render_scene(
                ctx,
                plan.problem,
                instance_seed=int(instance_seed) + int(attempt_index),
            )
            break
        except Exception as exc:
            last_error = exc
            continue
    if rendered is None or ctx is None:
        raise RuntimeError(f"failed to generate {task.task_id}") from last_error

    image, noise_meta = apply_post_image_noise(
        rendered.image,
        instance_seed=int(instance_seed),
        params=task_params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    annotation_gt, projected_annotation = plan.bind_annotation(rendered)
    _prompt_defaults, prompt_artifacts = rectangular_solid_prompt_artifacts(
        prompt_defaults=prompt_defaults,
        prompt_task_key=str(plan.prompt_task_key),
        prompt_branch_key=str(plan.prompt_branch_key),
        json_example_annotation=annotation_gt.value,
        answer=int(plan.answer_gt.value),
        instance_seed=int(instance_seed),
    )
    trace_payload = _common_trace_payload(
        selected_branch=str(selected_branch),
        branch_probabilities=branch_probabilities,
        prompt_artifacts=prompt_artifacts,
        image_size=image.size,
        plan=plan,
        rendered=rendered,
        projected_annotation=projected_annotation,
        noise_meta=noise_meta,
        style_meta={
            "technical_diagram": dict(ctx.diagram_style_meta),
            "background": dict(ctx.background_meta),
        },
    )
    return TaskOutput(
        prompt=str(prompt_artifacts.prompt),
        answer_gt=plan.answer_gt,
        annotation_gt=annotation_gt,
        image=image,
        image_id="img0",
        trace_payload=trace_payload,
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
        query_id=str(selected_branch),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
    )


__all__ = ["RectangularSolidObjectivePlan", "run_rectangular_solid_public_entry"]
