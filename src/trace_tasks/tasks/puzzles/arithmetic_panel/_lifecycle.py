"""Scene-private lifecycle for arithmetic-constraint public tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec
from trace_tasks.tasks.shared.visual_defaults import (
    default_noise_fallback,
    load_scene_noise_defaults,
)
from trace_tasks.tasks.puzzles.shared.unit_size_jitter import (
    resolve_puzzle_unit_size_scale,
    with_puzzle_unit_size_jitter,
)

from .shared.annotations import target_bbox_annotation
from .shared.output import arithmetic_trace_params, build_arithmetic_trace_payload
from .shared.prompts import build_arithmetic_prompt_artifacts
from .shared.rendering import render_arithmetic_case
from .shared.sampling import resolve_scene_axes
from .shared.state import ArithmeticCase, SCENE_ID

AttemptCaseBuilder = Callable[[Any], ArithmeticCase]


@dataclass(frozen=True)
class ArithmeticObjectivePlan:
    """Task-owned arithmetic objective plan consumed by neutral lifecycle plumbing."""

    prompt_query_key: str
    attempt_namespace: str
    construct_case: AttemptCaseBuilder
    query_params: Mapping[str, Any] = field(default_factory=dict)


def run_arithmetic_lifecycle(
    *,
    task_id: str,
    domain: str,
    selected_query_id: str,
    query_probabilities: Mapping[str, float],
    gen_defaults: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    params: Mapping[str, Any],
    instance_seed: int,
    max_attempts: int,
    objective: ArithmeticObjectivePlan,
) -> TaskOutput:
    """Run scene rendering, prompt, annotation, and output assembly."""

    unit_scale, unit_meta = resolve_puzzle_unit_size_scale(
        params,
        render_defaults,
        instance_seed=int(instance_seed),
        namespace="puzzles.arithmetic_panel.unit_size",
        fallback_min=0.5,
        fallback_max=1.0,
    )
    effective_params = dict(params)
    effective_params["unit_size_scale"] = float(unit_scale)
    effective_params["unit_size_jitter"] = dict(unit_meta)
    axes = resolve_scene_axes(
        instance_seed=int(instance_seed),
        params=effective_params,
        gen_defaults=gen_defaults,
        namespace="puzzles.arithmetic_panel",
    )
    effective_params["scene_variant"] = str(axes.scene_variant)
    visual_defaults = load_scene_noise_defaults(
        domain=str(domain),
        scene_id=SCENE_ID,
        fallback=default_noise_fallback(apply_prob=0.12),
        merge_with_fallback=True,
    )

    for attempt_index in range(max(1, int(max_attempts))):
        rng = spawn_rng(
            int(instance_seed),
            f"{objective.attempt_namespace}.attempt.{int(attempt_index)}",
        )
        try:
            case = objective.construct_case(rng)
            rendered_context = render_arithmetic_case(
                case=case,
                instance_seed=int(instance_seed) + int(attempt_index),
                params=effective_params,
                render_defaults=render_defaults,
                visual_defaults=visual_defaults,
            )
            annotation_artifacts = target_bbox_annotation(
                rendered_scene=rendered_context.rendered_scene,
                target_item_id=str(case.target_item_id),
            )
        except ValueError:
            continue

        prompt_defaults, prompt_artifacts = build_arithmetic_prompt_artifacts(
            domain=str(domain),
            prompt_query_key=str(objective.prompt_query_key),
            instance_seed=int(instance_seed),
        )
        query_spec = build_prompt_query_spec(
            prompt_artifacts=prompt_artifacts,
            query_id=str(selected_query_id),
            params=arithmetic_trace_params(
                axes=axes,
                case=case,
                prompt_query_key=str(objective.prompt_query_key),
                extra_params={
                    **dict(objective.query_params),
                    "branch_probabilities": dict(query_probabilities),
                },
            ),
        )
        trace_payload = build_arithmetic_trace_payload(
            annotation_artifacts=annotation_artifacts,
            annotation_item_id=str(case.target_item_id),
            axes=axes,
            case=case,
            rendered_context=rendered_context,
            prompt_defaults=prompt_defaults,
            prompt_artifacts=prompt_artifacts,
            query_spec=query_spec,
            execution_extra={
                "answer": int(case.answer_value),
            },
        )
        return TaskOutput(
            prompt=str(prompt_artifacts.prompt),
            prompt_variants=dict(prompt_artifacts.prompt_variants),
            answer_gt=TypedValue(type="integer", value=int(case.answer_value)),
            annotation_gt=annotation_artifacts.annotation_gt,
            image=rendered_context.image,
            image_id="img0",
            trace_payload=trace_payload,
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=str(selected_query_id),
        )

    raise RuntimeError(
        f"{task_id} failed to generate an arithmetic-constraint instance"
    )


__all__ = [
    "ArithmeticObjectivePlan",
    "run_arithmetic_lifecycle",
]
