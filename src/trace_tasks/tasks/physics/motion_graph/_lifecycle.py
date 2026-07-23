"""Scene-private lifecycle helpers for motion-graph objectives."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Tuple

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.shared.config_defaults import (
    load_scene_generation_rendering_prompt_defaults,
    required_group_defaults,
)
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_json_example import build_prompt_json_examples
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec

from .shared.output import build_render_spec, build_segment_annotation_payload
from .shared.prompts import (
    PROMPT_BUNDLE_ID as DEFAULT_PROMPT_BUNDLE_ID,
    build_motion_graph_prompt_artifacts,
)
from .shared.rendering import render_average_speed_graph, render_state_choice_graph
from .shared.sampling import (
    make_average_speed_graph_spec,
    make_state_graph_spec,
    resolve_interval_axes,
    resolve_state_axes,
)
from .shared.state import (
    SCENE_ID,
    IntervalAxes,
    MotionGraphRenderDefaults,
    OPTION_LETTERS,
)


@dataclass(frozen=True)
class StateChoiceLifecyclePlan:
    """Task-owned objective parameters consumed by the shared state-choice lifecycle."""

    task_identifier: str
    namespace: str
    prompt_key: str
    public_branch_ids: Tuple[str, ...]
    default_branch_id: str
    operation_label: str
    prompt_branch_key: str
    state_profile: str
    state_support: Tuple[str, ...]
    graph_kind: str
    y_axis_label: str
    title: str
    error_label: str
    fallback_render_defaults: MotionGraphRenderDefaults = field(
        default_factory=MotionGraphRenderDefaults
    )


@dataclass(frozen=True)
class AverageSpeedLifecyclePlan:
    """Task-owned objective parameters for the average-speed lifecycle."""

    task_identifier: str
    namespace: str
    prompt_key: str
    prompt_branch_key: str
    public_branch_ids: Tuple[str, ...] = (SINGLE_QUERY_ID,)
    default_branch_id: str = SINGLE_QUERY_ID
    fallback_render_defaults: MotionGraphRenderDefaults = field(
        default_factory=lambda: MotionGraphRenderDefaults(
            canvas_height=720,
            plot_top_px=86,
            y_min=0,
            y_max=12,
            t_max=8,
        )
    )


def run_state_choice_lifecycle(
    *,
    domain: str,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    plan: StateChoiceLifecyclePlan,
) -> TaskOutput:
    """Run the common graph/render/prompt loop for one state-choice objective."""

    generation_defaults, rendering_defaults, prompt_defaults_group = (
        load_scene_generation_rendering_prompt_defaults(
            "physics",
            SCENE_ID,
            task_id=str(plan.task_identifier),
        )
    )
    branch_label, branch_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=dict(params or {}),
        supported_query_ids=tuple(str(value) for value in plan.public_branch_ids),
        default_query_id=str(plan.default_branch_id),
        task_id=str(plan.task_identifier),
        namespace=f"{plan.namespace}.branch",
    )
    last_error: Exception | None = None
    for attempt_index in range(max(1, int(max_attempts))):
        attempt_seed = int(instance_seed) + (attempt_index * 7919)
        try:
            axes = resolve_state_axes(
                instance_seed=attempt_seed,
                params=task_params,
                defaults=generation_defaults,
                namespace=str(plan.namespace),
                supported_states=tuple(str(value) for value in plan.state_support),
            )
            spec = make_state_graph_spec(
                attempt_seed,
                axes=axes,
                params=task_params,
                render_defaults=rendering_defaults,
                operation_label=str(plan.operation_label),
                state_profile=str(plan.state_profile),
                supported_states=tuple(str(value) for value in plan.state_support),
                graph_kind=str(plan.graph_kind),
                y_axis_label=str(plan.y_axis_label),
                title=str(plan.title),
                namespace=str(plan.namespace),
            )
            rendered = render_state_choice_graph(
                instance_seed=attempt_seed,
                params=task_params,
                spec=spec,
                render_defaults=rendering_defaults,
                fallback=plan.fallback_render_defaults,
                namespace=str(plan.namespace),
            )
        except Exception as exc:  # pragma: no cover - surfaced after retries.
            last_error = exc
            continue

        answer_gt = TypedValue("option_letter", str(spec.correct_option_letter))
        curve_segment = rendered.render_map.get("curve_segment_px")
        if not isinstance(curve_segment, list) or len(curve_segment) != 2:
            last_error = ValueError("state-choice renderer did not project curve segment")
            continue
        annotation_gt = TypedValue(
            "segment",
            [
                [round(float(point[0]), 3), round(float(point[1]), 3)]
                for point in curve_segment
            ],
        )
        json_example, json_example_answer_only = build_prompt_json_examples(
            annotation_value=annotation_gt.value,
            answer_type=str(answer_gt.type),
        )
        prompt_defaults = required_group_defaults(
            prompt_defaults_group,
            ("bundle_id", "task_key"),
            context=f"prompt defaults for {plan.task_identifier}",
        )
        prompt_artifacts = build_motion_graph_prompt_artifacts(
            domain=str(domain),
            bundle_id=str(prompt_defaults.get("bundle_id", DEFAULT_PROMPT_BUNDLE_ID)),
            task_key=str(prompt_defaults.get("task_key", plan.prompt_key)),
            prompt_branch_key=str(plan.prompt_branch_key),
            dynamic_slots={
                "json_example": str(json_example),
                "json_example_answer_only": str(json_example_answer_only),
            },
            instance_seed=attempt_seed,
        )
        query_spec = build_prompt_query_spec(
            prompt_artifacts=prompt_artifacts,
            query_id=str(branch_label),
            params={
                "query_id": str(branch_label),
                "motion_operation": str(plan.operation_label),
                "graph_kind": str(spec.graph_kind),
                "motion_state": str(spec.motion_state),
                "target_answer": str(spec.correct_option_letter),
                "answer_support": list(OPTION_LETTERS),
                "query_id_probabilities": dict(branch_probabilities),
                "scene_variant_probabilities": dict(axes.scene_style_probabilities),
                "motion_state_probabilities": dict(axes.motion_state_probabilities),
                "correct_option_letter_probabilities": dict(
                    axes.correct_option_letter_probabilities
                ),
            },
        )
        annotation_payload = build_segment_annotation_payload(annotation_gt.value)
        trace_payload = {
            "scene_ir": {
                "scene_kind": f"physics_motion_graph_{str(spec.graph_kind)}",
                "entities": [dict(entity) for entity in rendered.scene_entities],
                "relations": {
                    "query_id": str(branch_label),
                    "motion_operation": str(plan.operation_label),
                    "graph_kind": str(spec.graph_kind),
                    "motion_state": str(spec.motion_state),
                    "correct_option_letter": str(spec.correct_option_letter),
                },
            },
            "query_spec": query_spec,
            "render_spec": build_render_spec(
                rendered,
                scope="motion_graph_diagram",
            ),
            "render_map": {
                **dict(rendered.render_map),
                "query_id": str(branch_label),
                "motion_operation": str(plan.operation_label),
            },
            "execution_trace": {
                "query_id": str(branch_label),
                "motion_operation": str(plan.operation_label),
                "graph_kind": str(spec.graph_kind),
                "motion_state": str(spec.motion_state),
                "option_map": dict(spec.option_map),
                "correct_option_letter": str(spec.correct_option_letter),
                "target_segment": {
                    "index": int(spec.target_segment_index),
                    "t_start": int(spec.t_values[spec.target_segment_index]),
                    "t_end": int(spec.t_values[spec.target_segment_index + 1]),
                    "y_start": int(spec.y_values[spec.target_segment_index]),
                    "y_end": int(spec.y_values[spec.target_segment_index + 1]),
                },
            },
            "sampling": {
                "query_id_probabilities": dict(branch_probabilities),
                "scene_variant_probabilities": dict(axes.scene_style_probabilities),
                "motion_state_probabilities": dict(axes.motion_state_probabilities),
                "correct_option_letter_probabilities": dict(
                    axes.correct_option_letter_probabilities
                ),
            },
            "background": dict(rendered.render_map.get("background_style", {})),
            "post_image_noise": dict(rendered.render_map.get("post_image_noise", {})),
            **annotation_payload,
        }
        return TaskOutput(
            prompt=str(prompt_artifacts.prompt),
            prompt_variants=dict(prompt_artifacts.prompt_variants),
            answer_gt=answer_gt,
            annotation_gt=annotation_gt,
            image=rendered.image,
            image_id="img0",
            trace_payload=trace_payload,
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=str(branch_label),
        )
    raise RuntimeError(
        f"failed to generate motion graph {plan.error_label} instance after "
        f"{max_attempts} attempts: {last_error}"
    )


def run_average_speed_lifecycle(
    *,
    domain: str,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    plan: AverageSpeedLifecyclePlan,
) -> TaskOutput:
    """Run the graph/render/prompt loop for the average-speed value objective."""

    generation_defaults, rendering_defaults, prompt_defaults_group = (
        load_scene_generation_rendering_prompt_defaults(
            "physics",
            SCENE_ID,
            task_id=str(plan.task_identifier),
        )
    )
    branch_label, branch_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=dict(params or {}),
        supported_query_ids=tuple(str(value) for value in plan.public_branch_ids),
        default_query_id=str(plan.default_branch_id),
        task_id=str(plan.task_identifier),
        namespace=f"{plan.namespace}.branch",
    )
    last_error: Exception | None = None
    for attempt_index in range(max(1, int(max_attempts))):
        attempt_seed = int(instance_seed) + (attempt_index * 7919)
        try:
            axes: IntervalAxes = resolve_interval_axes(
                instance_seed=attempt_seed,
                params=task_params,
                defaults=generation_defaults,
                namespace=str(plan.namespace),
            )
            spec = make_average_speed_graph_spec(
                attempt_seed,
                axes=axes,
                params=task_params,
                defaults=generation_defaults,
                render_defaults=rendering_defaults,
                namespace=str(plan.namespace),
            )
            rendered = render_average_speed_graph(
                instance_seed=attempt_seed,
                params=task_params,
                spec=spec,
                render_defaults=rendering_defaults,
                fallback=plan.fallback_render_defaults,
                namespace=str(plan.namespace),
            )
        except Exception as exc:  # pragma: no cover - surfaced after retries.
            last_error = exc
            continue

        answer_gt = TypedValue("integer", int(spec.average_speed_m_s))
        distance_segment = rendered.render_map.get("distance_segment_px")
        if not isinstance(distance_segment, list) or len(distance_segment) != 2:
            last_error = ValueError("average-speed renderer did not project distance segment")
            continue
        annotation_gt = TypedValue(
            "segment",
            [
                [round(float(point[0]), 3), round(float(point[1]), 3)]
                for point in distance_segment
            ],
        )
        json_example, json_example_answer_only = build_prompt_json_examples(
            annotation_value=annotation_gt.value,
            answer_type=str(answer_gt.type),
        )
        prompt_defaults = required_group_defaults(
            prompt_defaults_group,
            ("bundle_id", "task_key"),
            context=f"prompt defaults for {plan.task_identifier}",
        )
        prompt_artifacts = build_motion_graph_prompt_artifacts(
            domain=str(domain),
            bundle_id=str(prompt_defaults.get("bundle_id", DEFAULT_PROMPT_BUNDLE_ID)),
            task_key=str(prompt_defaults.get("task_key", plan.prompt_key)),
            prompt_branch_key=str(plan.prompt_branch_key),
            dynamic_slots={
                "json_example": str(json_example),
                "json_example_answer_only": str(json_example_answer_only),
            },
            instance_seed=attempt_seed,
        )
        query_spec = build_prompt_query_spec(
            prompt_artifacts=prompt_artifacts,
            query_id=str(branch_label),
            params={
                "query_id": str(branch_label),
                "graph_kind": str(spec.graph_kind),
                "target_answer": int(spec.average_speed_m_s),
                "query_id_probabilities": dict(branch_probabilities),
                "scene_variant_probabilities": dict(axes.scene_style_probabilities),
            },
        )
        annotation_payload = build_segment_annotation_payload(annotation_gt.value)
        trace_payload = {
            "scene_ir": {
                "scene_kind": "physics_motion_graph_distance_time_average_speed",
                "entities": [dict(entity) for entity in rendered.scene_entities],
                "relations": {
                    "query_id": str(branch_label),
                    "graph_kind": str(spec.graph_kind),
                    "average_speed_m_s": int(spec.average_speed_m_s),
                },
            },
            "query_spec": query_spec,
            "render_spec": build_render_spec(
                rendered,
                scope="motion_graph_average_speed_diagram",
            ),
            "render_map": {
                **dict(rendered.render_map),
                "query_id": str(branch_label),
            },
            "execution_trace": {
                "query_id": str(branch_label),
                "graph_kind": str(spec.graph_kind),
                "t_start_s": int(spec.t_start),
                "t_end_s": int(spec.t_end),
                "delta_t_s": int(spec.t_end) - int(spec.t_start),
                "d_start_m": int(spec.d_start),
                "d_end_m": int(spec.d_end),
                "delta_d_m": int(spec.d_end) - int(spec.d_start),
                "average_speed_m_s": int(spec.average_speed_m_s),
                "t_values": [int(value) for value in spec.t_values],
                "distance_values_m": [int(value) for value in spec.distance_values],
                "rate_formula": "(d_end - d_start) / (t_end - t_start)",
            },
            "sampling": {
                "query_id_probabilities": dict(branch_probabilities),
                "scene_variant_probabilities": dict(axes.scene_style_probabilities),
            },
            "background": dict(rendered.render_map.get("background_style", {})),
            "post_image_noise": dict(rendered.render_map.get("post_image_noise", {})),
            **annotation_payload,
        }
        return TaskOutput(
            prompt=str(prompt_artifacts.prompt),
            prompt_variants=dict(prompt_artifacts.prompt_variants),
            answer_gt=answer_gt,
            annotation_gt=annotation_gt,
            image=rendered.image,
            image_id="img0",
            trace_payload=trace_payload,
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=str(branch_label),
        )
    raise RuntimeError(
        f"failed to generate motion graph average-speed instance after "
        f"{max_attempts} attempts: {last_error}"
    )


__all__ = [
    "AverageSpeedLifecyclePlan",
    "StateChoiceLifecyclePlan",
    "run_average_speed_lifecycle",
    "run_state_choice_lifecycle",
]
