"""Compute displacement over a marked velocity-time graph interval."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
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
from .shared.rendering import render_interval_graph
from .shared.sampling import (
    FLAT_VELOCITY_PROFILE,
    RAMPED_VELOCITY_PROFILE,
    make_interval_graph_spec,
    resolve_interval_axes,
)
from .shared.state import (
    SCENE_ID,
    IntervalAxes,
    MotionGraphRenderDefaults,
)


TASK_ID = "task_physics__motion_graph__interval_displacement_value"
TASK_NAMESPACE = "physics_motion_graph_interval_displacement_value"
TASK_PROMPT_KEY = "motion_graph_interval_displacement_value_query"
CONSTANT_VELOCITY_SEGMENT = "constant_velocity_interval_displacement"
CONSTANT_ACCELERATION_SEGMENT = "constant_acceleration_interval_displacement"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (
    CONSTANT_VELOCITY_SEGMENT,
    CONSTANT_ACCELERATION_SEGMENT,
)
_DEFAULTS = MotionGraphRenderDefaults(canvas_height=720, plot_top_px=86, y_min=0, y_max=8, t_max=8)
_SEGMENT_MODE_BY_BRANCH = {
    CONSTANT_VELOCITY_SEGMENT: FLAT_VELOCITY_PROFILE,
    CONSTANT_ACCELERATION_SEGMENT: RAMPED_VELOCITY_PROFILE,
}
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = (
    load_scene_generation_rendering_prompt_defaults(
        "physics",
        SCENE_ID,
        task_id=TASK_ID,
    )
)


def _select_branch(
    instance_seed: int,
    params: Mapping[str, Any],
) -> tuple[str, Dict[str, float], Dict[str, Any]]:
    """Select and validate the interval-displacement branch."""

    return select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=SUPPORTED_QUERY_IDS,
        default_query_id=CONSTANT_VELOCITY_SEGMENT,
        task_id=TASK_ID,
        namespace=f"{TASK_NAMESPACE}.branch",
    )


def _area_formula(branch_id: str) -> str:
    if str(branch_id) == CONSTANT_VELOCITY_SEGMENT:
        return "v * delta_t"
    return "((v_start + v_end) / 2) * delta_t"


@register_task
class PhysicsMotionGraphIntervalDisplacementValueTask:
    """Compute displacement over a marked interval of a velocity-time graph."""

    domain = "physics"
    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(
        self,
        instance_seed: int,
        *,
        params: Dict[str, Any],
        max_attempts: int,
    ) -> TaskOutput:
        """Generate a velocity-time graph and bind the interval displacement."""

        branch_id, branch_probabilities, task_params = _select_branch(
            int(instance_seed),
            dict(params or {}),
        )
        segment_profile = str(_SEGMENT_MODE_BY_BRANCH[str(branch_id)])
        last_error: Exception | None = None
        for attempt_index in range(max(1, int(max_attempts))):
            attempt_seed = int(instance_seed) + (attempt_index * 7919)
            try:
                axes: IntervalAxes = resolve_interval_axes(
                    instance_seed=attempt_seed,
                    params=task_params,
                    defaults=_GEN_DEFAULTS,
                    namespace=TASK_NAMESPACE,
                )
                spec = make_interval_graph_spec(
                    attempt_seed,
                    axes=axes,
                    params=task_params,
                    defaults=_GEN_DEFAULTS,
                    render_defaults=_RENDER_DEFAULTS,
                    segment_profile=segment_profile,
                    namespace=TASK_NAMESPACE,
                )
                rendered = render_interval_graph(
                    instance_seed=attempt_seed,
                    params=task_params,
                    spec=spec,
                    render_defaults=_RENDER_DEFAULTS,
                    fallback=_DEFAULTS,
                    namespace=TASK_NAMESPACE,
                )
            except Exception as exc:  # pragma: no cover - surfaced after retries.
                last_error = exc
                continue

            answer_gt = TypedValue("integer", int(spec.displacement_m))
            velocity_segment = rendered.render_map.get("velocity_segment_px")
            if not isinstance(velocity_segment, list) or len(velocity_segment) != 2:
                last_error = ValueError("interval renderer did not project velocity segment")
                continue
            annotation_gt = TypedValue(
                "segment",
                [
                    [round(float(point[0]), 3), round(float(point[1]), 3)]
                    for point in velocity_segment
                ],
            )
            json_example, json_example_answer_only = build_prompt_json_examples(
                annotation_value=annotation_gt.value,
                answer_type=str(answer_gt.type),
            )
            prompt_defaults = required_group_defaults(
                _PROMPT_DEFAULTS,
                ("bundle_id", "task_key"),
                context=f"prompt defaults for {TASK_ID}",
            )
            prompt_artifacts = build_motion_graph_prompt_artifacts(
                domain=self.domain,
                bundle_id=str(prompt_defaults.get("bundle_id", DEFAULT_PROMPT_BUNDLE_ID)),
                task_key=str(prompt_defaults.get("task_key", TASK_PROMPT_KEY)),
                prompt_branch_key=str(branch_id),
                dynamic_slots={
                    "json_example": str(json_example),
                    "json_example_answer_only": str(json_example_answer_only),
                },
                instance_seed=attempt_seed,
            )
            query_spec = build_prompt_query_spec(
                prompt_artifacts=prompt_artifacts,
                query_id=str(branch_id),
                params={
                    "query_id": str(branch_id),
                    "segment_mode": str(segment_profile),
                    "graph_kind": str(spec.graph_kind),
                    "target_answer": int(spec.displacement_m),
                    "query_id_probabilities": dict(branch_probabilities),
                    "scene_variant_probabilities": dict(axes.scene_style_probabilities),
                },
            )
            annotation_payload = build_segment_annotation_payload(annotation_gt.value)
            render_map = {
                **dict(rendered.render_map),
                "query_id": str(branch_id),
                "segment_mode": str(segment_profile),
            }
            trace_payload = {
                "scene_ir": {
                    "scene_kind": "physics_motion_graph_velocity_time_interval_displacement",
                    "entities": [dict(entity) for entity in rendered.scene_entities],
                    "relations": {
                        "query_id": str(branch_id),
                        "segment_mode": str(segment_profile),
                        "graph_kind": str(spec.graph_kind),
                        "displacement_m": int(spec.displacement_m),
                    },
                },
                "query_spec": query_spec,
                "render_spec": build_render_spec(
                    rendered,
                    scope="motion_graph_interval_diagram",
                ),
                "render_map": render_map,
                "execution_trace": {
                    "query_id": str(branch_id),
                    "segment_mode": str(segment_profile),
                    "graph_kind": str(spec.graph_kind),
                    "t_start_s": int(spec.t_start),
                    "t_end_s": int(spec.t_end),
                    "delta_t_s": int(spec.t_end) - int(spec.t_start),
                    "v_start_m_s": int(spec.v_start),
                    "v_end_m_s": int(spec.v_end),
                    "displacement_m": int(spec.displacement_m),
                    "t_values": [int(value) for value in spec.t_values],
                    "velocity_values_m_s": [int(value) for value in spec.velocity_values],
                    "area_formula": _area_formula(str(branch_id)),
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
                query_id=str(branch_id),
            )
        raise RuntimeError(
            f"failed to generate motion graph interval instance after "
            f"{max_attempts} attempts: {last_error}"
        )


__all__ = ["PhysicsMotionGraphIntervalDisplacementValueTask", "TASK_ID"]
