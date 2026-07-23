"""Match an unwrapped cylinder strip mark to a top-view rim candidate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import split_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.deterministic_sampling import resolve_selection_index
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.labeling import LABEL_POOL_SAFE_UPPER, assign_random_shuffled_labels
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.geometry.shared.option_count import resolve_geometry_option_count

from ._lifecycle import build_trace_payload, render_cylinder_wrap_runtime
from .shared.defaults import DOMAIN, SCENE_ID
from .shared.rendering import render_wrapped_mark_scene
from .shared.state import WrappedMarkProblem

TASK_ID = "task_geometry__cylinder_wrap__wrapped_mark_position_label"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = ("single",)
SCENE_VARIANT = "strip_to_rim_position"
FORMULA_SCHEMA = "strip_mark_to_rim_candidate"
PROMPT_FIELD_PREFIX = "wrapped_mark"

_SCENE_DEFAULTS = get_scene_defaults(DOMAIN, SCENE_ID)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    _SCENE_DEFAULTS,
    task_id=TASK_ID,
)


@dataclass(frozen=True)
class _WrappedMarkRequest:
    """Task-owned option and answer binding before rendering."""

    selected_query: str
    query_probabilities: Mapping[str, float]
    params: Mapping[str, Any]
    problem: WrappedMarkProblem
    target_index_probabilities: Mapping[str, float]
    option_count_probabilities: Mapping[str, float]


def _resolve_public_branch(*, instance_seed: int, params: Mapping[str, Any]) -> tuple[str, Mapping[str, float], Mapping[str, Any]]:
    selected_query, query_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=SUPPORTED_QUERY_IDS,
        default_query_id="single",
        task_id=TASK_ID,
    )
    return str(selected_query), dict(query_probabilities), dict(task_params)


def _target_index_probabilities(option_count: int, selected: int | None = None) -> Dict[str, float]:
    if selected is not None:
        return {str(int(selected)): 1.0}
    probability = 1.0 / float(option_count)
    return {str(index): float(probability) for index in range(int(option_count))}


def _resolve_wrapped_mark_request(*, instance_seed: int, params: Mapping[str, Any]) -> _WrappedMarkRequest:
    """Choose candidate count and target candidate in the public task file."""

    selected_query, query_probabilities, task_params = _resolve_public_branch(
        instance_seed=int(instance_seed),
        params=params,
    )
    option_count, option_count_probabilities = resolve_geometry_option_count(
        params=task_params,
        gen_defaults=_GEN_DEFAULTS,
        field_name="option_count",
        supported_counts=(4, 6),
        task_id=TASK_ID,
        instance_seed=int(instance_seed),
    )
    if int(option_count) < 4:
        raise ValueError("wrapped mark task requires at least four candidate positions")
    if int(option_count) > len(LABEL_POOL_SAFE_UPPER):
        raise ValueError("option_count exceeds safe label pool")
    explicit = task_params.get("target_index")
    if explicit is not None:
        target_index = int(explicit)
        if target_index < 0 or target_index >= int(option_count):
            raise ValueError("target_index is outside option count")
        target_index_probabilities = _target_index_probabilities(int(option_count), selected=target_index)
    else:
        index = resolve_selection_index(
            params=task_params,
            instance_seed=int(instance_seed),
            namespace=f"{TASK_ID}.target_index",
        )
        target_index = int(index) % int(option_count)
        target_index_probabilities = _target_index_probabilities(int(option_count))
    label_rng = spawn_rng(int(instance_seed), f"{TASK_ID}.labels")
    labels = assign_random_shuffled_labels(label_rng, object_count=int(option_count))
    problem = WrappedMarkProblem(
        option_count=int(option_count),
        target_index=int(target_index),
        option_labels=tuple(str(label) for label in labels),
        answer_label=str(labels[int(target_index)]),
    )
    return _WrappedMarkRequest(
        selected_query=selected_query,
        query_probabilities=query_probabilities,
        params=task_params,
        problem=problem,
        target_index_probabilities=target_index_probabilities,
        option_count_probabilities=dict(option_count_probabilities),
    )


@register_task
class GeometryCylinderWrapWrappedMarkPositionLabelTask:
    """Match a mark on an unwrapped strip to a labeled rim position."""

    task_id = TASK_ID
    reasoning_operations = ('transformation', 'matching')
    domain = DOMAIN
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS
    reasoning_kind = "cylinder_wrap"

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one position-matching instance; this file owns output binding."""

        request = _resolve_wrapped_mark_request(
            instance_seed=int(instance_seed),
            params=params,
        )
        runtime = render_cylinder_wrap_runtime(
            instance_seed=int(instance_seed),
            params=request.params,
            render_defaults=_RENDER_DEFAULTS,
            prompt_defaults=_PROMPT_DEFAULTS,
            field_prefix=PROMPT_FIELD_PREFIX,
            max_attempts=int(max_attempts),
            problem=request.problem,
            render_scene=render_wrapped_mark_scene,
        )

        annotation_value = dict(runtime.annotation_value)
        answer_gt = TypedValue(type="option_letter", value=str(runtime.rendered.answer))
        annotation_gt = TypedValue(type=str(runtime.rendered.annotation_type), value=dict(annotation_value))
        return TaskOutput(
            prompt=str(runtime.prompt_artifacts.prompt),
            answer_gt=answer_gt,
            annotation_gt=annotation_gt,
            image=runtime.image,
            image_id="img0",
            trace_payload=build_trace_payload(
                scene_variant=SCENE_VARIANT,
                formula_schema=FORMULA_SCHEMA,
                selected_query=str(request.selected_query),
                query_probabilities=dict(request.query_probabilities),
                rendered=runtime.rendered,
                prompt_defaults=runtime.prompt_defaults,
                prompt_artifacts=runtime.prompt_artifacts,
                render_meta=dict(runtime.render_meta),
                noise_meta=dict(runtime.noise_meta),
                image_size=(int(runtime.image.size[0]), int(runtime.image.size[1])),
                annotation_value=annotation_value,
                answer_value=str(runtime.rendered.answer),
                query_params={
                    "answer_support": list(LABEL_POOL_SAFE_UPPER),
                    "target_index_probabilities": dict(request.target_index_probabilities),
                    "option_count_probabilities": dict(request.option_count_probabilities),
                },
            ),
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=str(request.selected_query),
            prompt_variants=dict(runtime.prompt_artifacts.prompt_variants),
        )


__all__ = ["GeometryCylinderWrapWrappedMarkPositionLabelTask"]
