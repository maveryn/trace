"""Compute a cylinder side path length from its unwrapped rectangular net."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Mapping, Tuple

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.core.sampling import uniform_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.geometry.shared.pythagorean import IntegerRightTriangle, integer_right_triangles
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import split_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.deterministic_sampling import resolve_selection_index, uniform_probability_map
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions

from ._lifecycle import build_trace_payload, render_cylinder_wrap_runtime
from .shared.defaults import DOMAIN, SCENE_ID
from .shared.rendering import render_surface_path_scene
from .shared.state import SurfacePathProblem

TASK_ID = "task_geometry__cylinder_wrap__surface_path_length_value"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = ("single",)
SCENE_VARIANT = "surface_net_path"
FORMULA_SCHEMA = "surface_path_pythagorean_length"
PROMPT_FIELD_PREFIX = "surface_path"

_SCENE_DEFAULTS = get_scene_defaults(DOMAIN, SCENE_ID)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    _SCENE_DEFAULTS,
    task_id=TASK_ID,
)


@dataclass(frozen=True)
class _SurfacePathRequest:
    """Task-owned path-length sampling and answer binding."""

    selected_query: str
    query_probabilities: Mapping[str, float]
    params: Mapping[str, Any]
    problem: SurfacePathProblem
    answer_support: Tuple[int, ...]
    path_length_probabilities: Mapping[str, float]
    case_key: str


@lru_cache(maxsize=1)
def _cases_by_path_length() -> Mapping[int, Tuple[IntegerRightTriangle, ...]]:
    """Group readable integer right-triangle cases by path length."""

    grouped: dict[int, list[IntegerRightTriangle]] = {}
    for triangle in integer_right_triangles(
        min_leg=5,
        max_leg=110,
        max_hypotenuse=160,
        include_swapped=True,
    ):
        leg_ratio = float(triangle.leg_b) / float(triangle.leg_a)
        if not (0.4 <= leg_ratio <= 2.5):
            continue
        grouped.setdefault(int(triangle.hypotenuse), []).append(triangle)
    if len(grouped) < 50:
        raise RuntimeError("cylinder surface-path support unexpectedly small")
    return {
        int(path_length): tuple(cases)
        for path_length, cases in sorted(grouped.items(), key=lambda item: int(item[0]))
    }


def _answer_support() -> Tuple[int, ...]:
    return tuple(int(value) for value in _cases_by_path_length().keys())


def _resolve_public_branch(*, instance_seed: int, params: Mapping[str, Any]) -> tuple[str, Mapping[str, float], Mapping[str, Any]]:
    selected_query, query_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=SUPPORTED_QUERY_IDS,
        default_query_id="single",
        task_id=TASK_ID,
    )
    return str(selected_query), dict(query_probabilities), dict(task_params)


def _resolve_surface_path_request(*, instance_seed: int, params: Mapping[str, Any]) -> _SurfacePathRequest:
    """Choose the answer value first, then bind one compatible net."""

    selected_query, query_probabilities, task_params = _resolve_public_branch(
        instance_seed=int(instance_seed),
        params=params,
    )
    cases_by_answer = _cases_by_path_length()
    support = _answer_support()
    explicit = task_params.get("target_path_length")
    if explicit is not None:
        path_length = int(explicit)
        if path_length not in cases_by_answer:
            raise ValueError(f"target_path_length={path_length} is not supported by {TASK_ID}")
        probabilities = uniform_probability_map(support, selected=path_length)
    else:
        rng = spawn_rng(int(instance_seed), f"{TASK_ID}.path_length")
        path_length = int(uniform_choice(rng, support))
        probabilities = uniform_probability_map(support)
    compatible_cases = tuple(cases_by_answer[int(path_length)])
    rng = spawn_rng(int(instance_seed), f"{TASK_ID}.case.{path_length}")
    triangle = uniform_choice(rng, compatible_cases)
    problem = SurfacePathProblem(
        circumference=int(triangle.leg_a),
        height=int(triangle.leg_b),
        path_length=int(triangle.hypotenuse),
    )
    return _SurfacePathRequest(
        selected_query=selected_query,
        query_probabilities=query_probabilities,
        params=task_params,
        problem=problem,
        answer_support=support,
        path_length_probabilities=probabilities,
        case_key=str(triangle.key),
    )


@register_task
class GeometryCylinderWrapSurfacePathLengthValueTask:
    """Compute a marked path length on an unwrapped cylinder side."""

    task_id = TASK_ID
    reasoning_operations = ('topology', 'transformation', 'formula_evaluation')
    domain = DOMAIN
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS
    reasoning_kind = "cylinder_wrap"

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one path-length instance; this file owns output binding."""

        request = _resolve_surface_path_request(
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
            render_scene=render_surface_path_scene,
        )

        answer_gt = TypedValue(type="integer", value=int(runtime.rendered.answer))
        annotation_value = dict(runtime.annotation_value)
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
                answer_value=int(runtime.rendered.answer),
                query_params={
                    "answer_support": [int(value) for value in request.answer_support],
                    "path_length_probabilities": dict(request.path_length_probabilities),
                    "case_key": str(request.case_key),
                },
            ),
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=str(request.selected_query),
            prompt_variants=dict(runtime.prompt_artifacts.prompt_variants),
        )


__all__ = ["GeometryCylinderWrapSurfacePathLengthValueTask"]
