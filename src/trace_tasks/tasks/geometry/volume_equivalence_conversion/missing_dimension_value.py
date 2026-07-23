"""Solve a missing dimension after converting between equal-volume solids."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec

from .shared.annotations import MISSING_DIMENSION_ANNOTATION_KEYS, projected_annotation
from .shared.construction import (
    CONE_TO_CUBOID_HEIGHT_CASES,
    CUBOID_TO_CYLINDER_LENGTH_CASES,
    CYLINDER_TO_CONE_HEIGHT_CASES,
    bind_case_metadata,
    resolve_cone_to_cuboid_height,
    resolve_cuboid_to_cylinder_length,
    resolve_cylinder_to_cone_height,
    solid_volume,
)
from .shared.defaults import DOMAIN, SCENE_ID, load_volume_equivalence_defaults
from .shared.output import common_trace_sections, prepare_volume_equivalence_artifacts, prompt_render_spec
from .shared.rendering import render_missing_dimension_scene
from .shared.sampling import select_conversion_case
from .shared.state import RenderedScene, ResolvedProblem


TASK_ID = "task_geometry__volume_equivalence_conversion__missing_dimension_value"
TASK_ID_MISSING_DIMENSION = TASK_ID
QUERY_ID_CUBOID_TO_CYLINDER_LENGTH = "cuboid_to_cylinder_length"
QUERY_ID_CYLINDER_TO_CONE_HEIGHT = "cylinder_to_cone_height"
QUERY_ID_CONE_TO_CUBOID_HEIGHT = "cone_to_cuboid_height"
SUPPORTED_QUERY_IDS: tuple[str, ...] = (
    QUERY_ID_CUBOID_TO_CYLINDER_LENGTH,
    QUERY_ID_CYLINDER_TO_CONE_HEIGHT,
    QUERY_ID_CONE_TO_CUBOID_HEIGHT,
)
MISSING_DIMENSION_QUERY_IDS = SUPPORTED_QUERY_IDS
PROMPT_TASK_KEY = "missing_dimension_value_query"


def _resolve_problem(
    *,
    selected_branch: str,
    instance_seed: int,
    params: Mapping[str, Any],
) -> ResolvedProblem:
    if str(selected_branch) == QUERY_ID_CUBOID_TO_CYLINDER_LENGTH:
        cases = CUBOID_TO_CYLINDER_LENGTH_CASES
        resolver = resolve_cuboid_to_cylinder_length
        branch_name = "cuboid_to_cylinder_length"
    elif str(selected_branch) == QUERY_ID_CYLINDER_TO_CONE_HEIGHT:
        cases = CYLINDER_TO_CONE_HEIGHT_CASES
        resolver = resolve_cylinder_to_cone_height
        branch_name = "cylinder_to_cone_height"
    elif str(selected_branch) == QUERY_ID_CONE_TO_CUBOID_HEIGHT:
        cases = CONE_TO_CUBOID_HEIGHT_CASES
        resolver = resolve_cone_to_cuboid_height
        branch_name = "cone_to_cuboid_height"
    else:
        raise ValueError(f"unsupported query_id for {TASK_ID}: {selected_branch}")
    case, case_probabilities = select_conversion_case(
        branch_name=str(branch_name),
        cases=cases,
        instance_seed=int(instance_seed),
        params=params,
        namespace=f"{TASK_ID}.{selected_branch}.case",
    )
    problem = resolver(case)
    answer_support = [resolver(candidate).answer for candidate in cases]
    return bind_case_metadata(
        problem,
        case_probabilities=case_probabilities,
        answer_support=answer_support,
    )


def _query_params(
    *,
    selected_branch: str,
    branch_probabilities: Mapping[str, float],
    problem: ResolvedProblem,
) -> dict[str, Any]:
    return {
        "task_id": TASK_ID,
        "scene_id": SCENE_ID,
        "query_id": str(selected_branch),
        "query_id_probabilities": dict(branch_probabilities),
        "case_probabilities": dict(problem.case_probabilities),
        "answer_support_probabilities": dict(problem.answer_support_probabilities),
        "source_shape": problem.source.shape,
        "target_shape": problem.target.shape,
        "source_volume": int(solid_volume(problem.source)),
        "target_volume": int(solid_volume(problem.target)),
        "target_unknown_role": str(problem.target_unknown_role),
    }


def _trace_payload(
    *,
    selected_branch: str,
    branch_probabilities: Mapping[str, float],
    problem: ResolvedProblem,
    rendered: RenderedScene,
    prompt_artifacts: Any,
    annotation_value: Mapping[str, list[float]],
    noise_meta: Mapping[str, Any],
    image_size: tuple[int, int],
) -> dict[str, Any]:
    """Build task-specific trace sections from the resolved missing-dimension problem."""

    params = _query_params(
        selected_branch=str(selected_branch),
        branch_probabilities=branch_probabilities,
        problem=problem,
    )
    query_spec = build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts,
        query_id=str(selected_branch),
        params=params,
    )
    query_spec["task_id"] = TASK_ID
    query_spec["scene_id"] = SCENE_ID
    trace_payload = common_trace_sections(
        problem=problem,
        rendered=rendered,
        annotation_keys=MISSING_DIMENSION_ANNOTATION_KEYS,
        noise_meta=noise_meta,
        image_size=image_size,
        option_count=0,
    )
    trace_payload["scene_ir"].update({"task_id": TASK_ID, "query_id": str(selected_branch)})
    trace_payload["query_spec"] = query_spec
    trace_payload["render_spec"].update(
        {
            "task_id": TASK_ID,
            "query_id": str(selected_branch),
            "prompt": prompt_render_spec(prompt_artifacts),
        }
    )
    trace_payload["render_map"] = {"query_id": str(selected_branch), **dict(trace_payload["render_map"])}
    trace_payload["execution_trace"].update(
        {
            "task_id": TASK_ID,
            "query_id": str(selected_branch),
            "answer": int(problem.answer),
        }
    )
    trace_payload["witness_symbolic"] = dict(params)
    trace_payload["projected_annotation"] = projected_annotation(dict(annotation_value))
    return trace_payload


@register_task
class GeometryVolumeEquivalenceConversionMissingDimensionValueTask:
    """Solve a missing dimension after converting one solid to an equal-volume target solid."""

    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    domain = DOMAIN
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one missing-dimension formula task after binding its query branch."""

        selected_branch, branch_probabilities, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=params,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=SUPPORTED_QUERY_IDS[0],
            task_id=TASK_ID,
            namespace=f"{TASK_ID}.query",
        )
        problem = _resolve_problem(
            selected_branch=str(selected_branch),
            instance_seed=int(instance_seed),
            params=task_params,
        )
        _generation_defaults, render_defaults, prompt_defaults = load_volume_equivalence_defaults(TASK_ID)
        artifacts = prepare_volume_equivalence_artifacts(
            problem=problem,
            render_scene=render_missing_dimension_scene,
            prompt_task_key=PROMPT_TASK_KEY,
            prompt_branch_key=str(selected_branch),
            annotation_keys=MISSING_DIMENSION_ANNOTATION_KEYS,
            answer=int(problem.answer),
            instance_seed=int(instance_seed),
            params=task_params,
            max_attempts=int(max_attempts),
            render_defaults=render_defaults,
            prompt_defaults=prompt_defaults,
            random_namespace=TASK_ID,
        )
        return TaskOutput(
            prompt=str(artifacts.prompt_artifacts.prompt),
            answer_gt=TypedValue(type="integer", value=int(problem.answer)),
            annotation_gt=TypedValue(type="bbox_map", value=dict(artifacts.annotation_value)),
            image=artifacts.image,
            image_id="img0",
            trace_payload=_trace_payload(
                selected_branch=str(selected_branch),
                branch_probabilities=branch_probabilities,
                problem=problem,
                rendered=artifacts.rendered,
                prompt_artifacts=artifacts.prompt_artifacts,
                annotation_value=artifacts.annotation_value,
                noise_meta=artifacts.noise_meta,
                image_size=artifacts.image.size,
            ),
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=str(selected_branch),
            prompt_variants=dict(artifacts.prompt_artifacts.prompt_variants),
        )


__all__ = [
    "GeometryVolumeEquivalenceConversionMissingDimensionValueTask",
    "MISSING_DIMENSION_ANNOTATION_KEYS",
    "MISSING_DIMENSION_QUERY_IDS",
    "QUERY_ID_CONE_TO_CUBOID_HEIGHT",
    "QUERY_ID_CUBOID_TO_CYLINDER_LENGTH",
    "QUERY_ID_CYLINDER_TO_CONE_HEIGHT",
    "SCENE_ID",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
    "TASK_ID_MISSING_DIMENSION",
]
