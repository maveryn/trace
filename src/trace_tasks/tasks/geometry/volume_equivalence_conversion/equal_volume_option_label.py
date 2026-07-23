"""Select the option solid that has equal volume to the source solid."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.geometry.shared.option_count import resolve_geometry_option_count
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec

from .shared.annotations import OPTION_ANNOTATION_KEYS, projected_bbox_annotation
from .shared.construction import (
    CONE_SOURCE_OPTION_CASES,
    CUBOID_SOURCE_OPTION_CASES,
    CYLINDER_SOURCE_OPTION_CASES,
    bind_option_metadata,
    resolve_cone_matching_cylinder_option,
    resolve_cuboid_matching_cylinder_option,
    resolve_cylinder_matching_cone_option,
    solid_volume,
)
from .shared.defaults import DOMAIN, SCENE_ID, load_volume_equivalence_defaults
from .shared.output import common_trace_sections, prepare_volume_equivalence_artifacts, prompt_render_spec
from .shared.rendering import render_option_scene
from .shared.sampling import select_conversion_case
from .shared.state import RenderedScene, ResolvedProblem


TASK_ID = "task_geometry__volume_equivalence_conversion__equal_volume_option_label"
TASK_ID_EQUAL_VOLUME_OPTION = TASK_ID
QUERY_ID_CONE_MATCHES_CYLINDER_OPTION = "cone_matches_cylinder_option"
QUERY_ID_CYLINDER_MATCHES_CONE_OPTION = "cylinder_matches_cone_option"
QUERY_ID_CUBOID_MATCHES_CYLINDER_OPTION = "cuboid_matches_cylinder_option"
SUPPORTED_QUERY_IDS: tuple[str, ...] = (
    QUERY_ID_CONE_MATCHES_CYLINDER_OPTION,
    QUERY_ID_CYLINDER_MATCHES_CONE_OPTION,
    QUERY_ID_CUBOID_MATCHES_CYLINDER_OPTION,
)
EQUAL_VOLUME_OPTION_QUERY_IDS = SUPPORTED_QUERY_IDS
PROMPT_TASK_KEY = "equal_volume_option_label_query"
_OptionResolver = Callable[..., ResolvedProblem]


@dataclass(frozen=True)
class _OptionBranchProgram:
    source_cases: tuple[tuple[int, ...], ...]
    resolver: _OptionResolver
    branch_name: str

    def resolve(
        self,
        *,
        case: Sequence[int],
        option_count: int,
        instance_seed: int,
        params: Mapping[str, Any],
        branch_key: str,
    ) -> ResolvedProblem:
        return self.resolver(
            case,
            option_count=int(option_count),
            instance_seed=int(instance_seed),
            params=params,
            shuffle_namespace=f"{TASK_ID}.{branch_key}.option_distractors",
            label_namespace=f"{TASK_ID}.{branch_key}.answer_label",
        )


_PROGRAMS: dict[str, _OptionBranchProgram] = {
    QUERY_ID_CONE_MATCHES_CYLINDER_OPTION: (
        _OptionBranchProgram(
            source_cases=CONE_SOURCE_OPTION_CASES,
            resolver=resolve_cone_matching_cylinder_option,
            branch_name="cone_matches_cylinder_option",
        )
    ),
    QUERY_ID_CYLINDER_MATCHES_CONE_OPTION: (
        _OptionBranchProgram(
            source_cases=CYLINDER_SOURCE_OPTION_CASES,
            resolver=resolve_cylinder_matching_cone_option,
            branch_name="cylinder_matches_cone_option",
        )
    ),
    QUERY_ID_CUBOID_MATCHES_CYLINDER_OPTION: (
        _OptionBranchProgram(
            source_cases=CUBOID_SOURCE_OPTION_CASES,
            resolver=resolve_cuboid_matching_cylinder_option,
            branch_name="cuboid_matches_cylinder_option",
        )
    ),
}


def _resolve_problem(
    *,
    selected_branch: str,
    instance_seed: int,
    params: Mapping[str, Any],
) -> ResolvedProblem:
    generation_defaults, _render_defaults, _prompt_defaults = load_volume_equivalence_defaults(TASK_ID)
    option_count, option_count_probabilities = resolve_geometry_option_count(
        params=params,
        gen_defaults=generation_defaults,
        field_name="option_count",
        supported_counts=(4, 6),
        task_id=TASK_ID,
        instance_seed=int(instance_seed),
    )
    program = _PROGRAMS.get(str(selected_branch))
    if program is None:
        raise ValueError(f"unsupported query_id for {TASK_ID}: {selected_branch}")
    case, case_probabilities = select_conversion_case(
        branch_name=program.branch_name,
        cases=program.source_cases,
        instance_seed=int(instance_seed),
        params=params,
        namespace=f"{TASK_ID}.{selected_branch}.case",
    )
    problem = program.resolve(
        case=case,
        option_count=int(option_count),
        instance_seed=int(instance_seed),
        params=params,
        branch_key=str(selected_branch),
    )
    return bind_option_metadata(
        problem,
        case_probabilities=case_probabilities,
        option_count_probabilities=option_count_probabilities,
    )


def _query_params(
    *,
    selected_branch: str,
    branch_probabilities: Mapping[str, float],
    problem: ResolvedProblem,
) -> dict[str, Any]:
    option_labels = [str(option.label) for option in problem.option_specs]
    return {
        "task_id": TASK_ID,
        "scene_id": SCENE_ID,
        "query_id": str(selected_branch),
        "query_id_probabilities": dict(branch_probabilities),
        "case_probabilities": dict(problem.case_probabilities),
        "answer_support_probabilities": dict(problem.answer_support_probabilities),
        "option_count_probabilities": dict(problem.option_count_probabilities),
        "source_shape": problem.source.shape,
        "target_shape": problem.target.shape,
        "source_volume": int(solid_volume(problem.source)),
        "target_volume": int(solid_volume(problem.target)),
        "target_unknown_role": str(problem.target_unknown_role),
        "selected_option_label": str(problem.selected_option_label),
        "option_labels": option_labels,
        "option_shapes": {str(option.label): option.solid.shape for option in problem.option_specs},
        "option_volumes": {str(option.label): int(option.volume) for option in problem.option_specs},
    }


def _option_trace(problem: ResolvedProblem) -> list[dict[str, Any]]:
    """Serialize the visual option candidates for this selection objective."""

    return [
        {
            "label": str(option.label),
            "shape": option.solid.shape,
            "base_area": int(option.solid.base_area),
            "height": int(option.solid.height),
            "length": int(option.solid.length),
            "width": int(option.solid.width),
            "volume": int(option.volume),
            "is_selected": str(option.label) == str(problem.selected_option_label),
        }
        for option in problem.option_specs
    ]


def _selected_option_position(problem: ResolvedProblem) -> int:
    """Return the one-based visible position of the selected option."""

    labels = [str(option.label) for option in problem.option_specs]
    return int(labels.index(str(problem.selected_option_label)) + 1)


def _trace_payload(
    *,
    selected_branch: str,
    branch_probabilities: Mapping[str, float],
    problem: ResolvedProblem,
    rendered: RenderedScene,
    prompt_artifacts: Any,
    annotation_value: list[float],
    noise_meta: Mapping[str, Any],
    image_size: tuple[int, int],
) -> dict[str, Any]:
    """Build task-specific trace sections from the resolved option-selection problem."""

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
        annotation_keys=OPTION_ANNOTATION_KEYS,
        noise_meta=noise_meta,
        image_size=image_size,
        option_count=len(problem.option_specs),
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
            "selected_option_label": str(problem.selected_option_label),
            "selected_option_position": _selected_option_position(problem),
            "options": _option_trace(problem),
            "answer": str(problem.answer),
        }
    )
    trace_payload["witness_symbolic"] = dict(params)
    trace_payload["projected_annotation"] = projected_bbox_annotation(list(annotation_value))
    return trace_payload


@register_task
class GeometryVolumeEquivalenceConversionEqualVolumeOptionLabelTask:
    """Select the option solid that has equal volume to the source solid."""

    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation', 'matching')
    domain = DOMAIN
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one visual option-selection task after binding its query branch."""

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
            render_scene=render_option_scene,
            prompt_task_key=PROMPT_TASK_KEY,
            prompt_branch_key=str(selected_branch),
            annotation_keys=OPTION_ANNOTATION_KEYS,
            annotation_schema="bbox",
            answer=str(problem.answer),
            instance_seed=int(instance_seed),
            params=task_params,
            max_attempts=int(max_attempts),
            render_defaults=render_defaults,
            prompt_defaults=prompt_defaults,
            random_namespace=TASK_ID,
        )
        return TaskOutput(
            prompt=str(artifacts.prompt_artifacts.prompt),
            answer_gt=TypedValue(type="option_letter", value=str(problem.answer)),
            annotation_gt=TypedValue(type="bbox", value=list(artifacts.annotation_value)),
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
    "EQUAL_VOLUME_OPTION_QUERY_IDS",
    "GeometryVolumeEquivalenceConversionEqualVolumeOptionLabelTask",
    "OPTION_ANNOTATION_KEYS",
    "QUERY_ID_CONE_MATCHES_CYLINDER_OPTION",
    "QUERY_ID_CUBOID_MATCHES_CYLINDER_OPTION",
    "QUERY_ID_CYLINDER_MATCHES_CONE_OPTION",
    "SCENE_ID",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
    "TASK_ID_EQUAL_VOLUME_OPTION",
]
