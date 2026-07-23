"""Relation and case-binding helpers for container volume transfer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .measurements import bind_sampling_metadata, support_from_cases
from .state import ResolvedProblem


@dataclass(frozen=True)
class ContainerVolumeTaskBinding:
    """Prompt and answer binding selected by a public task."""

    prompt_task_key: str
    annotation_keys: tuple[str, ...]
    answer_hint_key: str
    answer_type: str


@dataclass(frozen=True)
class ContainerVolumeQueryProgram:
    """A concrete case sampler and resolver for one semantic query branch."""

    selector: Callable[..., tuple[tuple[int, ...], dict[str, float]]]
    resolver: Callable[[Sequence[int]], ResolvedProblem]
    support_cases: tuple[Sequence[int], ...]
    support_field: str
    namespace_suffix: str


def resolve_container_volume_problem(
    *,
    program: ContainerVolumeQueryProgram,
    query_probabilities: Mapping[str, float],
    instance_seed: int,
    params: Mapping[str, Any],
    random_namespace: str,
) -> ResolvedProblem:
    """Select a case and bind sampling metadata for one task-owned query."""

    case, case_probabilities = program.selector(
        instance_seed=int(instance_seed),
        params=params,
        namespace=str(random_namespace),
    )
    return bind_sampling_metadata(
        program.resolver(case),
        query_probabilities=dict(query_probabilities),
        case_probabilities=dict(case_probabilities),
        answer_support_probabilities=support_from_cases(
            program.support_cases,
            program.resolver,
            program.support_field,
        ),
        params=dict(params),
    )


def container_measurement_fields(problem: ResolvedProblem) -> dict[str, Any]:
    """Return JSON-ready measurement fields shared by trace sections."""

    return {
        "source_shape": str(problem.source_shape),
        "target_shape": str(problem.target_shape),
        "source_base_area": int(problem.source_base_area),
        "source_height": int(problem.source_height),
        "source_volume": int(problem.source_volume),
        "target_base_area": int(problem.target_base_area),
        "target_length": int(problem.target_length),
        "target_width": int(problem.target_width),
        "target_height": int(problem.target_height),
        "target_volume": int(problem.target_volume),
        "fill_count": int(problem.fill_count),
        "pour_count": int(problem.pour_count),
        "resulting_height": float(problem.resulting_height),
    }


__all__ = [
    "ContainerVolumeQueryProgram",
    "ContainerVolumeTaskBinding",
    "container_measurement_fields",
    "resolve_container_volume_problem",
]
