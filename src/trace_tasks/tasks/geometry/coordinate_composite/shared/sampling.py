"""Sampling helpers for coordinate-composite candidate-point selections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence, Tuple

from trace_tasks.core.sampling import uniform_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.shared.fixed_query import geometry_selected_probability_map, select_task_query_id

from .relations import transform_object, transform_point
from .state import GraphPoint, SceneObject


@dataclass(frozen=True)
class CandidatePointLabelBinding:
    """Randomized labels and the single label selected by a point predicate."""

    labeled_points: Tuple[Tuple[str, GraphPoint], ...]
    selected_label: str
    selected_point_graph: GraphPoint
    label_probabilities: dict[str, float]


@dataclass(frozen=True)
class CandidatePointCase:
    """Scene-object layout, candidate graph points, and allowed transforms."""

    case_id: str
    objects: Tuple[SceneObject, ...]
    candidate_points: Tuple[GraphPoint, ...]
    transforms: Tuple[str, ...]


@dataclass(frozen=True)
class ResolvedCandidatePointProblem:
    """Resolved candidate-point selection produced from task-owned cases."""

    selection_key: str
    case: CandidatePointCase
    transform: str
    objects: Tuple[SceneObject, ...]
    selected_label: str
    selected_point_graph: GraphPoint
    labeled_points: Tuple[Tuple[str, GraphPoint], ...]
    selection_probabilities: dict[str, float]
    case_probabilities: dict[str, float]
    transform_probabilities: dict[str, float]
    label_probabilities: dict[str, float]


def bind_unique_candidate_point_label(
    *,
    rng: Any,
    option_labels: Sequence[str],
    candidate_points: Sequence[GraphPoint],
    objects: Tuple[SceneObject, ...],
    predicate: Callable[[GraphPoint, Tuple[SceneObject, ...]], bool],
) -> CandidatePointLabelBinding:
    """Shuffle labels and bind the one candidate point satisfying ``predicate``."""

    points = tuple((float(point[0]), float(point[1])) for point in candidate_points)
    matching_indices = tuple(index for index, point in enumerate(points) if predicate(point, objects))
    if len(matching_indices) != 1:
        raise RuntimeError(f"candidate-point predicate produced {len(matching_indices)} selected points")

    labels = [str(label) for label in option_labels]
    rng.shuffle(labels)
    selected_index = int(matching_indices[0])
    selected_label = str(labels[selected_index])
    return CandidatePointLabelBinding(
        labeled_points=tuple((str(label), point) for label, point in zip(labels, points)),
        selected_label=selected_label,
        selected_point_graph=points[selected_index],
        label_probabilities=geometry_selected_probability_map(tuple(str(label) for label in option_labels), selected=selected_label),
    )


def resolve_candidate_point_problem(
    *,
    instance_seed: int,
    selection_key: str,
    selection_probabilities: Mapping[str, float],
    request_params: Mapping[str, Any],
    cases_by_selection: Mapping[str, Tuple[CandidatePointCase, ...]],
    predicate_for_selection: Callable[[str], Callable[[GraphPoint, Tuple[SceneObject, ...]], bool]],
    option_labels: Sequence[str],
    random_namespace: str,
) -> ResolvedCandidatePointProblem:
    """Resolve case, transform, and label binding after public key selection."""

    cases = tuple(cases_by_selection[str(selection_key)])
    explicit_case = request_params.get("case_id")
    if explicit_case is not None:
        case_id = str(explicit_case)
        matching = tuple(case for case in cases if str(case.case_id) == case_id)
        if not matching:
            raise ValueError(f"case_id={case_id!r} is not valid for {selection_key}")
        case = matching[0]
        case_probabilities = geometry_selected_probability_map((item.case_id for item in cases), selected=case.case_id)
    else:
        case = uniform_choice(spawn_rng(int(instance_seed), f"{random_namespace}.case"), cases)
        case_probabilities = geometry_selected_probability_map(tuple(item.case_id for item in cases))

    explicit_transform = request_params.get("transform")
    if explicit_transform is not None:
        transform = str(explicit_transform)
        if transform not in set(case.transforms):
            raise ValueError(f"transform={transform!r} is not valid for {selection_key}")
        transform_probabilities = geometry_selected_probability_map(case.transforms, selected=transform)
    else:
        transform = str(uniform_choice(spawn_rng(int(instance_seed), f"{random_namespace}.transform"), case.transforms))
        transform_probabilities = geometry_selected_probability_map(case.transforms)

    objects = tuple(transform_object(obj, transform) for obj in case.objects)
    candidate_points = tuple(transform_point(point, transform) for point in case.candidate_points)
    label_binding = bind_unique_candidate_point_label(
        rng=spawn_rng(int(instance_seed), f"{random_namespace}.labels"),
        option_labels=option_labels,
        candidate_points=candidate_points,
        objects=objects,
        predicate=predicate_for_selection(str(selection_key)),
    )
    return ResolvedCandidatePointProblem(
        selection_key=str(selection_key),
        case=case,
        transform=str(transform),
        objects=objects,
        selected_label=str(label_binding.selected_label),
        selected_point_graph=label_binding.selected_point_graph,
        labeled_points=label_binding.labeled_points,
        selection_probabilities=dict(selection_probabilities),
        case_probabilities=dict(case_probabilities),
        transform_probabilities=dict(transform_probabilities),
        label_probabilities=dict(label_binding.label_probabilities),
    )


def resolve_candidate_point_selection(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    supported_keys: Sequence[str],
    default_key: str,
    public_identifier: str,
    cases_by_selection: Mapping[str, Tuple[CandidatePointCase, ...]],
    predicate_for_selection: Callable[[str], Callable[[GraphPoint, Tuple[SceneObject, ...]], bool]],
    option_labels: Sequence[str],
) -> ResolvedCandidatePointProblem:
    """Select a public-supplied semantic key and resolve the candidate point."""

    selected_key, key_probabilities, request_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=tuple(str(key) for key in supported_keys),
        default_query_id=str(default_key),
        task_id=str(public_identifier),
    )
    return resolve_candidate_point_problem(
        instance_seed=int(instance_seed),
        selection_key=str(selected_key),
        selection_probabilities=dict(key_probabilities),
        request_params=request_params,
        cases_by_selection=cases_by_selection,
        predicate_for_selection=predicate_for_selection,
        option_labels=option_labels,
        random_namespace=f"{public_identifier}.{selected_key}",
    )


__all__ = [
    "CandidatePointCase",
    "CandidatePointLabelBinding",
    "ResolvedCandidatePointProblem",
    "bind_unique_candidate_point_label",
    "resolve_candidate_point_problem",
    "resolve_candidate_point_selection",
]
