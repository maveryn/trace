"""State records shared by object-cluster task objectives."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping

from trace_tasks.tasks.three_d.shared.object_scene import ObjectSceneRenderParams


@dataclass(frozen=True)
class ClusterSequenceItem:
    """One countable object before 3D placement and projection."""

    shape_type: str
    color_name: str
    matches_query: bool
    count_role: str


@dataclass(frozen=True)
class ClusterRequest:
    """Task-local objective plan consumed by the neutral scene lifecycle."""

    external_query: str
    prompt_query_key: str
    query_probabilities: Dict[str, float]
    scene_variant: str
    scene_probabilities: Dict[str, float]
    dataset: Dict[str, Any]
    count_probabilities: Dict[str, Any]
    prompt_slots: Dict[str, str]
    keyed_annotation: bool = False


BuildRequest = Callable[
    [
        int,
        Mapping[str, Any],
        Mapping[str, Any],
        Mapping[str, Any],
        ObjectSceneRenderParams,
    ],
    ClusterRequest,
]


@dataclass(frozen=True)
class PredicateTarget:
    """Internal predicate description independent of public task/query ids."""

    mode: str
    target_shape_type: str | None = None
    target_color_name: str | None = None
    target_shape_types: tuple[str, ...] = ()
    singleton_shape_types: tuple[str, ...] = ()
    target_object_name: str | None = None
    target_object_plural: str | None = None
    target_object_union_phrase: str | None = None
    target_color_names: tuple[str, ...] = ()
    left_operand_phrase: str | None = None
    right_operand_phrase: str | None = None
    arithmetic_operation: str | None = None
    left_count: int | None = None
    right_count: int | None = None
    target_property_phrase: str = "counted objects"
    extras: Mapping[str, Any] = field(default_factory=dict)


__all__ = [
    "BuildRequest",
    "ClusterRequest",
    "ClusterSequenceItem",
    "PredicateTarget",
]
