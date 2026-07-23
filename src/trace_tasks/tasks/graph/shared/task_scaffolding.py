"""Shared graph task scaffolding for deterministic support selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ....core.seed import hash64
from ...shared.deterministic_sampling import resolve_selection_index
from .task_support import graph_balanced_axis_count


@dataclass(frozen=True)
class GraphBalancedAxisSpec:
    """One already-resolved graph axis that should be decoupled from support sampling."""

    probabilities: Mapping[str, float]
    balance_flag_key: str
    explicit_keys: Sequence[str]
    weights_key: str


def graph_hashed_axis_selection_index(
    instance_seed: int,
    *,
    task_id: str,
    axis_name: str,
    selection_index: int,
    axis_values: Sequence[Any] = (),
) -> int:
    """Return a deterministic hashed support index for one task-local axis."""

    suffix = ":".join(str(value) for value in axis_values)
    namespace = f"{str(task_id)}:{str(axis_name)}"
    if suffix:
        namespace = f"{namespace}:{suffix}"
    return int(hash64(int(instance_seed), namespace, int(selection_index)))


def graph_decoupled_selection_index(
    instance_seed: int,
    *,
    params: Mapping[str, Any],
    task_id: str,
    namespace: str,
    gen_defaults: Mapping[str, Any],
    axis_specs: Sequence[GraphBalancedAxisSpec] = (),
) -> int:
    """Return a support index decoupled from balanced public/query axes."""

    selection_index = int(
        resolve_selection_index(
            params=params,
            instance_seed=int(instance_seed),
            namespace=f"{str(task_id)}:{str(namespace)}",
        )
    )
    divisor = 1
    for axis in axis_specs:
        divisor *= int(
            graph_balanced_axis_count(
                params=params,
                gen_defaults=gen_defaults,
                probabilities=axis.probabilities,
                balance_flag_key=str(axis.balance_flag_key),
                explicit_keys=tuple(str(key) for key in axis.explicit_keys),
                weights_key=str(axis.weights_key),
            )
        )
    if int(divisor) > 1:
        return int(selection_index // int(divisor))
    return int(selection_index)


__all__ = [
    "GraphBalancedAxisSpec",
    "graph_decoupled_selection_index",
    "graph_hashed_axis_selection_index",
]
