"""State records shared by environment illustration count tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class EnvironmentChoice:
    """Resolved semantic operands for one environment objective."""

    branch_index: int
    theme_id: str
    theme_probabilities: Dict[str, float]
    feature_type: str | None = None
    feature_type_probabilities: Dict[str, float] | None = None
    relation: str | None = None
    relation_probabilities: Dict[str, float] | None = None
    crossing_type: str | None = None
    crossing_type_probabilities: Dict[str, float] | None = None
    window_mode: str | None = None
    window_mode_probabilities: Dict[str, float] | None = None


@dataclass(frozen=True)
class BoundCountResult:
    """Task-bound answer, annotation, and trace fragments."""

    answer: int
    annotation_value: list[list[float]]
    render_map_extra: Dict[str, Any]
    scene_relations: Dict[str, Any]
    execution_extra: Dict[str, Any]
    witness_symbolic: Dict[str, Any]
    operand_params: Dict[str, Any]


__all__ = ["BoundCountResult", "EnvironmentChoice"]
