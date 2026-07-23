"""State containers shared by pixel-village task primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class PixelVillageCountBinding:
    """Bound answer/annotation fragments for a generated pixel-village count."""

    prompt_defaults: Mapping[str, Any]
    slots: Mapping[str, Any]
    answer: int
    annotation_value: Sequence[Sequence[float]]
    scene: Any
    entity_bboxes: Mapping[str, Sequence[float]]
    counted_entity_ids: Sequence[str]
    scene_relations: Mapping[str, Any]
    branch_params: Mapping[str, Any]
    render_map_extra: Mapping[str, Any]
    execution_trace: Mapping[str, Any]
    witness_symbolic: Mapping[str, Any]
    projected_annotation: Mapping[str, Any]


__all__ = ["PixelVillageCountBinding"]
