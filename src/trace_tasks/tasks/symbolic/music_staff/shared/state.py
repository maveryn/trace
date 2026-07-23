"""Passive state for symbolic music-staff scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

from .components import MusicSceneSpec


DOMAIN = "symbolic"
SCENE_ID = "music_staff"
SCENE_VARIANTS: Tuple[str, ...] = (
    "engraved_sheet",
    "exam_scan",
    "notebook_staff",
)


@dataclass(frozen=True)
class MusicStaffDataset:
    branch_key: str
    answer_type: str
    answer_value: int | str
    annotation_item_ids: Tuple[str, ...]
    spec: MusicSceneSpec
    scene_variant: str
    prompt_slots: dict[str, str]
    metadata: dict[str, Any]
    target_answer_support: Tuple[int | str, ...]
