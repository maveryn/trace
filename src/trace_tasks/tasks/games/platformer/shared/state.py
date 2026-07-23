"""Passive state records for platformer game scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from .defaults import SUPPORTED_PLATFORMER_SCENE_VARIANTS, SUPPORTED_PLATFORMER_STYLE_VARIANTS


@dataclass(frozen=True)
class PlatformerPlatform:
    """One visible side-scroller platform."""

    platform_id: str
    label: str
    x_norm: float
    y_norm: float
    width_norm: float
    height_norm: float
    color_index: int


@dataclass(frozen=True)
class PlatformerHazard:
    """One visible side-scroller hazard."""

    hazard_id: str
    label: str
    kind: str
    x_norm: float
    y_norm: float
    width_norm: float
    height_norm: float
    color_index: int


@dataclass(frozen=True)
class PlatformerCollectible:
    """One visible collectible coin or bonus item."""

    collectible_id: str
    x_norm: float
    y_norm: float
    radius_norm: float
    on_path: bool
    color_index: int
    kind: str = "coin"
    score_value: int | None = None


@dataclass(frozen=True)
class PlatformerSample:
    """Generated platformer scene state."""

    mode: str
    scene_variant: str
    style_variant: str
    answer: str | int
    player_x_norm: float
    player_y_norm: float
    path_points_norm: Tuple[Tuple[float, float], ...]
    visible_path_fraction: float
    platforms: Tuple[PlatformerPlatform, ...]
    hazards: Tuple[PlatformerHazard, ...]
    collectibles: Tuple[PlatformerCollectible, ...]
    target_platform_id: str | None
    target_platform_label: str | None
    target_collectible_ids: Tuple[str, ...]
    annotation_entity_ids: Tuple[str, ...]
    construction_mode: str


def platform_entity_id(index: int) -> str:
    """Return a stable entity id for one platform."""

    return f"platform_{int(index)}"


def hazard_entity_id(index: int) -> str:
    """Return a stable entity id for one hazard."""

    return f"hazard_{int(index)}"


def collectible_entity_id(index: int) -> str:
    """Return a stable entity id for one collectible."""

    return f"collectible_{int(index)}"


def validate_platformer_sample(sample: PlatformerSample) -> None:
    """Validate shared structural invariants for one platformer sample."""

    platform_ids = [str(platform.platform_id) for platform in sample.platforms]
    platform_labels = [str(platform.label) for platform in sample.platforms if str(platform.label)]
    hazard_ids = [str(hazard.hazard_id) for hazard in sample.hazards]
    hazard_labels = [str(hazard.label) for hazard in sample.hazards if str(hazard.label)]
    collectible_ids = [str(collectible.collectible_id) for collectible in sample.collectibles]
    if len(platform_ids) != len(set(platform_ids)):
        raise ValueError("platform ids must be unique")
    if len(platform_labels) != len(set(platform_labels)):
        raise ValueError("platform labels must be unique")
    if len(hazard_ids) != len(set(hazard_ids)):
        raise ValueError("hazard ids must be unique")
    if len(hazard_labels) != len(set(hazard_labels)):
        raise ValueError("hazard labels must be unique")
    if len(collectible_ids) != len(set(collectible_ids)):
        raise ValueError("collectible ids must be unique")
    if str(sample.scene_variant) not in SUPPORTED_PLATFORMER_SCENE_VARIANTS:
        raise ValueError("unsupported platformer scene variant")
    if str(sample.style_variant) not in SUPPORTED_PLATFORMER_STYLE_VARIANTS:
        raise ValueError("unsupported platformer style variant")

    known_entities = set(platform_ids) | set(hazard_ids) | set(collectible_ids) | {"player"}
    if not set(sample.annotation_entity_ids) <= known_entities:
        raise ValueError("platformer annotation references unknown entities")
    if sample.target_platform_id is not None and str(sample.target_platform_id) not in platform_ids:
        raise ValueError("target platform id is not present in the scene")
    if sample.target_platform_label is not None and str(sample.target_platform_label) not in platform_labels:
        raise ValueError("target platform label is not present in the scene")
    if not set(str(value) for value in sample.target_collectible_ids) <= set(collectible_ids):
        raise ValueError("target collectible ids must be present in the scene")
    if len(sample.path_points_norm) < 2:
        raise ValueError("platformer sample requires a visible path")


__all__ = [
    "SUPPORTED_PLATFORMER_SCENE_VARIANTS",
    "SUPPORTED_PLATFORMER_STYLE_VARIANTS",
    "PlatformerCollectible",
    "PlatformerHazard",
    "PlatformerPlatform",
    "PlatformerSample",
    "collectible_entity_id",
    "hazard_entity_id",
    "platform_entity_id",
    "validate_platformer_sample",
]
