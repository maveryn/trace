"""Passive state and constants for space-shooter game tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple


DOMAIN = "games"
SCENE_ID = "space_shooter"
SUPPORTED_SCENE_VARIANTS: Tuple[str, ...] = ("defense_wave",)
SUPPORTED_STYLE_VARIANTS: Tuple[str, ...] = (
    "neon",
    "deep_space",
    "vector",
    "amber",
    "terminal",
)
ENEMY_LABELS: Tuple[str, ...] = tuple(chr(ord("A") + index) for index in range(26))


@dataclass(frozen=True)
class SpaceEnemy:
    """One visible enemy ship."""

    enemy_id: str
    label: str
    lane: int
    y_slot: int
    dx_frac: float
    dy_px: float
    score_value: int | None = None


@dataclass(frozen=True)
class SpaceProjectile:
    """One visible projectile with an explicit side and direction."""

    projectile_id: str
    owner: str
    lane: int
    y_slot: int
    dx_frac: float
    dy_px: float


@dataclass(frozen=True)
class SpaceShooterSample:
    """Generated scene state plus task-owned answer and annotation ids."""

    lane_count: int
    scene_variant: str
    answer: int | str
    player_lane: int
    enemies: Tuple[SpaceEnemy, ...]
    projectiles: Tuple[SpaceProjectile, ...]
    safe_lane_indices: Tuple[int, ...]
    annotation_entity_ids: Tuple[str, ...]
    target_answer: int | None
    construction_mode: str
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class SceneAxes:
    """Resolved scene-level axes shared by all space-shooter objectives."""

    scene_variant: str
    style_variant: str
    lane_count: int
    enemy_count: int
    enemy_projectile_per_lane_support: Tuple[int, ...]
    scene_variant_probabilities: Dict[str, float]
    style_variant_probabilities: Dict[str, float]
    lane_count_probabilities: Dict[str, float]
    enemy_count_probabilities: Dict[str, float]


def lane_entity_id(lane: int) -> str:
    """Return the stable render/entity id for one bottom lane pad."""

    return f"lane_{int(lane)}"


def validate_basic_space_shooter_sample(sample: SpaceShooterSample) -> None:
    """Validate scene-wide invariants independent of the public objective."""

    lane_count = int(sample.lane_count)
    if lane_count <= 0:
        raise ValueError("space shooter lane_count must be positive")
    if not (0 <= int(sample.player_lane) < lane_count):
        raise ValueError("space shooter player_lane out of range")
    enemy_ids = [str(enemy.enemy_id) for enemy in sample.enemies]
    projectile_ids = [str(projectile.projectile_id) for projectile in sample.projectiles]
    if len(enemy_ids) != len(set(enemy_ids)):
        raise ValueError("space shooter enemy ids must be unique")
    if len(projectile_ids) != len(set(projectile_ids)):
        raise ValueError("space shooter projectile ids must be unique")
    if any(str(projectile.owner) not in {"enemy", "player"} for projectile in sample.projectiles):
        raise ValueError("space shooter projectile owner must be enemy or player")
    if len(set(str(enemy.label) for enemy in sample.enemies)) != len(sample.enemies):
        raise ValueError("space shooter enemy labels must be unique")
    known_entities = set(enemy_ids) | set(projectile_ids) | {
        lane_entity_id(lane) for lane in range(lane_count)
    }
    if not set(str(entity_id) for entity_id in sample.annotation_entity_ids) <= known_entities:
        raise ValueError("space shooter annotation references unknown entities")
    if not set(int(lane) for lane in sample.safe_lane_indices) <= set(range(lane_count)):
        raise ValueError("space shooter safe lanes out of range")
