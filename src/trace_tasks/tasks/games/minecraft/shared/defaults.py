"""Scene constants and fallback defaults for Minecraft-like games tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from trace_tasks.tasks.games.shared.visual_defaults import load_games_scene_noise_defaults


SCENE_ID = "minecraft"
DEFAULT_BRANCH_ID = "single"

STYLE_VARIANTS: Tuple[str, ...] = (
    "grass",
    "desert",
    "snow",
    "cave",
    "mesa",
)
RESOURCE_KINDS: Tuple[str, ...] = (
    "diamond_ore",
    "gold_ore",
)
ORE_KINDS: Tuple[str, ...] = ("gold_ore", "diamond_ore")

SAMPLE_KIND_TOP_RESOURCE = "top_resource"
SAMPLE_KIND_ROUTE_COST = "route_cost"
SAMPLE_KIND_HEIGHT_FILTER = "height_filter"

HEIGHT_CONDITION_EXACT = "exact"
HEIGHT_CONDITION_AT_LEAST = "at_least"
ROUTE_DISTRACTOR_MIN_TRACK_DISTANCE = 3


@dataclass(frozen=True)
class MinecraftDefaults:
    """Fallback axes and render defaults for visible block-world scenes."""

    grid_width_support: Tuple[int, ...] = (8, 9, 10)
    grid_depth_support: Tuple[int, ...] = (8, 9, 10)
    top_resource_answer_support: Tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    route_answer_support: Tuple[int, ...] = (0, 1, 2, 3, 4, 5)
    height_filter_answer_support: Tuple[int, ...] = (1, 2, 3, 4, 5)
    exact_target_height_support: Tuple[int, ...] = (2, 3, 4, 5)
    at_least_target_height_support: Tuple[int, ...] = (2, 3, 4)
    stack_height_support: Tuple[int, ...] = (1, 2, 3, 4, 5)
    route_grid_width_support: Tuple[int, ...] = (11,)
    route_grid_depth_support: Tuple[int, ...] = (8, 9, 10)
    canvas_width: int = 840
    canvas_height: int = 680
    tile_width_px: int = 58
    tile_height_px: int = 30
    cube_height_px: int = 30
    outline_width_px: int = 2
    player_marker_size_px: int = 28


DEFAULTS = MinecraftDefaults()
POST_IMAGE_NOISE_DEFAULTS = load_games_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)


def resource_prompt_name(kind: str) -> str:
    """Return the prompt-facing resource name for one block kind."""

    names = {
        "diamond_ore": "diamond ore",
        "gold_ore": "gold ore",
    }
    return names.get(str(kind), str(kind).replace("_", " "))


__all__ = [
    "DEFAULTS",
    "DEFAULT_BRANCH_ID",
    "HEIGHT_CONDITION_AT_LEAST",
    "HEIGHT_CONDITION_EXACT",
    "MinecraftDefaults",
    "ORE_KINDS",
    "POST_IMAGE_NOISE_DEFAULTS",
    "RESOURCE_KINDS",
    "ROUTE_DISTRACTOR_MIN_TRACK_DISTANCE",
    "SAMPLE_KIND_HEIGHT_FILTER",
    "SAMPLE_KIND_ROUTE_COST",
    "SAMPLE_KIND_TOP_RESOURCE",
    "SCENE_ID",
    "STYLE_VARIANTS",
    "resource_prompt_name",
]
