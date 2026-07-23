"""Canonical named object resources for Trace 3D scenes.

This module is the ownership point for object pools, display names, base
dimensions, and scene-role profiles used by the three_d domain. Scene modules
should select objects from these definitions instead of maintaining independent
object inventories.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, Mapping, Sequence, Tuple


OBJECT_SCENE_ID = "object_scene"
OBJECT_CLUSTER_SCENE_ID = "object_cluster"
ROOM_SCENE_ID = "room"
STREET_SCENE_ID = "street"
WAREHOUSE_SCENE_ID = "warehouse"
SURFACE_FIXTURE_SCENE_ID = "surface_fixture"

SURFACE_FIXTURE_OBJECT_TYPES: Tuple[str, ...] = (
    "wall_tile_panel",
    "perforated_panel",
    "slot_board",
    "compartment_tray",
    "vent_panel",
    "window_grid",
    "door_bank",
    "drawer_pull_panel",
    "brick_wall",
    "paver_floor",
    "locker_bank",
    "mailbox_bank",
    "server_rack",
    "control_panel",
    "solar_panel_array",
    "screw_plate",
    "hex_nut_plate",
    "washer_plate",
    "socket_bank",
    "hook_board",
    "indicator_light_panel",
    "bracket_panel",
    "u_bolt_plate",
    "pipe_rack",
)

SURFACE_FIXTURE_DISPLAY_NAME: Mapping[str, str] = {
    "wall_tile_panel": "tiled wall panel",
    "perforated_panel": "perforated metal panel",
    "slot_board": "slotted board",
    "compartment_tray": "compartment tray",
    "vent_panel": "vent panel",
    "window_grid": "window grid",
    "door_bank": "bank of small doors",
    "drawer_pull_panel": "drawer front panel",
    "brick_wall": "brick wall",
    "paver_floor": "paver floor",
    "locker_bank": "locker bank",
    "mailbox_bank": "mailbox bank",
    "server_rack": "server rack",
    "control_panel": "control panel",
    "solar_panel_array": "solar panel array",
    "screw_plate": "screw plate",
    "hex_nut_plate": "hex-nut plate",
    "washer_plate": "washer plate",
    "socket_bank": "socket bank",
    "hook_board": "hook board",
    "indicator_light_panel": "indicator light panel",
    "bracket_panel": "bracket panel",
    "u_bolt_plate": "U-bolt plate",
    "pipe_rack": "pipe rack",
}


@dataclass(frozen=True)
class ThreeDObjectProfile:
    """One domain-owned rendering profile for a canonical named object."""

    profile_id: str
    canonical_id: str
    source_scene: str
    role: str
    object_type: str
    display_name: str
    size_class: str
    renderer: str
    resource_kind: str = "standalone"
    support_required: bool = False
    base_object_type: str | None = None
    support_profile_id: str | None = None
    mounting: str | None = None
    dimensions_xyz: Tuple[float, ...] | None = None


def _canonical_id(display_name: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", str(display_name).strip().lower())
    return cleaned.strip("_") or "object"


OBJECT_SCENE_SMALL_SHAPE_TYPES: Tuple[str, ...] = (
    "sphere",
    "cube",
    "cylinder",
    "cone",
    "arrow",
    "sword",
    "shield",
    "diamond",
    "heart",
    "key",
    "crown",
    "anchor",
    "horseshoe",
    "hammer",
    "gear",
    "bell",
    "trophy",
    "open_book",
    "mushroom",
    "lantern",
    "candle",
    "goblet",
    "mail_envelope",
    "compass",
    "flask",
    "clock",
    "apple",
    "carrot",
    "fish",
    "leaf",
    "glove",
    "hat",
    "helmet",
    "cup",
    "bottle",
    "umbrella",
    "calculator",
    "dice",
    "kite",
    "cactus",
    "drum",
    "ruler",
    "remote_control",
    "plug",
    "torus",
    "pyramid",
    "star_prism",
    "half_cylinder",
)
OBJECT_CLUSTER_EXTRA_SHAPE_TYPES: Tuple[str, ...] = (
    "pen",
    "pencil",
    "card",
    "bookmark",
    "candy_disc",
    "cd",
    "berry",
    "button",
    "screw",
    "hex_nut",
    "plate",
    "fork",
    "spoon",
    "bowl",
    "basket",
    "puzzle_piece",
    "small_box",
    "mini_chair",
    "mini_table",
    "flower",
    "jar",
    "can",
    "lid",
    "tube",
    "clip",
    "socket",
    "chess_piece",
    "light_bulb",
    "egg",
    "paint_brush",
    "stick",
    "ticket",
    "marble",
    "bead",
    "bolt",
    "cushion",
    "stool",
    "tray",
    "coaster",
    "rose",
    "tomato",
    "hook",
    "tape_roll",
    "bag",
)
OBJECT_CLUSTER_SHAPE_TYPES: Tuple[str, ...] = OBJECT_SCENE_SMALL_SHAPE_TYPES + OBJECT_CLUSTER_EXTRA_SHAPE_TYPES
THREE_D_NAMED_OBJECT_SHAPE_TYPES: Tuple[str, ...] = tuple(
    shape
    for shape in (
        "arrow",
        "apple",
        "anchor",
        "sphere",
        "basket",
        "bell",
        "open_book",
        "bottle",
        "bowl",
        "button",
        "cactus",
        "calculator",
        "candle",
        "card",
        "carrot",
        "mini_chair",
        "chess_piece",
        "clock",
        "compass",
        "cone",
        "crown",
        "cube",
        "cup",
        "cylinder",
        "diamond",
        "dice",
        "mail_envelope",
        "fish",
        "flower",
        "glove",
        "half_cylinder",
        "hat",
        "heart",
        "helmet",
        "horseshoe",
        "jar",
        "key",
        "kite",
        "lantern",
        "leaf",
        "mushroom",
        "pencil",
        "plate",
        "plug",
        "puzzle_piece",
        "pyramid",
        "remote_control",
        "torus",
        "ruler",
        "screw",
        "shield",
        "star_prism",
        "stick",
        "stool",
        "sword",
        "mini_table",
        "tray",
        "trophy",
        "umbrella",
    )
    if shape in set(OBJECT_CLUSTER_SHAPE_TYPES)
)
OBJECT_SCENE_CONTEXT_SHAPE_TYPES: Tuple[str, ...] = (
    "arch",
    "table",
    "shelf",
    "open_box",
    "refrigerator",
    "washing_machine",
    "vending_machine",
    "trash_bin",
    "bench",
    "piano",
    "locker",
    "cabinet",
    "sofa",
    "barrel",
    "chair",
)
OBJECT_SCENE_SHAPE_TYPES: Tuple[str, ...] = OBJECT_SCENE_SMALL_SHAPE_TYPES + OBJECT_SCENE_CONTEXT_SHAPE_TYPES
OBJECT_SCENE_NAMED_CANDIDATE_SHAPE_TYPES: Tuple[str, ...] = tuple(
    shape for shape in THREE_D_NAMED_OBJECT_SHAPE_TYPES if shape in set(OBJECT_SCENE_SMALL_SHAPE_TYPES)
)
OBJECT_SCENE_NAME_BY_SHAPE_TYPE: Dict[str, str] = {
    "sphere": "ball",
    "cube": "cube",
    "cylinder": "cylinder",
    "cone": "cone",
    "arrow": "arrow",
    "sword": "sword",
    "shield": "shield",
    "diamond": "diamond",
    "heart": "heart",
    "key": "key",
    "crown": "crown",
    "anchor": "anchor",
    "horseshoe": "horseshoe",
    "hammer": "hammer",
    "gear": "gear",
    "bell": "bell",
    "trophy": "trophy",
    "open_book": "book",
    "mushroom": "mushroom",
    "lantern": "lantern",
    "candle": "candle",
    "goblet": "goblet",
    "mail_envelope": "envelope",
    "compass": "compass",
    "flask": "flask",
    "clock": "clock",
    "apple": "apple",
    "carrot": "carrot",
    "fish": "fish",
    "leaf": "leaf",
    "glove": "glove",
    "hat": "hat",
    "helmet": "helmet",
    "cup": "cup",
    "bottle": "bottle",
    "umbrella": "umbrella",
    "calculator": "calculator",
    "dice": "dice",
    "kite": "kite",
    "cactus": "cactus",
    "drum": "drum",
    "ruler": "ruler",
    "remote_control": "remote control",
    "plug": "plug",
    "torus": "ring",
    "pyramid": "pyramid",
    "wedge": "ramp",
    "star_prism": "star",
    "half_cylinder": "half cylinder",
    "arch": "arch",
    "table": "table",
    "shelf": "shelf",
    "open_box": "open box",
    "refrigerator": "refrigerator",
    "washing_machine": "washing machine",
    "vending_machine": "vending machine",
    "trash_bin": "trash bin",
    "bench": "bench",
    "piano": "piano",
    "locker": "locker",
    "cabinet": "cabinet",
    "sofa": "sofa",
    "barrel": "barrel",
    "chair": "chair",
}
OBJECT_CLUSTER_EXTRA_NAME_BY_SHAPE_TYPE: Dict[str, str] = {
    "pen": "pen",
    "pencil": "pencil",
    "card": "card",
    "bookmark": "bookmark",
    "candy_disc": "candy",
    "cd": "CD",
    "berry": "berry",
    "button": "button",
    "screw": "screw",
    "hex_nut": "hex nut",
    "plate": "plate",
    "fork": "fork",
    "spoon": "spoon",
    "bowl": "bowl",
    "basket": "basket",
    "puzzle_piece": "puzzle piece",
    "small_box": "small box",
    "mini_chair": "chair",
    "mini_table": "table",
    "flower": "flower",
    "jar": "jar",
    "can": "can",
    "lid": "lid",
    "tube": "tube",
    "clip": "clip",
    "socket": "socket",
    "chess_piece": "chess piece",
    "light_bulb": "light bulb",
    "egg": "egg",
    "paint_brush": "paint brush",
    "stick": "stick",
    "ticket": "ticket",
    "marble": "marble",
    "bead": "bead",
    "bolt": "bolt",
    "cushion": "cushion",
    "stool": "stool",
    "tray": "tray",
    "coaster": "coaster",
    "rose": "rose",
    "tomato": "tomato",
    "hook": "hook",
    "tape_roll": "tape roll",
    "bag": "bag",
}
OBJECT_CLUSTER_NAME_BY_SHAPE_TYPE: Dict[str, str] = {
    **OBJECT_SCENE_NAME_BY_SHAPE_TYPE,
    **OBJECT_CLUSTER_EXTRA_NAME_BY_SHAPE_TYPE,
}
OBJECT_SCENE_SMALL_DIMENSIONS: Dict[str, Tuple[float, float, float]] = {
    "sphere": (0.48, 0.48, 0.48),
    "cube": (0.52, 0.52, 0.52),
    "cylinder": (0.50, 0.50, 0.64),
    "cone": (0.54, 0.54, 0.68),
    "arrow": (0.52, 0.73, 0.24),
    "sword": (0.50, 0.73, 0.24),
    "shield": (0.64, 0.20, 0.74),
    "diamond": (0.64, 0.64, 0.72),
    "heart": (0.66, 0.20, 0.64),
    "key": (0.46, 0.72, 0.22),
    "crown": (0.68, 0.20, 0.66),
    "anchor": (0.64, 0.20, 0.92),
    "horseshoe": (0.64, 0.20, 0.78),
    "hammer": (0.56, 0.72, 0.24),
    "gear": (0.70, 0.70, 0.30),
    "bell": (0.66, 0.20, 0.74),
    "trophy": (0.70, 0.20, 0.82),
    "open_book": (0.72, 0.20, 0.58),
    "mushroom": (0.66, 0.22, 0.72),
    "lantern": (0.62, 0.20, 0.82),
    "candle": (0.36, 0.36, 0.88),
    "goblet": (0.58, 0.20, 0.78),
    "mail_envelope": (0.72, 0.18, 0.52),
    "compass": (0.62, 0.20, 0.62),
    "flask": (0.58, 0.20, 0.78),
    "clock": (0.58, 0.20, 0.68),
    "apple": (0.58, 0.58, 0.58),
    "carrot": (0.44, 0.22, 0.76),
    "fish": (0.76, 0.26, 0.48),
    "leaf": (0.68, 0.18, 0.42),
    "glove": (0.66, 0.18, 0.70),
    "hat": (0.76, 0.56, 0.52),
    "helmet": (0.74, 0.56, 0.62),
    "cup": (0.50, 0.50, 0.62),
    "bottle": (0.42, 0.42, 0.82),
    "umbrella": (0.72, 0.20, 0.82),
    "calculator": (0.60, 0.20, 0.78),
    "dice": (0.54, 0.54, 0.54),
    "kite": (0.64, 0.18, 0.74),
    "cactus": (0.58, 0.22, 0.84),
    "drum": (0.62, 0.62, 0.82),
    "ruler": (0.42, 0.70, 0.12),
    "remote_control": (0.46, 0.18, 0.78),
    "plug": (0.68, 0.22, 0.54),
    "torus": (0.62, 0.62, 0.30),
    "pyramid": (0.58, 0.58, 0.58),
    "wedge": (0.70, 0.52, 0.54),
    "star_prism": (0.72, 0.72, 0.42),
    "half_cylinder": (0.72, 0.52, 0.50),
}
OBJECT_CLUSTER_EXTRA_DIMENSIONS: Dict[str, Tuple[float, float, float]] = {
    "pen": (0.30, 0.88, 0.16),
    "pencil": (0.32, 0.90, 0.18),
    "card": (0.66, 0.88, 0.06),
    "bookmark": (0.40, 1.02, 0.06),
    "candy_disc": (0.42, 0.42, 0.12),
    "cd": (0.54, 0.54, 0.08),
    "berry": (0.34, 0.34, 0.34),
    "button": (0.42, 0.42, 0.10),
    "screw": (0.30, 0.98, 0.15),
    "hex_nut": (0.48, 0.48, 0.16),
    "plate": (0.58, 0.58, 0.10),
    "fork": (0.48, 0.82, 0.12),
    "spoon": (0.52, 0.84, 0.14),
    "bowl": (0.52, 0.52, 0.26),
    "basket": (0.64, 0.58, 0.38),
    "puzzle_piece": (0.66, 0.66, 0.10),
    "small_box": (0.48, 0.42, 0.32),
    "mini_chair": (0.66, 0.60, 0.72),
    "mini_table": (0.72, 0.62, 0.46),
    "flower": (0.58, 0.58, 0.68),
    "jar": (0.46, 0.46, 0.60),
    "can": (0.44, 0.44, 0.56),
    "lid": (0.46, 0.46, 0.10),
    "tube": (0.30, 0.88, 0.18),
    "clip": (0.48, 0.62, 0.18),
    "socket": (0.62, 0.52, 0.34),
    "chess_piece": (0.58, 0.58, 0.78),
    "light_bulb": (0.54, 0.54, 0.70),
    "egg": (0.50, 0.50, 0.58),
    "paint_brush": (0.62, 0.96, 0.22),
    "stick": (0.16, 1.18, 0.12),
    "ticket": (0.64, 0.86, 0.06),
    "marble": (0.34, 0.34, 0.34),
    "bead": (0.36, 0.36, 0.18),
    "bolt": (0.36, 0.82, 0.20),
    "cushion": (0.62, 0.58, 0.16),
    "stool": (0.60, 0.56, 0.52),
    "tray": (0.86, 0.66, 0.18),
    "coaster": (0.46, 0.46, 0.08),
    "rose": (0.54, 0.54, 0.68),
    "tomato": (0.46, 0.46, 0.42),
    "hook": (0.46, 0.62, 0.42),
    "tape_roll": (0.58, 0.58, 0.18),
    "bag": (0.58, 0.46, 0.62),
}
OBJECT_CLUSTER_DIMENSIONS: Dict[str, Tuple[float, float, float]] = {
    **OBJECT_SCENE_SMALL_DIMENSIONS,
    **OBJECT_CLUSTER_EXTRA_DIMENSIONS,
}
OBJECT_SCENE_CONTEXT_DIMENSIONS: Dict[str, Tuple[float, float, float]] = {
    "arch": (1.92, 0.66, 1.84),
    "table": (2.06, 1.42, 1.10),
    "shelf": (1.78, 0.68, 1.96),
    "open_box": (1.62, 1.22, 0.80),
    "refrigerator": (1.18, 0.82, 2.08),
    "washing_machine": (1.18, 0.82, 1.10),
    "vending_machine": (1.24, 0.86, 2.12),
    "trash_bin": (1.02, 0.88, 1.28),
    "bench": (2.42, 0.86, 1.08),
    "piano": (2.34, 1.46, 1.18),
    "locker": (1.14, 0.72, 2.04),
    "cabinet": (1.42, 0.72, 1.92),
    "sofa": (2.12, 1.12, 0.92),
    "barrel": (1.00, 1.00, 1.90),
    "chair": (1.10, 1.02, 1.54),
}
SPATIAL_REFERENCE_NEAREST_REFERENCE_SHAPE_TYPES: Tuple[str, ...] = (
    "sphere",
    "cube",
    "cylinder",
    "cone",
    "torus",
    "pyramid",
    "arch",
    "table",
    "shelf",
    "open_box",
    "refrigerator",
    "washing_machine",
    "vending_machine",
    "trash_bin",
    "bench",
    "piano",
    "locker",
    "cabinet",
    "sofa",
    "barrel",
    "chair",
)
SPATIAL_OBJECT_RELATION_ON_TOP_PROP_TYPES: Tuple[str, ...] = ("table", "shelf")
SPATIAL_OBJECT_RELATION_UNDER_PROP_TYPES: Tuple[str, ...] = ("arch", "table")
SPATIAL_OBJECT_RELATION_INSIDE_PROP_TYPES: Tuple[str, ...] = ("open_box",)
SPATIAL_OCCLUSION_REFERENCE_SHAPE_TYPES: Tuple[str, ...] = (
    "arch",
    "table",
    "shelf",
    "refrigerator",
    "washing_machine",
    "vending_machine",
    "trash_bin",
    "bench",
    "piano",
    "locker",
    "cabinet",
    "sofa",
    "barrel",
    "chair",
)
SPATIAL_BETWEEN_REFERENCE_SHAPE_TYPES: Tuple[str, ...] = (
    "arch",
    "table",
    "shelf",
    "open_box",
    "refrigerator",
    "washing_machine",
    "vending_machine",
    "trash_bin",
    "bench",
    "piano",
    "locker",
    "cabinet",
    "sofa",
    "barrel",
    "chair",
)
SPATIAL_HEIGHT_SUPPORT_PLACEMENTS: Tuple[Tuple[str, str | None, Tuple[float, float]], ...] = (
    ("floor_spot", None, (-2.58, -2.42)),
    ("lowest_platform", "platform", (2.46, -2.18)),
    ("low_platform", "platform", (-0.74, -1.62)),
    ("mid_platform", "platform", (2.34, 0.16)),
    ("upper_platform", "platform", (1.46, 2.24)),
    ("high_platform", "platform", (-1.44, 2.20)),
)
SPATIAL_HEIGHT_FLOOR_CANDIDATE_SHAPE_TYPES: Tuple[str, ...] = (
    "sphere",
    "cube",
    "cylinder",
    "cone",
    "pyramid",
    "torus",
    "dice",
    "apple",
    "hat",
    "helmet",
    "mushroom",
    "drum",
)
SPATIAL_HEIGHT_ELEVATED_CANDIDATE_SHAPE_TYPES: Tuple[str, ...] = (
    "sphere",
    "cube",
    "cylinder",
    "cone",
    "pyramid",
    "torus",
    "dice",
    "cup",
    "bottle",
    "drum",
    "bell",
    "trophy",
    "lantern",
    "candle",
    "goblet",
    "flask",
    "clock",
    "apple",
    "cactus",
    "hat",
    "helmet",
    "mushroom",
)
# Height-extremum candidates need reliable support contact and unique object-name
# descriptors. Keep this pool narrower than the general object_cluster pool.
SPATIAL_HEIGHT_SAFE_CANDIDATE_SHAPE_TYPES: Tuple[str, ...] = (
    "sphere",
    "cube",
    "cylinder",
    "cone",
    "pyramid",
    "torus",
    "dice",
    "apple",
    "helmet",
    "mushroom",
    "cup",
    "trophy",
)


ROOM_QUERY_TARGET_TYPES: Tuple[str, ...] = (
    "tv",
    "clock",
    "picture_frame",
    "mirror",
    "wall_shelf",
    "wall_fan",
    "air_conditioner",
    "hanging_coat",
)
ROOM_OBJECT_PROMPT_NAMES: Dict[str, Tuple[str, str]] = {
    "tv": ("TV", "TVs"),
    "clock": ("clock", "clocks"),
    "picture_frame": ("picture frame", "picture frames"),
    "mirror": ("mirror", "mirrors"),
    "wall_shelf": ("shelf", "shelves"),
    "wall_fan": ("fan", "fans"),
    "air_conditioner": ("air conditioner", "air conditioners"),
    "hanging_coat": ("coat", "coats"),
    "poster": ("poster", "posters"),
    "wall_lamp": ("wall lamp", "wall lamps"),
    "speaker": ("speaker", "speakers"),
    "wall_cabinet": ("wall cabinet", "wall cabinets"),
}
ROOM_EXTRA_WALL_TYPES: Tuple[str, ...] = ("poster", "wall_lamp", "speaker", "wall_cabinet")
ROOM_FLOOR_DISTRACTOR_TYPES: Tuple[str, ...] = (
    "tv",
    "clock",
    "picture_frame",
    "mirror",
    "wall_shelf",
    "wall_fan",
    "air_conditioner",
    "hanging_coat",
)
ROOM_SURFACE_DISTRACTOR_TYPES: Tuple[str, ...] = ("tv", "clock", "picture_frame")
ROOM_SURFACE_PROP_SHAPES_BY_SCENE: Dict[str, Tuple[str, ...]] = {
    "living_room": ("media_console", "side_table", "desk"),
    "office_room": ("desk", "side_table", "media_console"),
    "studio_room": ("bed", "desk", "side_table"),
}
ROOM_SURFACE_PROP_TYPES: Tuple[str, ...] = ("media_console", "side_table", "desk", "bed")
ROOM_FLOOR_PROP_SHAPES: Tuple[str, ...] = (
    "sofa",
    "armchair",
    "media_console",
    "side_table",
    "desk",
    "bed",
    "plant",
    "floor_lamp",
    "box",
    "ball",
)
ROOM_FRONT_FLOOR_PROP_SHAPES: Tuple[str, ...] = ("armchair", "plant", "floor_lamp", "box", "side_table")
ROOM_CAMERA_DISTANCE_CANDIDATE_WALL_OBJECT_TYPES: Tuple[str, ...] = (
    "tv",
    "clock",
    "picture_frame",
    "mirror",
    "wall_fan",
    "air_conditioner",
    "hanging_coat",
)
ROOM_CAMERA_DISTANCE_CONTEXT_WALL_OBJECT_TYPES: Tuple[str, ...] = ("poster", "wall_lamp", "speaker", "wall_cabinet")
ROOM_SAME_WALL_REFERENCE_WALL_OBJECT_TYPES: Tuple[str, ...] = (
    "tv",
    "clock",
    "mirror",
    "wall_fan",
    "air_conditioner",
    "hanging_coat",
)
ROOM_SIDE_RELATION_REFERENCE_OBJECT_TYPE = "tv"
ROOM_WALL_BASE_DIMENSIONS: Dict[str, Tuple[float, float]] = {
    "tv": (1.16, 0.66),
    "clock": (0.76, 0.76),
    "picture_frame": (0.72, 0.58),
    "mirror": (0.76, 0.98),
    "wall_shelf": (1.12, 0.24),
    "wall_fan": (1.04, 1.04),
    "air_conditioner": (1.08, 0.48),
    "hanging_coat": (0.90, 1.12),
    "poster": (0.64, 0.86),
    "wall_lamp": (0.62, 0.78),
    "speaker": (0.56, 0.78),
    "wall_cabinet": (0.78, 0.68),
}
ROOM_SURFACE_DISTRACTOR_SPECS: Dict[str, Dict[str, object]] = {
    "tv": {"prompt_name": "tabletop TV", "dimensions_xyz": (0.66, 0.10, 0.38), "color_role": "tv_floor"},
    "clock": {"prompt_name": "tabletop clock", "dimensions_xyz": (0.42, 0.16, 0.42), "color_role": "clock_floor"},
    "picture_frame": {
        "prompt_name": "tabletop picture frame",
        "dimensions_xyz": (0.34, 0.06, 0.34),
        "color_role": "picture_frame_floor",
    },
}
ROOM_FLOOR_DISTRACTOR_SPECS: Dict[str, Dict[str, object]] = {
    "tv": {"prompt_name": "floor-standing TV", "dimensions_xyz": (0.95, 0.16, 0.58), "color_role": "tv_floor"},
    "clock": {"prompt_name": "floor clock", "dimensions_xyz": (0.48, 0.24, 0.98), "color_role": "clock_floor"},
    "picture_frame": {
        "prompt_name": "leaning picture frame",
        "dimensions_xyz": (0.55, 0.08, 0.58),
        "color_role": "picture_frame_floor",
    },
    "mirror": {"prompt_name": "standing mirror", "dimensions_xyz": (0.58, 0.10, 0.96), "color_role": "mirror_floor"},
    "wall_fan": {"prompt_name": "floor fan", "dimensions_xyz": (0.66, 0.20, 0.82), "color_role": "fan_floor"},
    "air_conditioner": {
        "prompt_name": "portable air conditioner",
        "dimensions_xyz": (0.46, 0.26, 0.76),
        "color_role": "ac_floor",
    },
    "hanging_coat": {"prompt_name": "coat on stand", "dimensions_xyz": (0.62, 0.18, 1.02), "color_role": "coat_floor"},
    "wall_shelf": {"prompt_name": "standing shelf", "dimensions_xyz": (0.70, 0.34, 1.05), "color_role": "shelf_floor"},
}
ROOM_FLOOR_PROP_SPECS: Dict[str, Dict[str, object]] = {
    "sofa": {"object_type": "sofa", "prompt_name": "sofa", "dimensions_xyz": (1.58, 0.74, 0.62), "color_role": "sofa"},
    "armchair": {
        "object_type": "armchair",
        "prompt_name": "armchair",
        "dimensions_xyz": (0.82, 0.72, 0.70),
        "color_role": "armchair",
    },
    "side_table": {
        "object_type": "side_table",
        "prompt_name": "side table",
        "dimensions_xyz": (0.58, 0.48, 0.52),
        "color_role": "wood",
        "shape_type": "table",
    },
    "desk": {
        "object_type": "desk",
        "prompt_name": "desk",
        "dimensions_xyz": (1.36, 0.70, 0.74),
        "color_role": "wood",
        "shape_type": "table",
    },
    "media_console": {
        "object_type": "media_console",
        "prompt_name": "media console",
        "dimensions_xyz": (1.34, 0.38, 0.46),
        "color_role": "wood",
    },
    "bed": {"object_type": "bed", "prompt_name": "bed", "dimensions_xyz": (1.62, 1.04, 0.62), "color_role": "bed"},
    "plant": {
        "object_type": "plant",
        "prompt_name": "plant",
        "dimensions_xyz": (0.45, 0.45, 0.72),
        "color_role": "plant",
        "shape_type": "cylinder",
    },
    "floor_lamp": {
        "object_type": "floor_lamp",
        "prompt_name": "floor lamp",
        "dimensions_xyz": (0.36, 0.36, 1.42),
        "color_role": "lamp",
        "shape_type": "cylinder",
    },
    "ball": {
        "object_type": "ball",
        "prompt_name": "ball",
        "dimensions_xyz": (0.42, 0.42, 0.42),
        "color_role": "toy",
        "shape_type": "sphere",
    },
    "box": {"object_type": "box", "prompt_name": "box", "dimensions_xyz": (0.52, 0.48, 0.46), "color_role": "box"},
}


STREET_OBJECT_TYPES: Tuple[str, ...] = (
    "car",
    "taxi",
    "van",
    "bus",
    "delivery_truck",
    "pickup_truck",
    "scooter",
    "motorcycle",
    "bicycle",
    "traffic_cone",
    "fire_hydrant",
    "trash_bin",
    "mailbox",
    "construction_barrier",
    "road_barrel",
)
STREET_VEHICLE_OBJECT_TYPES = {"car", "taxi", "van", "bus", "delivery_truck", "pickup_truck"}
STREET_CONTEXT_OBJECT_TYPES: Tuple[str, ...] = (
    "building",
    "store",
    "office_building",
    "tree",
    "shrub",
    "traffic_light",
    "street_sign",
    "bench",
)
STREET_SAME_ROAD_REFERENCE_OBJECT_TYPES: Tuple[str, ...] = (
    "car",
    "taxi",
    "van",
    "bus",
    "pickup_truck",
    "scooter",
    "motorcycle",
    "bicycle",
    "fire_hydrant",
    "mailbox",
    "trash_bin",
)
STREET_LANE_AHEAD_REFERENCE_OBJECT_TYPE = "car"
STREET_OBJECT_NAMES: Dict[str, str] = {
    "car": "car",
    "taxi": "taxi",
    "van": "van",
    "bus": "bus",
    "delivery_truck": "delivery truck",
    "pickup_truck": "pickup truck",
    "scooter": "scooter",
    "motorcycle": "motorcycle",
    "male_pedestrian": "pedestrian",
    "female_pedestrian": "pedestrian",
    "pedestrian": "pedestrian",
    "bicycle": "bicycle",
    "traffic_cone": "traffic cone",
    "fire_hydrant": "fire hydrant",
    "trash_bin": "trash bin",
    "mailbox": "mailbox",
    "construction_barrier": "construction barrier",
    "road_barrel": "road barrel",
    "building": "building",
    "store": "store",
    "office_building": "office building",
    "tree": "tree",
    "shrub": "shrub",
    "traffic_light": "traffic light",
    "street_sign": "street sign",
    "bench": "bench",
}
STREET_OBJECT_COLORS: Dict[str, Tuple[int, int, int]] = {
    "car": (65, 121, 205),
    "taxi": (231, 181, 55),
    "van": (112, 131, 151),
    "bus": (218, 111, 59),
    "delivery_truck": (204, 83, 77),
    "pickup_truck": (74, 151, 111),
    "scooter": (150, 86, 177),
    "motorcycle": (45, 65, 86),
    "male_pedestrian": (52, 135, 160),
    "female_pedestrian": (91, 141, 166),
    "pedestrian": (52, 135, 160),
    "bicycle": (70, 83, 107),
    "traffic_cone": (225, 121, 45),
    "fire_hydrant": (218, 58, 49),
    "trash_bin": (79, 121, 93),
    "mailbox": (42, 104, 164),
    "construction_barrier": (229, 131, 48),
    "road_barrel": (223, 102, 45),
    "building": (150, 142, 131),
    "store": (171, 138, 96),
    "office_building": (108, 145, 166),
    "tree": (77, 132, 76),
    "shrub": (70, 145, 84),
    "traffic_light": (63, 70, 78),
    "street_sign": (75, 132, 94),
    "bench": (142, 106, 78),
}
STREET_OBJECT_BASE_DIMENSIONS: Dict[str, Tuple[float, float, float]] = {
    "car": (1.14, 0.52, 0.48),
    "taxi": (1.14, 0.52, 0.50),
    "van": (1.28, 0.60, 0.74),
    "bus": (1.74, 0.66, 0.82),
    "delivery_truck": (1.54, 0.66, 0.80),
    "pickup_truck": (1.32, 0.58, 0.60),
    "scooter": (0.72, 0.28, 0.62),
    "motorcycle": (0.94, 0.36, 0.62),
    "male_pedestrian": (0.40, 0.32, 0.92),
    "female_pedestrian": (0.40, 0.32, 0.92),
    "pedestrian": (0.40, 0.32, 0.92),
    "bicycle": (0.78, 0.22, 0.48),
    "traffic_cone": (0.30, 0.30, 0.50),
    "fire_hydrant": (0.26, 0.26, 0.86),
    "trash_bin": (0.46, 0.46, 0.62),
    "mailbox": (0.64, 0.40, 0.74),
    "construction_barrier": (0.84, 0.22, 0.58),
    "road_barrel": (0.38, 0.38, 0.66),
    "building": (1.24, 1.16, 1.50),
    "store": (1.78, 1.14, 1.02),
    "office_building": (1.24, 1.10, 1.82),
    "tree": (0.62, 0.62, 1.18),
    "shrub": (0.74, 0.58, 0.42),
    "traffic_light": (0.18, 0.18, 1.78),
    "street_sign": (0.18, 0.18, 1.04),
    "bench": (1.24, 0.46, 0.62),
}
STREET_RADIAL_OBJECT_TYPES = {
    "pedestrian",
    "male_pedestrian",
    "female_pedestrian",
    "traffic_cone",
    "fire_hydrant",
    "trash_bin",
    "road_barrel",
    "shrub",
}
BUILDING_STYLES: Tuple[str, ...] = (
    "office_glass",
    "apartment_brick",
    "glass_tower",
    "retail_corner",
    "cafe_shop",
    "market_shop",
    "bookstore_shop",
    "concrete_midrise",
    "stucco_walkup",
)
BUILDING_STYLE_POOLS: Dict[str, Tuple[str, ...]] = {
    "downtown_intersection": (
        "office_glass",
        "glass_tower",
        "apartment_brick",
        "concrete_midrise",
        "retail_corner",
        "cafe_shop",
        "market_shop",
    ),
    "neighborhood_intersection": (
        "apartment_brick",
        "cafe_shop",
        "market_shop",
        "bookstore_shop",
        "stucco_walkup",
        "retail_corner",
        "concrete_midrise",
        "office_glass",
    ),
    "transit_intersection": (
        "office_glass",
        "concrete_midrise",
        "retail_corner",
        "cafe_shop",
        "market_shop",
        "apartment_brick",
        "glass_tower",
    ),
}
BUILDING_STYLE_DISPLAY_NAMES: Dict[str, str] = {
    "office_glass": "glass office building",
    "apartment_brick": "brick apartment building",
    "glass_tower": "glass tower",
    "retail_corner": "corner shop building",
    "cafe_shop": "cafe storefront",
    "market_shop": "market storefront",
    "bookstore_shop": "bookstore storefront",
    "concrete_midrise": "concrete midrise building",
    "stucco_walkup": "stucco walkup building",
}
BUILDING_STYLE_BASE_COLORS: Dict[str, Tuple[int, int, int]] = {
    "office_glass": (108, 145, 166),
    "apartment_brick": (158, 94, 76),
    "glass_tower": (92, 139, 184),
    "retail_corner": (184, 150, 106),
    "cafe_shop": (185, 130, 103),
    "market_shop": (158, 166, 107),
    "bookstore_shop": (128, 120, 158),
    "concrete_midrise": (142, 149, 154),
    "stucco_walkup": (190, 168, 137),
}
BUILDING_STYLE_DIMENSION_FACTORS: Dict[str, Tuple[float, float, float]] = {
    "office_glass": (1.10, 0.96, 1.06),
    "apartment_brick": (1.02, 1.06, 1.02),
    "glass_tower": (0.84, 0.88, 1.24),
    "retail_corner": (1.28, 1.08, 0.72),
    "cafe_shop": (1.52, 1.10, 0.66),
    "market_shop": (1.64, 1.14, 0.72),
    "bookstore_shop": (1.42, 1.08, 0.70),
    "concrete_midrise": (1.08, 0.98, 0.94),
    "stucco_walkup": (1.04, 1.02, 0.86),
}


WAREHOUSE_OBJECT_TYPES: Tuple[str, ...] = (
    "crate_stack",
    "pallet_load",
    "barrel",
    "traffic_cone",
    "floor_sign",
    "tool_cart",
    "pallet_jack",
    "forklift",
    "box_stack",
    "tire_stack",
    "storage_bin",
    "ladder",
    "workbench",
    "rolling_bin",
    "trash_can",
    "warning_bollard",
    "wrapped_bundle",
    "fire_extinguisher",
    "stacked_pipes",
)
WAREHOUSE_SMALL_OBJECT_CANDIDATE_TYPES: Tuple[str, ...] = (
    "crate_stack",
    "barrel",
    "traffic_cone",
    "floor_sign",
    "box_stack",
    "tire_stack",
    "storage_bin",
    "rolling_bin",
    "trash_can",
    "warning_bollard",
    "wrapped_bundle",
    "fire_extinguisher",
)
WAREHOUSE_CONTEXT_OBJECT_TYPES: Tuple[str, ...] = (
    "shelf_rack",
    "charging_dock",
    "conveyor",
    "pallet",
    "crate_stack",
    "barrel",
    "storage_bin",
    "tool_cart",
    "traffic_cone",
    "workbench",
    "rolling_bin",
    "trash_can",
    "warning_bollard",
    "wrapped_bundle",
    "fire_extinguisher",
    "stacked_pipes",
)
WAREHOUSE_OBJECT_NAMES: Dict[str, str] = {
    "warehouse_robot": "robot",
    "shelf_rack": "shelf rack",
    "crate_stack": "crate stack",
    "pallet_load": "loaded pallet",
    "barrel": "barrel",
    "traffic_cone": "traffic cone",
    "floor_sign": "floor sign",
    "tool_cart": "tool cart",
    "pallet_jack": "pallet jack",
    "forklift": "forklift",
    "box_stack": "box stack",
    "tire_stack": "tire stack",
    "storage_bin": "storage bin",
    "ladder": "ladder",
    "charging_dock": "charging dock",
    "conveyor": "conveyor",
    "pallet": "pallet",
    "workbench": "workbench",
    "rolling_bin": "rolling bin",
    "trash_can": "trash can",
    "warning_bollard": "warning bollard",
    "wrapped_bundle": "wrapped bundle",
    "fire_extinguisher": "fire extinguisher",
    "stacked_pipes": "stacked pipes",
}
WAREHOUSE_OBJECT_COLORS: Dict[str, Tuple[int, int, int]] = {
    "warehouse_robot": (72, 111, 182),
    "shelf_rack": (124, 130, 140),
    "crate_stack": (178, 111, 62),
    "pallet_load": (191, 137, 74),
    "barrel": (63, 126, 171),
    "traffic_cone": (224, 113, 40),
    "floor_sign": (230, 188, 62),
    "tool_cart": (77, 148, 129),
    "pallet_jack": (196, 76, 64),
    "forklift": (225, 169, 52),
    "box_stack": (192, 154, 93),
    "tire_stack": (55, 61, 68),
    "storage_bin": (83, 142, 91),
    "ladder": (115, 128, 142),
    "charging_dock": (80, 92, 105),
    "conveyor": (99, 112, 128),
    "pallet": (151, 109, 70),
    "workbench": (96, 112, 118),
    "rolling_bin": (68, 137, 152),
    "trash_can": (83, 101, 96),
    "warning_bollard": (222, 184, 54),
    "wrapped_bundle": (187, 174, 149),
    "fire_extinguisher": (204, 48, 43),
    "stacked_pipes": (119, 131, 142),
}
WAREHOUSE_OBJECT_BASE_DIMENSIONS: Dict[str, Tuple[float, float, float]] = {
    "warehouse_robot": (1.04, 0.78, 0.88),
    "shelf_rack": (2.78, 0.42, 1.56),
    "crate_stack": (0.92, 0.76, 0.90),
    "pallet_load": (1.08, 0.82, 0.82),
    "barrel": (0.40, 0.40, 1.30),
    "traffic_cone": (0.42, 0.42, 0.72),
    "floor_sign": (0.62, 0.28, 0.68),
    "tool_cart": (0.90, 0.52, 0.80),
    "pallet_jack": (1.12, 0.48, 0.46),
    "forklift": (1.48, 0.78, 1.08),
    "box_stack": (0.96, 0.72, 0.84),
    "tire_stack": (0.56, 0.56, 0.56),
    "storage_bin": (0.62, 0.58, 0.64),
    "ladder": (0.58, 1.05, 1.12),
    "charging_dock": (1.00, 0.42, 0.58),
    "conveyor": (1.84, 0.62, 0.56),
    "pallet": (1.08, 0.76, 0.24),
    "workbench": (1.06, 0.58, 0.68),
    "rolling_bin": (0.78, 0.66, 0.78),
    "trash_can": (0.46, 0.46, 0.88),
    "warning_bollard": (0.30, 0.30, 0.94),
    "wrapped_bundle": (1.02, 0.74, 0.66),
    "fire_extinguisher": (0.26, 0.26, 0.86),
    "stacked_pipes": (1.30, 0.56, 0.52),
}
WAREHOUSE_RADIAL_OBJECT_TYPES = {
    "barrel",
    "traffic_cone",
    "tire_stack",
    "storage_bin",
    "trash_can",
    "warning_bollard",
    "fire_extinguisher",
}
WAREHOUSE_NEAREST_REFERENCE_OBJECT_TYPE = "red_sphere"
WAREHOUSE_NEAREST_REFERENCE_OBJECT_NAME = "red sphere"
WAREHOUSE_NEAREST_REFERENCE_OBJECT_RGB: Tuple[int, int, int] = (214, 50, 50)
WAREHOUSE_NEAREST_REFERENCE_OBJECT_DIMENSIONS: Tuple[float, float, float] = (0.52, 0.52, 0.52)
WAREHOUSE_NEAREST_OBJECT_CANDIDATE_TYPES: Tuple[str, ...] = tuple(
    object_type
    for object_type in WAREHOUSE_SMALL_OBJECT_CANDIDATE_TYPES
    if object_type not in {"floor_sign", "warning_bollard", "fire_extinguisher"}
)
WAREHOUSE_ROBOT_HEADINGS: Tuple[str, ...] = ("east", "north", "west", "south")
WAREHOUSE_ROBOT_DESIGNS: Tuple[str, ...] = ("low_cart", "sensor_tower", "stacker_bot")
WAREHOUSE_SHELF_RACK_STYLES: Tuple[str, ...] = ("open_frame", "loaded_bins", "mixed_crates", "tall_sparse", "heavy_low")
WAREHOUSE_ROBOT_BASE_COLORS: Tuple[Tuple[int, int, int], ...] = (
    (55, 112, 190),
    (62, 145, 132),
    (205, 128, 48),
    (101, 98, 171),
    (74, 91, 112),
    (182, 74, 66),
)
WAREHOUSE_ROBOT_ACCENT_COLORS: Tuple[Tuple[int, int, int], ...] = (
    (235, 190, 55),
    (236, 236, 228),
    (52, 58, 70),
    (232, 104, 59),
    (79, 184, 219),
)
WAREHOUSE_SHELF_FRAME_COLORS: Tuple[Tuple[int, int, int], ...] = (
    (86, 96, 110),
    (45, 101, 148),
    (72, 118, 100),
    (168, 105, 55),
    (126, 87, 148),
    (92, 99, 88),
    (182, 146, 57),
)
WAREHOUSE_SHELF_LOAD_COLORS: Tuple[Tuple[int, int, int], ...] = (
    (82, 133, 146),
    (172, 107, 58),
    (93, 140, 91),
    (188, 154, 86),
    (83, 92, 111),
    (178, 73, 60),
)


def _make_profile(
    *,
    source_scene: str,
    role: str,
    object_type: str,
    display_name: str,
    size_class: str,
    renderer: str,
    resource_kind: str = "standalone",
    support_required: bool = False,
    base_object_type: str | None = None,
    support_profile_id: str | None = None,
    mounting: str | None = None,
    dimensions_xyz: Sequence[float] | None = None,
) -> ThreeDObjectProfile:
    return ThreeDObjectProfile(
        profile_id=f"{source_scene}:{role}:{object_type}",
        canonical_id=_canonical_id(display_name),
        source_scene=str(source_scene),
        role=str(role),
        object_type=str(object_type),
        display_name=str(display_name),
        size_class=str(size_class),
        renderer=str(renderer),
        resource_kind=str(resource_kind),
        support_required=bool(support_required),
        base_object_type=str(base_object_type) if base_object_type is not None else None,
        support_profile_id=str(support_profile_id) if support_profile_id is not None else None,
        mounting=str(mounting) if mounting is not None else None,
        dimensions_xyz=tuple(float(value) for value in dimensions_xyz) if dimensions_xyz is not None else None,
    )


def _build_profiles() -> Tuple[ThreeDObjectProfile, ...]:
    profiles = []
    for object_type in OBJECT_SCENE_SMALL_SHAPE_TYPES:
        profiles.append(
            _make_profile(
                source_scene=OBJECT_SCENE_ID,
                role="spatial_small_shape",
                object_type=object_type,
                display_name=OBJECT_SCENE_NAME_BY_SHAPE_TYPE[object_type],
                size_class="small",
                renderer="object_scene_shape",
                resource_kind="standalone",
                dimensions_xyz=OBJECT_SCENE_SMALL_DIMENSIONS[object_type],
            )
        )
    for object_type in OBJECT_SCENE_CONTEXT_SHAPE_TYPES:
        profiles.append(
            _make_profile(
                source_scene=OBJECT_SCENE_ID,
                role="spatial_context_shape",
                object_type=object_type,
                display_name=OBJECT_SCENE_NAME_BY_SHAPE_TYPE[object_type],
                size_class="large",
                renderer="object_scene_shape",
                resource_kind="standalone",
                dimensions_xyz=OBJECT_SCENE_CONTEXT_DIMENSIONS[object_type],
            )
        )
    for object_type in OBJECT_CLUSTER_SHAPE_TYPES:
        profiles.append(
            _make_profile(
                source_scene=OBJECT_CLUSTER_SCENE_ID,
                role="cluster_small_shape",
                object_type=object_type,
                display_name=OBJECT_CLUSTER_NAME_BY_SHAPE_TYPE[object_type],
                size_class="small",
                renderer="object_scene_shape",
                resource_kind="standalone",
                dimensions_xyz=OBJECT_CLUSTER_DIMENSIONS[object_type],
            )
        )
    for object_type in (*ROOM_QUERY_TARGET_TYPES, *ROOM_EXTRA_WALL_TYPES):
        profiles.append(
            _make_profile(
                source_scene=ROOM_SCENE_ID,
                role="room_wall_object",
                object_type=object_type,
                display_name=ROOM_OBJECT_PROMPT_NAMES[object_type][0],
                size_class="small",
                renderer="room_wall_object",
                resource_kind="mounted",
                support_required=True,
                base_object_type=object_type,
                mounting="wall_mounted",
                dimensions_xyz=ROOM_WALL_BASE_DIMENSIONS[object_type],
            )
        )
    for object_type, spec in ROOM_SURFACE_DISTRACTOR_SPECS.items():
        profiles.append(
            _make_profile(
                source_scene=ROOM_SCENE_ID,
                role="room_surface_variant",
                object_type=object_type,
                display_name=str(spec["prompt_name"]),
                size_class="small",
                renderer="room_floor_object",
                resource_kind="composite",
                support_required=True,
                base_object_type=object_type,
                mounting="on_furniture",
                dimensions_xyz=spec["dimensions_xyz"],
            )
        )
    for prop_shape in ROOM_FLOOR_PROP_SHAPES:
        spec = ROOM_FLOOR_PROP_SPECS[prop_shape]
        profiles.append(
            _make_profile(
                source_scene=ROOM_SCENE_ID,
                role="room_floor_prop",
                object_type=prop_shape,
                display_name=str(spec["prompt_name"]),
                size_class="large",
                renderer="room_floor_object",
                resource_kind="standalone",
                mounting="floor",
                dimensions_xyz=spec["dimensions_xyz"],
            )
        )
    for object_type in ROOM_FLOOR_DISTRACTOR_TYPES:
        spec = ROOM_FLOOR_DISTRACTOR_SPECS[object_type]
        profiles.append(
            _make_profile(
                source_scene=ROOM_SCENE_ID,
                role="room_floor_variant",
                object_type=object_type,
                display_name=str(spec["prompt_name"]),
                size_class="large",
                renderer="room_floor_object",
                resource_kind="variant",
                base_object_type=object_type,
                mounting="floor",
                dimensions_xyz=spec["dimensions_xyz"],
            )
        )
    for object_type in STREET_OBJECT_TYPES:
        profiles.append(
            _make_profile(
                source_scene=STREET_SCENE_ID,
                role="street_candidate",
                object_type=object_type,
                display_name=STREET_OBJECT_NAMES[object_type],
                size_class="large" if object_type in STREET_VEHICLE_OBJECT_TYPES else "small",
                renderer="street_object",
                resource_kind="standalone",
                mounting="ground",
                dimensions_xyz=STREET_OBJECT_BASE_DIMENSIONS[object_type],
            )
        )
    for object_type in STREET_CONTEXT_OBJECT_TYPES:
        profiles.append(
            _make_profile(
                source_scene=STREET_SCENE_ID,
                role="street_context",
                object_type=object_type,
                display_name=STREET_OBJECT_NAMES[object_type],
                size_class="large",
                renderer="street_object",
                resource_kind="scene_support" if object_type in {"building", "store", "office_building"} else "standalone",
                mounting="ground",
                dimensions_xyz=STREET_OBJECT_BASE_DIMENSIONS[object_type],
            )
        )
    for object_type in SURFACE_FIXTURE_OBJECT_TYPES:
        profiles.append(
            _make_profile(
                source_scene=SURFACE_FIXTURE_SCENE_ID,
                role="surface_fixture_variant",
                object_type=object_type,
                display_name=SURFACE_FIXTURE_DISPLAY_NAME[object_type],
                size_class="large",
                renderer="surface_fixture",
                resource_kind="scene_support",
                mounting="surface",
                dimensions_xyz=(4.0, 0.12, 2.4),
            )
        )
    warehouse_seen = []
    for object_type in ("warehouse_robot", *WAREHOUSE_OBJECT_TYPES, *WAREHOUSE_CONTEXT_OBJECT_TYPES):
        if object_type not in warehouse_seen:
            warehouse_seen.append(object_type)
    warehouse_large = {
        "warehouse_robot",
        "shelf_rack",
        "crate_stack",
        "pallet_load",
        "forklift",
        "box_stack",
        "charging_dock",
        "conveyor",
        "pallet",
        "workbench",
        "wrapped_bundle",
    }
    for object_type in warehouse_seen:
        profiles.append(
            _make_profile(
                source_scene=WAREHOUSE_SCENE_ID,
                role="warehouse_object" if object_type in WAREHOUSE_OBJECT_TYPES or object_type == "warehouse_robot" else "warehouse_context",
                object_type=object_type,
                display_name=WAREHOUSE_OBJECT_NAMES[object_type],
                size_class="large" if object_type in warehouse_large else "small",
                renderer="warehouse_object",
                resource_kind="scene_support" if object_type == "shelf_rack" else "standalone",
                mounting="floor",
                dimensions_xyz=WAREHOUSE_OBJECT_BASE_DIMENSIONS[object_type],
            )
        )
    profiles.append(
        _make_profile(
            source_scene=WAREHOUSE_SCENE_ID,
            role="warehouse_reference",
            object_type=WAREHOUSE_NEAREST_REFERENCE_OBJECT_TYPE,
            display_name=WAREHOUSE_NEAREST_REFERENCE_OBJECT_NAME,
            size_class="small",
            renderer="warehouse_object",
            resource_kind="reference",
            mounting="floor",
            dimensions_xyz=WAREHOUSE_NEAREST_REFERENCE_OBJECT_DIMENSIONS,
        )
    )
    return tuple(profiles)


THREE_D_OBJECT_PROFILES: Tuple[ThreeDObjectProfile, ...] = _build_profiles()
_PROFILE_BY_ID: Dict[str, ThreeDObjectProfile] = {profile.profile_id: profile for profile in THREE_D_OBJECT_PROFILES}


def object_profiles(
    *,
    source_scene: str | None = None,
    role: str | None = None,
    size_class: str | None = None,
    resource_kind: str | None = None,
) -> Tuple[ThreeDObjectProfile, ...]:
    """Return canonical object profiles filtered by scene role."""

    profiles = THREE_D_OBJECT_PROFILES
    if source_scene is not None:
        profiles = tuple(profile for profile in profiles if profile.source_scene == str(source_scene))
    if role is not None:
        profiles = tuple(profile for profile in profiles if profile.role == str(role))
    if size_class is not None:
        profiles = tuple(profile for profile in profiles if profile.size_class == str(size_class))
    if resource_kind is not None:
        profiles = tuple(profile for profile in profiles if profile.resource_kind == str(resource_kind))
    return tuple(profiles)


def object_profile_by_id(profile_id: str) -> ThreeDObjectProfile:
    """Return one canonical object profile by stable profile id."""

    try:
        return _PROFILE_BY_ID[str(profile_id)]
    except KeyError as exc:
        raise KeyError(f"unknown three_d object profile_id: {profile_id}") from exc


def object_profile_or_none(
    *,
    source_scene: str,
    object_type: str,
    role: str | None = None,
    resource_kind: str | None = None,
) -> ThreeDObjectProfile | None:
    """Return one profile for a scene/object filter, or ``None`` when unavailable or ambiguous."""

    matches = object_profiles(source_scene=str(source_scene), role=role, resource_kind=resource_kind)
    matches = tuple(profile for profile in matches if profile.object_type == str(object_type))
    if len(matches) != 1:
        return None
    return matches[0]


def object_profile(
    *,
    source_scene: str,
    object_type: str,
    role: str | None = None,
    resource_kind: str | None = None,
) -> ThreeDObjectProfile:
    """Return exactly one profile for a scene/object filter."""

    profile = object_profile_or_none(
        source_scene=str(source_scene),
        object_type=str(object_type),
        role=role,
        resource_kind=resource_kind,
    )
    if profile is None:
        role_text = f", role={role}" if role is not None else ""
        kind_text = f", resource_kind={resource_kind}" if resource_kind is not None else ""
        raise KeyError(f"expected one three_d object profile for scene={source_scene}, object_type={object_type}{role_text}{kind_text}")
    return profile


def scene_profile_ids(*, source_scene: str, role: str | None = None, resource_kind: str | None = None) -> Tuple[str, ...]:
    """Return stable profile ids available to one scene/profile filter."""

    return tuple(
        profile.profile_id
        for profile in object_profiles(source_scene=str(source_scene), role=role, resource_kind=resource_kind)
    )


def profile_display_name(
    *,
    source_scene: str,
    object_type: str,
    role: str | None = None,
    fallback: str | None = None,
) -> str:
    """Return the prompt-facing display name for one scene object profile."""

    profile = object_profile_or_none(source_scene=str(source_scene), role=role, object_type=str(object_type))
    if profile is not None:
        return str(profile.display_name)
    if fallback is not None:
        return str(fallback)
    return str(object_type).replace("_", " ")


def profile_dimensions_xyz(
    *,
    source_scene: str,
    object_type: str,
    role: str | None = None,
) -> Tuple[float, ...] | None:
    """Return profile dimensions for one scene object, when the registry owns them."""

    profile = object_profile_or_none(source_scene=str(source_scene), role=role, object_type=str(object_type))
    return tuple(float(value) for value in profile.dimensions_xyz) if profile is not None and profile.dimensions_xyz is not None else None


def profiles_by_canonical_id() -> Mapping[str, Tuple[ThreeDObjectProfile, ...]]:
    grouped: Dict[str, Tuple[ThreeDObjectProfile, ...]] = {}
    for profile in THREE_D_OBJECT_PROFILES:
        grouped[profile.canonical_id] = (*grouped.get(profile.canonical_id, ()), profile)
    return grouped
