"""Scene state catalogs for synthetic 3D surface-fixture panels."""

from __future__ import annotations

from typing import Mapping, Tuple

from trace_tasks.tasks.shared.color_format import format_named_color_with_hex
from trace_tasks.tasks.shared.named_colors import available_named_colors


SCENE_ID = "surface_fixture"

SUPPORTED_SCENE_VARIANTS: Tuple[str, ...] = (
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

ELEMENT_TYPE_BY_SCENE_VARIANT: Mapping[str, str] = {
    "wall_tile_panel": "tile",
    "perforated_panel": "hole",
    "slot_board": "slot",
    "compartment_tray": "compartment",
    "vent_panel": "vent",
    "window_grid": "window",
    "door_bank": "door",
    "drawer_pull_panel": "drawer_pull",
    "brick_wall": "brick",
    "paver_floor": "paver",
    "locker_bank": "locker",
    "mailbox_bank": "mailbox",
    "server_rack": "drive_bay",
    "control_panel": "button",
    "solar_panel_array": "solar_panel",
    "screw_plate": "screw",
    "hex_nut_plate": "hex_nut",
    "washer_plate": "washer",
    "socket_bank": "socket",
    "hook_board": "hook",
    "indicator_light_panel": "light",
    "bracket_panel": "bracket",
    "u_bolt_plate": "u_bolt",
    "pipe_rack": "pipe",
}

SCENE_VARIANT_BY_ELEMENT_TYPE: Mapping[str, str] = {
    str(element_type): str(scene_variant)
    for scene_variant, element_type in ELEMENT_TYPE_BY_SCENE_VARIANT.items()
}

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

ELEMENT_DISPLAY_NAME: Mapping[str, str] = {
    "tile": "tile",
    "hole": "hole",
    "slot": "slot",
    "compartment": "compartment",
    "vent": "vent",
    "window": "window",
    "door": "door",
    "drawer_pull": "drawer pull",
    "brick": "brick",
    "paver": "paver",
    "locker": "locker",
    "mailbox": "mailbox",
    "drive_bay": "drive bay",
    "button": "button",
    "solar_panel": "solar panel",
    "screw": "screw",
    "hex_nut": "hex nut",
    "washer": "washer",
    "socket": "socket",
    "hook": "hook",
    "light": "light",
    "bracket": "bracket",
    "u_bolt": "U-bolt",
    "pipe": "pipe",
}

ELEMENT_PLURAL: Mapping[str, str] = {
    "tile": "tiles",
    "hole": "holes",
    "slot": "slots",
    "compartment": "compartments",
    "vent": "vents",
    "window": "windows",
    "door": "doors",
    "drawer_pull": "drawer pulls",
    "brick": "bricks",
    "paver": "pavers",
    "locker": "lockers",
    "mailbox": "mailboxes",
    "drive_bay": "drive bays",
    "button": "buttons",
    "solar_panel": "solar panels",
    "screw": "screws",
    "hex_nut": "hex nuts",
    "washer": "washers",
    "socket": "sockets",
    "hook": "hooks",
    "light": "lights",
    "bracket": "brackets",
    "u_bolt": "U-bolts",
    "pipe": "pipes",
}

SEMANTIC_COLOR_RGB: Mapping[str, Tuple[int, int, int]] = {
    str(name): (int(rgb[0]), int(rgb[1]), int(rgb[2]))
    for name, rgb in available_named_colors()
}

SEMANTIC_COLOR_SUPPORT: Tuple[str, ...] = tuple(SEMANTIC_COLOR_RGB.keys())

COLORABLE_SCENE_VARIANTS: Tuple[str, ...] = (
    "wall_tile_panel",
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

COLOR_READOUT_SCENE_VARIANTS: Tuple[str, ...] = (
    "wall_tile_panel",
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
    "socket_bank",
    "indicator_light_panel",
)

SURFACE_FIXTURE_OBJECT_TYPES: Tuple[str, ...] = SUPPORTED_SCENE_VARIANTS


def semantic_color_label(color_name: str) -> str:
    """Return the visible prompt label for one semantic color."""

    rgb = SEMANTIC_COLOR_RGB[str(color_name)]
    return format_named_color_with_hex(str(color_name), rgb)


__all__ = [
    "COLORABLE_SCENE_VARIANTS",
    "COLOR_READOUT_SCENE_VARIANTS",
    "ELEMENT_DISPLAY_NAME",
    "ELEMENT_PLURAL",
    "ELEMENT_TYPE_BY_SCENE_VARIANT",
    "SCENE_ID",
    "SCENE_VARIANT_BY_ELEMENT_TYPE",
    "SEMANTIC_COLOR_RGB",
    "SEMANTIC_COLOR_SUPPORT",
    "SUPPORTED_SCENE_VARIANTS",
    "SURFACE_FIXTURE_DISPLAY_NAME",
    "SURFACE_FIXTURE_OBJECT_TYPES",
    "semantic_color_label",
]
