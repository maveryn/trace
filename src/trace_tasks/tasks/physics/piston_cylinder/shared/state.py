"""Passive state records for piston-cylinder diagrams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image


SCENE_ID = "piston_cylinder"
SCENE_NAMESPACE = "physics_piston_cylinder"
SUPPORTED_ORIENTATIONS: tuple[str, ...] = ("vertical_pair", "horizontal_pair")


@dataclass(frozen=True)
class PistonRenderDefaults:
    """Stable fallback defaults for piston-cylinder rendering."""

    canvas_width: int = 1160
    canvas_height: int = 740
    panel_left_px: int = 58
    panel_top_px: int = 52
    panel_right_margin_px: int = 58
    panel_bottom_margin_px: int = 58
    cylinder_width_px: int = 238
    cylinder_height_px: int = 330
    horizontal_cylinder_width_px: int = 342
    horizontal_cylinder_height_px: int = 208
    piston_thickness_px: int = 24
    label_font_size_px: int = 26
    state_font_size_px: int = 24
    title_font_size_px: int = 30


@dataclass(frozen=True)
class PistonGenerationDefaults:
    """Stable fallback supports for constant-pressure boundary-work scenes."""

    pressure_mpa_support: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    volume_l_support: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 9)
    min_volume_delta: int = 2
    max_volume_delta: int = 6
    boundary_work_answer_support: tuple[int, ...] = (
        -36,
        -30,
        -24,
        -20,
        -18,
        -16,
        -15,
        -12,
        -10,
        -9,
        -8,
        -6,
        -5,
        -4,
        -3,
        -2,
        2,
        3,
        4,
        5,
        6,
        8,
        9,
        10,
        12,
        15,
        16,
        18,
        20,
        24,
        30,
        36,
    )


@dataclass(frozen=True)
class PistonScenario:
    """Resolved constant-pressure boundary-work scenario."""

    pressure_mpa: int
    initial_volume_l: int
    final_volume_l: int
    boundary_work_kj: int
    orientation: str
    orientation_probabilities: dict[str, float]
    pressure_probabilities: dict[str, float]
    initial_volume_probabilities: dict[str, float]
    final_volume_probabilities: dict[str, float]
    target_answer_probabilities: dict[str, float]


@dataclass(frozen=True)
class RenderedPistonScene:
    """Rendered piston-cylinder scene and projected metadata."""

    image: Image.Image
    annotation_bbox_map: dict[str, list[float]]
    scene_entities: list[dict[str, Any]]
    render_map: dict[str, Any]
    background_meta: dict[str, Any]
    diagram_style_meta: dict[str, Any]
    post_noise_meta: dict[str, Any]
    font_family: str


__all__ = [
    "PistonGenerationDefaults",
    "PistonRenderDefaults",
    "PistonScenario",
    "RenderedPistonScene",
    "SCENE_ID",
    "SCENE_NAMESPACE",
    "SUPPORTED_ORIENTATIONS",
]
