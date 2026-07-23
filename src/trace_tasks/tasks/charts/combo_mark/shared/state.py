"""State containers for combo-mark chart rendering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from PIL import Image


@dataclass(frozen=True)
class RenderParams:
    canvas_width: int
    canvas_height: int
    plot_left: int
    plot_right: int
    plot_top: int
    plot_bottom: int
    axis_width: int
    grid_width: int
    line_width: int
    point_radius: int
    tick_font_size: int
    label_font_size: int
    value_font_size: int
    legend_font_size: int
    primary_rgb: tuple[int, int, int]
    primary_alt_rgb: tuple[int, int, int]
    line_rgb: tuple[int, int, int]
    area_rgb: tuple[int, int, int]
    axis_rgb: tuple[int, int, int]
    grid_rgb: tuple[int, int, int]
    text_rgb: tuple[int, int, int]
    text_stroke_rgb: tuple[int, int, int]
    panel_rgb: tuple[int, int, int]
    layout_jitter_meta: Mapping[str, Any]


@dataclass(frozen=True)
class ComboScene:
    image: Image.Image
    labels: tuple[str, ...]
    primary_values: tuple[int, ...]
    line_values: tuple[int, ...]
    primary_points: tuple[tuple[float, float], ...]
    line_points: tuple[tuple[float, float], ...]
    entities: tuple[dict[str, Any], ...]
    scene_variant: str
    primary_name: str
    line_name: str
    primary_axis_max: int
    line_axis_max: int
    plot_bbox: tuple[int, int, int, int]
    legend_bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class ComboDataset:
    labels: tuple[str, ...]
    primary_values: tuple[int, ...]
    line_values: tuple[int, ...]
    primary_name: str
    line_name: str
    scene_variant: str
    label_count_range: tuple[int, int]
    scene_variant_probabilities: dict[str, float]


@dataclass(frozen=True)
class ComboRenderArtifacts:
    scene: ComboScene
    render_params: RenderParams
    background_style: dict[str, Any]
    font_assets: dict[str, Any]
    post_image_noise: dict[str, Any]


__all__ = [
    "ComboDataset",
    "ComboRenderArtifacts",
    "ComboScene",
    "RenderParams",
]
