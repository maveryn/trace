"""Scene-local defaults for function-graph geometry tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Tuple

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.shared.config_defaults import split_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.geometry.shared.background_defaults import load_geometry_background_defaults
from trace_tasks.tasks.geometry.shared.noise_defaults import load_geometry_noise_defaults

DOMAIN = "geometry"
SCENE_ID = "function_graph"


@dataclass(frozen=True)
class FunctionGraphDefaults:
    """Fallback scene constants shared by function-graph objectives."""

    canvas_size_min: int = 640
    canvas_size_max: int = 720
    graph_cells_min: int = 20
    graph_cells_max: int = 20
    line_width: int = 4
    marker_radius: int = 7
    label_font_size_min: int = 16
    label_font_size_max: int = 24
    average_rate_support: Tuple[float, ...] = (-2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0)
    sinusoid_turning_support: Tuple[int, ...] = (3, 4)
    sinusoid_local_support: Tuple[int, ...] = (2,)
    piecewise_turning_support: Tuple[int, ...] = (2, 3, 4, 5, 6)
    piecewise_local_support: Tuple[int, ...] = (2, 3, 4, 5, 6)
    piecewise_turning_x_positions: Tuple[int, ...] = (-9, -7, -5, -3, -2, -1, 0, 1, 2, 3, 5, 7, 9)


DEFAULTS = FunctionGraphDefaults()
SCENE_DEFAULTS = get_scene_defaults(DOMAIN, SCENE_ID)
GEN_DEFAULTS, RENDER_DEFAULTS, PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    SCENE_DEFAULTS if isinstance(SCENE_DEFAULTS, Mapping) else {},
)
POST_IMAGE_BACKGROUND_DEFAULTS = load_geometry_background_defaults(scene_id=SCENE_ID)
POST_IMAGE_NOISE_DEFAULTS = load_geometry_noise_defaults(scene_id=SCENE_ID)


def int_tuple_default(key: str, fallback: Sequence[int]) -> Tuple[int, ...]:
    """Resolve one non-empty integer support sequence from scene defaults."""

    raw_value = GEN_DEFAULTS.get(str(key), fallback)
    if not isinstance(raw_value, Sequence) or isinstance(raw_value, (str, bytes)):
        raise ValueError(f"{key} must be a sequence of integers for {SCENE_ID}")
    values = tuple(int(value) for value in raw_value)
    if not values:
        raise ValueError(f"{key} cannot be empty for {SCENE_ID}")
    return values


def float_tuple_default(key: str, fallback: Sequence[float]) -> Tuple[float, ...]:
    """Resolve one non-empty float support sequence from scene defaults."""

    raw_value = GEN_DEFAULTS.get(str(key), fallback)
    if not isinstance(raw_value, Sequence) or isinstance(raw_value, (str, bytes)):
        raise ValueError(f"{key} must be a sequence of numbers for {SCENE_ID}")
    values = tuple(round(float(value), 1) for value in raw_value)
    if not values:
        raise ValueError(f"{key} cannot be empty for {SCENE_ID}")
    return values
