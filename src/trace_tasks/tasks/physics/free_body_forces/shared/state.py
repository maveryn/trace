"""Passive state records for free-body-force diagrams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image


SCENE_ID = "free_body_forces"
SCENE_NAMESPACE = "physics_free_body_forces"

SUPPORTED_SCENE_VARIANTS: tuple[str, ...] = ("clean_table", "gridded_table", "lab_card")
OPTION_LETTERS: tuple[str, ...] = ("A", "B", "C", "D", "E", "F", "G", "H")
DIRECTION_NAMES: tuple[str, ...] = (
    "east",
    "northeast",
    "north",
    "northwest",
    "west",
    "southwest",
    "south",
    "southeast",
)
DIRECTION_VECTORS: dict[str, tuple[int, int]] = {
    "east": (1, 0),
    "northeast": (1, 1),
    "north": (0, 1),
    "northwest": (-1, 1),
    "west": (-1, 0),
    "southwest": (-1, -1),
    "south": (0, -1),
    "southeast": (1, -1),
}
CARDINAL_VECTORS: dict[str, tuple[int, int]] = {
    "east": (1, 0),
    "west": (-1, 0),
    "north": (0, 1),
    "south": (0, -1),
}
VECTOR_TO_DIRECTION: dict[tuple[int, int], str] = {
    value: key for key, value in DIRECTION_VECTORS.items()
}


@dataclass(frozen=True)
class ForceSpec:
    """One visible applied-force arrow and its symbolic vector."""

    force_id: str
    direction: str
    magnitude_n: int
    vector: tuple[int, int]


@dataclass(frozen=True)
class SamplingAxes:
    """Resolved non-query sampling axes for one force diagram."""

    scene_variant: str
    net_force_direction: str
    correct_option_letter: str
    accent_color_name: str
    scene_variant_probabilities: dict[str, float]
    net_force_direction_probabilities: dict[str, float]
    correct_option_letter_probabilities: dict[str, float]
    accent_color_name_probabilities: dict[str, float]


@dataclass(frozen=True)
class ForceScenario:
    """Resolved free-body-force scenario and option mapping."""

    scene_variant: str
    net_force_direction: str
    correct_option_letter: str
    option_directions: dict[str, str]
    force_specs: tuple[ForceSpec, ...]
    resultant_vector: tuple[int, int]


@dataclass(frozen=True)
class RenderedForceScene:
    """Rendered force diagram and projected applied-force witnesses."""

    image: Image.Image
    annotation_bbox_map: dict[str, list[float]]
    scene_entities: list[dict[str, Any]]
    render_map: dict[str, Any]
    background_meta: dict[str, Any]
    diagram_style_meta: dict[str, Any]
    post_noise_meta: dict[str, Any]
    font_family: str
