"""Passive state objects for cuboid orthographic-view rendering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

from PIL import Image

Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]
Color = Tuple[int, int, int]


@dataclass(frozen=True)
class CuboidDimensions:
    """Integer cuboid dimensions used by the orthographic scene grammar."""

    length: int
    width: int
    height: int

    @property
    def top_view_perimeter(self) -> int:
        return 2 * (int(self.length) + int(self.width))

    @property
    def front_view_perimeter(self) -> int:
        return 2 * (int(self.length) + int(self.height))

    @property
    def right_view_perimeter(self) -> int:
        return 2 * (int(self.width) + int(self.height))

    @property
    def volume(self) -> int:
        return int(self.length) * int(self.width) * int(self.height)

    @property
    def surface_area(self) -> int:
        return 2 * (
            (int(self.length) * int(self.width))
            + (int(self.length) * int(self.height))
            + (int(self.width) * int(self.height))
        )


@dataclass(frozen=True)
class RenderedCuboidViewsScene:
    """Rendered scene plus projected visual witnesses for the public task."""

    image: Image.Image
    annotation_bboxes: Mapping[str, BBox]
    annotation_roles: Tuple[str, ...]
    label_bboxes: Dict[str, BBox]
    scene_entities: Tuple[Dict[str, Any], ...]
    render_map: Dict[str, Any]
