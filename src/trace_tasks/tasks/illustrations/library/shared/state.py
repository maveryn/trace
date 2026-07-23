"""Scene state records for the library illustration package."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Tuple

from PIL import Image

from ...shared.object_catalog import label_map_for_tag, public_name_map_for_tag, variant_ids_with_tag
from ...shared.object_library import BBox, RGB


LIBRARY_SETTING_IDS: Tuple[str, ...] = variant_ids_with_tag("library_setting")
LIBRARY_SECTION_TYPES: Tuple[str, ...] = variant_ids_with_tag("library_section")
LIBRARY_SECTION_LABELS: dict[str, str] = label_map_for_tag("library_section")
LIBRARY_SECTION_DISPLAY_NAMES: dict[str, str] = public_name_map_for_tag("library_section")
BOOK_ORIENTATIONS: Tuple[str, ...] = variant_ids_with_tag("library_book_orientation")


@dataclass(frozen=True)
class LibraryBookSpec:
    """Requested semantic properties for one rendered book."""

    section_key: str
    color_name: str
    orientation: str = "upright"
    role: str = "distractor"
    attributes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LibrarySectionSpec:
    """Requested books for one labeled library section."""

    section_key: str
    book_specs: Tuple[LibraryBookSpec, ...]
    role: str = "distractor"
    attributes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LibraryBook:
    """One rendered book with trace-ready metadata."""

    book_id: str
    section_id: str
    section_key: str
    section_name: str
    color_name: str
    color_rgb: RGB
    color_label: str
    orientation: str
    bbox_xyxy: BBox
    row_index: int
    slot_index: int
    role: str
    attributes: Mapping[str, Any]


@dataclass(frozen=True)
class LibrarySection:
    """One labeled shelf section."""

    section_id: str
    section_key: str
    section_name: str
    label: str
    bbox_xyxy: BBox
    label_bbox_xyxy: BBox
    shelf_bboxes_xyxy: Tuple[BBox, ...]
    book_ids: Tuple[str, ...]
    role: str
    attributes: Mapping[str, Any]


@dataclass(frozen=True)
class LibraryDecor:
    """Non-query visual decor drawn in the library."""

    decor_id: str
    decor_type: str
    bbox_xyxy: BBox
    attributes: Mapping[str, Any]
    object_record: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class RenderedLibraryScene:
    """Rendered library scene plus metadata for verifier-side projection."""

    image: Image.Image
    setting_id: str
    sections: Tuple[LibrarySection, ...]
    books: Tuple[LibraryBook, ...]
    decor: Tuple[LibraryDecor, ...]
    canvas_width: int
    canvas_height: int
    render_scale: int
    style_id: str
    layout: Mapping[str, Any]


def library_section_display_name(section_key: str) -> str:
    """Return prompt-facing library section text."""

    return LIBRARY_SECTION_DISPLAY_NAMES.get(str(section_key), str(section_key).replace("_", " ").title())


__all__ = [
    "BOOK_ORIENTATIONS",
    "LIBRARY_SECTION_DISPLAY_NAMES",
    "LIBRARY_SECTION_LABELS",
    "LIBRARY_SECTION_TYPES",
    "LIBRARY_SETTING_IDS",
    "LibraryBook",
    "LibraryBookSpec",
    "LibraryDecor",
    "LibrarySection",
    "LibrarySectionSpec",
    "RenderedLibraryScene",
    "library_section_display_name",
]
