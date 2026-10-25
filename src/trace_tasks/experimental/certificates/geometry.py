"""Coordinate frames and source-pixel annotation views (DESIGN 2.2, 2.3).

Every geometric quantity is scored in source-image pixels. A response frame maps to source
pixels by a per-axis scale: ``px_source`` is the identity, ``rel1000`` uses ``x*W/1000`` and
``y*H/1000``, and the diagnostic ``px_shown`` frame undoes Qwen2.5-VL's 28-px smart resize.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Context, Decimal
from numbers import Real
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from trace_tasks.core.reward_scoring import _normalize_bbox

from .types import COORD_FRAMES

Box = Tuple[float, float, float, float]
Point = Tuple[float, float]

# Qwen2.5-VL / EasyR1 image geometry (EasyR1 process_image bounds; trace-tool agent defaults).
QWEN25_FACTOR = 28
QWEN25_MIN_PIXELS = 262144
QWEN25_MAX_PIXELS = 4194304

BOX_ANNOTATION_TYPES = ("bbox", "bbox_set", "bbox_sequence", "bbox_map", "bbox_set_map")
POINT_ANNOTATION_TYPES = ("point", "point_set", "point_sequence", "point_map", "point_set_map")
SEGMENT_ANNOTATION_TYPES = ("segment", "segment_set")
_MAX_ANNOTATION_DEPTH = 6


@dataclass(frozen=True)
class Frame:
    """Response frame -> source pixels: ``x_src = x * scale_x``, ``y_src = y * scale_y``."""

    name: str
    scale_x: float
    scale_y: float

    @property
    def is_identity(self) -> bool:
        return self.scale_x == 1.0 and self.scale_y == 1.0

    def box(self, value: Any) -> Optional[Box]:
        """Legacy-normalized box (``_normalize_bbox``) mapped to source pixels, or None."""
        normalized = normalize_box(value)
        if normalized is None:
            return None
        x0, y0, x1, y1 = normalized
        return (x0 * self.scale_x, y0 * self.scale_y, x1 * self.scale_x, y1 * self.scale_y)

    def point(self, value: Any) -> Optional[Point]:
        """``[x, y]`` of two finite non-bool numbers mapped to source pixels, or None."""
        normalized = normalize_point(value)
        if normalized is None:
            return None
        return (normalized[0] * self.scale_x, normalized[1] * self.scale_y)


def as_finite_float(value: Any) -> Optional[float]:
    """``float(value)`` for a finite non-bool real number, else None.

    A JSON integer too large for a float (``10**400`` is valid RFC 8259 JSON) is not a
    coordinate: ``float()`` would raise OverflowError, so it reads as None (never a crash).
    """
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    try:
        out = float(value)
    except (OverflowError, ValueError, TypeError):
        return None
    return out if math.isfinite(out) else None


def _finite_number(value: Any) -> bool:
    return as_finite_float(value) is not None


def normalize_box(value: Any) -> Optional[Box]:
    """Legacy ``_normalize_bbox``: 4 finite non-bool numbers, sorted corners, non-zero area.

    Numbers that do not fit a float make the box invalid (None) instead of raising.
    """
    try:
        normalized = _normalize_bbox(value)
    except (OverflowError, ValueError, TypeError):
        return None
    return None if normalized is None else tuple(normalized)  # type: ignore[return-value]


def sanitize_numbers(value: Any, *, _depth: int = 0) -> Any:
    """Copy of a parsed JSON value with every number that does not fit a float replaced by
    None, so TRACE's legacy normalizers (which call ``float()``) reject it instead of raising.
    Containers deeper than the legacy annotation types are returned unchanged."""
    if _depth > 2 * _MAX_ANNOTATION_DEPTH:
        return value
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, Real):
        return value if as_finite_float(value) is not None or isinstance(value, float) else None
    if isinstance(value, dict):
        return {key: sanitize_numbers(item, _depth=_depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_numbers(item, _depth=_depth + 1) for item in value]
    return value


def normalize_point(value: Any) -> Optional[Point]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 2:
        return None
    x, y = as_finite_float(value[0]), as_finite_float(value[1])
    if x is None or y is None:
        return None
    return (x, y)


def source_frame(coord_frame: str, image_size: Sequence[int]) -> Frame:
    """The public contract frame of a row (``px_source`` or ``rel1000``)."""
    width, height = _size(image_size)
    if coord_frame == "px_source":
        return Frame("px_source", 1.0, 1.0)
    if coord_frame == "rel1000":
        return Frame("rel1000", width / 1000.0, height / 1000.0)
    raise ValueError(f"coord_frame must be one of {COORD_FRAMES}, got {coord_frame!r}")


def smart_resize(
    height: float,
    width: float,
    factor: int = QWEN25_FACTOR,
    min_pixels: int = QWEN25_MIN_PIXELS,
    max_pixels: int = QWEN25_MAX_PIXELS,
) -> Tuple[int, int]:
    """Qwen-VL smart resize: ``(height, width)`` rounded to ``factor`` with area in bounds."""
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return int(h_bar), int(w_bar)


def shown_size(
    image_size: Sequence[int],
    *,
    factor: int = QWEN25_FACTOR,
    min_pixels: int = QWEN25_MIN_PIXELS,
    max_pixels: int = QWEN25_MAX_PIXELS,
) -> Tuple[int, int]:
    """(W, H) of the image as Qwen2.5-VL sees it."""
    width, height = _size(image_size)
    shown_h, shown_w = smart_resize(height, width, factor, min_pixels, max_pixels)
    return shown_w, shown_h


def shown_frame(image_size: Sequence[int], **resize: int) -> Frame:
    """Diagnostic frame: coordinates given in Qwen2.5-VL's shown image, mapped to source px."""
    width, height = _size(image_size)
    shown_w, shown_h = shown_size((width, height), **resize)
    return Frame("px_shown", width / shown_w, height / shown_h)


def _size(image_size: Sequence[int]) -> Tuple[int, int]:
    width, height = int(image_size[0]), int(image_size[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"image size must be positive, got {image_size!r}")
    return width, height


def _leaf_size(annotation_type: str) -> int:
    if annotation_type in BOX_ANNOTATION_TYPES:
        return 4
    if annotation_type in POINT_ANNOTATION_TYPES or annotation_type in SEGMENT_ANNOTATION_TYPES:
        return 2
    raise ValueError(f"unsupported annotation type {annotation_type!r}")


def scale_annotation(value: Any, annotation_type: str, frame: Frame) -> Any:
    """Map a legacy annotation value to source pixels, keeping its structure.

    In the identity frame the value is returned unchanged, so legacy G is untouched. Otherwise
    every list of exactly 4 (box types) or 2 (point/segment types) finite numbers is scaled;
    anything else is left as is and fails the legacy normalizer exactly as before. JSON-string
    annotations are not decoded (they only occur in malformed responses).
    """
    if frame.is_identity:
        return value
    leaf = _leaf_size(annotation_type)

    def walk(node: Any, depth: int) -> Any:
        if depth > _MAX_ANNOTATION_DEPTH:
            return node  # deeper than any annotation type; the legacy normalizer rejects it
        if isinstance(node, list):
            if len(node) == leaf and all(_finite_number(item) for item in node):
                scales = (frame.scale_x, frame.scale_y) * (leaf // 2)
                return [float(item) * scale for item, scale in zip(node, scales)]
            return [walk(item, depth + 1) for item in node]
        if isinstance(node, dict):
            return {key: walk(item, depth + 1) for key, item in node.items()}
        return node

    return walk(value, 0)


@dataclass(frozen=True)
class AnnotationView:
    """Index-aligned source-pixel geometry of a model annotation, per role.

    ``roles[""]`` holds the list for ``bbox``/``bbox_set``/``bbox_sequence`` (and the point
    analogues); map types use their keys (for example ``filter_cells``, ``player_cell``).
    Every entry is a source-pixel box/point or None when that element is malformed, so a
    certificate index stays bound to the same position the model wrote.
    """

    annotation_type: str
    geometry: str
    roles: Mapping[str, Tuple[Optional[Tuple[float, ...]], ...]]
    structure_ok: bool
    raw: Any = None
    # Identity resolutions of this view, keyed by (role, thresholds, candidate boxes), so the
    # checks, diagnostics and identity report of one scoring call resolve each box once.
    cache: Dict[Any, Any] = field(default_factory=dict, compare=False, repr=False, hash=False)

    def items(self, role: str = "") -> Tuple[Optional[Tuple[float, ...]], ...]:
        return tuple(self.roles.get(role, ()))

    def size(self, role: str = "") -> int:
        return len(self.roles.get(role, ()))

    def max_size(self) -> int:
        """Length of the longest role list (0 for an empty or unparsable annotation)."""
        return max((len(items) for items in self.roles.values()), default=0)


def _role_name(key: Any) -> str:
    """A map annotation key as a role name ("" when it has no text form, which marks the
    structure invalid). A Python-literal annotation (legacy extraction) can key a map by an
    integer beyond ``sys.int_max_str_digits``, or a tuple holding one, whose ``str()`` raises
    (review VER-1); such a key never names a role."""
    if isinstance(key, str):
        return key.strip()
    try:
        return str(key).strip()
    except (ValueError, TypeError, RecursionError):
        return ""


def annotation_view(value: Any, annotation_type: str, frame: Frame) -> AnnotationView:
    """Build an :class:`AnnotationView` for ``value`` (response frame) in source pixels."""
    if annotation_type in BOX_ANNOTATION_TYPES:
        geometry, convert, leaf = "box", frame.box, 4
    elif annotation_type in POINT_ANNOTATION_TYPES:
        geometry, convert, leaf = "point", frame.point, 2
    else:
        raise ValueError(f"annotation views support box and point types, got {annotation_type!r}")

    def is_leaf(node: Any) -> bool:
        return isinstance(node, list) and len(node) == leaf and all(_finite_number(i) for i in node)

    def as_list(node: Any) -> Optional[Tuple[Optional[Tuple[float, ...]], ...]]:
        if is_leaf(node):
            return (convert(node),)
        if isinstance(node, dict) and geometry == "box" and "bboxes" in node:
            node = node["bboxes"]
        if not isinstance(node, list):
            return None
        return tuple(convert(item) for item in node)

    roles: Dict[str, Tuple[Optional[Tuple[float, ...]], ...]] = {}
    structure_ok = True
    single = annotation_type in ("bbox", "point")
    is_map = annotation_type.endswith("_map")
    if not is_map:
        if single:
            roles[""] = (convert(value),)
            structure_ok = roles[""][0] is not None
        else:
            items = as_list(value)
            structure_ok = items is not None
            roles[""] = items or ()
        return AnnotationView(annotation_type, geometry, roles, structure_ok, value)

    if not isinstance(value, dict):
        return AnnotationView(annotation_type, geometry, {}, False, value)
    set_map = annotation_type.endswith("_set_map")
    for key, item in value.items():
        role = _role_name(key)
        if not role or role in roles:
            structure_ok = False
            continue
        if set_map:
            items = as_list(item)
            structure_ok = structure_ok and items is not None
            roles[role] = items or ()
        else:
            converted = convert(item)
            structure_ok = structure_ok and converted is not None
            roles[role] = (converted,)
    return AnnotationView(annotation_type, geometry, roles, structure_ok, value)


# A private decimal context: rounding never reads or sets the thread's ambient context, and
# 400 digits cover every finite float (|x| < 1.8e308).
_ROUND_CONTEXT_PREC = 400


def round_half_up(value: float) -> int:
    """Round half away from zero on the decimal repr (278.5 -> 279), as the SFT target does.

    Hermetic: a fresh private ``decimal.Context`` per call, so neither the caller's context
    (precision, traps) nor its flags are read or changed."""
    context = Context(prec=_ROUND_CONTEXT_PREC, rounding=ROUND_HALF_UP)
    return int(Decimal(repr(float(value))).quantize(Decimal(1), rounding=ROUND_HALF_UP, context=context))


def round_box_half_up(box: Sequence[float]) -> list[int]:
    return [round_half_up(float(coord)) for coord in box]


__all__ = [
    "AnnotationView",
    "Box",
    "Frame",
    "Point",
    "QWEN25_FACTOR",
    "QWEN25_MAX_PIXELS",
    "QWEN25_MIN_PIXELS",
    "annotation_view",
    "normalize_box",
    "normalize_point",
    "round_box_half_up",
    "round_half_up",
    "sanitize_numbers",
    "as_finite_float",
    "scale_annotation",
    "shown_frame",
    "shown_size",
    "smart_resize",
    "source_frame",
]
