"""Games-domain readable text wrappers."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Mapping, Sequence, Tuple

from PIL import ImageDraw, ImageFont

from ....core.seed import hash64
from ...shared.text_legibility import (
    CONTEXT_TEXT_MIN_CONTRAST_RATIO,
    LARGE_TEXT_MIN_CONTRAST_RATIO,
    ReadableTextStyle,
    draw_centered_readable_text,
    draw_readable_text,
    draw_text_traced as _draw_text_traced,
    normalize_rgb,
    resolve_readable_text_style,
    text_bbox,
)

Color = Tuple[int, int, int]

_REQUIRED_GAME_TEXT_ROLES = {
    "readout",
    "board_mark",
    "game_symbol",
    "mini_board_label",
}


def _is_required_game_role(role: str, required: bool) -> bool:
    if bool(required):
        return True
    role_text = str(role)
    if role_text in _REQUIRED_GAME_TEXT_ROLES:
        return True
    return role_text.endswith("_label") or role_text.endswith("_readout")


def _style_passes(style: ReadableTextStyle) -> bool:
    try:
        return bool(style.metadata().get("passes", False))
    except Exception:
        return False


def _coerce_surfaces(
    *,
    fill_rgb: Sequence[int],
    stroke_rgb: Sequence[int] | None,
    surface_rgbs: Sequence[Sequence[int]] | None,
) -> tuple[Color, ...]:
    if surface_rgbs:
        return tuple(normalize_rgb(color) for color in surface_rgbs if len(color) >= 3)
    if stroke_rgb is not None:
        return (normalize_rgb(stroke_rgb),)
    # Conservative fallback for old call sites that did not state a surface.
    fill = normalize_rgb(fill_rgb)
    return ((255, 255, 255),) if sum(fill) < 384 else ((10, 14, 22),)


def _seed_for_text(
    *,
    instance_seed: int | None,
    namespace: str,
    text: str,
    xy_key: Sequence[float] | None,
    surface_rgbs: Sequence[Sequence[int]],
) -> int:
    if instance_seed is not None:
        return int(instance_seed)
    surface_key = ";".join(",".join(str(int(channel)) for channel in color[:3]) for color in surface_rgbs)
    return hash64(0, f"{namespace}:{text}:{surface_key}")


@lru_cache(maxsize=16384)
def _resolve_game_text_style_cached(
    *,
    seed: int,
    namespace: str,
    role: str,
    surfaces: tuple[Color, ...],
    preferred: tuple[Color, ...],
    candidates: tuple[Color, ...],
    required: bool,
    min_contrast_ratio: float,
    min_lab_distance: float,
) -> ReadableTextStyle:
    attempts = (
        (float(min_contrast_ratio), float(min_lab_distance)),
        (float(LARGE_TEXT_MIN_CONTRAST_RATIO), 0.0),
        (float(CONTEXT_TEXT_MIN_CONTRAST_RATIO), 0.0),
        (1.0, 0.0),
    )
    last_style: ReadableTextStyle | None = None
    for contrast, lab_distance in attempts:
        style = resolve_readable_text_style(
            instance_seed=int(seed),
            namespace=str(namespace),
            role=str(role),
            surface_rgbs=surfaces,
            preferred_rgbs=preferred,
            candidate_rgbs=candidates or None,
            min_contrast_ratio=float(contrast),
            min_lab_distance=float(lab_distance),
            required=bool(required),
        )
        last_style = style
        if _style_passes(style):
            return style
    return last_style  # type: ignore[return-value]


def resolve_game_text_style(
    *,
    role: str,
    text: str,
    fill_rgb: Sequence[int],
    stroke_rgb: Sequence[int] | None = None,
    surface_rgbs: Sequence[Sequence[int]] | None = None,
    preferred_rgbs: Sequence[Sequence[int]] | None = None,
    candidate_rgbs: Sequence[Sequence[int]] | None = None,
    instance_seed: int | None = None,
    namespace: str = "games.text",
    xy_key: Sequence[float] | None = None,
    required: bool = True,
    min_contrast_ratio: float = LARGE_TEXT_MIN_CONTRAST_RATIO,
    min_lab_distance: float = 24.0,
) -> ReadableTextStyle:
    """Resolve readable nonsemantic game text ink for one visible surface."""

    surfaces = _coerce_surfaces(fill_rgb=fill_rgb, stroke_rgb=stroke_rgb, surface_rgbs=surface_rgbs)
    preferred = tuple(preferred_rgbs or (normalize_rgb(fill_rgb),))
    candidates = tuple(normalize_rgb(color) for color in (candidate_rgbs or ()))
    seed = _seed_for_text(
        instance_seed=instance_seed,
        namespace=str(namespace),
        text=str(text),
        xy_key=xy_key,
        surface_rgbs=surfaces,
    )
    return _resolve_game_text_style_cached(
        seed=int(seed),
        namespace=str(namespace),
        role=str(role),
        surfaces=surfaces,
        preferred=preferred,
        candidates=candidates,
        required=bool(required),
        min_contrast_ratio=round(float(min_contrast_ratio), 4),
        min_lab_distance=round(float(min_lab_distance), 4),
    )


def draw_game_text_traced(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float] | Sequence[float],
    text: str,
    *args: Any,
    role: str = "visible_text",
    required: bool = False,
    extra_metadata: Mapping[str, Any] | None = None,
    surface_rgbs: Sequence[Sequence[int]] | None = None,
    preferred_rgbs: Sequence[Sequence[int]] | None = None,
    candidate_rgbs: Sequence[Sequence[int]] | None = None,
    instance_seed: int | None = None,
    namespace: str | None = None,
    min_contrast_ratio: float = LARGE_TEXT_MIN_CONTRAST_RATIO,
    min_lab_distance: float = 24.0,
    **kwargs: Any,
) -> dict[str, Any]:
    """Draw PIL text as games readout/context text with readable ink."""

    font = kwargs.get("font")
    fill = kwargs.get("fill")
    if font is None or fill is None or kwargs.get("anchor") is not None:
        return _draw_text_traced(
            draw,
            xy,
            text,
            *args,
            role=role,
            required=required,
            extra_metadata=extra_metadata,
            **kwargs,
        )
    stroke_width = max(0, int(kwargs.get("stroke_width", 0) or 0))
    stroke_fill = kwargs.get("stroke_fill")
    fill_rgb = normalize_rgb(fill)
    stroke_rgb = normalize_rgb(stroke_fill) if stroke_fill is not None else None
    effective_required = _is_required_game_role(str(role), bool(required))
    text_style = resolve_game_text_style(
        role=str(role),
        text=str(text),
        fill_rgb=fill_rgb,
        stroke_rgb=stroke_rgb,
        surface_rgbs=surface_rgbs,
        preferred_rgbs=preferred_rgbs,
        candidate_rgbs=candidate_rgbs,
        instance_seed=instance_seed,
        namespace=str(namespace or f"games.text.{role}"),
        xy_key=(float(xy[0]), float(xy[1])),  # type: ignore[index]
        required=effective_required,
        min_contrast_ratio=float(min_contrast_ratio),
        min_lab_distance=float(min_lab_distance),
    )
    metadata = dict(extra_metadata or {})
    metadata.setdefault("source", "games_text_helper")
    return draw_readable_text(
        draw,
        xy=(float(xy[0]), float(xy[1])),  # type: ignore[index]
        text=str(text),
        font=font,
        style=text_style,
        stroke_width=max(1, int(stroke_width)) if effective_required else max(0, int(stroke_width)),
        extra_metadata=metadata,
    )


def draw_centered_game_text_traced(
    draw: ImageDraw.ImageDraw,
    *,
    center: tuple[float, float],
    text: str,
    font: ImageFont.ImageFont,
    fill_rgb: Sequence[int],
    stroke_rgb: Sequence[int] | None = None,
    stroke_width: int = 0,
    role: str = "readout",
    required: bool = False,
    extra_metadata: Mapping[str, Any] | None = None,
    surface_rgbs: Sequence[Sequence[int]] | None = None,
    preferred_rgbs: Sequence[Sequence[int]] | None = None,
    candidate_rgbs: Sequence[Sequence[int]] | None = None,
    instance_seed: int | None = None,
    namespace: str | None = None,
    min_contrast_ratio: float = LARGE_TEXT_MIN_CONTRAST_RATIO,
    min_lab_distance: float = 24.0,
) -> dict[str, Any]:
    """Draw centered games text with readable ink and trace metadata."""

    effective_required = _is_required_game_role(str(role), bool(required))
    fill = normalize_rgb(fill_rgb)
    stroke = normalize_rgb(stroke_rgb) if stroke_rgb is not None else None
    text_style = resolve_game_text_style(
        role=str(role),
        text=str(text),
        fill_rgb=fill,
        stroke_rgb=stroke,
        surface_rgbs=surface_rgbs,
        preferred_rgbs=preferred_rgbs,
        candidate_rgbs=candidate_rgbs,
        instance_seed=instance_seed,
        namespace=str(namespace or f"games.text.{role}"),
        xy_key=(float(center[0]), float(center[1])),
        required=effective_required,
        min_contrast_ratio=float(min_contrast_ratio),
        min_lab_distance=float(min_lab_distance),
    )
    metadata = dict(extra_metadata or {})
    metadata.setdefault("source", "games_text_helper")
    return draw_centered_readable_text(
        draw,
        center=(float(center[0]), float(center[1])),
        text=str(text),
        font=font,
        style=text_style,
        stroke_width=max(1, int(stroke_width)) if effective_required else max(0, int(stroke_width)),
        extra_metadata=metadata,
    )


def draw_centered_game_text(
    draw: ImageDraw.ImageDraw,
    *,
    text: str,
    center: Tuple[float, float],
    font: ImageFont.ImageFont,
    fill: Sequence[int],
    stroke_fill: Sequence[int],
    stroke_width: int = 1,
    role: str = "readout",
    required: bool = False,
    surface_rgbs: Sequence[Sequence[int]] | None = None,
    preferred_rgbs: Sequence[Sequence[int]] | None = None,
    candidate_rgbs: Sequence[Sequence[int]] | None = None,
    instance_seed: int | None = None,
    namespace: str | None = None,
) -> list[float]:
    """Draw centered games text and return the rendered text bbox."""

    record = draw_centered_game_text_traced(
        draw,
        center=(float(center[0]), float(center[1])),
        text=str(text),
        font=font,
        fill_rgb=fill,
        stroke_rgb=stroke_fill,
        stroke_width=max(0, int(stroke_width)),
        role=str(role),
        required=bool(required),
        surface_rgbs=surface_rgbs,
        preferred_rgbs=preferred_rgbs,
        candidate_rgbs=candidate_rgbs,
        instance_seed=instance_seed,
        namespace=namespace,
    )
    bbox = record.get("bbox_px")
    if isinstance(bbox, list) and len(bbox) == 4:
        return [round(float(value), 3) for value in bbox]
    measured = text_bbox(draw, (0.0, 0.0), str(text), font, stroke_width=max(0, int(stroke_width)))
    return [round(float(value), 3) for value in measured]


__all__ = [
    "draw_centered_game_text",
    "draw_centered_game_text_traced",
    "draw_game_text_traced",
    "resolve_game_text_style",
]
