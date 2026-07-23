"""Shared canvas profiles for illustration scenes.

Canvas profile is render metadata, not a task/query axis. Explicit
``canvas_width`` and ``canvas_height`` params remain a narrow override for
tests and targeted debugging.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Mapping, Sequence, Tuple

from PIL import Image

from ....core.seed import spawn_rng
from ....core.sampling import uniform_choice
from ...shared.config_defaults import group_default


CANVAS_PROFILE_LANDSCAPE = "landscape"
CANVAS_PROFILE_SQUARE = "square"
CANVAS_PROFILE_PORTRAIT = "portrait"
CANVAS_PROFILE_CUSTOM = "custom"
CANVAS_PROFILE_SUPPORT: Tuple[str, ...] = (
    CANVAS_PROFILE_LANDSCAPE,
    CANVAS_PROFILE_SQUARE,
    CANVAS_PROFILE_PORTRAIT,
)
MAX_RECONSTRUCTION_OUTPUT_PIXELS = 1_280_000


@dataclass(frozen=True)
class CanvasProfile:
    """Resolved illustration canvas profile."""

    profile_id: str
    width: int
    height: int
    probabilities: Dict[str, float]

    @property
    def size(self) -> Tuple[int, int]:
        return (int(self.width), int(self.height))

    def trace(self) -> Dict[str, Any]:
        return {
            "canvas_profile": str(self.profile_id),
            "canvas_profile_size": [int(self.width), int(self.height)],
            "canvas_profile_probabilities": dict(self.probabilities),
        }


PROFILE_SIZES: Dict[str, Tuple[int, int]] = {
    CANVAS_PROFILE_LANDSCAPE: (1200, 800),
    CANVAS_PROFILE_SQUARE: (960, 960),
    CANVAS_PROFILE_PORTRAIT: (800, 1200),
}


def _uniform_profile_probability_map(values: Sequence[str]) -> Dict[str, float]:
    support = tuple(str(value) for value in values)
    if not support:
        return {}
    probability = 1.0 / float(len(support))
    return {str(value): float(probability) for value in support}


def _support_from(params: Mapping[str, Any], defaults: Mapping[str, Any]) -> Tuple[str, ...]:
    raw = params.get("canvas_profile_support", group_default(defaults, "canvas_profile_support", CANVAS_PROFILE_SUPPORT))
    if isinstance(raw, str):
        values = (raw,)
    elif isinstance(raw, Sequence):
        values = tuple(raw)
    else:
        values = tuple(CANVAS_PROFILE_SUPPORT)
    support = tuple(dict.fromkeys(str(value) for value in values if str(value) in set(CANVAS_PROFILE_SUPPORT)))
    if not support:
        raise ValueError("canvas_profile_support must include at least one supported illustration canvas profile")
    return support


def resolve_canvas_profile(
    *,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    fallback_width: int,
    fallback_height: int,
    instance_seed: int | None = None,
    namespace: str = "illustrations:canvas_profile",
) -> CanvasProfile:
    """Resolve a render-only canvas profile for one illustration instance."""

    if "canvas_width" in params or "canvas_height" in params:
        width = int(params.get("canvas_width", fallback_width))
        height = int(params.get("canvas_height", fallback_height))
        return CanvasProfile(
            profile_id=CANVAS_PROFILE_CUSTOM,
            width=int(width),
            height=int(height),
            probabilities={CANVAS_PROFILE_CUSTOM: 1.0},
        )

    support = _support_from(params, defaults)
    explicit = params.get("canvas_profile", group_default(defaults, "canvas_profile", None))
    if explicit is not None:
        profile_id = str(explicit)
        if profile_id not in set(support):
            raise ValueError(f"canvas_profile must be one of {support}")
        width, height = PROFILE_SIZES[profile_id]
        return CanvasProfile(
            profile_id=profile_id,
            width=int(width),
            height=int(height),
            probabilities={profile_id: 1.0},
        )

    if instance_seed is not None:
        rng = spawn_rng(int(instance_seed), str(namespace))
        profile_id = str(uniform_choice(rng, support, sort_keys=False))
    else:
        profile_id = str(support[0])
    width, height = PROFILE_SIZES[profile_id]
    return CanvasProfile(
        profile_id=profile_id,
        width=int(width),
        height=int(height),
        probabilities=_uniform_profile_probability_map(support),
    )


def resolve_profile_render_params(
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    *,
    prefix: str,
    fallback_width: int,
    fallback_height: int,
    fallback_scale: int,
    instance_seed: int | None = None,
    namespace: str = "illustrations:canvas_profile",
) -> Dict[str, Any]:
    """Resolve canvas profile plus render scale for a scene render."""

    profile = resolve_canvas_profile(
        params=params,
        defaults=render_defaults,
        fallback_width=int(fallback_width),
        fallback_height=int(fallback_height),
        instance_seed=instance_seed,
        namespace=namespace,
    )
    return {
        "canvas_width": int(profile.width),
        "canvas_height": int(profile.height),
        "render_scale": int(params.get("render_scale", group_default(render_defaults, f"{prefix}_render_scale", int(fallback_scale)))),
        **profile.trace(),
    }


def resolve_reconstruction_source_profile(
    *,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    fallback_source_width: int,
    fallback_source_height: int,
    instance_seed: int,
    namespace: str,
) -> CanvasProfile:
    """Resolve the source-scene profile for visual reconstruction tasks."""

    if "source_width" in params or "source_height" in params:
        width = int(params.get("source_width", fallback_source_width))
        height = int(params.get("source_height", fallback_source_height))
        return CanvasProfile(
            profile_id=CANVAS_PROFILE_CUSTOM,
            width=int(width),
            height=int(height),
            probabilities={CANVAS_PROFILE_CUSTOM: 1.0},
        )
    return resolve_canvas_profile(
        params=params,
        defaults=defaults,
        fallback_width=int(fallback_source_width),
        fallback_height=int(fallback_source_height),
        instance_seed=int(instance_seed),
        namespace=str(namespace),
    )


def reconstruction_grid_for_profile(profile: CanvasProfile | Mapping[str, Any] | str) -> Tuple[int, int]:
    """Return rows/cols whose cells are square for the selected source profile."""

    if isinstance(profile, CanvasProfile):
        if profile.profile_id == CANVAS_PROFILE_CUSTOM:
            return reconstruction_grid_for_size(profile.width, profile.height)
        profile_id = profile.profile_id
    elif isinstance(profile, Mapping):
        profile_id = str(profile.get("canvas_profile", profile.get("profile_id", "")))
    else:
        profile_id = str(profile)
    if profile_id == CANVAS_PROFILE_SQUARE:
        return (3, 3)
    if profile_id == CANVAS_PROFILE_PORTRAIT:
        return (3, 2)
    return (2, 3)


def reconstruction_grid_for_size(width: int, height: int) -> Tuple[int, int]:
    """Return rows/cols whose cells are square-ish for an illustration source size."""

    w = int(width)
    h = int(height)
    if w <= 0 or h <= 0:
        raise ValueError("reconstruction source size must be positive")
    if abs(w - h) <= max(2, int(round(0.01 * max(w, h)))):
        return (3, 3)
    if h > w:
        return (3, 2)
    return (2, 3)


def reconstruction_option_labels(rows: int, cols: int) -> Tuple[str, ...]:
    """Return stable option labels for a reconstruction grid."""

    count = int(rows) * int(cols)
    if count < 1:
        raise ValueError("reconstruction grid must have at least one cell")
    return tuple(chr(ord("A") + index) for index in range(count))


def scale_bbox(box: Sequence[float], *, scale_x: float, scale_y: float) -> list[float]:
    """Scale one bbox by x/y factors."""

    return [
        round(float(box[0]) * float(scale_x), 3),
        round(float(box[1]) * float(scale_y), 3),
        round(float(box[2]) * float(scale_x), 3),
        round(float(box[3]) * float(scale_y), 3),
    ]


def scale_bbox_map(boxes: Mapping[str, Sequence[float]], *, scale_x: float, scale_y: float) -> Dict[str, list[float]]:
    """Scale one label/key to bbox mapping."""

    return {str(key): scale_bbox(value, scale_x=float(scale_x), scale_y=float(scale_y)) for key, value in boxes.items()}


def resize_to_max_pixels(image: Image.Image, *, max_pixels: int = MAX_RECONSTRUCTION_OUTPUT_PIXELS) -> Tuple[Image.Image, float, float]:
    """Downscale an image if needed and return the x/y scale factors."""

    width, height = image.size
    pixels = int(width) * int(height)
    if pixels <= int(max_pixels):
        return image, 1.0, 1.0
    scale = math.sqrt(float(max_pixels) / float(max(1, pixels)))
    new_width = max(1, int(math.floor(float(width) * scale)))
    new_height = max(1, int(math.floor(float(height) * scale)))
    resized = image.resize((int(new_width), int(new_height)), Image.Resampling.LANCZOS)
    return resized, float(new_width) / float(width), float(new_height) / float(height)


__all__ = [
    "CANVAS_PROFILE_CUSTOM",
    "CANVAS_PROFILE_LANDSCAPE",
    "CANVAS_PROFILE_PORTRAIT",
    "CANVAS_PROFILE_SQUARE",
    "CANVAS_PROFILE_SUPPORT",
    "CanvasProfile",
    "MAX_RECONSTRUCTION_OUTPUT_PIXELS",
    "PROFILE_SIZES",
    "reconstruction_grid_for_size",
    "reconstruction_grid_for_profile",
    "reconstruction_option_labels",
    "resize_to_max_pixels",
    "resolve_canvas_profile",
    "resolve_profile_render_params",
    "resolve_reconstruction_source_profile",
    "scale_bbox",
    "scale_bbox_map",
]
