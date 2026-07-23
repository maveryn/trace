"""Per-icon noise helpers for Trace icon tasks.

These helpers sample subtle edits per icon instance, apply them before
compositing, and keep the icon alpha channel stable so bbox annotation stays
semantically grounded.
"""

from __future__ import annotations

import io
import random
from copy import deepcopy
from typing import Any, Dict, Mapping, Sequence, Tuple

from PIL import Image, ImageFilter

from ....core.visual.noise import TRACE_DEFAULT_NOISE_VALUE_RANGES


NoiseEdit = Tuple[str, Dict[str, float]]

_ALLOWED_EDIT_TYPES = {"blur", "downsample", "jpeg", "noise"}


def default_icon_noise_value_ranges() -> Dict[str, Dict[str, Tuple[float, float]]]:
    """Return a deep-copied default per-icon noise value-range mapping."""

    return deepcopy(TRACE_DEFAULT_NOISE_VALUE_RANGES)


def sample_icon_noise_edits(
    rng: random.Random,
    *,
    edit_types: Sequence[str],
    edit_value_ranges: Mapping[str, Mapping[str, Tuple[float, float]]],
    edit_count_range: Sequence[int],
) -> Tuple[NoiseEdit, ...]:
    """Sample per-icon subtle edits deterministically from configured ranges."""

    normalized_edit_types = [
        str(edit_type).strip().lower()
        for edit_type in edit_types
        if str(edit_type).strip().lower() in _ALLOWED_EDIT_TYPES
        and str(edit_type).strip().lower() in edit_value_ranges
    ]
    if not normalized_edit_types:
        return ()
    lo = max(0, int(edit_count_range[0]))
    hi = max(lo, int(edit_count_range[1]))
    if hi <= 0:
        return ()
    edit_count = int(rng.randint(lo, hi))
    if edit_count <= 0:
        return ()
    chosen = rng.sample(normalized_edit_types, k=min(int(edit_count), len(normalized_edit_types)))
    edits = []
    for edit_type in chosen:
        params: Dict[str, float] = {}
        for param_name, bounds in edit_value_ranges.get(str(edit_type), {}).items():
            lo_value = float(bounds[0])
            hi_value = float(bounds[1])
            sampled = float(rng.uniform(min(lo_value, hi_value), max(lo_value, hi_value)))
            if str(param_name) == "quality":
                params[str(param_name)] = float(int(round(sampled)))
            else:
                params[str(param_name)] = float(sampled)
        edits.append((str(edit_type), params))
    return tuple(edits)


def serialize_icon_noise_edits(edits: Sequence[NoiseEdit]) -> Tuple[Dict[str, Any], ...]:
    """Return one trace-friendly serialization of per-icon noise edits."""

    rows = []
    for edit_type, params in edits:
        rows.append(
            {
                "type": str(edit_type),
                "params": {
                    str(key): round(float(value), 6)
                    for key, value in dict(params).items()
                },
            }
        )
    return tuple(rows)


def _apply_noise_blend(image: Image.Image, rng: random.Random, alpha: float) -> Image.Image:
    """Blend grayscale noise into an RGB icon patch."""

    base = image.convert("RGB")
    width, height = base.size
    noise = Image.new("RGB", (width, height))
    pixels = noise.load()
    for y in range(height):
        for x in range(width):
            value = int(rng.randint(0, 255))
            pixels[x, y] = (value, value, value)
    return Image.blend(base, noise, max(0.0, min(1.0, float(alpha))))


def _apply_single_icon_edit(image: Image.Image, edit_type: str, params: Mapping[str, float], rng: random.Random) -> Image.Image:
    """Apply one subtle edit to an RGB icon patch."""

    base = image.convert("RGB")
    if str(edit_type) == "blur":
        return base.filter(ImageFilter.GaussianBlur(radius=float(params.get("radius", 0.4))))
    if str(edit_type) == "downsample":
        scale = max(0.05, min(1.0, float(params.get("scale", 0.9))))
        width, height = base.size
        down_width = max(1, int(round(width * scale)))
        down_height = max(1, int(round(height * scale)))
        low = base.resize((down_width, down_height), resample=Image.Resampling.BILINEAR)
        return low.resize((width, height), resample=Image.Resampling.NEAREST)
    if str(edit_type) == "jpeg":
        quality = max(5, min(95, int(round(float(params.get("quality", 80.0))))))
        buffer = io.BytesIO()
        base.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")
    if str(edit_type) == "noise":
        return _apply_noise_blend(base, rng, float(params.get("alpha", 0.05)))
    return base


def apply_icon_noise_edits_rgba(
    image: Image.Image,
    *,
    edits: Sequence[NoiseEdit],
    rng: random.Random,
) -> Image.Image:
    """Apply subtle RGB edits while preserving the original alpha mask."""

    if not edits:
        return image
    rgba = image.convert("RGBA")
    alpha = rgba.getchannel("A")
    rgb = rgba.convert("RGB")
    edited_rgb = rgb
    for edit_type, params in edits:
        edited_rgb = _apply_single_icon_edit(edited_rgb, str(edit_type), params, rng)
    edited_rgba = edited_rgb.convert("RGBA")
    edited_rgba.putalpha(alpha)
    return edited_rgba


__all__ = [
    "NoiseEdit",
    "apply_icon_noise_edits_rgba",
    "default_icon_noise_value_ranges",
    "sample_icon_noise_edits",
    "serialize_icon_noise_edits",
]
