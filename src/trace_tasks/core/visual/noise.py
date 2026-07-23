"""Deterministic post-image noise augmentation helpers for Trace tasks.

This module applies coordinate-preserving post-render noise edits with
deterministic sampling. Scenes provide their own default config; this
module only merges defaults with per-task/per-instance override keys.
"""

from __future__ import annotations

import io
import random
from copy import deepcopy
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from ..seed import spawn_rng
from .ranges import normalize_non_negative_int_range


TRACE_DEFAULT_NOISE_EDIT_TYPES: Tuple[str, ...] = (
    "blur",
    "downsample",
    "directional_blur",
    "edge_soften",
    "unsharp_mask",
    "jpeg",
    "posterize_quantization",
    "noise",
    "gaussian_noise",
    "poisson_noise",
    "salt_pepper_noise",
    "speckle_noise",
    "dust_speckle",
    "brightness_contrast",
    "exposure_shift",
    "gamma_shift",
    "low_contrast_fade",
    "uneven_illumination",
    "screen_or_paper_texture",
    "scanline_texture",
    "subpixel_display_texture",
    "neutral_moire_texture",
    "ink_bleed",
    "local_contrast_jitter",
    "vignette",
)

_ALLOWED_EDIT_TYPES = set(TRACE_DEFAULT_NOISE_EDIT_TYPES)

TRACE_DEFAULT_NOISE_VALUE_RANGES: Dict[str, Dict[str, Tuple[float, float]]] = {
    "blur": {"radius": (0.2, 0.6)},
    "directional_blur": {"length": (3.0, 7.0), "amount": (0.18, 0.42)},
    "downsample": {"scale": (0.85, 0.95)},
    "edge_soften": {"amount": (0.10, 0.32)},
    "unsharp_mask": {"radius": (0.4, 1.2), "percent": (60.0, 140.0), "threshold": (1.0, 5.0)},
    "jpeg": {"quality": (70.0, 90.0)},
    "posterize_quantization": {"levels": (40.0, 80.0), "amount": (0.10, 0.28)},
    "noise": {"alpha": (0.03, 0.08)},
    "gaussian_noise": {"sigma": (3.0, 12.0)},
    "poisson_noise": {"peak": (500.0, 1200.0)},
    "salt_pepper_noise": {"amount": (0.0015, 0.006)},
    "speckle_noise": {"sigma": (0.01, 0.035)},
    "dust_speckle": {"amount": (0.001, 0.005), "alpha": (0.25, 0.55)},
    "brightness_contrast": {"delta": (0.02, 0.07)},
    "exposure_shift": {"delta": (0.02, 0.08)},
    "gamma_shift": {"gamma": (0.88, 1.14)},
    "low_contrast_fade": {"contrast_drop": (0.06, 0.18), "fade_alpha": (0.03, 0.12)},
    "uneven_illumination": {"strength": (0.04, 0.14)},
    "screen_or_paper_texture": {"alpha": (0.015, 0.05), "grain_sigma": (4.0, 12.0)},
    "scanline_texture": {"alpha": (0.01, 0.03), "period": (3.0, 7.0)},
    "subpixel_display_texture": {"alpha": (0.003, 0.012), "period": (3.0, 5.0)},
    "neutral_moire_texture": {"alpha": (0.004, 0.018), "period": (10.0, 26.0)},
    "ink_bleed": {"amount": (0.05, 0.18)},
    "local_contrast_jitter": {"strength": (0.02, 0.08), "grid_size": (6.0, 14.0)},
    "vignette": {"strength": (0.06, 0.20)},
}

_DEFAULT_NOISE_CONFIG: Dict[str, Any] = {
    "apply_prob": 0.0,
    "edit_types": list(TRACE_DEFAULT_NOISE_EDIT_TYPES),
    "edit_count_range": [1, 2],
    "value_ranges": deepcopy(TRACE_DEFAULT_NOISE_VALUE_RANGES),
}


def _clamp_prob(value: Any, default: float) -> float:
    """Clamp a probability-like value to [0, 1] with fallback."""
    try:
        prob = float(value)
    except Exception:
        prob = float(default)
    return max(0.0, min(1.0, prob))


def _normalize_value_ranges(raw: Any, fallback: Mapping[str, Mapping[str, Tuple[float, float]]]) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """Normalize per-edit parameter ranges from config-like mappings."""
    out: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for edit_type, param_map in fallback.items():
        edit_key = str(edit_type).strip().lower()
        if edit_key not in _ALLOWED_EDIT_TYPES or not isinstance(param_map, Mapping):
            continue
        clean_params: Dict[str, Tuple[float, float]] = {}
        for param_name, bounds in param_map.items():
            if not isinstance(bounds, (list, tuple)) or len(bounds) < 2:
                continue
            lo = float(bounds[0])
            hi = float(bounds[1])
            clean_params[str(param_name)] = (min(lo, hi), max(lo, hi))
        if clean_params:
            out[edit_key] = clean_params
    if not isinstance(raw, Mapping):
        return out
    for edit_type, param_map in raw.items():
        edit_key = str(edit_type).strip().lower()
        if edit_key not in _ALLOWED_EDIT_TYPES or not isinstance(param_map, Mapping):
            continue
        clean_params: Dict[str, Tuple[float, float]] = {}
        for param_name, bounds in param_map.items():
            if not isinstance(bounds, (list, tuple)) or len(bounds) < 2:
                continue
            lo = float(bounds[0])
            hi = float(bounds[1])
            clean_params[str(param_name)] = (min(lo, hi), max(lo, hi))
        if clean_params:
            out[edit_key] = clean_params
    return out


def _odd_int(value: float, *, minimum: int = 3) -> int:
    out = max(int(minimum), int(round(float(value))))
    if out % 2 == 0:
        out += 1
    return out


def _normalize_edit_types(
    raw_types: Any,
    *,
    value_ranges: Mapping[str, Mapping[str, Tuple[float, float]]],
) -> List[str]:
    """Normalize edit-type list and keep only edits with declared ranges."""
    if not isinstance(raw_types, (list, tuple)):
        return []
    edit_types = [
        str(item).strip().lower()
        for item in raw_types
        if str(item).strip().lower() in _ALLOWED_EDIT_TYPES
    ]
    return [edit_type for edit_type in edit_types if edit_type in value_ranges]


def _sample_edit_params(
    edit_type: str,
    rng: random.Random,
    value_ranges: Mapping[str, Mapping[str, Tuple[float, float]]],
) -> Dict[str, float]:
    """Sample parameter values for one edit type from configured ranges."""
    params: Dict[str, float] = {}
    for param_name, (lo, hi) in value_ranges.get(edit_type, {}).items():
        value = float(rng.uniform(float(lo), float(hi)))
        if str(param_name) == "quality":
            params[str(param_name)] = float(int(round(value)))
        elif str(param_name) in {"grid_size", "levels", "period", "percent", "threshold"}:
            params[str(param_name)] = float(max(1, int(round(value))))
        elif str(edit_type) == "directional_blur" and str(param_name) == "length":
            params[str(param_name)] = float(_odd_int(value, minimum=3))
        else:
            params[str(param_name)] = float(value)
    if str(edit_type) == "directional_blur":
        params["angle_degrees"] = float(rng.choice([0, 90]))
    if str(edit_type) in {"scanline_texture", "subpixel_display_texture", "uneven_illumination"}:
        params["axis_code"] = float(rng.choice([0, 1]))
    if str(edit_type) == "neutral_moire_texture":
        params["axis_code"] = float(rng.choice([0, 1, 2]))
        params["phase"] = float(rng.uniform(0.0, 6.283185307179586))
    if str(edit_type) in {"exposure_shift", "uneven_illumination"}:
        params["polarity"] = float(rng.choice([-1, 1]))
    if str(edit_type) == "exposure_shift":
        params["factor"] = float(1.0 + (params["polarity"] * float(params.get("delta", 0.03))))
    return params


def _apply_noise_blend(image: Image.Image, rng: random.Random, alpha: float) -> Image.Image:
    """Blend grayscale random noise with an image using alpha mixing."""
    base = image.convert("RGB")
    width, height = base.size
    noise = Image.new("RGB", (width, height))
    pixels = noise.load()
    for y in range(height):
        for x in range(width):
            value = int(rng.randint(0, 255))
            pixels[x, y] = (value, value, value)
    return Image.blend(base, noise, max(0.0, min(1.0, float(alpha))))


def _np_rng(rng: random.Random) -> np.random.Generator:
    return np.random.default_rng(rng.getrandbits(63))


def _apply_gaussian_noise(image: Image.Image, rng: random.Random, sigma: float) -> Image.Image:
    np_rng = _np_rng(rng)
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    noisy = arr + np_rng.normal(0.0, float(sigma), size=arr.shape)
    return Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8))


def _apply_poisson_noise(image: Image.Image, rng: random.Random, peak: float) -> Image.Image:
    np_rng = _np_rng(rng)
    peak_value = max(1.0, float(peak))
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    noisy = np_rng.poisson(np.clip(arr, 0.0, 1.0) * peak_value) / peak_value
    return Image.fromarray(np.clip(noisy * 255.0, 0, 255).astype(np.uint8))


def _apply_speckle_noise(image: Image.Image, rng: random.Random, sigma: float) -> Image.Image:
    np_rng = _np_rng(rng)
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    multiplier = 1.0 + np_rng.normal(0.0, float(sigma), size=arr.shape[:2] + (1,))
    return Image.fromarray(np.clip(arr * multiplier, 0, 255).astype(np.uint8))


def _apply_dust_speckle(image: Image.Image, rng: random.Random, *, amount: float, alpha: float) -> Image.Image:
    np_rng = _np_rng(rng)
    arr = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    height, width = arr.shape[:2]
    mask = np_rng.random((height, width)) < max(0.0, min(1.0, float(amount)))
    light_dust = np_rng.random((height, width)) < 0.55
    target = np.zeros_like(arr)
    target[light_dust] = 255.0
    blend = max(0.0, min(1.0, float(alpha)))
    arr[mask] = (arr[mask] * (1.0 - blend)) + (target[mask] * blend)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _apply_brightness_contrast(image: Image.Image, rng: random.Random, delta: float) -> Image.Image:
    brightness = 1.0 + rng.choice([-1.0, 1.0]) * float(rng.uniform(0.0, float(delta)))
    contrast = 1.0 + rng.choice([-1.0, 1.0]) * float(rng.uniform(0.0, float(delta)))
    out = ImageEnhance.Brightness(image.convert("RGB")).enhance(max(0.5, min(1.5, brightness)))
    return ImageEnhance.Contrast(out).enhance(max(0.5, min(1.5, contrast)))


def _apply_gamma_shift(image: Image.Image, gamma: float) -> Image.Image:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    luma = (0.299 * arr[:, :, 0]) + (0.587 * arr[:, :, 1]) + (0.114 * arr[:, :, 2])
    gamma_value = max(0.55, min(1.65, float(gamma)))
    adjusted_luma = (np.clip(luma / 255.0, 0.0, 1.0) ** gamma_value) * 255.0
    delta = (adjusted_luma - luma)[:, :, None]
    return Image.fromarray(np.clip(arr + delta, 0, 255).astype(np.uint8))


def _apply_exposure_shift(image: Image.Image, factor: float) -> Image.Image:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    return Image.fromarray(np.clip(arr * max(0.5, min(1.5, float(factor))), 0, 255).astype(np.uint8))


def _apply_uneven_illumination(image: Image.Image, *, strength: float, axis_code: float, polarity: float) -> Image.Image:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    height, width = arr.shape[:2]
    if int(round(float(axis_code))) == 1:
        ramp = np.linspace(-1.0, 1.0, height, dtype=np.float32).reshape(height, 1, 1)
    else:
        ramp = np.linspace(-1.0, 1.0, width, dtype=np.float32).reshape(1, width, 1)
    multiplier = 1.0 + (np.sign(float(polarity) or 1.0) * max(0.0, min(1.0, float(strength))) * ramp)
    return Image.fromarray(np.clip(arr * multiplier, 0, 255).astype(np.uint8))


def _apply_directional_blur(image: Image.Image, *, length: float, amount: float, angle_degrees: float) -> Image.Image:
    base = image.convert("RGB")
    kernel_size = _odd_int(float(length), minimum=3)
    pad = kernel_size // 2
    arr = np.asarray(base, dtype=np.float32)
    if int(round(float(angle_degrees))) % 180 == 90:
        padded = np.pad(arr, ((pad, pad), (0, 0), (0, 0)), mode="edge")
        blurred = sum(padded[offset : offset + arr.shape[0], :, :] for offset in range(kernel_size)) / float(kernel_size)
    else:
        padded = np.pad(arr, ((0, 0), (pad, pad), (0, 0)), mode="edge")
        blurred = sum(padded[:, offset : offset + arr.shape[1], :] for offset in range(kernel_size)) / float(kernel_size)
    alpha = max(0.0, min(1.0, float(amount)))
    out = (arr * (1.0 - alpha)) + (blurred * alpha)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def _apply_edge_soften(image: Image.Image, amount: float) -> Image.Image:
    base = image.convert("RGB")
    softened = base.filter(ImageFilter.SMOOTH_MORE)
    return Image.blend(base, softened, max(0.0, min(1.0, float(amount))))


def _apply_unsharp_mask(image: Image.Image, *, radius: float, percent: float, threshold: float) -> Image.Image:
    return image.convert("RGB").filter(
        ImageFilter.UnsharpMask(
            radius=max(0.1, float(radius)),
            percent=max(1, int(round(float(percent)))),
            threshold=max(0, int(round(float(threshold)))),
        )
    )


def _apply_ink_bleed(image: Image.Image, amount: float) -> Image.Image:
    base = image.convert("RGB")
    arr = np.asarray(base, dtype=np.float32)
    gray = np.asarray(base.convert("L"), dtype=np.float32)
    spread = np.asarray(base.convert("L").filter(ImageFilter.MinFilter(3)), dtype=np.float32)
    dark_delta = np.clip((gray - spread) / 255.0, 0.0, 1.0)
    factor = 1.0 - (max(0.0, min(1.0, float(amount))) * dark_delta[:, :, None])
    return Image.fromarray(np.clip(arr * factor, 0, 255).astype(np.uint8))


def _apply_salt_pepper_noise(image: Image.Image, rng: random.Random, amount: float) -> Image.Image:
    np_rng = _np_rng(rng)
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    height, width = arr.shape[:2]
    mask = np_rng.random((height, width)) < max(0.0, min(1.0, float(amount)))
    salt = np_rng.random((height, width)) < 0.5
    arr[mask & salt] = 255
    arr[mask & ~salt] = 0
    return Image.fromarray(arr)


def _apply_posterize_quantization(image: Image.Image, *, levels: float, amount: float) -> Image.Image:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    level_count = max(2.0, float(levels))
    luma = (0.299 * arr[:, :, 0]) + (0.587 * arr[:, :, 1]) + (0.114 * arr[:, :, 2])
    step = 255.0 / max(1.0, level_count - 1.0)
    quantized_luma = np.round(luma / step) * step
    delta = (quantized_luma - luma)[:, :, None] * max(0.0, min(1.0, float(amount)))
    return Image.fromarray(np.clip(arr + delta, 0, 255).astype(np.uint8))


def _apply_low_contrast_fade(image: Image.Image, *, contrast_drop: float, fade_alpha: float) -> Image.Image:
    base = image.convert("RGB")
    contrast = max(0.25, min(1.0, 1.0 - float(contrast_drop)))
    low_contrast = ImageEnhance.Contrast(base).enhance(contrast)
    paper = Image.new("RGB", base.size, (246, 246, 238))
    return Image.blend(low_contrast, paper, max(0.0, min(1.0, float(fade_alpha))))


def _apply_scanline_texture(image: Image.Image, *, alpha: float, period: float, axis_code: float) -> Image.Image:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    height, width = arr.shape[:2]
    line_period = max(2, int(round(float(period))))
    if int(round(float(axis_code))) == 1:
        coords = np.arange(width).reshape(1, width)
        mask = ((coords % line_period) == 0).repeat(height, axis=0)
    else:
        coords = np.arange(height).reshape(height, 1)
        mask = ((coords % line_period) == 0).repeat(width, axis=1)
    multiplier = np.ones((height, width), dtype=np.float32)
    multiplier[mask] -= max(0.0, min(0.25, float(alpha)))
    return Image.fromarray(np.clip(arr * multiplier[:, :, None], 0, 255).astype(np.uint8))


def _apply_subpixel_display_texture(image: Image.Image, *, alpha: float, period: float, axis_code: float) -> Image.Image:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    height, width = arr.shape[:2]
    line_period = max(3, int(round(float(period))))
    if int(round(float(axis_code))) == 1:
        coords = np.arange(width, dtype=np.float32).reshape(1, width)
        pattern = ((coords % line_period) / max(1.0, float(line_period - 1)) - 0.5).repeat(height, axis=0)
    else:
        coords = np.arange(height, dtype=np.float32).reshape(height, 1)
        pattern = ((coords % line_period) / max(1.0, float(line_period - 1)) - 0.5).repeat(width, axis=1)
    multiplier = 1.0 + (pattern * 2.0 * max(0.0, min(0.05, float(alpha))))
    return Image.fromarray(np.clip(arr * multiplier[:, :, None], 0, 255).astype(np.uint8))


def _apply_neutral_moire_texture(
    image: Image.Image,
    *,
    alpha: float,
    period: float,
    axis_code: float,
    phase: float,
) -> Image.Image:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    height, width = arr.shape[:2]
    yy = np.arange(height, dtype=np.float32).reshape(height, 1)
    xx = np.arange(width, dtype=np.float32).reshape(1, width)
    wave_period = max(4.0, float(period))
    if int(round(float(axis_code))) == 1:
        pattern = np.sin(((yy / wave_period) * 6.283185307179586) + float(phase))
    elif int(round(float(axis_code))) == 2:
        pattern = np.sin((((xx + yy) / wave_period) * 6.283185307179586) + float(phase))
    else:
        pattern = np.sin(((xx / wave_period) * 6.283185307179586) + float(phase))
    multiplier = 1.0 + (pattern[:, :, None] * max(0.0, min(0.05, float(alpha))))
    return Image.fromarray(np.clip(arr * multiplier, 0, 255).astype(np.uint8))


def _apply_screen_or_paper_texture(image: Image.Image, rng: random.Random, *, alpha: float, grain_sigma: float) -> Image.Image:
    base = image.convert("RGB")
    width, height = base.size
    np_rng = _np_rng(rng)
    arr = np.asarray(base, dtype=np.float32)
    grain = np_rng.normal(0.0, float(grain_sigma), size=(height, width, 1))
    y = np.arange(height, dtype=np.float32).reshape(height, 1, 1)
    stripe_period = float(np_rng.integers(4, 9))
    stripes = np.sin((y / stripe_period) * np.pi * 2.0) * float(grain_sigma) * 0.45
    textured = arr + (grain + stripes) * max(0.0, min(1.0, float(alpha))) * 4.0
    return Image.fromarray(np.clip(textured, 0, 255).astype(np.uint8))


def _apply_local_contrast_jitter(
    image: Image.Image,
    rng: random.Random,
    *,
    strength: float,
    grid_size: float,
) -> Image.Image:
    base = image.convert("RGB")
    arr = np.asarray(base, dtype=np.float32)
    height, width = arr.shape[:2]
    cells = max(2, int(round(float(grid_size))))
    np_rng = _np_rng(rng)
    low_res = 1.0 + np_rng.uniform(-float(strength), float(strength), size=(cells, cells)).astype(np.float32)
    factor_img = Image.fromarray(np.clip(low_res * 127.5, 0, 255).astype(np.uint8))
    factor_img = factor_img.resize((width, height), resample=Image.BILINEAR)
    factor = (np.asarray(factor_img, dtype=np.float32) / 127.5)[:, :, None]
    out = ((arr - 127.5) * factor) + 127.5
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def _apply_vignette(image: Image.Image, strength: float) -> Image.Image:
    base = image.convert("RGB")
    arr = np.asarray(base, dtype=np.float32)
    height, width = arr.shape[:2]
    yy = np.linspace(-1.0, 1.0, height, dtype=np.float32).reshape(height, 1)
    xx = np.linspace(-1.0, 1.0, width, dtype=np.float32).reshape(1, width)
    distance = np.clip(np.sqrt((xx * xx) + (yy * yy)) / np.sqrt(2.0), 0.0, 1.0)
    mask = 1.0 - (max(0.0, min(1.0, float(strength))) * (distance ** 1.8))
    out = arr * mask[:, :, None]
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def _apply_single_edit(image: Image.Image, edit_type: str, params: Mapping[str, float], rng: random.Random) -> Image.Image:
    """Apply one supported post-image edit and return the transformed image."""
    base = image.convert("RGB")
    if edit_type == "blur":
        radius = float(params.get("radius", 0.4))
        return base.filter(ImageFilter.GaussianBlur(radius=radius))
    if edit_type == "directional_blur":
        return _apply_directional_blur(
            base,
            length=float(params.get("length", 3.0)),
            amount=float(params.get("amount", 0.25)),
            angle_degrees=float(params.get("angle_degrees", 0.0)),
        )
    if edit_type == "downsample":
        scale = max(0.05, min(1.0, float(params.get("scale", 0.9))))
        width, height = base.size
        down_w = max(1, int(round(width * scale)))
        down_h = max(1, int(round(height * scale)))
        low = base.resize((down_w, down_h), resample=Image.BILINEAR)
        return low.resize((width, height), resample=Image.NEAREST)
    if edit_type == "edge_soften":
        return _apply_edge_soften(base, amount=float(params.get("amount", 0.18)))
    if edit_type == "unsharp_mask":
        return _apply_unsharp_mask(
            base,
            radius=float(params.get("radius", 0.8)),
            percent=float(params.get("percent", 100.0)),
            threshold=float(params.get("threshold", 3.0)),
        )
    if edit_type == "jpeg":
        quality = int(round(float(params.get("quality", 80.0))))
        quality = max(5, min(95, quality))
        buf = io.BytesIO()
        base.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    if edit_type == "posterize_quantization":
        return _apply_posterize_quantization(
            base,
            levels=float(params.get("levels", 64.0)),
            amount=float(params.get("amount", 0.16)),
        )
    if edit_type == "noise":
        alpha = float(params.get("alpha", 0.05))
        return _apply_noise_blend(base, rng, alpha)
    if edit_type == "gaussian_noise":
        return _apply_gaussian_noise(base, rng, sigma=float(params.get("sigma", 6.0)))
    if edit_type == "poisson_noise":
        return _apply_poisson_noise(base, rng, peak=float(params.get("peak", 900.0)))
    if edit_type == "salt_pepper_noise":
        return _apply_salt_pepper_noise(base, rng, amount=float(params.get("amount", 0.004)))
    if edit_type == "speckle_noise":
        return _apply_speckle_noise(base, rng, sigma=float(params.get("sigma", 0.016)))
    if edit_type == "dust_speckle":
        return _apply_dust_speckle(
            base,
            rng,
            amount=float(params.get("amount", 0.002)),
            alpha=float(params.get("alpha", 0.35)),
        )
    if edit_type == "brightness_contrast":
        return _apply_brightness_contrast(base, rng, delta=float(params.get("delta", 0.04)))
    if edit_type == "exposure_shift":
        return _apply_exposure_shift(base, factor=float(params.get("factor", 1.0)))
    if edit_type == "gamma_shift":
        return _apply_gamma_shift(base, gamma=float(params.get("gamma", 1.0)))
    if edit_type == "low_contrast_fade":
        return _apply_low_contrast_fade(
            base,
            contrast_drop=float(params.get("contrast_drop", 0.08)),
            fade_alpha=float(params.get("fade_alpha", 0.04)),
        )
    if edit_type == "uneven_illumination":
        return _apply_uneven_illumination(
            base,
            strength=float(params.get("strength", 0.06)),
            axis_code=float(params.get("axis_code", 0.0)),
            polarity=float(params.get("polarity", 1.0)),
        )
    if edit_type == "screen_or_paper_texture":
        return _apply_screen_or_paper_texture(
            base,
            rng,
            alpha=float(params.get("alpha", 0.02)),
            grain_sigma=float(params.get("grain_sigma", 6.0)),
        )
    if edit_type == "scanline_texture":
        return _apply_scanline_texture(
            base,
            alpha=float(params.get("alpha", 0.012)),
            period=float(params.get("period", 4.0)),
            axis_code=float(params.get("axis_code", 0.0)),
        )
    if edit_type == "subpixel_display_texture":
        return _apply_subpixel_display_texture(
            base,
            alpha=float(params.get("alpha", 0.006)),
            period=float(params.get("period", 3.0)),
            axis_code=float(params.get("axis_code", 0.0)),
        )
    if edit_type == "neutral_moire_texture":
        return _apply_neutral_moire_texture(
            base,
            alpha=float(params.get("alpha", 0.008)),
            period=float(params.get("period", 16.0)),
            axis_code=float(params.get("axis_code", 0.0)),
            phase=float(params.get("phase", 0.0)),
        )
    if edit_type == "ink_bleed":
        return _apply_ink_bleed(base, amount=float(params.get("amount", 0.08)))
    if edit_type == "local_contrast_jitter":
        return _apply_local_contrast_jitter(
            base,
            rng,
            strength=float(params.get("strength", 0.04)),
            grid_size=float(params.get("grid_size", 8.0)),
        )
    if edit_type == "vignette":
        return _apply_vignette(base, strength=float(params.get("strength", 0.1)))
    return base


def _serialize_edits(edits: Sequence[Tuple[str, Mapping[str, float]]]) -> List[Dict[str, Any]]:
    """Serialize sampled edit operations to trace-friendly metadata rows."""
    rows: List[Dict[str, Any]] = []
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
    return rows


def _normalize_default_config(default_config: Mapping[str, Any] | None) -> Dict[str, Any]:
    """Normalize caller-provided scene defaults against global fallback."""
    base = deepcopy(_DEFAULT_NOISE_CONFIG)
    if not isinstance(default_config, Mapping):
        return base

    base_apply_prob = float(base["apply_prob"])
    base["apply_prob"] = _clamp_prob(default_config.get("apply_prob", base_apply_prob), base_apply_prob)

    value_ranges = _normalize_value_ranges(default_config.get("value_ranges", base["value_ranges"]), fallback=base["value_ranges"])
    base["value_ranges"] = value_ranges

    base["edit_types"] = _normalize_edit_types(
        default_config.get("edit_types", base["edit_types"]),
        value_ranges=value_ranges,
    )

    edit_count_range = normalize_non_negative_int_range(
        default_config.get("edit_count_range", base["edit_count_range"]),
        fallback_min=int(base["edit_count_range"][0]),
        fallback_max=int(base["edit_count_range"][1]),
    )
    base["edit_count_range"] = [int(edit_count_range[0]), int(edit_count_range[1])]
    return base


def _resolve_noise_overrides(params: Mapping[str, Any]) -> Dict[str, Any]:
    """Collect noise overrides from nested visual config."""
    merged: Dict[str, Any] = {}
    visual = params.get("visual")
    if isinstance(visual, Mapping):
        noise_cfg = visual.get("noise")
        if isinstance(noise_cfg, Mapping):
            merged.update(dict(noise_cfg))
    return merged


def _resolve_post_noise_config(params: Mapping[str, Any], *, default_config: Mapping[str, Any] | None) -> Dict[str, Any]:
    """Merge default and override noise config into a normalized payload."""
    base = _normalize_default_config(default_config)
    overrides = _resolve_noise_overrides(params)

    apply_prob = _clamp_prob(overrides.get("apply_prob", base.get("apply_prob", 0.0)), float(base.get("apply_prob", 0.0)))
    value_ranges = _normalize_value_ranges(
        overrides.get("value_ranges", base.get("value_ranges", {})),
        fallback=base.get("value_ranges", {}),
    )

    edit_types = _normalize_edit_types(
        overrides.get("edit_types", base.get("edit_types", [])),
        value_ranges=value_ranges,
    )

    edit_count_range = normalize_non_negative_int_range(
        overrides.get("edit_count_range", base.get("edit_count_range", [1, 1])),
        fallback_min=int(base.get("edit_count_range", [1, 1])[0]),
        fallback_max=int(base.get("edit_count_range", [1, 1])[1]),
    )

    return {
        "apply_prob": float(apply_prob),
        "edit_types": list(edit_types),
        "edit_count_range": [int(edit_count_range[0]), int(edit_count_range[1])],
        "value_ranges": value_ranges,
    }


def _sample_noise_edits(
    rng: random.Random,
    *,
    edit_types: Sequence[str],
    value_ranges: Mapping[str, Mapping[str, Tuple[float, float]]],
    edit_count_range: Sequence[int],
) -> List[Tuple[str, Dict[str, float]]]:
    """Sample edit operations and parameters for one image augmentation pass."""
    if not edit_types:
        return []
    lo = max(0, int(edit_count_range[0]))
    hi = max(lo, int(edit_count_range[1]))
    if hi <= 0:
        return []
    count = int(rng.randint(lo, hi))
    if count <= 0:
        return []
    chosen = rng.sample(list(edit_types), k=min(int(count), len(edit_types)))
    return [(edit_type, _sample_edit_params(edit_type, rng, value_ranges)) for edit_type in chosen]


def apply_post_image_noise(
    image: Image.Image,
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    default_config: Mapping[str, Any] | None = None,
) -> Tuple[Image.Image, Dict[str, Any]]:
    """Apply deterministic post-composite noise edits and return trace metadata."""
    cfg = _resolve_post_noise_config(params, default_config=default_config)
    enabled = bool(float(cfg["apply_prob"]) > 0.0 and cfg["edit_types"] and int(cfg["edit_count_range"][1]) > 0)

    meta: Dict[str, Any] = {
        "enabled": bool(enabled),
        "applied": False,
        "apply_prob": float(cfg["apply_prob"]),
        "edit_types": list(cfg["edit_types"]),
        "edit_count_range": [int(cfg["edit_count_range"][0]), int(cfg["edit_count_range"][1])],
        "edits": [],
    }

    if not enabled:
        return image, meta

    gate_rng = spawn_rng(instance_seed, "visual.noise_gate")
    if gate_rng.random() > float(cfg["apply_prob"]):
        return image, meta

    select_rng = spawn_rng(instance_seed, "visual.noise_select")
    edits = _sample_noise_edits(
        select_rng,
        edit_types=cfg["edit_types"],
        value_ranges=cfg["value_ranges"],
        edit_count_range=cfg["edit_count_range"],
    )
    if not edits:
        return image, meta

    apply_rng = spawn_rng(instance_seed, "visual.noise_apply")
    output = image.convert("RGB")
    for edit_type, edit_params in edits:
        output = _apply_single_edit(output, edit_type, edit_params, apply_rng)

    meta["applied"] = True
    meta["edits"] = _serialize_edits(edits)
    return output, meta
