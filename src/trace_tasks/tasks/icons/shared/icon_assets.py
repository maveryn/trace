"""Shared curated icon asset helpers for Trace icon tasks."""

from __future__ import annotations

from functools import lru_cache
from io import BytesIO
from pathlib import Path
import hashlib
from typing import Sequence, Tuple

import cairosvg
from PIL import Image, ImageOps

from ....resources import resource_path
from .icon_noise import NoiseEdit, apply_icon_noise_edits_rgba
from .icon_transform import IDENTITY_TRANSFORM_ID, apply_canonical_transform


_ASSET_ROOT = resource_path("assets", "icons")
_SVG_ROOT = _ASSET_ROOT / "svgs"
_MANIFEST_MAP = {
    "all": "all_icons.txt",
    "all_icons": "all_icons.txt",
    "all_icons.txt": "all_icons.txt",
    "non_symmetry": "non_symmetry.txt",
    "non_symmetry.txt": "non_symmetry.txt",
    "symmetry": "symmetry.txt",
    "symmetry.txt": "symmetry.txt",
}


@lru_cache(maxsize=1)
def _svg_index_by_manifest_id() -> dict[str, Path]:
    """Index curated SVGs by manifest id without light/dark filename prefixes."""

    mapping: dict[str, Path] = {}
    for path in _SVG_ROOT.glob("*.svg"):
        stem = str(path.stem)
        canonical = stem
        if canonical.startswith("light-"):
            canonical = canonical[len("light-") :]
        elif canonical.startswith("dark-"):
            canonical = canonical[len("dark-") :]
        mapping[str(canonical)] = path
    return mapping


def icon_asset_root() -> Path:
    """Return the Trace-side curated icon asset root."""

    return _ASSET_ROOT


def icon_svg_path(icon_id: str) -> Path:
    """Return the SVG path for one curated icon identifier."""

    icon_name = str(icon_id).strip()
    if not icon_name:
        raise ValueError("icon_id must be non-empty")
    path = _svg_index_by_manifest_id().get(icon_name)
    if path is None or not path.exists():
        raise FileNotFoundError(_SVG_ROOT / f"{icon_name}.svg")
    return path


@lru_cache(maxsize=32)
def load_icon_manifest(manifest_name: str) -> Tuple[str, ...]:
    """Load one curated icon manifest as a deterministic tuple of ids."""

    manifest_key = str(manifest_name).strip()
    key = _MANIFEST_MAP.get(manifest_key)
    if key is None:
        supported = ", ".join(available_manifests())
        raise ValueError(f"unsupported icon manifest {manifest_name!r}; supported manifests: {supported}")
    path = _ASSET_ROOT / str(key)
    if not path.exists():
        raise FileNotFoundError(path)
    icons = tuple(
        str(line).strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if str(line).strip()
    )
    if not icons:
        raise ValueError(f"manifest {path} resolved no icons")
    return icons


def resolve_icon_pool(manifest_name: str) -> Tuple[str, ...]:
    """Resolve one named curated icon pool."""

    return load_icon_manifest(str(manifest_name))


@lru_cache(maxsize=4096)
def _render_base_icon(icon_id: str, size_px: int) -> Image.Image:
    """Render one SVG icon to RGBA before color/transform post-processing."""

    size = max(8, int(size_px))
    png_bytes = cairosvg.svg2png(url=str(icon_svg_path(icon_id)), output_width=size, output_height=size)
    image = Image.open(BytesIO(png_bytes)).convert("RGBA")
    return image


def _tight_alpha_crop(image: Image.Image) -> Image.Image:
    """Return the tight alpha crop for one RGBA icon image."""

    alpha = image.getchannel("A")
    bbox = alpha.getbbox()
    if bbox is None:
        raise ValueError("rendered icon has empty alpha")
    return image.crop(bbox)


def render_icon_rgba(
    *,
    icon_id: str,
    size_px: int,
    tint_rgb: Tuple[int, int, int],
    rotation_degrees: int = 0,
    mirror_x: bool = False,
    noise_edits: Sequence[NoiseEdit] = (),
    noise_seed: int | None = None,
) -> Image.Image:
    """Render one cropped/tinted/rotated icon as RGBA.

    Trace icon tasks use icons as silhouettes. We preserve that here by
    using the rendered alpha channel and filling it with one deterministic tint.
    """

    base = _render_base_icon(str(icon_id), int(size_px))
    alpha = base.getchannel("A")
    tinted = Image.new("RGBA", base.size, tuple(int(value) for value in tint_rgb) + (0,))
    tinted.putalpha(alpha)
    if bool(mirror_x):
        tinted = ImageOps.mirror(tinted)
    rotation = int(rotation_degrees) % 360
    if rotation:
        tinted = tinted.rotate(rotation, expand=True, resample=Image.Resampling.BICUBIC)
    cropped = _tight_alpha_crop(tinted)
    if noise_edits:
        import random

        cropped = apply_icon_noise_edits_rgba(
            cropped,
            edits=tuple(noise_edits),
            rng=random.Random(int(noise_seed if noise_seed is not None else 0)),
        )
    return cropped


def render_icon_transformed_rgba(
    *,
    icon_id: str,
    size_px: int,
    tint_rgb: Tuple[int, int, int],
    transform_id: str = IDENTITY_TRANSFORM_ID,
    noise_edits: Sequence[NoiseEdit] = (),
    noise_seed: int | None = None,
) -> Image.Image:
    """Render one icon with a canonical D4 transform applied."""

    base = _render_base_icon(str(icon_id), max(8, int(size_px)))
    alpha = base.getchannel("A")
    tinted = Image.new("RGBA", base.size, tuple(int(value) for value in tint_rgb) + (0,))
    tinted.putalpha(alpha)
    transformed = apply_canonical_transform(tinted, str(transform_id))
    cropped = _tight_alpha_crop(transformed)
    if noise_edits:
        import random

        cropped = apply_icon_noise_edits_rgba(
            cropped,
            edits=tuple(noise_edits),
            rng=random.Random(int(noise_seed if noise_seed is not None else 0)),
        )
    return cropped


@lru_cache(maxsize=32768)
def icon_transform_signature(
    icon_id: str,
    size_px: int,
    transform_id: str = IDENTITY_TRANSFORM_ID,
) -> Tuple[int, int, str]:
    """Return a stable alpha-signature for one rendered icon transform.

    The signature is used to reject icon/transform pairs that would look
    visually equivalent in transformation tasks even when the transform names
    differ algebraically.
    """

    base = _render_base_icon(str(icon_id), max(8, int(size_px)))
    alpha = base.getchannel("A")
    rgba = Image.new("RGBA", base.size, (0, 0, 0, 0))
    rgba.putalpha(alpha)
    transformed = apply_canonical_transform(rgba, str(transform_id))
    cropped = _tight_alpha_crop(transformed)
    alpha_bytes = cropped.getchannel("A").tobytes()
    digest = hashlib.md5(alpha_bytes).hexdigest()
    return int(cropped.size[0]), int(cropped.size[1]), str(digest)


def available_manifests() -> Tuple[str, ...]:
    """Return the supported curated manifest keys."""

    return tuple(sorted({str(key) for key in _MANIFEST_MAP.keys() if key.endswith(".txt") is False}))


__all__ = [
    "available_manifests",
    "icon_asset_root",
    "icon_transform_signature",
    "icon_svg_path",
    "load_icon_manifest",
    "render_icon_rgba",
    "render_icon_transformed_rgba",
    "resolve_icon_pool",
]
