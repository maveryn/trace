"""Canonical icon-transform helpers for Trace icon tasks."""

from __future__ import annotations

from typing import Tuple

from PIL import Image, ImageOps


IDENTITY_TRANSFORM_ID = "identity"
NON_IDENTITY_TRANSFORM_IDS: Tuple[str, ...] = (
    "rot90",
    "rot180",
    "rot270",
    "flip_h",
    "flip_v",
    "flip_diag_main",
    "flip_diag_anti",
)
CANONICAL_TRANSFORM_IDS: Tuple[str, ...] = (IDENTITY_TRANSFORM_ID, *NON_IDENTITY_TRANSFORM_IDS)


def apply_canonical_transform(image: Image.Image, transform_id: str) -> Image.Image:
    """Apply one canonical square-symmetry transform to an RGBA icon image."""

    transform_key = str(transform_id).strip().lower()
    if transform_key == IDENTITY_TRANSFORM_ID:
        return image.copy()
    if transform_key == "rot90":
        return image.rotate(90, expand=True, resample=Image.Resampling.BICUBIC)
    if transform_key == "rot180":
        return image.rotate(180, expand=True, resample=Image.Resampling.BICUBIC)
    if transform_key == "rot270":
        return image.rotate(270, expand=True, resample=Image.Resampling.BICUBIC)
    if transform_key == "flip_h":
        return ImageOps.mirror(image)
    if transform_key == "flip_v":
        return ImageOps.flip(image)
    if transform_key == "flip_diag_main":
        return image.transpose(Image.Transpose.TRANSPOSE)
    if transform_key == "flip_diag_anti":
        return image.transpose(Image.Transpose.TRANSVERSE)
    raise ValueError(f"unsupported canonical transform: {transform_id}")


__all__ = [
    "CANONICAL_TRANSFORM_IDS",
    "IDENTITY_TRANSFORM_ID",
    "NON_IDENTITY_TRANSFORM_IDS",
    "apply_canonical_transform",
]
