"""Shared task-level visual-default loading helpers.

These helpers keep task modules focused on fallback policy while reusing one
implementation for scene visual section loading.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Mapping

from ...core.visual.defaults import load_domain_visual_section, load_scene_visual_section
from ...core.visual.noise import TRACE_DEFAULT_NOISE_EDIT_TYPES, TRACE_DEFAULT_NOISE_VALUE_RANGES


def default_noise_fallback(*, apply_prob: float) -> Dict[str, Any]:
    """Build a standard deterministic post-noise fallback config."""
    return {
        "apply_prob": float(apply_prob),
        "edit_types": list(TRACE_DEFAULT_NOISE_EDIT_TYPES),
        "edit_count_range": [1, 2],
        "value_ranges": deepcopy(TRACE_DEFAULT_NOISE_VALUE_RANGES),
    }


def load_domain_background_defaults(
    *,
    domain: str,
    fallback: Mapping[str, Any],
    merge_with_fallback: bool = True,
) -> Dict[str, Any]:
    """Load domain-level background defaults with shared section plumbing."""
    return load_domain_visual_section(
        domain=str(domain),
        section="background",
        fallback=dict(fallback),
        merge_with_fallback=bool(merge_with_fallback),
    )


def load_scene_background_defaults(
    *,
    domain: str,
    scene_id: str,
    fallback: Mapping[str, Any],
    merge_with_fallback: bool = True,
) -> Dict[str, Any]:
    """Load scene background defaults with shared section plumbing."""

    return load_scene_visual_section(
        domain=str(domain),
        scene_id=str(scene_id),
        section="background",
        fallback=dict(fallback),
        merge_with_fallback=bool(merge_with_fallback),
    )


def load_scene_noise_defaults(
    *,
    domain: str,
    scene_id: str,
    fallback: Mapping[str, Any],
    merge_with_fallback: bool = False,
) -> Dict[str, Any]:
    """Load scene post-noise defaults with shared section plumbing."""

    return load_scene_visual_section(
        domain=str(domain),
        scene_id=str(scene_id),
        section="noise",
        fallback=dict(fallback),
        merge_with_fallback=bool(merge_with_fallback),
    )


def load_domain_noise_defaults(
    *,
    domain: str,
    fallback: Mapping[str, Any],
    merge_with_fallback: bool = False,
) -> Dict[str, Any]:
    """Load domain-level post-noise defaults with shared section plumbing."""
    return load_domain_visual_section(
        domain=str(domain),
        section="noise",
        fallback=dict(fallback),
        merge_with_fallback=bool(merge_with_fallback),
    )


__all__ = [
    "default_noise_fallback",
    "load_domain_background_defaults",
    "load_domain_noise_defaults",
    "load_scene_background_defaults",
    "load_scene_noise_defaults",
    "load_scene_background_defaults",
    "load_scene_noise_defaults",
]
