"""Helpers for loading domain and scene visual defaults with fallbacks."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Mapping

from ..scene_config import get_domain_defaults, get_scene_defaults


def _resolve_visual_section(
    visual: Mapping[str, Any],
    *,
    section: str,
    fallback: Mapping[str, Any],
    merge_with_fallback: bool,
) -> Dict[str, Any]:
    """Resolve one visual subsection with fallback handling."""
    section_value = visual.get(str(section), {})
    if not isinstance(section_value, Mapping):
        return deepcopy(dict(fallback))

    if bool(merge_with_fallback):
        merged = deepcopy(dict(fallback))
        merged.update(dict(section_value))
        return merged
    return dict(section_value)


def load_domain_visual_section(
    *,
    domain: str,
    section: str,
    fallback: Mapping[str, Any],
    merge_with_fallback: bool,
) -> Dict[str, Any]:
    """Load one domain-level visual section (`background`/`noise`)."""
    cfg = get_domain_defaults(str(domain))
    visual = cfg.get("visual", {})
    if not isinstance(visual, Mapping):
        return deepcopy(dict(fallback))
    return _resolve_visual_section(
        visual,
        section=str(section),
        fallback=fallback,
        merge_with_fallback=bool(merge_with_fallback),
    )


def load_scene_visual_section(
    *,
    domain: str,
    scene_id: str,
    section: str,
    fallback: Mapping[str, Any],
    merge_with_fallback: bool,
) -> Dict[str, Any]:
    """Load one scene visual section (`background`/`noise`) with fallback handling."""

    cfg = get_scene_defaults(str(domain), str(scene_id))
    visual = cfg.get("visual", {})
    if not isinstance(visual, Mapping):
        return deepcopy(dict(fallback))
    return _resolve_visual_section(
        visual,
        section=str(section),
        fallback=fallback,
        merge_with_fallback=bool(merge_with_fallback),
    )
