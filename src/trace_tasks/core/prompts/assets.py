"""Prompt bundle asset loading and caching."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict

from ...resources import resource_path, safe_resource_join
from .schema import PromptBundle, parse_prompt_bundle


_CACHE: Dict[str, PromptBundle] = {}


def _prompt_root() -> Path:
    """Resolve the prompt-bundle root directory."""
    override = os.getenv("TRACE_PROMPT_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return resource_path("prompts")


def _bundle_rel_path(domain: str, scene_id: str, bundle_id: str) -> Path:
    """Build bundle path relative to the prompt root."""
    return Path(str(domain)) / str(scene_id) / f"{str(bundle_id)}.json"


def _bundle_abs_path(domain: str, scene_id: str, bundle_id: str) -> Path:
    """Build absolute bundle path for one domain/scene/bundle id."""
    return safe_resource_join(
        _prompt_root(),
        _bundle_rel_path(domain, scene_id, bundle_id),
    )

def load_prompt_bundle(domain: str, scene_id: str, bundle_id: str) -> PromptBundle:
    """Load a prompt bundle for one domain/scene pair."""
    abs_path = _bundle_abs_path(domain, scene_id, bundle_id)
    cache_key = str(abs_path.resolve())
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    if not abs_path.exists():
        raise FileNotFoundError(f"prompt bundle not found: {abs_path}")

    payload = abs_path.read_bytes()
    raw = json.loads(payload.decode("utf-8"))
    bundle = parse_prompt_bundle(
        raw,
        source_path=str(_bundle_rel_path(domain, scene_id, bundle_id)),
        source_hash=hashlib.sha256(payload).hexdigest(),
    )
    _CACHE[cache_key] = bundle
    return bundle


def load_scene_prompt_bundle(domain: str, scene_id: str, bundle_id: str) -> PromptBundle:
    """Alias for scene prompt-bundle loading."""

    return load_prompt_bundle(domain=domain, scene_id=scene_id, bundle_id=bundle_id)
