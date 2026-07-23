"""Scene-based domain configuration helpers."""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml

from ..resources import resource_path, safe_resource_join


_CACHE_BY_PATH: Dict[str, Dict[str, Any]] = {}
_SECTION_SHARED_KEY = "shared"
_SECTION_TASK_OVERRIDES_KEY = "task_overrides"


def _config_root() -> Path:
    """Resolve the domain config root with optional environment override."""

    override = os.getenv("TRACE_DOMAIN_CONFIG_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return resource_path("configs", "domains")


def _domain_config_path(domain: str) -> Path:
    """Return the domain-default config path."""

    return safe_resource_join(_config_root(), str(domain), "base.yaml")


def _scene_config_path(domain: str, scene_id: str) -> Path:
    """Return the scene config path."""

    return safe_resource_join(_config_root(), str(domain), f"{str(scene_id)}.yaml")


def _load_config(path: Path) -> Dict[str, Any]:
    """Load and cache one YAML config file as a plain mapping."""

    key = str(path.resolve())
    if key in _CACHE_BY_PATH:
        return _CACHE_BY_PATH[key]
    if not path.exists():
        _CACHE_BY_PATH[key] = {}
        return _CACHE_BY_PATH[key]
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raw = {}
    _CACHE_BY_PATH[key] = dict(raw)
    return _CACHE_BY_PATH[key]


def _deep_merge_mappings(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Deep-merge two mappings with override precedence."""

    merged: Dict[str, Any] = {}
    ordered_keys = list(base.keys()) + [key for key in override.keys() if key not in base]
    for key in ordered_keys:
        if key in override:
            over_value = override[key]
            if key in base and isinstance(base[key], Mapping) and isinstance(over_value, Mapping):
                merged[key] = _deep_merge_mappings(base[key], over_value)
            else:
                merged[key] = deepcopy(over_value)
        else:
            merged[key] = deepcopy(base[key])
    return merged


def resolve_scene_section_defaults(
    mapping: Mapping[str, Any],
    section: str,
    *,
    task_id: str | None = None,
) -> Dict[str, Any]:
    """Resolve one config section for a specific task."""

    value = mapping.get(str(section), {})
    if not isinstance(value, Mapping):
        return {}

    resolved: Dict[str, Any] = {}
    shared = value.get(_SECTION_SHARED_KEY, {})
    if isinstance(shared, Mapping):
        resolved = deepcopy(dict(shared))

    if task_id:
        overrides = value.get(_SECTION_TASK_OVERRIDES_KEY, {})
        if isinstance(overrides, Mapping):
            task_values = overrides.get(str(task_id), {})
            if isinstance(task_values, Mapping):
                resolved = _deep_merge_mappings(resolved, dict(task_values))
    return resolved


def get_domain_defaults(domain: str) -> Dict[str, Any]:
    """Return deep-copied defaults for a domain."""

    cfg = _load_config(_domain_config_path(domain))
    return deepcopy(dict(cfg)) if isinstance(cfg, Mapping) else {}


def get_scene_defaults(domain: str, scene_id: str) -> Dict[str, Any]:
    """Return merged domain and scene defaults."""

    domain_cfg = _load_config(_domain_config_path(domain))
    scene_cfg = _load_config(_scene_config_path(domain, scene_id))
    domain_map = dict(domain_cfg) if isinstance(domain_cfg, Mapping) else {}
    scene_map = dict(scene_cfg) if isinstance(scene_cfg, Mapping) else {}
    return deepcopy(_deep_merge_mappings(domain_map, scene_map))


__all__ = ["get_domain_defaults", "get_scene_defaults", "resolve_scene_section_defaults"]
