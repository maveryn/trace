"""Task-level config fallback helpers shared across task modules."""

from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Sequence, Tuple

from ...resources import resource_path
from ...core.scene_config import get_scene_defaults, resolve_scene_section_defaults


_PROMPT_CODE_DEFAULTS_CACHE: Dict[str, Dict[str, Any]] = {}


def _prompt_code_defaults_for_bundle(bundle_id: str) -> Dict[str, Any]:
    """Return prompt-asset code defaults for one bundle id.

    This is a migration bridge for scene-package tasks whose prompt prose has
    moved out of YAML but whose task code still resolves code-facing prompt
    defaults before rendering v1 prompt assets.
    """

    bundle_key = str(bundle_id).strip()
    if not bundle_key:
        return {}
    if bundle_key in _PROMPT_CODE_DEFAULTS_CACHE:
        return dict(_PROMPT_CODE_DEFAULTS_CACHE[bundle_key])

    defaults: Dict[str, Any] = {}
    prompt_root = resource_path("prompts")
    for path in prompt_root.glob(f"**/{bundle_key}.json"):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        code_defaults = raw.get("code_prompt_defaults")
        if isinstance(code_defaults, Mapping):
            defaults.update({str(key): value for key, value in code_defaults.items()})
        static_slots = raw.get("static_slots_by_key")
        if isinstance(static_slots, Mapping):
            for slot_values in static_slots.values():
                if isinstance(slot_values, Mapping):
                    defaults.update({str(key): value for key, value in slot_values.items()})
    _PROMPT_CODE_DEFAULTS_CACHE[bundle_key] = dict(defaults)
    return dict(defaults)


def group_default(mapping: Mapping[str, Any], key: str, fallback: Any) -> Any:
    """Return `mapping[key]` when present; otherwise return `fallback`."""
    if key in mapping:
        return mapping.get(key)
    return fallback


def required_group_default(mapping: Mapping[str, Any], key: str, *, context: str) -> Any:
    """Return a required config value and fail fast when missing/empty."""
    if key not in mapping:
        bundle_id = mapping.get("bundle_id")
        if bundle_id is not None:
            prompt_defaults = _prompt_code_defaults_for_bundle(str(bundle_id))
            if key in prompt_defaults:
                value = prompt_defaults.get(key)
                if value is None:
                    raise ValueError(f"prompt asset key '{key}' in {context} cannot be null")
                if isinstance(value, str) and not str(value).strip():
                    raise ValueError(f"prompt asset key '{key}' in {context} cannot be empty")
                return value
        raise ValueError(f"missing required config key '{key}' in {context}")
    value = mapping.get(key)
    if value is None:
        raise ValueError(f"config key '{key}' in {context} cannot be null")
    if isinstance(value, str) and not str(value).strip():
        raise ValueError(f"config key '{key}' in {context} cannot be empty")
    return value


def required_group_defaults(
    mapping: Mapping[str, Any],
    keys: Sequence[str],
    *,
    context: str,
) -> Dict[str, Any]:
    """Return required config values for `keys` with one fail-fast validation pass."""
    return {
        str(key): required_group_default(mapping, str(key), context=context)
        for key in [str(item) for item in keys]
    }


def resolve_optional_int_bounds(
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    *,
    min_key: str,
    max_key: str,
    context: str,
) -> Tuple[int | None, int | None]:
    """Resolve optional inclusive integer bounds from params/defaults."""
    raw_min = params.get(str(min_key), group_default(defaults, str(min_key), None))
    raw_max = params.get(str(max_key), group_default(defaults, str(max_key), None))
    min_value = int(raw_min) if raw_min is not None else None
    max_value = int(raw_max) if raw_max is not None else None
    if min_value is not None and max_value is not None and int(min_value) > int(max_value):
        raise ValueError(f"{min_key} must be <= {max_key} in {context}")
    return min_value, max_value


def resolve_required_int_bounds(
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    *,
    min_key: str,
    max_key: str,
    fallback_min: int,
    fallback_max: int,
    context: str,
) -> Tuple[int, int]:
    """Resolve required inclusive integer bounds from params/defaults/fallbacks."""
    min_value = int(params.get(str(min_key), group_default(defaults, str(min_key), int(fallback_min))))
    max_value = int(params.get(str(max_key), group_default(defaults, str(max_key), int(fallback_max))))
    if int(min_value) > int(max_value):
        raise ValueError(f"{min_key} must be <= {max_key} in {context}")
    return int(min_value), int(max_value)


def resolve_required_float_bounds(
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    *,
    min_key: str,
    max_key: str,
    fallback_min: float,
    fallback_max: float,
    context: str,
) -> Tuple[float, float]:
    """Resolve required inclusive float bounds from params/defaults/fallbacks."""
    min_value = float(params.get(str(min_key), group_default(defaults, str(min_key), float(fallback_min))))
    max_value = float(params.get(str(max_key), group_default(defaults, str(max_key), float(fallback_max))))
    if float(min_value) > float(max_value):
        raise ValueError(f"{min_key} must be <= {max_key} in {context}")
    return float(min_value), float(max_value)


def split_generation_rendering_prompt_defaults(
    mapping: Mapping[str, Any],
    *,
    task_id: str | None = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Return `generation`, `rendering`, and `prompt` defaults sections."""
    generation = resolve_scene_section_defaults(mapping, "generation", task_id=task_id)
    rendering = resolve_scene_section_defaults(mapping, "rendering", task_id=task_id)
    prompt = resolve_scene_section_defaults(mapping, "prompt", task_id=task_id)
    return generation, rendering, prompt


def split_scene_generation_rendering_prompt_defaults(
    mapping: Mapping[str, Any],
    *,
    task_id: str | None = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Return scene-based `generation`, `rendering`, and `prompt` defaults sections."""

    generation = resolve_scene_section_defaults(mapping, "generation", task_id=task_id)
    rendering = resolve_scene_section_defaults(mapping, "rendering", task_id=task_id)
    prompt = resolve_scene_section_defaults(mapping, "prompt", task_id=task_id)
    return generation, rendering, prompt


def load_scene_generation_rendering_prompt_defaults(
    domain: str,
    scene_id: str,
    *,
    task_id: str | None = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Load and split scene-based defaults for source-layout scene-package tasks."""

    mapping = get_scene_defaults(str(domain), str(scene_id))
    return split_scene_generation_rendering_prompt_defaults(
        mapping if isinstance(mapping, Mapping) else {},
        task_id=task_id,
    )
