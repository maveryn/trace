"""Deterministic helpers for low-risk render parameter variation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ...core.seed import hash64
from .config_defaults import group_default


def _selection_index(*, instance_seed: int | None, namespace: str, key: str, salt: int) -> int:
    """Return a stable per-instance selection index for one visual knob."""

    seed = 0 if instance_seed is None else int(instance_seed)
    return abs(int(hash64(int(seed), f"{namespace}:{key}", int(salt))))


def resolve_render_int(
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    key: str,
    fallback: int,
    *,
    instance_seed: int | None,
    namespace: str,
) -> int:
    """Resolve an integer render parameter, honoring optional ``_min``/``_max`` bounds."""

    if params.get(str(key)) is not None:
        return int(params[str(key)])
    low_raw = params.get(f"{str(key)}_min", group_default(defaults, f"{str(key)}_min", None))
    high_raw = params.get(f"{str(key)}_max", group_default(defaults, f"{str(key)}_max", None))
    if low_raw is not None or high_raw is not None:
        base = int(group_default(defaults, str(key), int(fallback)))
        low = int(base if low_raw is None else low_raw)
        high = int(base if high_raw is None else high_raw)
        if int(low) > int(high):
            raise ValueError(f"{key}_min must be <= {key}_max")
        span = int(high) - int(low) + 1
        index = _selection_index(instance_seed=instance_seed, namespace=str(namespace), key=str(key), salt=9187)
        return int(low) + int(index % int(span))
    return int(group_default(defaults, str(key), int(fallback)))


def resolve_render_float(
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    key: str,
    fallback: float,
    *,
    instance_seed: int | None,
    namespace: str,
    steps: int = 1000,
) -> float:
    """Resolve a float render parameter, honoring optional ``_min``/``_max`` bounds."""

    if params.get(str(key)) is not None:
        return float(params[str(key)])
    low_raw = params.get(f"{str(key)}_min", group_default(defaults, f"{str(key)}_min", None))
    high_raw = params.get(f"{str(key)}_max", group_default(defaults, f"{str(key)}_max", None))
    if low_raw is not None or high_raw is not None:
        base = float(group_default(defaults, str(key), float(fallback)))
        low = float(base if low_raw is None else low_raw)
        high = float(base if high_raw is None else high_raw)
        if float(low) > float(high):
            raise ValueError(f"{key}_min must be <= {key}_max")
        bins = max(1, int(steps))
        index = _selection_index(instance_seed=instance_seed, namespace=str(namespace), key=str(key), salt=10601)
        return float(low) + ((float(high) - float(low)) * (float(index % (bins + 1)) / float(bins)))
    return float(group_default(defaults, str(key), float(fallback)))


def resolve_render_rgb(
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    key: str,
    fallback: Sequence[int],
    *,
    instance_seed: int | None,
    namespace: str,
) -> tuple[int, int, int]:
    """Resolve an RGB parameter, honoring optional ``_options`` color palettes."""

    if params.get(str(key)) is not None:
        raw = params[str(key)]
    else:
        options = params.get(f"{str(key)}_options", group_default(defaults, f"{str(key)}_options", None))
        if options is not None:
            option_list = [item for item in options if isinstance(item, Sequence) and len(item) == 3]
            if not option_list:
                raise ValueError(f"{key}_options must contain at least one RGB triple")
            index = _selection_index(instance_seed=instance_seed, namespace=str(namespace), key=str(key), salt=12517)
            raw = option_list[int(index % len(option_list))]
        else:
            raw = group_default(defaults, str(key), list(fallback))
    if not isinstance(raw, Sequence) or len(raw) != 3:
        raise ValueError(f"{key} must resolve to an RGB triple")
    return tuple(int(value) for value in raw)


def _resolve_bool(value: Any, fallback: bool) -> bool:
    """Resolve one boolean-ish render config value."""

    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "y", "on", "always"}:
            return True
        if text in {"0", "false", "no", "n", "off", "never"}:
            return False
    return bool(fallback)


def resolve_layout_jitter(
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    *,
    instance_seed: int | None,
    namespace: str,
) -> dict[str, Any]:
    """Resolve deterministic nonsemantic layout jitter settings."""

    enabled = _resolve_bool(
        params.get("layout_jitter_enabled", group_default(defaults, "layout_jitter_enabled", False)),
        False,
    )
    min_margin = max(0, int(params.get("layout_jitter_min_margin_px", group_default(defaults, "layout_jitter_min_margin_px", 24))))
    mode_raw = str(params.get("layout_jitter_mode", group_default(defaults, "layout_jitter_mode", "pixel")))
    mode = mode_raw.strip().lower().replace("-", "_")
    if mode in {"fraction", "fractional", "fraction_of_slack", "slack"}:
        mode = "slack_fraction"
    if mode not in {"pixel", "slack_fraction"}:
        raise ValueError(f"unsupported layout_jitter_mode: {mode_raw!r}")
    if not enabled:
        return {
            "enabled": False,
            "mode": str(mode),
            "requested_dx_px": 0,
            "requested_dy_px": 0,
            "dx_px": 0,
            "dy_px": 0,
            "min_margin_px": int(min_margin),
        }
    if mode == "slack_fraction":
        return {
            "enabled": True,
            "mode": "slack_fraction",
            "requested_x_slack_fraction": float(
                resolve_render_float(
                    params,
                    defaults,
                    "layout_jitter_x_slack_fraction",
                    0.0,
                    instance_seed=instance_seed,
                    namespace=str(namespace),
                    steps=1000,
                )
            ),
            "requested_y_slack_fraction": float(
                resolve_render_float(
                    params,
                    defaults,
                    "layout_jitter_y_slack_fraction",
                    0.0,
                    instance_seed=instance_seed,
                    namespace=str(namespace),
                    steps=1000,
                )
            ),
            "requested_dx_px": 0,
            "requested_dy_px": 0,
            "dx_px": 0,
            "dy_px": 0,
            "min_margin_px": int(min_margin),
        }
    return {
        "enabled": True,
        "mode": "pixel",
        "requested_dx_px": int(
            resolve_render_int(
                params,
                defaults,
                "layout_jitter_x_px",
                0,
                instance_seed=instance_seed,
                namespace=str(namespace),
            )
        ),
        "requested_dy_px": int(
            resolve_render_int(
                params,
                defaults,
                "layout_jitter_y_px",
                0,
                instance_seed=instance_seed,
                namespace=str(namespace),
            )
        ),
        "dx_px": 0,
        "dy_px": 0,
        "min_margin_px": int(min_margin),
    }


def apply_layout_jitter_to_margins(
    *,
    left_px: int,
    right_px: int,
    top_px: int,
    bottom_px: int,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    instance_seed: int | None,
    namespace: str,
) -> tuple[int, int, int, int, dict[str, Any]]:
    """Shift a centered layout by trading pixels between opposite margins.

    Positive x jitter moves the layout right; positive y jitter moves it down.
    Plot/panel dimensions are preserved because opposite margins are adjusted
    by equal and opposite amounts.
    """

    jitter = resolve_layout_jitter(
        params,
        defaults,
        instance_seed=instance_seed,
        namespace=str(namespace),
    )
    return apply_resolved_layout_jitter_to_margins(
        left_px=int(left_px),
        right_px=int(right_px),
        top_px=int(top_px),
        bottom_px=int(bottom_px),
        jitter=jitter,
    )


def apply_resolved_layout_jitter_to_margins(
    *,
    left_px: int | float,
    right_px: int | float,
    top_px: int | float,
    bottom_px: int | float,
    jitter: Mapping[str, Any] | None,
) -> tuple[int, int, int, int, dict[str, Any]]:
    """Clamp a pre-resolved layout jitter request against concrete margins."""

    jitter_base = dict(jitter or {})
    min_margin = int(jitter_base.get("min_margin_px", 24))
    mode = str(jitter_base.get("mode", "pixel")).strip().lower().replace("-", "_")
    if mode in {"fraction", "fractional", "fraction_of_slack", "slack"}:
        mode = "slack_fraction"
    max_dx = max(0, int(right_px) - int(min_margin))
    min_dx = -max(0, int(left_px) - int(min_margin))
    max_dy = max(0, int(bottom_px) - int(min_margin))
    min_dy = -max(0, int(top_px) - int(min_margin))
    if mode == "slack_fraction":
        x_fraction = max(-1.0, min(1.0, float(jitter_base.get("requested_x_slack_fraction", 0.0))))
        y_fraction = max(-1.0, min(1.0, float(jitter_base.get("requested_y_slack_fraction", 0.0))))
        requested_dx = int(round(x_fraction * (float(max_dx) if x_fraction >= 0.0 else float(-min_dx))))
        requested_dy = int(round(y_fraction * (float(max_dy) if y_fraction >= 0.0 else float(-min_dy))))
    else:
        mode = "pixel"
        requested_dx = int(jitter_base.get("requested_dx_px", 0))
        requested_dy = int(jitter_base.get("requested_dy_px", 0))
    dx = max(int(min_dx), min(int(max_dx), int(requested_dx)))
    dy = max(int(min_dy), min(int(max_dy), int(requested_dy)))
    resolved = dict(jitter_base)
    resolved.update(
        {
            "enabled": bool(jitter_base.get("enabled", False)),
            "mode": str(mode),
            "requested_dx_px": int(requested_dx),
            "requested_dy_px": int(requested_dy),
            "dx_px": int(dx),
            "dy_px": int(dy),
            "min_margin_px": int(min_margin),
            "clamp_x_px": [int(min_dx), int(max_dx)],
            "clamp_y_px": [int(min_dy), int(max_dy)],
        }
    )
    return (
        int(left_px) + int(dx),
        int(right_px) - int(dx),
        int(top_px) + int(dy),
        int(bottom_px) - int(dy),
        resolved,
    )
