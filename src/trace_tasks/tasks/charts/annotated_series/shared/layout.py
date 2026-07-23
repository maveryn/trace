"""Layout helpers for the annotated-series chart scene."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.charts.annotated_series.shared.defaults import (
    CONTEXT_PARAM_KEYS,
    FALLBACK_CHART_DEFAULTS,
    SCENE_NAMESPACE,
    rendering_value,
)
from trace_tasks.tasks.shared.visual_style.context_layer import resolve_dashboard_context_layout


def context_params(params: Mapping[str, Any]) -> dict[str, Any]:
    resolved: dict[str, Any] = {}
    for key in CONTEXT_PARAM_KEYS:
        value = rendering_value(params, str(key), None)
        if value is not None:
            resolved[str(key)] = value
    return resolved


def context_profile(params: Mapping[str, Any]) -> str:
    raw_profile = rendering_value(params, "chart_context_profile", rendering_value(params, "context_text_profile", "report_paragraph"))
    normalized = str(raw_profile).strip().casefold().replace("-", "_").replace(" ", "_")
    if normalized in {"dense", "clean_minimal", "clean_minimal_only", "dense_clean_minimal"}:
        return "dense_clean_minimal"
    return "report_paragraph"


def choose_context_mode(*, params: Mapping[str, Any], instance_seed: int) -> str:
    if not bool(rendering_value(params, "context_text_enabled", False)):
        return "clean"
    supported = ("clean", "minimal", "paragraph_box")
    explicit_mode = rendering_value(params, "chart_context_mode", rendering_value(params, "context_text_mode", None))
    if explicit_mode is not None:
        normalized = _normalize_context_mode(str(explicit_mode))
        if normalized not in set(supported):
            raise ValueError(f"unsupported annotated-series chart context mode: {explicit_mode!r}")
        return str(normalized)
    raw_weights = rendering_value(
        params,
        "chart_context_mode_weights",
        rendering_value(params, "context_text_mode_weights", {"clean": 0.3, "minimal": 0.4, "paragraph_box": 0.3}),
    )
    if not isinstance(raw_weights, Mapping):
        raw_weights = {"clean": 1.0}
    normalized_weights: dict[str, float] = {key: 0.0 for key in supported}
    for raw_mode, raw_weight in raw_weights.items():
        mode = _normalize_context_mode(str(raw_mode))
        if mode not in normalized_weights:
            continue
        normalized_weights[str(mode)] += max(0.0, float(raw_weight))
    weights: list[tuple[str, float]] = []
    for mode in supported:
        weight = max(0.0, float(normalized_weights.get(str(mode), 0.0)))
        if weight > 0.0:
            weights.append((str(mode), float(weight)))
    if not weights:
        return "clean"
    rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.context_text_mode")
    cursor = rng.random() * sum(weight for _, weight in weights)
    running = 0.0
    for mode, weight in weights:
        running += float(weight)
        if cursor <= running:
            return str(mode)
    return str(weights[-1][0])


def _normalize_context_mode(value: str) -> str:
    normalized = str(value).strip().casefold().replace("-", "_").replace(" ", "_")
    if normalized in {"light", "light_context"}:
        return "minimal"
    if normalized in {"large", "large_distractor", "paragraph", "right_sidebar", "left_sidebar", "bottom_band", "sidebar"}:
        return "paragraph_box"
    return str(normalized)


def resolve_context_layout(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    canvas_width: int,
    canvas_height: int,
) -> dict[str, Any]:
    """Resolve non-answer context placement for the annotated-series scene.

    This helper chooses scene-level context geometry only. It does not know the
    public task objective, answer labels, or annotation target.
    """
    mode = choose_context_mode(params=params, instance_seed=int(instance_seed))
    resolved_context_params = context_params(params)
    profile = context_profile(params)
    top_reserved = int(resolved_context_params.get("context_text_top_reserved_px", 64))
    bottom_reserved = int(resolved_context_params.get("context_text_bottom_reserved_px", 28))
    if str(mode) == "clean":
        return {
            "enabled": False,
            "mode": "clean",
            "layout_mode": "chart_context:clean",
            "placement": "none",
            "context_profile": str(profile),
            "context_params": resolved_context_params,
            "top_reserved_px": int(top_reserved),
            "bottom_reserved_px": int(bottom_reserved),
        }
    if str(mode) == "minimal":
        return {
            "enabled": True,
            "mode": "minimal",
            "layout_mode": "chart_context:minimal",
            "placement": "top_bottom_notes",
            "box_count": 0,
            "context_profile": str(profile),
            "context_params": resolved_context_params,
            "top_reserved_px": int(top_reserved),
            "bottom_reserved_px": int(bottom_reserved),
        }
    layout = resolve_dashboard_context_layout(
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.context",
        params={
            **resolved_context_params,
            "context_text_enabled": True,
            "context_text_layout_mode": "chart_context:paragraph_box",
        },
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        top_reserved_px=int(top_reserved),
        bottom_reserved_px=int(bottom_reserved),
        left_margin_px=int(resolved_context_params.get("context_text_left_margin_px", 24)),
        right_margin_px=int(resolved_context_params.get("context_text_right_margin_px", 24)),
    )
    return {
        **dict(layout),
        "enabled": True,
        "mode": "paragraph_box",
        "layout_mode": "chart_context:paragraph_box",
        "context_profile": str(profile),
        "context_params": resolved_context_params,
    }


def apply_context_margin_overrides(
    params: Mapping[str, Any],
    *,
    context_layout: Mapping[str, Any],
) -> dict[str, Any]:
    resolved = dict(params)
    if not bool(context_layout.get("enabled", False)):
        return resolved

    base_left = int(rendering_value(params, "plot_margin_left_px", FALLBACK_CHART_DEFAULTS.plot_margin_left_px))
    base_right = int(rendering_value(params, "plot_margin_right_px", FALLBACK_CHART_DEFAULTS.plot_margin_right_px))
    base_bottom = int(
        rendering_value(params, "plot_margin_bottom_px", FALLBACK_CHART_DEFAULTS.plot_margin_bottom_px)
    )
    placement = str(context_layout.get("placement", "none"))
    if placement == "right_sidebar":
        sidebar_width = int(context_layout.get("sidebar_width_px", 0))
        sidebar_gap = int(context_layout.get("sidebar_gap_px", 14))
        resolved["plot_margin_right_px"] = int(base_right + max(0, sidebar_width) + max(0, sidebar_gap))
        resolved["plot_margin_left_px"] = int(base_left)
        resolved["layout_jitter_x_px"] = 0
    elif placement == "bottom_band":
        bottom_height = int(context_layout.get("bottom_band_height_px", 0))
        bottom_gap = int(context_layout.get("bottom_band_gap_px", 14))
        resolved["plot_margin_bottom_px"] = int(base_bottom + max(0, bottom_height) + max(0, bottom_gap))
        resolved["layout_jitter_y_px"] = 0
    return resolved


__all__ = [
    "apply_context_margin_overrides",
    "choose_context_mode",
    "context_params",
    "context_profile",
    "resolve_context_layout",
]
