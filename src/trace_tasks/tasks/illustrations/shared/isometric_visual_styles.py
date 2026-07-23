"""Shared visual style adapter for isometric illustration scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.shared.color_distance import coerce_rgb
from trace_tasks.tasks.shared.visual_style.surface_tones import DARK_SURFACE_TONE_IDS, DEFAULT_SURFACE_TONES


RGB = Tuple[int, int, int]


@dataclass(frozen=True)
class IsometricIllustrationTone:
    """Resolved neutral treatment mapped onto an isometric illustration renderer."""

    tone_id: str
    canvas_rgb: RGB
    terrain_edge_rgb: RGB
    terrain_shadow_rgb: RGB
    terrain_light_rgb: RGB
    label_fill_rgb: RGB
    label_outline_rgb: RGB
    label_text_rgb: RGB
    label_stroke_rgb: RGB
    ambient_tint_rgb: RGB
    semantic_tint_strength: float
    is_dark: bool

    def trace_metadata(self) -> Dict[str, Any]:
        return {
            "background_tone_id": str(self.tone_id),
            "background_tone_rgb": list(self.canvas_rgb),
            "background_rgb": list(self.canvas_rgb),
            "background_edge_rgb": list(self.terrain_edge_rgb),
            "background_shadow_rgb": list(self.terrain_shadow_rgb),
            "background_light_rgb": list(self.terrain_light_rgb),
            "label_fill_rgb": list(self.label_fill_rgb),
            "label_outline_rgb": list(self.label_outline_rgb),
            "label_text_rgb": list(self.label_text_rgb),
            "label_stroke_rgb": list(self.label_stroke_rgb),
            "ambient_tint_rgb": list(self.ambient_tint_rgb),
            "semantic_tint_strength": round(float(self.semantic_tint_strength), 4),
            "background_tone_family": "dark" if bool(self.is_dark) else "light",
        }


def _blend(left: RGB, right: RGB, weight: float) -> RGB:
    clipped = max(0.0, min(1.0, float(weight)))
    return tuple(
        int(round(float(left[index]) * (1.0 - clipped) + float(right[index]) * clipped))
        for index in range(3)
    )


def _shade(color: RGB, delta: int) -> RGB:
    return tuple(max(0, min(255, int(channel) + int(delta))) for channel in color)


def _normalize_rgb(value: Any, fallback: RGB) -> RGB:
    return coerce_rgb(value, fallback)


def _style_mapping(config: Mapping[str, Any]) -> Dict[str, Dict[str, RGB]]:
    raw = config.get("surface_tones")
    if not isinstance(raw, Mapping):
        return {str(style_id): dict(spec) for style_id, spec in DEFAULT_SURFACE_TONES.items()}
    styles = raw.get("styles", raw)
    if not isinstance(styles, Mapping):
        return {str(style_id): dict(spec) for style_id, spec in DEFAULT_SURFACE_TONES.items()}

    out: Dict[str, Dict[str, RGB]] = {}
    for style_id, spec in styles.items():
        if not isinstance(spec, Mapping):
            continue
        default_spec = dict(DEFAULT_SURFACE_TONES.get(str(style_id), next(iter(DEFAULT_SURFACE_TONES.values()))))
        normalized: Dict[str, RGB] = {}
        for field, fallback in default_spec.items():
            normalized[str(field)] = _normalize_rgb(spec.get(str(field), fallback), fallback)
        out[str(style_id)] = dict(normalized)
    return out or {str(style_id): dict(spec) for style_id, spec in DEFAULT_SURFACE_TONES.items()}


def _weights(config: Mapping[str, Any], *, style_ids: Tuple[str, ...]) -> Dict[str, float]:
    raw_section = config.get("surface_tones")
    raw_weights = raw_section.get("weights") if isinstance(raw_section, Mapping) else None
    if not isinstance(raw_weights, Mapping):
        return {str(style_id): 1.0 for style_id in style_ids}

    weights: Dict[str, float] = {}
    for style_id in style_ids:
        value = raw_weights.get(str(style_id), 0.0)
        try:
            numeric = float(value)
        except Exception:
            numeric = 0.0
        if numeric > 0.0:
            weights[str(style_id)] = float(numeric)
    return weights or {str(style_id): 1.0 for style_id in style_ids}


def _select_style_id(
    *,
    explicit_id: Any,
    styles: Mapping[str, Mapping[str, RGB]],
    weights: Mapping[str, float],
    instance_seed: int,
    namespace: str,
) -> str:
    if explicit_id is not None:
        tone_id = str(explicit_id)
        if tone_id not in styles:
            raise ValueError(f"unsupported isometric illustration tone id: {tone_id}")
        return str(tone_id)

    ordered = [(str(style_id), float(weights.get(str(style_id), 0.0))) for style_id in sorted(styles)]
    ordered = [(style_id, weight) for style_id, weight in ordered if weight > 0.0]
    if not ordered:
        ordered = [(str(style_id), 1.0) for style_id in sorted(styles)]
    total = sum(weight for _style_id, weight in ordered)
    rng = spawn_rng(int(instance_seed), str(namespace))
    cursor = float(rng.random()) * float(total)
    running = 0.0
    for style_id, weight in ordered:
        running += float(weight)
        if cursor <= running:
            return str(style_id)
    return str(ordered[-1][0])


def resolve_isometric_illustration_tone(
    *,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
) -> IsometricIllustrationTone:
    """Resolve one shared neutral tone for an isometric illustration scene."""

    merged = dict(render_defaults)
    merged.update(dict(params))
    styles = _style_mapping(merged)
    weights = _weights(merged, style_ids=tuple(styles))
    tone_id = _select_style_id(
        explicit_id=params.get("isometric_tone_id", params.get("background_tone_id", params.get("surface_tone_id"))),
        styles=styles,
        weights=weights,
        instance_seed=int(instance_seed),
        namespace=str(namespace),
    )
    spec = dict(styles[str(tone_id)])
    floor = _normalize_rgb(params.get("background_rgb", params.get("canvas_rgb", spec["floor_rgb"])), spec["floor_rgb"])
    grid = _normalize_rgb(params.get("terrain_shadow_rgb", spec["grid_rgb"]), spec["grid_rgb"])
    edge = _normalize_rgb(params.get("terrain_edge_rgb", spec["edge_rgb"]), spec["edge_rgb"])
    accent = _normalize_rgb(params.get("terrain_light_rgb", spec["surface_accent_rgb"]), spec["surface_accent_rgb"])
    text = _normalize_rgb(params.get("label_text_rgb", spec["text_rgb"]), spec["text_rgb"])
    stroke = _normalize_rgb(params.get("label_stroke_rgb", spec["text_stroke_rgb"]), spec["text_stroke_rgb"])
    is_dark = str(tone_id) in DARK_SURFACE_TONE_IDS
    label_fill = _blend(accent, floor, 0.25) if is_dark else _blend((255, 255, 244), floor, 0.18)
    ambient = accent if is_dark else floor
    return IsometricIllustrationTone(
        tone_id=str(tone_id),
        canvas_rgb=floor,
        terrain_edge_rgb=edge,
        terrain_shadow_rgb=grid,
        terrain_light_rgb=accent,
        label_fill_rgb=label_fill,
        label_outline_rgb=edge,
        label_text_rgb=text,
        label_stroke_rgb=stroke,
        ambient_tint_rgb=ambient,
        semantic_tint_strength=0.045 if is_dark else 0.08,
        is_dark=bool(is_dark),
    )


def tint_isometric_semantic_rgb(color: RGB, tone: IsometricIllustrationTone, *, strength: float | None = None) -> RGB:
    """Lightly tint a semantic scene color while preserving its category identity."""

    weight = float(tone.semantic_tint_strength if strength is None else strength)
    return _blend(tuple(int(value) for value in color), tuple(tone.ambient_tint_rgb), weight)


def isometric_terrain_triplet(
    fill_rgb: RGB,
    tone: IsometricIllustrationTone,
    *,
    shadow_delta: int = -48,
    light_delta: int = 38,
) -> Tuple[RGB, RGB, RGB]:
    """Return fill/shadow/light colors for one semantic isometric tile."""

    fill = tint_isometric_semantic_rgb(fill_rgb, tone)
    dark = _blend(_shade(fill, int(shadow_delta)), tone.terrain_shadow_rgb, 0.2)
    light = _blend(_shade(fill, int(light_delta)), tone.terrain_light_rgb, 0.16)
    return fill, dark, light


__all__ = [
    "IsometricIllustrationTone",
    "isometric_terrain_triplet",
    "resolve_isometric_illustration_tone",
    "tint_isometric_semantic_rgb",
]
