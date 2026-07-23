"""Shared visual style resolvers for three_d scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.shared.color_distance import coerce_rgb
from trace_tasks.tasks.shared.visual_style.surface_tones import (
    DEFAULT_SURFACE_TONES as DEFAULT_THREE_D_SURFACE_TONES,
    LIGHT_SURFACE_TEXT_RGB,
    LIGHT_SURFACE_TEXT_STROKE_RGB,
)


RGB = Tuple[int, int, int]


@dataclass(frozen=True)
class ThreeDSurfaceTone:
    """Resolved neutral surface palette for one three_d render."""

    tone_id: str
    floor_rgb: RGB
    grid_rgb: RGB
    edge_rgb: RGB
    text_rgb: RGB
    text_stroke_rgb: RGB
    surface_accent_rgb: RGB

    def trace_metadata(self) -> Dict[str, Any]:
        return {
            "background_tone_id": str(self.tone_id),
            "background_tone_rgb": list(self.floor_rgb),
            "background_grid_rgb": list(self.grid_rgb),
            "background_edge_rgb": list(self.edge_rgb),
            "background_surface_accent_rgb": list(self.surface_accent_rgb),
        }


@dataclass(frozen=True)
class ConveyorBeltStyle:
    """Resolved belt palette for straight and carousel conveyor scenes."""

    style_id: str
    fill_rgb: RGB
    fill_alt_rgb: RGB
    fill_secondary_rgb: RGB
    outline_rgb: RGB
    outline_secondary_rgb: RGB
    rail_rgb: RGB
    arrow_rgb: RGB
    shadow_rgb: RGB

    def trace_metadata(self) -> Dict[str, Any]:
        return {
            "conveyor_belt_style_id": str(self.style_id),
            "conveyor_belt_fill_rgb": list(self.fill_rgb),
            "conveyor_belt_fill_alt_rgb": list(self.fill_alt_rgb),
            "conveyor_belt_fill_secondary_rgb": list(self.fill_secondary_rgb),
            "conveyor_belt_outline_rgb": list(self.outline_rgb),
            "conveyor_belt_arrow_rgb": list(self.arrow_rgb),
        }



DEFAULT_CONVEYOR_BELT_STYLES: Mapping[str, Mapping[str, RGB]] = {
    "galvanized_steel": {
        "fill_rgb": (196, 211, 222),
        "fill_alt_rgb": (205, 219, 229),
        "fill_secondary_rgb": (188, 204, 216),
        "outline_rgb": (75, 91, 108),
        "outline_secondary_rgb": (87, 103, 121),
        "rail_rgb": (112, 126, 139),
        "arrow_rgb": (132, 145, 157),
        "shadow_rgb": (94, 104, 114),
    },
    "dark_rubber": {
        "fill_rgb": (112, 119, 125),
        "fill_alt_rgb": (124, 132, 138),
        "fill_secondary_rgb": (103, 111, 118),
        "outline_rgb": (48, 55, 62),
        "outline_secondary_rgb": (58, 66, 74),
        "rail_rgb": (92, 100, 108),
        "arrow_rgb": (172, 180, 186),
        "shadow_rgb": (70, 78, 86),
    },
    "airport_silver": {
        "fill_rgb": (211, 219, 224),
        "fill_alt_rgb": (221, 228, 232),
        "fill_secondary_rgb": (201, 211, 218),
        "outline_rgb": (89, 102, 113),
        "outline_secondary_rgb": (101, 114, 125),
        "rail_rgb": (134, 145, 154),
        "arrow_rgb": (143, 154, 164),
        "shadow_rgb": (104, 112, 121),
    },
    "blue_gray_belt": {
        "fill_rgb": (169, 190, 205),
        "fill_alt_rgb": (181, 202, 216),
        "fill_secondary_rgb": (158, 179, 195),
        "outline_rgb": (66, 84, 100),
        "outline_secondary_rgb": (76, 94, 111),
        "rail_rgb": (101, 119, 135),
        "arrow_rgb": (120, 136, 150),
        "shadow_rgb": (84, 97, 108),
    },
    "muted_teal": {
        "fill_rgb": (164, 194, 193),
        "fill_alt_rgb": (176, 205, 204),
        "fill_secondary_rgb": (153, 183, 183),
        "outline_rgb": (62, 92, 94),
        "outline_secondary_rgb": (75, 104, 106),
        "rail_rgb": (98, 126, 127),
        "arrow_rgb": (118, 143, 144),
        "shadow_rgb": (80, 100, 102),
    },
    "graphite": {
        "fill_rgb": (132, 141, 149),
        "fill_alt_rgb": (145, 154, 162),
        "fill_secondary_rgb": (121, 130, 139),
        "outline_rgb": (55, 63, 72),
        "outline_secondary_rgb": (66, 74, 84),
        "rail_rgb": (103, 112, 121),
        "arrow_rgb": (168, 176, 184),
        "shadow_rgb": (76, 84, 93),
    },
    "olive_gray": {
        "fill_rgb": (174, 188, 170),
        "fill_alt_rgb": (186, 199, 181),
        "fill_secondary_rgb": (162, 177, 159),
        "outline_rgb": (76, 92, 73),
        "outline_secondary_rgb": (89, 105, 86),
        "rail_rgb": (112, 127, 108),
        "arrow_rgb": (135, 147, 130),
        "shadow_rgb": (85, 96, 84),
    },
    "tan_industrial": {
        "fill_rgb": (195, 183, 158),
        "fill_alt_rgb": (207, 196, 171),
        "fill_secondary_rgb": (183, 172, 149),
        "outline_rgb": (97, 84, 66),
        "outline_secondary_rgb": (110, 97, 78),
        "rail_rgb": (136, 124, 101),
        "arrow_rgb": (151, 139, 117),
        "shadow_rgb": (104, 94, 79),
    },
    "pale_metal": {
        "fill_rgb": (220, 226, 226),
        "fill_alt_rgb": (230, 235, 235),
        "fill_secondary_rgb": (210, 218, 219),
        "outline_rgb": (94, 105, 107),
        "outline_secondary_rgb": (106, 117, 119),
        "rail_rgb": (143, 153, 154),
        "arrow_rgb": (151, 160, 161),
        "shadow_rgb": (111, 119, 120),
    },
    "slate_rail": {
        "fill_rgb": (181, 194, 204),
        "fill_alt_rgb": (193, 206, 215),
        "fill_secondary_rgb": (170, 184, 195),
        "outline_rgb": (63, 78, 92),
        "outline_secondary_rgb": (75, 90, 105),
        "rail_rgb": (82, 97, 112),
        "arrow_rgb": (127, 141, 154),
        "shadow_rgb": (83, 96, 107),
    },
}


def _normalize_rgb_mapping(value: Any, fallback: RGB) -> RGB:
    return coerce_rgb(value, fallback)


def _style_mapping(config: Mapping[str, Any], *, key: str, defaults: Mapping[str, Mapping[str, RGB]]) -> Dict[str, Dict[str, RGB]]:
    raw = config.get(str(key))
    if not isinstance(raw, Mapping):
        return {str(style_id): dict(spec) for style_id, spec in defaults.items()}
    styles = raw.get("styles", raw)
    if not isinstance(styles, Mapping):
        return {str(style_id): dict(spec) for style_id, spec in defaults.items()}
    out: Dict[str, Dict[str, RGB]] = {}
    for style_id, spec in styles.items():
        if not isinstance(spec, Mapping):
            continue
        default_spec = dict(defaults.get(str(style_id), next(iter(defaults.values()))))
        normalized: Dict[str, RGB] = {}
        for field, fallback in default_spec.items():
            normalized[str(field)] = _normalize_rgb_mapping(spec.get(str(field), fallback), fallback)
        out[str(style_id)] = dict(normalized)
    return out or {str(style_id): dict(spec) for style_id, spec in defaults.items()}


def _weights(config: Mapping[str, Any], *, key: str, style_ids: Tuple[str, ...]) -> Dict[str, float]:
    raw_section = config.get(str(key))
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
        style_id = str(explicit_id)
        if style_id not in styles:
            raise ValueError(f"unsupported three_d visual style id: {style_id}")
        return str(style_id)
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


def resolve_three_d_surface_tone(
    *,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
) -> ThreeDSurfaceTone:
    """Resolve one neutral surface tone for a three_d render."""

    merged = dict(render_defaults)
    merged.update(dict(params))
    explicit_floor = params.get("floor_rgb")
    explicit_tone = params.get("surface_tone_id", params.get("background_tone_id"))
    if explicit_floor is not None and explicit_tone is None:
        floor = _normalize_rgb_mapping(explicit_floor, (232, 239, 242))
        grid = _normalize_rgb_mapping(params.get("grid_rgb", merged.get("grid_rgb", (184, 197, 207))), (184, 197, 207))
        edge = _normalize_rgb_mapping(params.get("edge_rgb", merged.get("edge_rgb", (93, 108, 124))), (93, 108, 124))
        return ThreeDSurfaceTone(
            tone_id="custom",
            floor_rgb=floor,
            grid_rgb=grid,
            edge_rgb=edge,
            text_rgb=_normalize_rgb_mapping(params.get("text_rgb", merged.get("text_rgb", (30, 34, 42))), (30, 34, 42)),
            text_stroke_rgb=_normalize_rgb_mapping(
                params.get("text_stroke_rgb", merged.get("text_stroke_rgb", (255, 255, 255))),
                (255, 255, 255),
            ),
            surface_accent_rgb=_normalize_rgb_mapping(params.get("surface_accent_rgb", grid), grid),
        )

    styles = _style_mapping(merged, key="surface_tones", defaults=DEFAULT_THREE_D_SURFACE_TONES)
    weights = _weights(merged, key="surface_tones", style_ids=tuple(styles))
    tone_id = _select_style_id(
        explicit_id=explicit_tone,
        styles=styles,
        weights=weights,
        instance_seed=int(instance_seed),
        namespace=str(namespace),
    )
    spec = dict(styles[str(tone_id)])
    floor = _normalize_rgb_mapping(params.get("floor_rgb", spec["floor_rgb"]), spec["floor_rgb"])
    grid = _normalize_rgb_mapping(params.get("grid_rgb", spec["grid_rgb"]), spec["grid_rgb"])
    edge = _normalize_rgb_mapping(params.get("edge_rgb", spec["edge_rgb"]), spec["edge_rgb"])
    return ThreeDSurfaceTone(
        tone_id=str(tone_id),
        floor_rgb=floor,
        grid_rgb=grid,
        edge_rgb=edge,
        text_rgb=_normalize_rgb_mapping(
            params.get("text_rgb", merged.get("text_rgb", spec.get("text_rgb", LIGHT_SURFACE_TEXT_RGB))),
            spec.get("text_rgb", LIGHT_SURFACE_TEXT_RGB),
        ),
        text_stroke_rgb=_normalize_rgb_mapping(
            params.get("text_stroke_rgb", merged.get("text_stroke_rgb", spec.get("text_stroke_rgb", LIGHT_SURFACE_TEXT_STROKE_RGB))),
            spec.get("text_stroke_rgb", LIGHT_SURFACE_TEXT_STROKE_RGB),
        ),
        surface_accent_rgb=_normalize_rgb_mapping(params.get("surface_accent_rgb", spec["surface_accent_rgb"]), spec["surface_accent_rgb"]),
    )


def resolve_conveyor_belt_style(
    *,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
) -> ConveyorBeltStyle:
    """Resolve one realistic conveyor belt style for conveyor scenes."""

    merged = dict(render_defaults)
    merged.update(dict(params))
    styles = _style_mapping(merged, key="conveyor_belt_styles", defaults=DEFAULT_CONVEYOR_BELT_STYLES)
    weights = _weights(merged, key="conveyor_belt_styles", style_ids=tuple(styles))
    style_id = _select_style_id(
        explicit_id=params.get("conveyor_belt_style_id", params.get("belt_style_id")),
        styles=styles,
        weights=weights,
        instance_seed=int(instance_seed),
        namespace=str(namespace),
    )
    spec = dict(styles[str(style_id)])
    return ConveyorBeltStyle(
        style_id=str(style_id),
        fill_rgb=_normalize_rgb_mapping(params.get("belt_fill_rgb", spec["fill_rgb"]), spec["fill_rgb"]),
        fill_alt_rgb=_normalize_rgb_mapping(params.get("belt_fill_alt_rgb", spec["fill_alt_rgb"]), spec["fill_alt_rgb"]),
        fill_secondary_rgb=_normalize_rgb_mapping(
            params.get("belt_fill_secondary_rgb", spec["fill_secondary_rgb"]),
            spec["fill_secondary_rgb"],
        ),
        outline_rgb=_normalize_rgb_mapping(params.get("belt_outline_rgb", spec["outline_rgb"]), spec["outline_rgb"]),
        outline_secondary_rgb=_normalize_rgb_mapping(
            params.get("belt_outline_secondary_rgb", spec["outline_secondary_rgb"]),
            spec["outline_secondary_rgb"],
        ),
        rail_rgb=_normalize_rgb_mapping(params.get("belt_rail_rgb", spec["rail_rgb"]), spec["rail_rgb"]),
        arrow_rgb=_normalize_rgb_mapping(params.get("belt_arrow_rgb", spec["arrow_rgb"]), spec["arrow_rgb"]),
        shadow_rgb=_normalize_rgb_mapping(params.get("belt_shadow_rgb", spec["shadow_rgb"]), spec["shadow_rgb"]),
    )


__all__ = [
    "ConveyorBeltStyle",
    "DEFAULT_CONVEYOR_BELT_STYLES",
    "DEFAULT_THREE_D_SURFACE_TONES",
    "ThreeDSurfaceTone",
    "resolve_conveyor_belt_style",
    "resolve_three_d_surface_tone",
]
