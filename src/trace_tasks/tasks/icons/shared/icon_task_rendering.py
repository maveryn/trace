"""Shared render-param and noise helpers for icon scenes."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Mapping, Sequence, Tuple

from ....core.seed import hash64, spawn_rng
from ....core.visual.noise import TRACE_DEFAULT_NOISE_VALUE_RANGES
from ...shared.config_defaults import group_default
from ...shared.render_variation import resolve_render_rgb
from ...shared.text_legibility import (
    resolve_readable_text_style,
    text_legibility_summary_from_records,
)
from .icon_noise import NoiseEdit, sample_icon_noise_edits
from .scene_style import (
    icon_canvas_style_trace,
    icon_canvas_style_with_chrome,
    resolve_icon_canvas_style,
)


def _normalize_noise_value_ranges(raw: Any, fallback: Mapping[str, Mapping[str, Tuple[float, float]]]) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """Normalize per-icon noise value ranges from config-like mappings."""

    if not isinstance(raw, Mapping):
        raw = fallback
    normalized: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for edit_type, param_map in raw.items():
        if not isinstance(param_map, Mapping):
            continue
        edit_key = str(edit_type).strip().lower()
        normalized_params: Dict[str, Tuple[float, float]] = {}
        for param_name, bounds in param_map.items():
            if not isinstance(bounds, (list, tuple)) or len(bounds) < 2:
                continue
            lo = float(bounds[0])
            hi = float(bounds[1])
            normalized_params[str(param_name)] = (min(lo, hi), max(lo, hi))
        if normalized_params:
            normalized[edit_key] = normalized_params
    return normalized


def resolve_icon_rgb_param(
    *,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    key: str,
    fallback: Sequence[int],
    instance_seed: int | None,
) -> Tuple[int, int, int]:
    """Resolve one icon chrome RGB value with optional deterministic palettes."""

    return resolve_render_rgb(
        params,
        render_defaults,
        str(key),
        tuple(int(value) for value in fallback),
        instance_seed=instance_seed,
        namespace="icons.render",
    )


def resolve_icon_render_params(
    *,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    fallback_defaults: Any,
    instance_seed: int | None = None,
) -> Dict[str, Any]:
    """Resolve common rendering params for icon scenes."""

    canvas_style, canvas_style_metadata = resolve_icon_canvas_style(
        params=params,
        render_defaults=render_defaults,
        instance_seed=instance_seed,
    )

    def has_local_rgb_override(key: str) -> bool:
        return params.get(str(key)) is not None or params.get(f"{str(key)}_options") is not None

    def resolve_chrome_rgb(key: str, style_attr: str, fallback: Sequence[int]) -> Tuple[int, int, int]:
        if canvas_style is not None and not has_local_rgb_override(str(key)):
            return tuple(int(value) for value in getattr(canvas_style, str(style_attr)))
        return resolve_icon_rgb_param(
            params=params,
            render_defaults=render_defaults,
            key=str(key),
            fallback=fallback,
            instance_seed=instance_seed,
        )

    render_params: Dict[str, Any] = {
        "canvas_width": int(params.get("canvas_width", group_default(render_defaults, "canvas_width", fallback_defaults.canvas_width))),
        "canvas_height": int(
            params.get("canvas_height", group_default(render_defaults, "canvas_height", fallback_defaults.canvas_height))
        ),
        "reference_panel_width_px": int(
            params.get(
                "reference_panel_width_px",
                group_default(
                    render_defaults,
                    "reference_panel_width_px",
                    getattr(fallback_defaults, "reference_panel_width_px", 0),
                ),
            )
        ),
        "panel_gap_px": int(
            params.get(
                "panel_gap_px",
                group_default(render_defaults, "panel_gap_px", getattr(fallback_defaults, "panel_gap_px", 0)),
            )
        ),
        "outer_margin_px": int(
            params.get("outer_margin_px", group_default(render_defaults, "outer_margin_px", fallback_defaults.outer_margin_px))
        ),
        "panel_padding_px": int(
            params.get("panel_padding_px", group_default(render_defaults, "panel_padding_px", fallback_defaults.panel_padding_px))
        ),
        "panel_corner_radius_px": int(
            params.get(
                "panel_corner_radius_px",
                group_default(render_defaults, "panel_corner_radius_px", fallback_defaults.panel_corner_radius_px),
            )
        ),
        "scene_icon_size_min_px": int(
            params.get(
                "scene_icon_size_min_px",
                group_default(render_defaults, "scene_icon_size_min_px", fallback_defaults.scene_icon_size_min_px),
            )
        ),
        "scene_icon_size_max_px": int(
            params.get(
                "scene_icon_size_max_px",
                group_default(render_defaults, "scene_icon_size_max_px", fallback_defaults.scene_icon_size_max_px),
            )
        ),
        "reference_icon_size_px": int(
            params.get(
                "reference_icon_size_px",
                group_default(
                    render_defaults,
                    "reference_icon_size_px",
                    getattr(fallback_defaults, "reference_icon_size_px", 0),
                ),
            )
        ),
        "reference_icon_size_min_px": int(
            params.get(
                "reference_icon_size_min_px",
                group_default(
                    render_defaults,
                    "reference_icon_size_min_px",
                    getattr(
                        fallback_defaults,
                        "reference_icon_size_min_px",
                        getattr(fallback_defaults, "reference_icon_size_px", 0),
                    ),
                ),
            )
        ),
        "reference_icon_size_max_px": int(
            params.get(
                "reference_icon_size_max_px",
                group_default(
                    render_defaults,
                    "reference_icon_size_max_px",
                    getattr(
                        fallback_defaults,
                        "reference_icon_size_max_px",
                        getattr(fallback_defaults, "reference_icon_size_px", 0),
                    ),
                ),
            )
        ),
        "scene_max_overlap_fraction": float(
            params.get(
                "scene_max_overlap_fraction",
                group_default(
                    render_defaults,
                    "scene_max_overlap_fraction",
                    getattr(fallback_defaults, "scene_max_overlap_fraction", 0.10),
                ),
            )
        ),
        "scene_placement_max_attempts": int(
            params.get(
                "scene_placement_max_attempts",
                group_default(
                    render_defaults,
                    "scene_placement_max_attempts",
                    getattr(fallback_defaults, "scene_placement_max_attempts", 120),
                ),
            )
        ),
        "scene_size_shrink_rounds": int(
            params.get(
                "scene_size_shrink_rounds",
                group_default(
                    render_defaults,
                    "scene_size_shrink_rounds",
                    getattr(fallback_defaults, "scene_size_shrink_rounds", 6),
                ),
            )
        ),
        "scene_size_shrink_factor": float(
            params.get(
                "scene_size_shrink_factor",
                group_default(
                    render_defaults,
                    "scene_size_shrink_factor",
                    getattr(fallback_defaults, "scene_size_shrink_factor", 0.90),
                ),
            )
        ),
        "palette_size_min": int(
            params.get("palette_size_min", group_default(render_defaults, "palette_size_min", fallback_defaults.palette_size_min))
        ),
        "palette_size_max": int(
            params.get("palette_size_max", group_default(render_defaults, "palette_size_max", fallback_defaults.palette_size_max))
        ),
        "color_channel_min": int(
            params.get("color_channel_min", group_default(render_defaults, "color_channel_min", fallback_defaults.color_channel_min))
        ),
        "color_channel_max": int(
            params.get("color_channel_max", group_default(render_defaults, "color_channel_max", fallback_defaults.color_channel_max))
        ),
        "min_color_distance": float(
            params.get("min_color_distance", group_default(render_defaults, "min_color_distance", fallback_defaults.min_color_distance))
        ),
        "color_distance_space": str(
            params.get(
                "color_distance_space",
                group_default(render_defaults, "color_distance_space", fallback_defaults.color_distance_space),
            )
        ),
        "panel_title_font_size_px": int(
            params.get(
                "panel_title_font_size_px",
                group_default(render_defaults, "panel_title_font_size_px", fallback_defaults.panel_title_font_size_px),
            )
        ),
        "background_color_rgb": resolve_chrome_rgb(
            "background_color_rgb",
            "background_rgb",
            fallback_defaults.background_color_rgb,
        ),
        "panel_fill_rgb": resolve_chrome_rgb(
            "panel_fill_rgb",
            "panel_fill_rgb",
            fallback_defaults.panel_fill_rgb,
        ),
        "panel_border_rgb": resolve_chrome_rgb(
            "panel_border_rgb",
            "panel_border_rgb",
            fallback_defaults.panel_border_rgb,
        ),
        "header_text_rgb": resolve_chrome_rgb(
            "header_text_rgb",
            "text_rgb",
            fallback_defaults.header_text_rgb,
        ),
        "icon_noise_edit_types": tuple(
            str(value).strip().lower()
            for value in params.get(
                "icon_noise_edit_types",
                group_default(
                    render_defaults,
                    "icon_noise_edit_types",
                    getattr(fallback_defaults, "icon_noise_edit_types", ("blur", "downsample", "jpeg", "noise")),
                ),
            )
            if str(value).strip()
        ),
        "icon_noise_edit_count_range": tuple(
            int(value)
            for value in params.get(
                "icon_noise_edit_count_range",
                group_default(
                    render_defaults,
                    "icon_noise_edit_count_range",
                    getattr(fallback_defaults, "icon_noise_edit_count_range", (0, 2)),
                ),
            )
        ),
        "icon_noise_value_ranges": _normalize_noise_value_ranges(
            params.get(
                "icon_noise_value_ranges",
                group_default(
                    render_defaults,
                    "icon_noise_value_ranges",
                    deepcopy(
                        getattr(
                            fallback_defaults,
                            "icon_noise_value_ranges",
                            TRACE_DEFAULT_NOISE_VALUE_RANGES,
                        )
                    ),
                ),
            ),
            fallback=deepcopy(
                getattr(
                    fallback_defaults,
                    "icon_noise_value_ranges",
                    TRACE_DEFAULT_NOISE_VALUE_RANGES,
                )
            ),
        ),
    }
    header_style = resolve_readable_text_style(
        instance_seed=int(instance_seed or 0),
        namespace="icons.panel_header_text",
        role="icon_panel_header_text",
        surface_rgbs=(
            tuple(int(value) for value in render_params["panel_fill_rgb"]),
            tuple(int(value) for value in render_params["background_color_rgb"]),
        ),
        preferred_rgbs=(tuple(int(value) for value in render_params["header_text_rgb"]),),
    )
    render_params["header_text_rgb"] = tuple(int(value) for value in header_style.fill_rgb)
    render_params["header_text_stroke_rgb"] = tuple(int(value) for value in render_params["panel_fill_rgb"])
    header_record = header_style.metadata()
    header_record["stroke_rgb"] = list(render_params["header_text_stroke_rgb"])
    render_params["text_legibility"] = text_legibility_summary_from_records([header_record])
    render_params["text_color_policy"] = "read_required_text_uses_random_nonsemantic_readable_ink"
    resolved_canvas_style = icon_canvas_style_with_chrome(
        canvas_style,
        background_rgb=render_params["background_color_rgb"],
        panel_fill_rgb=render_params["panel_fill_rgb"],
        panel_border_rgb=render_params["panel_border_rgb"],
        header_text_rgb=render_params["header_text_rgb"],
        header_text_stroke_rgb=render_params["header_text_stroke_rgb"],
    )
    render_params["_icon_canvas_style_object"] = resolved_canvas_style
    render_params["icon_canvas_style"] = icon_canvas_style_trace(
        resolved_canvas_style,
        canvas_style_metadata,
    )
    return render_params


def resolve_icon_cell_render_params(
    *,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    fallback_defaults: Any,
    instance_seed: int | None = None,
) -> Dict[str, Any]:
    """Resolve shared render params for icon tasks built from labeled cells."""

    render_params = resolve_icon_render_params(
        params=params,
        render_defaults=render_defaults,
        fallback_defaults=fallback_defaults,
        instance_seed=instance_seed,
    )
    render_params["cell_padding_px"] = int(
        params.get(
            "cell_padding_px",
            group_default(render_defaults, "cell_padding_px", getattr(fallback_defaults, "cell_padding_px", 0)),
        )
    )
    render_params["cell_icon_padding_px"] = int(
        params.get(
            "cell_icon_padding_px",
            group_default(
                render_defaults,
                "cell_icon_padding_px",
                getattr(fallback_defaults, "cell_icon_padding_px", 0),
            ),
        )
    )
    render_params["cell_corner_radius_px"] = int(
        params.get(
            "cell_corner_radius_px",
            group_default(
                render_defaults,
                "cell_corner_radius_px",
                getattr(fallback_defaults, "cell_corner_radius_px", 0),
            ),
        )
    )
    render_params["cell_box_width_min_px"] = int(
        params.get(
            "cell_box_width_min_px",
            group_default(
                render_defaults,
                "cell_box_width_min_px",
                getattr(fallback_defaults, "cell_box_width_min_px", 0),
            ),
        )
    )
    render_params["cell_box_width_max_px"] = int(
        params.get(
            "cell_box_width_max_px",
            group_default(
                render_defaults,
                "cell_box_width_max_px",
                getattr(fallback_defaults, "cell_box_width_max_px", 0),
            ),
        )
    )
    render_params["cell_box_height_min_px"] = int(
        params.get(
            "cell_box_height_min_px",
            group_default(
                render_defaults,
                "cell_box_height_min_px",
                getattr(fallback_defaults, "cell_box_height_min_px", 0),
            ),
        )
    )
    render_params["cell_box_height_max_px"] = int(
        params.get(
            "cell_box_height_max_px",
            group_default(
                render_defaults,
                "cell_box_height_max_px",
                getattr(fallback_defaults, "cell_box_height_max_px", 0),
            ),
        )
    )
    render_params["cell_border_rgb"] = resolve_icon_rgb_param(
        params=params,
        render_defaults=render_defaults,
        key="cell_border_rgb",
        fallback=getattr(fallback_defaults, "cell_border_rgb", (218, 223, 233)),
        instance_seed=instance_seed,
    )
    render_params["cell_label_font_size_px"] = int(
        params.get(
            "cell_label_font_size_px",
            group_default(
                render_defaults,
                "cell_label_font_size_px",
                getattr(fallback_defaults, "cell_label_font_size_px", 0),
            ),
        )
    )
    render_params["cell_label_color_rgb"] = resolve_icon_rgb_param(
        params=params,
        render_defaults=render_defaults,
        key="cell_label_color_rgb",
        fallback=getattr(fallback_defaults, "cell_label_color_rgb", getattr(fallback_defaults, "header_text_rgb", (70, 78, 96))),
        instance_seed=instance_seed,
    )
    cell_label_style = resolve_readable_text_style(
        instance_seed=int(instance_seed or 0),
        namespace="icons.cell_label_text",
        role="icon_cell_label_text",
        surface_rgbs=(
            tuple(int(value) for value in render_params["panel_fill_rgb"]),
            tuple(int(value) for value in render_params["background_color_rgb"]),
        ),
        preferred_rgbs=(tuple(int(value) for value in render_params["cell_label_color_rgb"]),),
    )
    render_params["cell_label_color_rgb"] = tuple(int(value) for value in cell_label_style.fill_rgb)
    render_params["cell_label_stroke_rgb"] = tuple(int(value) for value in render_params["panel_fill_rgb"])
    cell_label_record = cell_label_style.metadata()
    cell_label_record["stroke_rgb"] = list(render_params["cell_label_stroke_rgb"])
    previous_legibility = render_params.get("text_legibility")
    previous_records = []
    if isinstance(previous_legibility, Mapping) and isinstance(previous_legibility.get("records"), list):
        previous_records = [dict(record) for record in previous_legibility["records"] if isinstance(record, Mapping)]
    render_params["text_legibility"] = text_legibility_summary_from_records(
        [*previous_records, cell_label_record]
    )
    render_params["scene_content_side_padding_px"] = int(
        params.get(
            "scene_content_side_padding_px",
            group_default(
                render_defaults,
                "scene_content_side_padding_px",
                getattr(fallback_defaults, "scene_content_side_padding_px", 12),
            ),
        )
    )
    render_params["scene_content_bottom_padding_px"] = int(
        params.get(
            "scene_content_bottom_padding_px",
            group_default(
                render_defaults,
                "scene_content_bottom_padding_px",
                getattr(fallback_defaults, "scene_content_bottom_padding_px", 12),
            ),
        )
    )
    render_params["scene_content_top_offset_px"] = int(
        params.get(
            "scene_content_top_offset_px",
            group_default(
                render_defaults,
                "scene_content_top_offset_px",
                getattr(fallback_defaults, "scene_content_top_offset_px", 40),
            ),
        )
    )
    render_params["missing_mark_font_size_px"] = int(
        params.get(
            "missing_mark_font_size_px",
            group_default(
                render_defaults,
                "missing_mark_font_size_px",
                getattr(fallback_defaults, "missing_mark_font_size_px", 0),
            ),
        )
    )
    render_params["missing_mark_color_rgb"] = resolve_icon_rgb_param(
        params=params,
        render_defaults=render_defaults,
        key="missing_mark_color_rgb",
        fallback=getattr(fallback_defaults, "missing_mark_color_rgb", getattr(fallback_defaults, "header_text_rgb", (70, 78, 96))),
        instance_seed=instance_seed,
    )
    return render_params


def icon_render_style_trace(
    *,
    render_params: Mapping[str, Any],
    sampled_palette_rgb: Tuple[Tuple[int, int, int], ...],
) -> Dict[str, Any]:
    """Return the canonical render-style trace block for icon tasks."""

    style_trace = {
        "background_color_rgb": list(render_params["background_color_rgb"]),
        "panel_fill_rgb": list(render_params["panel_fill_rgb"]),
        "panel_border_rgb": list(render_params["panel_border_rgb"]),
        "header_text_rgb": list(render_params["header_text_rgb"]),
        "header_text_stroke_rgb": list(render_params.get("header_text_stroke_rgb", (255, 255, 255))),
        "text_color_policy": str(
            render_params.get("text_color_policy", "read_required_text_uses_random_nonsemantic_readable_ink")
        ),
        "text_legibility": dict(render_params.get("text_legibility", {})),
        "icon_canvas_style": dict(render_params.get("icon_canvas_style", {"enabled": False})),
        "palette_size_min": int(render_params["palette_size_min"]),
        "palette_size_max": int(render_params["palette_size_max"]),
        "sampled_palette_rgb": [list(color) for color in sampled_palette_rgb],
        "color_channel_min": int(render_params["color_channel_min"]),
        "color_channel_max": int(render_params["color_channel_max"]),
        "min_color_distance": float(render_params["min_color_distance"]),
        "color_distance_space": str(render_params["color_distance_space"]),
        "scene_max_overlap_fraction": float(render_params["scene_max_overlap_fraction"]),
        "scene_placement_max_attempts": int(render_params["scene_placement_max_attempts"]),
        "scene_size_shrink_rounds": int(render_params["scene_size_shrink_rounds"]),
        "scene_size_shrink_factor": float(render_params["scene_size_shrink_factor"]),
        "scene_icon_size_min_px": int(render_params["scene_icon_size_min_px"]),
        "scene_icon_size_max_px": int(render_params["scene_icon_size_max_px"]),
        "reference_icon_size_px": int(render_params["reference_icon_size_px"]),
        "reference_icon_size_min_px": int(render_params["reference_icon_size_min_px"]),
        "reference_icon_size_max_px": int(render_params["reference_icon_size_max_px"]),
        "icon_noise_edit_types": [str(value) for value in render_params["icon_noise_edit_types"]],
        "icon_noise_edit_count_range": [
            int(render_params["icon_noise_edit_count_range"][0]),
            int(render_params["icon_noise_edit_count_range"][1]),
        ],
        "icon_noise_value_ranges": {
            str(edit_type): {
                str(param): [float(bounds[0]), float(bounds[1])]
                for param, bounds in params.items()
            }
            for edit_type, params in render_params["icon_noise_value_ranges"].items()
        },
    }
    if "cell_label_color_rgb" in render_params:
        style_trace["cell_label_color_rgb"] = list(render_params["cell_label_color_rgb"])
    if "cell_label_stroke_rgb" in render_params:
        style_trace["cell_label_stroke_rgb"] = list(render_params["cell_label_stroke_rgb"])
    return style_trace


def sample_icon_instance_noise(
    *,
    instance_seed: int,
    namespace: str,
    render_params: Mapping[str, Any],
) -> Tuple[Tuple[NoiseEdit, ...], int]:
    """Sample one deterministic per-icon noise payload and return its apply seed."""

    sample_rng = spawn_rng(int(instance_seed), f"{namespace}:sample")
    edits = sample_icon_noise_edits(
        sample_rng,
        edit_types=tuple(render_params["icon_noise_edit_types"]),
        edit_value_ranges=render_params["icon_noise_value_ranges"],
        edit_count_range=tuple(render_params["icon_noise_edit_count_range"]),
    )
    apply_seed = int(hash64(int(instance_seed), f"{namespace}:apply"))
    return tuple(edits), int(apply_seed)


__all__ = [
    "icon_render_style_trace",
    "resolve_icon_cell_render_params",
    "resolve_icon_render_params",
    "resolve_icon_rgb_param",
    "sample_icon_instance_noise",
]
