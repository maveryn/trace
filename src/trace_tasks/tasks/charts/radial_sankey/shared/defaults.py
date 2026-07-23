"""Configuration defaults and render parameter resolution for radial Sankey charts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from trace_tasks.core.sampling import uniform_choice
from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.charts.shared.flow import resolve_flow_required_int_bounds
from trace_tasks.tasks.charts.shared.visual_defaults import (
    chart_font_asset_metadata,
    load_chart_scene_background_defaults,
    load_chart_scene_noise_defaults,
    sample_chart_font_family,
)
from trace_tasks.tasks.shared.bbox_projection import round_bbox
from trace_tasks.tasks.shared.config_defaults import (
    split_scene_generation_rendering_prompt_defaults,
)
from trace_tasks.tasks.shared.render_variation import apply_layout_jitter_to_margins, resolve_render_rgb

from .state import RGB, SCENE_ID, SCENE_NAMESPACE, RadialRenderParams


SCENE_DEFAULTS = get_scene_defaults("charts", SCENE_ID)
GEN_DEFAULTS, RENDER_DEFAULTS, PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    SCENE_DEFAULTS if isinstance(SCENE_DEFAULTS, Mapping) else {}
)
POST_IMAGE_BACKGROUND_DEFAULTS = load_chart_scene_background_defaults(scene_id=SCENE_ID)
POST_IMAGE_NOISE_DEFAULTS = load_chart_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.0)

TITLE_OPTIONS: tuple[str, ...] = (
    "Radial Transfer Map",
    "Circular Flow Summary",
    "Endpoint Flow Ring",
    "Transfer Chord Diagram",
    "Radial Sankey Routing",
)

RADIAL_COLOR_SCHEMES: tuple[dict[str, Any], ...] = (
    {
        "name": "lagoon_coral",
        "source_node_fill_rgb": (34, 99, 137),
        "target_node_fill_rgb": (184, 82, 77),
        "ring_line_rgb": (122, 151, 166),
        "value_label_fill_rgb": (255, 255, 252),
        "value_label_border_rgb": (87, 101, 112),
        "flow_palette_rgb": (
            (39, 125, 161),
            (235, 130, 75),
            (68, 159, 119),
            (181, 86, 110),
            (219, 181, 73),
            (86, 117, 186),
            (143, 108, 179),
        ),
    },
    {
        "name": "forest_plum",
        "source_node_fill_rgb": (71, 121, 65),
        "target_node_fill_rgb": (111, 75, 150),
        "ring_line_rgb": (136, 157, 129),
        "value_label_fill_rgb": (254, 255, 249),
        "value_label_border_rgb": (91, 98, 84),
        "flow_palette_rgb": (
            (75, 142, 84),
            (176, 117, 64),
            (128, 88, 164),
            (207, 154, 65),
            (64, 139, 151),
            (197, 92, 113),
            (99, 118, 62),
        ),
    },
    {
        "name": "ink_gold",
        "source_node_fill_rgb": (45, 74, 112),
        "target_node_fill_rgb": (163, 104, 37),
        "ring_line_rgb": (142, 138, 123),
        "value_label_fill_rgb": (255, 253, 243),
        "value_label_border_rgb": (89, 85, 74),
        "flow_palette_rgb": (
            (53, 91, 146),
            (207, 157, 60),
            (143, 83, 149),
            (70, 147, 133),
            (197, 91, 78),
            (98, 118, 177),
            (168, 122, 51),
        ),
    },
    {
        "name": "berry_teal",
        "source_node_fill_rgb": (44, 128, 133),
        "target_node_fill_rgb": (148, 69, 118),
        "ring_line_rgb": (127, 155, 158),
        "value_label_fill_rgb": (253, 252, 255),
        "value_label_border_rgb": (88, 91, 110),
        "flow_palette_rgb": (
            (42, 145, 151),
            (192, 83, 129),
            (95, 135, 205),
            (228, 151, 70),
            (96, 160, 94),
            (154, 104, 190),
            (204, 92, 84),
        ),
    },
    {
        "name": "copper_blue",
        "source_node_fill_rgb": (57, 88, 145),
        "target_node_fill_rgb": (176, 94, 58),
        "ring_line_rgb": (141, 150, 165),
        "value_label_fill_rgb": (255, 254, 250),
        "value_label_border_rgb": (82, 89, 103),
        "flow_palette_rgb": (
            (57, 101, 177),
            (205, 105, 64),
            (73, 150, 112),
            (194, 151, 54),
            (128, 95, 181),
            (68, 142, 174),
            (181, 82, 95),
        ),
    },
    {
        "name": "slate_citrus",
        "source_node_fill_rgb": (70, 83, 105),
        "target_node_fill_rgb": (105, 128, 48),
        "ring_line_rgb": (143, 150, 153),
        "value_label_fill_rgb": (255, 255, 248),
        "value_label_border_rgb": (86, 94, 96),
        "flow_palette_rgb": (
            (78, 96, 125),
            (151, 164, 55),
            (212, 128, 60),
            (72, 151, 164),
            (175, 82, 132),
            (113, 113, 188),
            (89, 145, 93),
        ),
    },
)
RADIAL_COLOR_SCHEME_BY_NAME = {str(scheme["name"]): dict(scheme) for scheme in RADIAL_COLOR_SCHEMES}


def render_style_seed(params: Mapping[str, Any]) -> int:
    try:
        return int(params.get("_render_style_seed", params.get("_sample_cursor", 0)) or 0)
    except Exception:
        return 0


def int_param(params: Mapping[str, Any], key: str, fallback: int) -> int:
    return int(params.get(str(key), RENDER_DEFAULTS.get(str(key), int(fallback))))


def gen_int_param(params: Mapping[str, Any], key: str, fallback: int) -> int:
    return int(params.get(str(key), GEN_DEFAULTS.get(str(key), int(fallback))))


def required_int_bounds(
    params: Mapping[str, Any],
    *,
    min_key: str,
    max_key: str,
    fallback_min: int,
    fallback_max: int,
    context: str,
) -> tuple[int, int]:
    return resolve_flow_required_int_bounds(
        params,
        GEN_DEFAULTS,
        min_key=str(min_key),
        max_key=str(max_key),
        fallback_min=int(fallback_min),
        fallback_max=int(fallback_max),
        context=str(context),
    )


def as_rgb(raw: Any, fallback: RGB) -> RGB:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != 3:
        return tuple(int(value) for value in fallback)
    return (int(raw[0]), int(raw[1]), int(raw[2]))


def rgb_param(params: Mapping[str, Any], key: str, fallback: RGB) -> RGB:
    return resolve_render_rgb(
        params,
        RENDER_DEFAULTS,
        str(key),
        fallback,
        instance_seed=render_style_seed(params),
        namespace=SCENE_NAMESPACE,
    )


def resolve_radial_color_scheme(params: Mapping[str, Any], *, instance_seed: int) -> dict[str, Any]:
    explicit = params.get("radial_color_scheme", RENDER_DEFAULTS.get("radial_color_scheme"))
    if explicit is not None:
        name = str(explicit)
        if name not in RADIAL_COLOR_SCHEME_BY_NAME:
            raise ValueError(f"unknown radial_color_scheme: {name}")
        selected = dict(RADIAL_COLOR_SCHEME_BY_NAME[name])
    else:
        raw_options = params.get("radial_color_scheme_options", RENDER_DEFAULTS.get("radial_color_scheme_options"))
        option_names = (
            [str(value) for value in raw_options if str(value) in RADIAL_COLOR_SCHEME_BY_NAME]
            if isinstance(raw_options, Sequence) and not isinstance(raw_options, (str, bytes))
            else []
        )
        if not option_names:
            option_names = [str(scheme["name"]) for scheme in RADIAL_COLOR_SCHEMES]
        scheme_name = uniform_choice(
            spawn_rng(
                int(instance_seed),
                f"{SCENE_NAMESPACE}.radial_color_scheme",
            ),
            tuple(option_names),
        )
        selected = dict(RADIAL_COLOR_SCHEME_BY_NAME[str(scheme_name)])

    palette = [as_rgb(color, (52, 111, 179)) for color in selected.get("flow_palette_rgb", ())]
    if len(palette) < 3:
        palette = [
            (52, 111, 179),
            (219, 124, 62),
            (68, 156, 118),
            (149, 111, 190),
            (215, 171, 63),
            (70, 144, 169),
        ]
    rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.flow_palette.{selected['name']}")
    rng.shuffle(palette)
    selected["flow_palette_rgb"] = tuple(palette)
    return selected


def clamp_bbox(bbox: Sequence[float], *, width: int, height: int) -> list[float]:
    x0, y0, x1, y1 = [float(value) for value in bbox]
    x0 = max(0.0, min(float(width), x0))
    y0 = max(0.0, min(float(height), y0))
    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))
    if x1 <= x0:
        x1 = min(float(width), x0 + 1.0)
    if y1 <= y0:
        y1 = min(float(height), y0 + 1.0)
    return round_bbox((x0, y0, x1, y1))


def resolve_render_params(params: Mapping[str, Any]) -> RadialRenderParams:
    """Resolve scene-level visual parameters while preserving canvas and bbox-safe jitter invariants."""

    outer = int_param(params, "outer_margin_px", 36)
    color_scheme = resolve_radial_color_scheme(params, instance_seed=render_style_seed(params))
    jitter_left, _jitter_right, jitter_top, _jitter_bottom, layout_jitter_meta = apply_layout_jitter_to_margins(
        left_px=int(outer),
        right_px=int(outer),
        top_px=int(outer),
        bottom_px=int(outer),
        params=params,
        defaults=RENDER_DEFAULTS,
        instance_seed=render_style_seed(params),
        namespace=f"{SCENE_NAMESPACE}.layout",
    )
    return RadialRenderParams(
        canvas_width=int_param(params, "canvas_width", 1328),
        canvas_height=int_param(params, "canvas_height", 960),
        outer_margin_px=int(outer),
        panel_padding_px=int_param(params, "panel_padding_px", 28),
        title_band_height_px=int_param(params, "title_band_height_px", 62),
        ring_radius_px=int_param(params, "radial_ring_radius_px", 360),
        chord_radius_inset_px=int_param(params, "radial_chord_radius_inset_px", 54),
        node_width_px=int_param(params, "radial_node_width_px", 78),
        node_height_px=int_param(params, "radial_node_height_px", 50),
        node_border_width_px=int_param(params, "node_border_width_px", 2),
        min_flow_width_px=int_param(params, "radial_min_flow_width_px", int_param(params, "min_flow_width_px", 6)),
        max_flow_width_px=int_param(params, "radial_max_flow_width_px", int_param(params, "max_flow_width_px", 18)),
        value_label_font_size_px=int_param(params, "value_label_font_size_px", 25),
        value_label_gap_px=max(8, int_param(params, "value_label_gap_px", 16)),
        node_label_font_size_px=int_param(params, "node_label_font_size_px", 30),
        title_font_size_px=int_param(params, "title_font_size_px", 31),
        panel_fill_rgb=rgb_param(params, "panel_fill_rgb", (252, 253, 251)),
        panel_border_rgb=rgb_param(params, "panel_border_rgb", (70, 80, 90)),
        plot_fill_rgb=rgb_param(params, "plot_fill_rgb", (255, 255, 255)),
        ring_line_rgb=rgb_param(params, "radial_ring_line_rgb", as_rgb(color_scheme.get("ring_line_rgb"), (170, 178, 188))),
        source_node_fill_rgb=rgb_param(params, "radial_source_node_fill_rgb", as_rgb(color_scheme.get("source_node_fill_rgb"), (42, 99, 150))),
        target_node_fill_rgb=rgb_param(params, "radial_target_node_fill_rgb", as_rgb(color_scheme.get("target_node_fill_rgb"), (130, 83, 148))),
        node_border_rgb=rgb_param(params, "node_border_rgb", (30, 38, 46)),
        node_text_rgb=rgb_param(params, "node_text_rgb", (255, 255, 255)),
        value_label_fill_rgb=rgb_param(params, "value_label_fill_rgb", as_rgb(color_scheme.get("value_label_fill_rgb"), (255, 255, 255))),
        value_label_border_rgb=rgb_param(params, "value_label_border_rgb", as_rgb(color_scheme.get("value_label_border_rgb"), (82, 88, 96))),
        value_label_text_rgb=rgb_param(params, "value_label_text_rgb", (28, 34, 42)),
        title_color_rgb=rgb_param(params, "title_color_rgb", (32, 38, 46)),
        color_scheme_name=str(color_scheme["name"]),
        flow_palette_rgb=tuple(as_rgb(color, (52, 111, 179)) for color in color_scheme["flow_palette_rgb"]),
        flow_alpha=max(40, min(220, int_param(params, "radial_flow_alpha", int_param(params, "flow_alpha", 138)))),
        layout_offset_x_px=int(jitter_left) - int(outer),
        layout_offset_y_px=int(jitter_top) - int(outer),
        layout_jitter_meta=dict(layout_jitter_meta),
    )


def prompt_bundle_id() -> str:
    return str(PROMPT_DEFAULTS.get("bundle_id", "charts_radial_sankey_v1"))


def font_assets_payload(*, chart_font_family: str) -> dict[str, Any]:
    return chart_font_asset_metadata(str(chart_font_family))


def sample_chart_font(instance_seed: int, params: Mapping[str, Any]) -> str:
    return str(
        sample_chart_font_family(
            instance_seed=int(instance_seed),
            namespace=f"{SCENE_NAMESPACE}.chart_font",
            params=params,
        )
    )
