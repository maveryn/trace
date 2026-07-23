"""Scene-neutral sampling helpers for named-strip icon rows."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from ....shared.config_defaults import group_default
from ...shared.icon_style import sample_icon_palette
from ...shared.icon_task_rendering import sample_icon_instance_noise
from ...shared.procedural_named_icon_field_scene import (
    resolve_named_icon_fill_style_probabilities,
    rotation_for_named_shape,
)
from ...shared.procedural_named_icons import (
    PROCEDURAL_NAMED_ICON_SHAPES,
    sample_procedural_named_icon_fill_style,
    validate_procedural_named_icon_fill_style_support,
)

from .defaults import DEFAULT_RENDER
from .state import NamedStripIconPlan


def named_strip_shape_support(params: Mapping[str, Any], gen_defaults: Mapping[str, Any]) -> Tuple[str, ...]:
    """Resolve the procedural named-icon support set for one strip."""

    raw = params.get("shape_id_support", group_default(gen_defaults, "shape_id_support", PROCEDURAL_NAMED_ICON_SHAPES))
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("shape_id_support must be a sequence")
    values = tuple(dict.fromkeys(str(value).strip() for value in raw if str(value).strip()))
    unsupported = sorted(set(values) - set(PROCEDURAL_NAMED_ICON_SHAPES))
    if unsupported:
        raise ValueError(f"unsupported procedural named icon shapes: {unsupported}")
    if len(values) < 8:
        raise ValueError("named-strip rows need at least eight supported icon shapes")
    return values


def named_strip_fill_style_support(params: Mapping[str, Any], gen_defaults: Mapping[str, Any]) -> Tuple[str, ...]:
    """Resolve non-semantic named-icon fill styles for one strip."""

    raw = params.get(
        "named_icon_fill_style_support",
        group_default(gen_defaults, "named_icon_fill_style_support", DEFAULT_RENDER.named_icon_fill_style_support),
    )
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raw = DEFAULT_RENDER.named_icon_fill_style_support
    return validate_procedural_named_icon_fill_style_support(tuple(str(value) for value in raw))


def named_strip_fill_style_probabilities(
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    support: Sequence[str],
) -> Dict[str, float]:
    """Resolve non-semantic fill-style probabilities for one strip."""

    return resolve_named_icon_fill_style_probabilities(params, gen_defaults, tuple(str(value) for value in support))


def target_runs(shape_ids: Sequence[str], *, target_shape_id: str) -> Tuple[Tuple[int, int], ...]:
    """Return contiguous runs of `target_shape_id` in reading order."""

    runs: List[Tuple[int, int]] = []
    start = None
    for index, shape_id in enumerate(shape_ids):
        if str(shape_id) == str(target_shape_id):
            if start is None:
                start = int(index)
        elif start is not None:
            runs.append((int(start), int(index) - 1))
            start = None
    if start is not None:
        runs.append((int(start), len(shape_ids) - 1))
    return tuple(runs)


def target_run_lengths(runs: Sequence[Tuple[int, int]]) -> Tuple[int, ...]:
    """Return inclusive run lengths from `(start, end)` pairs."""

    return tuple(int(end) - int(start) + 1 for start, end in runs)


def build_named_strip_icon_plans(
    *,
    shape_ids: Sequence[str],
    fill_style_support: Sequence[str],
    fill_style_probabilities: Mapping[str, float],
    instance_seed: int,
    render_params: Mapping[str, Any],
    rng,
) -> Tuple[Tuple[NamedStripIconPlan, ...], Tuple[Tuple[int, int, int], ...]]:
    """Sample per-cell procedural icon render plans for one named strip."""

    palette_size = int(rng.randint(int(render_params["palette_size_min"]), int(render_params["palette_size_max"])))
    palette = sample_icon_palette(
        rng,
        palette_size=int(palette_size),
        channel_min=int(render_params["color_channel_min"]),
        channel_max=int(render_params["color_channel_max"]),
        anchor_colors=(
            tuple(int(value) for value in render_params["background_color_rgb"]),
            tuple(int(value) for value in render_params["panel_fill_rgb"]),
            tuple(int(value) for value in render_params["panel_border_rgb"]),
            tuple(int(value) for value in render_params["header_text_rgb"]),
        ),
        min_color_distance=float(render_params["min_color_distance"]),
        distance_space=str(render_params["color_distance_space"]),
    )
    min_size = max(12, int(render_params["scene_icon_size_min_px"]))
    max_size = max(min_size, int(render_params["scene_icon_size_max_px"]))
    plans: List[NamedStripIconPlan] = []
    for cell_index, shape_id in enumerate(shape_ids):
        noise_edits, noise_seed = sample_icon_instance_noise(
            instance_seed=int(instance_seed),
            namespace=f"icons.named_strip.cell_{int(cell_index)}",
            render_params=render_params,
        )
        plans.append(
            NamedStripIconPlan(
                cell_index=int(cell_index),
                shape_id=str(shape_id),
                tint_rgb=tuple(int(value) for value in rng.choice(palette)),
                fill_style=sample_procedural_named_icon_fill_style(
                    rng,
                    support=tuple(str(value) for value in fill_style_support),
                    probabilities={str(key): float(value) for key, value in fill_style_probabilities.items()},
                ),
                nominal_size_px=int(rng.randint(int(min_size), int(max_size))),
                rotation_degrees=rotation_for_named_shape(rng, str(shape_id)),
                noise_edits=tuple(noise_edits),
                noise_seed=int(noise_seed),
            )
        )
    return tuple(plans), tuple(tuple(int(channel) for channel in color) for color in palette)


__all__ = [
    "build_named_strip_icon_plans",
    "named_strip_fill_style_probabilities",
    "named_strip_fill_style_support",
    "named_strip_shape_support",
    "target_run_lengths",
    "target_runs",
]
