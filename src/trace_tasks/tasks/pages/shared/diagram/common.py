"""Shared page-diagram helpers reused across multiple page task families."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from PIL import ImageDraw

from .....core.seed import spawn_rng
from ....shared.drawing import draw_centered_text
from ....shared.name_assets import load_short_name_manifest
from ....shared.render_variation import (
    apply_resolved_layout_jitter_to_margins,
    resolve_render_int,
    resolve_render_rgb,
)
from ....shared.text_rendering import fit_font_to_box
from ....shared.variant_sampling import apply_balanced_variant_sampling, resolve_variant


def resolve_diagrams_axis_variant(
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    supported_variants: Sequence[str],
    task_id: str,
    explicit_key: str,
    weights_key: str,
    balance_flag_key: str,
    axis_namespace: str,
) -> Tuple[str, Dict[str, float]]:
    """Resolve one semantic or visual page-diagram axis with deterministic balancing."""

    rng = spawn_rng(int(instance_seed), f"{task_id}.{axis_namespace}")
    selected_variant, probabilities = resolve_variant(
        rng,
        params=params,
        gen_defaults=gen_defaults,
        supported_variants=[str(item) for item in supported_variants],
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
    )
    balanced = apply_balanced_variant_sampling(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        selected_variant=str(selected_variant),
        variant_probabilities=probabilities,
        supported_variants=[str(item) for item in supported_variants],
        balance_flag_key=str(balance_flag_key),
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
        sampling_namespace=f"{task_id}:{axis_namespace}",
    )
    return str(balanced), {str(key): float(value) for key, value in probabilities.items()}


def projected_diagram_bbox_annotation(
    bbox_map: Mapping[str, Sequence[float]],
    item_ids: Sequence[str],
) -> Dict[str, Any]:
    """Project ordered diagram ids into prompt-facing `bbox_set` annotation."""

    return {
        "bbox_set": [
            list(bbox_map[str(item_id)])
            for item_id in [str(item) for item in item_ids]
            if str(item_id) in bbox_map
        ]
    }


def projected_diagram_bbox_sequence_annotation(
    bbox_map: Mapping[str, Sequence[float]],
    item_ids: Sequence[str],
) -> Dict[str, Any]:
    """Project ordered diagram ids into prompt-facing `bbox_sequence` annotation."""

    return {
        "type": "bbox_sequence",
        "bbox_sequence": [
            list(bbox_map[str(item_id)])
            for item_id in [str(item) for item in item_ids]
            if str(item_id) in bbox_map
        ],
    }


def sample_diagram_short_names(*, count: int, rng) -> list[str]:
    """Sample unique short visible names for diagram labels."""

    names = load_short_name_manifest()
    if int(count) > len(names):
        raise ValueError("requested more diagram names than the shared short-name manifest contains")
    return [str(label) for label in rng.sample(list(names), int(count))]


def resolve_diagrams_int_param(
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    key: str,
    fallback: int,
    *,
    instance_seed: int | None = None,
    namespace: str = "pages.diagram",
) -> int:
    """Resolve one integer-valued diagrams config parameter."""

    return resolve_render_int(
        params,
        defaults,
        key,
        fallback,
        instance_seed=instance_seed,
        namespace=str(namespace),
    )


def resolve_diagrams_rgb_triple(
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    key: str,
    fallback: Sequence[int],
    *,
    instance_seed: int | None = None,
    namespace: str = "pages.diagram",
) -> tuple[int, int, int]:
    """Resolve one RGB triple from scene defaults and task params."""

    return resolve_render_rgb(
        params,
        defaults,
        key,
        fallback,
        instance_seed=instance_seed,
        namespace=str(namespace),
    )


def round_diagram_bbox(bbox: Sequence[float]) -> list[float]:
    """Round one bbox into trace-safe float precision."""

    return [round(float(value), 3) for value in bbox]


def resolve_diagram_panel_geometry(
    *,
    canvas_width: int,
    canvas_height: int,
    outer_margin_px: int,
    title_band_height_px: int,
    panel_padding_px: int,
) -> Tuple[Tuple[float, float, float, float], Tuple[float, float, float, float], Tuple[float, float, float, float]]:
    """Resolve the outer panel, title band, and inner content geometry for one diagram scene."""

    margin = float(outer_margin_px)
    panel = (
        margin,
        margin,
        float(int(canvas_width) - int(outer_margin_px)),
        float(int(canvas_height) - int(outer_margin_px)),
    )
    title_band = (
        float(panel[0]),
        float(panel[1]),
        float(panel[2]),
        float(panel[1] + int(title_band_height_px)),
    )
    content = (
        float(panel[0] + int(panel_padding_px)),
        float(title_band[3] + int(panel_padding_px)),
        float(panel[2] - int(panel_padding_px)),
        float(panel[3] - int(panel_padding_px)),
    )
    return panel, title_band, content


def resolve_jittered_diagram_panel_geometry(
    *,
    canvas_width: int,
    canvas_height: int,
    outer_margin_px: int,
    title_band_height_px: int,
    panel_padding_px: int,
    layout_jitter_meta: Mapping[str, Any] | None,
) -> Tuple[
    Tuple[float, float, float, float],
    Tuple[float, float, float, float],
    Tuple[float, float, float, float],
    Dict[str, Any],
]:
    """Resolve panel geometry after applying whole-panel layout jitter."""

    margin = float(outer_margin_px)
    jitter_left, _jitter_right, jitter_top, _jitter_bottom, resolved_jitter = apply_resolved_layout_jitter_to_margins(
        left_px=float(margin),
        right_px=float(margin),
        top_px=float(margin),
        bottom_px=float(margin),
        jitter=layout_jitter_meta,
    )
    panel = (
        float(jitter_left),
        float(jitter_top),
        float(int(canvas_width) - int(outer_margin_px) + int(resolved_jitter.get("dx_px", 0))),
        float(int(canvas_height) - int(outer_margin_px) + int(resolved_jitter.get("dy_px", 0))),
    )
    title_band = (
        float(panel[0]),
        float(panel[1]),
        float(panel[2]),
        float(panel[1] + int(title_band_height_px)),
    )
    content = (
        float(panel[0] + int(panel_padding_px)),
        float(title_band[3] + int(panel_padding_px)),
        float(panel[2] - int(panel_padding_px)),
        float(panel[3] - int(panel_padding_px)),
    )
    return panel, title_band, content, dict(resolved_jitter)


def draw_diagram_text_in_box(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: Sequence[float],
    text: str,
    font_size_px: int,
    bold: bool,
    fill: Sequence[int],
    stroke_fill: Sequence[int],
    padding_px: int,
) -> list[float]:
    """Draw one centered fitted string inside a box and return the rendered text bbox."""

    left, top, right, bottom = [float(value) for value in bbox]
    font = fit_font_to_box(
        draw,
        text=str(text),
        max_width=float(right - left - (2.0 * float(padding_px))),
        max_height=float(bottom - top - (2.0 * float(padding_px))),
        bold=bool(bold),
        min_size_px=max(10, int(float(font_size_px) * 0.58)),
        max_size_px=int(font_size_px),
        fill_ratio=0.98,
    )
    return draw_centered_text(
        draw,
        text=str(text),
        center=(0.5 * float(left + right), 0.5 * float(top + bottom)),
        font=font,
        fill=tuple(int(value) for value in fill),
        stroke_fill=tuple(int(value) for value in stroke_fill),
        stroke_width=1,
    )


__all__ = [
    "draw_diagram_text_in_box",
    "projected_diagram_bbox_annotation",
    "projected_diagram_bbox_sequence_annotation",
    "resolve_jittered_diagram_panel_geometry",
    "resolve_diagram_panel_geometry",
    "resolve_diagrams_int_param",
    "resolve_diagrams_rgb_triple",
    "resolve_diagrams_axis_variant",
    "round_diagram_bbox",
    "sample_diagram_short_names",
]
