"""Surface-fixture cell layout helpers."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .state import SEMANTIC_COLOR_RGB
from .rendering import layout_surface_element_grid


VALID_LAYOUT_STYLES: Tuple[str, ...] = (
    "uniform_grid",
    "variable_grid",
    "brick_grid",
    "jittered_grid",
    "loose_rows",
    "panel_scatter",
)

VALID_LAYOUT_FAMILIES: Tuple[str, ...] = (
    "strict_grid",
    "tiled_staggered",
    "loose_mounted_rows",
    "panel_scatter",
)

SURFACE_LAYOUT_STYLE_WEIGHTS_BY_FAMILY: Mapping[str, Mapping[str, float]] = {
    "strict_grid": {"uniform_grid": 0.80, "variable_grid": 0.20},
    "tiled_staggered": {"uniform_grid": 0.40, "variable_grid": 0.25, "brick_grid": 0.35},
    "loose_mounted_rows": {"jittered_grid": 0.35, "loose_rows": 0.55, "uniform_grid": 0.10},
    "panel_scatter": {"panel_scatter": 0.75, "jittered_grid": 0.25},
}

SURFACE_LAYOUT_FAMILY_WEIGHTS_BY_VARIANT: Mapping[str, Mapping[str, float]] = {
    "wall_tile_panel": {"strict_grid": 0.35, "tiled_staggered": 0.65},
    "perforated_panel": {"tiled_staggered": 1.0},
    "slot_board": {"loose_mounted_rows": 1.0},
    "compartment_tray": {"strict_grid": 1.0},
    "vent_panel": {"loose_mounted_rows": 1.0},
    "window_grid": {"strict_grid": 1.0},
    "door_bank": {"strict_grid": 1.0},
    "drawer_pull_panel": {"loose_mounted_rows": 1.0},
    "brick_wall": {"tiled_staggered": 1.0},
    "paver_floor": {"tiled_staggered": 1.0},
    "locker_bank": {"strict_grid": 1.0},
    "mailbox_bank": {"strict_grid": 1.0},
    "server_rack": {"strict_grid": 1.0},
    "control_panel": {"loose_mounted_rows": 0.45, "panel_scatter": 0.55},
    "solar_panel_array": {"strict_grid": 1.0},
    "screw_plate": {"panel_scatter": 1.0},
    "hex_nut_plate": {"panel_scatter": 1.0},
    "washer_plate": {"panel_scatter": 1.0},
    "socket_bank": {"loose_mounted_rows": 1.0},
    "hook_board": {"loose_mounted_rows": 1.0},
    "indicator_light_panel": {"loose_mounted_rows": 0.55, "panel_scatter": 0.45},
    "bracket_panel": {"loose_mounted_rows": 1.0},
    "u_bolt_plate": {"loose_mounted_rows": 1.0},
    "pipe_rack": {"strict_grid": 0.30, "loose_mounted_rows": 0.70},
}


def _normalized_weights(weights: Mapping[str, float], *, fallback: str = "uniform_grid") -> Dict[str, float]:
    total = float(sum(max(0.0, float(value)) for value in weights.values()))
    if total <= 0.0:
        return {str(fallback): 1.0}
    return {
        str(style): float(value) / total
        for style, value in weights.items()
        if float(value) > 0.0
    }


def _choose_weighted(weights: Mapping[str, float], *, rng: Any) -> str:
    threshold = float(rng.random())
    cumulative = 0.0
    selected = next(iter(weights))
    for key, probability in weights.items():
        cumulative += float(probability)
        selected = str(key)
        if threshold <= cumulative:
            break
    return selected


def _aggregate_style_probabilities(family_probabilities: Mapping[str, float]) -> Dict[str, float]:
    style_probabilities: Dict[str, float] = {}
    for family, family_probability in family_probabilities.items():
        style_weights = _normalized_weights(SURFACE_LAYOUT_STYLE_WEIGHTS_BY_FAMILY[str(family)])
        for style, style_probability in style_weights.items():
            style_probabilities[str(style)] = style_probabilities.get(str(style), 0.0) + (
                float(family_probability) * float(style_probability)
            )
    return {
        style: probability
        for style, probability in style_probabilities.items()
        if probability > 0.0
    }


def resolve_repeated_layout_style(
    *,
    scene_variant: str,
    rng: Any,
    params: Mapping[str, Any],
) -> Tuple[str, str, Dict[str, float], Dict[str, float]]:
    """Select a controlled placement family and style for repeated-element counting."""

    valid_families = set(VALID_LAYOUT_FAMILIES)
    family_probabilities = _normalized_weights(
        SURFACE_LAYOUT_FAMILY_WEIGHTS_BY_VARIANT.get(
            str(scene_variant),
            {"strict_grid": 1.0},
        ),
        fallback="strict_grid",
    )
    if not set(family_probabilities).issubset(valid_families):
        invalid = sorted(set(family_probabilities) - valid_families)
        raise ValueError(f"unsupported layout family for {scene_variant}: {invalid}")

    explicit_family = params.get("layout_family")
    if explicit_family is not None:
        family = str(explicit_family)
        if family not in family_probabilities:
            raise ValueError(f"layout_family {family} is not allowed for {scene_variant}")
        family_probabilities = {family: 1.0}
    else:
        family = _choose_weighted(family_probabilities, rng=rng)

    style_weights = _normalized_weights(SURFACE_LAYOUT_STYLE_WEIGHTS_BY_FAMILY[str(family)])
    explicit_style = params.get("layout_style")
    if explicit_style is not None:
        style = str(explicit_style)
        if style not in set(VALID_LAYOUT_STYLES):
            raise ValueError(f"unsupported surface fixture layout_style: {style}")
        if style not in style_weights:
            raise ValueError(f"layout_style {style} is not allowed for layout_family {family}")
        style_probabilities = {style: 1.0}
    else:
        style = _choose_weighted(style_weights, rng=rng)
        style_probabilities = _aggregate_style_probabilities(family_probabilities)
    return str(family), str(style), dict(family_probabilities), dict(style_probabilities)


def _clamp_interval(center: float, span: float, lower: float, upper: float) -> Tuple[float, float]:
    half = float(span) * 0.5
    center = min(float(upper) - half, max(float(lower) + half, float(center)))
    return float(center - half), float(center + half)


def _scatter_bounds(
    *,
    rows: int,
    cols: int,
    present_indices: Sequence[int],
    rng: Any,
    u_pad: float,
    v_pad: float,
) -> Dict[int, Tuple[float, float, float, float]]:
    indices = [int(index) for index in present_indices]
    count = max(1, len(indices))
    base_w = min(0.145, max(0.075, 0.62 / max(3.0, math.sqrt(float(count)) * 1.45)))
    base_h = min(0.145, max(0.075, 0.58 / max(3.0, math.sqrt(float(count)) * 1.40)))
    min_sep = min(0.13, max(0.068, 0.43 / math.sqrt(float(count))))
    centers: List[Tuple[float, float]] = []
    bounds: Dict[int, Tuple[float, float, float, float]] = {}

    for rank, index in enumerate(indices):
        width = base_w * float(rng.uniform(0.84, 1.16))
        height = base_h * float(rng.uniform(0.84, 1.16))
        center_u = 0.5
        center_v = 0.5
        for _attempt in range(180):
            center_u = float(rng.uniform(float(u_pad) + width * 0.5, 1.0 - float(u_pad) - width * 0.5))
            center_v = float(rng.uniform(float(v_pad) + height * 0.5, 1.0 - float(v_pad) - height * 0.5))
            if all((center_u - u) ** 2 + ((center_v - v) * 1.12) ** 2 >= min_sep**2 for u, v in centers):
                break
        else:
            row = int(rank // max(1, int(cols)))
            col = int(rank % max(1, int(cols)))
            cell_w = (1.0 - 2.0 * float(u_pad)) / float(max(1, int(cols)))
            cell_h = (1.0 - 2.0 * float(v_pad)) / float(max(1, int(rows)))
            center_u = float(u_pad) + (float(col) + 0.5) * cell_w + float(rng.uniform(-0.16, 0.16)) * cell_w
            center_v = float(v_pad) + (float(row) + 0.5) * cell_h + float(rng.uniform(-0.14, 0.14)) * cell_h
        centers.append((float(center_u), float(center_v)))
        u0, u1 = _clamp_interval(center_u, width, u_pad, 1.0 - u_pad)
        v0, v1 = _clamp_interval(center_v, height, v_pad, 1.0 - v_pad)
        bounds[int(index)] = (u0, u1, v0, v1)
    return bounds


def layout_cells(
    *,
    scene_variant: str,
    element_type: str,
    rows: int,
    cols: int,
    present_indices: Sequence[int],
    target_indices: Sequence[int],
    rng: Any,
    layout_style: str,
    color_by_index: Mapping[int, str] | None = None,
    reference_index: int | None = None,
    include_absent: bool = False,
    semantic_color: bool = False,
) -> List[Dict[str, Any]]:
    """Create projected-cell records before pixel rendering."""

    color_by_index = color_by_index or {}
    present_set = {int(index) for index in present_indices}
    target_set = {int(index) for index in target_indices}
    u_pad = 0.065
    v_pad = 0.075
    gap = 0.016
    if str(layout_style) in {"variable_grid", "loose_rows"}:
        col_weights = [float(rng.uniform(0.78, 1.22)) for _ in range(int(cols))]
        row_weights = [float(rng.uniform(0.82, 1.18)) for _ in range(int(rows))]
    else:
        col_weights = [1.0 for _ in range(int(cols))]
        row_weights = [1.0 for _ in range(int(rows))]
    col_total = sum(col_weights) or 1.0
    row_total = sum(row_weights) or 1.0
    col_edges = [u_pad]
    for weight in col_weights:
        col_edges.append(float(col_edges[-1]) + (1.0 - 2.0 * u_pad) * float(weight) / float(col_total))
    row_edges = [v_pad]
    for weight in row_weights:
        row_edges.append(float(row_edges[-1]) + (1.0 - 2.0 * v_pad) * float(weight) / float(row_total))

    scatter_bounds = (
        _scatter_bounds(rows=int(rows), cols=int(cols), present_indices=present_indices, rng=rng, u_pad=u_pad, v_pad=v_pad)
        if str(layout_style) == "panel_scatter"
        else {}
    )

    cells: List[Dict[str, Any]] = []
    for flat_index in range(int(rows) * int(cols)):
        row = int(flat_index // int(cols))
        col = int(flat_index % int(cols))
        u0 = float(col_edges[col])
        u1 = float(col_edges[col + 1])
        v0 = float(row_edges[row])
        v1 = float(row_edges[row + 1])
        if str(layout_style) == "brick_grid" and row % 2 == 1:
            shift = ((u1 - u0) * 0.22)
            u0 = max(u_pad, u0 + shift)
            u1 = min(1.0 - u_pad, u1 + shift)
        elif str(layout_style) == "jittered_grid":
            width = float(u1 - u0)
            height = float(v1 - v0)
            scale_u = float(rng.uniform(0.72, 0.92))
            scale_v = float(rng.uniform(0.72, 0.92))
            center_u = (u0 + u1) * 0.5 + float(rng.uniform(-0.13, 0.13)) * width
            center_v = (v0 + v1) * 0.5 + float(rng.uniform(-0.10, 0.10)) * height
            u0, u1 = _clamp_interval(center_u, width * scale_u, u0, u1)
            v0, v1 = _clamp_interval(center_v, height * scale_v, v0, v1)
        elif str(layout_style) == "loose_rows":
            width = float(u1 - u0)
            height = float(v1 - v0)
            scale_u = float(rng.uniform(0.62, 0.88))
            scale_v = float(rng.uniform(0.62, 0.88))
            center_u = (u0 + u1) * 0.5 + float(rng.uniform(-0.20, 0.20)) * width
            center_v = (v0 + v1) * 0.5 + float(rng.uniform(-0.08, 0.08)) * height
            u0, u1 = _clamp_interval(center_u, width * scale_u, u0, u1)
            v0, v1 = _clamp_interval(center_v, height * scale_v, v0, v1)
        elif str(layout_style) == "panel_scatter" and int(flat_index) in scatter_bounds:
            u0, u1, v0, v1 = scatter_bounds[int(flat_index)]
        is_present = int(flat_index) in present_set
        if not is_present and not bool(include_absent):
            continue
        color_name = str(color_by_index.get(int(flat_index), ""))
        count_role = "target" if int(flat_index) in target_set else "distractor"
        if reference_index is not None and int(flat_index) == int(reference_index):
            count_role = "reference"
        element_id = f"{element_type}_{flat_index:02d}" if is_present else f"missing_{element_type}_{flat_index:02d}"
        cells.append(
            {
                "element_id": str(element_id),
                "cell_id": f"cell_{flat_index:02d}",
                "flat_index": int(flat_index),
                "element_type": str(element_type),
                "row": int(row),
                "column": int(col),
                "u0": float(u0 + gap),
                "u1": float(u1 - gap),
                "v0": float(v0 + gap),
                "v1": float(v1 - gap),
                "present": bool(is_present),
                "color_name": str(color_name),
                "fill_rgb": list(SEMANTIC_COLOR_RGB[color_name]) if color_name else None,
                "semantic_color": bool(semantic_color and color_name),
                "count_role": str(count_role),
                "layout_style": str(layout_style),
            }
        )
    return cells


def target_ids_from_indices(cells: Sequence[Mapping[str, Any]], target_indices: Sequence[int]) -> List[str]:
    """Return element ids for target flat indices."""

    target_set = {int(index) for index in target_indices}
    return [str(cell["element_id"]) for cell in cells if int(cell["flat_index"]) in target_set]


def grid_for_total(total_slots: int, *, min_cols: int = 3) -> Tuple[int, int]:
    """Return a stable rows/columns grid for a target slot count."""

    rows, cols = layout_surface_element_grid(int(total_slots))
    cols = max(int(min_cols), int(cols))
    rows = int((int(total_slots) + int(cols) - 1) // int(cols))
    return int(rows), int(cols)


__all__ = [
    "SURFACE_LAYOUT_FAMILY_WEIGHTS_BY_VARIANT",
    "SURFACE_LAYOUT_STYLE_WEIGHTS_BY_FAMILY",
    "VALID_LAYOUT_FAMILIES",
    "VALID_LAYOUT_STYLES",
    "grid_for_total",
    "layout_cells",
    "resolve_repeated_layout_style",
    "target_ids_from_indices",
]
