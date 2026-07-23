"""Sampling and layout helpers for concept-map scene packages."""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Dict, Mapping, Sequence

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.pages.shared.diagram.common import (
    resolve_jittered_diagram_panel_geometry,
    round_diagram_bbox,
)
from trace_tasks.tasks.shared.config_defaults import group_default, resolve_required_int_bounds
from trace_tasks.tasks.shared.deterministic_sampling import resolve_selection_index
from trace_tasks.tasks.shared.render_variation import resolve_layout_jitter, resolve_render_int
from trace_tasks.tasks.shared.variant_sampling import apply_balanced_variant_sampling, resolve_variant

from .defaults import (
    CHILD_NODE_SHAPES,
    CHILDREN_TOTAL_KIND,
    CONTEXTS,
    CONTEXT_VARIANTS,
    GENERATION_DEFAULTS,
    LAYOUT_VARIANTS,
    MARKED_TOTAL_KIND,
    MARKERS,
    NAMESPACE_ROOT,
    NODE_SHAPE_PROFILES,
    RANKED_CHILD_KIND,
    RENDERING_DEFAULTS,
    SCENE,
    STYLE_VARIANTS,
)
from .state import ConceptMapCase


def resolve_named_axis(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    supported_values: Sequence[str],
    explicit_key: str,
    weights_key: str,
    balance_flag_key: str,
    namespace: str,
) -> tuple[str, Dict[str, float]]:
    """Resolve one balanced concept-map scene axis."""

    rng = spawn_rng(int(instance_seed), f"{NAMESPACE_ROOT}.{namespace}")
    selected, probabilities = resolve_variant(
        rng,
        params=params,
        gen_defaults=GENERATION_DEFAULTS,
        supported_variants=[str(item) for item in supported_values],
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
    )
    balanced = apply_balanced_variant_sampling(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=GENERATION_DEFAULTS,
        selected_variant=str(selected),
        variant_probabilities=probabilities,
        supported_variants=[str(item) for item in supported_values],
        balance_flag_key=str(balance_flag_key),
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
        sampling_namespace=f"{NAMESPACE_ROOT}.{namespace}",
    )
    return str(balanced), {str(key): float(value) for key, value in sorted(probabilities.items())}


def _bbox_center(bbox: Sequence[float]) -> tuple[float, float]:
    return ((float(bbox[0]) + float(bbox[2])) * 0.5, (float(bbox[1]) + float(bbox[3])) * 0.5)


def _node_bbox(center: tuple[float, float], width: float, height: float) -> list[float]:
    cx, cy = float(center[0]), float(center[1])
    return round_diagram_bbox((cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0))


def _resolve_node_shape_profile(instance_seed: int) -> tuple[str, str]:
    profile = NODE_SHAPE_PROFILES[abs(int(instance_seed)) % len(NODE_SHAPE_PROFILES)]
    central_shape = {
        "mixed_hub_circle": "circle",
        "oval_branch_mix": "ellipse",
        "mixed_cards_ovals": "pill",
    }[profile]
    return str(profile), str(central_shape)


def _assign_branch_child_shapes(
    branches: list[Dict[str, Any]],
    *,
    shape_profile: str,
    instance_seed: int,
) -> None:
    branch_shape_cycles = {
        "mixed_hub_circle": ("ellipse", "rounded_rect", "pill", "ellipse"),
        "oval_branch_mix": ("ellipse", "pill", "circle", "rounded_rect"),
        "mixed_cards_ovals": ("rounded_rect", "ellipse", "pill", "circle"),
    }
    branch_cycle = branch_shape_cycles[str(shape_profile)]
    child_offset = abs(int(instance_seed // 17)) % len(CHILD_NODE_SHAPES)
    for branch_index, branch in enumerate(branches):
        branch_shape = branch_cycle[(branch_index + abs(int(instance_seed))) % len(branch_cycle)]
        if str(branch_shape) == "circle" and len(str(branch["label"])) > 11:
            branch_shape = "ellipse"
        branch["shape"] = str(branch_shape)
        for child_index, child in enumerate(branch["children"]):
            child["shape"] = CHILD_NODE_SHAPES[(branch_index + child_index + child_offset) % len(CHILD_NODE_SHAPES)]


def _branch_dimensions(branch: Mapping[str, Any], width: float, height: float) -> tuple[float, float]:
    if str(branch.get("shape", "rounded_rect")) == "circle":
        diameter = float(group_default(RENDERING_DEFAULTS, "branch_circle_diameter_px", 86))
        return diameter, diameter
    return float(width), float(height)


def child_sort_key(child: Mapping[str, Any]) -> tuple[float, float, str]:
    """Sort children by visible reading order."""

    bbox = child["bbox"]
    return (float(bbox[1]), float(bbox[0]), str(child["label"]))


def ordinal_label(value: int) -> str:
    """Return a compact ordinal label."""

    value = int(value)
    suffix = "th"
    if value % 100 not in {11, 12, 13}:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")
    return f"{value}{suffix}"


def _select_branch_by_answer_index(
    *,
    branches: Sequence[Dict[str, Any]],
    answer_index: int,
    min_count: int,
    max_count: int,
) -> tuple[int, int]:
    target = int(min_count) + (abs(int(answer_index)) % (int(max_count) - int(min_count) + 1))
    eligible = [idx for idx, branch in enumerate(branches) if len(branch["children"]) == int(target)]
    if not eligible:
        closest_distance = min(abs(len(branch["children"]) - int(target)) for branch in branches)
        eligible = [
            idx for idx, branch in enumerate(branches)
            if abs(len(branch["children"]) - int(target)) == int(closest_distance)
        ]
    return int(eligible[abs(int(answer_index)) % len(eligible)]), int(target)


def build_concept_map_case(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    case_kind: str,
    case_defaults: Mapping[str, Any] | None = None,
) -> ConceptMapCase:
    """Resolve one complete concept-map scene and selected target."""

    context_id, context_probabilities = resolve_named_axis(
        params=params,
        instance_seed=int(instance_seed),
        supported_values=CONTEXT_VARIANTS,
        explicit_key="context_id",
        weights_key="context_weights",
        balance_flag_key="balanced_context_sampling",
        namespace="context",
    )
    layout_variant, layout_probabilities = resolve_named_axis(
        params=params,
        instance_seed=int(instance_seed),
        supported_values=LAYOUT_VARIANTS,
        explicit_key="layout_variant",
        weights_key="layout_weights",
        balance_flag_key="balanced_layout_sampling",
        namespace="layout",
    )
    style_variant, style_probabilities = resolve_named_axis(
        params=params,
        instance_seed=int(instance_seed),
        supported_values=STYLE_VARIANTS,
        explicit_key="style_variant",
        weights_key="style_weights",
        balance_flag_key="balanced_style_sampling",
        namespace="style",
    )
    rng = spawn_rng(int(instance_seed), f"{NAMESPACE_ROOT}.scene")
    scene, selection = _build_scene_and_selection(
        case_kind=str(case_kind),
        context_id=str(context_id),
        layout_variant=str(layout_variant),
        style_variant=str(style_variant),
        rng=rng,
        params=params,
        case_defaults=case_defaults or {},
        instance_seed=int(instance_seed),
    )
    return ConceptMapCase(
        case_kind=str(case_kind),
        scene=scene,
        selection=selection,
        context_probabilities=dict(context_probabilities),
        layout_probabilities=dict(layout_probabilities),
        style_probabilities=dict(style_probabilities),
    )


def _bounds(
    params: Mapping[str, Any],
    case_defaults: Mapping[str, Any],
    *,
    min_key: str,
    max_key: str,
    fallback_min: int,
    fallback_max: int,
    context: str,
) -> tuple[int, int]:
    merged_defaults = dict(GENERATION_DEFAULTS)
    merged_defaults.update(dict(case_defaults))
    return resolve_required_int_bounds(
        params,
        merged_defaults,
        min_key=str(min_key),
        max_key=str(max_key),
        fallback_min=int(fallback_min),
        fallback_max=int(fallback_max),
        context=str(context),
    )


def _build_scene_and_selection(
    *,
    case_kind: str,
    context_id: str,
    layout_variant: str,
    style_variant: str,
    rng,
    params: Mapping[str, Any],
    case_defaults: Mapping[str, Any],
    instance_seed: int,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Build one full diagram before target binding; sampled axes remain scene-level."""

    branch_min, branch_max = _bounds(
        params,
        case_defaults,
        min_key="branch_count_min",
        max_key="branch_count_max",
        fallback_min=5,
        fallback_max=7,
        context="concept-map branch count",
    )
    child_min, child_max = _bounds(
        params,
        case_defaults,
        min_key="child_count_min",
        max_key="child_count_max",
        fallback_min=3,
        fallback_max=8,
        context="concept-map child count",
    )
    marker_min, marker_max = _bounds(
        params,
        case_defaults,
        min_key="marked_count_min",
        max_key="marked_count_max",
        fallback_min=1,
        fallback_max=5,
        context="concept-map marked count",
    )
    if str(layout_variant) == "radial_mind_map":
        branch_max = min(int(branch_max), 5)
        branch_min = min(int(branch_min), int(branch_max))
        target_child_max = min(int(child_max), 6)
        target_marker_max = min(int(marker_max), 4)
        distractor_child_min = max(2, min(int(child_min), 3))
        distractor_child_max = min(int(child_max), 3)
    else:
        target_child_max = int(child_max)
        target_marker_max = int(marker_max)
        distractor_child_min = max(2, min(int(child_min), 4))
        distractor_child_max = min(int(child_max), 4)
    target_child_min = max(1, min(int(child_min), int(target_child_max)))
    target_marker_min = min(int(marker_min), int(target_marker_max))
    distractor_child_max = max(int(distractor_child_min), int(distractor_child_max))
    branch_count = int(rng.randint(int(branch_min), int(branch_max)))
    context = CONTEXTS[str(context_id)]
    branch_items = list(context["branches"].items())
    branch_offset = int(rng.randrange(len(branch_items)))
    selected_branch_items = [branch_items[(branch_offset + idx) % len(branch_items)] for idx in range(branch_count)]
    sampling_index = abs(
        int(
            params.get(
                "_sample_cursor",
                resolve_selection_index(
                    params=params,
                    instance_seed=int(instance_seed),
                    namespace=f"{NAMESPACE_ROOT}.answer",
                ),
            )
        )
    )
    forced_branch_index: int | None = None
    forced_count: int | None = None
    forced_marker = MARKERS[abs(int(sampling_index)) % len(MARKERS)]

    if str(case_kind) == CHILDREN_TOTAL_KIND:
        forced_count = int(target_child_min) + (
            abs(int(sampling_index)) % (int(target_child_max) - int(target_child_min) + 1)
        )
        forced_branch_index = abs(int(sampling_index // max(1, int(target_child_max - target_child_min + 1)))) % int(branch_count)
    elif str(case_kind) == MARKED_TOTAL_KIND:
        forced_count = int(target_marker_min) + (
            abs(int(sampling_index)) % (int(target_marker_max) - int(target_marker_min) + 1)
        )
        forced_branch_index = abs(int(sampling_index // max(1, int(target_marker_max - target_marker_min + 1)))) % int(branch_count)
    elif str(case_kind) == RANKED_CHILD_KIND:
        forced_branch_index = abs(int(sampling_index)) % int(branch_count)
        ranked_count_min = max(6, int(target_child_min))
        ranked_count_max = max(int(ranked_count_min), min(int(target_child_max), 7))
        forced_count = int(ranked_count_min) + (
            abs(int(sampling_index // max(1, int(branch_count)))) % (int(ranked_count_max) - int(ranked_count_min) + 1)
        )

    branches: list[Dict[str, Any]] = []
    used_labels: set[str] = set()
    for branch_index, (branch_label, item_pool) in enumerate(selected_branch_items):
        item_count = int(rng.randint(int(distractor_child_min), int(distractor_child_max)))
        if forced_branch_index is not None and int(branch_index) == int(forced_branch_index):
            item_count = int(forced_count or item_count)
            if str(case_kind) == MARKED_TOTAL_KIND:
                item_count = max(int(item_count), int(forced_count or item_count) + 1)
            item_count = min(int(target_child_max), int(item_count))
        pool = [str(item) for item in item_pool if str(item) not in used_labels]
        if len(pool) < int(item_count):
            pool = [str(item) for item in item_pool]
        labels = rng.sample(pool, int(item_count))
        for label in labels:
            used_labels.add(str(label))
        marker_ids: list[str] = []
        if str(case_kind) == MARKED_TOTAL_KIND and int(branch_index) == int(forced_branch_index):
            forced_positions = set(rng.sample(list(range(int(item_count))), int(forced_count or 1)))
            alternatives = [marker for marker in MARKERS if marker["marker_id"] != forced_marker["marker_id"]]
            for child_index in range(int(item_count)):
                if int(child_index) in forced_positions:
                    marker_ids.append(str(forced_marker["marker_id"]))
                else:
                    marker_ids.append(str(rng.choice(alternatives)["marker_id"]))
        else:
            for child_index in range(int(item_count)):
                marker_ids.append(str(MARKERS[(branch_index + child_index + abs(int(instance_seed))) % len(MARKERS)]["marker_id"]))
        branches.append(
            {
                "branch_id": f"branch_{branch_index}",
                "label": str(branch_label),
                "children": [
                    {
                        "node_id": f"child_{branch_index}_{child_index}",
                        "label": str(label),
                        "marker_id": str(marker_ids[child_index]),
                        "branch_id": f"branch_{branch_index}",
                    }
                    for child_index, label in enumerate(labels)
                ],
            }
        )

    canvas_width = resolve_render_int(params, RENDERING_DEFAULTS, "canvas_width", 1500, instance_seed=instance_seed, namespace=SCENE)
    canvas_height = resolve_render_int(params, RENDERING_DEFAULTS, "canvas_height", 1050, instance_seed=instance_seed, namespace=SCENE)
    outer_margin = resolve_render_int(params, RENDERING_DEFAULTS, "outer_margin_px", 46, instance_seed=instance_seed, namespace=SCENE)
    title_height = resolve_render_int(params, RENDERING_DEFAULTS, "title_band_height_px", 64, instance_seed=instance_seed, namespace=SCENE)
    panel_padding = resolve_render_int(params, RENDERING_DEFAULTS, "panel_padding_px", 26, instance_seed=instance_seed, namespace=SCENE)
    jitter_meta = resolve_layout_jitter(
        params,
        RENDERING_DEFAULTS,
        instance_seed=int(instance_seed),
        namespace=f"{NAMESPACE_ROOT}.panel",
    )
    panel, title_bbox, content, jitter = resolve_jittered_diagram_panel_geometry(
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        outer_margin_px=int(outer_margin),
        title_band_height_px=int(title_height),
        panel_padding_px=int(panel_padding),
        layout_jitter_meta=jitter_meta,
    )

    node_shape_profile = _assign_layout(
        branches=branches,
        central_label=str(context["central"]),
        content_bbox=content,
        layout_variant=str(layout_variant),
        instance_seed=int(instance_seed),
    )
    selection = _build_selection(
        case_kind=str(case_kind),
        branches=branches,
        answer_index=int(sampling_index),
        marker=forced_marker,
        child_min=int(target_child_min),
        child_max=int(target_child_max),
        marker_min=int(target_marker_min),
        marker_max=int(target_marker_max),
    )
    scene = {
        "scene_id": SCENE,
        "context_id": str(context_id),
        "scene_title": str(context["title"]),
        "central_label": str(context["central"]),
        "layout_variant": str(layout_variant),
        "style_variant": str(style_variant),
        "node_shape_profile": str(node_shape_profile),
        "branch_count": int(len(branches)),
        "child_count": int(sum(len(branch["children"]) for branch in branches)),
        "branches": deepcopy(branches),
        "canvas_width": int(canvas_width),
        "canvas_height": int(canvas_height),
        "panel_bbox": list(panel),
        "title_bbox": list(title_bbox),
        "content_bbox": list(content),
        "layout_jitter": dict(jitter),
    }
    return scene, selection


def _assign_layout(
    *,
    branches: list[Dict[str, Any]],
    central_label: str,
    content_bbox: Sequence[float],
    layout_variant: str,
    instance_seed: int,
) -> str:
    """Place all branch and child nodes while keeping reading-order boxes stable."""

    content = [float(value) for value in content_bbox]
    width = float(content[2] - content[0])
    height = float(content[3] - content[1])
    shape_profile, central_shape = _resolve_node_shape_profile(int(instance_seed))
    _assign_branch_child_shapes(branches, shape_profile=str(shape_profile), instance_seed=int(instance_seed))
    central_w = float(group_default(RENDERING_DEFAULTS, "central_node_width_px", 190))
    central_h = float(group_default(RENDERING_DEFAULTS, "central_node_height_px", 88))
    if str(central_shape) == "circle":
        central_diameter = float(group_default(RENDERING_DEFAULTS, "central_circle_diameter_px", 146))
        central_w = central_diameter
        central_h = central_diameter
    branch_w = float(group_default(RENDERING_DEFAULTS, "branch_node_width_px", 168))
    branch_h = float(group_default(RENDERING_DEFAULTS, "branch_node_height_px", 48))
    child_w = float(group_default(RENDERING_DEFAULTS, "child_node_width_px", 142))
    child_h = float(group_default(RENDERING_DEFAULTS, "child_node_height_px", 38))
    cx = float(content[0] + width * 0.5)
    cy = float(content[1] + height * 0.5)
    if str(layout_variant) == "left_right_map":
        cx = float(content[0] + width * 0.50)
    elif str(layout_variant) == "clustered_map":
        cy = float(content[1] + height * 0.38)
    central = {
        "node_id": "central",
        "label": str(central_label),
        "bbox": _node_bbox((cx, cy), central_w, central_h),
        "shape": str(central_shape),
    }

    if str(layout_variant) == "left_right_map":
        _assign_left_right_layout(branches, central, content, branch_w, branch_h, child_w, child_h)
    elif str(layout_variant) == "clustered_map":
        _assign_clustered_layout(branches, central, content, branch_w, branch_h, child_w, child_h)
    else:
        _assign_radial_layout(branches, central, content, branch_w, branch_h, child_w, child_h, int(instance_seed))
    for branch in branches:
        branch["children"].sort(key=child_sort_key)
    central["label"] = str(central_label)
    for branch in branches:
        branch["central"] = deepcopy(central)
    return str(shape_profile)


def _assign_radial_layout(
    branches: list[Dict[str, Any]],
    central: Dict[str, Any],
    content: Sequence[float],
    branch_w: float,
    branch_h: float,
    child_w: float,
    child_h: float,
    instance_seed: int,
) -> None:
    """Position branches around the central topic and push children outward."""

    center = _bbox_center(central["bbox"])
    count = len(branches)
    branch_radius = 235.0
    start = -math.pi / 2.0 + ((abs(int(instance_seed)) % 9) - 4) * 0.025
    for index, branch in enumerate(branches):
        angle = float(start + (2.0 * math.pi * index / max(1, count)))
        bx = float(center[0] + math.cos(angle) * branch_radius)
        by = float(center[1] + math.sin(angle) * branch_radius)
        bw, bh = _branch_dimensions(branch, branch_w, branch_h)
        branch["bbox"] = _node_bbox((bx, by), bw, bh)
        branch["anchor"] = (bx, by)
        children = branch["children"]
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        row_step = 43.0
        col_step = child_w + 12.0
        for child_index, child in enumerate(children):
            if abs(cos_a) >= abs(sin_a):
                cols = 2 if len(children) >= 5 else 1
                rows = int(math.ceil(len(children) / cols))
                local_col = child_index % cols
                local_row = child_index // cols
                total_h = (rows - 1) * row_step
                start_y = max(
                    float(content[1] + child_h / 2.0),
                    min(float(content[3] - child_h / 2.0 - total_h), by - total_h / 2.0),
                )
                if cos_a >= 0:
                    rx = float(content[2] - child_w / 2.0 - 8.0 - local_col * col_step)
                else:
                    rx = float(content[0] + child_w / 2.0 + 8.0 + local_col * col_step)
                ry = float(start_y + local_row * row_step)
            else:
                if sin_a > 0:
                    cols = min(2, max(1, len(children)))
                else:
                    cols = min(4, max(1, len(children)))
                rows = int(math.ceil(len(children) / cols))
                local_col = child_index % cols
                local_row = child_index // cols
                total_w = (cols - 1) * col_step
                outward_shift = cos_a * min(110.0, child_w * 0.75) if sin_a > 0 else 0.0
                start_x = max(
                    float(content[0] + child_w / 2.0),
                    min(float(content[2] - child_w / 2.0 - total_w), bx - total_w / 2.0 + outward_shift),
                )
                rx = float(start_x + local_col * col_step)
                if sin_a < 0:
                    ry = float(content[1] + child_h / 2.0 + 8.0 + local_row * row_step)
                else:
                    ry = float(content[3] - child_h / 2.0 - 8.0 - (rows - 1 - local_row) * row_step)
            child["bbox"] = _node_bbox((rx, ry), child_w, child_h)


def _assign_left_right_layout(
    branches: list[Dict[str, Any]],
    central: Dict[str, Any],
    content: Sequence[float],
    branch_w: float,
    branch_h: float,
    child_w: float,
    child_h: float,
) -> None:
    """Arrange branches on both sides of the hub with children inside side bands."""

    center = _bbox_center(central["bbox"])
    left = [branch for idx, branch in enumerate(branches) if idx % 2 == 0]
    right = [branch for idx, branch in enumerate(branches) if idx % 2 == 1]
    for side, group in (("left", left), ("right", right)):
        if not group:
            continue
        x_branch = float(center[0] - 260.0) if side == "left" else float(center[0] + 260.0)
        x_child = float(content[0] + child_w / 2.0 + 8.0) if side == "left" else float(content[2] - child_w / 2.0 - 8.0)
        child_col_step = float(child_w + 12.0)
        y_min = float(content[1] + 64.0)
        y_max = float(content[3] - 64.0)
        band_height = float((y_max - y_min) / max(1, len(group)))
        for order, branch in enumerate(group):
            band_top = float(y_min + order * band_height)
            band_bottom = float(y_min + (order + 1) * band_height)
            y = float((band_top + band_bottom) * 0.5)
            bw, bh = _branch_dimensions(branch, branch_w, branch_h)
            branch["bbox"] = _node_bbox((x_branch, y), bw, bh)
            branch["anchor"] = (x_branch, y)
            child_count = len(branch["children"])
            cols = 2 if child_count >= 5 else 1
            rows = int(math.ceil(child_count / cols))
            row_step = 43.0
            if rows > 1:
                row_step = min(row_step, max(child_h + 5.0, (band_height - child_h - 12.0) / (rows - 1)))
            total_h = (rows - 1) * row_step
            start_y = max(
                float(band_top + child_h / 2.0 + 4.0),
                min(float(band_bottom - child_h / 2.0 - total_h - 4.0), y - total_h / 2.0),
            )
            for child_index, child in enumerate(branch["children"]):
                local_col = child_index % cols
                local_row = child_index // cols
                if side == "left":
                    child_x = float(x_child + local_col * child_col_step)
                else:
                    child_x = float(x_child - local_col * child_col_step)
                child_y = float(start_y + local_row * row_step)
                child["bbox"] = _node_bbox((child_x, child_y), child_w, child_h)


def _assign_clustered_layout(
    branches: list[Dict[str, Any]],
    central: Dict[str, Any],
    content: Sequence[float],
    branch_w: float,
    branch_h: float,
    child_w: float,
    child_h: float,
) -> None:
    columns = 3 if len(branches) >= 6 else 2
    top_y = float(content[1] + 92.0)
    bottom_y = float(content[3] - 160.0)
    x_gap = float((content[2] - content[0]) / columns)
    for index, branch in enumerate(branches):
        col = index % columns
        row = index // columns
        rows = int(math.ceil(len(branches) / columns))
        bx = float(content[0] + x_gap * (col + 0.5))
        by = top_y if rows <= 1 else top_y + (bottom_y - top_y) * row / max(1, rows - 1)
        bw, bh = _branch_dimensions(branch, branch_w, branch_h)
        branch["bbox"] = _node_bbox((bx, by), bw, bh)
        branch["anchor"] = (bx, by)
        child_count = len(branch["children"])
        cols = 2 if child_count >= 5 else 1
        for child_index, child in enumerate(branch["children"]):
            local_col = child_index % cols
            local_row = child_index // cols
            child_x = bx + (local_col - (cols - 1) / 2.0) * (child_w + 12.0)
            child_y = by + bh / 2.0 + child_h / 2.0 + 10.0 + local_row * 42.0
            child_y = min(float(content[3] - child_h / 2.0), child_y)
            child["bbox"] = _node_bbox((child_x, child_y), child_w, child_h)


def _build_selection(
    *,
    case_kind: str,
    branches: list[Dict[str, Any]],
    answer_index: int,
    marker: Mapping[str, Any],
    child_min: int,
    child_max: int,
    marker_min: int,
    marker_max: int,
) -> Dict[str, Any]:
    """Select the task target after layout so annotations refer to final node ids."""

    if str(case_kind) == CHILDREN_TOTAL_KIND:
        branch_index, _target = _select_branch_by_answer_index(
            branches=branches,
            answer_index=int(answer_index),
            min_count=int(child_min),
            max_count=int(child_max),
        )
        branch = branches[int(branch_index)]
        return {
            "branch_id": str(branch["branch_id"]),
            "branch_label": str(branch["label"]),
            "answer": int(len(branch["children"])),
            "annotation_node_ids": [str(child["node_id"]) for child in branch["children"]],
        }
    if str(case_kind) == RANKED_CHILD_KIND:
        desired_rank = 2 + (abs(int(answer_index)) % 4)
        eligible = [branch for branch in branches if len(branch["children"]) > int(desired_rank)]
        if not eligible:
            eligible = [branch for branch in branches if len(branch["children"]) >= 3]
        branch = eligible[abs(int(answer_index)) % len(eligible)]
        ordered = sorted(branch["children"], key=child_sort_key)
        allowed_ranks = list(range(2, min(5, len(ordered) - 1) + 1))
        if not allowed_ranks:
            allowed_ranks = [min(2, len(ordered))]
        rank = int(allowed_ranks[abs(int(answer_index)) % len(allowed_ranks)])
        target = ordered[int(rank) - 1]
        return {
            "branch_id": str(branch["branch_id"]),
            "branch_label": str(branch["label"]),
            "answer_node_id": str(target["node_id"]),
            "rank": int(rank),
            "rank_ordinal": ordinal_label(int(rank)),
            "reading_order": "from top to bottom, breaking ties from left to right",
            "answer": str(target["label"]),
            "annotation_node_ids": [str(target["node_id"])],
        }
    if str(case_kind) != MARKED_TOTAL_KIND:
        raise ValueError(f"unsupported concept-map case kind: {case_kind}")
    target = int(marker_min) + (abs(int(answer_index)) % (int(marker_max) - int(marker_min) + 1))
    eligible = [
        branch for branch in branches
        if sum(1 for child in branch["children"] if str(child["marker_id"]) == str(marker["marker_id"])) == int(target)
    ]
    if not eligible:
        closest_distance = min(
            abs(sum(1 for child in branch["children"] if str(child["marker_id"]) == str(marker["marker_id"])) - int(target))
            for branch in branches
        )
        eligible = [
            branch for branch in branches
            if abs(sum(1 for child in branch["children"] if str(child["marker_id"]) == str(marker["marker_id"])) - int(target))
            == int(closest_distance)
        ]
    branch = eligible[abs(int(answer_index)) % len(eligible)]
    matched = [child for child in branch["children"] if str(child["marker_id"]) == str(marker["marker_id"])]
    return {
        "branch_id": str(branch["branch_id"]),
        "branch_label": str(branch["label"]),
        "marker_id": str(marker["marker_id"]),
        "marker_label": str(marker["label"]).removesuffix(" marker"),
        "marker_display_label": str(marker["label"]),
        "answer": int(len(matched)),
        "annotation_node_ids": [str(child["node_id"]) for child in matched],
    }
