"""Scene-local sampling primitives for process-flow page scenes."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any, Dict, Mapping, Sequence

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.pages.shared.common import resolve_pages_axis_variant
from trace_tasks.tasks.pages.shared.diagram.common import resolve_jittered_diagram_panel_geometry
from trace_tasks.tasks.shared.config_defaults import resolve_required_int_bounds
from trace_tasks.tasks.shared.render_variation import resolve_layout_jitter, resolve_render_int

from .defaults import GENERATION_DEFAULTS, NAMESPACE_ROOT, RENDER_FALLBACKS, RENDERING_DEFAULTS
from .state import (
    CONDITION_POOLS,
    CONTEXTS,
    LAYOUT_VARIANTS,
    ROLE_DESCRIPTIONS,
    SCENE_VARIANTS,
    SHAPE_DESCRIPTIONS,
    STATUS_POOL,
    STYLE_VARIANTS,
    BBox,
    ProcessFlowRenderParams,
    ProcessFlowSceneCase,
)


def _round_box(box: Sequence[float]) -> list[float]:
    return [round(float(value), 3) for value in box]


def _shape_for_role(role: str) -> str:
    return {
        "start": "ellipse",
        "decision": "diamond",
        "review": "rounded",
        "data": "parallelogram",
        "output": "ellipse",
    }.get(str(role), "rounded")


_CLEAN_BRANCH_PATTERN_COUNT = 4


def _clean_branch_lane_pattern(*, lane_count: int, pattern_index: int) -> Dict[str, int]:
    """Return a split/merge lane skeleton that avoids branch-flow wire clutter."""

    lane_count = max(3, int(lane_count))
    center = max(1, min(lane_count - 2, (lane_count - 1) // 2))
    upper = max(0, center - 1)
    lower = min(lane_count - 1, center + 1)
    if lower == upper:
        lower = min(lane_count - 1, upper + 1)

    branch_a = (upper, lower)
    branch_b = (upper, lower)
    variant = int(pattern_index) % _CLEAN_BRANCH_PATTERN_COUNT
    if variant == 1:
        branch_a = (lower, upper)
    elif variant == 2:
        branch_b = (lower, upper)
    elif variant == 3:
        branch_a = (lower, upper)
        branch_b = (lower, upper)

    return {
        "n0": center,
        "n1": center,
        "n2": center,
        "n3": branch_a[0],
        "n4": branch_a[1],
        "n5": center,
        "n6": center,
        "n7": branch_b[0],
        "n8": branch_b[1],
        "n9": center,
    }


def _resolve_axis(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    supported: Sequence[str],
    explicit_key: str,
    weights_key: str,
    balance_flag_key: str,
    namespace: str,
) -> tuple[str, Dict[str, float]]:
    return resolve_pages_axis_variant(
        params=params,
        gen_defaults=GENERATION_DEFAULTS,
        instance_seed=int(instance_seed),
        supported_variants=tuple(str(value) for value in supported),
        task_id=f"{NAMESPACE_ROOT}.{namespace}",
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
        balance_flag_key=str(balance_flag_key),
        axis_namespace=str(namespace),
    )


def resolve_render_params(
    params: Mapping[str, Any],
    *,
    instance_seed: int,
) -> ProcessFlowRenderParams:
    """Resolve render parameters and layout jitter for one process-flow scene."""

    def _int(key: str, fallback: int) -> int:
        return resolve_render_int(
            params,
            RENDERING_DEFAULTS,
            key,
            fallback,
            instance_seed=int(instance_seed),
            namespace=NAMESPACE_ROOT,
        )

    jitter_meta = resolve_layout_jitter(
        params,
        RENDERING_DEFAULTS,
        instance_seed=int(instance_seed),
        namespace=f"{NAMESPACE_ROOT}.layout",
    )
    return ProcessFlowRenderParams(
        canvas_width=_int("canvas_width", int(RENDER_FALLBACKS["canvas_width"])),
        canvas_height=_int("canvas_height", int(RENDER_FALLBACKS["canvas_height"])),
        outer_margin_px=_int("outer_margin_px", int(RENDER_FALLBACKS["outer_margin_px"])),
        panel_padding_px=_int("panel_padding_px", int(RENDER_FALLBACKS["panel_padding_px"])),
        panel_corner_radius_px=_int(
            "panel_corner_radius_px",
            int(RENDER_FALLBACKS["panel_corner_radius_px"]),
        ),
        title_band_height_px=_int(
            "title_band_height_px",
            int(RENDER_FALLBACKS["title_band_height_px"]),
        ),
        lane_header_height_px=_int(
            "lane_header_height_px",
            int(RENDER_FALLBACKS["lane_header_height_px"]),
        ),
        node_width_px=_int("node_width_px", int(RENDER_FALLBACKS["node_width_px"])),
        node_height_px=_int("node_height_px", int(RENDER_FALLBACKS["node_height_px"])),
        node_corner_radius_px=_int(
            "node_corner_radius_px",
            int(RENDER_FALLBACKS["node_corner_radius_px"]),
        ),
        node_border_width_px=_int(
            "node_border_width_px",
            int(RENDER_FALLBACKS["node_border_width_px"]),
        ),
        edge_width_px=_int("edge_width_px", int(RENDER_FALLBACKS["edge_width_px"])),
        arrow_head_length_px=_int(
            "arrow_head_length_px",
            int(RENDER_FALLBACKS["arrow_head_length_px"]),
        ),
        arrow_head_width_px=_int(
            "arrow_head_width_px",
            int(RENDER_FALLBACKS["arrow_head_width_px"]),
        ),
        title_font_size_px=_int(
            "title_font_size_px",
            int(RENDER_FALLBACKS["title_font_size_px"]),
        ),
        lane_font_size_px=_int("lane_font_size_px", int(RENDER_FALLBACKS["lane_font_size_px"])),
        node_label_font_size_px=_int(
            "node_label_font_size_px",
            int(RENDER_FALLBACKS["node_label_font_size_px"]),
        ),
        badge_font_size_px=_int("badge_font_size_px", int(RENDER_FALLBACKS["badge_font_size_px"])),
        edge_label_font_size_px=_int(
            "edge_label_font_size_px",
            int(RENDER_FALLBACKS["edge_label_font_size_px"]),
        ),
        layout_jitter_meta=dict(jitter_meta),
    )


def panel_geometry(
    render_params: ProcessFlowRenderParams,
) -> tuple[BBox, BBox, BBox, ProcessFlowRenderParams]:
    """Return canonical diagram panel geometry with resolved layout jitter."""

    panel, title_bbox, content, jitter_meta = resolve_jittered_diagram_panel_geometry(
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
        outer_margin_px=int(render_params.outer_margin_px),
        title_band_height_px=int(render_params.title_band_height_px),
        panel_padding_px=int(render_params.panel_padding_px),
        layout_jitter_meta=dict(render_params.layout_jitter_meta),
    )
    resolved_params = replace(render_params, layout_jitter_meta=dict(jitter_meta))
    return (
        tuple(float(value) for value in panel),
        tuple(float(value) for value in title_bbox),
        tuple(float(value) for value in content),
        resolved_params,
    )


def _make_base_nodes(
    *,
    rng: Any,
    context: Mapping[str, Any],
    lanes: Sequence[str],
    target_node_count: int,
    endpoint_first_choice: int | None,
    lane_pattern_index: int | None,
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]], Dict[str, str]]:
    """Build the directed process graph with stable decision-path support."""

    labels = list(dict.fromkeys(str(label) for label in context["steps"]))
    if len(labels) > 3:
        offset = int(rng.randrange(0, min(3, len(labels))))
        labels = labels[offset:] + labels[:offset]
    lane_count = len(lanes)
    pattern_index = int(lane_pattern_index or 0) % _CLEAN_BRANCH_PATTERN_COUNT
    lane_pattern = _clean_branch_lane_pattern(
        lane_count=int(lane_count),
        pattern_index=int(pattern_index),
    )
    lane_for = {
        node_ref: max(0, min(int(lane_index), lane_count - 1))
        for node_ref, lane_index in lane_pattern.items()
    }
    roles = {
        "n0": "start",
        "n1": "process",
        "n2": "decision",
        "n3": "process",
        "n4": "review",
        "n5": "data",
        "n6": "decision",
        "n7": "process",
        "n8": "output",
        "n9": "output",
    }
    levels = {
        "n0": 0,
        "n1": 1,
        "n2": 2,
        "n3": 3,
        "n4": 3,
        "n5": 4,
        "n6": 5,
        "n7": 6,
        "n8": 6,
        "n9": 7,
    }
    nodes: list[Dict[str, Any]] = []
    for index, node_ref in enumerate([f"n{item}" for item in range(10)]):
        role = roles[node_ref]
        label = labels[index % len(labels)]
        if role == "start":
            label = labels[0]
        elif node_ref == "n9":
            label = labels[9 % len(labels)]
        nodes.append(
            {
                "node_id": node_ref,
                "bbox_id": node_ref,
                "label": label,
                "lane": str(lanes[lane_for[node_ref] % lane_count]),
                "lane_index": int(lane_for[node_ref] % lane_count),
                "role": role,
                "shape": _shape_for_role(role),
                "status": str(rng.choice(STATUS_POOL)),
                "level": int(levels[node_ref]),
                "order": int(index),
            }
        )
    cond_pair_a = list(rng.choice(CONDITION_POOLS))
    cond_pair_b = list(rng.choice(CONDITION_POOLS))
    if cond_pair_b == cond_pair_a:
        next_index = (list(CONDITION_POOLS).index(tuple(cond_pair_a)) + 2) % len(CONDITION_POOLS)
        cond_pair_b = list(CONDITION_POOLS[next_index])
    condition_map = {
        "first_left": str(cond_pair_a[0]),
        "first_right": str(cond_pair_a[1]),
        "second_left": str(cond_pair_b[0]),
        "second_right": str(cond_pair_b[1]),
    }
    edges = [
        {"edge_id": "e0", "source": "n0", "target": "n1", "label": "", "kind": "branch_clean"},
        {"edge_id": "e1", "source": "n1", "target": "n2", "label": "", "kind": "branch_clean"},
        {
            "edge_id": "e2",
            "source": "n2",
            "target": "n3",
            "label": condition_map["first_left"],
            "kind": "decision_clean",
        },
        {
            "edge_id": "e3",
            "source": "n2",
            "target": "n4",
            "label": condition_map["first_right"],
            "kind": "decision_clean",
        },
        {"edge_id": "e4", "source": "n3", "target": "n5", "label": "", "kind": "branch_clean"},
        {"edge_id": "e5", "source": "n4", "target": "n5", "label": "", "kind": "branch_clean"},
        {"edge_id": "e6", "source": "n5", "target": "n6", "label": "", "kind": "branch_clean"},
        {
            "edge_id": "e7",
            "source": "n6",
            "target": "n7",
            "label": condition_map["second_left"],
            "kind": "decision_clean",
        },
        {
            "edge_id": "e8",
            "source": "n6",
            "target": "n8",
            "label": condition_map["second_right"],
            "kind": "decision_clean",
        },
        {"edge_id": "e9", "source": "n7", "target": "n9", "label": "", "kind": "branch_clean"},
        {"edge_id": "e10", "source": "n8", "target": "n9", "label": "", "kind": "branch_clean"},
    ]

    extra_count = max(0, int(target_node_count) - len(nodes))
    route_branch = "n3" if endpoint_first_choice == 0 else "n4" if endpoint_first_choice == 1 else None
    extra_sources = ["n3", "n4", "n5"]
    if route_branch in extra_sources:
        extra_sources = [source for source in extra_sources if source != route_branch]
    extra_labels = [label for label in labels if label not in {str(node["label"]) for node in nodes}]
    for extra_idx in range(extra_count):
        role = str(rng.choice(("process", "review", "data")))
        node_ref = f"n{10 + extra_idx}"
        lane_index = int(rng.randrange(0, lane_count))
        label = str(extra_labels[extra_idx % len(extra_labels)] if extra_labels else f"Step {extra_idx + 1}")
        nodes.append(
            {
                "node_id": node_ref,
                "bbox_id": node_ref,
                "label": label,
                "lane": str(lanes[lane_index]),
                "lane_index": int(lane_index),
                "role": role,
                "shape": _shape_for_role(role),
                "status": str(rng.choice(STATUS_POOL)),
                "level": int(rng.choice([3, 4, 5, 6])),
                "order": int(10 + extra_idx),
            }
        )
        source = str(rng.choice(extra_sources))
        target = "n5" if source in {"n3", "n4"} else "n6"
        edges.append(
            {
                "edge_id": f"e{11 + (2 * extra_idx)}",
                "source": source,
                "target": node_ref,
                "label": str(rng.choice(("audit", "copy", "notify", "check"))),
                "kind": "side",
            }
        )
        edges.append(
            {
                "edge_id": f"e{12 + (2 * extra_idx)}",
                "source": node_ref,
                "target": target,
                "label": "",
                "kind": "auto",
            }
        )
    return nodes, edges, condition_map


def _make_handoff_chain_nodes(
    *,
    rng: Any,
    context: Mapping[str, Any],
    lanes: Sequence[str],
    target_node_count: int,
    target_cross_count: int,
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]], Dict[str, str]]:
    """Build a clean adjacent-stage process chain for handoff-count tasks."""

    labels = list(dict.fromkeys(str(label) for label in context["steps"]))
    if len(labels) > 3:
        offset = int(rng.randrange(0, min(3, len(labels))))
        labels = labels[offset:] + labels[:offset]
    lane_count = int(len(lanes))
    stage_count = max(7, min(9, int(target_node_count)))
    cross_count = max(3, min(int(target_cross_count), stage_count - 1))
    if cross_count >= 7 and stage_count >= 9 and lane_count >= 3:
        base_lane = int(rng.randrange(1, lane_count - 1))
        paired_lane = base_lane + int(rng.choice((-1, 1)))
        lane_indices = [base_lane]
        for transition_index in range(stage_count - 1):
            current = int(lane_indices[-1])
            if transition_index < cross_count:
                lane_indices.append(paired_lane if current == base_lane else base_lane)
            else:
                lane_indices.append(current)
    else:
        transition_indices = set(rng.sample(range(stage_count - 1), k=int(cross_count)))
        lane_indices = [int(rng.randrange(0, lane_count))]
        for transition_index in range(stage_count - 1):
            current = int(lane_indices[-1])
            if transition_index in transition_indices:
                options = []
                for delta in (-1, 1):
                    candidate = current + delta
                    if 0 <= candidate < lane_count:
                        options.append(int(candidate))
                if not options:
                    options = [idx for idx in range(lane_count) if idx != current]
                lane_indices.append(int(rng.choice(options)))
            else:
                lane_indices.append(current)

    roles: list[str] = ["start"]
    middle_roles = ("process", "review", "data")
    for index in range(1, stage_count - 1):
        roles.append(str(middle_roles[(index + int(rng.randrange(0, len(middle_roles)))) % len(middle_roles)]))
    roles.append("output")

    nodes: list[Dict[str, Any]] = []
    for index in range(stage_count):
        node_ref = f"n{index}"
        role = roles[index]
        nodes.append(
            {
                "node_id": node_ref,
                "bbox_id": node_ref,
                "label": str(labels[index % len(labels)]),
                "lane": str(lanes[int(lane_indices[index]) % lane_count]),
                "lane_index": int(lane_indices[index] % lane_count),
                "role": str(role),
                "shape": _shape_for_role(role),
                "status": str(rng.choice(STATUS_POOL)),
                "level": int(index),
                "order": int(index),
            }
        )

    edges: list[Dict[str, Any]] = []
    for index in range(stage_count - 1):
        edges.append(
            {
                "edge_id": f"e{index}",
                "source": f"n{index}",
                "target": f"n{index + 1}",
                "label": "",
                "kind": "handoff_chain",
            }
        )
    return nodes, edges, {}


def assign_layout(
    *,
    nodes: list[Dict[str, Any]],
    lanes: Sequence[str],
    content_bbox: Sequence[float],
    layout_variant: str,
    flow_family: str,
    render_params: ProcessFlowRenderParams,
    instance_seed: int,
    namespace: str,
) -> tuple[list[Dict[str, Any]], Dict[str, list[float]]]:
    """Place nodes into lane bands while preserving source lane membership."""

    rng = spawn_rng(int(instance_seed), f"{namespace}.layout.{layout_variant}")
    left, top, right, bottom = [float(value) for value in content_bbox]
    lane_bboxes: Dict[str, list[float]] = {}
    vertical = str(layout_variant) in {"vertical_swimlane", "staggered_columns"}
    lane_header_h = float(
        render_params.lane_header_height_px
        if vertical
        else max(112.0 if str(flow_family) == "handoff_chain" else 88.0, render_params.lane_header_height_px)
    )
    node_w_default = float(render_params.node_width_px)
    node_h_default = float(render_params.node_height_px)
    max_level = max(int(node["level"]) for node in nodes)
    occupancy: Dict[tuple[int, int], int] = {}
    if vertical:
        lane_w = float(right - left) / float(len(lanes))
        flow_top = top + lane_header_h
        flow_bottom = bottom
        for idx, lane in enumerate(lanes):
            lx0 = left + (idx * lane_w)
            lx1 = left + ((idx + 1) * lane_w)
            lane_bboxes[str(lane)] = _round_box([lx0, top, lx1, bottom])
        level_gap = float(flow_bottom - flow_top) / float(max(1, max_level))
        node_w = min(node_w_default, max(104.0, lane_w - 26.0))
        node_h = node_h_default
        for node in nodes:
            lane_index = int(node["lane_index"])
            level = int(node["level"])
            key = (lane_index, level)
            slot = occupancy.get(key, 0)
            occupancy[key] = slot + 1
            lane_center = left + ((lane_index + 0.5) * lane_w)
            if str(layout_variant) == "staggered_columns":
                x_jitter = rng.uniform(-0.12, 0.12) * lane_w
            else:
                x_jitter = rng.uniform(-7.0, 7.0)
            y_jitter = (slot * (node_h * 0.50)) + rng.uniform(-8.0, 8.0)
            if slot:
                x_jitter += ((-1) ** slot) * min(24.0, 0.12 * lane_w)
            cx = max(
                left + 0.5 * node_w + 8.0,
                min(right - 0.5 * node_w - 8.0, lane_center + x_jitter),
            )
            cy = max(
                flow_top + 0.5 * node_h + 8.0,
                min(flow_bottom - 0.5 * node_h - 8.0, flow_top + (level * level_gap) + y_jitter),
            )
            node["center"] = [round(cx, 3), round(cy, 3)]
            node["width"] = round(node_w, 3)
            node["height"] = round(node_h, 3)
            node["bbox"] = _round_box(
                [cx - 0.5 * node_w, cy - 0.5 * node_h, cx + 0.5 * node_w, cy + 0.5 * node_h]
            )
        for lane_index in range(len(lanes)):
            lane_nodes = sorted(
                [node for node in nodes if int(node["lane_index"]) == int(lane_index)],
                key=lambda item: (int(item["level"]), int(item["order"])),
            )
            if len(lane_nodes) <= 1:
                continue
            y_min = flow_top + (0.5 * node_h) + 10.0
            y_max = flow_bottom - (0.5 * node_h) - 10.0
            gap = float(y_max - y_min) / float(max(1, len(lane_nodes) - 1))
            for idx, node in enumerate(lane_nodes):
                cx = float(node["center"][0])
                cy = float(y_min + (idx * gap))
                node["center"] = [round(cx, 3), round(cy, 3)]
                node["bbox"] = _round_box(
                    [cx - 0.5 * node_w, cy - 0.5 * node_h, cx + 0.5 * node_w, cy + 0.5 * node_h]
                )
    else:
        lane_h = float(bottom - top) / float(len(lanes))
        flow_left = left + lane_header_h
        flow_right = right
        for idx, lane in enumerate(lanes):
            ly0 = top + (idx * lane_h)
            ly1 = top + ((idx + 1) * lane_h)
            lane_bboxes[str(lane)] = _round_box([left, ly0, right, ly1])
        raw_level_gap = float(flow_right - flow_left) / float(max(1, max_level + 1))
        if str(flow_family) == "handoff_chain":
            node_w = min(node_w_default, max(84.0, raw_level_gap * 0.66))
            node_h = min(node_h_default, max(48.0, lane_h - 26.0))
        elif str(flow_family) == "branch_flow":
            node_w = min(node_w_default, max(86.0, raw_level_gap * 0.72))
            node_h = min(node_h_default, max(50.0, lane_h - 26.0))
        else:
            node_w = min(node_w_default, max(110.0, raw_level_gap * 0.88))
            node_h = min(node_h_default, max(50.0, lane_h - 22.0))
        x_min = flow_left + (0.5 * node_w) + 10.0
        x_max = flow_right - (0.5 * node_w) - 10.0
        level_gap = float(x_max - x_min) / float(max(1, max_level))
        for node in nodes:
            lane_index = int(node["lane_index"])
            level = int(node["level"])
            key = (lane_index, level)
            slot = occupancy.get(key, 0)
            occupancy[key] = slot + 1
            lane_center = top + ((lane_index + 0.5) * lane_h)
            x_jitter = (slot * (node_w * 0.18)) + rng.uniform(-2.0, 2.0)
            if str(layout_variant) == "compact_rows":
                y_jitter = rng.uniform(-0.10, 0.10) * lane_h
            elif str(flow_family) == "handoff_chain":
                y_jitter = rng.uniform(-3.0, 3.0)
            else:
                y_jitter = rng.uniform(-7.0, 7.0)
            cy = max(
                top + 0.5 * node_h + 7.0,
                min(bottom - 0.5 * node_h - 7.0, lane_center + y_jitter),
            )
            cx = max(
                x_min,
                min(x_max, x_min + (level * level_gap) + x_jitter),
            )
            node["center"] = [round(cx, 3), round(cy, 3)]
            node["width"] = round(node_w, 3)
            node["height"] = round(node_h, 3)
            node["bbox"] = _round_box(
                [cx - 0.5 * node_w, cy - 0.5 * node_h, cx + 0.5 * node_w, cy + 0.5 * node_h]
            )
        # Keep horizontal swimlane x-positions tied to process level. Reflowing
        # each lane independently makes later process steps appear to the left
        # of earlier steps and produces backtracking/crossing arrows.
    return nodes, lane_bboxes


def build_scene_case(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    namespace: str,
    endpoint_first_choice: int | None = None,
) -> ProcessFlowSceneCase:
    """Sample and lay out one complete process-flow diagram state."""

    flow_family = str(params.get("flow_family", "branch_flow")).strip() or "branch_flow"
    scene_variant, scene_probs = _resolve_axis(
        params=params,
        instance_seed=int(instance_seed),
        supported=SCENE_VARIANTS,
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        namespace="scene_variant",
    )
    layout_variant, layout_probs = _resolve_axis(
        params=params,
        instance_seed=int(instance_seed),
        supported=LAYOUT_VARIANTS,
        explicit_key="layout_variant",
        weights_key="layout_variant_weights",
        balance_flag_key="balanced_layout_variant_sampling",
        namespace="layout_variant",
    )
    style_variant, style_probs = _resolve_axis(
        params=params,
        instance_seed=int(instance_seed),
        supported=STYLE_VARIANTS,
        explicit_key="style_variant",
        weights_key="style_variant_weights",
        balance_flag_key="balanced_style_variant_sampling",
        namespace="style_variant",
    )
    lane_min, lane_max = resolve_required_int_bounds(
        params,
        GENERATION_DEFAULTS,
        min_key="lane_count_min",
        max_key="lane_count_max",
        fallback_min=3,
        fallback_max=5,
        context="process_flow lanes",
    )
    node_min, node_max = resolve_required_int_bounds(
        params,
        GENERATION_DEFAULTS,
        min_key="node_count_min",
        max_key="node_count_max",
        fallback_min=10,
        fallback_max=14,
        context="process_flow nodes",
    )
    rng = spawn_rng(int(instance_seed), f"{namespace}.dataset")
    lane_count = int(rng.randint(int(lane_min), int(lane_max)))
    node_count = int(rng.randint(int(node_min), int(node_max)))
    context = deepcopy(CONTEXTS[str(scene_variant)])
    lanes = list(context["lanes"])
    rng.shuffle(lanes)
    lanes = [str(lane) for lane in lanes[:lane_count]]
    if flow_family == "handoff_chain":
        lane_pattern_index = -1
        target_cross_count = int(params.get("target_cross_count", rng.randint(3, 6)))
        nodes, edges, condition_map = _make_handoff_chain_nodes(
            rng=rng,
            context=context,
            lanes=lanes,
            target_node_count=node_count,
            target_cross_count=target_cross_count,
        )
    else:
        raw_lane_pattern_index = params.get("lane_pattern_index")
        if raw_lane_pattern_index is None:
            lane_pattern_index = int(rng.randrange(_CLEAN_BRANCH_PATTERN_COUNT))
        else:
            lane_pattern_index = int(raw_lane_pattern_index) % _CLEAN_BRANCH_PATTERN_COUNT
        nodes, edges, condition_map = _make_base_nodes(
            rng=rng,
            context=context,
            lanes=lanes,
            target_node_count=node_count,
            endpoint_first_choice=endpoint_first_choice,
            lane_pattern_index=int(lane_pattern_index),
        )
    render_params = resolve_render_params(params, instance_seed=int(instance_seed))
    panel, title_bbox, content, render_params = panel_geometry(render_params)
    nodes, lane_bboxes = assign_layout(
        nodes=nodes,
        lanes=lanes,
        content_bbox=content,
        layout_variant=str(layout_variant),
        flow_family=str(flow_family),
        render_params=render_params,
        instance_seed=int(instance_seed),
        namespace=str(namespace),
    )
    return ProcessFlowSceneCase(
        scene_variant=str(scene_variant),
        layout_variant=str(layout_variant),
        style_variant=str(style_variant),
        flow_family=str(flow_family),
        lane_pattern_index=int(lane_pattern_index),
        scene_title=str(context["title"]),
        lanes=tuple(str(lane) for lane in lanes),
        nodes=tuple(dict(node) for node in nodes),
        edges=tuple(dict(edge) for edge in edges),
        condition_map=dict(condition_map),
        panel_bbox=tuple(float(value) for value in panel),
        title_bbox=tuple(float(value) for value in title_bbox),
        content_bbox=tuple(float(value) for value in content),
        lane_bboxes=dict(lane_bboxes),
        render_params=render_params,
        scene_variant_probabilities=dict(scene_probs),
        layout_variant_probabilities=dict(layout_probs),
        style_variant_probabilities=dict(style_probs),
    )


def choose_answer_bucket_candidate(
    candidates: Sequence[Mapping[str, Any]],
    *,
    rng: Any,
    min_answer: int = 1,
) -> Dict[str, Any]:
    """Choose a candidate by answer bucket first, then by candidate."""

    pool = [dict(candidate) for candidate in candidates]
    if not pool:
        raise ValueError("process-flow candidate selector received no candidates")
    eligible = [candidate for candidate in pool if int(candidate.get("answer", 0)) >= int(min_answer)]
    if eligible:
        pool = eligible
    by_answer: Dict[int, list[Dict[str, Any]]] = {}
    for candidate in pool:
        by_answer.setdefault(int(candidate["answer"]), []).append(candidate)
    answer_bucket = int(rng.choice(sorted(by_answer)))
    return dict(rng.choice(by_answer[answer_bucket]))


def choose_from_answer_support(
    candidates: Sequence[Mapping[str, Any]],
    *,
    rng: Any,
    answer_support: Sequence[int],
) -> Dict[str, Any]:
    """Choose a candidate near a seeded answer support value."""

    by_answer: Dict[int, list[Dict[str, Any]]] = {}
    for candidate in candidates:
        by_answer.setdefault(int(candidate["answer"]), []).append(dict(candidate))
    if not by_answer:
        raise ValueError("process-flow answer-support selector received no candidates")
    support = [int(value) for value in answer_support]
    target = int(support[int(rng.randrange(len(support)))])
    if target in by_answer:
        answer_bucket = int(target)
    else:
        available = sorted(by_answer)
        distance = min(abs(int(value) - int(target)) for value in available)
        tied = [int(value) for value in available if abs(int(value) - int(target)) == int(distance)]
        answer_bucket = int(max(tied))
    return dict(rng.choice(by_answer[int(answer_bucket)]))


def node_filter_candidates(
    *,
    nodes: Sequence[Mapping[str, Any]],
    filter_kind: str,
) -> list[Dict[str, Any]]:
    """Return count candidates for one visible node-filter family."""

    candidates: list[Dict[str, Any]] = []
    visible_status_nodes = [node for node in nodes if str(node["role"]) not in {"start", "output"}]
    if str(filter_kind) == "shape":
        for shape, description in SHAPE_DESCRIPTIONS.items():
            ids = [str(node["node_id"]) for node in nodes if str(node["shape"]) == str(shape)]
            if ids:
                candidates.append(
                    {
                        "shape": str(shape),
                        "shape_description": str(description),
                        "shape_filter_description": str(description),
                        "filter_mode": "include",
                        "answer": int(len(ids)),
                        "annotation_node_ids": ids,
                        "answer_type": "integer",
                    }
                )
            complement_ids = [
                str(node["node_id"]) for node in nodes if str(node["shape"]) != str(shape)
            ]
            if complement_ids:
                candidates.append(
                    {
                        "shape": str(shape),
                        "shape_description": str(description),
                        "shape_filter_description": f"not {description}",
                        "filter_mode": "exclude",
                        "answer": int(len(complement_ids)),
                        "annotation_node_ids": complement_ids,
                        "answer_type": "integer",
                    }
                )
    elif str(filter_kind) == "status":
        for status in STATUS_POOL:
            ids = [
                str(node["node_id"])
                for node in visible_status_nodes
                if str(node["status"]) == str(status)
            ]
            if ids:
                candidates.append(
                    {
                        "status_name": str(status),
                        "status_filter_description": f"marked {status}",
                        "filter_mode": "include",
                        "answer": int(len(ids)),
                        "annotation_node_ids": ids,
                        "answer_type": "integer",
                    }
                )
            complement_ids = [
                str(node["node_id"])
                for node in visible_status_nodes
                if str(node["status"]) != str(status)
            ]
            if complement_ids:
                candidates.append(
                    {
                        "status_name": str(status),
                        "status_filter_description": f"not marked {status}",
                        "filter_mode": "exclude",
                        "answer": int(len(complement_ids)),
                        "annotation_node_ids": complement_ids,
                        "answer_type": "integer",
                    }
                )
    elif str(filter_kind) == "role":
        for role, description in ROLE_DESCRIPTIONS.items():
            ids = [str(node["node_id"]) for node in nodes if str(node["role"]) == str(role)]
            if ids:
                candidates.append(
                    {
                        "role_name": str(role),
                        "role_description": str(description),
                        "role_filter_description": str(description),
                        "filter_mode": "include",
                        "answer": int(len(ids)),
                        "annotation_node_ids": ids,
                        "answer_type": "integer",
                    }
                )
            complement_ids = [
                str(node["node_id"]) for node in nodes if str(node["role"]) != str(role)
            ]
            if complement_ids:
                candidates.append(
                    {
                        "role_name": str(role),
                        "role_description": str(description),
                        "role_filter_description": f"not {description}",
                        "filter_mode": "exclude",
                        "answer": int(len(complement_ids)),
                        "annotation_node_ids": complement_ids,
                        "answer_type": "integer",
                    }
                )
    else:
        raise ValueError(f"unsupported process-flow node filter kind: {filter_kind}")
    return candidates


def condition_path_candidate(
    *,
    rng: Any,
    nodes: Sequence[Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Return one decision-path endpoint candidate over the sampled graph."""

    first_choice = int(rng.randrange(0, 2))
    second_choice = int(rng.randrange(0, 2))
    first_edge_ref = "e2" if first_choice == 0 else "e3"
    second_edge_ref = "e7" if second_choice == 0 else "e8"
    first_target = "n3" if first_choice == 0 else "n4"
    second_target = "n7" if second_choice == 0 else "n8"
    edge_by_ref = {str(edge["edge_id"]): edge for edge in edges}
    node_by_ref = {str(node["node_id"]): node for node in nodes}
    path_node_refs = ["n0", "n1", "n2", first_target, "n5", "n6", second_target]
    labels = [str(edge_by_ref[first_edge_ref]["label"]), str(edge_by_ref[second_edge_ref]["label"])]
    return {
        "answer": str(node_by_ref[second_target]["label"]),
        "answer_node_id": str(second_target),
        "annotation_roles": [
            {"key": "start_step", "kind": "node", "id": "n0"},
            {"key": "first_decision_label", "kind": "edge_label", "id": first_edge_ref},
            {"key": "second_decision_label", "kind": "edge_label", "id": second_edge_ref},
            {"key": "endpoint_step", "kind": "node", "id": second_target},
        ],
        "annotation_node_ids": [str(node_ref) for node_ref in path_node_refs],
        "annotation_edge_label_ids": [first_edge_ref, second_edge_ref],
        "condition_labels": labels,
        "condition_sequence_text": f"\"{labels[0]}\" then \"{labels[1]}\"",
        "start_label": str(node_by_ref["n0"]["label"]),
        "path_node_labels": [str(node_by_ref[node_ref]["label"]) for node_ref in path_node_refs],
        "answer_type": "string",
    }


def handoff_candidates(
    *,
    nodes: Sequence[Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
    handoff_scope: str,
) -> list[Dict[str, Any]]:
    """Return cross-lane arrow candidates for all-lane or named-lane scopes."""

    node_by_ref = {str(node["node_id"]): node for node in nodes}
    cross_edges = [
        edge
        for edge in edges
        if str(node_by_ref[str(edge["source"])]["lane"])
        != str(node_by_ref[str(edge["target"])]["lane"])
    ]
    if not cross_edges:
        raise ValueError("process-flow scene has no cross-lane handoff arrows")
    if str(handoff_scope) == "all":
        return [
            {
                "answer": int(len(cross_edges)),
                "annotation_edge_ids": [str(edge["edge_id"]) for edge in cross_edges],
                "answer_type": "integer",
            }
        ]

    candidates: list[Dict[str, Any]] = []
    lanes = sorted({str(node["lane"]) for node in nodes})
    for lane in lanes:
        if str(handoff_scope) == "lane_outgoing":
            selected_edges = [
                edge
                for edge in cross_edges
                if str(node_by_ref[str(edge["source"])]["lane"]) == str(lane)
            ]
        elif str(handoff_scope) == "lane_involved":
            selected_edges = [
                edge
                for edge in cross_edges
                if str(node_by_ref[str(edge["source"])]["lane"]) == str(lane)
                or str(node_by_ref[str(edge["target"])]["lane"]) == str(lane)
            ]
        else:
            raise ValueError(f"unsupported process-flow handoff scope: {handoff_scope}")
        if selected_edges:
            candidates.append(
                {
                    "lane_name": str(lane),
                    "answer": int(len(selected_edges)),
                    "annotation_edge_ids": [str(edge["edge_id"]) for edge in selected_edges],
                    "answer_type": "integer",
                }
            )
    return candidates
