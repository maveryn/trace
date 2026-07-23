"""Graph-domain adapter for shared structured-information styling."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ...shared.visual_style.information_scene import (
    Color,
    InformationSceneStyle,
    resolve_information_scene_style,
)


GraphInformationStyle = InformationSceneStyle

NODE_LINK_INFORMATION_SCENE_TREATMENTS: tuple[str, ...] = (
    "clean_default",
    "report_card",
    "academic_figure",
    "journal_appendix",
    "print_scan_sheet",
)


def resolve_graph_information_style(
    *,
    instance_seed: int,
    params: Mapping[str, Any] | None,
    scene_id: str,
    protected_colors: Sequence[Color] | None = None,
    allow_dark: bool = False,
) -> tuple[GraphInformationStyle, dict[str, Any]]:
    """Resolve one graph presentation style without changing topology."""

    resolved_params = params or {}
    default_treatments: tuple[str, ...] | None = None
    if str(scene_id) == "node_link":
        default_treatments = NODE_LINK_INFORMATION_SCENE_TREATMENTS
    return resolve_information_scene_style(
        instance_seed=int(instance_seed),
        namespace=f"graph.{str(scene_id)}.information_scene_style",
        treatments=resolved_params.get("information_scene_treatments", default_treatments),
        treatment_weights=resolved_params.get("information_scene_treatment_weights", {}),
        palettes=resolved_params.get("information_scene_palettes"),
        palette_weights=resolved_params.get("information_scene_palette_weights", {}),
        chrome_modes=resolved_params.get("information_scene_chrome_modes"),
        chrome_mode_weights=resolved_params.get("information_scene_chrome_mode_weights", {}),
        allow_dark=bool(allow_dark),
        protected_colors=protected_colors or (),
    )


def infer_graph_scene_id(task_id: str) -> str:
    """Infer the public graph scene id for shared render styling."""

    text = str(task_id)
    if "adjacency" in text:
        return "adjacency"
    if "binary_tree" in text or "bst_" in text or "heap_property" in text:
        return "binary_tree"
    if "graph_options" in text:
        return "graph_options"
    if "metro" in text:
        return "metro"
    if "pipe" in text:
        return "pipe_network"
    if "automaton" in text:
        return "automaton"
    if "flow_network" in text or "max_flow" in text or "flow_value" in text or "min_cut" in text:
        return "flow_network"
    if "mst_weight" in text or "minimum_spanning_tree" in text:
        return "node_link"
    return "node_link"


def graph_surface_roles_from_information_style(style: GraphInformationStyle) -> dict[str, tuple[int, int, int]]:
    """Map shared roles into graph panel and neutral ink roles."""

    return {
        "background_color_rgb": tuple(int(value) for value in style.canvas_rgb),
        "panel_fill_rgb": tuple(int(value) for value in style.surface_rgb),
        "panel_border_rgb": tuple(int(value) for value in style.panel_border_rgb),
        "title_color_rgb": tuple(int(value) for value in style.text_rgb),
        "edge_color_rgb": tuple(int(value) for value in style.connector_rgb),
        "label_text_rgb": tuple(int(value) for value in style.text_rgb),
        "label_stroke_rgb": tuple(int(value) for value in style.text_stroke_rgb),
    }


__all__ = [
    "GraphInformationStyle",
    "NODE_LINK_INFORMATION_SCENE_TREATMENTS",
    "graph_surface_roles_from_information_style",
    "infer_graph_scene_id",
    "resolve_graph_information_style",
]
