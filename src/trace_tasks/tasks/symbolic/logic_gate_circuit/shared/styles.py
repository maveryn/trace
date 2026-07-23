"""Rendering style resolution for logic-gate circuit scenes."""

from __future__ import annotations

from typing import Any, Mapping

from ...shared.scene_style import SymbolicSceneStyle, make_symbolic_scene_background, resolve_symbolic_scene_style

from .state import LogicGateRenderParams


def resolve_render_params(defaults: Mapping[str, Any]) -> LogicGateRenderParams:
    """Resolve renderer dimensions from scene defaults."""

    return LogicGateRenderParams(
        canvas_width=int(defaults.get("logic_canvas_width", defaults.get("canvas_width", 1180))),
        canvas_height=int(defaults.get("logic_canvas_height", defaults.get("canvas_height", 820))),
        card_corner_radius_px=int(defaults.get("logic_card_corner_radius_px", defaults.get("panel_corner_radius_px", 18))),
        card_border_width_px=int(defaults.get("logic_card_border_width_px", defaults.get("panel_border_width_px", 2))),
        gate_width_px=int(defaults.get("logic_gate_width_px", 72)),
        gate_height_px=int(defaults.get("logic_gate_height_px", 46)),
        wire_width_px=int(defaults.get("logic_wire_width_px", 3)),
        node_radius_px=int(defaults.get("logic_node_radius_px", 5)),
        label_font_size_px=int(defaults.get("logic_label_font_size_px", defaults.get("label_font_size_px", 22))),
        small_font_size_px=int(defaults.get("logic_small_font_size_px", defaults.get("small_font_size_px", 16))),
        gate_font_size_px=int(defaults.get("logic_gate_font_size_px", 15)),
        table_font_size_px=int(defaults.get("logic_table_font_size_px", 20)),
    )


def resolve_background(*, instance_seed: int, canvas_width: int, canvas_height: int) -> tuple[object, dict[str, Any], SymbolicSceneStyle, dict[str, Any]]:
    """Resolve the symbolic scene style and background image."""

    scene_style, scene_style_meta = resolve_symbolic_scene_style(
        instance_seed=int(instance_seed),
        namespace="logic_gate_circuit.background",
    )
    background, background_meta = make_symbolic_scene_background(
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        style=scene_style,
    )
    return background, dict(background_meta), scene_style, dict(scene_style_meta)
