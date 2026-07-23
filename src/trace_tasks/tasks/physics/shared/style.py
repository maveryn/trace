"""Shared physics-domain visual-theme helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, Tuple

from ...shared.named_colors import available_named_colors, darken_color, named_color


Color = Tuple[int, int, int]
SUPPORTED_PHYSICS_COLOR_NAMES: Tuple[str, ...] = tuple(str(name) for name, _ in available_named_colors())


@dataclass(frozen=True)
class PhysicsLeverTheme:
    """Resolved per-instance lever-balance theme derived from one named accent color."""

    accent_color_name: str
    beam_fill_rgb: Color
    beam_outline_rgb: Color
    beam_tick_rgb: Color
    distance_text_rgb: Color
    weight_fill_rgb: Color
    weight_outline_rgb: Color
    weight_text_rgb: Color
    fulcrum_fill_rgb: Color
    fulcrum_outline_rgb: Color
    texture_rgb: Color


@dataclass(frozen=True)
class PhysicsCircuitTheme:
    """Resolved per-instance resistor-network theme derived from one named accent color."""

    accent_color_name: str
    wire_rgb: Color
    resistor_fill_rgb: Color
    resistor_outline_rgb: Color
    resistor_text_rgb: Color
    missing_resistor_fill_rgb: Color
    missing_resistor_outline_rgb: Color
    missing_resistor_text_rgb: Color
    terminal_fill_rgb: Color
    terminal_outline_rgb: Color
    terminal_text_rgb: Color


@dataclass(frozen=True)
class PhysicsOpticsTheme:
    """Resolved per-instance optics theme derived from one named accent color."""

    accent_color_name: str
    board_grid_rgb: Color
    board_outline_rgb: Color
    mirror_rgb: Color
    target_fill_rgb: Color
    target_outline_rgb: Color
    target_text_rgb: Color
    source_fill_rgb: Color
    source_outline_rgb: Color
    source_text_rgb: Color
    ray_rgb: Color
    bounce_fill_rgb: Color
    bounce_outline_rgb: Color


@dataclass(frozen=True)
class PhysicsSpringTheme:
    """Resolved per-instance spring-extension theme derived from one named accent color."""

    accent_color_name: str
    card_fill_rgb: Color
    card_outline_rgb: Color
    support_fill_rgb: Color
    support_outline_rgb: Color
    ruler_rgb: Color
    ruler_text_rgb: Color
    spring_rgb: Color
    weight_fill_rgb: Color
    weight_outline_rgb: Color
    weight_text_rgb: Color
    marker_fill_rgb: Color
    marker_outline_rgb: Color
    missing_fill_rgb: Color
    missing_outline_rgb: Color
    missing_text_rgb: Color
    texture_rgb: Color


@dataclass(frozen=True)
class PhysicsPulleyTheme:
    """Resolved per-instance pulley-system theme derived from one named accent color."""

    accent_color_name: str
    frame_fill_rgb: Color
    frame_outline_rgb: Color
    rope_rgb: Color
    pulley_fill_rgb: Color
    pulley_outline_rgb: Color
    load_fill_rgb: Color
    load_outline_rgb: Color
    load_text_rgb: Color
    effort_rgb: Color
    missing_fill_rgb: Color
    missing_outline_rgb: Color
    missing_text_rgb: Color
    texture_rgb: Color


@dataclass(frozen=True)
class PhysicsCollisionTheme:
    """Resolved per-instance sticky-collision theme derived from one named accent color."""

    accent_color_name: str
    table_fill_rgb: Color
    table_outline_rgb: Color
    grid_rgb: Color
    puck_a_fill_rgb: Color
    puck_b_fill_rgb: Color
    puck_outline_rgb: Color
    puck_text_rgb: Color
    collision_fill_rgb: Color
    collision_outline_rgb: Color
    label_fill_rgb: Color
    label_outline_rgb: Color
    label_text_rgb: Color
    motion_arrow_rgb: Color
    final_arrow_rgb: Color
    option_arrow_rgb: Color
    option_outline_rgb: Color


@dataclass(frozen=True)
class PhysicsHydraulicTheme:
    """Resolved per-instance hydraulic-piston theme derived from one named accent color."""

    accent_color_name: str
    chamber_fill_rgb: Color
    chamber_outline_rgb: Color
    piston_fill_rgb: Color
    piston_outline_rgb: Color
    fluid_fill_rgb: Color
    fluid_outline_rgb: Color
    pipe_fill_rgb: Color
    pipe_outline_rgb: Color
    force_rgb: Color
    label_fill_rgb: Color
    label_outline_rgb: Color
    label_text_rgb: Color
    missing_fill_rgb: Color
    missing_outline_rgb: Color
    missing_text_rgb: Color
    texture_rgb: Color


@dataclass(frozen=True)
class PhysicsPVDiagramTheme:
    """Resolved per-instance pressure-volume diagram theme derived from one named accent color."""

    accent_color_name: str
    plot_fill_rgb: Color
    paper_plot_fill_rgb: Color
    plot_outline_rgb: Color
    grid_rgb: Color
    axis_rgb: Color
    axis_text_rgb: Color
    process_rgb: Color
    guide_rgb: Color
    work_fill_rgb: Color
    label_fill_rgb: Color
    label_outline_rgb: Color
    label_text_rgb: Color


@dataclass(frozen=True)
class PhysicsElectrostaticsTheme:
    """Resolved per-instance electrostatics field-map theme derived from one named accent color."""

    accent_color_name: str
    board_fill_rgb: Color
    board_alt_fill_rgb: Color
    board_outline_rgb: Color
    grid_rgb: Color
    axis_rgb: Color
    axis_text_rgb: Color
    positive_fill_rgb: Color
    positive_outline_rgb: Color
    positive_text_rgb: Color
    negative_fill_rgb: Color
    negative_outline_rgb: Color
    negative_text_rgb: Color
    point_fill_rgb: Color
    point_outline_rgb: Color
    point_text_rgb: Color
    target_fill_rgb: Color
    target_outline_rgb: Color
    option_arrow_rgb: Color
    option_outline_rgb: Color
    guide_rgb: Color
    label_fill_rgb: Color
    label_outline_rgb: Color
    label_text_rgb: Color


@dataclass(frozen=True)
class PhysicsMagnetismTheme:
    """Resolved per-instance magnetism field-force theme derived from one named accent color."""

    accent_color_name: str
    panel_fill_rgb: Color
    panel_alt_fill_rgb: Color
    panel_outline_rgb: Color
    grid_rgb: Color
    field_symbol_rgb: Color
    particle_fill_rgb: Color
    particle_outline_rgb: Color
    particle_text_rgb: Color
    velocity_arrow_rgb: Color
    force_arrow_rgb: Color
    path_rgb: Color
    option_arrow_rgb: Color
    option_outline_rgb: Color
    label_fill_rgb: Color
    label_outline_rgb: Color
    label_text_rgb: Color
    missing_fill_rgb: Color
    missing_outline_rgb: Color
    missing_text_rgb: Color


@dataclass(frozen=True)
class PhysicsWavesTheme:
    """Resolved per-instance wave-interference theme derived from one named accent color."""

    accent_color_name: str
    tank_fill_rgb: Color
    tank_alt_fill_rgb: Color
    tank_outline_rgb: Color
    grid_rgb: Color
    crest_rgb: Color
    trough_rgb: Color
    source_fill_rgb: Color
    source_outline_rgb: Color
    source_text_rgb: Color
    candidate_fill_rgb: Color
    candidate_outline_rgb: Color
    candidate_text_rgb: Color
    point_fill_rgb: Color
    point_outline_rgb: Color
    guide_rgb: Color
    label_fill_rgb: Color
    label_outline_rgb: Color
    label_text_rgb: Color


def _blend_with_white(color: Sequence[int], *, color_weight: float) -> Color:
    """Blend one RGB color toward white by the requested color weight."""

    weight = max(0.0, min(1.0, float(color_weight)))
    if len(color) < 3:
        raise ValueError("physics color blends require three RGB channels")
    return tuple(
        max(0, min(255, int(round((255.0 * (1.0 - weight)) + (float(int(channel)) * weight)))))
        for channel in color[:3]
    )


def _style_rgb(diagram_style: Any, role: str) -> Color:
    """Return one RGB role from a technical diagram style."""

    value = getattr(diagram_style, role)
    return tuple(int(channel) for channel in value[:3])


def build_physics_lever_theme(accent_color_name: str, *, diagram_style: Any | None = None) -> PhysicsLeverTheme:
    """Resolve one readable lever-balance theme from a named accent color."""

    if diagram_style is not None:
        return PhysicsLeverTheme(
            accent_color_name=str(accent_color_name),
            beam_fill_rgb=_style_rgb(diagram_style, "fill_rgb"),
            beam_outline_rgb=_style_rgb(diagram_style, "stroke_rgb"),
            beam_tick_rgb=_style_rgb(diagram_style, "axis_rgb"),
            distance_text_rgb=_style_rgb(diagram_style, "label_rgb"),
            weight_fill_rgb=_style_rgb(diagram_style, "muted_fill_rgb"),
            weight_outline_rgb=_style_rgb(diagram_style, "secondary_stroke_rgb"),
            weight_text_rgb=_style_rgb(diagram_style, "label_rgb"),
            fulcrum_fill_rgb=_style_rgb(diagram_style, "accent_rgb"),
            fulcrum_outline_rgb=_style_rgb(diagram_style, "stroke_rgb"),
            texture_rgb=_style_rgb(diagram_style, "canvas_accent_rgb"),
        )

    accent_rgb = tuple(int(channel) for channel in named_color(str(accent_color_name)))
    accent_dark_rgb = darken_color(accent_rgb, factor=0.60)
    accent_deep_rgb = darken_color(accent_rgb, factor=0.42)
    beam_fill_rgb = _blend_with_white(accent_rgb, color_weight=0.34)
    weight_fill_rgb = _blend_with_white(accent_rgb, color_weight=0.14)
    fulcrum_fill_rgb = _blend_with_white(accent_rgb, color_weight=0.78)
    texture_rgb = _blend_with_white(accent_dark_rgb, color_weight=0.42)
    neutral_text_rgb = (42, 46, 52)
    return PhysicsLeverTheme(
        accent_color_name=str(accent_color_name),
        beam_fill_rgb=tuple(int(channel) for channel in beam_fill_rgb),
        beam_outline_rgb=tuple(int(channel) for channel in accent_deep_rgb),
        beam_tick_rgb=tuple(int(channel) for channel in accent_dark_rgb),
        distance_text_rgb=tuple(int(channel) for channel in accent_deep_rgb),
        weight_fill_rgb=tuple(int(channel) for channel in weight_fill_rgb),
        weight_outline_rgb=tuple(int(channel) for channel in accent_dark_rgb),
        weight_text_rgb=tuple(int(channel) for channel in neutral_text_rgb),
        fulcrum_fill_rgb=tuple(int(channel) for channel in fulcrum_fill_rgb),
        fulcrum_outline_rgb=tuple(int(channel) for channel in accent_deep_rgb),
        texture_rgb=tuple(int(channel) for channel in texture_rgb),
    )


def build_physics_circuit_theme(accent_color_name: str, *, diagram_style: Any | None = None) -> PhysicsCircuitTheme:
    """Resolve one readable resistor-network theme from a named accent color."""

    if diagram_style is not None:
        return PhysicsCircuitTheme(
            accent_color_name=str(accent_color_name),
            wire_rgb=_style_rgb(diagram_style, "stroke_rgb"),
            resistor_fill_rgb=_style_rgb(diagram_style, "muted_fill_rgb"),
            resistor_outline_rgb=_style_rgb(diagram_style, "secondary_stroke_rgb"),
            resistor_text_rgb=_style_rgb(diagram_style, "label_rgb"),
            missing_resistor_fill_rgb=(255, 231, 231),
            missing_resistor_outline_rgb=(187, 56, 56),
            missing_resistor_text_rgb=(167, 38, 38),
            terminal_fill_rgb=_style_rgb(diagram_style, "accent_rgb"),
            terminal_outline_rgb=_style_rgb(diagram_style, "stroke_rgb"),
            terminal_text_rgb=_style_rgb(diagram_style, "label_rgb"),
        )

    accent_rgb = tuple(int(channel) for channel in named_color(str(accent_color_name)))
    accent_dark_rgb = darken_color(accent_rgb, factor=0.58)
    accent_deep_rgb = darken_color(accent_rgb, factor=0.40)
    resistor_fill_rgb = _blend_with_white(accent_rgb, color_weight=0.18)
    terminal_fill_rgb = _blend_with_white(accent_rgb, color_weight=0.72)
    missing_fill_rgb = (255, 231, 231)
    missing_outline_rgb = (187, 56, 56)
    missing_text_rgb = (167, 38, 38)
    return PhysicsCircuitTheme(
        accent_color_name=str(accent_color_name),
        wire_rgb=tuple(int(channel) for channel in accent_deep_rgb),
        resistor_fill_rgb=tuple(int(channel) for channel in resistor_fill_rgb),
        resistor_outline_rgb=tuple(int(channel) for channel in accent_dark_rgb),
        resistor_text_rgb=(39, 43, 49),
        missing_resistor_fill_rgb=tuple(int(channel) for channel in missing_fill_rgb),
        missing_resistor_outline_rgb=tuple(int(channel) for channel in missing_outline_rgb),
        missing_resistor_text_rgb=tuple(int(channel) for channel in missing_text_rgb),
        terminal_fill_rgb=tuple(int(channel) for channel in terminal_fill_rgb),
        terminal_outline_rgb=tuple(int(channel) for channel in accent_deep_rgb),
        terminal_text_rgb=(39, 43, 49),
    )


def build_physics_optics_theme(accent_color_name: str, *, diagram_style: Any | None = None) -> PhysicsOpticsTheme:
    """Resolve one readable optics theme from a named accent color."""

    if diagram_style is not None:
        return PhysicsOpticsTheme(
            accent_color_name=str(accent_color_name),
            board_grid_rgb=_style_rgb(diagram_style, "grid_minor_rgb"),
            board_outline_rgb=_style_rgb(diagram_style, "axis_rgb"),
            mirror_rgb=_style_rgb(diagram_style, "stroke_rgb"),
            target_fill_rgb=_style_rgb(diagram_style, "accent_rgb"),
            target_outline_rgb=_style_rgb(diagram_style, "stroke_rgb"),
            target_text_rgb=_style_rgb(diagram_style, "label_rgb"),
            source_fill_rgb=_style_rgb(diagram_style, "muted_fill_rgb"),
            source_outline_rgb=_style_rgb(diagram_style, "secondary_stroke_rgb"),
            source_text_rgb=_style_rgb(diagram_style, "label_rgb"),
            ray_rgb=_style_rgb(diagram_style, "secondary_accent_rgb"),
            bounce_fill_rgb=_style_rgb(diagram_style, "highlight_rgb"),
            bounce_outline_rgb=_style_rgb(diagram_style, "secondary_stroke_rgb"),
        )

    accent_rgb = tuple(int(channel) for channel in named_color(str(accent_color_name)))
    accent_dark_rgb = darken_color(accent_rgb, factor=0.60)
    accent_deep_rgb = darken_color(accent_rgb, factor=0.42)
    board_grid_rgb = _blend_with_white(accent_dark_rgb, color_weight=0.18)
    board_outline_rgb = _blend_with_white(accent_dark_rgb, color_weight=0.52)
    target_fill_rgb = tuple(int(channel) for channel in accent_dark_rgb)
    source_fill_rgb = _blend_with_white(accent_rgb, color_weight=0.74)
    return PhysicsOpticsTheme(
        accent_color_name=str(accent_color_name),
        board_grid_rgb=tuple(int(channel) for channel in board_grid_rgb),
        board_outline_rgb=tuple(int(channel) for channel in board_outline_rgb),
        mirror_rgb=tuple(int(channel) for channel in accent_deep_rgb),
        target_fill_rgb=tuple(int(channel) for channel in target_fill_rgb),
        target_outline_rgb=tuple(int(channel) for channel in accent_dark_rgb),
        target_text_rgb=(43, 47, 53),
        source_fill_rgb=tuple(int(channel) for channel in source_fill_rgb),
        source_outline_rgb=tuple(int(channel) for channel in accent_deep_rgb),
        source_text_rgb=(39, 43, 49),
        ray_rgb=(212, 92, 36),
        bounce_fill_rgb=(245, 205, 92),
        bounce_outline_rgb=(176, 116, 22),
    )


def build_physics_spring_theme(accent_color_name: str, *, diagram_style: Any | None = None) -> PhysicsSpringTheme:
    """Resolve one readable spring-extension theme from a named accent color."""

    if diagram_style is not None:
        return PhysicsSpringTheme(
            accent_color_name=str(accent_color_name),
            card_fill_rgb=_style_rgb(diagram_style, "panel_fill_rgb"),
            card_outline_rgb=_style_rgb(diagram_style, "panel_border_rgb"),
            support_fill_rgb=_style_rgb(diagram_style, "fill_rgb"),
            support_outline_rgb=_style_rgb(diagram_style, "stroke_rgb"),
            ruler_rgb=_style_rgb(diagram_style, "axis_rgb"),
            ruler_text_rgb=_style_rgb(diagram_style, "label_rgb"),
            spring_rgb=_style_rgb(diagram_style, "stroke_rgb"),
            weight_fill_rgb=_style_rgb(diagram_style, "muted_fill_rgb"),
            weight_outline_rgb=_style_rgb(diagram_style, "secondary_stroke_rgb"),
            weight_text_rgb=_style_rgb(diagram_style, "label_rgb"),
            marker_fill_rgb=_style_rgb(diagram_style, "accent_rgb"),
            marker_outline_rgb=_style_rgb(diagram_style, "stroke_rgb"),
            missing_fill_rgb=(255, 231, 231),
            missing_outline_rgb=(187, 56, 56),
            missing_text_rgb=(167, 38, 38),
            texture_rgb=_style_rgb(diagram_style, "canvas_accent_rgb"),
        )

    accent_rgb = tuple(int(channel) for channel in named_color(str(accent_color_name)))
    accent_dark_rgb = darken_color(accent_rgb, factor=0.60)
    accent_deep_rgb = darken_color(accent_rgb, factor=0.42)
    accent_mid_rgb = darken_color(accent_rgb, factor=0.74)
    return PhysicsSpringTheme(
        accent_color_name=str(accent_color_name),
        card_fill_rgb=_blend_with_white(accent_rgb, color_weight=0.08),
        card_outline_rgb=tuple(int(channel) for channel in accent_deep_rgb),
        support_fill_rgb=_blend_with_white(accent_rgb, color_weight=0.30),
        support_outline_rgb=tuple(int(channel) for channel in accent_deep_rgb),
        ruler_rgb=tuple(int(channel) for channel in accent_dark_rgb),
        ruler_text_rgb=(41, 45, 51),
        spring_rgb=tuple(int(channel) for channel in accent_deep_rgb),
        weight_fill_rgb=_blend_with_white(accent_rgb, color_weight=0.18),
        weight_outline_rgb=tuple(int(channel) for channel in accent_dark_rgb),
        weight_text_rgb=(39, 43, 49),
        marker_fill_rgb=_blend_with_white(accent_rgb, color_weight=0.52),
        marker_outline_rgb=tuple(int(channel) for channel in accent_mid_rgb),
        missing_fill_rgb=(255, 231, 231),
        missing_outline_rgb=(187, 56, 56),
        missing_text_rgb=(167, 38, 38),
        texture_rgb=_blend_with_white(accent_dark_rgb, color_weight=0.30),
    )


def build_physics_pulley_theme(accent_color_name: str, *, diagram_style: Any | None = None) -> PhysicsPulleyTheme:
    """Resolve one readable pulley-system theme from a named accent color."""

    if diagram_style is not None:
        return PhysicsPulleyTheme(
            accent_color_name=str(accent_color_name),
            frame_fill_rgb=_style_rgb(diagram_style, "fill_rgb"),
            frame_outline_rgb=_style_rgb(diagram_style, "stroke_rgb"),
            rope_rgb=_style_rgb(diagram_style, "secondary_stroke_rgb"),
            pulley_fill_rgb=_style_rgb(diagram_style, "muted_fill_rgb"),
            pulley_outline_rgb=_style_rgb(diagram_style, "stroke_rgb"),
            load_fill_rgb=_style_rgb(diagram_style, "panel_fill_rgb"),
            load_outline_rgb=_style_rgb(diagram_style, "panel_border_rgb"),
            load_text_rgb=_style_rgb(diagram_style, "label_rgb"),
            effort_rgb=_style_rgb(diagram_style, "secondary_accent_rgb"),
            missing_fill_rgb=(255, 231, 231),
            missing_outline_rgb=(187, 56, 56),
            missing_text_rgb=(167, 38, 38),
            texture_rgb=_style_rgb(diagram_style, "canvas_accent_rgb"),
        )

    accent_rgb = tuple(int(channel) for channel in named_color(str(accent_color_name)))
    accent_dark_rgb = darken_color(accent_rgb, factor=0.60)
    accent_deep_rgb = darken_color(accent_rgb, factor=0.42)
    return PhysicsPulleyTheme(
        accent_color_name=str(accent_color_name),
        frame_fill_rgb=_blend_with_white(accent_rgb, color_weight=0.12),
        frame_outline_rgb=tuple(int(channel) for channel in accent_deep_rgb),
        rope_rgb=tuple(int(channel) for channel in accent_deep_rgb),
        pulley_fill_rgb=_blend_with_white(accent_rgb, color_weight=0.20),
        pulley_outline_rgb=tuple(int(channel) for channel in accent_dark_rgb),
        load_fill_rgb=_blend_with_white(accent_rgb, color_weight=0.16),
        load_outline_rgb=tuple(int(channel) for channel in accent_dark_rgb),
        load_text_rgb=(39, 43, 49),
        effort_rgb=(212, 92, 36),
        missing_fill_rgb=(255, 231, 231),
        missing_outline_rgb=(187, 56, 56),
        missing_text_rgb=(167, 38, 38),
        texture_rgb=_blend_with_white(accent_dark_rgb, color_weight=0.28),
    )


def build_physics_collision_theme(accent_color_name: str, *, diagram_style: Any | None = None) -> PhysicsCollisionTheme:
    """Resolve one readable sticky-collision theme from a named accent color."""

    if diagram_style is not None:
        return PhysicsCollisionTheme(
            accent_color_name=str(accent_color_name),
            table_fill_rgb=_style_rgb(diagram_style, "panel_fill_rgb"),
            table_outline_rgb=_style_rgb(diagram_style, "panel_border_rgb"),
            grid_rgb=_style_rgb(diagram_style, "grid_minor_rgb"),
            puck_a_fill_rgb=_style_rgb(diagram_style, "fill_rgb"),
            puck_b_fill_rgb=_style_rgb(diagram_style, "highlight_rgb"),
            puck_outline_rgb=_style_rgb(diagram_style, "stroke_rgb"),
            puck_text_rgb=_style_rgb(diagram_style, "label_rgb"),
            collision_fill_rgb=_style_rgb(diagram_style, "muted_fill_rgb"),
            collision_outline_rgb=_style_rgb(diagram_style, "secondary_stroke_rgb"),
            label_fill_rgb=_style_rgb(diagram_style, "label_fill_rgb"),
            label_outline_rgb=_style_rgb(diagram_style, "label_border_rgb"),
            label_text_rgb=_style_rgb(diagram_style, "label_rgb"),
            motion_arrow_rgb=_style_rgb(diagram_style, "secondary_accent_rgb"),
            final_arrow_rgb=_style_rgb(diagram_style, "stroke_rgb"),
            option_arrow_rgb=_style_rgb(diagram_style, "secondary_stroke_rgb"),
            option_outline_rgb=_style_rgb(diagram_style, "panel_border_rgb"),
        )

    accent_rgb = tuple(int(channel) for channel in named_color(str(accent_color_name)))
    accent_dark_rgb = darken_color(accent_rgb, factor=0.60)
    accent_deep_rgb = darken_color(accent_rgb, factor=0.42)
    return PhysicsCollisionTheme(
        accent_color_name=str(accent_color_name),
        table_fill_rgb=(255, 255, 255),
        table_outline_rgb=tuple(int(channel) for channel in accent_deep_rgb),
        grid_rgb=_blend_with_white(accent_dark_rgb, color_weight=0.16),
        puck_a_fill_rgb=_blend_with_white(accent_rgb, color_weight=0.28),
        puck_b_fill_rgb=(250, 226, 157),
        puck_outline_rgb=tuple(int(channel) for channel in accent_dark_rgb),
        puck_text_rgb=(38, 42, 48),
        collision_fill_rgb=_blend_with_white(accent_rgb, color_weight=0.14),
        collision_outline_rgb=tuple(int(channel) for channel in accent_deep_rgb),
        label_fill_rgb=(255, 255, 255),
        label_outline_rgb=tuple(int(channel) for channel in accent_dark_rgb),
        label_text_rgb=(39, 43, 49),
        motion_arrow_rgb=(207, 82, 40),
        final_arrow_rgb=tuple(int(channel) for channel in accent_deep_rgb),
        option_arrow_rgb=(58, 71, 88),
        option_outline_rgb=tuple(int(channel) for channel in accent_dark_rgb),
    )


def build_physics_hydraulic_theme(accent_color_name: str, *, diagram_style: Any | None = None) -> PhysicsHydraulicTheme:
    """Resolve one readable hydraulic-piston theme from a named accent color."""

    if diagram_style is not None:
        return PhysicsHydraulicTheme(
            accent_color_name=str(accent_color_name),
            chamber_fill_rgb=_style_rgb(diagram_style, "panel_fill_rgb"),
            chamber_outline_rgb=_style_rgb(diagram_style, "panel_border_rgb"),
            piston_fill_rgb=_style_rgb(diagram_style, "fill_rgb"),
            piston_outline_rgb=_style_rgb(diagram_style, "stroke_rgb"),
            fluid_fill_rgb=_style_rgb(diagram_style, "muted_fill_rgb"),
            fluid_outline_rgb=_style_rgb(diagram_style, "secondary_stroke_rgb"),
            pipe_fill_rgb=_style_rgb(diagram_style, "panel_alt_fill_rgb"),
            pipe_outline_rgb=_style_rgb(diagram_style, "stroke_rgb"),
            force_rgb=_style_rgb(diagram_style, "secondary_accent_rgb"),
            label_fill_rgb=_style_rgb(diagram_style, "label_fill_rgb"),
            label_outline_rgb=_style_rgb(diagram_style, "label_border_rgb"),
            label_text_rgb=_style_rgb(diagram_style, "label_rgb"),
            missing_fill_rgb=(255, 231, 231),
            missing_outline_rgb=(187, 56, 56),
            missing_text_rgb=(167, 38, 38),
            texture_rgb=_style_rgb(diagram_style, "canvas_accent_rgb"),
        )

    accent_rgb = tuple(int(channel) for channel in named_color(str(accent_color_name)))
    accent_dark_rgb = darken_color(accent_rgb, factor=0.60)
    accent_deep_rgb = darken_color(accent_rgb, factor=0.42)
    return PhysicsHydraulicTheme(
        accent_color_name=str(accent_color_name),
        chamber_fill_rgb=_blend_with_white(accent_rgb, color_weight=0.06),
        chamber_outline_rgb=tuple(int(channel) for channel in accent_deep_rgb),
        piston_fill_rgb=_blend_with_white(accent_rgb, color_weight=0.30),
        piston_outline_rgb=tuple(int(channel) for channel in accent_deep_rgb),
        fluid_fill_rgb=_blend_with_white(accent_rgb, color_weight=0.44),
        fluid_outline_rgb=tuple(int(channel) for channel in accent_dark_rgb),
        pipe_fill_rgb=_blend_with_white(accent_rgb, color_weight=0.36),
        pipe_outline_rgb=tuple(int(channel) for channel in accent_deep_rgb),
        force_rgb=(212, 92, 36),
        label_fill_rgb=(255, 255, 255),
        label_outline_rgb=tuple(int(channel) for channel in accent_dark_rgb),
        label_text_rgb=(39, 43, 49),
        missing_fill_rgb=(255, 231, 231),
        missing_outline_rgb=(187, 56, 56),
        missing_text_rgb=(167, 38, 38),
        texture_rgb=_blend_with_white(accent_dark_rgb, color_weight=0.26),
    )


def build_physics_pv_diagram_theme(accent_color_name: str, *, diagram_style: Any | None = None) -> PhysicsPVDiagramTheme:
    """Resolve one readable pressure-volume diagram theme from a named accent color."""

    if diagram_style is not None:
        return PhysicsPVDiagramTheme(
            accent_color_name=str(accent_color_name),
            plot_fill_rgb=_style_rgb(diagram_style, "panel_fill_rgb"),
            paper_plot_fill_rgb=_style_rgb(diagram_style, "panel_alt_fill_rgb"),
            plot_outline_rgb=_style_rgb(diagram_style, "panel_border_rgb"),
            grid_rgb=_style_rgb(diagram_style, "grid_minor_rgb"),
            axis_rgb=_style_rgb(diagram_style, "axis_rgb"),
            axis_text_rgb=_style_rgb(diagram_style, "label_rgb"),
            process_rgb=_style_rgb(diagram_style, "secondary_accent_rgb"),
            guide_rgb=_style_rgb(diagram_style, "guide_rgb"),
            work_fill_rgb=_style_rgb(diagram_style, "muted_fill_rgb"),
            label_fill_rgb=_style_rgb(diagram_style, "label_fill_rgb"),
            label_outline_rgb=_style_rgb(diagram_style, "label_border_rgb"),
            label_text_rgb=_style_rgb(diagram_style, "label_rgb"),
        )

    accent_rgb = tuple(int(channel) for channel in named_color(str(accent_color_name)))
    accent_dark_rgb = darken_color(accent_rgb, factor=0.60)
    accent_deep_rgb = darken_color(accent_rgb, factor=0.42)
    return PhysicsPVDiagramTheme(
        accent_color_name=str(accent_color_name),
        plot_fill_rgb=(255, 255, 255),
        paper_plot_fill_rgb=(255, 252, 242),
        plot_outline_rgb=tuple(int(channel) for channel in accent_deep_rgb),
        grid_rgb=_blend_with_white(accent_dark_rgb, color_weight=0.15),
        axis_rgb=tuple(int(channel) for channel in accent_deep_rgb),
        axis_text_rgb=(39, 43, 49),
        process_rgb=(207, 82, 40),
        guide_rgb=_blend_with_white(accent_dark_rgb, color_weight=0.30),
        work_fill_rgb=_blend_with_white(accent_rgb, color_weight=0.18),
        label_fill_rgb=(255, 255, 255),
        label_outline_rgb=tuple(int(channel) for channel in accent_dark_rgb),
        label_text_rgb=(39, 43, 49),
    )


def build_physics_electrostatics_theme(accent_color_name: str, *, diagram_style: Any | None = None) -> PhysicsElectrostaticsTheme:
    """Resolve one readable electrostatics field-map theme from a named accent color."""

    if diagram_style is not None:
        return PhysicsElectrostaticsTheme(
            accent_color_name=str(accent_color_name),
            board_fill_rgb=_style_rgb(diagram_style, "panel_fill_rgb"),
            board_alt_fill_rgb=_style_rgb(diagram_style, "panel_alt_fill_rgb"),
            board_outline_rgb=_style_rgb(diagram_style, "panel_border_rgb"),
            grid_rgb=_style_rgb(diagram_style, "grid_minor_rgb"),
            axis_rgb=_style_rgb(diagram_style, "axis_rgb"),
            axis_text_rgb=_style_rgb(diagram_style, "label_rgb"),
            positive_fill_rgb=(255, 226, 221),
            positive_outline_rgb=(187, 58, 46),
            positive_text_rgb=(142, 40, 32),
            negative_fill_rgb=(222, 237, 255),
            negative_outline_rgb=(55, 91, 172),
            negative_text_rgb=(38, 66, 132),
            point_fill_rgb=_style_rgb(diagram_style, "muted_fill_rgb"),
            point_outline_rgb=_style_rgb(diagram_style, "secondary_stroke_rgb"),
            point_text_rgb=_style_rgb(diagram_style, "label_rgb"),
            target_fill_rgb=_style_rgb(diagram_style, "accent_rgb"),
            target_outline_rgb=_style_rgb(diagram_style, "stroke_rgb"),
            option_arrow_rgb=_style_rgb(diagram_style, "stroke_rgb"),
            option_outline_rgb=_style_rgb(diagram_style, "panel_border_rgb"),
            guide_rgb=_style_rgb(diagram_style, "guide_rgb"),
            label_fill_rgb=_style_rgb(diagram_style, "label_fill_rgb"),
            label_outline_rgb=_style_rgb(diagram_style, "label_border_rgb"),
            label_text_rgb=_style_rgb(diagram_style, "label_rgb"),
        )

    accent_rgb = tuple(int(channel) for channel in named_color(str(accent_color_name)))
    accent_dark_rgb = darken_color(accent_rgb, factor=0.60)
    accent_deep_rgb = darken_color(accent_rgb, factor=0.42)
    return PhysicsElectrostaticsTheme(
        accent_color_name=str(accent_color_name),
        board_fill_rgb=(255, 255, 255),
        board_alt_fill_rgb=(255, 252, 242),
        board_outline_rgb=tuple(int(channel) for channel in accent_deep_rgb),
        grid_rgb=_blend_with_white(accent_dark_rgb, color_weight=0.16),
        axis_rgb=tuple(int(channel) for channel in accent_deep_rgb),
        axis_text_rgb=(39, 43, 49),
        positive_fill_rgb=(255, 226, 221),
        positive_outline_rgb=(187, 58, 46),
        positive_text_rgb=(142, 40, 32),
        negative_fill_rgb=(222, 237, 255),
        negative_outline_rgb=(55, 91, 172),
        negative_text_rgb=(38, 66, 132),
        point_fill_rgb=_blend_with_white(accent_rgb, color_weight=0.26),
        point_outline_rgb=tuple(int(channel) for channel in accent_dark_rgb),
        point_text_rgb=(39, 43, 49),
        target_fill_rgb=_blend_with_white(accent_rgb, color_weight=0.48),
        target_outline_rgb=tuple(int(channel) for channel in accent_deep_rgb),
        option_arrow_rgb=(58, 71, 88),
        option_outline_rgb=tuple(int(channel) for channel in accent_dark_rgb),
        guide_rgb=_blend_with_white(accent_dark_rgb, color_weight=0.32),
        label_fill_rgb=(255, 255, 255),
        label_outline_rgb=tuple(int(channel) for channel in accent_dark_rgb),
        label_text_rgb=(39, 43, 49),
    )


def build_physics_magnetism_theme(accent_color_name: str, *, diagram_style: Any | None = None) -> PhysicsMagnetismTheme:
    """Resolve one readable magnetism field-force theme from a named accent color."""

    if diagram_style is not None:
        return PhysicsMagnetismTheme(
            accent_color_name=str(accent_color_name),
            panel_fill_rgb=_style_rgb(diagram_style, "panel_fill_rgb"),
            panel_alt_fill_rgb=_style_rgb(diagram_style, "panel_alt_fill_rgb"),
            panel_outline_rgb=_style_rgb(diagram_style, "panel_border_rgb"),
            grid_rgb=_style_rgb(diagram_style, "grid_minor_rgb"),
            field_symbol_rgb=_style_rgb(diagram_style, "secondary_stroke_rgb"),
            particle_fill_rgb=_style_rgb(diagram_style, "muted_fill_rgb"),
            particle_outline_rgb=_style_rgb(diagram_style, "stroke_rgb"),
            particle_text_rgb=_style_rgb(diagram_style, "label_rgb"),
            velocity_arrow_rgb=_style_rgb(diagram_style, "secondary_accent_rgb"),
            force_arrow_rgb=_style_rgb(diagram_style, "stroke_rgb"),
            path_rgb=_style_rgb(diagram_style, "guide_rgb"),
            option_arrow_rgb=_style_rgb(diagram_style, "secondary_stroke_rgb"),
            option_outline_rgb=_style_rgb(diagram_style, "panel_border_rgb"),
            label_fill_rgb=_style_rgb(diagram_style, "label_fill_rgb"),
            label_outline_rgb=_style_rgb(diagram_style, "label_border_rgb"),
            label_text_rgb=_style_rgb(diagram_style, "label_rgb"),
            missing_fill_rgb=(255, 231, 231),
            missing_outline_rgb=(187, 56, 56),
            missing_text_rgb=(167, 38, 38),
        )

    accent_rgb = tuple(int(channel) for channel in named_color(str(accent_color_name)))
    accent_dark_rgb = darken_color(accent_rgb, factor=0.60)
    accent_deep_rgb = darken_color(accent_rgb, factor=0.42)
    return PhysicsMagnetismTheme(
        accent_color_name=str(accent_color_name),
        panel_fill_rgb=(255, 255, 255),
        panel_alt_fill_rgb=(255, 252, 242),
        panel_outline_rgb=tuple(int(channel) for channel in accent_deep_rgb),
        grid_rgb=_blend_with_white(accent_dark_rgb, color_weight=0.14),
        field_symbol_rgb=_blend_with_white(accent_deep_rgb, color_weight=0.70),
        particle_fill_rgb=_blend_with_white(accent_rgb, color_weight=0.30),
        particle_outline_rgb=tuple(int(channel) for channel in accent_deep_rgb),
        particle_text_rgb=(39, 43, 49),
        velocity_arrow_rgb=(207, 82, 40),
        force_arrow_rgb=tuple(int(channel) for channel in accent_deep_rgb),
        path_rgb=(70, 92, 122),
        option_arrow_rgb=(58, 71, 88),
        option_outline_rgb=tuple(int(channel) for channel in accent_dark_rgb),
        label_fill_rgb=(255, 255, 255),
        label_outline_rgb=tuple(int(channel) for channel in accent_dark_rgb),
        label_text_rgb=(39, 43, 49),
        missing_fill_rgb=(255, 231, 231),
        missing_outline_rgb=(187, 56, 56),
        missing_text_rgb=(167, 38, 38),
    )


def build_physics_waves_theme(accent_color_name: str, *, diagram_style: Any | None = None) -> PhysicsWavesTheme:
    """Resolve one readable wave-interference theme from a named accent color."""

    if diagram_style is not None:
        return PhysicsWavesTheme(
            accent_color_name=str(accent_color_name),
            tank_fill_rgb=_style_rgb(diagram_style, "panel_fill_rgb"),
            tank_alt_fill_rgb=_style_rgb(diagram_style, "panel_alt_fill_rgb"),
            tank_outline_rgb=_style_rgb(diagram_style, "panel_border_rgb"),
            grid_rgb=_style_rgb(diagram_style, "grid_minor_rgb"),
            crest_rgb=_style_rgb(diagram_style, "stroke_rgb"),
            trough_rgb=_style_rgb(diagram_style, "secondary_stroke_rgb"),
            source_fill_rgb=_style_rgb(diagram_style, "accent_rgb"),
            source_outline_rgb=_style_rgb(diagram_style, "stroke_rgb"),
            source_text_rgb=_style_rgb(diagram_style, "label_rgb"),
            candidate_fill_rgb=_style_rgb(diagram_style, "label_fill_rgb"),
            candidate_outline_rgb=_style_rgb(diagram_style, "label_border_rgb"),
            candidate_text_rgb=_style_rgb(diagram_style, "label_rgb"),
            point_fill_rgb=_style_rgb(diagram_style, "highlight_rgb"),
            point_outline_rgb=_style_rgb(diagram_style, "secondary_stroke_rgb"),
            guide_rgb=_style_rgb(diagram_style, "guide_rgb"),
            label_fill_rgb=_style_rgb(diagram_style, "label_fill_rgb"),
            label_outline_rgb=_style_rgb(diagram_style, "label_border_rgb"),
            label_text_rgb=_style_rgb(diagram_style, "label_rgb"),
        )

    accent_rgb = tuple(int(channel) for channel in named_color(str(accent_color_name)))
    accent_dark_rgb = darken_color(accent_rgb, factor=0.60)
    accent_deep_rgb = darken_color(accent_rgb, factor=0.42)
    return PhysicsWavesTheme(
        accent_color_name=str(accent_color_name),
        tank_fill_rgb=(255, 255, 255),
        tank_alt_fill_rgb=(246, 252, 255),
        tank_outline_rgb=tuple(int(channel) for channel in accent_deep_rgb),
        grid_rgb=_blend_with_white(accent_dark_rgb, color_weight=0.14),
        crest_rgb=tuple(int(channel) for channel in accent_deep_rgb),
        trough_rgb=_blend_with_white(accent_deep_rgb, color_weight=0.46),
        source_fill_rgb=_blend_with_white(accent_rgb, color_weight=0.36),
        source_outline_rgb=tuple(int(channel) for channel in accent_deep_rgb),
        source_text_rgb=(39, 43, 49),
        candidate_fill_rgb=(255, 255, 255),
        candidate_outline_rgb=tuple(int(channel) for channel in accent_dark_rgb),
        candidate_text_rgb=(39, 43, 49),
        point_fill_rgb=(250, 226, 157),
        point_outline_rgb=(176, 116, 22),
        guide_rgb=(207, 82, 40),
        label_fill_rgb=(255, 255, 255),
        label_outline_rgb=tuple(int(channel) for channel in accent_dark_rgb),
        label_text_rgb=(39, 43, 49),
    )


__all__ = [
    "PhysicsCircuitTheme",
    "PhysicsCollisionTheme",
    "PhysicsElectrostaticsTheme",
    "PhysicsHydraulicTheme",
    "PhysicsLeverTheme",
    "PhysicsMagnetismTheme",
    "PhysicsOpticsTheme",
    "PhysicsPVDiagramTheme",
    "PhysicsPulleyTheme",
    "PhysicsSpringTheme",
    "PhysicsWavesTheme",
    "SUPPORTED_PHYSICS_COLOR_NAMES",
    "build_physics_circuit_theme",
    "build_physics_collision_theme",
    "build_physics_electrostatics_theme",
    "build_physics_hydraulic_theme",
    "build_physics_lever_theme",
    "build_physics_magnetism_theme",
    "build_physics_optics_theme",
    "build_physics_pv_diagram_theme",
    "build_physics_pulley_theme",
    "build_physics_spring_theme",
    "build_physics_waves_theme",
]
