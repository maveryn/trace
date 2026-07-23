"""Tests for shared technical-diagram visual styling."""

from __future__ import annotations

from trace_tasks.tasks.physics.shared.diagram_style import (
    PHYSICS_ELECTROSTATICS_SEMANTIC_COLORS,
    physics_electrostatics_theme_from_diagram_style,
    resolve_physics_diagram_style,
)
from trace_tasks.tasks.physics.analog_meter.meter_readout_value import PhysicsAnalogMeterReadoutValueTask
from trace_tasks.tasks.physics.motion_graph.interval_displacement_value import PhysicsMotionGraphIntervalDisplacementValueTask
from trace_tasks.tasks.physics.pv_diagram.pv_work_value import PhysicsPVDiagramWorkValueTask
from trace_tasks.tasks.geometry.shared.diagram_style import resolve_geometry_diagram_style
from trace_tasks.tasks.shared.text_legibility import READ_REQUIRED_TEXT_MIN_CONTRAST_RATIO
from trace_tasks.tasks.shared.visual_style.technical_diagram import (
    TECHNICAL_DIAGRAM_PROFILE_ANALYTICAL,
    TECHNICAL_DIAGRAM_PROFILE_GRAPH_PAPER,
    TECHNICAL_DIAGRAM_PROFILE_THEME_IDS,
    TECHNICAL_DIAGRAM_THEMES,
    TECHNICAL_DIAGRAM_PALETTES,
    TECHNICAL_DIAGRAM_TREATMENTS,
    make_technical_diagram_background,
    resolve_technical_diagram_style,
)


def _relative_luminance(rgb: tuple[int, int, int]) -> float:
    def channel(value: int) -> float:
        normalized = max(0.0, min(1.0, float(value) / 255.0))
        if normalized <= 0.03928:
            return normalized / 12.92
        return ((normalized + 0.055) / 1.055) ** 2.4

    return (0.2126 * channel(rgb[0])) + (0.7152 * channel(rgb[1])) + (0.0722 * channel(rgb[2]))


def _contrast_ratio(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
    lum_a = _relative_luminance(a)
    lum_b = _relative_luminance(b)
    lighter = max(lum_a, lum_b)
    darker = min(lum_a, lum_b)
    return (lighter + 0.05) / (darker + 0.05)


def test_technical_diagram_registry_breadth_and_metadata() -> None:
    style, metadata = resolve_technical_diagram_style(
        instance_seed=12001,
        namespace="tests.technical_diagram",
        theme_profile=TECHNICAL_DIAGRAM_PROFILE_ANALYTICAL,
        allow_dark=True,
        protected_colors=PHYSICS_ELECTROSTATICS_SEMANTIC_COLORS,
    )

    assert len(TECHNICAL_DIAGRAM_TREATMENTS) >= 20
    assert len(TECHNICAL_DIAGRAM_PALETTES) >= 20
    assert len(TECHNICAL_DIAGRAM_PROFILE_THEME_IDS[TECHNICAL_DIAGRAM_PROFILE_ANALYTICAL]) == 25
    assert len(TECHNICAL_DIAGRAM_PROFILE_THEME_IDS[TECHNICAL_DIAGRAM_PROFILE_GRAPH_PAPER]) == 25
    for profile in (TECHNICAL_DIAGRAM_PROFILE_ANALYTICAL, TECHNICAL_DIAGRAM_PROFILE_GRAPH_PAPER):
        themes = [TECHNICAL_DIAGRAM_THEMES[theme_id] for theme_id in TECHNICAL_DIAGRAM_PROFILE_THEME_IDS[profile]]
        assert sum("light" in theme.compatibility for theme in themes) == 20
        assert sum("dark" in theme.compatibility for theme in themes) == 5
    assert all(
        TECHNICAL_DIAGRAM_TREATMENTS[TECHNICAL_DIAGRAM_THEMES[theme_id].treatment_id].grid_kind == "none"
        for theme_id in TECHNICAL_DIAGRAM_PROFILE_THEME_IDS[TECHNICAL_DIAGRAM_PROFILE_ANALYTICAL]
    )
    assert all(
        TECHNICAL_DIAGRAM_TREATMENTS[TECHNICAL_DIAGRAM_THEMES[theme_id].treatment_id].grid_kind in {"square", "lab_grid"}
        for theme_id in TECHNICAL_DIAGRAM_PROFILE_THEME_IDS[TECHNICAL_DIAGRAM_PROFILE_GRAPH_PAPER]
    )
    assert metadata["kind"] == "technical_diagram_style"
    assert metadata["technical_profile"] == TECHNICAL_DIAGRAM_PROFILE_ANALYTICAL
    assert metadata["theme_id"] in TECHNICAL_DIAGRAM_PROFILE_THEME_IDS[TECHNICAL_DIAGRAM_PROFILE_ANALYTICAL]
    assert metadata["treatment"] == style.treatment
    assert metadata["palette_id"] == style.palette_id
    assert metadata["roles_rgb"]["axis"] == list(style.axis_rgb)
    assert metadata["grid_style"]["kind"] == style.grid_kind
    assert metadata["frame_style"]["mode"] == style.frame_mode
    assert set(metadata["frame_style"]["default_mode_weights"]) == {"none", "plain_outline", "matching_outline"}
    assert metadata["stroke_widths"]["axis_px"] == style.axis_stroke_width_px
    assert metadata["protected_colors_rgb"]
    assert metadata["contrast_checks"]["min_paper_ink_lab_distance"] > 0


def test_technical_diagram_background_is_deterministic() -> None:
    style, _ = resolve_technical_diagram_style(
        instance_seed=12002,
        namespace="tests.technical_diagram.background",
        treatments=("subtle_scan_sheet",),
        palettes=("neutral_ink",),
        frame_modes=("matching_outline",),
    )
    image_a, meta_a = make_technical_diagram_background(
        canvas_width=320,
        canvas_height=220,
        style=style,
        instance_seed=12002,
        namespace="tests.technical_diagram.background",
    )
    image_b, meta_b = make_technical_diagram_background(
        canvas_width=320,
        canvas_height=220,
        style=style,
        instance_seed=12002,
        namespace="tests.technical_diagram.background",
    )

    assert image_a.size == (320, 220)
    assert meta_a == meta_b
    assert image_a.tobytes() == image_b.tobytes()
    assert meta_a["style_spec"]["kind"] == "technical_diagram_style"
    assert meta_a["style_spec"]["frame_mode"] == "matching_outline"


def test_geometry_adapter_strengthens_label_contrast_on_light_panels() -> None:
    style, metadata = resolve_geometry_diagram_style(
        instance_seed=12004,
        params={
            "technical_diagram_treatments": ("exam_problem_box",),
            "technical_diagram_palettes": ("graphite_blue",),
        },
        scene_id="bearing_route",
        allow_dark=False,
    )

    anchors = (
        style.canvas_rgb,
        style.paper_rgb,
        style.panel_fill_rgb,
        style.panel_alt_fill_rgb,
        style.option_fill_rgb,
    )
    assert style.label_stroke_rgb == style.label_rgb
    assert min(_contrast_ratio(style.label_rgb, anchor) for anchor in anchors) >= READ_REQUIRED_TEXT_MIN_CONTRAST_RATIO
    assert metadata["geometry_label_contrast_guard"]["enabled"] is True
    assert metadata["geometry_label_contrast_guard"]["shared_text_legibility"]["passes"] is True
    assert metadata["roles_rgb"]["label"] == list(style.label_rgb)
    assert metadata["roles_rgb"]["label_stroke"] == list(style.label_stroke_rgb)
    assert metadata["technical_profile"] == TECHNICAL_DIAGRAM_PROFILE_ANALYTICAL
    assert metadata["theme_id"] in TECHNICAL_DIAGRAM_PROFILE_THEME_IDS[TECHNICAL_DIAGRAM_PROFILE_ANALYTICAL]


def test_geometry_graph_paper_profile_uses_coordinate_safe_themes() -> None:
    style, metadata = resolve_geometry_diagram_style(
        instance_seed=12005,
        params={},
        scene_id="graph_paper",
    )

    assert metadata["geometry_style_profile"] == "coordinate_grid"
    assert metadata["technical_profile"] == TECHNICAL_DIAGRAM_PROFILE_GRAPH_PAPER
    assert metadata["theme_id"] in TECHNICAL_DIAGRAM_PROFILE_THEME_IDS[TECHNICAL_DIAGRAM_PROFILE_GRAPH_PAPER]
    assert style.grid_kind in {"square", "lab_grid"}


def test_physics_diagram_adapter_preserves_electrostatics_semantic_colors() -> None:
    style, metadata = resolve_physics_diagram_style(
        instance_seed=12003,
        params={},
        scene_id="electrostatics_field_map",
        protected_colors=PHYSICS_ELECTROSTATICS_SEMANTIC_COLORS,
    )
    theme = physics_electrostatics_theme_from_diagram_style(style, accent_color_name="blue")

    assert theme.positive_outline_rgb == (187, 58, 46)
    assert theme.negative_outline_rgb == (55, 91, 172)
    assert theme.axis_rgb == style.axis_rgb
    assert theme.grid_rgb == style.grid_minor_rgb
    assert metadata["selection"]["allow_dark"] is True
    assert metadata["technical_profile"] == TECHNICAL_DIAGRAM_PROFILE_ANALYTICAL


def test_physics_default_profile_keeps_analytical_themes_when_guides_are_requested() -> None:
    style, metadata = resolve_physics_diagram_style(
        instance_seed=12006,
        params={},
        scene_id="wire_magnetism",
        require_grid=True,
    )

    assert metadata["technical_profile"] == TECHNICAL_DIAGRAM_PROFILE_ANALYTICAL
    assert metadata["theme_id"] in TECHNICAL_DIAGRAM_PROFILE_THEME_IDS[TECHNICAL_DIAGRAM_PROFILE_ANALYTICAL]
    assert len(metadata["selection"]["available_theme_ids"]) == 25
    assert metadata["selection"]["allow_dark"] is True
    assert metadata["selection"]["require_grid"] is False
    assert style.grid_kind == "none"


def test_physics_graph_paper_profile_is_explicit_opt_in() -> None:
    style, metadata = resolve_physics_diagram_style(
        instance_seed=12007,
        params={},
        scene_id="motion_graph",
        style_profile="graph_paper",
    )

    assert metadata["technical_profile"] == TECHNICAL_DIAGRAM_PROFILE_GRAPH_PAPER
    assert metadata["theme_id"] in TECHNICAL_DIAGRAM_PROFILE_THEME_IDS[TECHNICAL_DIAGRAM_PROFILE_GRAPH_PAPER]
    assert len(metadata["selection"]["available_theme_ids"]) == 25
    assert metadata["selection"]["allow_dark"] is True
    assert metadata["selection"]["require_grid"] is True
    assert style.grid_kind in {"square", "lab_grid"}


def test_physics_scene_generation_uses_expected_technical_profiles() -> None:
    analog = PhysicsAnalogMeterReadoutValueTask().generate(
        12101,
        params={"post_image_noise": {"enabled": False}},
        max_attempts=20,
    )
    motion = PhysicsMotionGraphIntervalDisplacementValueTask().generate(
        12102,
        params={"post_image_noise": {"enabled": False}},
        max_attempts=20,
    )
    pv = PhysicsPVDiagramWorkValueTask().generate(
        12103,
        params={"post_image_noise": {"enabled": False}},
        max_attempts=20,
    )

    assert analog.trace_payload["render_spec"]["technical_diagram_style"]["technical_profile"] == (
        TECHNICAL_DIAGRAM_PROFILE_ANALYTICAL
    )
    assert motion.trace_payload["render_spec"]["technical_diagram_style"]["technical_profile"] == (
        TECHNICAL_DIAGRAM_PROFILE_GRAPH_PAPER
    )
    assert pv.trace_payload["render_spec"]["technical_diagram_style"]["technical_profile"] == (
        TECHNICAL_DIAGRAM_PROFILE_GRAPH_PAPER
    )
