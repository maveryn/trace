"""Tests for shared structured-information visual style sampling."""

from __future__ import annotations

from PIL import Image

from trace_tasks.tasks.shared.visual_style.information_scene import (
    INFORMATION_SCENE_PALETTES,
    INFORMATION_SCENE_TREATMENTS,
    information_scene_style_from_metadata,
    information_scene_style_metadata,
    make_information_scene_background,
    resolve_information_scene_style,
    resolve_information_scene_style_from_request,
)
from trace_tasks.tasks.shared.visual_style.request import build_visual_style_request
from trace_tasks.tasks.charts.shared.information_style import resolve_chart_information_style


def test_information_scene_style_has_expected_breadth() -> None:
    assert len(INFORMATION_SCENE_TREATMENTS) >= 20
    assert len(INFORMATION_SCENE_PALETTES) >= 20
    light_treatments = {
        treatment_id
        for treatment_id, treatment in INFORMATION_SCENE_TREATMENTS.items()
        if "light" in set(treatment.compatibility)
    }
    dark_treatments = {
        treatment_id
        for treatment_id, treatment in INFORMATION_SCENE_TREATMENTS.items()
        if "dark" in set(treatment.compatibility)
    }
    assert len(light_treatments) == 20
    assert len(dark_treatments) == 5


def test_information_scene_style_is_deterministic_and_metadata_complete() -> None:
    left_style, left_meta = resolve_information_scene_style(
        instance_seed=123,
        namespace="test.information_scene",
        allow_dark=False,
    )
    right_style, right_meta = resolve_information_scene_style(
        instance_seed=123,
        namespace="test.information_scene",
        allow_dark=False,
    )
    assert left_style == right_style
    assert left_meta == right_meta
    assert left_meta["kind"] == "information_scene_style"
    assert left_meta["style_pack"] == left_style.style_pack
    assert left_meta["surface_style"]["chrome_mode"] == left_style.chrome_mode
    assert len(left_meta["roles_rgb"]["canvas"]) == 3
    assert len(left_meta["contrast_checks"]) > 0
    assert left_meta["semantic_color_policy"] == "style_nonsemantic_roles_only"


def test_information_scene_background_records_style() -> None:
    style, _meta = resolve_information_scene_style(
        instance_seed=456,
        namespace="test.information_scene.background",
        treatments=("report_card",),
        palettes=("neutral_report",),
        chrome_modes=("thin_frame",),
        allow_dark=False,
    )
    image, background_meta = make_information_scene_background(
        canvas_width=320,
        canvas_height=220,
        style=style,
        instance_seed=456,
        namespace="test.information_scene.background",
    )
    assert isinstance(image, Image.Image)
    assert image.size == (320, 220)
    assert background_meta["selected_style"] == f"information_scene_style:{style.style_pack}"
    assert background_meta["style_spec"] == information_scene_style_metadata(style)


def test_information_scene_style_round_trips_from_metadata() -> None:
    style, meta = resolve_information_scene_style(
        instance_seed=567,
        namespace="test.information_scene.roundtrip",
        treatments=("desktop_app_window",),
        palettes=("metro_bright",),
        chrome_modes=("accent_frame",),
        protected_colors=((0, 114, 178),),
        allow_dark=False,
    )
    rebuilt = information_scene_style_from_metadata(meta)
    assert rebuilt == style
    image, background_meta = make_information_scene_background(
        canvas_width=240,
        canvas_height=180,
        style=rebuilt,
        instance_seed=567,
        namespace="test.information_scene.roundtrip",
    )
    assert isinstance(image, Image.Image)
    assert background_meta["style_spec"]["style_pack"] == style.style_pack


def test_information_scene_protected_color_metadata() -> None:
    _style, meta = resolve_information_scene_style(
        instance_seed=789,
        namespace="test.information_scene.protected",
        protected_colors=((30, 126, 147),),
        allow_dark=False,
    )
    assert meta["protected_colors_rgb"] == [[30, 126, 147]]
    assert meta["selection"]["min_protected_rgb_distance_required"] > 0
    assert "protected_palette_filter_fallback" in meta["selection"]


def test_information_scene_request_records_policy_and_can_enable_dark() -> None:
    request = build_visual_style_request(
        domain="charts",
        scene_id="histogram",
        routing_key="distribution",
        instance_seed=42,
        params={},
        style_family="information_scene",
        allow_dark=True,
        allow_colored_surface=True,
        protected_colors=((20, 120, 200),),
        required_text_roles=("axis_tick", "chart_label"),
    )
    _style, meta = resolve_information_scene_style_from_request(
        request,
        treatments=("dark_analytics_board",),
        palettes=("dark_analytics",),
    )
    assert meta["style_request"]["domain"] == "charts"
    assert meta["style_request"]["style_family"] == "information_scene"
    assert meta["style_request"]["allow_dark"] is True
    assert meta["style_request"]["protected_colors_rgb"] == [[20, 120, 200]]
    assert meta["palette_id"] == "dark_analytics"


def test_chart_information_style_reads_group_defaults() -> None:
    _style, meta = resolve_chart_information_style(
        instance_seed=91,
        params={},
        scene_id="histogram",
    )
    assert meta["style_request"]["domain"] == "charts"
    assert meta["style_request"]["allow_dark"] is True
    assert meta["style_request"]["allow_colored_surface"] is True
    assert "dark_analytics_board" in set(meta["selection"]["eligible_treatments"])
