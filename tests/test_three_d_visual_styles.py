"""Tests for shared three_d visual style resolution."""

from __future__ import annotations

from pathlib import Path

import yaml

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.shared.config_defaults import split_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.visual_style.surface_tones import DEFAULT_SURFACE_TONES
from trace_tasks.tasks.three_d.shared.object_scene import _resolve_render_params as resolve_object_scene_render_params
from trace_tasks.tasks.three_d.shared.visual_styles import DEFAULT_CONVEYOR_BELT_STYLES, DEFAULT_THREE_D_SURFACE_TONES
from trace_tasks.tasks.three_d.street.shared.state import _resolve_render_params as resolve_street_render_params
from trace_tasks.tasks.three_d.warehouse.shared.state import _resolve_render_params as resolve_warehouse_render_params


def _render_defaults(scene_id: str) -> dict:
    scene_defaults = get_scene_defaults("three_d", scene_id)
    _gen_defaults, render_defaults, _prompt_defaults = split_scene_generation_rendering_prompt_defaults(scene_defaults)
    return dict(render_defaults)


def test_large_object_three_d_scenes_do_not_sample_portrait_canvas() -> None:
    resolvers = {
        "object_scene": resolve_object_scene_render_params,
        "room": resolve_object_scene_render_params,
        "street": resolve_street_render_params,
        "warehouse": resolve_warehouse_render_params,
    }

    for scene_id, resolver in resolvers.items():
        render_defaults = _render_defaults(scene_id)
        assert float(render_defaults["canvas_preset_weights"]["portrait"]) == 0.0
        presets = {
            str(
                resolver(
                    {},
                    render_defaults=render_defaults,
                    instance_seed=2026062700 + seed,
                    namespace=f"test.no_portrait.{scene_id}",
                ).canvas_preset
            )
            for seed in range(48)
        }
        assert presets <= {"landscape", "square"}, scene_id
        assert presets == {"landscape", "square"}, scene_id


def test_three_d_surface_tone_pool_has_expected_light_dark_split() -> None:
    dark_tones = {
        "charcoal_concrete",
        "deep_warehouse",
        "midnight_steel",
        "dark_sage_floor",
        "graphite_plaster",
    }

    assert len(DEFAULT_THREE_D_SURFACE_TONES) == 25
    assert dark_tones.issubset(set(DEFAULT_THREE_D_SURFACE_TONES))
    assert len(set(DEFAULT_THREE_D_SURFACE_TONES) - dark_tones) == 20
    for tone_id, tone in DEFAULT_THREE_D_SURFACE_TONES.items():
        assert {"floor_rgb", "grid_rgb", "edge_rgb", "surface_accent_rgb", "text_rgb", "text_stroke_rgb"}.issubset(tone), tone_id


def test_three_d_surface_tones_use_shared_pool() -> None:
    assert DEFAULT_THREE_D_SURFACE_TONES == DEFAULT_SURFACE_TONES


def test_shared_three_d_surface_tones_cover_current_scene_renderers() -> None:
    resolvers = {
        "object_cluster": resolve_object_scene_render_params,
        "object_scene": resolve_object_scene_render_params,
        "room": resolve_object_scene_render_params,
        "surface_fixture": resolve_object_scene_render_params,
        "conveyor": resolve_object_scene_render_params,
        "carousel": resolve_object_scene_render_params,
        "street": resolve_street_render_params,
        "warehouse": resolve_warehouse_render_params,
    }
    approved_tones = set(DEFAULT_THREE_D_SURFACE_TONES)
    dark_tones = {
        "charcoal_concrete",
        "deep_warehouse",
        "midnight_steel",
        "dark_sage_floor",
        "graphite_plaster",
    }
    for scene_id, resolver in resolvers.items():
        render_defaults = _render_defaults(scene_id)
        tone_ids = {
            str(
                resolver(
                    {},
                    render_defaults=render_defaults,
                    instance_seed=2026062500 + seed,
                    namespace=f"test.{scene_id}",
                ).background_tone_id
            )
            for seed in range(100)
        }
        assert tone_ids.issubset(approved_tones), scene_id
        assert "custom" not in tone_ids, scene_id
        assert len(tone_ids) >= 12, scene_id
        assert tone_ids & dark_tones, scene_id


def test_scene_configs_do_not_pin_shared_surface_tones() -> None:
    blocked_keys = {
        "floor_rgb",
        "grid_rgb",
        "edge_rgb",
        "text_rgb",
        "text_stroke_rgb",
        "sidewalk_rgb",
        "curb_rgb",
        "aisle_rgb",
        "shelf_zone_rgb",
    }
    config_root = Path("src/trace_tasks/resources/configs/domains/three_d")

    def walk(value: object, *, path: str) -> list[str]:
        if isinstance(value, dict):
            hits: list[str] = []
            for key, child in value.items():
                next_path = f"{path}.{key}" if path else str(key)
                if key in blocked_keys:
                    hits.append(next_path)
                hits.extend(walk(child, path=next_path))
            return hits
        if isinstance(value, list):
            hits = []
            for index, child in enumerate(value):
                hits.extend(walk(child, path=f"{path}[{index}]"))
            return hits
        return []

    for config_path in sorted(config_root.glob("*.yaml")):
        if config_path.name == "base.yaml":
            continue
        data = yaml.safe_load(config_path.read_text()) or {}
        assert walk(data, path=config_path.name) == []


def test_surface_tone_override_stays_deterministic() -> None:
    render_defaults = _render_defaults("object_scene")
    render_params = resolve_object_scene_render_params(
        {"background_tone_id": "pale_sand"},
        render_defaults=render_defaults,
        instance_seed=7,
        namespace="test.override",
    )

    assert render_params.background_tone_id == "pale_sand"
    assert render_params.floor_rgb == DEFAULT_THREE_D_SURFACE_TONES["pale_sand"]["floor_rgb"]
    assert render_params.grid_rgb == DEFAULT_THREE_D_SURFACE_TONES["pale_sand"]["grid_rgb"]


def test_dark_surface_tone_override_uses_style_text_colors() -> None:
    render_defaults = _render_defaults("conveyor")
    render_params = resolve_object_scene_render_params(
        {"background_tone_id": "charcoal_concrete"},
        render_defaults=render_defaults,
        instance_seed=17,
        namespace="test.dark_override",
    )
    expected = DEFAULT_THREE_D_SURFACE_TONES["charcoal_concrete"]

    assert render_params.background_tone_id == "charcoal_concrete"
    assert render_params.floor_rgb == expected["floor_rgb"]
    assert render_params.text_rgb == expected["text_rgb"]
    assert render_params.text_stroke_rgb == expected["text_stroke_rgb"]


def test_custom_floor_override_keeps_custom_text_defaults() -> None:
    render_defaults = _render_defaults("conveyor")
    render_params = resolve_object_scene_render_params(
        {"floor_rgb": (12, 13, 14), "grid_rgb": (40, 41, 42)},
        render_defaults=render_defaults,
        instance_seed=19,
        namespace="test.custom_floor",
    )

    assert render_params.background_tone_id == "custom"
    assert render_params.floor_rgb == (12, 13, 14)
    assert render_params.grid_rgb == (40, 41, 42)
    assert render_params.text_rgb == (30, 34, 42)
    assert render_params.text_stroke_rgb == (255, 255, 255)


def test_conveyor_belt_styles_are_only_enabled_for_belt_scenes() -> None:
    approved_styles = set(DEFAULT_CONVEYOR_BELT_STYLES)
    for scene_id in ("conveyor", "carousel"):
        render_defaults = _render_defaults(scene_id)
        style_ids = {
            str(
                resolve_object_scene_render_params(
                    {},
                    render_defaults=render_defaults,
                    instance_seed=2026062600 + seed,
                    namespace=f"test.{scene_id}",
                ).conveyor_belt_style_id
            )
            for seed in range(24)
        }
        assert style_ids.issubset(approved_styles), scene_id
        assert len(style_ids) >= 4, scene_id

    render_defaults = _render_defaults("object_scene")
    render_params = resolve_object_scene_render_params(
        {},
        render_defaults=render_defaults,
        instance_seed=2026062601,
        namespace="test.object_scene",
    )
    assert render_params.conveyor_belt_style_id is None


def test_conveyor_belt_style_override_stays_deterministic() -> None:
    render_defaults = _render_defaults("conveyor")
    render_params = resolve_object_scene_render_params(
        {"conveyor_belt_style_id": "dark_rubber"},
        render_defaults=render_defaults,
        instance_seed=11,
        namespace="test.belt_override",
    )

    assert render_params.conveyor_belt_style_id == "dark_rubber"
    assert render_params.conveyor_belt_fill_rgb == DEFAULT_CONVEYOR_BELT_STYLES["dark_rubber"]["fill_rgb"]
    assert render_params.conveyor_belt_outline_rgb == DEFAULT_CONVEYOR_BELT_STYLES["dark_rubber"]["outline_rgb"]
