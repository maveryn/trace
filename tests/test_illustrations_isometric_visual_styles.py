"""Tests for shared isometric illustration background tone support."""

from __future__ import annotations

from pathlib import Path

import yaml

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.illustrations.isometric_farmstead.shared.rendering import render_isometric_farmstead_scene
from trace_tasks.tasks.illustrations.isometric_harbor.shared.rendering import render_isometric_harbor_scene
from trace_tasks.tasks.illustrations.isometric_quarry.shared.rendering import render_isometric_quarry_scene
from trace_tasks.tasks.illustrations.shared.isometric_visual_styles import resolve_isometric_illustration_tone
from trace_tasks.tasks.shared.config_defaults import split_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.visual_style.surface_tones import DARK_SURFACE_TONE_IDS, DEFAULT_SURFACE_TONES


ISOMETRIC_SCENES = ("isometric_farmstead", "isometric_harbor", "isometric_quarry")


def _render_defaults(scene_id: str) -> dict:
    scene_defaults = get_scene_defaults("illustrations", scene_id)
    _gen_defaults, render_defaults, _prompt_defaults = split_scene_generation_rendering_prompt_defaults(scene_defaults)
    return dict(render_defaults)


def test_isometric_tone_resolver_uses_shared_25_tone_pool() -> None:
    approved_tones = set(DEFAULT_SURFACE_TONES)
    for scene_id in ISOMETRIC_SCENES:
        render_defaults = _render_defaults(scene_id)
        tone_ids = {
            resolve_isometric_illustration_tone(
                params={},
                render_defaults=render_defaults,
                instance_seed=2026062600 + seed,
                namespace=f"test.{scene_id}",
            ).tone_id
            for seed in range(100)
        }
        assert tone_ids.issubset(approved_tones), scene_id
        assert "custom" not in tone_ids, scene_id
        assert len(tone_ids) >= 12, scene_id
        assert tone_ids & DARK_SURFACE_TONE_IDS, scene_id


def test_isometric_renderers_record_actual_background_tone_metadata() -> None:
    renderers = {
        "isometric_farmstead": render_isometric_farmstead_scene,
        "isometric_harbor": render_isometric_harbor_scene,
        "isometric_quarry": render_isometric_quarry_scene,
    }
    for scene_id, renderer in renderers.items():
        scene = renderer(
            17,
            width=640,
            height=480,
            canvas_profile="landscape",
            render_style_params={"background_tone_id": "charcoal_concrete"},
            render_style_defaults=_render_defaults(scene_id),
        )
        assert scene.trace["background_tone_id"] == "charcoal_concrete"
        assert scene.trace["background_tone_rgb"] == list(DEFAULT_SURFACE_TONES["charcoal_concrete"]["floor_rgb"])
        assert scene.trace["background_rgb"] == list(DEFAULT_SURFACE_TONES["charcoal_concrete"]["floor_rgb"])
        assert scene.trace["background_tone_family"] == "dark"


def test_isometric_scene_configs_do_not_pin_shared_surface_tones() -> None:
    blocked_keys = {
        "surface_tones",
        "background_rgb",
        "canvas_rgb",
        "terrain_shadow_rgb",
        "terrain_edge_rgb",
        "terrain_light_rgb",
        "label_text_rgb",
        "label_stroke_rgb",
    }
    for scene_id in ISOMETRIC_SCENES:
        config_path = Path("src/trace_tasks/resources/configs/domains/illustrations") / f"{scene_id}.yaml"
        data = yaml.safe_load(config_path.read_text()) or {}
        hits: list[str] = []

        def walk(value: object, *, path: str) -> None:
            if isinstance(value, dict):
                for key, child in value.items():
                    next_path = f"{path}.{key}" if path else str(key)
                    if str(key) in blocked_keys:
                        hits.append(next_path)
                    walk(child, path=next_path)
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    walk(child, path=f"{path}[{index}]")

        walk(data, path=config_path.name)
        assert hits == [], scene_id
