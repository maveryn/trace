"""Regression tests for shared scene config helpers.

Domain-specific default assertions live in focused `tests/test_*_scene_config.py`
files. Keep this module limited to cross-domain config-loader and helper invariants.
"""

from __future__ import annotations

import pytest

import trace_tasks.core.scene_config as scene_config
from trace_tasks.core.scene_config import (
    get_domain_defaults,
    get_scene_defaults,
    resolve_scene_section_defaults,
)
from trace_tasks.tasks.shared.visual_defaults import load_scene_background_defaults, load_scene_noise_defaults
from trace_tasks.tasks.shared.config_defaults import (
    required_group_default,
    required_group_defaults,
    resolve_optional_int_bounds,
    resolve_required_float_bounds,
    resolve_required_int_bounds,
)


def test_domain_defaults_and_missing_group_behavior() -> None:
    assert get_scene_defaults("missing_domain", "missing_group") == {}
    domain_cfg = get_domain_defaults("geometry")
    cfg = get_scene_defaults("geometry", "missing_group")
    assert cfg["rendering"]["shared"] == domain_cfg["rendering"]["shared"]


def test_scene_defaults_and_section_resolution(tmp_path, monkeypatch) -> None:
    config_root = tmp_path / "configs" / "domains"
    domain_dir = config_root / "demo"
    domain_dir.mkdir(parents=True)
    (domain_dir / "base.yaml").write_text(
        """
generation:
  shared:
    base_only: true
    shared_value: base
rendering:
  shared:
    canvas_width: 100
    color: base
  task_overrides:
    task_demo__sample_scene__lookup_value:
      canvas_width: 120
visual:
  background:
    enabled: false
  noise:
    apply_prob: 0.1
""",
        encoding="utf-8",
    )
    (domain_dir / "sample_scene.yaml").write_text(
        """
generation:
  shared:
    scene_only: true
    shared_value: scene
rendering:
  shared:
    canvas_height: 200
    color: scene
  task_overrides:
    task_demo__sample_scene__lookup_value:
      canvas_height: 240
visual:
  background:
    enabled: true
    style_id: scene_paper
  noise:
    apply_prob: 0.35
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("TRACE_DOMAIN_CONFIG_ROOT", str(config_root))
    scene_config._CACHE_BY_PATH.clear()

    cfg = get_scene_defaults("demo", "sample_scene")
    assert cfg["generation"]["shared"]["base_only"] is True
    assert cfg["generation"]["shared"]["scene_only"] is True
    assert cfg["generation"]["shared"]["shared_value"] == "scene"
    assert get_scene_defaults("demo", "missing_scene")["generation"]["shared"]["shared_value"] == "base"

    shared_rendering = resolve_scene_section_defaults(cfg, "rendering")
    task_rendering = resolve_scene_section_defaults(
        cfg,
        "rendering",
        task_id="task_demo__sample_scene__lookup_value",
    )
    assert shared_rendering["canvas_width"] == 100
    assert shared_rendering["canvas_height"] == 200
    assert shared_rendering["color"] == "scene"
    assert task_rendering["canvas_width"] == 120
    assert task_rendering["canvas_height"] == 240


def test_scene_visual_defaults_load_with_fallbacks(tmp_path, monkeypatch) -> None:
    config_root = tmp_path / "configs" / "domains"
    domain_dir = config_root / "demo"
    domain_dir.mkdir(parents=True)
    (domain_dir / "base.yaml").write_text("visual: {}\n", encoding="utf-8")
    (domain_dir / "sample_scene.yaml").write_text(
        """
visual:
  background:
    enabled: true
    style_id: scene_paper
  noise:
    apply_prob: 0.35
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("TRACE_DOMAIN_CONFIG_ROOT", str(config_root))
    scene_config._CACHE_BY_PATH.clear()

    background = load_scene_background_defaults(
        domain="demo",
        scene_id="sample_scene",
        fallback={"fallback_key": "kept", "enabled": False},
        merge_with_fallback=True,
    )
    noise = load_scene_noise_defaults(
        domain="demo",
        scene_id="sample_scene",
        fallback={"apply_prob": 0.0, "fallback_key": "dropped"},
        merge_with_fallback=False,
    )
    assert background == {"fallback_key": "kept", "enabled": True, "style_id": "scene_paper"}
    assert noise == {"apply_prob": 0.35}


def test_section_defaults_require_shared_and_task_overrides_schema() -> None:
    mapping = {
        "rendering": {
            "canvas_size_min": 111,
            "shared": {"canvas_size_min": 512},
            "task_overrides": {"demo_task": {"canvas_size_min": 768}},
        }
    }
    shared_only = resolve_scene_section_defaults(mapping, "rendering")
    task_specific = resolve_scene_section_defaults(mapping, "rendering", task_id="demo_task")
    assert int(shared_only["canvas_size_min"]) == 512
    assert int(task_specific["canvas_size_min"]) == 768
    assert resolve_scene_section_defaults({"rendering": {"canvas_size_min": 111}}, "rendering") == {}


def test_required_group_helpers_enforce_presence_and_nonempty() -> None:
    assert required_group_default({"value": 3}, "value", context="test") == 3
    resolved = required_group_defaults({"a": 1, "b": "ok"}, ("a", "b"), context="test")
    assert resolved == {"a": 1, "b": "ok"}
    with pytest.raises(ValueError):
        required_group_default({}, "value", context="test")
    with pytest.raises(ValueError):
        required_group_default({"value": "   "}, "value", context="test")
    with pytest.raises(ValueError):
        required_group_defaults({"a": 1}, ("a", "b"), context="test")


def test_resolve_numeric_bounds_helpers() -> None:
    assert resolve_optional_int_bounds(
        {"answer_min": 3},
        {"answer_max": 9},
        min_key="answer_min",
        max_key="answer_max",
        context="test",
    ) == (3, 9)
    assert resolve_optional_int_bounds(
        {},
        {},
        min_key="answer_min",
        max_key="answer_max",
        context="test",
    ) == (None, None)
    with pytest.raises(ValueError):
        resolve_optional_int_bounds(
            {"answer_min": 10, "answer_max": 2},
            {},
            min_key="answer_min",
            max_key="answer_max",
            context="test",
        )
    assert resolve_required_int_bounds(
        {"min": 2},
        {"max": 8},
        min_key="min",
        max_key="max",
        fallback_min=1,
        fallback_max=9,
        context="test",
    ) == (2, 8)
    assert resolve_required_float_bounds(
        {"min_f": 0.1},
        {"max_f": 0.6},
        min_key="min_f",
        max_key="max_f",
        fallback_min=0.0,
        fallback_max=1.0,
        context="test",
    ) == (0.1, 0.6)
    with pytest.raises(ValueError):
        resolve_required_int_bounds(
            {"min": 9, "max": 2},
            {},
            min_key="min",
            max_key="max",
            fallback_min=0,
            fallback_max=1,
            context="test",
        )
    with pytest.raises(ValueError):
        resolve_required_float_bounds(
            {"min_f": 0.9, "max_f": 0.2},
            {},
            min_key="min_f",
            max_key="max_f",
            fallback_min=0.0,
            fallback_max=1.0,
            context="test",
        )
