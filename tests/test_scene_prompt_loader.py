"""Tests for scene-routed prompt bundle loading."""

from __future__ import annotations

import json

import trace_tasks.core.prompts.assets as prompt_assets
from trace_tasks.core.prompts import load_scene_prompt_bundle, render_prompt


def _write_prompt_bundle(path, *, bundle_id: str) -> None:
    variants = [f"variant {index}" for index in range(5)]
    payload = {
        "bundle_id": bundle_id,
        "schema_version": "v0",
        "scene_templates": {"demo_scene": [f"Scene {item}: {{thing}}" for item in variants]},
        "task_templates": {"demo_task": [f"Task {item}: {{question}}" for item in variants]},
        "answer_or_annotation_templates": {"answer_only": ["" for _ in variants]},
        "required_slots_by_key": {
            "scene:demo_scene": ["thing"],
            "task:demo_task": ["question"],
        },
    }
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_scene_prompt_bundle_uses_scene_path(tmp_path, monkeypatch) -> None:
    prompt_root = tmp_path / "prompts"
    bundle_path = prompt_root / "demo" / "sample_scene" / "demo_scene_bundle_v0.json"
    _write_prompt_bundle(bundle_path, bundle_id="demo_scene_bundle_v0")
    monkeypatch.setenv("TRACE_PROMPT_ROOT", str(prompt_root))
    prompt_assets._CACHE.clear()

    bundle = load_scene_prompt_bundle("demo", "sample_scene", "demo_scene_bundle_v0")
    assert bundle.bundle_id == "demo_scene_bundle_v0"
    assert bundle.source_path == "demo/sample_scene/demo_scene_bundle_v0.json"

    rendered = render_prompt(
        domain="demo",
        scene_id="sample_scene",
        bundle_id="demo_scene_bundle_v0",
        scene_key="demo_scene",
        task_key="demo_task",
        answer_or_annotation_key="answer_only",
        slots={"thing": "a chart", "question": "What is shown?"},
        instance_seed=123,
    )
    assert "a chart" in rendered.prompt
    assert "What is shown?" in rendered.prompt
    assert rendered.metadata["template_paths"] == ["demo/sample_scene/demo_scene_bundle_v0.json"]
