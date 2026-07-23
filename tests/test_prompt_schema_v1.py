"""Tests for scene prompt asset schema v1."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

import pytest

import trace_tasks.core.prompts.assets as prompt_assets
from trace_tasks.core.prompts import load_scene_prompt_bundle, render_prompt
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec, build_prompt_trace_artifacts, render_scene_prompt_variants


_VARIANTS = tuple(range(5))


def _variant_templates(text: str) -> list[str]:
    return [f"{text} Variant {index}." for index in _VARIANTS]


def _valid_bundle() -> dict[str, Any]:
    return {
        "bundle_id": "demo_v1",
        "schema_version": "v1",
        "templates": {
            "scene": {
                "demo_scene": _variant_templates("Scene shows {object_description}")
            },
            "task": {
                "demo_task": _variant_templates("Use target label {target_label}")
            },
            "query": {
                "demo_query": _variant_templates("Which option is {target_label}?")
            },
            "output": {
                "answer_only": _variant_templates(
                    "Answer format: {answer_hint}\nExample JSON:\n{json_example_answer_only}"
                ),
                "answer_and_annotation": _variant_templates(
                    "Annotation format: {annotation_hint}\n"
                    "Answer format: {answer_hint}\n"
                    "Example JSON:\n{json_example}"
                ),
            },
        },
        "static_slots_by_key": {
            "scene:demo_scene": {
                "object_description": "a labeled option panel",
            },
            "query:demo_query": {
                "answer_hint": 'set "answer" to a single option label A, B, C, or D',
                "annotation_hint": 'set "annotation" to one bbox around the selected option',
                "json_example": '{"annotation":[[1,2,3,4]],"answer":"B"}',
                "json_example_answer_only": '{"answer":"B"}',
            },
        },
        "dynamic_slots": {
            "target_label": {
                "type": "label",
                "scope": "query:demo_query",
                "description": "The target option label for this generated instance.",
            }
        },
        "required_slots_by_key": {
            "query:demo_query": ["target_label"],
        },
    }


def _write_bundle(prompt_root: Path, payload: Mapping[str, Any]) -> None:
    bundle_path = prompt_root / "demo" / "demo_scene" / "demo_v1.json"
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    bundle_path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture()
def prompt_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "prompts"
    monkeypatch.setenv("TRACE_PROMPT_ROOT", str(root))
    prompt_assets._CACHE.clear()
    return root


def test_v1_scene_prompt_rendering_resolves_static_and_dynamic_slots(prompt_root: Path) -> None:
    _write_bundle(prompt_root, _valid_bundle())

    bundle = load_scene_prompt_bundle("demo", "demo_scene", "demo_v1")
    assert bundle.schema_version == "v1"
    assert bundle.source_path == "demo/demo_scene/demo_v1.json"
    assert bundle.source_hash

    rendered = render_prompt(
        domain="demo",
        scene_id="demo_scene",
        bundle_id="demo_v1",
        scene_key="demo_scene",
        task_key="demo_task",
        query_key="demo_query",
        answer_or_annotation_key="answer_and_annotation",
        dynamic_slots={"target_label": "C"},
        instance_seed=123,
    )

    assert "a labeled option panel" in rendered.prompt
    assert "C" in rendered.prompt
    assert '"annotation"' in rendered.prompt
    assert '"answer"' in rendered.prompt

    metadata = rendered.metadata
    assert metadata["prompt_schema_version"] == "v1"
    assert metadata["schema_version"] == "v1"
    assert metadata["prompt_bundle_path"] == "demo/demo_scene/demo_v1.json"
    assert metadata["prompt_bundle_hash"] == bundle.source_hash
    assert metadata["selected_keys"] == {
        "scene": "demo_scene",
        "task": "demo_task",
        "query": "demo_query",
        "output": "answer_and_annotation",
    }
    assert set(metadata["selected_indices"]) == {"scene", "task", "query", "output"}
    assert metadata["variant_counts"] == {
        "scene": 5,
        "task": 5,
        "query": 5,
        "output": 5,
    }
    assert metadata["slot_values"]["target_label"] == "C"
    assert metadata["slot_values"]["object_description"] == "a labeled option panel"
    assert metadata["slot_sources"]["target_label"] == "dynamic:target_label"
    assert metadata["slot_sources"]["object_description"] == "static:scene:demo_scene"


def test_v1_prompt_variant_wrapper_accepts_dynamic_slots(prompt_root: Path) -> None:
    _write_bundle(prompt_root, _valid_bundle())

    rendered = render_scene_prompt_variants(
        domain="demo",
        scene_id="demo_scene",
        bundle_id="demo_v1",
        scene_key="demo_scene",
        task_key="demo_task",
        query_key="demo_query",
        dynamic_slots={"target_label": "A"},
        instance_seed=456,
    )

    assert sorted(rendered.prompt_variants) == ["answer_and_annotation", "answer_only"]
    assert rendered.active_mode == "answer_and_annotation"
    assert "A" in rendered.prompt


def test_prompt_query_spec_uses_canonical_trace_shape(prompt_root: Path) -> None:
    _write_bundle(prompt_root, _valid_bundle())

    rendered = render_scene_prompt_variants(
        domain="demo",
        scene_id="demo_scene",
        bundle_id="demo_v1",
        scene_key="demo_scene",
        task_key="demo_task",
        query_key="demo_query",
        dynamic_slots={"target_label": "D"},
        instance_seed=789,
    )
    artifacts = build_prompt_trace_artifacts(rendered)

    query_spec = build_prompt_query_spec(
        prompt_artifacts=artifacts,
        query_id="demo_query",
        params={"target_label": "D", "query_id": "demo_query"},
    )

    assert query_spec["query_id"] == "demo_query"
    assert query_spec["template_id"] == "demo_v1"
    assert query_spec["params"]["query_id"] == "demo_query"
    assert query_spec["params"]["target_label"] == "D"
    assert query_spec["prompt_variant"]["prompt_bundle_id"] == "demo_v1"
    assert query_spec["prompt_variant_active_key"] == "answer_and_annotation"
    assert sorted(query_spec["prompt_variants"]) == ["answer_and_annotation", "answer_only"]


def test_prompt_query_spec_rejects_conflicting_query_id(prompt_root: Path) -> None:
    _write_bundle(prompt_root, _valid_bundle())

    rendered = render_scene_prompt_variants(
        domain="demo",
        scene_id="demo_scene",
        bundle_id="demo_v1",
        scene_key="demo_scene",
        task_key="demo_task",
        query_key="demo_query",
        dynamic_slots={"target_label": "A"},
        instance_seed=789,
    )
    artifacts = build_prompt_trace_artifacts(rendered)

    with pytest.raises(ValueError, match="conflicts"):
        build_prompt_query_spec(
            prompt_artifacts=artifacts,
            query_id="demo_query",
            params={"query_id": "other_query"},
        )


def test_v1_rejects_arbitrary_slots(prompt_root: Path) -> None:
    _write_bundle(prompt_root, _valid_bundle())

    with pytest.raises(ValueError, match="requires dynamic_slots"):
        render_prompt(
            domain="demo",
            scene_id="demo_scene",
            bundle_id="demo_v1",
            scene_key="demo_scene",
            task_key="demo_task",
            query_key="demo_query",
            answer_or_annotation_key="answer_only",
            slots={"target_label": "A"},
            instance_seed=123,
        )


def test_v1_rejects_missing_required_dynamic_slot(prompt_root: Path) -> None:
    _write_bundle(prompt_root, _valid_bundle())

    with pytest.raises(ValueError, match="missing required prompt slots"):
        render_prompt(
            domain="demo",
            scene_id="demo_scene",
            bundle_id="demo_v1",
            scene_key="demo_scene",
            task_key="demo_task",
            query_key="demo_query",
            answer_or_annotation_key="answer_only",
            dynamic_slots={},
            instance_seed=123,
        )


def test_v1_rejects_undeclared_template_placeholders(prompt_root: Path) -> None:
    payload = _valid_bundle()
    payload["templates"]["task"]["demo_task"][0] = "Missing {not_declared}."
    _write_bundle(prompt_root, payload)

    with pytest.raises(ValueError, match="not declared"):
        load_scene_prompt_bundle("demo", "demo_scene", "demo_v1")


def test_v1_rejects_static_dynamic_slot_collisions(prompt_root: Path) -> None:
    payload = _valid_bundle()
    payload["static_slots_by_key"]["task:demo_task"] = {"target_label": "A"}
    _write_bundle(prompt_root, payload)

    with pytest.raises(ValueError, match="static/dynamic prompt slot collision"):
        load_scene_prompt_bundle("demo", "demo_scene", "demo_v1")


def test_v1_rejects_nested_static_placeholders(prompt_root: Path) -> None:
    payload = _valid_bundle()
    payload["static_slots_by_key"]["scene:demo_scene"]["object_description"] = "a {bad} panel"
    _write_bundle(prompt_root, payload)

    with pytest.raises(ValueError, match="nested placeholders"):
        load_scene_prompt_bundle("demo", "demo_scene", "demo_v1")


def test_v1_rejects_malformed_json_examples(prompt_root: Path) -> None:
    payload = _valid_bundle()
    payload["static_slots_by_key"]["query:demo_query"]["json_example"] = '{"answer":"B"}'
    _write_bundle(prompt_root, payload)

    with pytest.raises(ValueError, match="annotation key"):
        load_scene_prompt_bundle("demo", "demo_scene", "demo_v1")


def test_v1_rejects_unknown_dynamic_slots(prompt_root: Path) -> None:
    _write_bundle(prompt_root, _valid_bundle())

    with pytest.raises(ValueError, match="not declared"):
        render_prompt(
            domain="demo",
            scene_id="demo_scene",
            bundle_id="demo_v1",
            scene_key="demo_scene",
            task_key="demo_task",
            query_key="demo_query",
            answer_or_annotation_key="answer_only",
            dynamic_slots={"target_label": "A", "answer_hint": "override"},
            instance_seed=123,
        )


def test_v1_rejects_dynamic_slot_type_mismatches(prompt_root: Path) -> None:
    payload = copy.deepcopy(_valid_bundle())
    payload["templates"]["task"]["demo_task"] = _variant_templates("There are {target_count} targets")
    payload["templates"]["query"]["demo_query"] = _variant_templates("Count the matching targets")
    payload["dynamic_slots"] = {
        "target_count": {
            "type": "integer",
            "scope": "task:demo_task",
            "description": "Target count.",
        }
    }
    payload["required_slots_by_key"] = {"task:demo_task": ["target_count"]}
    _write_bundle(prompt_root, payload)

    with pytest.raises(ValueError, match="must be an integer"):
        render_prompt(
            domain="demo",
            scene_id="demo_scene",
            bundle_id="demo_v1",
            scene_key="demo_scene",
            task_key="demo_task",
            query_key="demo_query",
            answer_or_annotation_key="answer_only",
            dynamic_slots={"target_count": "3"},
            instance_seed=123,
        )
