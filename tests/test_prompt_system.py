"""Tests for the current Trace prompt bundle system."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

from trace_tasks.core.prompts import load_prompt_bundle, render_prompt, render_prompt_variants
from trace_tasks.core.prompts.schema import PROMPT_SCHEMA_V1, REQUIRED_PROMPT_VARIANTS
from trace_tasks.core.seed import hash64
from trace_tasks.tasks import TASK_REGISTRY, create_task
from trace_tasks.tasks.shared.prompt_json_example import (
    build_prompt_json_examples,
    dump_prompt_json_examples,
)

ANSWER_FORMAT_TEXT = re.compile(
    r'(Answer format:|Answer field:|Required answer format:|Final answer format:|Use this answer format:|Format for the "answer" field:)'
)
ANNOTATION_FORMAT_TEXT = re.compile(
    r'(Annotation format:|Annotation field:|Required annotation format:|Final annotation format:|Use this annotation format:|Format for the "annotation" field:)'
)


def _active_prompt_bundle_coords() -> list[tuple[str, str, str]]:
    """Return prompt bundles referenced by current domain/scene configs."""

    coords: set[tuple[str, str, str]] = set()

    def _walk(value: object) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "bundle_id" and isinstance(child, str):
                    coords.add((domain, scene_id, str(child)))
                _walk(child)
        elif isinstance(value, list):
            for child in value:
                _walk(child)

    for path in sorted(Path("src/trace_tasks/resources/configs/domains").glob("*/*.yaml")):
        domain = str(path.parent.name)
        scene_id = str(path.stem)
        _walk(yaml.safe_load(path.read_text()) or {})

    missing = [
        f"{domain}/{scene_id}/{bundle_id}"
        for domain, scene_id, bundle_id in sorted(coords)
        if not (
            Path("src/trace_tasks/resources/prompts")
            / domain
            / scene_id
            / f"{bundle_id}.json"
        ).exists()
    ]
    assert missing == []
    return sorted(coords)


def _semantic_prompt_part(prompt: str) -> str:
    """Return prompt text before output-mode formatting instructions."""

    markers = (
        "return json",
        "Return a JSON object",
        "Return only",
        "Respond with JSON only",
        "Use JSON only",
        "End your response",
        "Annotation format:",
        "Annotation field:",
        "Required annotation format:",
        "Final annotation format:",
        "Use this annotation format:",
        'Format for the "annotation" field:',
        "Answer format:",
        "Answer field:",
        "Required answer format:",
        "Final answer format:",
        "Use this answer format:",
        'Format for the "answer" field:',
    )
    text = str(prompt)
    lowered = text.lower()
    positions = [lowered.find(marker.lower()) for marker in markers if lowered.find(marker.lower()) >= 0]
    return text[: min(positions)].strip() if positions else text.strip()


def _render_current_v1_prompt(*, output_key: str = "answer_and_annotation"):
    return render_prompt(
        domain="geometry",
        scene_id="composite_shape",
        bundle_id="geometry_composite_shape_v1",
        scene_key="composite_shape_scene",
        task_key="composite_shape_query",
        query_key="l_profile_area",
        answer_or_annotation_key=output_key,
        dynamic_slots={
            "annotation_keys": "A, B, C, D",
            "json_example": '{"annotation":{"A":[120,180],"B":[220,180]},"answer":18}',
            "json_example_answer_only": '{"answer":18}',
        },
        instance_seed=4242,
    )


def test_render_prompt_is_deterministic_for_current_v1_bundle() -> None:
    a = _render_current_v1_prompt()
    b = _render_current_v1_prompt()

    assert a.prompt == b.prompt
    assert a.metadata == b.metadata
    assert a.metadata["prompt_bundle_id"] == "geometry_composite_shape_v1"
    assert a.metadata["answer_or_annotation_key"] == "answer_and_annotation"
    assert "Example JSON" in a.prompt
    assert ANNOTATION_FORMAT_TEXT.search(a.prompt) is not None
    assert ANSWER_FORMAT_TEXT.search(a.prompt) is not None


def test_prompt_bundle_contract_and_required_slots_for_current_v1_bundle() -> None:
    bundle = load_prompt_bundle(
        "geometry",
        "composite_shape",
        "geometry_composite_shape_v1",
    )

    assert bundle.schema_version == PROMPT_SCHEMA_V1
    assert bundle.allow_empty_task_templates is True
    assert len(bundle.scene_templates["composite_shape_scene"]) == REQUIRED_PROMPT_VARIANTS
    assert len(bundle.task_templates["composite_shape_query"]) == REQUIRED_PROMPT_VARIANTS
    assert len(bundle.query_templates["l_profile_area"]) == REQUIRED_PROMPT_VARIANTS
    assert len(bundle.answer_or_annotation_templates["answer_only"]) == REQUIRED_PROMPT_VARIANTS
    assert len(bundle.answer_or_annotation_templates["answer_and_annotation"]) == REQUIRED_PROMPT_VARIANTS
    assert bundle.required_slots_by_key["scene:composite_shape_scene"] == ("object_description",)
    assert bundle.required_slots_by_key["output:answer_only"] == (
        "answer_hint",
        "json_example_answer_only",
    )
    assert bundle.required_slots_by_key["output:answer_and_annotation"] == (
        "annotation_keys",
        "answer_hint",
        "json_example",
    )

    with pytest.raises(ValueError, match="missing required prompt slots"):
        render_prompt(
            domain="geometry",
            scene_id="composite_shape",
            bundle_id="geometry_composite_shape_v1",
            scene_key="composite_shape_scene",
            task_key="composite_shape_query",
            query_key="l_profile_area",
            answer_or_annotation_key="answer_and_annotation",
            dynamic_slots={},
            instance_seed=9999,
        )


def test_render_prompt_variants_contains_answer_only_and_answer_and_annotation() -> None:
    results = render_prompt_variants(
        domain="geometry",
        scene_id="composite_shape",
        bundle_id="geometry_composite_shape_v1",
        scene_key="composite_shape_scene",
        task_key="composite_shape_query",
        query_key="l_profile_area",
        answer_or_annotation_keys=("answer_only", "answer_and_annotation"),
        dynamic_slots={
            "annotation_keys": "A, B, C, D",
            "json_example": '{"annotation":{"A":[120,180],"B":[220,180]},"answer":18}',
            "json_example_answer_only": '{"answer":18}',
        },
        instance_seed=4242,
    )

    assert sorted(results.keys()) == ["answer_and_annotation", "answer_only"]
    assert '"annotation"' not in results["answer_only"].prompt
    assert '"annotation"' in results["answer_and_annotation"].prompt
    assert '"answer"' in results["answer_only"].prompt
    assert ANSWER_FORMAT_TEXT.search(results["answer_only"].prompt) is not None
    assert ANNOTATION_FORMAT_TEXT.search(results["answer_and_annotation"].prompt) is not None
    assert results["answer_only"].metadata["answer_or_annotation_key"] == "answer_only"
    assert results["answer_and_annotation"].metadata["answer_or_annotation_key"] == "answer_and_annotation"


def test_active_config_prompt_bundles_exist_and_load() -> None:
    coords = _active_prompt_bundle_coords()
    assert coords

    for domain, scene_id, bundle_id in coords:
        bundle = load_prompt_bundle(domain, scene_id, bundle_id)
        assert bundle.bundle_id == bundle_id
        assert bundle.scene_templates
        assert bundle.task_templates
        assert "answer_only" in bundle.answer_or_annotation_templates
        assert "answer_and_annotation" in bundle.answer_or_annotation_templates
        for templates in (
            *bundle.scene_templates.values(),
            *bundle.task_templates.values(),
            *bundle.query_templates.values(),
            *bundle.answer_or_annotation_templates.values(),
        ):
            assert len(templates) == REQUIRED_PROMPT_VARIANTS, bundle.source_path


def test_active_prompt_bundles_use_current_output_format_language() -> None:
    banned_query_patterns = (
        re.compile(r"\bAnswer with\b", flags=re.IGNORECASE),
        re.compile(r"\bRespond with\b", flags=re.IGNORECASE),
        re.compile(r"\bReturn only\b", flags=re.IGNORECASE),
        re.compile(r"\bReturn the\b", flags=re.IGNORECASE),
        re.compile(r"\bGive the final\b", flags=re.IGNORECASE),
    )

    for domain, scene_id, bundle_id in _active_prompt_bundle_coords():
        bundle = load_prompt_bundle(domain, scene_id, bundle_id)
        for template in bundle.answer_or_annotation_templates["answer_only"]:
            assert "Return a valid JSON object" not in str(template), bundle.source_path
            assert ANSWER_FORMAT_TEXT.search(str(template)) is not None, bundle.source_path
            assert ANNOTATION_FORMAT_TEXT.search(str(template)) is None, bundle.source_path
        for template in bundle.answer_or_annotation_templates["answer_and_annotation"]:
            assert "Return a valid JSON object" not in str(template), bundle.source_path
            assert ANSWER_FORMAT_TEXT.search(str(template)) is not None, bundle.source_path
            assert ANNOTATION_FORMAT_TEXT.search(str(template)) is not None, bundle.source_path

        for templates in bundle.query_templates.values():
            for template in templates:
                assert all(pattern.search(str(template)) is None for pattern in banned_query_patterns), bundle.source_path


def _example_answer_matches_type(answer_type: str, value: object) -> bool:
    if answer_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if answer_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if answer_type == "string":
        return isinstance(value, str)
    if answer_type == "option_letter":
        return isinstance(value, str) and len(value) == 1 and value.isalpha() and value.upper() == value
    if answer_type == "pi_expression":
        return isinstance(value, str)
    return True


def test_active_tasks_answer_only_prompts_stay_answer_only() -> None:
    banned_annotation_word = re.compile(r"\bannotation\b", flags=re.IGNORECASE)
    banned_bbox_word = re.compile(r"\bbbox\b|bounding box", flags=re.IGNORECASE)

    for task_id in sorted(TASK_REGISTRY):
        task = create_task(task_id)
        out = None
        last_error: Exception | None = None
        for sample_idx in range(8):
            try:
                out = task.generate(
                    hash64(20260411, f"{task_id}:answer_only_prompt_audit", sample_idx),
                    params={},
                    max_attempts=128,
                )
                break
            except Exception as exc:  # pragma: no cover - depends on generation retries
                last_error = exc
                continue
        if out is None:
            raise AssertionError(
                f"{task_id} failed answer audit generation across 8 deterministic seeds"
            ) from last_error

        prompt = str(out.prompt_variants.get("answer_only", ""))
        annotation_prompt = str(out.prompt_variants.get("answer_and_annotation", ""))
        assert prompt, task_id
        assert annotation_prompt, task_id
        assert _semantic_prompt_part(prompt) == _semantic_prompt_part(annotation_prompt), task_id
        assert "Example JSON:" in prompt, task_id
        assert '"annotation"' not in prompt, task_id
        output_text = prompt[len(_semantic_prompt_part(prompt)) :]
        assert ANNOTATION_FORMAT_TEXT.search(output_text) is None, task_id
        assert banned_bbox_word.search(output_text) is None, task_id

        query_spec = out.trace_payload.get("query_spec", {})
        prompt_variants = query_spec.get("prompt_variants", {})
        answer_only_variant = prompt_variants.get("answer_only", {})
        metadata = answer_only_variant.get("metadata", {})
        slot_values = metadata.get("slot_values", {})

        answer_hint = str(slot_values.get("answer_hint", ""))
        if answer_hint:
            assert banned_annotation_word.search(answer_hint) is None, task_id
            assert banned_bbox_word.search(answer_hint) is None, task_id

        example = json.loads(str(slot_values.get("json_example_answer_only", "")))
        assert list(example.keys()) == ["answer"], task_id
        assert _example_answer_matches_type(str(out.answer_gt.type), example["answer"]), task_id


def test_prompt_json_examples_use_non_degenerate_point_layouts() -> None:
    example_json, _ = build_prompt_json_examples(
        annotation_value={"A": [9, 9], "B": [8, 8], "C": [7, 7], "D": [6, 6]},
        answer_type="integer",
    )
    payload = json.loads(example_json)
    assert payload["annotation"] == {
        "A": [0, 0],
        "B": [4, 0],
        "C": [4, 2],
        "D": [0, 2],
    }


def test_dump_prompt_json_examples_uses_compact_answer_contract() -> None:
    answer_and_annotation, answer_only = dump_prompt_json_examples(
        annotation={"angle": [120, 180], "side": [220, 240]},
        answer="12π",
        ensure_ascii=False,
    )
    assert answer_and_annotation == '{"annotation":{"angle":[120,180],"side":[220,240]},"answer":"12π"}'
    assert answer_only == '{"answer":"12π"}'
