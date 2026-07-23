"""Contract tests for symbolic Morse-code notation tasks."""

from __future__ import annotations

from trace_tasks.core.prompts import load_prompt_bundle
from trace_tasks.core.prompts.schema import REQUIRED_PROMPT_VARIANTS
from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.taxonomy import resolve_task_taxonomy
from trace_tasks.tasks.registry import TASK_REGISTRY
from trace_tasks.tasks.shared.word_assets import load_short_word_bank_by_length
from trace_tasks.tasks.symbolic.morse_code.morse_word_read_label import (
    INTERNAL_QUERY_KEY as MORSE_WORD_READ_INTERNAL_QUERY_KEY,
)
from trace_tasks.tasks.symbolic.morse_code.morse_word_read_label import (
    TASK_ID as MORSE_WORD_READ_TASK_ID,
)
from trace_tasks.tasks.symbolic.morse_code.morse_word_read_label import SymbolicMorseWordReadLabelTask
from trace_tasks.tasks.symbolic.morse_code.word_morse_match_label import (
    INTERNAL_QUERY_KEY as WORD_MORSE_MATCH_INTERNAL_QUERY_KEY,
)
from trace_tasks.tasks.symbolic.morse_code.word_morse_match_label import (
    TASK_ID as WORD_MORSE_MATCH_TASK_ID,
)
from trace_tasks.tasks.symbolic.morse_code.word_morse_match_label import SymbolicMorseWordMatchLabelTask
from trace_tasks.tasks.symbolic.morse_code.shared.sampling import MORSE_WORD_BANK_BY_LENGTH


def _assert_bbox_in_image(bbox: list[float], *, width: int, height: int) -> None:
    assert len(bbox) == 4
    assert 0 <= float(bbox[0]) < float(bbox[2]) <= width
    assert 0 <= float(bbox[1]) < float(bbox[3]) <= height


def _assert_same_prefix_options(metadata: dict) -> None:
    words = tuple(str(word) for word in metadata["option_words"].values())
    prefix = str(metadata["shared_prefix"])
    assert prefix
    assert int(metadata["shared_prefix_length"]) == len(prefix)
    assert int(metadata["shared_prefix_length"]) >= 1
    assert all(word.startswith(prefix) for word in words)
    assert int(metadata["prefix_candidate_count"]) >= len(words)
    assert str(metadata["word_option_sampling_strategy"]).startswith("same_prefix")


def test_morse_code_tasks_are_registered_and_taxonomized() -> None:
    assert TASK_REGISTRY[MORSE_WORD_READ_TASK_ID] is SymbolicMorseWordReadLabelTask
    assert TASK_REGISTRY[WORD_MORSE_MATCH_TASK_ID] is SymbolicMorseWordMatchLabelTask
    for task_id in (MORSE_WORD_READ_TASK_ID, WORD_MORSE_MATCH_TASK_ID):
        task = TASK_REGISTRY[task_id]()
        taxonomy = resolve_task_taxonomy(task_id)
        assert task.domain == "symbolic"
        assert not hasattr(task, "scene_id")
        assert tuple(task.supported_query_ids) == (SINGLE_QUERY_ID,)
        assert taxonomy.domain == "symbolic"
        assert taxonomy.scene_id == "morse_code"
        assert taxonomy.source_scene_id == ""


def test_morse_word_tasks_use_shared_short_word_assets() -> None:
    shared_bank = load_short_word_bank_by_length(min_length=3, max_length=5)
    assert MORSE_WORD_BANK_BY_LENGTH is shared_bank
    assert {length: len(words) for length, words in shared_bank.items()} == {3: 1000, 4: 1000, 5: 1000}
    assert {"amy", "john", "james"}.issubset({word for words in shared_bank.values() for word in words})


def test_morse_word_read_contract() -> None:
    out = SymbolicMorseWordReadLabelTask().generate(
        2026062501,
        params={"target_word": "james", "correct_label": "D", "scene_variant": "clean_card"},
        max_attempts=12,
    )
    trace = out.trace_payload
    metadata = trace["execution_trace"]["morse_metadata"]

    assert out.scene_id == "morse_code"
    assert out.query_id == SINGLE_QUERY_ID
    assert trace["query_spec"]["internal_query_id"] == MORSE_WORD_READ_INTERNAL_QUERY_KEY
    assert trace["execution_trace"]["internal_query_id"] == MORSE_WORD_READ_INTERNAL_QUERY_KEY
    assert out.answer_gt.type == "string"
    assert out.answer_gt.value == "D"
    assert out.annotation_gt.type == "bbox_map"
    assert sorted(out.annotation_gt.value.keys()) == ["selected_option", "source_code"]
    assert metadata["target_word"] == "james"
    assert metadata["word_length"] == 5
    assert metadata["option_words"]["D"] == "james"
    assert metadata["target_codes"] == [".---", ".-", "--", ".", "..."]
    assert len(metadata["option_words"]) == 4
    assert {len(word) for word in metadata["option_words"].values()} == {5}
    _assert_same_prefix_options(metadata)
    assert trace["render_map"]["annotation_source"] == "item_bboxes_px"
    assert trace["projected_annotation"]["bbox_map"] == out.annotation_gt.value
    assert trace["execution_trace"]["answer_value"] == out.answer_gt.value

    width, height = out.image.size
    for bbox in out.annotation_gt.value.values():
        _assert_bbox_in_image(bbox, width=width, height=height)


def test_word_morse_match_contract() -> None:
    out = SymbolicMorseWordMatchLabelTask().generate(
        2026062502,
        params={"target_word": "leaf", "correct_label": "B", "scene_variant": "exam_scan"},
        max_attempts=12,
    )
    trace = out.trace_payload
    metadata = trace["execution_trace"]["morse_metadata"]

    assert out.scene_id == "morse_code"
    assert out.query_id == SINGLE_QUERY_ID
    assert trace["query_spec"]["internal_query_id"] == WORD_MORSE_MATCH_INTERNAL_QUERY_KEY
    assert trace["execution_trace"]["internal_query_id"] == WORD_MORSE_MATCH_INTERNAL_QUERY_KEY
    assert out.answer_gt.type == "string"
    assert out.answer_gt.value == "B"
    assert out.annotation_gt.type == "bbox_map"
    assert sorted(out.annotation_gt.value.keys()) == ["selected_option", "source_word"]
    assert metadata["source_word"] == "leaf"
    assert metadata["word_length"] == 4
    assert metadata["option_words"]["B"] == "leaf"
    assert metadata["option_codes"]["B"] == metadata["source_codes"]
    assert metadata["source_codes"] == [".-..", ".", ".-", "..-."]
    assert len(metadata["option_words"]) == 4
    assert {len(word) for word in metadata["option_words"].values()} == {4}
    _assert_same_prefix_options(metadata)
    assert trace["render_map"]["annotation_source"] == "item_bboxes_px"
    assert trace["projected_annotation"]["bbox_map"] == out.annotation_gt.value
    assert trace["execution_trace"]["answer_value"] == out.answer_gt.value

    width, height = out.image.size
    for bbox in out.annotation_gt.value.values():
        _assert_bbox_in_image(bbox, width=width, height=height)


def test_morse_generation_is_deterministic() -> None:
    params = {"scene_variant": "notebook_card", "target_word": "emily", "correct_label": "D"}
    out_a = SymbolicMorseWordReadLabelTask().generate(2026062599, params=params, max_attempts=12)
    out_b = SymbolicMorseWordReadLabelTask().generate(2026062599, params=params, max_attempts=12)

    assert out_a.prompt == out_b.prompt
    assert out_a.answer_gt == out_b.answer_gt
    assert out_a.annotation_gt == out_b.annotation_gt
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.image.tobytes() == out_b.image.tobytes()


def test_symbolic_morse_code_prompt_bundle_supports_word_tasks() -> None:
    bundle = load_prompt_bundle("symbolic", "morse_code", "symbolic_morse_code_v1")
    assert "morse_code" in bundle.scene_templates
    assert len(bundle.task_templates["morse_word_read_label"]) == REQUIRED_PROMPT_VARIANTS
    assert len(bundle.task_templates["word_morse_match_label"]) == REQUIRED_PROMPT_VARIANTS
    assert not bundle.query_templates
    assert list(bundle.required_slots_by_key["scene:morse_code"]) == ["object_description"]
    assert list(bundle.required_slots_by_key["task:morse_word_read_label"]) == []
    assert list(bundle.required_slots_by_key["task:word_morse_match_label"]) == []
