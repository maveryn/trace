"""Contract tests for symbolic Braille notation tasks."""

from __future__ import annotations

from trace_tasks.core.prompts import load_prompt_bundle
from trace_tasks.core.prompts.schema import REQUIRED_PROMPT_VARIANTS
from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.taxonomy import resolve_task_taxonomy
from trace_tasks.tasks.registry import TASK_REGISTRY
from trace_tasks.tasks.shared.word_assets import load_short_word_bank_by_length
from trace_tasks.tasks.symbolic.braille_cell.matching_pattern_label import (
    INTERNAL_QUERY_KEY as MATCHING_PATTERN_INTERNAL_QUERY_KEY,
)
from trace_tasks.tasks.symbolic.braille_cell.matching_pattern_label import (
    SCENE_ID,
)
from trace_tasks.tasks.symbolic.braille_cell.matching_pattern_label import (
    TASK_ID as MATCHING_PATTERN_TASK_ID,
)
from trace_tasks.tasks.symbolic.braille_cell.matching_pattern_label import SymbolicBrailleMatchingPatternLabelTask
from trace_tasks.tasks.symbolic.braille_cell.braille_word_read_label import (
    INTERNAL_QUERY_KEY as BRAILLE_WORD_READ_INTERNAL_QUERY_KEY,
)
from trace_tasks.tasks.symbolic.braille_cell.braille_word_read_label import (
    TASK_ID as BRAILLE_WORD_READ_TASK_ID,
)
from trace_tasks.tasks.symbolic.braille_cell.braille_word_read_label import SymbolicBrailleWordReadLabelTask
from trace_tasks.tasks.symbolic.braille_cell.word_braille_match_label import (
    INTERNAL_QUERY_KEY as WORD_BRAILLE_MATCH_INTERNAL_QUERY_KEY,
)
from trace_tasks.tasks.symbolic.braille_cell.word_braille_match_label import (
    TASK_ID as WORD_BRAILLE_MATCH_TASK_ID,
)
from trace_tasks.tasks.symbolic.braille_cell.word_braille_match_label import SymbolicBrailleWordMatchLabelTask
from trace_tasks.tasks.symbolic.braille_cell.shared.sampling import BRAILLE_WORD_BANK_BY_LENGTH


def _assert_same_prefix_options(metadata: dict) -> None:
    words = tuple(str(word) for word in metadata["option_words"].values())
    prefix = str(metadata["shared_prefix"])
    assert prefix
    assert int(metadata["shared_prefix_length"]) == len(prefix)
    assert int(metadata["shared_prefix_length"]) >= 1
    assert all(word.startswith(prefix) for word in words)
    assert int(metadata["prefix_candidate_count"]) >= len(words)
    assert str(metadata["word_option_sampling_strategy"]).startswith("same_prefix")


def test_braille_tasks_are_registered_and_taxonomized() -> None:
    assert TASK_REGISTRY[MATCHING_PATTERN_TASK_ID] is SymbolicBrailleMatchingPatternLabelTask
    assert TASK_REGISTRY[BRAILLE_WORD_READ_TASK_ID] is SymbolicBrailleWordReadLabelTask
    assert TASK_REGISTRY[WORD_BRAILLE_MATCH_TASK_ID] is SymbolicBrailleWordMatchLabelTask
    for task_id in (
        MATCHING_PATTERN_TASK_ID,
        BRAILLE_WORD_READ_TASK_ID,
        WORD_BRAILLE_MATCH_TASK_ID,
    ):
        task = TASK_REGISTRY[task_id]()
        taxonomy = resolve_task_taxonomy(task_id)
        assert task.domain == "symbolic"
        assert not hasattr(task, "scene_id")
        assert tuple(task.supported_query_ids) == (SINGLE_QUERY_ID,)
        assert taxonomy.domain == "symbolic"
        assert taxonomy.scene_id == SCENE_ID
        assert taxonomy.source_scene_id == ""


def test_braille_word_tasks_use_shared_short_word_assets() -> None:
    shared_bank = load_short_word_bank_by_length(min_length=3, max_length=5)
    assert BRAILLE_WORD_BANK_BY_LENGTH is shared_bank
    assert {length: len(words) for length, words in shared_bank.items()} == {3: 1000, 4: 1000, 5: 1000}
    assert {"amy", "john", "james"}.issubset({word for words in shared_bank.values() for word in words})


def test_braille_matching_pattern_contract() -> None:
    out = SymbolicBrailleMatchingPatternLabelTask().generate(
        2026060502,
        params={"correct_label": "C", "scene_variant": "notebook_card"},
        max_attempts=12,
    )
    trace = out.trace_payload
    option_patterns = trace["execution_trace"]["braille_metadata"]["option_patterns"]

    assert out.scene_id == SCENE_ID
    assert out.query_id == SINGLE_QUERY_ID
    assert trace["query_spec"]["internal_query_id"] == MATCHING_PATTERN_INTERNAL_QUERY_KEY
    assert trace["execution_trace"]["internal_query_id"] == MATCHING_PATTERN_INTERNAL_QUERY_KEY
    assert out.answer_gt.type == "string"
    assert out.answer_gt.value == "C"
    assert out.annotation_gt.type == "bbox_map"
    assert sorted(out.annotation_gt.value.keys()) == ["reference_cell", "selected_option"]
    assert len(option_patterns) == 6
    assert option_patterns["C"] == trace["execution_trace"]["braille_metadata"]["reference_pattern"]
    assert all(label in option_patterns for label in ("A", "B", "C", "D", "E", "F"))
    assert trace["render_map"]["annotation_source"] == "item_bboxes_px"
    assert trace["projected_annotation"]["bbox_map"] == out.annotation_gt.value
    assert trace["execution_trace"]["answer_value"] == out.answer_gt.value

    width, height = out.image.size
    for bbox in out.annotation_gt.value.values():
        assert len(bbox) == 4
        assert 0 <= float(bbox[0]) < float(bbox[2]) <= width
        assert 0 <= float(bbox[1]) < float(bbox[3]) <= height


def test_braille_word_read_contract() -> None:
    out = SymbolicBrailleWordReadLabelTask().generate(
        2026060503,
        params={
            "target_word": "james",
            "correct_label": "D",
            "scene_variant": "clean_card",
        },
        max_attempts=12,
    )
    trace = out.trace_payload
    metadata = trace["execution_trace"]["braille_metadata"]

    assert out.scene_id == SCENE_ID
    assert out.query_id == SINGLE_QUERY_ID
    assert trace["query_spec"]["internal_query_id"] == BRAILLE_WORD_READ_INTERNAL_QUERY_KEY
    assert trace["execution_trace"]["internal_query_id"] == BRAILLE_WORD_READ_INTERNAL_QUERY_KEY
    assert out.answer_gt.type == "string"
    assert out.answer_gt.value == "D"
    assert out.annotation_gt.type == "bbox_map"
    assert sorted(out.annotation_gt.value.keys()) == ["selected_option", "source_plate"]
    assert metadata["target_word"] == "james"
    assert metadata["word_length"] == 5
    assert metadata["option_words"]["D"] == "james"
    assert len(metadata["option_words"]) == 4
    assert {len(word) for word in metadata["option_words"].values()} == {5}
    _assert_same_prefix_options(metadata)
    assert trace["render_map"]["annotation_source"] == "item_bboxes_px"
    assert trace["projected_annotation"]["bbox_map"] == out.annotation_gt.value
    assert trace["execution_trace"]["answer_value"] == out.answer_gt.value

    width, height = out.image.size
    for bbox in out.annotation_gt.value.values():
        assert len(bbox) == 4
        assert 0 <= float(bbox[0]) < float(bbox[2]) <= width
        assert 0 <= float(bbox[1]) < float(bbox[3]) <= height


def test_word_braille_match_contract() -> None:
    out = SymbolicBrailleWordMatchLabelTask().generate(
        2026060504,
        params={
            "target_word": "leaf",
            "correct_label": "B",
            "scene_variant": "exam_scan",
        },
        max_attempts=12,
    )
    trace = out.trace_payload
    metadata = trace["execution_trace"]["braille_metadata"]

    assert out.scene_id == SCENE_ID
    assert out.query_id == SINGLE_QUERY_ID
    assert trace["query_spec"]["internal_query_id"] == WORD_BRAILLE_MATCH_INTERNAL_QUERY_KEY
    assert trace["execution_trace"]["internal_query_id"] == WORD_BRAILLE_MATCH_INTERNAL_QUERY_KEY
    assert out.answer_gt.type == "string"
    assert out.answer_gt.value == "B"
    assert out.annotation_gt.type == "bbox_map"
    assert sorted(out.annotation_gt.value.keys()) == ["selected_option", "source_word"]
    assert metadata["source_word"] == "leaf"
    assert metadata["word_length"] == 4
    assert metadata["option_words"]["B"] == "leaf"
    assert metadata["option_patterns"]["B"] == metadata["source_patterns"]
    assert len(metadata["option_words"]) == 4
    assert {len(word) for word in metadata["option_words"].values()} == {4}
    _assert_same_prefix_options(metadata)
    assert trace["render_map"]["annotation_source"] == "item_bboxes_px"
    assert trace["projected_annotation"]["bbox_map"] == out.annotation_gt.value
    assert trace["execution_trace"]["answer_value"] == out.answer_gt.value

    width, height = out.image.size
    for bbox in out.annotation_gt.value.values():
        assert len(bbox) == 4
        assert 0 <= float(bbox[0]) < float(bbox[2]) <= width
        assert 0 <= float(bbox[1]) < float(bbox[3]) <= height


def test_braille_generation_is_deterministic() -> None:
    params = {"scene_variant": "exam_scan", "correct_label": "D", "reference_raised_dot_count": 3}
    out_a = SymbolicBrailleMatchingPatternLabelTask().generate(2026060599, params=params, max_attempts=12)
    out_b = SymbolicBrailleMatchingPatternLabelTask().generate(2026060599, params=params, max_attempts=12)

    assert out_a.prompt == out_b.prompt
    assert out_a.answer_gt == out_b.answer_gt
    assert out_a.annotation_gt == out_b.annotation_gt
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.image.tobytes() == out_b.image.tobytes()


def test_symbolic_notation_prompt_bundle_supports_braille_queries() -> None:
    bundle = load_prompt_bundle("symbolic", "braille_cell", "symbolic_braille_cell_v1")
    assert "braille_cell" in bundle.scene_templates
    assert len(bundle.task_templates["braille_matching_pattern_label"]) == REQUIRED_PROMPT_VARIANTS
    assert len(bundle.task_templates["braille_word_read_label"]) == REQUIRED_PROMPT_VARIANTS
    assert len(bundle.task_templates["word_braille_match_label"]) == REQUIRED_PROMPT_VARIANTS
    assert not bundle.query_templates
    assert list(bundle.required_slots_by_key["scene:braille_cell"]) == ["object_description"]
    assert list(bundle.required_slots_by_key["task:braille_matching_pattern_label"]) == []
    assert list(bundle.required_slots_by_key["task:braille_word_read_label"]) == []
    assert list(bundle.required_slots_by_key["task:word_braille_match_label"]) == []
