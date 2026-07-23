"""Behavior tests for elapsed-time and sequence-completion clock tasks."""

from __future__ import annotations

from collections import Counter

from trace_tasks.core.seed import hash64
from trace_tasks.tasks.shared.time_format import add_clock_minutes, format_clock_hhmm
from trace_tasks.tasks.symbolic.clock.elapsed_time_value import SymbolicClockElapsedTimeValueTask
from trace_tasks.tasks.symbolic.clock.sequence_completion_label import SymbolicClockSequenceCompletionLabelTask
from tests.helpers import extract_prompt_json_example


def test_symbolic_clock_elapsed_time_contract_matches_trace() -> None:
    task = SymbolicClockElapsedTimeValueTask()
    out = task.generate(
        20800,
        params={
            "scene_variant": "classic",
            "style_variant": "marker",
            "accent_color_name": "blue",
            "start_hour": 2,
            "start_minute": 15,
            "elapsed_minutes": 75,
        },
        max_attempts=20,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]

    assert out.query_id == "single"
    assert out.answer_gt.type == "option_letter"
    assert out.annotation_gt.type == "bbox"
    assert trace["scene_ir"]["scene_kind"] == "symbolic_clock_elapsed_time_pair"
    assert int(execution["elapsed_minutes"]) == 75
    assert int(execution["answer_value"]) == 75
    assert int(execution["option_values_by_label"][out.answer_gt.value]) == 75
    assert str(execution["start_time_text"]) == "02:15"
    assert str(execution["end_time_text"]) == str(format_clock_hhmm(add_clock_minutes(int(execution["start_total_minutes"]), 75)))
    assert trace["projected_annotation"]["bbox"] == out.annotation_gt.value
    assert trace["projected_annotation"]["pixel_bbox"] == out.annotation_gt.value
    assert trace["render_map"]["selected_option_bbox_px"] == out.annotation_gt.value
    assert set(execution["source_clock_bboxes_px"].keys()) == {"start_clock", "end_clock"}
    assert trace["render_spec"]["clock_style"]["font"]["source"] == "global_font_pool"


def test_symbolic_clock_elapsed_time_prompt_examples_match_contract() -> None:
    task = SymbolicClockElapsedTimeValueTask()
    out = task.generate(20810, params={}, max_attempts=20)
    answer_and_annotation = extract_prompt_json_example(out.prompt_variants["answer_and_annotation"])
    answer_only = extract_prompt_json_example(out.prompt_variants["answer_only"])
    assert answer_and_annotation == {
        "annotation": [304, 670, 436, 736],
        "answer": "C",
    }
    assert answer_only == {"answer": "C"}


def test_symbolic_clock_elapsed_time_sampling_covers_answers_and_styles() -> None:
    task = SymbolicClockElapsedTimeValueTask()
    durations: Counter[int] = Counter()
    scene_variants: Counter[str] = Counter()
    style_variants: Counter[str] = Counter()
    accents: Counter[str] = Counter()
    for index in range(90):
        out = task.generate(
            hash64(20820, "symbolic_clock_elapsed_time", index),
            params={},
            max_attempts=20,
        )
        execution = out.trace_payload["execution_trace"]
        durations[int(execution["elapsed_minutes"])] += 1
        scene_variants[str(execution["scene_variant"])] += 1
        style_variants[str(execution["style_variant"])] += 1
        accents[str(execution["accent_color_name"])] += 1

    assert len(durations) >= 8
    assert set(scene_variants.keys()) == {"classic", "minimal", "outline"}
    assert set(style_variants.keys()) == {"accented", "marker", "studio"}
    assert len(accents) >= 8


def test_symbolic_clock_sequence_completion_contract_matches_trace() -> None:
    task = SymbolicClockSequenceCompletionLabelTask()
    out = task.generate(
        20840,
        params={
            "scene_variant": "minimal",
            "style_variant": "accented",
            "accent_color_name": "green",
            "sequence_step_minutes": 30,
            "missing_slot_index": 2,
            "answer_label": "C",
        },
        max_attempts=20,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]

    assert out.query_id == "single"
    assert out.answer_gt.type == "string"
    assert out.answer_gt.value == "C"
    assert out.annotation_gt.type == "bbox_map"
    assert set(out.annotation_gt.value.keys()) == {"sequence_panel", "correct_option"}
    assert trace["scene_ir"]["scene_kind"] == "symbolic_clock_sequence_completion_panel"
    assert int(execution["missing_slot_index"]) == 2
    assert int(execution["sequence_step_minutes"]) == 30
    assert str(execution["correct_label"]) == "C"
    assert str(execution["option_time_text_by_label"]["C"]) == str(execution["missing_time_text"])
    assert len(execution["visible_sequence_time_texts"]) == 4
    assert execution["visible_sequence_time_texts"][2] is None
    assert len(set(execution["option_time_text_by_label"].values())) == 4
    assert trace["projected_annotation"]["bbox_map"] == out.annotation_gt.value
    assert trace["render_map"]["correct_option_bbox_px"] == out.annotation_gt.value["correct_option"]
    assert trace["render_spec"]["clock_style"]["font"]["source"] == "global_font_pool"


def test_symbolic_clock_sequence_completion_prompt_examples_match_contract() -> None:
    task = SymbolicClockSequenceCompletionLabelTask()
    out = task.generate(20850, params={}, max_attempts=20)
    answer_and_annotation = extract_prompt_json_example(out.prompt_variants["answer_and_annotation"])
    answer_only = extract_prompt_json_example(out.prompt_variants["answer_only"])
    assert answer_and_annotation == {
        "annotation": {
            "sequence_panel": [50, 70, 930, 290],
            "correct_option": [275, 430, 480, 650],
        },
        "answer": "B",
    }
    assert answer_only == {"answer": "B"}


def test_symbolic_clock_sequence_completion_sampling_covers_internal_axes() -> None:
    task = SymbolicClockSequenceCompletionLabelTask()
    missing_slots: Counter[int] = Counter()
    answer_labels: Counter[str] = Counter()
    steps: Counter[int] = Counter()
    scene_variants: Counter[str] = Counter()
    for index in range(120):
        out = task.generate(
            hash64(20860, "symbolic_clock_sequence_completion", index),
            params={},
            max_attempts=20,
        )
        execution = out.trace_payload["execution_trace"]
        missing_slots[int(execution["missing_slot_index"])] += 1
        answer_labels[str(execution["correct_label"])] += 1
        steps[int(execution["sequence_step_minutes"])] += 1
        scene_variants[str(execution["scene_variant"])] += 1
        assert len(execution["option_labels"]) == 4
        assert len(set(execution["option_time_text_by_label"].values())) == 4

    assert set(missing_slots.keys()) == {0, 1, 2, 3}
    assert set(answer_labels.keys()) == {"A", "B", "C", "D"}
    assert set(steps.keys()) == {15, 30, 45, 60, 90}
    assert set(scene_variants.keys()) == {"classic", "minimal", "outline"}
