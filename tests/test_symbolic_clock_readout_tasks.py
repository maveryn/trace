"""Behavior tests for clock-readout task."""

from __future__ import annotations

from collections import Counter

import pytest

from trace_tasks.core.seed import hash64
from trace_tasks.tasks.symbolic.clock.alarm_wait_time_value import (
    ALARM_HAND_COLOR_RGB,
    SymbolicClockAlarmWaitTimeValueTask,
)
from trace_tasks.tasks.symbolic.clock.full_time_readout import SymbolicClockFullTimeReadoutTask
from trace_tasks.tasks.symbolic.clock.hand_angle_value import SymbolicClockHandAngleValueTask
from trace_tasks.tasks.symbolic.clock.offset_readout import SymbolicClockOffsetReadoutTask
from trace_tasks.tasks.shared.time_artifact_style import (
    SUPPORTED_TIME_ARTIFACT_CLOCK_COLOR_NAMES,
    SUPPORTED_TIME_ARTIFACT_CLOCK_STYLE_VARIANTS,
)
from trace_tasks.tasks.shared.time_format import (
    add_clock_minutes,
    clock_hand_angle_gap_deg,
    clock_hand_pair_angle_gaps_deg,
    clock_total_minutes,
    clock_total_seconds,
    format_clock_hhmm,
    format_clock_hhmmss,
)
from tests.helpers import extract_prompt_json_example


def test_symbolic_clock_readout_contract_matches_trace() -> None:
    task_cases = (
        (SymbolicClockOffsetReadoutTask(), "minutes_after", "after"),
        (SymbolicClockOffsetReadoutTask(), "minutes_before", "before"),
    )
    scene_variants = ("classic", "minimal", "outline")
    style_variants = ("studio", "accented", "marker")
    accent_colors = ("blue", "orange", "magenta")
    for query_id_index, (task, expected_query_id, expected_direction) in enumerate(task_cases):
        for scene_index, scene_variant in enumerate(scene_variants):
            seed = 20300 + (query_id_index * 20) + scene_index
            out = task.generate(
                seed,
                params={
                    "query_id": expected_query_id,
                    "scene_variant": scene_variant,
                    "style_variant": style_variants[scene_index],
                    "accent_color_name": accent_colors[scene_index],
                    "delta_minutes": 25,
                },
                max_attempts=20,
            )
            trace = out.trace_payload
            execution = trace["execution_trace"]
            scene_entities = trace["scene_ir"]["entities"]
            hand_entities = [entity for entity in scene_entities if entity["entity_kind"] == "clock_hand"]

            assert out.answer_gt.type == "option_letter"
            assert out.annotation_gt.type == "bbox"
            assert sorted(out.prompt_variants.keys()) == ["answer_and_annotation", "answer_only"]
            expected_hand_count = 2
            assert trace["scene_ir"]["scene_kind"] == "symbolic_clock_single"
            assert out.query_id == expected_query_id
            assert str(execution["query_id"]) == out.query_id
            assert str(execution["offset_unit"]) == "minutes"
            assert str(execution["offset_direction"]) == expected_direction
            assert str(execution["scene_variant"]) == str(scene_variant)
            assert str(execution["style_variant"]) == str(style_variants[scene_index])
            assert str(execution["accent_color_name"]) == str(accent_colors[scene_index])
            assert len(hand_entities) == expected_hand_count
            shown_total_minutes = int(execution["shown_total_minutes"])
            shown_text = str(format_clock_hhmm(int(shown_total_minutes)))
            assert str(execution["shown_time_text"]) == shown_text
            assert trace["projected_annotation"]["bbox"] == out.annotation_gt.value
            assert trace["projected_annotation"]["pixel_bbox"] == out.annotation_gt.value
            assert len(execution["supporting_segments"]) == 2
            assert len(trace["render_map"]["hand_bboxes_px"]) == expected_hand_count
            assert trace["render_map"]["annotation_source"] == "selected_answer_option_bbox_px"
            assert execution["selected_option_bbox_px"] == out.annotation_gt.value
            assert trace["render_map"]["selected_option_bbox_px"] == out.annotation_gt.value
            assert execution["answer_label"] == out.answer_gt.value
            assert trace["render_spec"]["clock_style"]["font"]["source"] == "global_font_pool"
            assert trace["render_spec"]["clock_style"]["font"]["font_family"]
            assert str(trace["render_spec"]["clock_style"]["accent_color_name"]) == str(accent_colors[scene_index])
            assert str(trace["render_spec"]["clock_style"]["style_variant"]) == str(style_variants[scene_index])
            assert isinstance(trace["render_spec"]["clock_style"]["resolved_colors_rgb"], dict)
            assert int(execution["minute_support"][2]) == 5
            assert float(execution["min_hand_angle_gap_deg"]) == pytest.approx(10.0)

            if expected_direction == "after":
                expected = format_clock_hhmm(add_clock_minutes(int(shown_total_minutes), 25))
                assert str(execution["answer_value"]) == str(expected)
                assert str(execution["option_text_by_label"][out.answer_gt.value]) == str(expected)
                assert int(execution["delta_minutes"]) == 25
            else:
                expected = format_clock_hhmm(add_clock_minutes(int(shown_total_minutes), -25))
                assert str(execution["answer_value"]) == str(expected)
                assert str(execution["option_text_by_label"][out.answer_gt.value]) == str(expected)
                assert int(execution["delta_minutes"]) == 25


def test_symbolic_clock_prompt_examples_match_variant_offsets() -> None:
    option_example = [224, 770, 316, 836]
    expected = (
        (SymbolicClockOffsetReadoutTask(), "minutes_after", (
            {"annotation": option_example, "answer": "C"},
            {"answer": "C"},
        )),
        (SymbolicClockOffsetReadoutTask(), "minutes_before", (
            {"annotation": option_example, "answer": "C"},
            {"answer": "C"},
        )),
    )
    for index, (task, query_id, (expected_answer_and_annotation, expected_answer_only)) in enumerate(expected, start=20340):
        out = task.generate(
            index,
            params={"query_id": query_id, "delta_minutes": 25},
            max_attempts=20,
        )
        answer_and_annotation = extract_prompt_json_example(out.prompt_variants["answer_and_annotation"])
        answer_only = extract_prompt_json_example(out.prompt_variants["answer_only"])
        assert answer_and_annotation == expected_answer_and_annotation
        assert answer_only == expected_answer_only


def test_symbolic_clock_balanced_sampling_defaults_cover_public_axes() -> None:
    task = SymbolicClockOffsetReadoutTask()
    query_ids: Counter[str] = Counter()
    scene_variants: Counter[str] = Counter()
    style_variants: Counter[str] = Counter()
    accent_color_names: Counter[str] = Counter()
    for index in range(60):
        out = task.generate(
            hash64(20380, task.task_id, index),
            params={},
            max_attempts=20,
        )
        execution = out.trace_payload["execution_trace"]
        query_ids[str(execution["query_id"])] += 1
        scene_variants[str(execution["scene_variant"])] += 1
        style_variants[str(execution["style_variant"])] += 1
        accent_color_names[str(execution["accent_color_name"])] += 1
    assert set(query_ids.keys()) == {"minutes_after", "minutes_before"}
    assert set(scene_variants.keys()) == {"classic", "minimal", "outline"}
    assert set(style_variants.keys()) == set(SUPPORTED_TIME_ARTIFACT_CLOCK_STYLE_VARIANTS)
    assert set(accent_color_names.keys()) == set(SUPPORTED_TIME_ARTIFACT_CLOCK_COLOR_NAMES)


def test_symbolic_clock_marker_style_uses_dot_minor_ticks() -> None:
    task = SymbolicClockOffsetReadoutTask()
    out = task.generate(
        20400,
        params={
            "query_id": "minutes_after",
            "delta_minutes": 25,
            "scene_variant": "classic",
            "style_variant": "marker",
            "accent_color_name": "cyan",
        },
        max_attempts=20,
    )
    clock_style = out.trace_payload["render_spec"]["clock_style"]
    assert str(clock_style["style_variant"]) == "marker"
    assert str(clock_style["minor_tick_mode"]) == "dot"
    assert list(clock_style["resolved_colors_rgb"]["minute_hand"]) != list(clock_style["resolved_colors_rgb"]["hour_hand"])


def test_symbolic_clock_hand_angle_contract_matches_trace() -> None:
    task = SymbolicClockHandAngleValueTask()
    out = task.generate(
        20410,
        params={
            "shown_hour": 3,
            "shown_minute": 0,
            "scene_variant": "classic",
            "style_variant": "accented",
            "accent_color_name": "green",
        },
        max_attempts=20,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]
    shown_total_minutes = int(execution["shown_total_minutes"])

    assert out.answer_gt.type == "option_letter"
    assert out.annotation_gt.type == "bbox"
    assert out.query_id == "single"
    assert trace["scene_ir"]["scene_kind"] == "symbolic_clock_single"
    assert str(execution["shown_time_text"]) == "03:00"
    assert int(execution["hand_angle_deg"]) == int(round(clock_hand_angle_gap_deg(shown_total_minutes)))
    assert int(execution["answer_value"]) == int(execution["hand_angle_deg"])
    assert int(execution["answer_value"]) == 90
    assert str(execution["option_values_by_label"][out.answer_gt.value]) == str(execution["answer_value"])
    assert trace["projected_annotation"]["bbox"] == out.annotation_gt.value
    assert trace["projected_annotation"]["pixel_bbox"] == out.annotation_gt.value
    assert trace["render_map"]["annotation_source"] == "selected_answer_option_bbox_px"
    assert len(trace["render_map"]["hand_tips_px"]) == 2
    assert len(execution["supporting_segments"]) == 2


def test_symbolic_clock_hand_angle_prompt_examples_match_contract() -> None:
    out = SymbolicClockHandAngleValueTask().generate(
        20415,
        params={"shown_hour": 3, "shown_minute": 0},
        max_attempts=20,
    )
    answer_and_annotation = extract_prompt_json_example(out.prompt_variants["answer_and_annotation"])
    answer_only = extract_prompt_json_example(out.prompt_variants["answer_only"])
    assert answer_and_annotation == {
        "annotation": [224, 770, 316, 836],
        "answer": "C",
    }
    assert answer_only == {"answer": "C"}


def test_symbolic_clock_hand_angle_sampling_covers_integer_angles() -> None:
    task = SymbolicClockHandAngleValueTask()
    answers: Counter[int] = Counter()
    scene_variants: Counter[str] = Counter()
    for index in range(80):
        out = task.generate(
            hash64(20416, task.task_id, index),
            params={},
            max_attempts=20,
        )
        execution = out.trace_payload["execution_trace"]
        answer = int(execution["answer_value"])
        answers[answer] += 1
        scene_variants[str(execution["scene_variant"])] += 1
        assert 15 <= answer <= 180
        assert answer % 5 == 0
        assert answer == int(execution["hand_angle_deg"])
        assert out.answer_gt.type == "option_letter"
        assert out.annotation_gt.type == "bbox"
    assert len(answers) >= 8
    assert set(scene_variants.keys()) == {"classic", "minimal", "outline"}


def test_symbolic_clock_full_time_readout_contract_matches_trace() -> None:
    task = SymbolicClockFullTimeReadoutTask()
    shown_total_seconds = clock_total_seconds(3, 25, 40)
    out = task.generate(
        20430,
        params={
            "shown_hour": 3,
            "shown_minute": 25,
            "shown_second": 40,
            "scene_variant": "classic",
            "style_variant": "marker",
            "accent_color_name": "blue",
        },
        max_attempts=20,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]
    hand_entities = [
        entity
        for entity in trace["scene_ir"]["entities"]
        if entity["entity_kind"] == "clock_hand"
    ]

    assert out.answer_gt.type == "option_letter"
    assert out.annotation_gt.type == "bbox"
    assert out.query_id == "single"
    assert trace["scene_ir"]["scene_kind"] == "symbolic_clock_single"
    assert len(hand_entities) == 3
    assert {entity["attrs"]["hand_kind"] for entity in hand_entities} == {
        "hour",
        "minute",
        "second",
    }
    assert int(execution["shown_total_seconds"]) == shown_total_seconds
    assert str(execution["shown_time_text"]) == "03:25:40"
    assert str(execution["answer_value"]) == format_clock_hhmmss(shown_total_seconds)
    assert str(execution["option_text_by_label"][out.answer_gt.value]) == "03:25:40"
    assert trace["projected_annotation"]["bbox"] == out.annotation_gt.value
    assert trace["projected_annotation"]["pixel_bbox"] == out.annotation_gt.value
    assert len(execution["supporting_segments"]) == 3
    assert execution["supporting_parts"] == ["selected_answer_option"]
    assert len(trace["render_map"]["hand_bboxes_px"]) == 3
    assert len(trace["render_map"]["hand_tips_px"]) == 3
    assert bool(trace["render_spec"]["clock_style"]["show_second_hand"]) is True
    assert min(execution["hand_angle_gaps_deg"]) >= 15.0
    assert [
        round(float(value), 6)
        for value in clock_hand_pair_angle_gaps_deg(shown_total_seconds)
    ] == execution["hand_angle_gaps_deg"]


def test_symbolic_clock_full_time_readout_prompt_examples_match_contract() -> None:
    out = SymbolicClockFullTimeReadoutTask().generate(
        20431,
        params={"shown_hour": 3, "shown_minute": 25, "shown_second": 40},
        max_attempts=20,
    )
    answer_and_annotation = extract_prompt_json_example(
        out.prompt_variants["answer_and_annotation"]
    )
    answer_only = extract_prompt_json_example(out.prompt_variants["answer_only"])
    assert answer_and_annotation == {
        "annotation": [224, 770, 316, 836],
        "answer": "C",
    }
    assert answer_only == {"answer": "C"}


def test_symbolic_clock_full_time_readout_sampling_covers_seconds_and_styles() -> None:
    task = SymbolicClockFullTimeReadoutTask()
    seconds: Counter[int] = Counter()
    scene_variants: Counter[str] = Counter()
    for index in range(80):
        out = task.generate(
            hash64(20432, task.task_id, index),
            params={},
            max_attempts=20,
        )
        execution = out.trace_payload["execution_trace"]
        seconds[int(execution["shown_second"])] += 1
        scene_variants[str(execution["scene_variant"])] += 1
        assert out.answer_gt.type == "option_letter"
        assert out.annotation_gt.type == "bbox"
        assert int(execution["shown_second"]) % 5 == 0
    assert len(seconds) >= 6
    assert set(scene_variants.keys()) == {"classic", "minimal", "outline"}


def test_symbolic_clock_alarm_wait_time_contract_matches_trace() -> None:
    task = SymbolicClockAlarmWaitTimeValueTask()
    out = task.generate(
        20450,
        params={
            "shown_hour": 3,
            "shown_minute": 25,
            "alarm_hour": 7,
            "scene_variant": "classic",
            "style_variant": "accented",
            "accent_color_name": "blue",
        },
        max_attempts=20,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]
    hand_entities = [
        entity
        for entity in trace["scene_ir"]["entities"]
        if entity["entity_kind"] == "clock_hand"
    ]
    hand_kinds = {str(entity["attrs"]["hand_kind"]) for entity in hand_entities}
    clock_colors = trace["render_spec"]["clock_style"]["resolved_colors_rgb"]
    prompt_lower = str(out.prompt).lower()

    assert out.answer_gt.type == "option_letter"
    assert out.annotation_gt.type == "bbox"
    assert out.query_id == "single"
    assert trace["scene_ir"]["scene_kind"] == "symbolic_clock_alarm"
    assert hand_kinds == {"hour", "minute", "alarm"}
    assert str(execution["shown_time_text"]) == "03:25"
    assert int(execution["alarm_hour"]) == 7
    assert str(execution["alarm_time_text"]) == "07:00"
    assert int(execution["wait_minutes"]) == 215
    assert int(execution["answer_value"]) == 215
    assert str(execution["option_text_by_label"][out.answer_gt.value]) == "215"
    assert str(execution["alarm_hand_scale"]) == "hour"
    assert min(float(value) for value in execution["alarm_hand_angle_gaps_deg"]) >= 20.0
    assert float(execution["current_hand_angle_gap_deg"]) >= 10.0
    assert list(clock_colors["alarm_hand"]) == list(ALARM_HAND_COLOR_RGB)
    assert list(clock_colors["hour_hand"]) != list(ALARM_HAND_COLOR_RGB)
    assert list(clock_colors["minute_hand"]) != list(ALARM_HAND_COLOR_RGB)
    assert trace["projected_annotation"]["bbox"] == out.annotation_gt.value
    assert trace["projected_annotation"]["pixel_bbox"] == out.annotation_gt.value
    assert len(trace["render_map"]["hand_bboxes_px"]) == 3
    assert len(trace["render_map"]["hand_tips_px"]) == 3
    assert execution["supporting_parts"] == ["selected_answer_option"]
    assert "red alarm hand" in prompt_lower
    assert ":00" in out.prompt
    assert "hour scale" in prompt_lower or "hour-scale" in prompt_lower


def test_symbolic_clock_alarm_wait_time_prompt_examples_match_contract() -> None:
    out = SymbolicClockAlarmWaitTimeValueTask().generate(
        20451,
        params={"shown_hour": 3, "shown_minute": 25, "alarm_hour": 7},
        max_attempts=20,
    )
    answer_and_annotation = extract_prompt_json_example(out.prompt_variants["answer_and_annotation"])
    answer_only = extract_prompt_json_example(out.prompt_variants["answer_only"])
    assert answer_and_annotation == {
        "annotation": [224, 770, 316, 836],
        "answer": "C",
    }
    assert answer_only == {"answer": "C"}


def test_symbolic_clock_alarm_wait_time_sampling_respects_visual_constraints() -> None:
    task = SymbolicClockAlarmWaitTimeValueTask()
    alarm_hours: Counter[int] = Counter()
    answers: Counter[int] = Counter()
    accent_colors: Counter[str] = Counter()
    for index in range(80):
        out = task.generate(
            hash64(20452, task.task_id, index),
            params={},
            max_attempts=20,
        )
        execution = out.trace_payload["execution_trace"]
        alarm_hours[int(execution["alarm_hour"])] += 1
        answers[int(execution["answer_value"])] += 1
        accent_colors[str(execution["accent_color_name"])] += 1
        assert out.answer_gt.type == "option_letter"
        assert out.annotation_gt.type == "bbox"
        assert 1 <= int(execution["answer_value"]) <= 720
        assert int(execution["answer_value"]) == int(execution["wait_minutes"])
        assert int(execution["alarm_minute"]) == 0
        assert str(execution["alarm_hand_scale"]) == "hour"
        assert min(float(value) for value in execution["alarm_hand_angle_gaps_deg"]) >= float(execution["min_alarm_hand_gap_deg"])
        assert float(execution["current_hand_angle_gap_deg"]) >= float(execution["min_hand_angle_gap_deg"])
    assert len(alarm_hours) >= 6
    assert len(answers) >= 10
    assert not (set(accent_colors) & {"red", "orange", "magenta", "maroon"})


def test_symbolic_clock_rejects_near_overlapping_explicit_time() -> None:
    task = SymbolicClockOffsetReadoutTask()
    with pytest.raises(ValueError):
        task.generate(
            20420,
            params={"query_id": "minutes_after", "shown_hour": 12, "shown_minute": 0},
            max_attempts=20,
        )


def test_symbolic_clock_alarm_wait_time_rejects_overlapping_alarm_hand() -> None:
    task = SymbolicClockAlarmWaitTimeValueTask()
    with pytest.raises(ValueError):
        task.generate(
            20453,
            params={"shown_hour": 3, "shown_minute": 0, "alarm_hour": 3},
            max_attempts=20,
        )
