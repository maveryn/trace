"""Behavior tests for analog/digital clock match-panel task."""

from __future__ import annotations

from collections import Counter

from trace_tasks.core.seed import hash64
from trace_tasks.tasks.symbolic.clock.equivalent_time_label import SymbolicClockMatchPanelTask, SUPPORTED_DIGITAL_DISPLAY_PALETTES
from trace_tasks.tasks.shared.time_artifact_style import (
    SUPPORTED_TIME_ARTIFACT_CLOCK_COLOR_NAMES,
    SUPPORTED_TIME_ARTIFACT_CLOCK_STYLE_VARIANTS,
)
from trace_tasks.tasks.shared.time_format import format_clock_hhmm
from tests.helpers import extract_prompt_json_example


def test_symbolic_clock_equivalent_time_contract_matches_trace() -> None:
    task = SymbolicClockMatchPanelTask()
    for index, query_id in enumerate(("analog_reference_digital_options", "digital_reference_analog_options")):
        out = task.generate(
            20600 + index,
            params={
                "query_id": query_id,
                "scene_variant": "classic",
                "style_variant": "marker",
                "accent_color_name": "blue",
                "digital_display_palette": "blue_lcd",
                "answer_label": "D",
            },
            max_attempts=20,
        )
        trace = out.trace_payload
        execution = trace["execution_trace"]

        assert out.answer_gt.type == "string"
        assert out.answer_gt.value == "D"
        assert out.annotation_gt.type == "bbox_map"
        assert set(out.annotation_gt.value.keys()) == {"reference", "correct_option"}
        assert trace["scene_ir"]["scene_kind"] == "symbolic_clock_equivalent_time_panel"
        assert out.query_id == query_id
        assert str(execution["query_id"]) == query_id
        assert str(execution["correct_label"]) == "D"
        assert str(execution["digital_display_palette"]) == "blue_lcd"
        assert str(execution["option_time_text_by_label"]["D"]) == str(execution["target_time_text"])
        assert str(format_clock_hhmm(int(execution["target_total_minutes"]))) == str(execution["target_time_text"])
        assert trace["projected_annotation"]["bbox_map"] == out.annotation_gt.value
        assert trace["render_map"]["correct_option_bbox_px"] == out.annotation_gt.value["correct_option"]
        assert trace["render_spec"]["clock_style"]["font"]["source"] == "global_font_pool"
        assert trace["render_spec"]["clock_style"]["digital_display_palette"] == "blue_lcd"
        assert set(trace["render_spec"]["clock_style"]["digital_display_colors_rgb"].keys()) == {
            "case_fill",
            "case_outline",
            "screen_fill",
            "screen_outline",
            "text",
            "glow",
            "shadow",
        }
        assert trace["render_spec"]["post_image_noise"]["apply_prob"] == 0.4
        if query_id == "analog_reference_digital_options":
            assert str(execution["reference_representation"]) == "analog"
            assert str(execution["option_representation"]) == "digital"
        else:
            assert str(execution["reference_representation"]) == "digital"
            assert str(execution["option_representation"]) == "analog"


def test_symbolic_clock_equivalent_time_prompt_examples_match_contract() -> None:
    task = SymbolicClockMatchPanelTask()
    for index, query_id in enumerate(("analog_reference_digital_options", "digital_reference_analog_options"), start=20620):
        out = task.generate(
            index,
            params={"query_id": query_id},
            max_attempts=20,
        )
        answer_and_annotation = extract_prompt_json_example(out.prompt_variants["answer_and_annotation"])
        answer_only = extract_prompt_json_example(out.prompt_variants["answer_only"])
        assert answer_and_annotation == {
            "annotation": {
                "reference": [315, 74, 587, 346],
                "correct_option": [382, 442, 652, 602],
            },
            "answer": "B",
        }
        assert answer_only == {"answer": "B"}


def test_symbolic_clock_equivalent_time_balanced_sampling_defaults_cover_axes() -> None:
    task = SymbolicClockMatchPanelTask()
    query_ids: Counter[str] = Counter()
    scene_variants: Counter[str] = Counter()
    style_variants: Counter[str] = Counter()
    accent_color_names: Counter[str] = Counter()
    digital_display_palettes: Counter[str] = Counter()
    answer_labels: Counter[str] = Counter()
    for index in range(90):
        out = task.generate(
            hash64(20640, "symbolic_clock_equivalent_time", index),
            params={},
            max_attempts=20,
        )
        execution = out.trace_payload["execution_trace"]
        query_ids[str(execution["query_id"])] += 1
        scene_variants[str(execution["scene_variant"])] += 1
        style_variants[str(execution["style_variant"])] += 1
        accent_color_names[str(execution["accent_color_name"])] += 1
        digital_display_palettes[str(execution["digital_display_palette"])] += 1
        answer_labels[str(execution["correct_label"])] += 1
        assert len(execution["option_labels"]) == 6
        assert len(set(execution["option_time_text_by_label"].values())) == 6

    assert set(query_ids.keys()) == {"analog_reference_digital_options", "digital_reference_analog_options"}
    assert set(scene_variants.keys()) == {"classic", "minimal", "outline"}
    assert set(style_variants.keys()) == set(SUPPORTED_TIME_ARTIFACT_CLOCK_STYLE_VARIANTS)
    assert set(accent_color_names.keys()) == set(SUPPORTED_TIME_ARTIFACT_CLOCK_COLOR_NAMES)
    assert set(digital_display_palettes.keys()) == set(SUPPORTED_DIGITAL_DISPLAY_PALETTES)
    assert set(answer_labels.keys()) == {"A", "B", "C", "D", "E", "F"}
