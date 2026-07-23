"""Behavior tests for clock-compare task."""

from __future__ import annotations

from collections import Counter

from trace_tasks.core.seed import hash64
from trace_tasks.tasks.symbolic.clock.time_extremum_label import SymbolicClockCompareTask
from trace_tasks.tasks.symbolic.clock.time_order_label import SymbolicClockTimeOrderLabelTask
from trace_tasks.tasks.shared.time_artifact_style import (
    SUPPORTED_TIME_ARTIFACT_CLOCK_COLOR_NAMES,
    SUPPORTED_TIME_ARTIFACT_CLOCK_STYLE_VARIANTS,
)
from tests.helpers import extract_prompt_json_example


def test_symbolic_clock_compare_contract_matches_trace() -> None:
    task = SymbolicClockCompareTask()
    query_ids = ("earliest_time_label", "latest_time_label")
    scene_variants = ("classic", "outline")
    style_variants = ("studio", "marker")
    accent_colors = ("blue", "orange")
    for query_id_index, query_id in enumerate(query_ids):
        for scene_index, scene_variant in enumerate(scene_variants):
            seed = 20500 + (query_id_index * 10) + scene_index
            out = task.generate(
                seed,
                params={
                    "query_id": query_id,
                    "scene_variant": scene_variant,
                    "style_variant": style_variants[scene_index],
                    "accent_color_name": accent_colors[scene_index],
                },
                max_attempts=20,
            )
            trace = out.trace_payload
            execution = trace["execution_trace"]
            shown_total_minutes_by_label = {
                str(label): int(total)
                for label, total in execution["shown_total_minutes_by_label"].items()
            }

            assert out.answer_gt.type == "string"
            assert out.annotation_gt.type == "bbox"
            expected_direction = "earliest" if str(query_id) == "earliest_time_label" else "latest"
            assert out.query_id == query_id
            assert str(execution["query_id"]) == query_id
            assert str(execution["extremum_direction"]) == expected_direction
            assert str(execution["scene_variant"]) == str(scene_variant)
            assert str(execution["style_variant"]) == str(style_variants[scene_index])
            assert str(execution["accent_color_name"]) == str(accent_colors[scene_index])
            assert trace["scene_ir"]["scene_kind"] == "symbolic_clock_grid"
            assert int(execution["clock_count"]) == 6
            assert len(execution["clock_labels"]) == 6
            assert len(set(execution["clock_labels"])) == len(execution["clock_labels"])
            assert execution["clock_labels"] == ["A", "B", "C", "D", "E", "F"]
            assert set(execution["clock_labels"]).issubset(set(execution["clock_label_pool"]))
            assert trace["projected_annotation"]["bbox"] == out.annotation_gt.value
            assert str(trace["render_map"]["winning_label"]) == str(out.answer_gt.value)
            assert trace["render_spec"]["post_image_noise"]["apply_prob"] == 0.5
            assert trace["render_spec"]["clock_style"]["font"]["source"] == "global_font_pool"
            assert trace["render_spec"]["clock_style"]["font"]["font_family"]

            expected_winner = (
                min(shown_total_minutes_by_label.items(), key=lambda item: item[1])[0]
                if expected_direction == "earliest"
                else max(shown_total_minutes_by_label.items(), key=lambda item: item[1])[0]
            )
            assert str(out.answer_gt.value) == str(expected_winner)
            assert out.annotation_gt.value == trace["render_map"]["winning_clock_bbox_px"]


def test_symbolic_clock_compare_prompt_examples_match_variants() -> None:
    task = SymbolicClockCompareTask()
    expected = {
        "earliest_time_label": (
            {"annotation": [368, 94, 552, 278], "answer": "B"},
            {"answer": "B"},
        ),
        "latest_time_label": (
            {"annotation": [368, 94, 552, 278], "answer": "B"},
            {"answer": "B"},
        ),
    }
    for index, (query_id, (expected_answer_and_annotation, expected_answer_only)) in enumerate(expected.items(), start=20540):
        out = task.generate(
            index,
            params={"query_id": query_id},
            max_attempts=20,
        )
        answer_and_annotation = extract_prompt_json_example(out.prompt_variants["answer_and_annotation"])
        answer_only = extract_prompt_json_example(out.prompt_variants["answer_only"])
        assert answer_and_annotation == expected_answer_and_annotation
        assert answer_only == expected_answer_only


def test_symbolic_clock_compare_explicit_clock_count_uses_six_clocks() -> None:
    task = SymbolicClockCompareTask()
    out = task.generate(
        20570,
        params={"query_id": "latest_time_label", "clock_count": 6},
        max_attempts=20,
    )
    execution = out.trace_payload["execution_trace"]
    assert int(execution["clock_count"]) == 6
    assert len(execution["clock_labels"]) == 6
    assert len(set(execution["clock_labels"])) == 6
    assert execution["clock_labels"] == ["A", "B", "C", "D", "E", "F"]
    assert str(out.answer_gt.value) in set(execution["clock_labels"])


def test_symbolic_clock_compare_balanced_sampling_defaults_cover_axes() -> None:
    task = SymbolicClockCompareTask()
    query_ids: Counter[str] = Counter()
    extremum_directions: Counter[str] = Counter()
    scene_variants: Counter[str] = Counter()
    style_variants: Counter[str] = Counter()
    accent_color_names: Counter[str] = Counter()
    clock_counts: Counter[int] = Counter()
    winner_labels: Counter[str] = Counter()
    for index in range(90):
        out = task.generate(
            hash64(20580, "symbolic_clock_compare", index),
            params={},
            max_attempts=20,
        )
        execution = out.trace_payload["execution_trace"]
        query_ids[str(execution["query_id"])] += 1
        extremum_directions[str(execution["extremum_direction"])] += 1
        scene_variants[str(execution["scene_variant"])] += 1
        style_variants[str(execution["style_variant"])] += 1
        accent_color_names[str(execution["accent_color_name"])] += 1
        clock_counts[int(execution["clock_count"])] += 1
        winner_labels[str(execution["winner_label"])] += 1

    assert set(query_ids.keys()) == {"earliest_time_label", "latest_time_label"}
    assert set(extremum_directions.keys()) == {"earliest", "latest"}
    assert set(scene_variants.keys()) == {"classic", "minimal", "outline"}
    assert set(style_variants.keys()) == set(SUPPORTED_TIME_ARTIFACT_CLOCK_STYLE_VARIANTS)
    assert set(accent_color_names.keys()) == set(SUPPORTED_TIME_ARTIFACT_CLOCK_COLOR_NAMES)
    assert set(clock_counts.keys()) == {6}
    assert set(winner_labels.keys()).issubset({"A", "B", "C", "D", "E", "F"})
    assert len(winner_labels) >= 4


def test_symbolic_clock_time_order_label_contract_matches_trace() -> None:
    task = SymbolicClockTimeOrderLabelTask()
    out = task.generate(
        20700,
        params={
            "scene_variant": "classic",
            "style_variant": "accented",
            "accent_color_name": "blue",
            "shown_total_minutes_by_label": {"A": 180, "B": 90, "C": 255, "D": 45},
            "answer_label": "3",
        },
        max_attempts=20,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]

    assert out.query_id == "single"
    assert out.answer_gt.type == "string"
    assert out.answer_gt.value == "3"
    assert out.annotation_gt.type == "bbox"
    assert trace["scene_ir"]["scene_kind"] == "symbolic_clock_time_order_panel"
    assert execution["clock_labels"] == ["A", "B", "C", "D"]
    assert execution["option_labels"] == ["1", "2", "3", "4", "5", "6"]
    assert execution["shown_time_text_by_label"] == {
        "A": "03:00",
        "B": "01:30",
        "C": "04:15",
        "D": "12:45",
    }
    assert execution["true_order_labels"] == ["D", "B", "A", "C"]
    assert execution["true_order_text"] == "D < B < A < C"
    assert execution["option_order_text_by_label"]["3"] == "D < B < A < C"
    assert len(set(execution["option_order_text_by_label"].values())) == 6
    assert trace["projected_annotation"]["bbox"] == out.annotation_gt.value
    assert trace["render_map"]["correct_label"] == out.answer_gt.value
    assert trace["render_map"]["correct_option_bbox_px"] == out.annotation_gt.value
    assert out.annotation_gt.value == trace["render_map"]["option_card_bboxes_px"]["3"]
    assert out.annotation_gt.value not in [
        trace["render_map"]["clocks_by_label"][label]["face_bbox_px"]
        for label in execution["clock_labels"]
    ]
    assert trace["render_spec"]["clock_style"]["font"]["source"] == "global_font_pool"


def test_symbolic_clock_time_order_label_prompt_examples_match_contract() -> None:
    task = SymbolicClockTimeOrderLabelTask()
    out = task.generate(20710, params={}, max_attempts=20)
    answer_and_annotation = extract_prompt_json_example(out.prompt_variants["answer_and_annotation"])
    answer_only = extract_prompt_json_example(out.prompt_variants["answer_only"])
    assert answer_and_annotation == {
        "annotation": [350, 450, 630, 538],
        "answer": "2",
    }
    assert answer_only == {"answer": "2"}


def test_symbolic_clock_time_order_label_sampling_covers_axes_and_options() -> None:
    task = SymbolicClockTimeOrderLabelTask()
    scene_variants: Counter[str] = Counter()
    style_variants: Counter[str] = Counter()
    accent_color_names: Counter[str] = Counter()
    answer_labels: Counter[str] = Counter()
    for index in range(90):
        out = task.generate(
            hash64(20720, "symbolic_clock_time_order_label", index),
            params={},
            max_attempts=20,
        )
        trace = out.trace_payload
        execution = trace["execution_trace"]
        scene_variants[str(execution["scene_variant"])] += 1
        style_variants[str(execution["style_variant"])] += 1
        accent_color_names[str(execution["accent_color_name"])] += 1
        answer_labels[str(execution["correct_label"])] += 1

        shown = {str(key): int(value) for key, value in execution["shown_total_minutes_by_label"].items()}
        expected_order = [
            str(label)
            for label, _total in sorted(shown.items(), key=lambda item: (int(item[1]), str(item[0])))
        ]
        assert execution["clock_labels"] == ["A", "B", "C", "D"]
        assert execution["option_labels"] == ["1", "2", "3", "4", "5", "6"]
        assert execution["true_order_labels"] == expected_order
        assert execution["option_order_labels_by_label"][str(out.answer_gt.value)] == expected_order
        assert len(set(execution["option_order_text_by_label"].values())) == 6
        assert trace["render_map"]["correct_option_bbox_px"] == out.annotation_gt.value

    assert set(scene_variants.keys()) == {"classic", "minimal", "outline"}
    assert set(style_variants.keys()) == set(SUPPORTED_TIME_ARTIFACT_CLOCK_STYLE_VARIANTS)
    assert set(accent_color_names.keys()) == set(SUPPORTED_TIME_ARTIFACT_CLOCK_COLOR_NAMES)
    assert set(answer_labels.keys()) == {"1", "2", "3", "4", "5", "6"}
