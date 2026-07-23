"""Contract tests for symbolic spinner tasks."""

from __future__ import annotations

from math import gcd

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks import TASK_REGISTRY
from trace_tasks.tasks.symbolic.spinner.multi_attribute_and_probability import (
    SUPPORTED_QUERY_IDS as AND_QUERY_IDS,
)
from trace_tasks.tasks.symbolic.spinner.multi_attribute_and_probability import (
    SymbolicSpinnerMultiAttributeAndProbabilityTask,
)
from trace_tasks.tasks.symbolic.spinner.multi_attribute_or_probability import (
    SUPPORTED_QUERY_IDS as OR_QUERY_IDS,
)
from trace_tasks.tasks.symbolic.spinner.multi_attribute_or_probability import (
    SymbolicSpinnerMultiAttributeOrProbabilityTask,
)
from trace_tasks.tasks.symbolic.spinner.single_attribute_probability import (
    SUPPORTED_QUERY_IDS as SINGLE_ATTRIBUTE_QUERY_IDS,
)
from trace_tasks.tasks.symbolic.spinner.single_attribute_probability import (
    SymbolicSpinnerSingleAttributeProbabilityTask,
)
from trace_tasks.tasks.symbolic.spinner.pair_color_event_probability import (
    SUPPORTED_QUERY_IDS as PAIR_QUERY_IDS,
)
from trace_tasks.tasks.symbolic.spinner.pair_color_event_probability import (
    SymbolicSpinnerPairColorEventProbabilityTask,
)


TASKS = (
    (
        "task_symbolic__spinner__single_attribute_probability",
        SymbolicSpinnerSingleAttributeProbabilityTask,
        set(SINGLE_ATTRIBUTE_QUERY_IDS),
        "bbox",
    ),
    (
        "task_symbolic__spinner__multi_attribute_and_probability",
        SymbolicSpinnerMultiAttributeAndProbabilityTask,
        set(AND_QUERY_IDS),
        "bbox",
    ),
    (
        "task_symbolic__spinner__multi_attribute_or_probability",
        SymbolicSpinnerMultiAttributeOrProbabilityTask,
        set(OR_QUERY_IDS),
        "bbox",
    ),
    (
        "task_symbolic__spinner__pair_color_event_probability",
        SymbolicSpinnerPairColorEventProbabilityTask,
        set(PAIR_QUERY_IDS),
        "bbox_map",
    ),
)


def _reduced_fraction(numerator: int, denominator: int) -> str:
    common = gcd(abs(int(numerator)), abs(int(denominator)))
    return f"{int(numerator) // common}/{int(denominator) // common}"


def _assert_bbox(bbox: object, image_size: tuple[int, int]) -> None:
    assert isinstance(bbox, list)
    assert len(bbox) == 4
    assert 0 <= float(bbox[0]) < float(bbox[2]) <= image_size[0]
    assert 0 <= float(bbox[1]) < float(bbox[3]) <= image_size[1]


def test_spinner_tasks_are_registered() -> None:
    for task_id, task_cls, _queries, _annotation_type in TASKS:
        assert TASK_REGISTRY[task_id] is task_cls
        task = task_cls()
        assert task.domain == "symbolic"
        assert not hasattr(task, "scene_id")


def test_spinner_tasks_emit_contracts() -> None:
    for index, (_task_id, task_cls, queries, annotation_type) in enumerate(TASKS):
        out = task_cls().generate(2026052500 + index, params={}, max_attempts=30)
        trace = out.trace_payload
        execution = trace["execution_trace"]
        event = execution["event"]

        assert out.scene_id == "spinner"
        assert out.query_id in queries
        assert out.answer_gt.type == "option_letter"
        assert out.annotation_gt.type == annotation_type
        assert trace["query_spec"]["params"]["query_id"] == out.query_id
        assert trace["render_spec"]["scene_id"] == "spinner"
        assert sorted(out.prompt_variants.keys()) == ["answer_and_annotation", "answer_only"]
        assert out.image.size == (
            int(trace["render_spec"]["canvas_width"]),
            int(trace["render_spec"]["canvas_height"]),
        )
        expected_fraction = _reduced_fraction(
            int(event["favorable_outcome_count"]),
            int(event["total_outcome_count"]),
        )
        assert execution["probability_fraction"] == expected_fraction
        assert out.answer_gt.value in execution["option_labels"]
        assert execution["answer_label"] == out.answer_gt.value
        assert execution["answer_value"] == out.answer_gt.value
        assert execution["option_text_by_label"][str(out.answer_gt.value)] == expected_fraction
        assert len(execution["option_text_by_label"]) == 6
        assert len(set(execution["option_text_by_label"].values())) == 6
        assert trace["query_spec"]["params"]["probability_fraction"] == expected_fraction
        assert trace["query_spec"]["params"]["correct_label"] == out.answer_gt.value
        assert trace["render_spec"]["option_card_layout"]["option_count"] == 6
        assert set(trace["render_map"]["option_bboxes_px"]) == set(execution["option_labels"])
        assert trace["render_map"]["selected_option_label"] == out.answer_gt.value
        assert 0 < int(event["favorable_outcome_count"]) < int(event["total_outcome_count"])
        assert execution["calculation_supporting_item_ids"]
        if execution["mode"] == "single":
            assert execution["annotation_item_ids"] == ["spinner_panel"]
            assert trace["render_map"]["annotation_source"] == "panel_bbox_px"
            assert trace["projected_annotation"]["bbox"] == out.annotation_gt.value
            _assert_bbox(out.annotation_gt.value, out.image.size)
        else:
            assert execution["annotation_item_ids"] == ["spinner_a_panel", "spinner_b_panel"]
            assert trace["render_map"]["annotation_source"] == "keyed_panel_bboxes_px"
            assert set(out.annotation_gt.value.keys()) == {"spinner_a", "spinner_b"}
            assert trace["projected_annotation"]["bbox_map"] == out.annotation_gt.value
            for bbox in out.annotation_gt.value.values():
                _assert_bbox(bbox, out.image.size)


def test_spinner_generation_is_deterministic() -> None:
    task = SymbolicSpinnerPairColorEventProbabilityTask()
    params = {"scene_variant": "spinner_card", "query_id": "pair_same_color_probability"}
    out_a = task.generate(2026052599, params=params, max_attempts=30)
    out_b = task.generate(2026052599, params=params, max_attempts=30)

    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.prompt == out_b.prompt
    assert out_a.image.tobytes() == out_b.image.tobytes()


def test_spinner_single_query_tasks_use_single_public_query() -> None:
    for cls in (SymbolicSpinnerMultiAttributeAndProbabilityTask, SymbolicSpinnerMultiAttributeOrProbabilityTask):
        out = cls().generate(2026052699, params={}, max_attempts=30)
        assert out.query_id == SINGLE_QUERY_ID
        assert out.trace_payload["query_spec"]["params"]["query_id"] == SINGLE_QUERY_ID
        assert out.trace_payload["query_spec"]["params"]["event_key"] in {
            "single_color_and_shape_probability",
            "single_color_or_shape_probability",
        }
