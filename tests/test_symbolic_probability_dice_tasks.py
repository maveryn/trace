"""Contract tests for dice probability puzzle tasks."""

from __future__ import annotations

from math import gcd

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks import TASK_REGISTRY
from trace_tasks.tasks.symbolic.dice.dice_conditional_event_value import (
    SUPPORTED_QUERY_IDS as CONDITIONAL_QUERY_IDS,
    SymbolicProbabilityDiceConditionalEventValueTask,
)
from trace_tasks.tasks.symbolic.dice.pair_attribute_combo_probability import (
    SUPPORTED_QUERY_IDS as PAIR_ATTRIBUTE_COMBO_QUERY_IDS,
    SymbolicProbabilityDicePairAttributeComboProbabilityTask,
)
from trace_tasks.tasks.symbolic.dice.pair_difference_probability import (
    SymbolicProbabilityDicePairDifferenceProbabilityTask,
)
from trace_tasks.tasks.symbolic.dice.pair_sum_probability import (
    SymbolicProbabilityDicePairSumProbabilityTask,
)
from trace_tasks.tasks.symbolic.dice.pair_sum_threshold_probability import (
    SUPPORTED_QUERY_IDS as PAIR_SUM_THRESHOLD_QUERY_IDS,
    SymbolicProbabilityDicePairSumThresholdProbabilityTask,
)
from trace_tasks.tasks.symbolic.dice.single_attribute_probability import (
    SUPPORTED_QUERY_IDS as SINGLE_ATTRIBUTE_QUERY_IDS,
    SymbolicProbabilityDiceSingleAttributeProbabilityTask,
)
from trace_tasks.tasks.symbolic.dice.single_threshold_probability import (
    SUPPORTED_QUERY_IDS as SINGLE_THRESHOLD_QUERY_IDS,
    SymbolicProbabilityDiceSingleThresholdProbabilityTask,
)
from trace_tasks.tasks.symbolic.dice.shared.rendering import SUPPORTED_DICE_VISUAL_STYLES


TASKS = (
    (
        "task_symbolic__dice__single_attribute_probability",
        SymbolicProbabilityDiceSingleAttributeProbabilityTask,
        set(SINGLE_ATTRIBUTE_QUERY_IDS),
    ),
    (
        "task_symbolic__dice__single_threshold_probability",
        SymbolicProbabilityDiceSingleThresholdProbabilityTask,
        set(SINGLE_THRESHOLD_QUERY_IDS),
    ),
    (
        "task_symbolic__dice__pair_sum_probability",
        SymbolicProbabilityDicePairSumProbabilityTask,
        {SINGLE_QUERY_ID},
    ),
    (
        "task_symbolic__dice__pair_sum_threshold_probability",
        SymbolicProbabilityDicePairSumThresholdProbabilityTask,
        set(PAIR_SUM_THRESHOLD_QUERY_IDS),
    ),
    (
        "task_symbolic__dice__pair_difference_probability",
        SymbolicProbabilityDicePairDifferenceProbabilityTask,
        {SINGLE_QUERY_ID},
    ),
    (
        "task_symbolic__dice__pair_attribute_combo_probability",
        SymbolicProbabilityDicePairAttributeComboProbabilityTask,
        set(PAIR_ATTRIBUTE_COMBO_QUERY_IDS),
    ),
    (
        "task_symbolic__dice__dice_conditional_event_value",
        SymbolicProbabilityDiceConditionalEventValueTask,
        set(CONDITIONAL_QUERY_IDS),
    ),
)


def _reduced_fraction(numerator: int, denominator: int) -> str:
    common = gcd(abs(int(numerator)), abs(int(denominator)))
    return f"{int(numerator) // common}/{int(denominator) // common}"


def test_dice_probability_tasks_are_registered() -> None:
    for task_id, task_cls, _queries in TASKS:
        assert TASK_REGISTRY[task_id] is task_cls
        task = task_cls()
        assert task.domain == "symbolic"
        assert not hasattr(task, "scene_id")


def test_dice_probability_tasks_emit_contracts() -> None:
    for index, (_task_id, task_cls, queries) in enumerate(TASKS):
        out = task_cls().generate(2026052600 + index, params={}, max_attempts=30)
        trace = out.trace_payload
        execution = trace["execution_trace"]
        event = execution["event"]

        assert out.scene_id == "dice"
        assert out.query_id in queries
        assert out.answer_gt.type == "option_letter"
        assert trace["query_spec"]["params"]["query_id"] == out.query_id
        assert trace["render_spec"]["scene_id"] == "dice"
        assert trace["render_spec"]["label_style"]["font"]["source"] == "global_font_pool"
        assert trace["render_spec"]["label_style"]["font"]["font_family"]
        dice_style = trace["render_spec"]["dice_visual_style"]
        assert dice_style["style_id"] in set(SUPPORTED_DICE_VISUAL_STYLES)
        assert dice_style["semantic_color_policy"]["die_face_colors_preserved"] is True
        assert dice_style["semantic_color_policy"]["pip_count_and_positions_preserved"] is True
        assert trace["render_spec"]["post_image_noise_policy"]["reason"] == "dice_color_and_pip_readability"
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
        if execution["mode"] == "pair":
            assert out.annotation_gt.type == "bbox_map"
            assert trace["render_map"]["annotation_source"] == "keyed_tray_bboxes_px"
            assert trace["projected_annotation"]["type"] == "bbox_map"
            assert trace["projected_annotation"]["bbox_map"] == out.annotation_gt.value
            assert trace["projected_annotation"]["pixel_bbox_map"] == out.annotation_gt.value
            assert set(out.annotation_gt.value) == {"tray_a", "tray_b"}
            assert execution["annotation_item_ids"] == ["tray_a", "tray_b"]
            assert execution["annotation_role_item_ids"] == {"tray_a": "tray_a", "tray_b": "tray_b"}
            assert event["favorable_pairs"]
            annotation_boxes = list(out.annotation_gt.value.values())
        else:
            assert out.annotation_gt.type == "bbox"
            assert trace["render_map"]["annotation_source"] == "tray_bbox_px"
            assert trace["projected_annotation"]["type"] == "bbox"
            assert trace["projected_annotation"]["bbox"] == out.annotation_gt.value
            assert trace["projected_annotation"]["pixel_bbox"] == out.annotation_gt.value
            assert execution["annotation_item_ids"] == ["tray"]
            assert execution["annotation_role_item_ids"] == {"dice_tray": "tray"}
            assert event["favorable_die_ids"]
            annotation_boxes = [out.annotation_gt.value]
        if execution["mode"] == "conditional":
            assert event["denominator_die_ids"]
            assert set(event["favorable_die_ids"]).issubset(set(event["denominator_die_ids"]))
        assert execution["calculation_supporting_item_ids"]
        for bbox in annotation_boxes:
            assert len(bbox) == 4
            assert 0 <= float(bbox[0]) < float(bbox[2]) <= out.image.size[0]
            assert 0 <= float(bbox[1]) < float(bbox[3]) <= out.image.size[1]


def test_dice_probability_generation_is_deterministic() -> None:
    task = SymbolicProbabilityDiceConditionalEventValueTask()
    params = {
        "scene_variant": "dice_tray_felt",
        "dice_visual_style": "inked_pips",
        "query_id": "conditional_color_given_value_property_probability",
    }
    out_a = task.generate(2026052699, params=params, max_attempts=30)
    out_b = task.generate(2026052699, params=params, max_attempts=30)

    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.prompt == out_b.prompt
    assert out_a.image.tobytes() == out_b.image.tobytes()
