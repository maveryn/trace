"""Behavior tests for pages paired-form reconciliation tasks."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from trace_tasks.tasks.pages.paired_forms.shortfall_minus_overage_value import (
    PROMPT_QUERY_KEY as SHORTFALL_PROMPT_QUERY_KEY,
    PagesPairedFormsShortfallMinusOverageValueTask,
)
from trace_tasks.tasks.pages.paired_forms.sum_absolute_quantity_differences_value import (
    PROMPT_QUERY_KEY as SUM_ABS_PROMPT_QUERY_KEY,
    PagesPairedFormsSumAbsoluteQuantityDifferencesValueTask,
)
from trace_tasks.tasks.pages.paired_forms.total_amount_delta_value import (
    PROMPT_QUERY_KEY as TOTAL_DELTA_PROMPT_QUERY_KEY,
    PagesPairedFormsTotalAmountDeltaValueTask,
)
from tests.helpers import extract_prompt_json_example


def _answer_from_trace(prompt_query_key: str, item_specs: Sequence[Mapping[str, Any]]) -> int:
    if str(prompt_query_key) == TOTAL_DELTA_PROMPT_QUERY_KEY:
        return sum(
            abs(int(spec["order_qty"]) - int(spec["received_qty"])) * int(spec["unit_value"])
            for spec in item_specs
            if int(spec["order_qty"]) != int(spec["received_qty"])
        )
    if str(prompt_query_key) == SHORTFALL_PROMPT_QUERY_KEY:
        return sum(
            (int(spec["order_qty"]) - int(spec["received_qty"])) * int(spec["unit_value"])
            for spec in item_specs
        )
    if str(prompt_query_key) == SUM_ABS_PROMPT_QUERY_KEY:
        return sum(abs(int(spec["order_qty"]) - int(spec["received_qty"])) for spec in item_specs)
    raise AssertionError(f"unknown prompt query key {prompt_query_key}")


def _task_cases():
    return (
        (PagesPairedFormsTotalAmountDeltaValueTask(), TOTAL_DELTA_PROMPT_QUERY_KEY),
        (PagesPairedFormsShortfallMinusOverageValueTask(), SHORTFALL_PROMPT_QUERY_KEY),
        (PagesPairedFormsSumAbsoluteQuantityDifferencesValueTask(), SUM_ABS_PROMPT_QUERY_KEY),
    )


def test_pages_paired_forms_contract_matches_trace() -> None:
    for index, (task, prompt_query_key) in enumerate(_task_cases()):
        out = task.generate(
            70080 + index,
            params={"scene_variant": "purchase_receipt_pair"},
            max_attempts=10,
        )
        trace = out.trace_payload
        execution = trace["execution_trace"]
        render = trace["render_spec"]
        render_map = trace["render_map"]
        item_specs = [dict(spec) for spec in execution["item_specs"]]
        annotation_bbox_ids = [str(item) for item in execution["annotation_bbox_ids"]]
        annotation_bboxes = [[float(value) for value in bbox] for bbox in out.annotation_gt.value]

        assert out.answer_gt.type == "integer"
        assert out.annotation_gt.type == "bbox_set"
        assert sorted(out.prompt_variants.keys()) == ["answer_and_annotation", "answer_only"]
        assert str(out.query_id) == "single"
        assert str(execution["query_id"]) == "single"
        assert str(execution["prompt_query_key"]) == str(prompt_query_key)
        assert str(execution["scene_variant"]) == "purchase_receipt_pair"
        assert str(execution["view_family"]) == "paired_forms_reconciliation"
        assert out.image.size == (int(render["canvas_width"]), int(render["canvas_height"]))
        assert int(out.answer_gt.value) == int(execution["answer_value"])
        assert int(out.answer_gt.value) == _answer_from_trace(str(prompt_query_key), item_specs)
        assert 4 <= int(execution["item_count"]) <= 6
        assert len(item_specs) == int(execution["item_count"])
        assert set(execution["receiving_item_order_ids"]) == {str(spec["item_id"]) for spec in item_specs}
        assert [str(item) for item in execution["receiving_item_order_ids"]] != [
            str(spec["item_id"]) for spec in item_specs
        ]
        assert trace["projected_annotation"]["type"] == "bbox_set"
        assert trace["projected_annotation"]["bbox_set"] == out.annotation_gt.value
        assert [str(item) for item in execution["supporting_bbox_ids"]] == annotation_bbox_ids
        assert all(str(item).startswith("recv:") for item in annotation_bbox_ids)
        assert set(annotation_bbox_ids) == {f"recv:{item_id}" for item_id in execution["mismatch_item_ids"]}

        expected_bboxes = [
            [float(value) for value in render_map["row_bboxes_px"][bbox_id]]
            for bbox_id in annotation_bbox_ids
        ]
        assert annotation_bboxes == expected_bboxes
        supporting_cell_bbox_ids = dict(execution["supporting_cell_bbox_ids"])
        assert supporting_cell_bbox_ids

        visible_numbers = {
            int(value)
            for spec in item_specs
            for value in (int(spec["order_qty"]), int(spec["received_qty"]), int(spec["unit_value"]))
        }
        assert int(out.answer_gt.value) not in visible_numbers

        assert 2 <= len(execution["mismatch_item_ids"]) <= 3
        assert len(annotation_bbox_ids) == len(execution["mismatch_item_ids"])
        if str(prompt_query_key) == SHORTFALL_PROMPT_QUERY_KEY:
            assert len(execution["shortfall_item_ids"]) >= 1
            assert len(execution["overage_item_ids"]) >= 1
        if str(prompt_query_key) in {TOTAL_DELTA_PROMPT_QUERY_KEY, SHORTFALL_PROMPT_QUERY_KEY}:
            assert sum(1 for role in supporting_cell_bbox_ids if role.endswith("_unit_value")) == len(
                execution["mismatch_item_ids"]
            )
        else:
            assert not any(role.endswith("_unit_value") for role in supporting_cell_bbox_ids)


def test_pages_paired_forms_prompt_examples_match_contract() -> None:
    expected_answers = {
        TOTAL_DELTA_PROMPT_QUERY_KEY: 864,
        SHORTFALL_PROMPT_QUERY_KEY: 324,
        SUM_ABS_PROMPT_QUERY_KEY: 42,
    }

    for index, (task, prompt_query_key) in enumerate(_task_cases(), start=70120):
        out = task.generate(index, params={}, max_attempts=10)
        answer_and_annotation = extract_prompt_json_example(out.prompt_variants["answer_and_annotation"])
        answer_only = extract_prompt_json_example(out.prompt_variants["answer_only"])

        assert answer_and_annotation["answer"] == expected_answers[str(prompt_query_key)]
        assert answer_only["answer"] == expected_answers[str(prompt_query_key)]
        assert isinstance(answer_and_annotation["annotation"], list)
        assert all(len(bbox) == 4 for bbox in answer_and_annotation["annotation"])


def test_pages_paired_forms_value_is_deterministic() -> None:
    task = PagesPairedFormsTotalAmountDeltaValueTask()
    params = {"scene_variant": "purchase_receipt_pair"}
    out_a = task.generate(70170, params=params, max_attempts=10)
    out_b = task.generate(70170, params=params, max_attempts=10)

    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.trace_payload["query_spec"]["prompt_variant"] == out_b.trace_payload["query_spec"]["prompt_variant"]
    assert out_a.prompt == out_b.prompt
    assert out_a.image.tobytes() == out_b.image.tobytes()


def test_pages_paired_forms_scene_variant_sampling_is_stable() -> None:
    scene_variants: set[str] = set()

    for index, (task, _) in enumerate(_task_cases(), start=70220):
        out = task.generate(index, params={}, max_attempts=10)
        execution = out.trace_payload["execution_trace"]
        assert str(execution["query_id"]) == "single"
        scene_variants.add(str(execution["scene_variant"]))

    assert scene_variants == {"purchase_receipt_pair"}


def test_pages_paired_forms_layout_jitter_stays_in_bounds() -> None:
    task = PagesPairedFormsTotalAmountDeltaValueTask()

    for dx, dy in ((24, 18), (-24, -18)):
        out = task.generate(
            70320 + dx + dy,
            params={
                "layout_jitter_enabled": True,
                "layout_jitter_x_px": dx,
                "layout_jitter_y_px": dy,
            },
            max_attempts=10,
        )
        width, height = out.image.size
        trace = out.trace_payload
        all_bboxes = []
        all_bboxes.extend(trace["render_map"]["panel_bboxes_px"].values())
        all_bboxes.extend(out.annotation_gt.value)
        for bbox in all_bboxes:
            x0, y0, x1, y1 = [float(value) for value in bbox]
            assert 0.0 <= x0 <= x1 <= float(width)
            assert 0.0 <= y0 <= y1 <= float(height)
