"""Contract smoke tests for candlestick/OHLC chart tasks."""

from __future__ import annotations

from collections import Counter

from trace_tasks.tasks import TASK_REGISTRY
from trace_tasks.tasks.charts.candlestick.directional_body_threshold_count import (
    DIRECTIONAL_BODY_THRESHOLD_COUNT_SUPPORT,
)


CANDLESTICK_TASKS = {
    "task_charts__candlestick__range_extremum_label": {
        "largest_body_range_label",
        "largest_wick_range_label",
        "smallest_body_range_label",
        "smallest_wick_range_label",
    },
    "task_charts__candlestick__counterfactual_close_value": {
        "close_after_body_decrease_value",
        "close_after_body_increase_value",
    },
    "task_charts__candlestick__directional_body_threshold_count": {
        "up_body_threshold_count",
        "down_body_threshold_count",
    },
}
WICK_RANGE_QUERY_IDS = {"largest_wick_range_label", "smallest_wick_range_label"}
BODY_RANGE_QUERY_IDS = {"largest_body_range_label", "smallest_body_range_label"}
COUNTERFACTUAL_QUERY_IDS = {"close_after_body_decrease_value", "close_after_body_increase_value"}
DIRECTIONAL_COUNT_QUERY_IDS = {"up_body_threshold_count", "down_body_threshold_count"}


def _candles_by_id(output):
    return {
        str(candle["candle_id"]): dict(candle)
        for candle in output.trace_payload["execution_trace"]["candles"]
    }


def _bbox_center(box):
    return [
        round((float(box[0]) + float(box[2])) / 2.0, 3),
        round((float(box[1]) + float(box[3])) / 2.0, 3),
    ]


def _bbox_vertical_segment(box):
    center_x = round((float(box[0]) + float(box[2])) / 2.0, 3)
    return [
        [center_x, round(float(box[1]), 3)],
        [center_x, round(float(box[3]), 3)],
    ]


def test_candlestick_tasks_registered() -> None:
    for task_id in CANDLESTICK_TASKS:
        assert TASK_REGISTRY.get(task_id) is not None


def test_candlestick_tasks_generate_default_query_outputs() -> None:
    for seed_index, (task_id, allowed_query_ids) in enumerate(sorted(CANDLESTICK_TASKS.items())):
        task = TASK_REGISTRY[task_id]()
        output = task.generate(
            700_000 + seed_index,
            params={},
            max_attempts=100,
        )
        assert output.scene_id == "candlestick"
        assert output.query_id in allowed_query_ids
        assert output.trace_payload["query_spec"]["params"]["query_id"] == output.query_id
        if output.query_id in BODY_RANGE_QUERY_IDS | WICK_RANGE_QUERY_IDS:
            assert output.annotation_gt.type == "segment"
            assert len(output.annotation_gt.value) == 2
            assert all(len(point) == 2 for point in output.annotation_gt.value)
            assert output.trace_payload["projected_annotation"]["type"] == "segment"
            assert output.trace_payload["projected_annotation"]["segment"] == output.annotation_gt.value
            assert output.trace_payload["projected_annotation"]["pixel_segment"] == output.annotation_gt.value
        elif output.query_id in DIRECTIONAL_COUNT_QUERY_IDS:
            assert output.annotation_gt.type == "bbox_set"
            assert all(len(box) == 4 for box in output.annotation_gt.value)
            assert output.trace_payload["projected_annotation"]["type"] == "bbox_set"
            assert output.trace_payload["projected_annotation"]["bbox_set"] == output.annotation_gt.value
        else:
            assert output.annotation_gt.type == "point"
            assert len(output.annotation_gt.value) == 2
            assert output.trace_payload["projected_annotation"]["type"] == "point"
            assert output.trace_payload["projected_annotation"]["point"] == output.annotation_gt.value
            assert output.trace_payload["projected_annotation"]["pixel_point"] == output.annotation_gt.value
        assert output.trace_payload["render_map"]["body_bboxes_px"]


def test_candlestick_tasks_generate_each_query_branch_and_answer_contract() -> None:
    seed_index = 0
    for task_id, allowed_query_ids in sorted(CANDLESTICK_TASKS.items()):
        task = TASK_REGISTRY[task_id]()
        for query_id in sorted(allowed_query_ids):
            output = task.generate(
                701_000 + seed_index,
                params={"query_id": query_id},
                max_attempts=100,
            )
            assert output.scene_id == "candlestick"
            assert output.query_id == query_id
            execution = output.trace_payload["execution_trace"]
            candles = _candles_by_id(output)

            if query_id in WICK_RANGE_QUERY_IDS:
                extremum = str(execution["extremum"])
                ranked = sorted(candles.values(), key=lambda candle: int(candle["wick_range"]))
                target = ranked[-1] if extremum == "largest" else ranked[0]
                assert output.answer_gt.type == "string"
                assert output.answer_gt.value == str(target["label"])
                assert execution["annotation_roles"] == ["answer_wick"]
                assert output.annotation_gt.type == "segment"
                assert output.annotation_gt.value == _bbox_vertical_segment(
                    output.trace_payload["render_map"]["wick_bboxes_px"][str(target["candle_id"])]
                )
                assert output.trace_payload["projected_annotation"]["type"] == "segment"
                assert output.trace_payload["projected_annotation"]["segment"] == output.annotation_gt.value
            elif query_id in BODY_RANGE_QUERY_IDS:
                extremum = str(execution["extremum"])
                ranked = sorted(candles.values(), key=lambda candle: int(candle["body_size"]))
                target = ranked[-1] if extremum == "largest" else ranked[0]
                assert output.answer_gt.type == "string"
                assert output.answer_gt.value == str(target["label"])
                assert execution["annotation_roles"] == ["answer_body"]
                assert output.annotation_gt.type == "segment"
                assert output.annotation_gt.value == _bbox_vertical_segment(
                    output.trace_payload["render_map"]["body_bboxes_px"][str(target["candle_id"])]
                )
                assert output.trace_payload["projected_annotation"]["type"] == "segment"
                assert output.trace_payload["projected_annotation"]["segment"] == output.annotation_gt.value
            elif query_id in COUNTERFACTUAL_QUERY_IDS:
                target = candles[str(execution["target_candle_id"])]
                new_body = int(execution["new_body_size"])
                if str(target["direction"]) == "up":
                    expected = int(target["open"]) + new_body
                else:
                    expected = int(target["open"]) - new_body
                assert output.answer_gt.type == "integer"
                assert output.answer_gt.value == expected
                assert execution["annotation_roles"] == ["target_body"]
                assert output.annotation_gt.type == "point"
                assert output.annotation_gt.value == _bbox_center(
                    output.trace_payload["render_map"]["body_bboxes_px"][str(target["candle_id"])]
                )
                assert output.trace_payload["projected_annotation"]["type"] == "point"
                assert output.trace_payload["projected_annotation"]["point"] == output.annotation_gt.value
            elif query_id in DIRECTIONAL_COUNT_QUERY_IDS:
                direction = "up" if str(query_id) == "up_body_threshold_count" else "down"
                threshold = int(execution["body_threshold"])
                same_direction = [
                    candle
                    for candle in execution["candles"]
                    if str(candle["direction"]) == direction
                ]
                expected_ids = [
                    str(candle["candle_id"])
                    for candle in same_direction
                    if int(candle["body_size"]) > threshold
                ]
                # Candle colour encodes only direction, so a colour-only count must
                # over-count: at least one same-direction body stays under the threshold.
                assert len(same_direction) > len(expected_ids)
                assert output.answer_gt.type == "integer"
                assert output.answer_gt.value == len(expected_ids)
                assert execution["matching_candle_ids"] == expected_ids
                assert output.annotation_gt.type == "bbox_set"
                expected_boxes = [
                    [
                        round(float(coordinate), 3)
                        for coordinate in output.trace_payload["render_map"]["body_bboxes_px"][candle_id]
                    ]
                    for candle_id in expected_ids
                ]
                assert output.annotation_gt.value == expected_boxes
            else:
                raise AssertionError(f"unsupported candlestick query id: {query_id}")

            seed_index += 1


def test_candlestick_directional_body_threshold_count_pins_explicit_target() -> None:
    task = TASK_REGISTRY["task_charts__candlestick__directional_body_threshold_count"]()
    for target_answer in (1, 2, 3, 4, 5):
        output = task.generate(
            702_000 + target_answer,
            params={
                "query_id": "down_body_threshold_count",
                "target_answer": target_answer,
                "body_threshold": 10,
            },
            max_attempts=100,
        )
        execution = output.trace_payload["execution_trace"]
        ups = [c for c in execution["candles"] if str(c["direction"]) == "up"]
        downs = [c for c in execution["candles"] if str(c["direction"]) == "down"]
        over = [c for c in downs if int(c["body_size"]) > 10]
        assert int(execution["body_threshold"]) == 10
        assert output.answer_gt.value == target_answer
        assert len(over) == target_answer
        assert len(downs) > len(over)
        assert len(ups) >= 2 and len(downs) >= 2
        assert len(output.annotation_gt.value) == target_answer


def test_candlestick_directional_body_threshold_count_spreads_answers_across_its_support() -> None:
    task = TASK_REGISTRY["task_charts__candlestick__directional_body_threshold_count"]()
    for query_id in sorted(DIRECTIONAL_COUNT_QUERY_IDS):
        counts: Counter[int] = Counter()
        for offset in range(24):
            output = task.generate(
                704_000 + offset,
                params={"query_id": query_id, "_sample_cursor": offset},
                max_attempts=100,
            )
            counts[int(output.answer_gt.value)] += 1
        total = sum(counts.values())
        # The public anti-degeneracy gate: at least four distinct answers and no
        # single answer above one third of the samples.
        assert len(counts) >= 4
        assert max(counts.values()) / total < 1 / 3
        assert set(counts) <= set(DIRECTIONAL_BODY_THRESHOLD_COUNT_SUPPORT)


def test_candlestick_directional_body_threshold_count_is_deterministic() -> None:
    task = TASK_REGISTRY["task_charts__candlestick__directional_body_threshold_count"]()
    params = {"query_id": "up_body_threshold_count"}
    first = task.generate(703_001, params=params, max_attempts=100)
    second = task.generate(703_001, params=params, max_attempts=100)
    assert first.prompt == second.prompt
    assert first.answer_gt.to_dict() == second.answer_gt.to_dict()
    assert first.annotation_gt.to_dict() == second.annotation_gt.to_dict()
    assert first.trace_payload["execution_trace"] == second.trace_payload["execution_trace"]
    assert first.image.tobytes() == second.image.tobytes()
