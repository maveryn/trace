"""Contract smoke tests for treemap composition chart tasks."""

from __future__ import annotations

from trace_tasks.tasks.registry import TASK_REGISTRY, ensure_scene_tasks_registered


TREEMAP_TASKS = {
    "task_charts__treemap__group_total_value": {
        "single",
    },
    "task_charts__treemap__parent_total_extremum_label": {
        "largest_parent_total",
        "smallest_parent_total",
    },
    "task_charts__treemap__repeated_leaf_aggregate_value": {
        "treemap_repeated_leaf_sum_value",
        "treemap_repeated_leaf_average_value",
    },
}


def test_treemap_tasks_registered() -> None:
    ensure_scene_tasks_registered("charts", "treemap")
    for task_id in TREEMAP_TASKS:
        assert task_id in TASK_REGISTRY


def test_treemap_tasks_generate_default_query_outputs() -> None:
    for seed_index, (task_id, allowed_query_ids) in enumerate(sorted(TREEMAP_TASKS.items())):
        task = TASK_REGISTRY[task_id]()
        output = task.generate(
            203_000 + seed_index,
            params={},
            max_attempts=160,
        )
        assert output.query_id in allowed_query_ids
        assert output.scene_id == "treemap"
        assert output.trace_payload["query_spec"]["params"]["query_id"] == output.query_id
        if task_id.endswith("__parent_total_extremum_label"):
            assert output.answer_gt.type == "string"
        else:
            assert output.answer_gt.type == "integer"
        assert output.annotation_gt.type == "bbox_set"
        assert output.annotation_gt.value
        assert output.trace_payload["projected_annotation"]["type"] == "bbox_set"
        assert output.trace_payload["projected_annotation"]["bbox_set"] == output.annotation_gt.value
        assert str(output.trace_payload["render_spec"]["font_assets"]["chart_font_family"]).strip()
        assert output.trace_payload["render_spec"]["value_source"] == "printed_leaf_values"
        assert output.trace_payload["render_map"]["leaf_traces"]
        assert output.trace_payload["render_map"]["parent_traces"]


def test_treemap_tasks_generate_each_query_branch() -> None:
    seed_index = 0
    for task_id, allowed_query_ids in sorted(TREEMAP_TASKS.items()):
        task = TASK_REGISTRY[task_id]()
        for query_id in sorted(allowed_query_ids):
            output = task.generate(
                204_000 + seed_index,
                params={"query_id": query_id},
                max_attempts=200,
            )
            assert output.query_id == query_id
            assert output.scene_id == "treemap"
            assert output.trace_payload["query_spec"]["params"]["query_id"] == query_id
            assert output.annotation_gt.value
            assert output.trace_payload["projected_annotation"]["type"] == "bbox_set"
            execution = output.trace_payload["execution_trace"]
            if task_id.endswith("__group_total_value"):
                parent_id = str(execution["parent_id"])
                parents = {
                    str(parent["parent_id"]): dict(parent)
                    for parent in execution["parents"]
                }
                assert output.answer_gt.type == "integer"
                assert output.answer_gt.value == int(parents[parent_id]["value"])
                assert output.annotation_gt.value == [
                    output.trace_payload["render_map"]["annotation_bbox_by_leaf_id"][leaf_id]
                    for leaf_id in parents[parent_id]["leaf_ids"]
                ]
            elif task_id.endswith("__parent_total_extremum_label"):
                totals = {
                    str(parent["parent_label"]): int(parent["parent_total"])
                    for parent in execution["parent_totals"]
                }
                expected = (
                    max(totals, key=totals.get)
                    if str(query_id) == "largest_parent_total"
                    else min(totals, key=totals.get)
                )
                assert output.answer_gt.type == "string"
                assert output.answer_gt.value == str(expected)
                assert execution["answer_parent_label"] == str(expected)
                assert execution["answer_parent_total"] == int(totals[str(expected)])
                assert output.annotation_gt.value == [
                    output.trace_payload["render_map"]["annotation_bbox_by_leaf_id"][leaf_id]
                    for leaf_id in execution["leaf_ids"]
                ]
            elif task_id.endswith("__repeated_leaf_aggregate_value"):
                values = [int(value) for value in execution["leaf_values"]]
                expected = sum(values)
                if str(query_id) == "treemap_repeated_leaf_average_value":
                    expected = expected // len(values)
                assert output.answer_gt.type == "integer"
                assert output.answer_gt.value == int(expected)
            seed_index += 1
