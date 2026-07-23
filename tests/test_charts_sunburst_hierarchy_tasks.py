"""Contract smoke tests for sunburst hierarchy chart tasks."""

from __future__ import annotations

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.charts.shared.label_assets import normalize_chart_label_for_collision
from trace_tasks.tasks.registry import TASK_REGISTRY, ensure_scene_tasks_registered


SUNBURST_TASKS = {
    "task_charts__sunburst__parent_total_value": {SINGLE_QUERY_ID},
    "task_charts__sunburst__parent_total_extremum_label": {
        "highest_parent_total_label",
        "lowest_parent_total_label",
    },
    "task_charts__sunburst__leaf_threshold_count_under_parent": {
        "above_threshold_leaf_count_under_parent",
        "below_threshold_leaf_count_under_parent",
    },
    "task_charts__sunburst__leaf_range_count_under_parent": {SINGLE_QUERY_ID},
}


def test_sunburst_tasks_registered() -> None:
    ensure_scene_tasks_registered("charts", "sunburst")
    assert all(task_id in TASK_REGISTRY for task_id in SUNBURST_TASKS)


def test_sunburst_tasks_generate_default_query_outputs() -> None:
    ensure_scene_tasks_registered("charts", "sunburst")
    for seed_index, (task_id, allowed_query_ids) in enumerate(sorted(SUNBURST_TASKS.items())):
        task = TASK_REGISTRY[task_id]()
        output = task.generate(
            103_000 + seed_index,
            params={},
            max_attempts=120,
        )
        assert output.scene_id == "sunburst"
        assert output.query_id in allowed_query_ids
        assert output.trace_payload["query_spec"]["params"]["query_id"] == output.query_id
        assert output.answer_gt.type in {"integer", "string"}
        assert output.annotation_gt.type == "point_set"
        assert output.annotation_gt.value
        assert output.trace_payload["query_spec"]["params"]["program_code"]
        assert output.trace_payload["projected_annotation"]["type"] == "point_set"
        assert output.trace_payload["projected_annotation"]["point_set"] == output.annotation_gt.value
        assert output.trace_payload["render_spec"]["not_to_scale"] is True
        assert output.trace_payload["render_spec"]["font_assets"]
        assert output.trace_payload["render_map"]["node_traces"]
        node_traces = output.trace_payload["render_map"]["node_traces"]
        visible_labels = [
            str(node["label"])
            for node in node_traces
            if str(node["level"]) in {"parent", "subgroup", "leaf"}
        ]
        normalized_labels = [normalize_chart_label_for_collision(label) for label in visible_labels]
        assert len(normalized_labels) == len(set(normalized_labels))
        label_caps = output.trace_payload["execution_trace"]["label_max_chars"]
        for node in node_traces:
            level = str(node["level"])
            if level in {"parent", "subgroup", "leaf"}:
                assert len(str(node["label"])) <= int(label_caps[level])


def test_sunburst_tasks_generate_each_query_branch() -> None:
    ensure_scene_tasks_registered("charts", "sunburst")
    seed_index = 0
    for task_id, allowed_query_ids in sorted(SUNBURST_TASKS.items()):
        task = TASK_REGISTRY[task_id]()
        for query_id in sorted(allowed_query_ids):
            output = task.generate(
                104_000 + seed_index,
                params={"query_id": query_id},
                max_attempts=160,
            )
            assert output.scene_id == "sunburst"
            assert output.query_id == query_id
            assert output.trace_payload["query_spec"]["params"]["query_id"] == query_id
            prompt_lower = output.prompt.lower()
            if "annotation" in prompt_lower:
                assert prompt_lower.find("annotation") < prompt_lower.find("answer")
            assert output.annotation_gt.value
            if query_id in {"highest_parent_total_label", "lowest_parent_total_label"}:
                execution = output.trace_payload["execution_trace"]
                assert set(execution["annotation_node_ids"]) == set(execution["answer_leaf_ids"])
                assert set(execution["leaf_ids"]) == set(execution["answer_leaf_ids"])
                totals = execution["parent_totals"]
                expected = max(totals, key=totals.get) if query_id.startswith("highest") else min(totals, key=totals.get)
                assert output.answer_gt.value == expected
            elif task_id.endswith("parent_total_value"):
                execution = output.trace_payload["execution_trace"]
                assert sum(execution["leaf_values"]) == output.answer_gt.value
                assert len(output.annotation_gt.value) == len(execution["leaf_ids"])
            elif task_id.endswith("leaf_threshold_count_under_parent"):
                execution = output.trace_payload["execution_trace"]
                threshold = int(execution["threshold_value"])
                values = [int(value) for value in execution["candidate_leaf_values"]]
                if query_id.startswith("above"):
                    expected_count = sum(1 for value in values if value > threshold)
                else:
                    expected_count = sum(1 for value in values if value < threshold)
                assert output.answer_gt.value == expected_count
                assert len(output.annotation_gt.value) == expected_count
            elif task_id.endswith("leaf_range_count_under_parent"):
                execution = output.trace_payload["execution_trace"]
                lower = int(execution["lower_value"])
                upper = int(execution["upper_value"])
                values = [int(value) for value in execution["candidate_leaf_values"]]
                expected_count = sum(1 for value in values if lower <= value <= upper)
                assert output.answer_gt.value == expected_count
                assert len(output.annotation_gt.value) == expected_count
            seed_index += 1
