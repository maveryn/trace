"""Behavior tests for graph structure-option matching tasks."""

from __future__ import annotations

from trace_tasks.core.seed import hash64
from trace_tasks.tasks import create_task


def _has_antiparallel_edges(edges: list[list[str]]) -> bool:
    directed_edges = {tuple(str(value) for value in edge) for edge in edges}
    return any((target, source) in directed_edges for source, target in directed_edges)


def test_graph_options_directed_samples_avoid_overlapped_reverse_arrows() -> None:
    """Directed option graphs should not draw opposite arrows on the same segment."""

    tasks = {
        "same_structure_label": create_task("task_graph__graph_options__same_structure_label"),
        "contained_subgraph_label": create_task("task_graph__graph_options__contained_subgraph_label"),
    }
    for semantic_name, task in tasks.items():
        for index in range(20):
            out = task.generate(
                int(hash64(19244, f"graph_options_directed_no_antiparallel:{semantic_name}", index)),
                params={"query_id": "single", "edge_mode": "directed"},
                max_attempts=240,
            )
            assert out.query_id == "single"
            execution = out.trace_payload["execution_trace"]
            specs = [execution["query_structure_spec"], *[option["structure_spec"] for option in execution["option_specs"]]]
            assert all(not _has_antiparallel_edges(spec["edges"]) for spec in specs)


def test_graph_options_default_to_four_visual_options_in_two_by_two_layout() -> None:
    """Graph-option tasks should use four larger visual option panels by default."""

    task_ids = (
        "task_graph__graph_options__same_structure_label",
        "task_graph__graph_options__contained_subgraph_label",
    )
    for task_id in task_ids:
        out = create_task(task_id).generate(2026062503, params={}, max_attempts=160)
        execution = out.trace_payload["execution_trace"]
        option_bboxes = out.trace_payload["render_map"]["option_panel_bboxes_px"]
        assert int(execution["option_count"]) == 4
        assert set(option_bboxes) == {"option_A", "option_B", "option_C", "option_D"}

        left_edges = {float(bbox[0]) for bbox in option_bboxes.values()}
        top_edges = {float(bbox[1]) for bbox in option_bboxes.values()}
        assert len(left_edges) == 2
        assert len(top_edges) == 2
