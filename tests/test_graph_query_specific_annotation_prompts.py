"""Regression tests for query-specific graph annotation instructions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from trace_tasks.tasks.graph.node_link.degree_extremum_value import (
    GraphComparisonExtremeDegreeValueTask,
)
from trace_tasks.tasks.graph.adjacency.directed_strong_component_count import (
    GraphCountingAdjacencyDirectedStrongComponentCountTask,
)
from trace_tasks.tasks.graph.binary_tree.child_structure_node_count import (
    GraphCountingBinaryTreeChildStructureNodeCountTask,
)
from trace_tasks.tasks.graph.node_link.cross_color_edge_count import (
    GraphCountingCrossColorEdgeCountTask,
)
from trace_tasks.tasks.graph.node_link.degree_value_filter_count import GraphCountingDegreeValueFilterCountTask
from trace_tasks.tasks.graph.node_link.named_node_degree_value import (
    GraphCountingNamedNodeDegreeValueTask,
)
from trace_tasks.tasks.graph.node_link.degree_after_removal_filter_count import (
    GraphCountingDegreeAfterRemovalFilterCountTask,
)
from trace_tasks.tasks.graph.adjacency.traversal_kth_label import (
    GraphOrderAdjacencyTraversalLabelTask,
)
from trace_tasks.tasks.graph.binary_tree.traversal_kth_label import (
    GraphOrderBinaryTreeTraversalLabelTask,
)
from trace_tasks.tasks.graph.node_link.shortest_path_length import GraphPathShortestPathLengthTask
from trace_tasks.tasks.graph.automaton.state_after_input_label import (
    GraphRelationAutomatonStateSimulationLabelTask,
)
from trace_tasks.tasks.graph.automaton.nfa_accepted_string_label import (
    GraphRelationAutomatonNfaAcceptedStringLabelTask,
)
from trace_tasks.tasks.graph.binary_tree.lowest_common_ancestor_label import (
    GraphRelationBinaryTreeLowestCommonAncestorLabelTask,
)
from trace_tasks.tasks.graph.node_link.component_size_after_edge_edit import (
    GraphRelationComponentSizeAfterEdgeEditTask,
)
from trace_tasks.tasks.graph.node_link.reachable_count_after_edge_edit import (
    GraphRelationReachableCountAfterEdgeEditTask,
)
from trace_tasks.tasks.graph.binary_tree.bst_path_operation_label import (
    GraphRelationBstPathOperationLabelTask,
)
from trace_tasks.tasks.graph.node_link.unique_related_node_label import (
    GraphRelationUniqueNodeLabelTask,
)

FORBIDDEN_GENERIC_ANNOTATION_PHRASES = (
    "requested degree condition",
    "requested source or sink condition",
    "hypothetical arrow edit",
    "queried edge or arrow",
    "visited search or insertion path",
    "hypothetical edge edit",
    "queried extreme degree value",
    "source or sink",
    "common-neighbor, common-successor, or common-predecessor",
)


def _annotation_sentence(prompt: str) -> str:
    marker = "Annotation format: "
    assert marker in str(prompt)
    return str(prompt).split(marker, 1)[1].split("\n", 1)[0]


@pytest.mark.parametrize(
    ("task_cls", "seed", "params", "expected_phrases"),
    [
        (
            GraphCountingDegreeValueFilterCountTask,
            30101,
            {
                "query_id": "undirected_degree_count",
                "query_degree": 2,
                "target_count": 1,
                "node_count": 6,
            },
            ("nodes with degree 2",),
        ),
        (
            GraphCountingDegreeValueFilterCountTask,
            30102,
            {
                "query_id": "directed_in_degree_count",
                "degree_mode": "in_degree",
                "query_degree": 1,
                "target_count": 1,
                "node_count": 6,
            },
            ("nodes with in-degree 1",),
        ),
        (
            GraphCountingDegreeValueFilterCountTask,
            30103,
            {
                "query_id": "directed_out_degree_count",
                "degree_mode": "out_degree",
                "query_degree": 1,
                "target_count": 1,
                "node_count": 6,
            },
            ("nodes with out-degree 1",),
        ),
        (
            GraphCountingDegreeAfterRemovalFilterCountTask,
            30106,
            {
                "query_id": "directed_in_degree_one_filter_remaining_count",
                "target_count": 4,
                "node_count": 7,
            },
            ("removing every node with in-degree 1",),
        ),
        (
            GraphCountingDegreeAfterRemovalFilterCountTask,
            30107,
            {
                "query_id": "directed_out_degree_one_filter_remaining_count",
                "target_count": 4,
                "node_count": 7,
            },
            ("removing every node with out-degree 1",),
        ),
        (
            GraphCountingCrossColorEdgeCountTask,
            30108,
            {"query_id": "directed_cross_color_edge_count", "target_count": 1, "node_count": 8},
            ("one arrow from", "node to"),
        ),
        (
            GraphCountingNamedNodeDegreeValueTask,
            30109,
            {
                "query_id": "directed_named_node_in_degree_value",
                "target_degree": 1,
                "node_count": 6,
            },
            ("arrow pointing into node",),
        ),
        (
            GraphCountingAdjacencyDirectedStrongComponentCountTask,
            30110,
            {
                "component_count": 2,
                "node_count": 6,
            },
            ("strongly connected component",),
        ),
        (
            GraphCountingBinaryTreeChildStructureNodeCountTask,
            30111,
            {"query_id": "leaf_node_count", "target_count": 3},
            ("every leaf node",),
        ),
        (
            GraphComparisonExtremeDegreeValueTask,
            30112,
            {
                "graph_directionality": "directed",
                "degree_mode": "out_degree",
                "extremum_mode": "max",
                "target_degree": 2,
                "node_count": 7,
            },
            ("maximum out-degree value",),
        ),
        (
            GraphPathShortestPathLengthTask,
            30113,
            {
                "query_id": "directed_shortest_path_length",
                "target_shortest_path_length": 3,
                "node_count": 7,
            },
            ("following arrow directions",),
        ),
        (
            GraphOrderAdjacencyTraversalLabelTask,
            30114,
            {
                "query_id": "bfs_kth_visit_label",
                "traversal_position": 3,
                "node_count": 6,
            },
            ("breadth-first search",),
        ),
        (
            GraphOrderBinaryTreeTraversalLabelTask,
            30115,
            {"query_id": "postorder_kth_node_label", "traversal_position": 3},
            ("postorder traversal",),
        ),
        (
            GraphRelationReachableCountAfterEdgeEditTask,
            30116,
            {
                "query_id": "reachable_count_after_edge_addition",
                "target_reachable_count": 3,
                "node_count": 7,
            },
            ("after adding an arrow",),
        ),
        (
            GraphRelationComponentSizeAfterEdgeEditTask,
            30117,
            {
                "query_id": "component_size_after_edge_removal",
                "target_component_size": 3,
                "node_count": 7,
            },
            ("after removing the edge",),
        ),
        (
            GraphRelationUniqueNodeLabelTask,
            30119,
            {"query_id": "unique_predecessor_label", "node_count": 6},
            ("arrow pointing into",),
        ),
        (
            GraphRelationBstPathOperationLabelTask,
            30120,
            {"query_id": "bst_insert_parent_label", "node_count": 7},
            ("visited insertion path", "new key would attach"),
        ),
        (
            GraphRelationAutomatonStateSimulationLabelTask,
            30121,
            {"query_id": "transition_step_state_label"},
            ("requested transition step",),
        ),
        (
            GraphRelationAutomatonNfaAcceptedStringLabelTask,
            30122,
            {},
            ("accepting NFA path",),
        ),
        (
            GraphRelationBinaryTreeLowestCommonAncestorLabelTask,
            30123,
            {},
            ("node_a", "node_b", "lowest_common_ancestor"),
        ),
    ],
)
def test_graph_annotation_prompt_matches_query_branch(
    task_cls: type,
    seed: int,
    params: Mapping[str, Any],
    expected_phrases: tuple[str, ...],
) -> None:
    task = task_cls()
    out = task.generate(seed, params=dict(params), max_attempts=300)

    sentence = _annotation_sentence(out.prompt_variants["answer_and_annotation"])
    sentence_lower = sentence.lower()
    for phrase in expected_phrases:
        assert phrase.lower() in sentence_lower
    for phrase in FORBIDDEN_GENERIC_ANNOTATION_PHRASES:
        assert phrase not in sentence_lower
