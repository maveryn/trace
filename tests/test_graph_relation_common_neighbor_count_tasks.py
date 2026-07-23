"""Behavior tests for graph common-neighbor relation task."""

from __future__ import annotations

import json
from collections import Counter

from trace_tasks.core.seed import hash64
from trace_tasks.tasks import TASK_REGISTRY
from trace_tasks.tasks.graph.node_link.common_related_node_count import GraphRelationCommonNeighborCountTask


def _extract_prompt_json_example(prompt: str) -> dict:
    marker = "Example JSON:\n"
    assert marker in str(prompt)
    payload = str(prompt).split(marker, 1)[1].strip()
    return json.loads(payload)


def _extract_annotation_format_sentence(prompt: str) -> str:
    marker = "Annotation format: "
    assert marker in str(prompt)
    return str(prompt).split(marker, 1)[1].split("\n", 1)[0]


def test_graph_relation_common_neighbor_undirected_contract_matches_trace() -> None:
    task = GraphRelationCommonNeighborCountTask()
    out = task.generate(
        20601,
        params={
            "common_neighbor_mode": "undirected_common_neighbor",
            "target_count": 2,
            "node_count": 8,
            "label_variant": "named",
            "edge_routing_variant": "mixed_arc",
            "layout_variant": "shell",
        },
        max_attempts=200,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]
    node_entities = [
        entity for entity in trace["scene_ir"]["entities"] if entity["entity_kind"] == "graph_node"
    ]

    assert "task_graph__node_link__common_related_node_count" in TASK_REGISTRY
    assert out.scene_id == "node_link"
    assert out.query_id == "undirected_common_neighbor_count"
    assert out.answer_gt.type == "integer"
    assert out.annotation_gt.type == "point_set"
    assert int(out.answer_gt.value) == 2
    assert trace["scene_ir"]["scene_kind"] == "graph_common_neighbor_relation"
    assert execution["query_id"] == "undirected_common_neighbor_count"
    assert execution["graph_directionality"] == "undirected"
    assert execution["common_neighbor_mode"] == "undirected_common_neighbor"
    assert execution["question_format"] == "common_neighbor_count"
    assert execution["layout_variant_requested"] == "shell"
    assert execution["edge_routing_variant"] == "mixed_arc"
    assert execution["label_variant"] == "named"
    assert 'node "' in str(out.prompt)

    query_a = str(execution["query_label_a"])
    query_b = str(execution["query_label_b"])
    adjacency = {str(key): {str(value) for value in values} for key, values in execution["adjacency_by_label"].items()}
    matching_labels = [str(label) for label in execution["matching_labels"]]
    expected = sorted(adjacency[query_a] & adjacency[query_b])
    assert expected == sorted(matching_labels)
    assert int(out.answer_gt.value) == len(matching_labels) == len(out.annotation_gt.value)
    assert trace["witness_symbolic"]["labels"] == matching_labels
    assert trace["witness_symbolic"]["common_neighbor_mode"] == "undirected_common_neighbor"
    assert trace["projected_annotation"]["type"] == "point_set"
    assert trace["projected_annotation"]["point_set"] == out.annotation_gt.value
    assert sum(1 for node in node_entities if bool(node["is_common_neighbor_node"])) == len(matching_labels)
    assert sorted(out.prompt_variants.keys()) == ["answer_and_annotation", "answer_only"]


def test_graph_relation_common_successor_and_predecessor_modes_use_edge_direction() -> None:
    task = GraphRelationCommonNeighborCountTask()
    successor_out = task.generate(
        20602,
        params={
            "common_neighbor_mode": "directed_common_successor",
            "target_count": 3,
            "node_count": 8,
            "label_variant": "letters",
        },
        max_attempts=200,
    )
    successor_trace = successor_out.trace_payload["execution_trace"]
    query_a = str(successor_trace["query_label_a"])
    query_b = str(successor_trace["query_label_b"])
    successors = {
        str(key): {str(value) for value in values}
        for key, values in successor_trace["successors_by_label"].items()
    }
    assert successor_out.query_id == "directed_common_successor_count"
    assert successor_trace["graph_directionality"] == "directed"
    assert successor_trace["question_format"] == "common_successor_count"
    assert sorted(successors[query_a] & successors[query_b]) == sorted(successor_trace["matching_labels"])
    assert int(successor_out.answer_gt.value) == 3

    predecessor_out = task.generate(
        20603,
        params={
            "common_neighbor_mode": "directed_common_predecessor",
            "target_count": 0,
            "node_count": 7,
            "label_variant": "numbers",
        },
        max_attempts=200,
    )
    predecessor_trace = predecessor_out.trace_payload["execution_trace"]
    query_a = str(predecessor_trace["query_label_a"])
    query_b = str(predecessor_trace["query_label_b"])
    predecessors = {
        str(key): {str(value) for value in values}
        for key, values in predecessor_trace["predecessors_by_label"].items()
    }
    assert predecessor_out.query_id == "directed_common_predecessor_count"
    assert predecessor_trace["graph_directionality"] == "directed"
    assert predecessor_trace["question_format"] == "common_predecessor_count"
    assert sorted(predecessors[query_a] & predecessors[query_b]) == []
    assert predecessor_trace["matching_labels"] == []
    assert predecessor_out.annotation_gt.value == []
    assert int(predecessor_out.answer_gt.value) == 0


def test_graph_relation_common_neighbor_prompt_examples_match_contract() -> None:
    task = GraphRelationCommonNeighborCountTask()
    out = task.generate(
        20604,
        params={
            "common_neighbor_mode": "undirected_common_neighbor",
            "target_count": 2,
            "label_variant": "numbers",
        },
        max_attempts=200,
    )
    answer_only = _extract_prompt_json_example(out.prompt_variants["answer_only"])
    answer_and_annotation = _extract_prompt_json_example(out.prompt_variants["answer_and_annotation"])
    assert answer_only == {"answer": 2}
    assert list(answer_and_annotation.keys()) == ["annotation", "answer"]
    assert answer_and_annotation["annotation"] == [[180, 220], [310, 180]]
    assert answer_and_annotation["answer"] == 2


def test_graph_relation_common_neighbor_annotation_hint_matches_query_branch() -> None:
    task = GraphRelationCommonNeighborCountTask()
    cases = [
        (
            "undirected_common_neighbor",
            "directly connected to both",
            ("point to directly", "points directly to both", "successor", "predecessor"),
        ),
        (
            "directed_common_successor",
            "point to directly",
            ("directly connected to both", "points directly to both", "predecessor", "neighbor"),
        ),
        (
            "directed_common_predecessor",
            "points directly to both",
            ("directly connected to both", "point to directly", "successor", "neighbor"),
        ),
    ]
    for index, (mode, required_phrase, forbidden_phrases) in enumerate(cases):
        out = task.generate(
            20620 + index,
            params={
                "common_neighbor_mode": mode,
                "target_count": 1,
                "label_variant": "letters",
            },
            max_attempts=200,
        )
        annotation_sentence = _extract_annotation_format_sentence(out.prompt_variants["answer_and_annotation"])
        assert required_phrase in annotation_sentence
        for phrase in forbidden_phrases:
            assert phrase not in annotation_sentence


def test_graph_relation_common_neighbor_balanced_sampling_includes_zero() -> None:
    task = GraphRelationCommonNeighborCountTask()
    query_ids: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    answers: Counter[int] = Counter()
    label_variants: Counter[str] = Counter()
    edge_routing_variants: Counter[str] = Counter()
    for index in range(100):
        out = task.generate(
            hash64(20610, "graph_relation_common_neighbor_count", index),
            params={},
            max_attempts=200,
        )
        execution = out.trace_payload["execution_trace"]
        query_ids[str(execution["query_id"])] += 1
        modes[str(execution["common_neighbor_mode"])] += 1
        answers[int(out.answer_gt.value)] += 1
        label_variants[str(execution["label_variant"])] += 1
        edge_routing_variants[str(execution["edge_routing_variant"])] += 1
        assert int(out.answer_gt.value) == int(execution["target_count"])
        assert 0 <= int(out.answer_gt.value) <= 4

    assert set(query_ids) == {
        "undirected_common_neighbor_count",
        "directed_common_successor_count",
        "directed_common_predecessor_count",
    }
    assert min(query_ids.values()) >= 20
    assert set(modes) == {
        "undirected_common_neighbor",
        "directed_common_successor",
        "directed_common_predecessor",
    }
    assert set(answers) == set(range(0, 5))
    assert all(count > 0 for count in answers.values())
    assert set(label_variants) == {"letters", "numbers", "named"}
    assert set(edge_routing_variants) == {"straight", "mixed_arc"}
