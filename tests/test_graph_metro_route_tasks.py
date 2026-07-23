from __future__ import annotations

from trace_tasks.tasks import TASK_REGISTRY
from trace_tasks.tasks.graph.metro.exact_distance_station_count import GraphRelationMetroExactDistanceCountTask
from trace_tasks.tasks.graph.metro.route_condition_station_count import GraphMetroRouteConditionStationCountTask
from trace_tasks.tasks.graph.metro.shortest_path_length import GraphPathMetroShortestPathLengthTask
from trace_tasks.tasks.graph.metro.station_membership_count import (
    GraphCountingTransferStationCountTask,
)


def test_metro_transfer_station_count_contract() -> None:
    out = GraphCountingTransferStationCountTask().generate(
        2026051907,
        params={"target_count": 3, "route_count": 4, "label_variant": "letters"},
        max_attempts=100,
    )
    trace = out.trace_payload["execution_trace"]
    query_spec = out.trace_payload["query_spec"]
    station_routes = trace["station_route_ids_by_label"]
    transfer_labels = sorted(
        (str(label) for label, route_ids in station_routes.items() if len(route_ids) >= 2)
    )
    assert out.scene_id == "metro"
    assert out.query_id == "single"
    assert query_spec["internal_query_id"] == "metro_transfer_station_count"
    assert out.answer_gt.type == "integer"
    assert out.answer_gt.value == len(transfer_labels) == 3
    assert out.annotation_gt.type == "point_set"
    assert len(out.annotation_gt.value) == len(transfer_labels)
    assert "transfer stations" in out.prompt or "multiple routes" in out.prompt


def test_metro_transfer_station_count_is_registered() -> None:
    assert "task_graph__metro__station_membership_count" in TASK_REGISTRY
    assert "task_graph__metro__exact_distance_station_count" in TASK_REGISTRY
    assert "task_graph__metro__route_condition_station_count" in TASK_REGISTRY
    assert "task_graph__metro__shortest_path_length" in TASK_REGISTRY


def test_metro_exact_distance_count_contract() -> None:
    out = GraphRelationMetroExactDistanceCountTask().generate(
        2026051910,
        params={"target_count": 4, "query_distance": 2, "route_count": 3, "label_variant": "letters"},
        max_attempts=100,
    )
    trace = out.trace_payload["execution_trace"]
    query_label = str(trace["query_label"])
    query_distance = int(trace["query_distance"])
    adjacency = trace["adjacency_by_label"]
    distances = {query_label: 0}
    queue = [query_label]
    for node in queue:
        for neighbor in adjacency[str(node)]:
            if str(neighbor) not in distances:
                distances[str(neighbor)] = distances[str(node)] + 1
                queue.append(str(neighbor))
    exact_labels = sorted(label for label, distance in distances.items() if int(distance) == query_distance)
    assert out.scene_id == "metro"
    assert out.query_id == "single"
    assert out.answer_gt.value == len(exact_labels) == 4
    assert out.annotation_gt.type == "point_set"
    assert len(out.annotation_gt.value) == len(exact_labels)


def test_metro_route_transfer_station_count_contract() -> None:
    out = GraphMetroRouteConditionStationCountTask().generate(
        2026062501,
        params={
            "query_id": "metro_route_transfer_station_count",
            "target_count": 2,
            "route_count": 4,
            "label_variant": "letters",
        },
        max_attempts=100,
    )
    trace = out.trace_payload["execution_trace"]
    route_id = trace["query_route_ids"][0]
    route_labels = set(trace["route_station_labels"][route_id])
    station_routes = trace["station_route_ids_by_label"]
    expected = sorted(
        label
        for label in route_labels
        if len(station_routes[str(label)]) >= 2
    )
    assert out.scene_id == "metro"
    assert out.query_id == "metro_route_transfer_station_count"
    assert out.answer_gt.type == "integer"
    assert out.answer_gt.value == len(expected) == 2
    assert out.annotation_gt.type == "point_set"
    assert len(out.annotation_gt.value) == len(expected)
    assert trace["matching_labels"] == expected
    assert trace["query_route_names"]


def test_metro_route_single_route_station_count_contract() -> None:
    out = GraphMetroRouteConditionStationCountTask().generate(
        2026062502,
        params={
            "query_id": "metro_route_single_route_station_count",
            "target_count": 3,
            "route_count": 4,
            "label_variant": "letters",
        },
        max_attempts=100,
    )
    trace = out.trace_payload["execution_trace"]
    route_id = trace["query_route_ids"][0]
    route_labels = set(trace["route_station_labels"][route_id])
    station_routes = trace["station_route_ids_by_label"]
    expected = sorted(
        label
        for label in route_labels
        if len(station_routes[str(label)]) == 1
    )
    assert out.scene_id == "metro"
    assert out.query_id == "metro_route_single_route_station_count"
    assert out.answer_gt.type == "integer"
    assert out.answer_gt.value == len(expected) == 3
    assert out.annotation_gt.type == "point_set"
    assert len(out.annotation_gt.value) == len(expected)
    assert trace["matching_labels"] == expected
    assert trace["query_route_names"]


def test_metro_shortest_path_length_contract() -> None:
    out = GraphPathMetroShortestPathLengthTask().generate(
        2026051911,
        params={"target_shortest_path_length": 4, "route_count": 4, "label_variant": "letters"},
        max_attempts=100,
    )
    trace = out.trace_payload["execution_trace"]
    assert out.scene_id == "metro"
    assert out.query_id == "single"
    assert out.answer_gt.value == int(trace["target_shortest_path_length"]) == 4
    assert out.annotation_gt.type == "point_sequence"
    assert len(out.annotation_gt.value) == out.answer_gt.value
    assert trace["matching_labels"][0] == trace["source_label"]
    assert trace["matching_labels"][-1] == trace["goal_label"]


def test_metro_transfer_station_count_is_deterministic() -> None:
    task = GraphCountingTransferStationCountTask()
    params = {"target_count": 5, "route_count": 5, "label_variant": "numbers"}
    left = task.generate(2026051908, params=params, max_attempts=100)
    right = task.generate(2026051908, params=params, max_attempts=100)
    assert left.answer_gt.value == right.answer_gt.value
    assert left.annotation_gt.value == right.annotation_gt.value
    assert left.trace_payload["execution_trace"]["route_station_labels"] == right.trace_payload["execution_trace"]["route_station_labels"]
