"""Node-link graph visual variant tests."""

from __future__ import annotations

import math

from trace_tasks.tasks.graph.node_link.largest_component_size import GraphComparisonLargestComponentSizeTask
from trace_tasks.tasks.graph.node_link.same_component_count import GraphRelationSameComponentCountTask
from trace_tasks.tasks.graph.node_link.degree_extremum_value import GraphComparisonExtremeDegreeValueTask
from trace_tasks.tasks.graph.node_link.articulation_point_count import GraphCountingArticulationPointCountTask
from trace_tasks.tasks.graph.node_link.bridge_count import GraphCountingBridgeCountTask
from trace_tasks.tasks.graph.node_link.degree_value_filter_count import GraphCountingDegreeValueFilterCountTask
from trace_tasks.tasks.graph.node_link.named_node_degree_value import GraphCountingNamedNodeDegreeValueTask
from trace_tasks.tasks.graph.node_link.topological_endpoint_node_label import GraphOrderTopologicalEndpointNodeLabelTask
from trace_tasks.tasks.graph.node_link.longest_path_length import GraphPathLongestPathLengthTask
from trace_tasks.tasks.graph.node_link.shortest_path_length import GraphPathShortestPathLengthTask
from trace_tasks.tasks.graph.node_link.common_related_node_count import GraphRelationCommonNeighborCountTask
from trace_tasks.tasks.graph.node_link.edge_between_nodes_label import GraphRelationEdgeBetweenNodesLabelTask
from trace_tasks.tasks.graph.node_link.reachable_count import GraphRelationReachableCountTask
from trace_tasks.tasks.graph.node_link.largest_chordless_cycle_size import GraphRelationLargestChordlessCycleSizeTask
from trace_tasks.tasks.graph.node_link.unique_cycle_size import GraphRelationUniqueCycleSizeTask


NODE_LINK_TASKS = (
    ("task_graph__node_link__largest_component_size", GraphComparisonLargestComponentSizeTask, {}),
    ("task_graph__node_link__articulation_point_count", GraphCountingArticulationPointCountTask, {}),
    ("task_graph__node_link__bridge_count", GraphCountingBridgeCountTask, {}),
    ("task_graph__node_link__degree_value_filter_count", GraphCountingDegreeValueFilterCountTask, {"query_id": "undirected_degree_count"}),
    ("task_graph__node_link__named_node_degree_value", GraphCountingNamedNodeDegreeValueTask, {}),
    ("task_graph__node_link__topological_endpoint_node_label", GraphOrderTopologicalEndpointNodeLabelTask, {}),
    ("task_graph__node_link__shortest_path_length", GraphPathShortestPathLengthTask, {}),
    ("task_graph__node_link__reachable_count", GraphRelationReachableCountTask, {}),
    ("task_graph__node_link__same_component_count", GraphRelationSameComponentCountTask, {}),
    ("task_graph__node_link__largest_chordless_cycle_size", GraphRelationLargestChordlessCycleSizeTask, {}),
    ("task_graph__node_link__unique_cycle_size", GraphRelationUniqueCycleSizeTask, {}),
    ("task_graph__node_link__degree_extremum_value", GraphComparisonExtremeDegreeValueTask, {}),
    ("task_graph__node_link__degree_value_filter_count", GraphCountingDegreeValueFilterCountTask, {"query_id": "directed_out_degree_count"}),
    ("task_graph__node_link__common_related_node_count", GraphRelationCommonNeighborCountTask, {}),
    ("task_graph__node_link__edge_between_nodes_label", GraphRelationEdgeBetweenNodesLabelTask, {}),
    ("task_graph__node_link__longest_path_length", GraphPathLongestPathLengthTask, {}),
)

LABEL_REFERENCING_TASKS = {
    GraphCountingNamedNodeDegreeValueTask,
    GraphPathShortestPathLengthTask,
    GraphRelationReachableCountTask,
    GraphRelationSameComponentCountTask,
    GraphRelationCommonNeighborCountTask,
    GraphRelationEdgeBetweenNodesLabelTask,
}

def _quadratic_point(
    start: tuple[float, float],
    control: tuple[float, float],
    end: tuple[float, float],
    t: float,
) -> tuple[float, float]:
    return (
        ((1.0 - t) * (1.0 - t) * start[0])
        + (2.0 * (1.0 - t) * t * control[0])
        + (t * t * end[0]),
        ((1.0 - t) * (1.0 - t) * start[1])
        + (2.0 * (1.0 - t) * t * control[1])
        + (t * t * end[1]),
    )


def _assert_arc_edges_clear_non_endpoint_nodes(
    *,
    node_entities: list[dict],
    edge_entities: list[dict],
) -> None:
    centers_by_label = {
        str(node["label"]): (float(node["center_px"][0]), float(node["center_px"][1]))
        for node in node_entities
    }
    radius_px = max(
        1,
        max(
            int(round((float(node["bbox_xyxy"][2]) - float(node["bbox_xyxy"][0])) / 2.0))
            for node in node_entities
        ),
    )
    min_clearance_px = float(max(48, int(radius_px) * 2 + 8))
    for edge in edge_entities:
        if edge["route_variant"] != "arc":
            continue
        start_label = str(edge["node_u_label"])
        end_label = str(edge["node_v_label"])
        control = (float(edge["control_px"][0]), float(edge["control_px"][1]))
        for index in range(1, 64):
            point = _quadratic_point(
                centers_by_label[start_label],
                control,
                centers_by_label[end_label],
                float(index) / 64.0,
            )
            for label, center in centers_by_label.items():
                if label in {start_label, end_label}:
                    continue
                assert (
                    math.hypot(point[0] - center[0], point[1] - center[1])
                    >= min_clearance_px
                )


def _minimum_incident_edge_angle_degrees(
    *,
    node_entities: list[dict],
    edge_entities: list[dict],
) -> float:
    centers_by_label = {
        str(node["label"]): (float(node["center_px"][0]), float(node["center_px"][1]))
        for node in node_entities
    }
    incident_by_label = {str(label): [] for label in centers_by_label}
    for edge in edge_entities:
        left = str(edge["node_u_label"])
        right = str(edge["node_v_label"])
        incident_by_label.setdefault(left, []).append(right)
        incident_by_label.setdefault(right, []).append(left)

    min_angle = 180.0
    for label, neighbors in incident_by_label.items():
        center = centers_by_label[str(label)]
        for index, left_label in enumerate(neighbors):
            for right_label in neighbors[index + 1 :]:
                left = centers_by_label[str(left_label)]
                right = centers_by_label[str(right_label)]
                left_vec = (float(left[0] - center[0]), float(left[1] - center[1]))
                right_vec = (float(right[0] - center[0]), float(right[1] - center[1]))
                left_norm = math.hypot(left_vec[0], left_vec[1])
                right_norm = math.hypot(right_vec[0], right_vec[1])
                if left_norm <= 0.0 or right_norm <= 0.0:
                    continue
                dot = max(
                    -1.0,
                    min(
                        1.0,
                        ((left_vec[0] * right_vec[0]) + (left_vec[1] * right_vec[1]))
                        / (left_norm * right_norm),
                    ),
                )
                min_angle = min(float(min_angle), float(math.degrees(math.acos(dot))))
    return float(min_angle)


def test_node_link_named_and_mixed_arcs_are_trace_recorded() -> None:
    saw_arc = False
    for index, (task_id, task_cls, extra_params) in enumerate(NODE_LINK_TASKS):
        params = {
            "label_variant": "named",
            "edge_routing_variant": "mixed_arc",
        }
        params.update(dict(extra_params))
        if task_id == "task_graph__node_link__shortest_path_length":
            params.update({"query_id": "directed_shortest_path_length"})

        out = task_cls().generate(
            930000 + index,
            params=params,
            max_attempts=200,
        )
        trace = out.trace_payload
        execution = trace["execution_trace"]
        style = trace["render_spec"]["style"]
        node_entities = [
            entity for entity in trace["scene_ir"]["entities"] if entity["entity_kind"] == "graph_node"
        ]
        edge_entities = [
            entity for entity in trace["scene_ir"]["entities"] if entity["entity_kind"] == "graph_edge"
        ]

        assert execution["label_variant"] == "named"
        assert execution["edge_routing_variant"] == "mixed_arc"
        assert style["edge_routing_variant"] == "mixed_arc"
        assert trace["render_spec"]["panel_geometry"]["font_family"]
        assert trace["render_spec"]["panel_geometry"]["layout_jitter"]["enabled"] is True
        assert all(0 < len(str(node["label"])) <= 5 for node in node_entities)
        assert all("route_variant" in edge and "control_px" in edge for edge in edge_entities)
        arc_edges = [edge for edge in edge_entities if edge["route_variant"] == "arc"]
        saw_arc = bool(arc_edges) or saw_arc
        assert all(edge["control_px"] is not None for edge in arc_edges)
        _assert_arc_edges_clear_non_endpoint_nodes(
            node_entities=node_entities,
            edge_entities=edge_entities,
        )
        if task_cls in LABEL_REFERENCING_TASKS:
            assert any(f'"{node["label"]}"' in str(out.prompt) for node in node_entities)
    assert saw_arc


def test_node_link_context_blocks_reserve_graph_content() -> None:
    for index, position in enumerate(("top", "bottom", "left", "right")):
        out = GraphCountingArticulationPointCountTask().generate(
            940100 + index,
            params={
                "context_block_probability": 1.0,
                "context_block_max_elements": 1,
                "context_block_position_weights": {position: 1.0},
                "context_block_clutter_level_weights": {"high": 1.0},
                "edge_routing_variant": "straight",
            },
            max_attempts=200,
        )
        panel_geometry = out.trace_payload["render_spec"]["panel_geometry"]
        reservation = panel_geometry["context_block_reservation"]
        context_blocks = [
            element
            for element in panel_geometry.get("context_text_elements", [])
            if element.get("kind") == "context_block"
        ]
        assert reservation["position"] == position
        assert context_blocks
        content = tuple(float(value) for value in reservation["final_content_xyxy"])
        block = tuple(float(value) for value in context_blocks[0]["bbox_xyxy"])
        if position == "top":
            assert content[1] > block[3]
        elif position == "bottom":
            assert content[3] < block[1]
        elif position == "left":
            assert content[0] > block[2]
        else:
            assert content[2] < block[0]
        for entity in out.trace_payload["scene_ir"]["entities"]:
            if entity["entity_kind"] != "graph_node":
                continue
            x, y = (float(entity["center_px"][0]), float(entity["center_px"][1]))
            assert content[0] <= x <= content[2]
            assert content[1] <= y <= content[3]


def test_mixed_arcs_do_not_duplicate_intervening_node_chain_regression() -> None:
    params = {
        "component_count": 4,
        "edge_count": 10,
        "edge_routing_variant": "mixed_arc",
        "graph_directionality": "undirected",
        "label_variant": "numbers",
        "layout_transform_variant": "rotate_90",
        "layout_variant": "circular",
        "node_color_name": "maroon",
        "node_count": 13,
        "node_shape_variant": "circle",
        "target_largest_component_size": 5,
        "topology_profile": "balanced",
    }

    out = GraphComparisonLargestComponentSizeTask().generate(
        287376090864390,
        params=params,
        max_attempts=200,
    )
    node_entities = [
        entity
        for entity in out.trace_payload["scene_ir"]["entities"]
        if entity["entity_kind"] == "graph_node"
    ]
    edge_entities = [
        entity
        for entity in out.trace_payload["scene_ir"]["entities"]
        if entity["entity_kind"] == "graph_edge"
    ]

    assert any(edge["route_variant"] == "arc" for edge in edge_entities)
    _assert_arc_edges_clear_non_endpoint_nodes(
        node_entities=node_entities,
        edge_entities=edge_entities,
    )


def test_path_spine_layout_fans_off_path_edges_regression() -> None:
    out = GraphPathShortestPathLengthTask().generate(
        6553976889450802,
        params={
            "query_id": "undirected_shortest_path_length",
            "node_count": 9,
            "target_shortest_path_length": 3,
            "topology_profile": "low_degree",
            "layout_variant": "path_spine",
            "label_variant": "numbers",
            "node_shape_variant": "rounded_square",
            "layout_transform_variant": "rotate_180",
            "edge_routing_variant": "straight",
            "node_color_name": "red",
        },
        max_attempts=200,
    )
    execution = out.trace_payload["execution_trace"]
    node_entities = [
        entity
        for entity in out.trace_payload["scene_ir"]["entities"]
        if entity["entity_kind"] == "graph_node"
    ]
    edge_entities = [
        entity
        for entity in out.trace_payload["scene_ir"]["entities"]
        if entity["entity_kind"] == "graph_edge"
    ]

    assert execution["layout_variant_used"] == "path_spine"
    assert _minimum_incident_edge_angle_degrees(
        node_entities=node_entities,
        edge_entities=edge_entities,
    ) >= 12.0
