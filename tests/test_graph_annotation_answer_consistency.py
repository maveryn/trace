"""Graph-domain annotation/answer consistency guardrails."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping

from trace_tasks.core.seed import hash64
from trace_tasks.core.taxonomy import TASK_TAXONOMY
from trace_tasks.tasks import create_task
from trace_tasks.tasks.registry import list_default_task_ids

LEN_EQ_ANSWER_TASKS = {
    "task_graph__adjacency__directed_pair_reciprocity_count",
    "task_graph__adjacency__directed_strong_component_count",
    "task_graph__adjacency__undirected_component_count",
    "task_graph__binary_tree__child_structure_node_count",
    "task_graph__binary_tree__depth_level_node_count",
    "task_graph__automaton__nondeterministic_state_count",
    "task_graph__flow_network__min_cut_edge_count",
    "task_graph__metro__exact_distance_station_count",
    "task_graph__metro__route_condition_station_count",
    "task_graph__metro__station_membership_count",
    "task_graph__node_link__articulation_point_count",
    "task_graph__node_link__bridge_count",
    "task_graph__node_link__common_related_node_count",
    "task_graph__node_link__component_size_after_edge_edit",
    "task_graph__node_link__cross_color_edge_count",
    "task_graph__node_link__degree_after_removal_filter_count",
    "task_graph__node_link__degree_value_filter_count",
    "task_graph__node_link__edge_color_count",
    "task_graph__node_link__edge_text_count",
    "task_graph__node_link__isolated_after_removal_count",
    "task_graph__node_link__largest_chordless_cycle_size",
    "task_graph__node_link__largest_component_size",
    "task_graph__node_link__named_node_degree_value",
    "task_graph__node_link__node_color_count",
    "task_graph__node_link__reachable_count",
    "task_graph__node_link__reachable_count_after_edge_edit",
    "task_graph__node_link__same_component_count",
    "task_graph__node_link__unique_cycle_size",
    "task_graph__pipe_network__bridge_count",
    "task_graph__pipe_network__pipe_exact_distance_count",
    "task_graph__pipe_network__pipe_reachable_junction_count",
}

PATH_LEN_EQ_ANSWER_PLUS_ONE_TASKS = {
    "task_graph__node_link__longest_path_length",
    "task_graph__pipe_network__shortest_path_length",
}

PATH_LEN_EQ_ANSWER_TASKS = {
    "task_graph__metro__shortest_path_length",
    "task_graph__node_link__shortest_path_length",
}

SINGLE_ANNOTATION_LABEL_TASKS = {
    "task_graph__graph_options__contained_subgraph_label",
    "task_graph__graph_options__same_structure_label",
    "task_graph__node_link__edge_between_nodes_label",
    "task_graph__node_link__unique_related_node_label",
}

GRAPH_QUERY_IDS = {
    "task_graph__adjacency__directed_strong_component_count": ("single",),
    "task_graph__adjacency__directed_pair_reciprocity_count": ("single",),
    "task_graph__adjacency__mst_weight": ("single",),
    "task_graph__adjacency__traversal_kth_label": (
        "bfs_kth_visit_label",
        "dfs_kth_visit_label",
    ),
    "task_graph__adjacency__undirected_component_count": ("single",),
    "task_graph__automaton__dfa_accepted_string_label": ("single",),
    "task_graph__automaton__nfa_accepted_string_label": ("single",),
    "task_graph__automaton__nondeterministic_state_count": ("single",),
    "task_graph__automaton__state_after_input_label": (
        "final_state_label",
        "transition_step_state_label",
    ),
    "task_graph__binary_tree__bst_path_operation_label": (
        "bst_insert_parent_label",
        "bst_search_terminal_label",
    ),
    "task_graph__binary_tree__heap_property_violation_label": (
        "heap_property_violation_label",
    ),
    "task_graph__binary_tree__child_structure_node_count": (
        "internal_node_count",
        "leaf_node_count",
        "single_child_node_count",
        "two_child_node_count",
    ),
    "task_graph__binary_tree__depth_level_node_count": ("single",),
    "task_graph__binary_tree__heap_property_violation_label": ("single",),
    "task_graph__binary_tree__local_relative_node_label": (
        "left_child_label",
        "parent_label",
        "right_child_label",
        "sibling_label",
    ),
    "task_graph__binary_tree__lowest_common_ancestor_label": ("single",),
    "task_graph__binary_tree__traversal_kth_label": (
        "inorder_kth_node_label",
        "level_order_kth_node_label",
        "postorder_kth_node_label",
        "preorder_kth_node_label",
    ),
    "task_graph__flow_network__max_flow_value": ("single",),
    "task_graph__flow_network__min_cut_edge_count": ("single",),
    "task_graph__graph_options__contained_subgraph_label": ("single",),
    "task_graph__graph_options__same_structure_label": ("single",),
    "task_graph__metro__exact_distance_station_count": ("single",),
    "task_graph__metro__route_condition_station_count": (
        "metro_route_single_route_station_count",
        "metro_route_transfer_station_count",
    ),
    "task_graph__metro__shortest_path_length": ("single",),
    "task_graph__metro__station_membership_count": ("single",),
    "task_graph__node_link__articulation_point_count": ("single",),
    "task_graph__node_link__bridge_count": ("single",),
    "task_graph__node_link__common_related_node_count": (
        "directed_common_predecessor_count",
        "directed_common_successor_count",
        "undirected_common_neighbor_count",
    ),
    "task_graph__node_link__component_size_after_edge_edit": (
        "component_size_after_edge_addition",
        "component_size_after_edge_removal",
    ),
    "task_graph__node_link__cross_color_edge_count": (
        "cross_color_edge_count",
        "directed_cross_color_edge_count",
    ),
    "task_graph__node_link__degree_extremum_value": (
        "directed_max_in_degree_value",
        "directed_max_out_degree_value",
        "undirected_max_degree_value",
        "undirected_min_degree_value",
    ),
    "task_graph__node_link__degree_after_removal_filter_count": (
        "directed_in_degree_one_filter_remaining_count",
        "directed_out_degree_one_filter_remaining_count",
        "undirected_degree_one_filter_remaining_count",
    ),
    "task_graph__node_link__degree_value_filter_count": (
        "directed_in_degree_count",
        "directed_out_degree_count",
        "undirected_degree_count",
    ),
    "task_graph__node_link__edge_between_nodes_label": (
        "directed_edge_between_nodes_label",
        "edge_between_nodes_label",
    ),
    "task_graph__node_link__edge_color_count": ("single",),
    "task_graph__node_link__edge_text_count": ("single",),
    "task_graph__node_link__hamiltonian_cycle_neighbor_label": (
        "next_in_hamiltonian_cycle_label",
        "previous_in_hamiltonian_cycle_label",
    ),
    "task_graph__node_link__isolated_after_removal_count": ("single",),
    "task_graph__node_link__largest_chordless_cycle_size": ("single",),
    "task_graph__node_link__largest_component_size": ("single",),
    "task_graph__node_link__longest_path_length": ("single",),
    "task_graph__node_link__mst_weight": ("single",),
    "task_graph__node_link__named_node_degree_value": (
        "directed_named_node_in_degree_value",
        "directed_named_node_out_degree_value",
        "directed_named_node_total_degree_value",
        "undirected_named_node_degree_value",
    ),
    "task_graph__node_link__node_color_count": ("single",),
    "task_graph__node_link__reachable_count": ("single",),
    "task_graph__node_link__reachable_count_after_edge_edit": (
        "reachable_count_after_edge_addition",
        "reachable_count_after_edge_removal",
    ),
    "task_graph__node_link__same_component_count": ("single",),
    "task_graph__node_link__shortest_path_length": (
        "directed_shortest_path_length",
        "undirected_shortest_path_length",
    ),
    "task_graph__node_link__topological_endpoint_node_label": (
        "first_in_topological_order_label",
        "last_in_topological_order_label",
    ),
    "task_graph__node_link__unique_cycle_size": ("single",),
    "task_graph__node_link__unique_related_node_label": (
        "unique_neighbor_label",
        "unique_predecessor_label",
        "unique_successor_label",
    ),
    "task_graph__pedigree_chart__relatedness_coefficient_label": ("single",),
    "task_graph__pedigree_chart__relationship_label": ("single",),
    "task_graph__phylogeny_tree__clade_leaf_count": ("single",),
    "task_graph__phylogeny_tree__mrca_clade_membership_count": ("single",),
    "task_graph__phylogeny_tree__sister_leaf_label": ("single",),
    "task_graph__phylogeny_tree__topology_outlier_label": ("single",),
    "task_graph__pipe_network__bridge_count": ("single",),
    "task_graph__pipe_network__pipe_exact_distance_count": ("single",),
    "task_graph__pipe_network__pipe_reachable_junction_count": ("single",),
    "task_graph__pipe_network__shortest_path_length": ("single",),
}


def _active_graph_task_ids() -> list[str]:
    """Return active/default graph task ids from the public registry surface."""

    return sorted(
        task_id
        for task_id in list_default_task_ids()
        if TASK_TAXONOMY.get(task_id) is not None
        and TASK_TAXONOMY[task_id].domain == "graph"
    )


def _assert_query_map_matches_active_graph_tasks(graph_task_ids: list[str]) -> None:
    """Keep explicit query branch validators synchronized with active graph tasks."""

    stale_query_task_ids = sorted(set(GRAPH_QUERY_IDS) - set(graph_task_ids))
    missing_query_task_ids = sorted(set(graph_task_ids) - set(GRAPH_QUERY_IDS))
    assert stale_query_task_ids == []
    assert missing_query_task_ids == []


def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _values_equal(left: Any, right: Any) -> bool:
    if _is_num(left) and _is_num(right):
        return abs(float(left) - float(right)) < 1e-9
    return str(left) == str(right)


def _canonical_edge(edge: Any) -> tuple[str, ...]:
    return tuple(sorted(str(value) for value in edge))


def _directed_edge(edge: Any) -> tuple[str, ...]:
    return tuple(str(value) for value in edge)


def _check_annotation_shape(annotation_type: str, annotation_value: Any) -> list[str]:
    errors: list[str] = []
    if annotation_type == "point":
        if not (
            isinstance(annotation_value, list)
            and len(annotation_value) == 2
            and all(_is_num(component) for component in annotation_value)
        ):
            errors.append(f"{annotation_type} value is not [x,y]: {annotation_value!r}")
    elif annotation_type == "bbox":
        if not (
            isinstance(annotation_value, list)
            and len(annotation_value) == 4
            and all(_is_num(component) for component in annotation_value)
        ):
            errors.append(f"{annotation_type} value is not [x0,y0,x1,y1]: {annotation_value!r}")
        elif not (
            float(annotation_value[2]) > float(annotation_value[0])
            and float(annotation_value[3]) > float(annotation_value[1])
        ):
            errors.append(f"{annotation_type} value has non-positive extent: {annotation_value!r}")
    elif annotation_type in {"point_set", "point_sequence"}:
        if not isinstance(annotation_value, list):
            return [f"{annotation_type} value is not a list"]
        for index, point in enumerate(annotation_value):
            if not (
                isinstance(point, list)
                and len(point) == 2
                and all(_is_num(component) for component in point)
            ):
                errors.append(f"{annotation_type}[{index}] is not [x,y]: {point!r}")
                break
    elif annotation_type in {"bbox_set", "bbox_sequence"}:
        if not isinstance(annotation_value, list):
            return [f"{annotation_type} value is not a list"]
        for index, bbox in enumerate(annotation_value):
            if not (
                isinstance(bbox, list)
                and len(bbox) == 4
                and all(_is_num(component) for component in bbox)
            ):
                errors.append(
                    f"{annotation_type}[{index}] is not [x0,y0,x1,y1]: {bbox!r}"
                )
                break
            if not (
                float(bbox[2]) > float(bbox[0]) and float(bbox[3]) > float(bbox[1])
            ):
                errors.append(
                    f"{annotation_type}[{index}] has non-positive extent: {bbox!r}"
                )
                break
    elif annotation_type == "segment_set":
        if not isinstance(annotation_value, list):
            return [f"{annotation_type} value is not a list"]
        for index, pair in enumerate(annotation_value):
            if not (
                isinstance(pair, list)
                and len(pair) == 2
                and all(
                    isinstance(point, list)
                    and len(point) == 2
                    and all(_is_num(component) for component in point)
                    for point in pair
                )
            ):
                errors.append(f"{annotation_type}[{index}] is not a segment: {pair!r}")
                break
    elif annotation_type == "point_map":
        if not isinstance(annotation_value, Mapping):
            return [f"{annotation_type} value is not an object"]
        for key, point in annotation_value.items():
            if not (
                isinstance(key, str)
                and isinstance(point, list)
                and len(point) == 2
                and all(_is_num(component) for component in point)
            ):
                errors.append(f"{annotation_type}[{key!r}] is not [x,y]: {point!r}")
                break
    elif annotation_type == "bbox_map":
        if not isinstance(annotation_value, Mapping):
            return [f"{annotation_type} value is not an object"]
        for key, bbox in annotation_value.items():
            if not (
                isinstance(key, str)
                and isinstance(bbox, list)
                and len(bbox) == 4
                and all(_is_num(component) for component in bbox)
            ):
                errors.append(
                    f"{annotation_type}[{key!r}] is not [x0,y0,x1,y1]: {bbox!r}"
                )
                break
            if not (
                float(bbox[2]) > float(bbox[0]) and float(bbox[3]) > float(bbox[1])
            ):
                errors.append(
                    f"{annotation_type}[{key!r}] has non-positive extent: {bbox!r}"
                )
                break
    else:
        errors.append(f"unhandled annotation type: {annotation_type!r}")
    return errors


def _check_len(
    errors: list[str], *, name: str, annotation_len: int | None, values: Any
) -> None:
    if isinstance(values, list) and annotation_len != len(values):
        errors.append(f"annotation length {annotation_len} != {name} length {len(values)}")


def _audit_graph_sample(row: Mapping[str, Any]) -> list[str]:
    task_id = str(row["task_id"])
    query_id = str(row["query_id"])
    answer_type = str(row["answer_type"])
    answer_value = row["answer_value"]
    annotation_type = str(row["annotation_type"])
    annotation_value = row["annotation_value"]
    execution_trace = row["execution_trace"]
    projected_annotation = row["projected_annotation"]
    annotation_len = (
        1
        if annotation_type in {"point", "bbox", "segment"}
        else (len(annotation_value) if isinstance(annotation_value, (list, Mapping)) else None)
    )

    errors = _check_annotation_shape(annotation_type, annotation_value)
    if isinstance(execution_trace, Mapping) and "answer" in execution_trace:
        if not _values_equal(answer_value, execution_trace["answer"]):
            errors.append(
                f"answer_gt {answer_value!r} != execution_trace.answer {execution_trace['answer']!r}"
            )
    if isinstance(projected_annotation, Mapping) and projected_annotation.get("type"):
        if str(projected_annotation["type"]) != annotation_type:
            errors.append(
                f"annotation_gt.type {annotation_type!r} != projected_annotation.type "
                f"{projected_annotation['type']!r}"
            )
        if (
            annotation_type == "bbox_map"
            and projected_annotation.get("bbox_map") != annotation_value
        ):
            errors.append(
                "bbox_map annotation does not match projected_annotation.bbox_map"
            )
        if (
            annotation_type == "point_map"
            and projected_annotation.get("point_map") != annotation_value
        ):
            errors.append(
                "point_map annotation does not match projected_annotation.point_map"
            )
        if annotation_type == "bbox" and projected_annotation.get("bbox") != annotation_value:
            errors.append("bbox annotation does not match projected_annotation.bbox")
        if annotation_type == "point" and projected_annotation.get("point") != annotation_value:
            errors.append("point annotation does not match projected_annotation.point")

    if task_id in LEN_EQ_ANSWER_TASKS:
        if answer_type != "integer":
            errors.append(f"count task expected integer answer, got {answer_type!r}")
        elif annotation_len != int(answer_value):
            errors.append(
                f"count annotation length {annotation_len} != answer {answer_value}"
            )

    if task_id in PATH_LEN_EQ_ANSWER_PLUS_ONE_TASKS:
        if annotation_len != int(answer_value) + 1:
            errors.append(
                f"path annotation length {annotation_len} != answer+1 {int(answer_value) + 1}"
            )
        _check_len(
            errors,
            name="trace path labels",
            annotation_len=annotation_len,
            values=execution_trace.get("matching_labels")
            or execution_trace.get("shortest_path_labels")
            or execution_trace.get("longest_path_labels"),
        )

    if task_id in PATH_LEN_EQ_ANSWER_TASKS:
        if annotation_len != int(answer_value):
            errors.append(
                f"path annotation length {annotation_len} != answer {answer_value}"
            )
        _check_len(
            errors,
            name="annotation path labels",
            annotation_len=annotation_len,
            values=execution_trace.get("annotation_labels")
            or execution_trace.get("matching_labels"),
        )

    if task_id in SINGLE_ANNOTATION_LABEL_TASKS and annotation_len != 1:
        errors.append(f"single-label annotation length {annotation_len} != 1")

    if (
        task_id
        not in {
            "task_graph__metro__shortest_path_length",
            "task_graph__node_link__hamiltonian_cycle_neighbor_label",
        }
        and execution_trace.get("matching_labels")
        and annotation_type in {
        "point_set",
        "point_sequence",
        "bbox_set",
        "bbox_sequence",
        }
    ):
        _check_len(
            errors,
            name="matching_labels",
            annotation_len=annotation_len,
            values=execution_trace.get("matching_labels"),
        )
    if execution_trace.get("matching_edges") and annotation_type in {"segment_set", "bbox_set"}:
        _check_len(
            errors,
            name="matching_edges",
            annotation_len=annotation_len,
            values=execution_trace.get("matching_edges"),
        )
    if "counted_edges" in execution_trace and annotation_type == "segment_set":
        _check_len(
            errors,
            name="counted_edges",
            annotation_len=annotation_len,
            values=execution_trace.get("counted_edges"),
        )
    if "annotation_edges" in execution_trace and annotation_type == "segment_set":
        _check_len(
            errors,
            name="annotation_edges",
            annotation_len=annotation_len,
            values=execution_trace.get("annotation_edges"),
        )
    if "annotation_cell_edges" in execution_trace and annotation_type == "bbox_set":
        _check_len(
            errors,
            name="annotation_cell_edges",
            annotation_len=annotation_len,
            values=execution_trace.get("annotation_cell_edges"),
        )
    if "annotation_labels" in execution_trace and annotation_type in {
        "bbox_set",
        "bbox_sequence",
    }:
        _check_len(
            errors,
            name="annotation_labels",
            annotation_len=annotation_len,
            values=execution_trace.get("annotation_labels"),
        )

    if task_id == "task_graph__node_link__degree_extremum_value":
        if int(annotation_len) != 1:
            errors.append(
                "degree-extremum annotation must contain exactly one unique "
                f"extremum node, got {annotation_len}"
            )
        labels = execution_trace.get("matching_labels") or []
        if execution_trace.get("target_degree") is not None and int(
            answer_value
        ) != int(execution_trace["target_degree"]):
            errors.append(
                f"degree-extremum answer {answer_value} != target_degree "
                f"{execution_trace['target_degree']}"
            )
        queried_degrees = execution_trace.get("queried_degrees_by_label") or {}
        mismatched = [
            label
            for label in labels
            if str(label) in queried_degrees
            and int(queried_degrees[str(label)]) != int(answer_value)
        ]
        if mismatched:
            errors.append(
                f"degree-extremum labels do not match answer: {mismatched[:3]!r}"
            )

    if task_id == "task_graph__node_link__topological_endpoint_node_label":
        order = execution_trace.get("topological_order_labels") or []
        if annotation_len != 1:
            errors.append(f"topological endpoint annotation length {annotation_len} != 1")
        if str(answer_value) != str(execution_trace.get("answer_label")):
            errors.append("topological endpoint answer does not match answer_label")
        if query_id == "first_in_topological_order_label" and order and str(answer_value) != str(order[0]):
            errors.append("topological endpoint answer is not the first order label")
        if query_id == "last_in_topological_order_label" and order and str(answer_value) != str(order[-1]):
            errors.append("topological endpoint answer is not the last order label")

    if task_id == "task_graph__node_link__mst_weight":
        mst_edges = execution_trace.get("minimum_spanning_tree_edges") or execution_trace.get("matching_edges") or []
        _check_len(
            errors,
            name="minimum_spanning_tree_edges",
            annotation_len=annotation_len,
            values=mst_edges,
        )
        if execution_trace.get("minimum_spanning_tree_total_weight") is not None:
            if int(answer_value) != int(
                execution_trace["minimum_spanning_tree_total_weight"]
            ):
                errors.append("MST answer does not match trace total weight")
        weights = {
            _canonical_edge(item["endpoints"]): int(item["weight"])
            for item in execution_trace.get("edge_weights_by_label", [])
            if isinstance(item, Mapping)
        }
        if mst_edges and weights:
            total = sum(weights.get(_canonical_edge(edge), 0) for edge in mst_edges)
            if total != int(answer_value):
                errors.append(
                    f"MST selected edge weights sum {total} != answer {answer_value}"
                )

    if task_id == "task_graph__adjacency__mst_weight":
        mst_edges = execution_trace.get("minimum_spanning_tree_edges") or []
        _check_len(
            errors,
            name="minimum_spanning_tree_edges",
            annotation_len=annotation_len,
            values=mst_edges,
        )
        if annotation_len != int(execution_trace.get("node_count", 0)) - 1:
            errors.append("adjacency MST annotation length does not equal node_count-1")

    if task_id == "task_graph__adjacency__directed_pair_reciprocity_count":
        counted_pairs = execution_trace.get("counted_pairs") or []
        annotation_cell_keys = execution_trace.get("annotation_cell_keys") or []
        if annotation_type != "segment_set":
            errors.append("adjacency reciprocity annotation must be segment_set")
        if int(answer_value) != len(counted_pairs):
            errors.append("adjacency reciprocity answer does not match counted_pairs length")
        if annotation_len != int(answer_value):
            errors.append("adjacency reciprocity annotation length must equal the answer")
        if len(annotation_cell_keys) != 2 * annotation_len:
            errors.append("adjacency reciprocity annotation_cell_keys length must be twice annotation length")
        for key in annotation_cell_keys:
            if "||" not in str(key):
                errors.append(f"invalid matrix cell key: {key!r}")
                break
            row_label, column_label = str(key).split("||", 1)
            if str(row_label) == str(column_label):
                errors.append("adjacency reciprocity annotation includes a diagonal cell")
                break

    if task_id == "task_graph__flow_network__max_flow_value":
        cut_edges = (
            execution_trace.get("minimum_cut_edges")
            or execution_trace.get("original_min_cut_edges")
            or []
        )
        _check_len(
            errors,
            name="minimum_cut_edges",
            annotation_len=annotation_len,
            values=cut_edges,
        )
        max_flow_value = execution_trace.get(
            "max_flow_value", execution_trace.get("original_max_flow_value")
        )
        if int(answer_value) != int(max_flow_value):
            errors.append("max-flow answer does not match trace max flow")
        capacities = {
            _directed_edge(item["edge"]): int(item["capacity"])
            for item in execution_trace.get("capacity_by_edge", [])
            if isinstance(item, Mapping)
        }
        if capacities:
            cut_capacity = sum(
                capacities.get(_directed_edge(edge), 0)
                for edge in cut_edges
            )
            if cut_capacity != int(answer_value):
                errors.append(
                    f"max-flow annotation cut capacity {cut_capacity} != answer {answer_value}"
                )

    if task_id in {
        "task_graph__binary_tree__traversal_kth_label",
        "task_graph__adjacency__traversal_kth_label",
        "task_graph__binary_tree__bst_path_operation_label",
    }:
        annotation_labels = execution_trace.get("annotation_labels") or []
        _check_len(
            errors,
            name="annotation_labels",
            annotation_len=annotation_len,
            values=annotation_labels,
        )
        if annotation_labels and str(annotation_labels[-1]) != str(answer_value):
            errors.append(
                f"last annotation label {annotation_labels[-1]!r} != answer {answer_value!r}"
            )
        if task_id == "task_graph__binary_tree__traversal_kth_label":
            if annotation_len != int(execution_trace.get("traversal_position")):
                errors.append(
                    "binary-tree traversal annotation length != traversal_position"
                )
        if task_id == "task_graph__adjacency__traversal_kth_label":
            visit_order = execution_trace.get("visit_order") or []
            if (
                annotation_labels
                and visit_order[: len(annotation_labels)] != annotation_labels
            ):
                errors.append(
                    "adjacency traversal annotation_labels is not a visit_order prefix"
                )
    if task_id == "task_graph__binary_tree__heap_property_violation_label":
        if str(answer_value) != str(execution_trace.get("answer_label")):
            errors.append("heap-property violation answer does not match answer_label")
        if annotation_type != "point_map":
            errors.append("heap-property violation annotation must be point_map")
        role_to_label = execution_trace.get("annotation_role_to_label") or {}
        if not isinstance(role_to_label, Mapping):
            errors.append("heap-property violation annotation_role_to_label is not a map")
        elif set(role_to_label) != {"parent", "child"}:
            errors.append("heap-property violation annotation keys must be parent/child")
        elif str(role_to_label.get("child")) != str(answer_value):
            errors.append("heap-property violation child annotation label != answer")
        if not isinstance(annotation_value, Mapping):
            errors.append("heap-property violation annotation value is not a map")
        elif set(annotation_value) != {"parent", "child"}:
            errors.append("heap-property violation annotation value keys must be parent/child")

    if task_id in {
        "task_graph__binary_tree__local_relative_node_label",
        "task_graph__binary_tree__lowest_common_ancestor_label",
    }:
        if str(answer_value) != str(execution_trace.get("answer_label")):
            errors.append("binary-tree relation answer does not match answer_label")
        if annotation_type != "point_map":
            errors.append("binary-tree relation annotation must be point_map")
        role_to_label = execution_trace.get("annotation_role_to_label") or {}
        if not isinstance(role_to_label, Mapping):
            errors.append("binary-tree relation annotation_role_to_label is not a map")
        elif not isinstance(annotation_value, Mapping):
            errors.append("binary-tree relation annotation value is not a map")
        elif set(role_to_label) != set(annotation_value):
            errors.append(
                "binary-tree relation annotation keys do not match annotation_role_to_label keys"
            )
        if (
            annotation_len is not None
            and annotation_len < len(execution_trace.get("query_labels") or []) + 1
        ):
            errors.append("binary-tree relation annotation omits a query or answer node")

    if task_id == "task_graph__automaton__state_after_input_label":
        state_path = execution_trace.get("annotation_state_path_labels") or []
        _check_len(
            errors,
            name="annotation_state_path_labels",
            annotation_len=annotation_len,
            values=state_path,
        )
        if state_path and str(state_path[-1]) != str(answer_value):
            errors.append(
                f"automaton state path ends at {state_path[-1]!r}, not answer {answer_value!r}"
            )
        expected_len = (
            int(execution_trace.get("input_length")) + 1
            if query_id == "final_state_label"
            else int(execution_trace.get("transition_step_count")) + 1
        )
        if annotation_len != expected_len:
            errors.append(
                f"automaton state path length {annotation_len} != expected {expected_len}"
            )

    if task_id in {
        "task_graph__automaton__dfa_accepted_string_label",
        "task_graph__automaton__nfa_accepted_string_label",
    }:
        _check_len(
            errors,
            name="accepting_path_labels",
            annotation_len=annotation_len,
            values=execution_trace.get("accepting_path_labels"),
        )
        if annotation_len != int(execution_trace.get("input_length")) + 1:
            errors.append("accepted-string path length != input_length+1")
        if str(answer_value) != str(execution_trace.get("answer_option_label")):
            errors.append("accepted-string answer does not match answer_option_label")

    if task_id in {"task_graph__node_link__edge_between_nodes_label"}:
        if str(answer_value) != str(execution_trace.get("target_edge_label")):
            errors.append("edge-attribute answer does not match target_edge_label")
    if task_id in {
        "task_graph__node_link__hamiltonian_cycle_neighbor_label",
        "task_graph__node_link__unique_related_node_label",
    }:
        if str(answer_value) != str(execution_trace.get("answer_label")):
            errors.append("node-label answer does not match answer_label")
    if task_id in {
        "task_graph__graph_options__contained_subgraph_label",
        "task_graph__graph_options__same_structure_label",
    }:
        if str(answer_value) != str(execution_trace.get("answer_option_label")):
            errors.append("structure-match answer does not match answer_option_label")
    return errors


def _collect_sample(output: Any, seed: int) -> dict[str, Any]:
    return {
        "task_id": str(
            output.trace_payload.get("query_spec", {}).get("task_id")
            or output.trace_payload.get("execution_trace", {}).get("task_id")
            or ""
        ),
        "query_id": str(output.query_id),
        "seed": int(seed),
        "answer_type": output.answer_gt.type,
        "answer_value": output.answer_gt.value,
        "annotation_type": output.annotation_gt.type,
        "annotation_value": output.annotation_gt.value,
        "execution_trace": output.trace_payload.get("execution_trace", {}),
        "projected_annotation": output.trace_payload.get("projected_annotation", {}),
    }


def _generate_contract_sample(task: Any, *, task_id: str, query_id: str, sample_index: int) -> tuple[Any, int]:
    """Generate one sample for audit, retrying rare infeasible deterministic axes."""

    params = {} if str(query_id) == "single" else {"query_id": str(query_id)}
    last_error: Exception | None = None
    for retry_index in range(8):
        seed = hash64(
            20260528,
            f"graph_annotation_contract:{task_id}:{query_id}",
            sample_index,
            retry_index,
        )
        try:
            return task.generate(seed, params=params, max_attempts=300), int(seed)
        except RuntimeError as exc:
            last_error = exc
            continue
    raise AssertionError(f"could not generate contract sample for {task_id}/{query_id}") from last_error


def test_graph_annotation_matches_answer_contracts() -> None:
    """Every graph query branch should keep answer, annotation, and trace aligned."""

    failures: list[tuple[str, str, int, list[str]]] = []
    graph_task_ids = _active_graph_task_ids()
    _assert_query_map_matches_active_graph_tasks(graph_task_ids)

    total_samples = 0
    query_bucket_counts: Counter[tuple[str, str]] = Counter()
    for task_id in graph_task_ids:
        task = create_task(task_id)
        for query_id in GRAPH_QUERY_IDS[task_id]:
            for sample_index in range(2):
                output, seed = _generate_contract_sample(
                    task,
                    task_id=task_id,
                    query_id=str(query_id),
                    sample_index=sample_index,
                )
                row = _collect_sample(output, seed)
                if not row["task_id"]:
                    row["task_id"] = task_id
                total_samples += 1
                query_bucket_counts[(task_id, str(query_id))] += 1
                errors = _audit_graph_sample(row)
                if errors:
                    failures.append((task_id, str(query_id), int(row["seed"]), errors))

    expected_total_samples = 2 * sum(len(query_ids) for query_ids in GRAPH_QUERY_IDS.values())
    assert total_samples == expected_total_samples
    assert len(query_bucket_counts) == sum(len(query_ids) for query_ids in GRAPH_QUERY_IDS.values())
    assert failures == []
