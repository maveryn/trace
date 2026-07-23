"""Sampling helpers for maze-exit puzzle instances."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Tuple

from trace_tasks.core.seed import spawn_rng

from trace_tasks.tasks.shared.config_defaults import group_default

from .defaults import to_int
from .state import EXIT_LABEL_POOL, SCENE_NAMESPACE, TARGET_REACHABILITY_VALUES, Cell
from .topology import (
    boundary_sides,
    exit_clockwise_sort_key,
    generate_spanning_tree,
    reachable_cells_from_start,
    shortest_path_between,
)

def resolve_int_bounds(
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    *,
    min_key: str,
    max_key: str,
    fallback_min: int,
    fallback_max: int,
) -> Tuple[int, int]:
    lower = to_int(params.get(str(min_key), group_default(defaults, str(min_key), int(fallback_min))), int(fallback_min))
    upper = to_int(params.get(str(max_key), group_default(defaults, str(max_key), int(fallback_max))), int(fallback_max))
    if int(lower) > int(upper):
        raise ValueError(f"{min_key} must be <= {max_key}")
    return int(lower), int(upper)
def sample_int(
    *,
    rng,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    key: str,
    min_key: str,
    max_key: str,
    fallback_min: int,
    fallback_max: int,
    sampling_offset: int,
) -> Tuple[int, Tuple[int, int]]:
    lower, upper = resolve_int_bounds(
        params,
        defaults,
        min_key=str(min_key),
        max_key=str(max_key),
        fallback_min=int(fallback_min),
        fallback_max=int(fallback_max),
    )
    explicit = params.get(str(key))
    if explicit is not None:
        value = to_int(explicit, int(lower))
        if not (int(lower) <= int(value) <= int(upper)):
            raise ValueError(f"{key} must be within [{lower}, {upper}]")
        return int(value), (int(lower), int(upper))
    support = list(range(int(lower), int(upper) + 1))
    _ = sampling_offset
    return int(rng.randint(int(lower), int(upper))), (int(lower), int(upper))
def sample_exit_target_index(*, params: Mapping[str, Any], rng, exit_count: int, offset: int) -> int:
    support = list(range(int(exit_count)))
    if "target_exit_index" in params:
        value = to_int(params["target_exit_index"], 0)
        if int(value) not in support:
            raise ValueError("target_exit_index must be within the active exit support")
        return int(value)
    _ = offset
    return int(rng.choice(support))


def build_maze_exit_dataset(
    *,
    request_kind: str,
    target_reachability: str | None,
    scene_variant: str,
    params: Mapping[str, Any],
    instance_seed: int,
    generation_defaults: Mapping[str, Any],
    max_attempts: int,
) -> Dict[str, Any]:
    """Build a maze instance without any public task identity branching.

    ``request_kind`` selects the neutral construction shape: either a single
    target exit with requested reachability or a nearest-exit request. Public
    task files bind those constructions to task ids, prompts, answers, and
    annotations.
    """

    is_label_request = str(request_kind) == "exit_label"
    is_nearest_request = str(request_kind) == "nearest_exit"
    if bool(is_label_request):
        if str(target_reachability) not in set(TARGET_REACHABILITY_VALUES):
            raise ValueError("target_reachability must be reachable or unreachable for exit_reachability_label")
        resolved_target_reachability = str(target_reachability)
    else:
        resolved_target_reachability = None
    rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.dataset")
    rows, row_range = sample_int(
        rng=rng,
        params=params,
        defaults=generation_defaults,
        key="maze_rows",
        min_key="maze_rows_min",
        max_key="maze_rows_max",
        fallback_min=7,
        fallback_max=10,
        sampling_offset=0,
    )
    cols, col_range = sample_int(
        rng=rng,
        params=params,
        defaults=generation_defaults,
        key="maze_cols",
        min_key="maze_cols_min",
        max_key="maze_cols_max",
        fallback_min=9,
        fallback_max=13,
        sampling_offset=3,
    )
    exit_count, exit_count_range = sample_int(
        rng=rng,
        params=params,
        defaults=generation_defaults,
        key="exit_count",
        min_key="exit_count_min",
        max_key="exit_count_max",
        fallback_min=5,
        fallback_max=8,
        sampling_offset=11,
    )
    if int(exit_count) > len(EXIT_LABEL_POOL):
        raise ValueError("exit_count cannot exceed the available exit-label pool")

    start_col_options = [max(1, int(cols) // 2 - 1), int(cols) // 2, min(int(cols) - 2, int(cols) // 2 + 1)]
    start_row_options = [max(1, int(rows) // 2 - 1), int(rows) // 2, min(int(rows) - 2, int(rows) // 2 + 1)]
    start = (int(rng.choice(start_col_options)), int(rng.choice(start_row_options)))
    if not (0 < start[0] < int(cols) - 1 and 0 < start[1] < int(rows) - 1):
        start = (max(1, min(int(cols) - 2, int(cols) // 2)), max(1, min(int(rows) - 2, int(rows) // 2)))

    all_boundary_cells = [
        (int(col), int(row))
        for row in range(int(rows))
        for col in range(int(cols))
        if bool(boundary_sides((int(col), int(row)), rows=int(rows), cols=int(cols)))
    ]
    attempt_count = max(120, int(max_attempts) * 16)
    for attempt in range(int(attempt_count)):
        tree_rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.maze_tree", index=int(attempt))
        candidate_cells = list(all_boundary_cells)
        tree_rng.shuffle(candidate_cells)
        if len(candidate_cells) < int(exit_count):
            continue
        exit_cells = [tuple(cell) for cell in candidate_cells[: int(exit_count)]]
        exits: List[Dict[str, Any]] = []
        for index, cell in enumerate(exit_cells):
            sides = list(boundary_sides(cell, rows=int(rows), cols=int(cols)))
            side = str(tree_rng.choice(sides))
            exits.append(
                {
                    "item_id": f"exit_{index + 1}",
                    "label": "",
                    "cell": [int(cell[0]), int(cell[1])],
                    "side": str(side),
                }
            )
        exits = sorted(exits, key=lambda item: exit_clockwise_sort_key(item, rows=int(rows), cols=int(cols)))
        for index, exit_spec in enumerate(exits):
            exit_spec["item_id"] = f"exit_{index + 1}"
            exit_spec["label"] = str(EXIT_LABEL_POOL[index])

        target_index = sample_exit_target_index(params=params, rng=tree_rng, exit_count=int(exit_count), offset=17)
        if bool(is_label_request) and str(resolved_target_reachability) == "reachable":
            reachable_indices = {int(target_index)}
        elif bool(is_label_request) and str(resolved_target_reachability) == "unreachable":
            reachable_indices = {index for index in range(int(exit_count)) if index != int(target_index)}
        elif bool(is_nearest_request):
            reachable_indices = set(range(int(exit_count)))
        else:
            raise ValueError(f"unsupported maze request_kind: {request_kind}")

        blocked_exit_cells = {
            tuple(int(value) for value in exit_spec["cell"])
            for index, exit_spec in enumerate(exits)
            if int(index) not in reachable_indices
        }
        allowed_cells = {
            (int(col), int(row))
            for row in range(int(rows))
            for col in range(int(cols))
            if (int(col), int(row)) not in blocked_exit_cells
        }
        try:
            open_edges = set(
                generate_spanning_tree(
                    rows=int(rows),
                    cols=int(cols),
                    start=tuple(start),
                    rng=tree_rng,
                    allowed_cells=allowed_cells,
                )
            )
        except ValueError:
            continue
        for index, exit_spec in enumerate(exits):
            cell = tuple(int(value) for value in exit_spec["cell"])
            if int(index) in reachable_indices and cell not in allowed_cells:
                raise ValueError("reachable exit cell was excluded from the maze")

        reachable_cells = set(reachable_cells_from_start(start=tuple(start), rows=int(rows), cols=int(cols), edges=tuple(open_edges)))
        for index, exit_spec in enumerate(exits):
            cell = tuple(int(value) for value in exit_spec["cell"])
            exit_spec["reachable"] = bool(cell in reachable_cells)
        if {index for index, exit_spec in enumerate(exits) if bool(exit_spec["reachable"])} != set(reachable_indices):
            continue

        reachable_exits = [exit_spec for exit_spec in exits if bool(exit_spec["reachable"])]
        unreachable_exits = [exit_spec for exit_spec in exits if not bool(exit_spec["reachable"])]
        reachable_labels = [str(exit_spec["label"]) for exit_spec in reachable_exits]
        unreachable_labels = [str(exit_spec["label"]) for exit_spec in unreachable_exits]
        if bool(is_label_request) and str(resolved_target_reachability) == "reachable":
            answer_exit = reachable_exits[0]
            answer_value: str | int = str(answer_exit["label"])
            supporting_item_ids = [str(answer_exit["item_id"])]
            annotation_policy = "single_reachable_exit_point"
            query_details: Dict[str, Any] = {"target_reachability": str(resolved_target_reachability)}
        elif bool(is_label_request) and str(resolved_target_reachability) == "unreachable":
            answer_exit = unreachable_exits[0]
            answer_value = str(answer_exit["label"])
            supporting_item_ids = [str(answer_exit["item_id"])]
            annotation_policy = "single_unreachable_exit_point"
            query_details = {"target_reachability": str(resolved_target_reachability)}
        elif bool(is_nearest_request):
            min_gap_edges = to_int(
                params.get(
                    "nearest_exit_min_gap_edges",
                    group_default(generation_defaults, "nearest_exit_min_gap_edges", 2),
                ),
                2,
            )
            path_records: List[Dict[str, Any]] = []
            for exit_spec in reachable_exits:
                path_cells = shortest_path_between(
                    start=tuple(start),
                    goal=tuple(int(value) for value in exit_spec["cell"]),
                    rows=int(rows),
                    cols=int(cols),
                    edges=tuple(open_edges),
                )
                if len(path_cells) < 2:
                    continue
                path_records.append(
                    {
                        "label": str(exit_spec["label"]),
                        "item_id": str(exit_spec["item_id"]),
                        "cell": [int(value) for value in exit_spec["cell"]],
                        "path_cells": [
                            [int(cell[0]), int(cell[1])] for cell in path_cells
                        ],
                        "path_length_edges": max(0, len(path_cells) - 1),
                    }
                )
            if len(path_records) != int(exit_count):
                continue
            path_records = sorted(
                path_records,
                key=lambda item: (int(item["path_length_edges"]), str(item["label"])),
            )
            nearest_record = path_records[0]
            if len(path_records) > 1:
                nearest_gap = int(path_records[1]["path_length_edges"]) - int(
                    nearest_record["path_length_edges"]
                )
            else:
                nearest_gap = 0
            if int(nearest_gap) < int(min_gap_edges):
                continue
            answer_value = str(nearest_record["label"])
            supporting_item_ids = [str(nearest_record["item_id"])]
            annotation_policy = "nearest_exit_point"
            query_details = {
                "nearest_label": str(nearest_record["label"]),
                "nearest_exit_item_id": str(nearest_record["item_id"]),
                "nearest_exit_cell": list(nearest_record["cell"]),
                "nearest_exit_path_cells": list(nearest_record["path_cells"]),
                "nearest_exit_path_length_edges": int(nearest_record["path_length_edges"]),
                "nearest_exit_margin_edges": int(nearest_gap),
                "nearest_exit_min_gap_edges": int(min_gap_edges),
                "exit_path_lengths_by_label": {
                    str(record["label"]): int(record["path_length_edges"])
                    for record in path_records
                },
                "exit_paths_by_label": {
                    str(record["label"]): [list(cell) for cell in record["path_cells"]]
                    for record in path_records
                },
            }
        else:
            raise ValueError(f"unsupported maze request_kind: {request_kind}")

        return {
            "target_reachability": str(resolved_target_reachability) if resolved_target_reachability is not None else None,
            "scene_variant": str(scene_variant),
            "request_kind": str(request_kind),
            "view_family": "topology_orthogonal_maze_exit_label",
            "topology_rule": "move_through_open_corridors_from_start_walls_block_motion",
            "maze_rows": int(rows),
            "maze_cols": int(cols),
            "maze_rows_range": [int(row_range[0]), int(row_range[1])],
            "maze_cols_range": [int(col_range[0]), int(col_range[1])],
            "start_cell": [int(start[0]), int(start[1])],
            "open_edges": [
                [[int(edge[0][0]), int(edge[0][1])], [int(edge[1][0]), int(edge[1][1])]]
                for edge in sorted(open_edges, key=lambda item: (item[0][1], item[0][0], item[1][1], item[1][0]))
            ],
            "exits": [dict(exit_spec) for exit_spec in exits],
            "exit_count": int(exit_count),
            "exit_count_range": [int(exit_count_range[0]), int(exit_count_range[1])],
            "reachable_exit_total": int(len(reachable_exits)),
            "reachable_exit_total_range": [int(len(reachable_exits)), int(len(reachable_exits))],
            "reachable_exit_labels": list(reachable_labels),
            "unreachable_exit_labels": list(unreachable_labels),
            "answer_value": answer_value,
            "supporting_item_ids": list(supporting_item_ids),
            "annotation_policy": str(annotation_policy),
            "query_details": dict(query_details),
            **dict(query_details),
            "solver_trace": {
                "start_cell": [int(start[0]), int(start[1])],
                "reachable_exit_labels": list(reachable_labels),
                "unreachable_exit_labels": list(unreachable_labels),
                "target_reachability": str(resolved_target_reachability) if resolved_target_reachability is not None else None,
                "answer_value": answer_value,
                "supporting_item_ids": list(supporting_item_ids),
                "annotation_policy": str(annotation_policy),
                **dict(query_details),
            },
        }

    raise ValueError("failed to build a maze with enough boundary leaf exits")


def sample_exit_label_maze(
    *,
    target_reachability: str,
    scene_variant: str,
    params: Mapping[str, Any],
    instance_seed: int,
    generation_defaults: Mapping[str, Any],
    max_attempts: int,
) -> Dict[str, Any]:
    """Build a maze with exactly one target reachable/unreachable exit."""

    return build_maze_exit_dataset(
        request_kind="exit_label",
        target_reachability=str(target_reachability),
        scene_variant=str(scene_variant),
        params=params,
        instance_seed=int(instance_seed),
        generation_defaults=generation_defaults,
        max_attempts=int(max_attempts),
    )


def sample_nearest_exit_maze(
    *,
    scene_variant: str,
    params: Mapping[str, Any],
    instance_seed: int,
    generation_defaults: Mapping[str, Any],
    max_attempts: int,
) -> Dict[str, Any]:
    """Build a maze with four reachable exits and one unique nearest exit."""

    return build_maze_exit_dataset(
        request_kind="nearest_exit",
        target_reachability=None,
        scene_variant=str(scene_variant),
        params=params,
        instance_seed=int(instance_seed),
        generation_defaults=generation_defaults,
        max_attempts=int(max_attempts),
    )


__all__ = [
    "sample_exit_label_maze",
    "sample_nearest_exit_maze",
]
