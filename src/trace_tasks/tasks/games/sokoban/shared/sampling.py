"""Identity-free Sokoban scene sampling primitives."""

from __future__ import annotations

from itertools import cycle
from string import ascii_uppercase
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from trace_tasks.core.sampling import shuffled_support
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.shared.color_format import format_named_color_with_hex
from trace_tasks.tasks.shared.named_colors import sample_named_color_palette
from trace_tasks.tasks.games.shared.sampling import (
    get_games_int_param as _get_int,
    get_games_int_range as _get_range,
    resolve_games_named_axis,
)
from trace_tasks.tasks.shared.support_sampling import resolve_integer_choice, resolve_integer_support

from .defaults import GEN_DEFAULTS
from .rules import (
    add_cells,
    json_safe,
    largest_component,
    manhattan,
    moves_from_path,
    sequence_description,
    sequence_text,
    simulate_grid_path,
    shortest_path,
)
from .state import (
    BOX_GOAL_DISTANCE_CONTRACT_KIND,
    BOX_GOAL_DISTANCE_OPTION_COUNT_SUPPORT,
    BOX_GOAL_STATUS_CONTRACT_KIND,
    BOX_GOAL_STATUS_COUNT_SUPPORT,
    BOX_GOAL_STATUS_MODE_OFF,
    BOX_GOAL_STATUS_MODE_ON,
    DIRECTIONS,
    PATH_CONTRACT_KIND,
    PATH_MODE_BLOCKED,
    PATH_MODE_SHORTEST,
    PATH_MODE_VALID,
    PATH_OPTION_COUNT_SUPPORT,
    PUSH_STAND_CONTRACT_KIND,
    PUSH_STAND_OPTION_COUNT_SUPPORT,
    RELATION_CONTRACT_KIND,
    RELATION_MODE_NEAREST_BOX,
    RELATION_MODE_NEAREST_TARGET,
    RELATION_MODE_RANKED_PAIR,
    RELATION_OPTION_COUNT_SUPPORT,
    SUPPORTED_SCENE_VARIANTS,
    Cell,
    SokobanAxes,
)


BOX_GOAL_PAIR_COLORS: Tuple[Tuple[int, int, int], ...] = (
    (210, 64, 72),
    (45, 116, 205),
    (55, 150, 88),
    (184, 98, 210),
    (224, 142, 46),
    (42, 156, 170),
)

_OPPOSITE_DIRECTIONS: Mapping[str, str] = {
    "U": "D",
    "D": "U",
    "L": "R",
    "R": "L",
}


def select_scene_axes(params: Mapping[str, Any], *, instance_seed: int, namespace: str) -> SokobanAxes:
    """Resolve scene-level rendering grammar axes."""

    scene_variant, scene_variant_probabilities = resolve_games_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=GEN_DEFAULTS,
        namespace="scene_variant",
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        supported_variants=SUPPORTED_SCENE_VARIANTS,
    )
    return SokobanAxes(
        scene_variant=str(scene_variant),
        scene_variant_probabilities=dict(scene_variant_probabilities),
    )


def select_option_count(
    params: Mapping[str, Any],
    *,
    instance_seed: int,
    namespace: str,
    family: str,
) -> tuple[int, tuple[int, ...], dict[str, float]]:
    """Resolve path or relation option counts from scene-level config."""

    if str(family) == "path":
        support_key = "path_option_count_support"
        fallback = PATH_OPTION_COUNT_SUPPORT
        balance_key = "balanced_path_option_count_sampling"
    elif str(family) == "relation":
        support_key = "relation_option_count_support"
        fallback = RELATION_OPTION_COUNT_SUPPORT
        balance_key = "balanced_relation_option_count_sampling"
    else:
        raise ValueError(f"unsupported Sokoban option family: {family}")

    option_count, probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=GEN_DEFAULTS,
        support_key=str(support_key),
        explicit_key="option_count",
        fallback_support=fallback,
        namespace=f"{namespace}.{family}.option_count",
        balanced_flag_key=str(balance_key),
        namespace_support_permutation=True,
    )
    support = resolve_integer_support(
        params,
        gen_defaults=GEN_DEFAULTS,
        key=str(support_key),
        fallback=fallback,
    )
    return int(option_count), tuple(int(value) for value in support), dict(probabilities)


def select_rank(option_count: int, *, instance_seed: int) -> int:
    """Choose the requested pair-rank axis for ranked relation scenes."""

    return 2 + (int(instance_seed) % min(3, int(option_count) - 1))


def select_box_goal_status_answer_count(
    params: Mapping[str, Any],
    *,
    instance_seed: int,
    namespace: str,
) -> tuple[int, tuple[int, ...], dict[str, float]]:
    """Resolve the requested box-on-goal/off-goal answer count."""

    answer_count, probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=GEN_DEFAULTS,
        support_key="box_goal_status_count_support",
        explicit_key="target_count",
        fallback_support=BOX_GOAL_STATUS_COUNT_SUPPORT,
        namespace=f"{namespace}.box_goal_status.answer_count",
        balanced_flag_key="balanced_box_goal_status_count_sampling",
        namespace_support_permutation=True,
    )
    support = resolve_integer_support(
        params,
        gen_defaults=GEN_DEFAULTS,
        key="box_goal_status_count_support",
        fallback=BOX_GOAL_STATUS_COUNT_SUPPORT,
    )
    return int(answer_count), tuple(int(value) for value in support), dict(probabilities)


def select_box_goal_distance_option_count(
    params: Mapping[str, Any],
    *,
    instance_seed: int,
    namespace: str,
) -> tuple[int, tuple[int, ...], dict[str, float]]:
    """Resolve how many labeled box-goal pairs appear in a distance comparison."""

    option_count, probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=GEN_DEFAULTS,
        support_key="box_goal_distance_option_count_support",
        explicit_key="option_count",
        fallback_support=BOX_GOAL_DISTANCE_OPTION_COUNT_SUPPORT,
        namespace=f"{namespace}.box_goal_distance.option_count",
        balanced_flag_key="balanced_box_goal_distance_option_count_sampling",
        namespace_support_permutation=True,
    )
    support = resolve_integer_support(
        params,
        gen_defaults=GEN_DEFAULTS,
        key="box_goal_distance_option_count_support",
        fallback=BOX_GOAL_DISTANCE_OPTION_COUNT_SUPPORT,
    )
    return int(option_count), tuple(int(value) for value in support), dict(probabilities)


def sample_base_board(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
    open_bias: bool = False,
) -> Dict[str, Any]:
    """Sample a connected board with border walls and sparse internal walls."""

    rng = spawn_rng(int(instance_seed), namespace)
    row_min, row_max = _get_range(
        params,
        GEN_DEFAULTS,
        min_key="board_rows_min",
        max_key="board_rows_max",
        fallback_min=6,
        fallback_max=9,
    )
    col_min, col_max = _get_range(
        params,
        GEN_DEFAULTS,
        min_key="board_cols_min",
        max_key="board_cols_max",
        fallback_min=6,
        fallback_max=9,
    )
    wall_min, wall_max = _get_range(
        params,
        GEN_DEFAULTS,
        min_key="internal_wall_count_min",
        max_key="internal_wall_count_max",
        fallback_min=2,
        fallback_max=9,
    )
    if open_bias:
        wall_max = max(0, min(int(wall_max), 4))
    for _attempt in range(256):
        rows = int(rng.randint(row_min, row_max))
        cols = int(rng.randint(col_min, col_max))
        walls: set[Cell] = set()
        for row in range(rows):
            walls.add((row, 0))
            walls.add((row, cols - 1))
        for col in range(cols):
            walls.add((0, col))
            walls.add((rows - 1, col))
        interior = [(row, col) for row in range(1, rows - 1) for col in range(1, cols - 1)]
        rng.shuffle(interior)
        wall_count = int(rng.randint(wall_min, wall_max))
        walls.update(interior[: min(wall_count, max(0, len(interior) // 4))])
        component = largest_component(rows, cols, walls)
        if len(component) >= max(12, int(0.55 * (rows - 2) * (cols - 2))):
            return {
                "rows": int(rows),
                "cols": int(cols),
                "walls": set(walls),
                "component": list(component),
            }
    raise ValueError("could not sample connected Sokoban board")


def sample_box_goal_status_dataset(
    *,
    status_mode: str,
    answer_count: int,
    params: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
) -> Dict[str, Any]:
    """Build a paired-color Sokoban board with a controlled box-goal status count."""

    if str(status_mode) not in {BOX_GOAL_STATUS_MODE_ON, BOX_GOAL_STATUS_MODE_OFF}:
        raise ValueError(f"unsupported Sokoban box-goal status mode: {status_mode}")
    rng = spawn_rng(int(instance_seed), f"{namespace}.box_goal_status.{status_mode}.{answer_count}")
    box_min, box_max = _get_range(
        params,
        GEN_DEFAULTS,
        min_key="box_goal_status_box_count_min",
        max_key="box_goal_status_box_count_max",
        fallback_min=3,
        fallback_max=5,
    )
    box_min = max(2, min(6, int(box_min)))
    box_max = max(box_min, min(6, int(box_max)))
    target_answer = int(answer_count)
    if target_answer < 1 or target_answer > 5:
        raise ValueError(f"unsupported Sokoban box-goal status answer count: {answer_count}")

    for attempt in range(256):
        minimum_boxes = max(box_min, target_answer + 1)
        if minimum_boxes > box_max:
            continue
        box_count = int(rng.randint(minimum_boxes, box_max))
        on_goal_count = target_answer if str(status_mode) == BOX_GOAL_STATUS_MODE_ON else box_count - target_answer
        if on_goal_count < 0 or on_goal_count > box_count:
            continue
        off_goal_count = box_count - on_goal_count
        if on_goal_count < 1 or off_goal_count < 1:
            continue
        board = sample_base_board(
            params=params,
            instance_seed=int(instance_seed) + attempt,
            namespace=f"{namespace}.box_goal_status.board",
            open_bias=True,
        )
        rows, cols, walls = int(board["rows"]), int(board["cols"]), set(board["walls"])
        component = list(board["component"])
        needed_cells = int(box_count + off_goal_count + 1)
        if len(component) < needed_cells:
            continue
        cells = _sample_distinct_cells(rng, component, needed_cells, forbidden=())
        labels = [f"B{idx}" for idx in range(1, box_count + 1)]
        target_labels = {box_label: f"T{box_label[1:]}" for box_label in labels}
        shuffled_labels = list(labels)
        rng.shuffle(shuffled_labels)
        on_goal_labels = set(shuffled_labels[:on_goal_count])
        target_cells = {target_labels[box_label]: tuple(cells[index]) for index, box_label in enumerate(labels)}
        off_cells = iter(cells[box_count : box_count + off_goal_count])
        boxes: Dict[str, Cell] = {}
        for box_label in labels:
            target_label = target_labels[box_label]
            boxes[box_label] = tuple(target_cells[target_label]) if box_label in on_goal_labels else tuple(next(off_cells))
        player = tuple(cells[-1])
        counted_labels = [
            box_label
            for box_label in labels
            if (box_label in on_goal_labels) == (str(status_mode) == BOX_GOAL_STATUS_MODE_ON)
        ]
        color_cycle = cycle(shuffled_support(rng, BOX_GOAL_PAIR_COLORS))
        colors = {box_label: list(next(color_cycle)) for box_label in labels}
        target_colors = {
            target_labels[box_label]: list(colors[box_label])
            for box_label in labels
        }
        matching_targets = {box_label: target_labels[box_label] for box_label in labels}
        return {
            "contract_kind": BOX_GOAL_STATUS_CONTRACT_KIND,
            "status_mode": str(status_mode),
            "rows": rows,
            "cols": cols,
            "walls": sorted([list(cell) for cell in walls]),
            "component_cells": sorted([list(cell) for cell in component]),
            "player_start": list(player),
            "boxes_start": {label: list(cell) for label, cell in sorted(boxes.items())},
            "targets": {label: list(cell) for label, cell in sorted(target_cells.items())},
            "matching_targets": dict(sorted(matching_targets.items())),
            "box_colors": dict(sorted(colors.items())),
            "target_colors": dict(sorted(target_colors.items())),
            "goal_status_count": int(target_answer),
            "box_count": int(box_count),
            "boxes_on_matching_goals": sorted(on_goal_labels),
            "boxes_off_matching_goals": sorted(label for label in labels if label not in on_goal_labels),
            "counted_box_labels": sorted(counted_labels),
            "annotation_cells": [list(boxes[label]) for label in sorted(counted_labels)],
            "option_count": 0,
            "option_specs": [],
            "solver_trace": {
                "box_count": int(box_count),
                "on_goal_count": int(on_goal_count),
                "off_goal_count": int(off_goal_count),
                "counted_box_labels": sorted(counted_labels),
            },
        }
    raise ValueError(f"could not build Sokoban box-goal status dataset for {status_mode}")


def sample_closest_box_goal_dataset(
    *,
    option_count: int,
    params: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
) -> Dict[str, Any]:
    """Build a labeled box-goal board with a unique closest Manhattan pair."""

    if int(option_count) not in BOX_GOAL_DISTANCE_OPTION_COUNT_SUPPORT:
        raise ValueError(f"unsupported Sokoban box-goal distance option count: {option_count}")
    rng = spawn_rng(int(instance_seed), f"{namespace}.closest_box_goal.{option_count}")
    for attempt in range(256):
        board = sample_base_board(
            params=params,
            instance_seed=int(instance_seed) + attempt,
            namespace=f"{namespace}.closest_box_goal.board",
            open_bias=True,
        )
        rows, cols, walls = int(board["rows"]), int(board["cols"]), set(board["walls"])
        component = list(board["component"])
        needed_cells = (2 * int(option_count)) + 1
        if len(component) < needed_cells:
            continue
        cells = _sample_distinct_cells(rng, component, needed_cells, forbidden=())
        labels = _choose_option_labels(int(option_count))
        boxes = {label: tuple(cells[index]) for index, label in enumerate(labels)}
        targets = {f"T{label}": tuple(cells[int(option_count) + index]) for index, label in enumerate(labels)}
        matching_targets = {label: f"T{label}" for label in labels}
        distances = {
            label: manhattan(boxes[label], targets[matching_targets[label]])
            for label in labels
        }
        min_distance = min(distances.values())
        if sum(1 for value in distances.values() if int(value) == int(min_distance)) != 1:
            continue
        answer_label = min(distances, key=lambda label: (int(distances[label]), str(label)))
        color_cycle = cycle(shuffled_support(rng, BOX_GOAL_PAIR_COLORS))
        colors = {label: list(next(color_cycle)) for label in labels}
        target_colors = {matching_targets[label]: list(colors[label]) for label in labels}
        option_specs = [
            {
                "kind": "box_goal_distance",
                "option_label": str(label),
                "box_label": str(label),
                "target_label": str(matching_targets[label]),
                "candidate_cells": [list(boxes[label])],
                "target_cell": list(targets[matching_targets[label]]),
                "distance": int(distances[label]),
                "is_correct": bool(str(label) == str(answer_label)),
                "option_id": f"option_{label}",
            }
            for label in labels
        ]
        return {
            "contract_kind": BOX_GOAL_DISTANCE_CONTRACT_KIND,
            "rows": rows,
            "cols": cols,
            "walls": sorted([list(cell) for cell in walls]),
            "component_cells": sorted([list(cell) for cell in component]),
            "player_start": list(cells[-1]),
            "boxes_start": {label: list(cell) for label, cell in sorted(boxes.items())},
            "targets": {label: list(cell) for label, cell in sorted(targets.items())},
            "matching_targets": dict(sorted(matching_targets.items())),
            "box_colors": dict(sorted(colors.items())),
            "target_colors": dict(sorted(target_colors.items())),
            "boxes_on_matching_goals": [],
            "boxes_off_matching_goals": list(labels),
            "show_box_labels": True,
            "option_count": int(option_count),
            "option_specs": option_specs,
            "answer_option_label": str(answer_label),
            "answer_cell": list(boxes[str(answer_label)]),
            "relation_support": {
                "distance_kind": "manhattan",
                "answer_box_label": str(answer_label),
                "answer_distance": int(distances[str(answer_label)]),
                "pair_distances": [
                    {
                        "box_label": str(label),
                        "target_label": str(matching_targets[label]),
                        "distance": int(distances[label]),
                    }
                    for label in labels
                ],
            },
            "solver_trace": {
                "distance_kind": "manhattan",
                "distances": {str(label): int(value) for label, value in sorted(distances.items())},
                "answer_box_label": str(answer_label),
                "answer_cell": list(boxes[str(answer_label)]),
            },
        }
    raise ValueError("could not build Sokoban closest box-goal dataset")


def sample_push_stand_cell_dataset(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
) -> Dict[str, Any]:
    """Build a straight-push setup with four labeled player stand cells."""

    rng = spawn_rng(int(instance_seed), f"{namespace}.push_stand_cell")
    for attempt in range(512):
        board = sample_base_board(
            params=params,
            instance_seed=int(instance_seed) + attempt,
            namespace=f"{namespace}.push_stand_cell.board",
            open_bias=True,
        )
        rows, cols, walls = int(board["rows"]), int(board["cols"]), set(board["walls"])
        component = set(tuple(cell) for cell in board["component"])
        if len(component) < 12:
            continue

        candidate_boxes = list(component)
        rng.shuffle(candidate_boxes)
        direction_keys = list(DIRECTIONS)
        rng.shuffle(direction_keys)
        for box_cell in candidate_boxes:
            adjacent_cells = {
                direction: add_cells(box_cell, delta)
                for direction, delta in DIRECTIONS.items()
            }
            if any(cell not in component for cell in adjacent_cells.values()):
                continue
            for push_direction in direction_keys:
                push_delta = DIRECTIONS[str(push_direction)]
                stand_direction = _OPPOSITE_DIRECTIONS[str(push_direction)]
                stand_cell = adjacent_cells[str(stand_direction)]
                target_distance_options = [2, 3, 4]
                rng.shuffle(target_distance_options)
                for target_distance in target_distance_options:
                    path_cells = [
                        add_cells(box_cell, (push_delta[0] * step, push_delta[1] * step))
                        for step in range(1, int(target_distance) + 1)
                    ]
                    if any(cell not in component for cell in path_cells):
                        continue
                    target_cell = path_cells[-1]
                    forbidden = {box_cell, target_cell, stand_cell, *adjacent_cells.values(), *path_cells}
                    palette = sample_named_color_palette(rng, palette_size=4)
                    if len(palette) < 4:
                        continue
                    target_color_name, target_color_rgb = palette[0]
                    distractor_colors = palette[1:]
                    open_cells = [cell for cell in component if cell not in forbidden]
                    if len(open_cells) < 3 + (2 * len(distractor_colors)):
                        continue

                    boxes: Dict[str, Cell] = {"target_box": tuple(box_cell)}
                    targets: Dict[str, Cell] = {"target_goal": tuple(target_cell)}
                    box_colors: Dict[str, List[int]] = {"target_box": list(target_color_rgb)}
                    target_colors: Dict[str, List[int]] = {"target_goal": list(target_color_rgb)}
                    matching_targets: Dict[str, str] = {"target_box": "target_goal"}

                    rng.shuffle(open_cells)
                    cursor = 0
                    for index, (color_name, color_rgb) in enumerate(distractor_colors[:2], start=1):
                        box_label = f"distractor_box_{index}"
                        target_label = f"distractor_goal_{index}"
                        boxes[box_label] = tuple(open_cells[cursor])
                        targets[target_label] = tuple(open_cells[cursor + 1])
                        box_colors[box_label] = list(color_rgb)
                        target_colors[target_label] = list(color_rgb)
                        matching_targets[box_label] = target_label
                        cursor += 2

                    occupied_by_boxes = set(boxes.values())
                    passable = set(component) - occupied_by_boxes
                    player_candidates = [
                        cell
                        for cell in open_cells[cursor:]
                        if cell in passable and shortest_path(passable, cell, stand_cell) is not None
                    ]
                    if not player_candidates:
                        continue
                    player = tuple(player_candidates[int(rng.randrange(len(player_candidates)))])

                    correct = {
                        "kind": "stand_cell",
                        "display_text": "stand",
                        "candidate_cells": [list(stand_cell)],
                        "stand_direction": str(stand_direction),
                    }
                    distractors = [
                        {
                            "kind": "stand_cell",
                            "display_text": "stand",
                            "candidate_cells": [list(cell)],
                            "stand_direction": str(direction),
                        }
                        for direction, cell in adjacent_cells.items()
                        if tuple(cell) != tuple(stand_cell)
                    ]
                    option_specs, answer_label = _assign_option_labels(
                        correct=correct,
                        distractors=distractors,
                        option_count=4,
                        instance_seed=int(instance_seed),
                    )
                    color_label = format_named_color_with_hex(str(target_color_name), target_color_rgb)
                    support = {
                        "target_color_name": str(target_color_name),
                        "target_color_rgb": list(target_color_rgb),
                        "target_color_label": str(color_label),
                        "target_box_label": "target_box",
                        "target_goal_label": "target_goal",
                        "target_box_cell": list(box_cell),
                        "target_goal_cell": list(target_cell),
                        "push_direction": str(push_direction),
                        "stand_direction": str(stand_direction),
                        "stand_cell": list(stand_cell),
                        "straight_path_cells": [list(cell) for cell in path_cells],
                        "correct_option_label": str(answer_label),
                    }
                    return {
                        "contract_kind": PUSH_STAND_CONTRACT_KIND,
                        "rows": rows,
                        "cols": cols,
                        "walls": sorted([list(cell) for cell in walls]),
                        "component_cells": sorted([list(cell) for cell in component]),
                        "player_start": list(player),
                        "boxes_start": {label: list(cell) for label, cell in sorted(boxes.items())},
                        "targets": {label: list(cell) for label, cell in sorted(targets.items())},
                        "matching_targets": dict(sorted(matching_targets.items())),
                        "box_colors": dict(sorted(box_colors.items())),
                        "target_colors": dict(sorted(target_colors.items())),
                        "option_count": 4,
                        "option_specs": option_specs,
                        "answer_option_label": str(answer_label),
                        "answer_cell": list(stand_cell),
                        "target_color_name": str(target_color_name),
                        "target_color_label": str(color_label),
                        "target_box_label": "target_box",
                        "target_goal_label": "target_goal",
                        "relation_support": support,
                        "solver_trace": dict(support),
                    }
    raise ValueError("could not build Sokoban push stand-cell dataset")


def _choose_option_labels(option_count: int) -> List[str]:
    return list(ascii_uppercase[: int(option_count)])


def _assign_option_labels(
    *,
    correct: Dict[str, Any],
    distractors: Sequence[Dict[str, Any]],
    option_count: int,
    instance_seed: int,
    cycle_stride: int = 3,
) -> Tuple[List[Dict[str, Any]], str]:
    labels = _choose_option_labels(option_count)
    correct_index = (int(instance_seed) // max(1, int(cycle_stride))) % int(option_count)
    option_specs: List[Dict[str, Any]] = []
    distractor_iter = iter(list(distractors))
    for index, label in enumerate(labels):
        payload = dict(correct) if int(index) == int(correct_index) else dict(next(distractor_iter))
        payload["option_label"] = str(label)
        payload["is_correct"] = bool(int(index) == int(correct_index))
        payload["option_id"] = f"option_{label}"
        option_specs.append(payload)
    return option_specs, str(labels[correct_index])


def _sample_distinct_cells(
    rng,
    cells: Sequence[Cell],
    count: int,
    *,
    forbidden: Iterable[Cell] = (),
) -> List[Cell]:
    forbidden_set = set(forbidden)
    candidates = [tuple(cell) for cell in cells if tuple(cell) not in forbidden_set]
    rng.shuffle(candidates)
    if len(candidates) < int(count):
        raise ValueError("not enough available Sokoban cells")
    return candidates[: int(count)]


def _mutated_sequence(base: Sequence[str], rng, *, alphabet: Sequence[str] = tuple(DIRECTIONS.keys())) -> List[str]:
    seq = [str(item) for item in base]
    if not seq:
        return [str(alphabet[int(rng.randrange(len(alphabet)))])]
    op = int(rng.randrange(3))
    if op == 0:
        idx = int(rng.randrange(len(seq)))
        choices = [move for move in alphabet if move != seq[idx]]
        seq[idx] = str(choices[int(rng.randrange(len(choices)))])
    elif op == 1 and len(seq) > 2:
        del seq[int(rng.randrange(len(seq)))]
    else:
        seq.insert(int(rng.randrange(len(seq) + 1)), str(alphabet[int(rng.randrange(len(alphabet)))]))
    return seq


def sample_path_sequence_dataset(
    *,
    path_mode: str,
    option_count: int,
    params: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
) -> Dict[str, Any]:
    """Build one path-option Sokoban dataset for a task-owned path mode."""

    rng = spawn_rng(int(instance_seed), f"{namespace}.path.{path_mode}")
    dist_min, dist_max = _get_range(
        params,
        GEN_DEFAULTS,
        min_key="path_length_min",
        max_key="path_length_max",
        fallback_min=4,
        fallback_max=12,
    )
    for attempt in range(256):
        board = sample_base_board(
            params=params,
            instance_seed=int(instance_seed) + attempt,
            namespace=f"{namespace}.path.board",
            open_bias=False,
        )
        rows, cols, walls = int(board["rows"]), int(board["cols"]), set(board["walls"])
        component = list(board["component"])
        box_count = int(rng.randint(1, 3))
        boxes_cells = _sample_distinct_cells(rng, component, box_count, forbidden=())
        boxes = {f"B{idx}": tuple(cell) for idx, cell in enumerate(boxes_cells, start=1)}
        passable = set(component) - set(boxes.values())
        candidate_pairs = []
        shuffled = list(passable)
        rng.shuffle(shuffled)
        for start in shuffled[: min(len(shuffled), 36)]:
            for goal in shuffled:
                if start == goal:
                    continue
                path = shortest_path(passable, start, goal)
                if path is None:
                    continue
                length = len(path) - 1
                if dist_min <= length <= dist_max:
                    candidate_pairs.append((start, goal, path))
            if candidate_pairs:
                break
        if not candidate_pairs:
            continue
        start, goal, path = candidate_pairs[int(rng.randrange(len(candidate_pairs)))]
        shortest_moves = moves_from_path(path)
        if str(path_mode) == PATH_MODE_VALID:
            correct_moves = list(shortest_moves)
            if len(path) >= 2:
                back = path[-2]
                out_move = moves_from_path([goal, back])[0]
                in_move = moves_from_path([back, goal])[0]
                correct_moves = list(shortest_moves) + [out_move, in_move]
            correct_kind = "valid_path"
        elif str(path_mode) == PATH_MODE_BLOCKED:
            prefix = list(shortest_moves[: max(1, len(shortest_moves) // 2)])
            prefix_end = simulate_grid_path(passable, start, prefix)["end"]
            blocked_moves = [
                move
                for move, delta in DIRECTIONS.items()
                if add_cells(prefix_end, delta) not in passable
            ]
            if not blocked_moves:
                continue
            correct_moves = prefix + [str(blocked_moves[int(rng.randrange(len(blocked_moves)))])]
            correct_kind = "blocked_path"
        elif str(path_mode) == PATH_MODE_SHORTEST:
            correct_moves = list(shortest_moves)
            correct_kind = "shortest_path"
        else:
            raise ValueError(f"unsupported Sokoban path mode: {path_mode}")
        seen = {sequence_text(correct_moves)}
        distractors: List[Dict[str, Any]] = []
        for _ in range(512):
            if len(distractors) >= int(option_count) - 1:
                break
            candidate = _mutated_sequence(shortest_moves, rng)
            sim = simulate_grid_path(passable, start, candidate)
            if str(path_mode) == PATH_MODE_BLOCKED:
                if sim["blocked_at_step"] is not None:
                    continue
            elif sim["blocked_at_step"] is None and sim["end"] == goal and (
                str(path_mode) != PATH_MODE_SHORTEST or len(candidate) == len(shortest_moves)
            ):
                continue
            key = sequence_text(candidate)
            if key in seen:
                continue
            seen.add(key)
            distractors.append({"kind": "move_sequence", "moves": list(candidate), "display_text": sequence_text(candidate)})
        if len(distractors) < int(option_count) - 1:
            continue
        correct = {"kind": "move_sequence", "moves": list(correct_moves), "display_text": sequence_text(correct_moves)}
        option_specs, answer_label = _assign_option_labels(
            correct=correct,
            distractors=distractors,
            option_count=int(option_count),
            instance_seed=int(instance_seed),
        )
        return {
            "contract_kind": PATH_CONTRACT_KIND,
            "path_mode": str(path_mode),
            "rows": rows,
            "cols": cols,
            "walls": sorted([list(cell) for cell in walls]),
            "component_cells": sorted([list(cell) for cell in component]),
            "player_start": list(start),
            "boxes_start": {label: list(cell) for label, cell in sorted(boxes.items())},
            "targets": {"G": list(goal)},
            "path_start": list(start),
            "path_goal": list(goal),
            "shortest_path_cells": [list(cell) for cell in path],
            "shortest_moves": list(shortest_moves),
            "correct_sequence_kind": str(correct_kind),
            "move_sequence": list(correct_moves),
            "move_sequence_text": sequence_text(correct_moves),
            "move_sequence_description": sequence_description(correct_moves),
            "option_count": int(option_count),
            "option_specs": option_specs,
            "answer_option_label": str(answer_label),
            "solver_trace": {
                "passable_cells": sorted([list(cell) for cell in passable]),
                "correct_sequence_simulation": simulate_grid_path(passable, start, correct_moves),
            },
        }
    raise ValueError(f"could not build Sokoban path-sequence dataset for {path_mode}")


def sample_relation_dataset(
    *,
    relation_mode: str,
    option_count: int,
    rank: int | None,
    params: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
) -> Dict[str, Any]:
    """Build one relation-option Sokoban dataset for a task-owned relation mode."""

    rng = spawn_rng(int(instance_seed), f"{namespace}.relation.{relation_mode}")
    for attempt in range(256):
        needed = 2 * int(option_count) + 1 if str(relation_mode) == RELATION_MODE_RANKED_PAIR else int(option_count) + 2
        board = sample_base_board(
            params=params,
            instance_seed=int(instance_seed) + attempt,
            namespace=f"{namespace}.relation.board",
            open_bias=True,
        )
        rows, cols, walls = int(board["rows"]), int(board["cols"]), set(board["walls"])
        component = list(board["component"])
        if len(component) < int(needed):
            continue
        cells = _sample_distinct_cells(rng, component, needed, forbidden=())
        player = tuple(cells[0])
        if str(relation_mode) == RELATION_MODE_NEAREST_TARGET:
            boxes = {"B1": tuple(cells[1])}
            targets = {f"T{idx}": tuple(cell) for idx, cell in enumerate(cells[2 : 2 + int(option_count)], start=1)}
            distances = {label: manhattan(boxes["B1"], cell) for label, cell in targets.items()}
            if len(set(distances.values())) < len(distances):
                continue
            answer_target = min(distances, key=lambda label: distances[label])
            correct = {
                "kind": "target_label",
                "display_text": str(answer_target),
                "target_label": str(answer_target),
                "candidate_cells": [list(targets[str(answer_target)])],
            }
            distractors = [
                {
                    "kind": "target_label",
                    "display_text": str(label),
                    "target_label": str(label),
                    "candidate_cells": [list(targets[str(label)])],
                }
                for label in sorted(targets)
                if label != answer_target
            ]
            marked_box_label = "B1"
            marked_target_label = ""
            query_entity_type = "box"
            query_entity_label = "B1"
            answer_cell = list(targets[str(answer_target)])
            support = {"answer_target_label": str(answer_target), "distances": dict(distances)}
        elif str(relation_mode) == RELATION_MODE_NEAREST_BOX:
            target = tuple(cells[1])
            targets = {"T1": target}
            boxes = {f"B{idx}": tuple(cell) for idx, cell in enumerate(cells[2 : 2 + int(option_count)], start=1)}
            distances = {label: manhattan(cell, target) for label, cell in boxes.items()}
            if len(set(distances.values())) < len(distances):
                continue
            answer_box = min(distances, key=lambda label: distances[label])
            correct = {
                "kind": "box_label",
                "display_text": str(answer_box),
                "box_label": str(answer_box),
                "candidate_cells": [list(boxes[str(answer_box)])],
            }
            distractors = [
                {
                    "kind": "box_label",
                    "display_text": str(label),
                    "box_label": str(label),
                    "candidate_cells": [list(boxes[str(label)])],
                }
                for label in sorted(boxes)
                if label != answer_box
            ]
            marked_box_label = ""
            marked_target_label = "T1"
            query_entity_type = "target"
            query_entity_label = "T1"
            answer_cell = list(boxes[str(answer_box)])
            support = {"answer_box_label": str(answer_box), "distances": dict(distances)}
        elif str(relation_mode) == RELATION_MODE_RANKED_PAIR:
            resolved_rank = select_rank(int(option_count), instance_seed=int(instance_seed)) if rank is None else int(rank)
            box_cells = cells[1 : 1 + int(option_count)]
            target_cells = cells[1 + int(option_count) : 1 + (2 * int(option_count))]
            boxes = {f"B{idx}": tuple(cell) for idx, cell in enumerate(box_cells, start=1)}
            targets = {f"T{idx}": tuple(cell) for idx, cell in enumerate(target_cells, start=1)}
            paired_labels = [
                (f"B{idx}", f"T{idx}", manhattan(boxes[f"B{idx}"], targets[f"T{idx}"]))
                for idx in range(1, int(option_count) + 1)
            ]
            if len(set(dist for _b, _t, dist in paired_labels)) < len(paired_labels):
                continue
            pair_options = sorted(paired_labels, key=lambda item: (item[2], item[0], item[1]))
            answer_pair = pair_options[int(resolved_rank) - 1]
            correct = {
                "kind": "pair_label",
                "display_text": f"{answer_pair[0]}-{answer_pair[1]}",
                "box_label": str(answer_pair[0]),
                "target_label": str(answer_pair[1]),
                "candidate_cells": [list(boxes[str(answer_pair[0])]), list(targets[str(answer_pair[1])])],
            }
            distractors = [
                {
                    "kind": "pair_label",
                    "display_text": f"{box_label}-{target_label}",
                    "box_label": str(box_label),
                    "target_label": str(target_label),
                    "candidate_cells": [list(boxes[str(box_label)]), list(targets[str(target_label)])],
                }
                for box_label, target_label, _dist in pair_options
                if (box_label, target_label) != (answer_pair[0], answer_pair[1])
            ]
            marked_box_label = ""
            marked_target_label = ""
            query_entity_type = "pair"
            query_entity_label = ""
            answer_cell = [list(boxes[str(answer_pair[0])]), list(targets[str(answer_pair[1])])]
            support = {
                "rank": int(resolved_rank),
                "rank_word": {2: "second", 3: "third", 4: "fourth"}.get(int(resolved_rank), str(resolved_rank)),
                "answer_pair": [str(answer_pair[0]), str(answer_pair[1])],
                "pair_distances": [
                    {"box_label": str(b), "target_label": str(t), "distance": int(d)}
                    for b, t, d in paired_labels
                ],
            }
        else:
            raise ValueError(f"unsupported Sokoban relation mode: {relation_mode}")
        if len(distractors) < int(option_count) - 1:
            continue
        option_specs, answer_label = _assign_option_labels(
            correct=correct,
            distractors=distractors,
            option_count=int(option_count),
            instance_seed=int(instance_seed),
        )
        return {
            "contract_kind": RELATION_CONTRACT_KIND,
            "relation_mode": str(relation_mode),
            "rows": rows,
            "cols": cols,
            "walls": sorted([list(cell) for cell in walls]),
            "component_cells": sorted([list(cell) for cell in component]),
            "player_start": list(player),
            "boxes_start": {label: list(cell) for label, cell in sorted(boxes.items())},
            "targets": {label: list(cell) for label, cell in sorted(targets.items())},
            "marked_box_label": str(marked_box_label),
            "marked_target_label": str(marked_target_label),
            "query_entity_type": str(query_entity_type),
            "query_entity_label": str(query_entity_label),
            "answer_cell": json_safe(answer_cell),
            "relation_support": support,
            "option_count": int(option_count),
            "option_specs": option_specs[: int(option_count)],
            "answer_option_label": str(answer_label),
            "solver_trace": dict(support),
        }
    raise ValueError(f"could not build Sokoban relation dataset for {relation_mode}")


__all__ = [
    "sample_base_board",
    "sample_box_goal_status_dataset",
    "sample_closest_box_goal_dataset",
    "sample_push_stand_cell_dataset",
    "sample_path_sequence_dataset",
    "sample_relation_dataset",
    "select_box_goal_distance_option_count",
    "select_box_goal_status_answer_count",
    "select_option_count",
    "select_rank",
    "select_scene_axes",
]
