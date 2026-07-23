"""Sampling primitives for rule-override board scene-package tasks."""

from __future__ import annotations

from typing import Any, Mapping, Sequence, Tuple

from trace_tasks.tasks.games.shared.sampling import resolve_games_named_axis
from trace_tasks.tasks.shared.support_sampling import resolve_integer_choice, resolve_integer_support

from .defaults import DEFAULTS
from .rules import (
    has_full_line,
    line_result_from_target_line,
    opponent,
    piece_result_from_counts,
    target_fewer_needed_for_result,
    target_line_needed_for_result,
)
from .state import (
    BoardPanel,
    LINE_BOARD_FAMILY,
    LINE_PLAYERS,
    PIECE_BOARD_FAMILY,
    PIECE_PLAYERS,
    RuleOverrideAxes,
    RuleOverrideSceneSample,
    SCENE_NAMESPACE,
    SUPPORTED_BOARD_STYLES,
)


def resolve_rule_override_axes(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    board_family: str,
    player_support: Sequence[str],
    board_size_support_key: str,
    board_size_fallback: Sequence[int],
) -> RuleOverrideAxes:
    """Resolve scene axes without using public task or query identities."""

    board_style, board_style_probabilities = resolve_games_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        namespace="board_style",
        explicit_key="board_style",
        weights_key="board_style_weights",
        balance_flag_key="balanced_board_style_sampling",
        supported_variants=SUPPORTED_BOARD_STYLES,
    )
    target_player, target_player_probabilities = resolve_games_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        namespace=f"{str(board_family)}.target_player",
        explicit_key="target_player",
        weights_key=f"{str(board_family)}_target_player_weights",
        balance_flag_key="balanced_target_player_sampling",
        supported_variants=tuple(str(value) for value in player_support),
    )
    board_count, board_count_probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        support_key="board_count_support",
        explicit_key="board_count",
        fallback_support=DEFAULTS.board_count_support,
        namespace=f"{SCENE_NAMESPACE}.board_count",
        balanced_flag_key="balanced_board_count_sampling",
        namespace_support_permutation=True,
    )
    board_size, board_size_probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        support_key=str(board_size_support_key),
        explicit_key="board_size",
        fallback_support=tuple(int(value) for value in board_size_fallback),
        namespace=f"{SCENE_NAMESPACE}.{str(board_family)}.board_size",
        balanced_flag_key="balanced_board_size_sampling",
        namespace_support_permutation=True,
    )
    raw_answer_support = resolve_integer_support(
        params,
        gen_defaults=gen_defaults,
        key="target_answer_support",
        fallback=DEFAULTS.target_answer_support,
    )
    conditional_answer_support = tuple(int(value) for value in raw_answer_support if 0 <= int(value) <= int(board_count))
    if not conditional_answer_support:
        conditional_answer_support = tuple(range(0, int(board_count) + 1))
    answer_params = {**dict(params), "target_answer_support": conditional_answer_support}
    target_answer, target_answer_probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=answer_params,
        gen_defaults={},
        support_key="target_answer_support",
        explicit_key="target_answer",
        fallback_support=conditional_answer_support,
        namespace=f"{SCENE_NAMESPACE}.{str(board_family)}.answer",
        balanced_flag_key="balanced_target_answer_sampling",
        namespace_support_permutation=True,
    )
    return RuleOverrideAxes(
        board_family=str(board_family),
        board_style=str(board_style),
        target_player=str(target_player),
        board_count=int(board_count),
        board_size=int(board_size),
        target_answer=int(target_answer),
        board_style_probabilities=dict(board_style_probabilities),
        target_player_probabilities=dict(target_player_probabilities),
        board_count_probabilities=dict(board_count_probabilities),
        board_size_probabilities=dict(board_size_probabilities),
        target_answer_probabilities=dict(target_answer_probabilities),
    )


def resolve_line_axes(*, instance_seed: int, params: Mapping[str, Any], gen_defaults: Mapping[str, Any]) -> RuleOverrideAxes:
    """Resolve axes for line-board samples."""

    return resolve_rule_override_axes(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        board_family=LINE_BOARD_FAMILY,
        player_support=LINE_PLAYERS,
        board_size_support_key="line_board_size_support",
        board_size_fallback=DEFAULTS.line_board_size_support,
    )


def resolve_piece_axes(*, instance_seed: int, params: Mapping[str, Any], gen_defaults: Mapping[str, Any]) -> RuleOverrideAxes:
    """Resolve axes for piece-count board samples."""

    return resolve_rule_override_axes(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        board_family=PIECE_BOARD_FAMILY,
        player_support=PIECE_PLAYERS,
        board_size_support_key="piece_board_size_support",
        board_size_fallback=DEFAULTS.piece_board_size_support,
    )


def line_cells_with_condition(*, rng: Any, size: int, target_player: str, has_target_line: bool) -> Tuple[Tuple[str, ...], ...]:
    """Sample an X/O board with or without a full target-player line."""

    target = str(target_player)
    other = opponent(target)
    if bool(has_target_line):
        cells = [["" for _ in range(int(size))] for _ in range(int(size))]
        line_type = str(rng.choice(("row", "col", "diag_down", "diag_up")))
        line_index = int(rng.randrange(0, int(size)))
        for index in range(int(size)):
            if line_type == "row":
                cells[line_index][index] = target
            elif line_type == "col":
                cells[index][line_index] = target
            elif line_type == "diag_down":
                cells[index][index] = target
            else:
                cells[index][int(size) - 1 - index] = target
        for row in range(int(size)):
            for col in range(int(size)):
                if cells[row][col]:
                    continue
                cells[row][col] = str(rng.choices(("", target, other), weights=(0.30, 0.18, 0.52), k=1)[0])
        return tuple(tuple(value for value in row) for row in cells)

    for _attempt in range(500):
        cells = []
        for _row in range(int(size)):
            cells.append(
                tuple(
                    str(rng.choices(("", target, other), weights=(0.35, 0.22, 0.43), k=1)[0])
                    for _col in range(int(size))
                )
            )
        if not has_full_line(cells, target):
            return tuple(cells)
    raise ValueError("failed to sample line board without target line")


def piece_cells_with_condition(*, rng: Any, size: int, target_player: str, target_has_fewer: bool) -> Tuple[Tuple[str, ...], ...]:
    """Sample a token board where the target player has fewer or more pieces."""

    target = str(target_player)
    other = opponent(target)
    total_cells = int(size) * int(size)
    if bool(target_has_fewer):
        target_count = int(rng.randrange(2, max(3, min(7, total_cells // 2))))
        other_count = int(rng.randrange(target_count + 1, min(total_cells - target_count, target_count + 6) + 1))
    else:
        other_count = int(rng.randrange(2, max(3, min(7, total_cells // 2))))
        target_count = int(rng.randrange(other_count + 1, min(total_cells - other_count, other_count + 6) + 1))
    positions = [(row, col) for row in range(int(size)) for col in range(int(size))]
    rng.shuffle(positions)
    cells = [["" for _ in range(int(size))] for _ in range(int(size))]
    for row, col in positions[:target_count]:
        cells[int(row)][int(col)] = target
    for row, col in positions[target_count : target_count + other_count]:
        cells[int(row)][int(col)] = other
    return tuple(tuple(value for value in row) for row in cells)


def sample_line_result_scene(*, rng: Any, axes: RuleOverrideAxes, target_result: str, rule_text: str) -> RuleOverrideSceneSample:
    """Construct line mini-boards with exactly the requested result count."""

    counted_flags = [True] * int(axes.target_answer) + [False] * int(axes.board_count - axes.target_answer)
    rng.shuffle(counted_flags)
    boards: list[BoardPanel] = []
    annotation_ids: list[str] = []
    for index, counted in enumerate(counted_flags):
        board_id = f"board_{int(index) + 1:02d}"
        has_target_line = target_line_needed_for_result(target_result=str(target_result), counted=bool(counted))
        cells = line_cells_with_condition(
            rng=rng,
            size=int(axes.board_size),
            target_player=str(axes.target_player),
            has_target_line=bool(has_target_line),
        )
        result = line_result_from_target_line(bool(has_target_line))
        target_stat = 1 if bool(has_target_line) else 0
        opponent_stat = 1 if has_full_line(cells, opponent(str(axes.target_player))) else 0
        if bool(counted):
            annotation_ids.append(board_id)
        boards.append(
            BoardPanel(
                board_id=str(board_id),
                label=f"Board {int(index) + 1}",
                cells=tuple(tuple(str(value) for value in row) for row in cells),
                counted=bool(counted),
                result=str(result),
                target_player=str(axes.target_player),
                target_stat=int(target_stat),
                opponent_stat=int(opponent_stat),
            )
        )
    return _build_scene_sample(axes=axes, rule_text=str(rule_text), boards=tuple(boards), annotation_ids=tuple(annotation_ids))


def sample_piece_result_scene(*, rng: Any, axes: RuleOverrideAxes, target_result: str, rule_text: str) -> RuleOverrideSceneSample:
    """Construct piece-count mini-boards with exactly the requested result count."""

    counted_flags = [True] * int(axes.target_answer) + [False] * int(axes.board_count - axes.target_answer)
    rng.shuffle(counted_flags)
    boards: list[BoardPanel] = []
    annotation_ids: list[str] = []
    for index, counted in enumerate(counted_flags):
        board_id = f"board_{int(index) + 1:02d}"
        target_has_fewer = target_fewer_needed_for_result(target_result=str(target_result), counted=bool(counted))
        cells = piece_cells_with_condition(
            rng=rng,
            size=int(axes.board_size),
            target_player=str(axes.target_player),
            target_has_fewer=bool(target_has_fewer),
        )
        flat = [value for row in cells for value in row]
        target_stat = sum(1 for value in flat if str(value) == str(axes.target_player))
        opponent_stat = sum(1 for value in flat if str(value) == opponent(str(axes.target_player)))
        result = piece_result_from_counts(target_count=int(target_stat), opponent_count=int(opponent_stat))
        if bool(counted):
            annotation_ids.append(board_id)
        boards.append(
            BoardPanel(
                board_id=str(board_id),
                label=f"Board {int(index) + 1}",
                cells=tuple(tuple(str(value) for value in row) for row in cells),
                counted=bool(counted),
                result=str(result),
                target_player=str(axes.target_player),
                target_stat=int(target_stat),
                opponent_stat=int(opponent_stat),
            )
        )
    return _build_scene_sample(axes=axes, rule_text=str(rule_text), boards=tuple(boards), annotation_ids=tuple(annotation_ids))


def _build_scene_sample(
    *,
    axes: RuleOverrideAxes,
    rule_text: str,
    boards: Tuple[BoardPanel, ...],
    annotation_ids: Tuple[str, ...],
) -> RuleOverrideSceneSample:
    """Build and validate the shared sampled-scene record."""

    sample = RuleOverrideSceneSample(
        board_family=str(axes.board_family),
        board_style=str(axes.board_style),
        target_player=str(axes.target_player),
        board_size=int(axes.board_size),
        answer=int(axes.target_answer),
        rule_text=str(rule_text),
        boards=tuple(boards),
        annotation_entity_ids=tuple(annotation_ids),
    )
    actual_answer = sum(1 for board in sample.boards if bool(board.counted))
    if int(actual_answer) != int(axes.target_answer):
        raise ValueError("sampled answer does not match requested target answer")
    return sample


__all__ = [
    "line_cells_with_condition",
    "piece_cells_with_condition",
    "resolve_line_axes",
    "resolve_piece_axes",
    "resolve_rule_override_axes",
    "sample_line_result_scene",
    "sample_piece_result_scene",
]
