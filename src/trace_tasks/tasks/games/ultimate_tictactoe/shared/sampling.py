"""Identity-free sampling helpers for Ultimate Tic-Tac-Toe boards."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Tuple

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.shared.support_sampling import resolve_integer_choice
from trace_tasks.tasks.shared.variant_sampling import apply_balanced_variant_sampling, resolve_variant

from .defaults import DEFAULTS, GEN_DEFAULTS
from .rules import (
    board_entity_id,
    cell_entity_id,
    immediate_winning_cells,
    local_board_for_status,
    open_cells_without_immediate_win,
    opponent_of,
    status_of,
)
from .state import (
    LOCAL_LINES,
    MACRO_LABELS,
    OPTION_LABELS,
    PLAYER_O,
    PLAYER_X,
    STATUS_DRAWN,
    STATUS_NEITHER_WON,
    STATUS_O_WON,
    STATUS_OPEN,
    STATUS_X_WON,
    UltimateLocalBoard,
    UltimateSample,
    SUPPORTED_STYLE_VARIANTS,
)


def inner_cycle_params(params: Mapping[str, Any], *, branch_count: int) -> dict[str, Any]:
    """Cycle task branches before target-answer axes."""

    resolved = dict(params)
    sampling_index = params.get("_sample_cursor")
    if sampling_index is None:
        return resolved
    resolved["_sample_cursor"] = abs(int(sampling_index)) // max(1, int(branch_count))
    return resolved


def sample_integer_axis(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    support_key: str,
    explicit_key: str,
    fallback_support: Sequence[int],
    namespace: str,
    balanced_flag_key: str,
) -> tuple[int, dict[str, float]]:
    """Resolve one balanced integer axis for task-owned construction."""

    value, probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=GEN_DEFAULTS,
        support_key=str(support_key),
        explicit_key=str(explicit_key),
        fallback_support=tuple(int(item) for item in fallback_support),
        namespace=str(namespace),
        balanced_flag_key=str(balanced_flag_key),
        namespace_support_permutation=True,
    )
    return int(value), dict(probabilities)


def sample_style_variant(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    namespace: str,
) -> tuple[str, dict[str, float]]:
    """Resolve the scene-local board style axis with standard games weighting."""

    rng = spawn_rng(int(instance_seed), f"{str(namespace)}.style_variant")
    style_variant, probabilities = resolve_variant(
        rng,
        params=params,
        gen_defaults=GEN_DEFAULTS,
        explicit_key="style_variant",
        weights_key="style_variant_weights",
        supported_variants=[str(item) for item in SUPPORTED_STYLE_VARIANTS],
    )
    style_variant = apply_balanced_variant_sampling(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=GEN_DEFAULTS,
        selected_variant=str(style_variant),
        variant_probabilities=probabilities,
        supported_variants=[str(item) for item in SUPPORTED_STYLE_VARIANTS],
        balance_flag_key="balanced_style_variant_sampling",
        explicit_key="style_variant",
        weights_key="style_variant_weights",
        sampling_namespace=f"{str(namespace)}.style_variant",
    )
    return str(style_variant), dict(probabilities)


def matching_status_indices(board: Sequence[UltimateLocalBoard], target_status: str) -> Tuple[int, ...]:
    """Return small-board indices matching the requested status category."""

    if str(target_status) == STATUS_X_WON:
        return tuple(index for index, local in enumerate(board) if local.status == STATUS_X_WON)
    if str(target_status) == STATUS_O_WON:
        return tuple(index for index, local in enumerate(board) if local.status == STATUS_O_WON)
    if str(target_status) == STATUS_DRAWN:
        return tuple(index for index, local in enumerate(board) if local.status == STATUS_DRAWN)
    return tuple(index for index, local in enumerate(board) if local.status in {STATUS_DRAWN, STATUS_OPEN})


def counts_for_status_target(rng, *, target_status: str, target_answer: int) -> dict[str, int]:
    """Choose local-board status counts while preserving the requested answer."""

    counts = {STATUS_X_WON: 0, STATUS_O_WON: 0, STATUS_DRAWN: 0, STATUS_OPEN: 0}
    if str(target_status) == STATUS_X_WON:
        counts[STATUS_X_WON] = int(target_answer)
        remaining = 9 - int(target_answer)
        counts[STATUS_O_WON] = int(rng.randrange(1, min(4, remaining) + 1))
        remaining -= counts[STATUS_O_WON]
        counts[STATUS_DRAWN] = int(rng.randrange(0, min(3, remaining) + 1))
        counts[STATUS_OPEN] = remaining - counts[STATUS_DRAWN]
    elif str(target_status) == STATUS_O_WON:
        counts[STATUS_O_WON] = int(target_answer)
        remaining = 9 - int(target_answer)
        counts[STATUS_X_WON] = int(rng.randrange(1, min(4, remaining) + 1))
        remaining -= counts[STATUS_X_WON]
        counts[STATUS_DRAWN] = int(rng.randrange(0, min(3, remaining) + 1))
        counts[STATUS_OPEN] = remaining - counts[STATUS_DRAWN]
    elif str(target_status) == STATUS_DRAWN:
        counts[STATUS_DRAWN] = int(target_answer)
        remaining = 9 - int(target_answer)
        if remaining < 2:
            raise ValueError("drawn-board target leaves no room for both winners")
        counts[STATUS_X_WON] = int(rng.randrange(1, min(4, remaining - 1) + 1))
        remaining -= counts[STATUS_X_WON]
        counts[STATUS_O_WON] = int(rng.randrange(1, min(4, remaining) + 1))
        remaining -= counts[STATUS_O_WON]
        counts[STATUS_OPEN] = remaining
    else:
        neither = int(target_answer)
        counts[STATUS_DRAWN] = int(rng.randrange(0, min(3, neither) + 1))
        counts[STATUS_OPEN] = neither - counts[STATUS_DRAWN]
        remaining = 9 - neither
        counts[STATUS_X_WON] = int(rng.randrange(0, remaining + 1))
        counts[STATUS_O_WON] = remaining - counts[STATUS_X_WON]
    return counts


def sample_status_board_count(
    rng,
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    target_status: str,
    support_key: str,
    fallback_support: Sequence[int],
    namespace: str,
    branch_count: int,
) -> UltimateSample:
    """Sample a board whose small-board status count equals a target answer."""

    target_answer, target_probabilities = sample_integer_axis(
        instance_seed=int(instance_seed),
        params=inner_cycle_params(params, branch_count=int(branch_count)),
        support_key=str(support_key),
        explicit_key="target_answer",
        fallback_support=tuple(int(value) for value in fallback_support),
        namespace=f"{str(namespace)}.target_answer",
        balanced_flag_key="balanced_target_answer_sampling",
    )
    counts = counts_for_status_target(rng, target_status=str(target_status), target_answer=int(target_answer))
    statuses: list[str] = []
    for status, count in counts.items():
        statuses.extend([str(status)] * int(count))
    if len(statuses) != 9:
        raise ValueError(f"status count construction produced {len(statuses)} boards")
    rng.shuffle(statuses)
    board = tuple(local_board_for_status(rng, status) for status in statuses)
    matching = matching_status_indices(board, str(target_status))
    if len(matching) != int(target_answer):
        raise ValueError("status target count mismatch")
    annotation_ids = tuple(board_entity_id(index) for index in matching)
    return UltimateSample(
        board=tuple(board),
        answer=int(target_answer),
        answer_type="integer",
        target_answer=int(target_answer),
        highlighted_board_index=None,
        option_cells=(),
        answer_cell=None,
        support_cells=(),
        annotation_entity_ids=tuple(annotation_ids),
        metadata={
            "target_answer": int(target_answer),
            "target_answer_probabilities": dict(target_probabilities),
            "target_status": str(target_status),
            "status_counts": dict(counts),
            "matching_small_boards": [MACRO_LABELS[index] for index in matching],
        },
    )


def make_tactic_cells(
    rng,
    *,
    acting_player: str,
    threat_player: str,
    blocking: bool,
) -> Tuple[Tuple[str, ...], int, Tuple[int, int], str]:
    """Construct a local board with a unique line-completion cell."""

    opponent = opponent_of(str(threat_player))
    for _attempt in range(800):
        line = tuple(rng.choice(LOCAL_LINES))
        answer_cell = int(rng.choice(line))
        support = tuple(int(index) for index in line if int(index) != int(answer_cell))
        cells = [""] * 9
        for index in support:
            cells[int(index)] = str(threat_player)
        available = [index for index in range(9) if index not in set(line)]
        rng.shuffle(available)
        for index in available[:2]:
            cells[int(index)] = str(opponent if rng.random() < 0.75 else threat_player)
        status, _winner_line = status_of(cells)
        if status != STATUS_OPEN:
            continue
        wins = immediate_winning_cells(cells, str(threat_player))
        if tuple(wins) != (int(answer_cell),):
            continue
        if bool(blocking) and immediate_winning_cells(cells, str(acting_player)):
            continue
        empties = [index for index, value in enumerate(cells) if not value]
        if len(empties) < 5:
            continue
        return tuple(cells), int(answer_cell), tuple(support), str(threat_player)
    raise ValueError("failed to construct unique local tactic")


def sample_answer_slot(option_count: int, *, instance_seed: int, params: Mapping[str, Any], namespace: str) -> int:
    """Choose where the correct option appears, respecting cursor controls."""

    explicit = params.get("answer_option_index")
    if explicit is not None:
        value = int(explicit)
        if 0 <= value < int(option_count):
            return int(value)
        raise ValueError("answer_option_index outside option_count")
    sampling_index = params.get("_sample_cursor")
    if sampling_index is not None:
        return abs(int(sampling_index)) % int(option_count)
    rng = spawn_rng(int(instance_seed), f"{str(namespace)}.answer_option_index")
    return int(rng.randrange(int(option_count)))


def sample_local_line_tactic(
    rng,
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    acting_player: str,
    threat_player: str,
    blocking: bool,
    namespace: str,
) -> UltimateSample:
    """Sample a highlighted local board with labeled candidate move cells."""

    option_count, option_probabilities = sample_integer_axis(
        instance_seed=int(instance_seed),
        params=params,
        support_key="option_count_support",
        explicit_key="option_count",
        fallback_support=DEFAULTS.option_count_support,
        namespace=f"{str(namespace)}.option_count",
        balanced_flag_key="balanced_target_answer_sampling",
    )
    tactic_cells, answer_cell, support_cells, threat_player_text = make_tactic_cells(
        rng,
        acting_player=str(acting_player),
        threat_player=str(threat_player),
        blocking=bool(blocking),
    )
    empties = [index for index, value in enumerate(tactic_cells) if not value]
    distractors = [index for index in empties if int(index) != int(answer_cell)]
    rng.shuffle(distractors)
    answer_slot = sample_answer_slot(
        int(option_count),
        instance_seed=int(instance_seed),
        params=params,
        namespace=str(namespace),
    )
    option_cells = distractors[: int(option_count) - 1]
    option_cells.insert(int(answer_slot), int(answer_cell))
    highlighted = int(rng.randrange(9))
    status_options = [STATUS_X_WON, STATUS_O_WON, STATUS_DRAWN, STATUS_OPEN]
    board_values: list[UltimateLocalBoard] = []
    for index in range(9):
        if int(index) == int(highlighted):
            board_values.append(UltimateLocalBoard(cells=tuple(tactic_cells), status=STATUS_OPEN, winning_line=None))
        else:
            board_values.append(local_board_for_status(rng, str(rng.choice(status_options))))
    answer_label = OPTION_LABELS[int(answer_slot)]
    annotation_ids = (cell_entity_id(int(highlighted), int(answer_cell)),)
    return UltimateSample(
        board=tuple(board_values),
        answer=str(answer_label),
        answer_type="string",
        target_answer=None,
        highlighted_board_index=int(highlighted),
        option_cells=tuple(int(cell) for cell in option_cells),
        answer_cell=int(answer_cell),
        support_cells=tuple(int(cell) for cell in support_cells),
        annotation_entity_ids=tuple(annotation_ids),
        metadata={
            "highlighted_small_board": MACRO_LABELS[int(highlighted)],
            "option_count": int(option_count),
            "option_count_probabilities": dict(option_probabilities),
            "answer_option_index": int(answer_slot),
            "answer_label": str(answer_label),
            "answer_cell": int(answer_cell) + 1,
            "support_cells": [int(cell) + 1 for cell in support_cells],
            "acting_player": str(acting_player),
            "threat_player": str(threat_player_text),
            "blocking_tactic": bool(blocking),
            "option_cells": [int(cell) + 1 for cell in option_cells],
        },
    )


def sample_non_threat_board(rng, *, player: str) -> UltimateLocalBoard:
    """Sample one non-counted small board for macro-threat construction."""

    status = str(rng.choice((STATUS_X_WON, STATUS_O_WON, STATUS_DRAWN, "open_no_target_threat")))
    if status == "open_no_target_threat":
        return UltimateLocalBoard(
            cells=open_cells_without_immediate_win(rng, player=str(player)),
            status=STATUS_OPEN,
            winning_line=None,
        )
    return local_board_for_status(rng, status)


def matching_macro_threat_indices(board: Sequence[UltimateLocalBoard], player: str) -> Tuple[int, ...]:
    """Return open small-board indices where the player has a one-move win."""

    return tuple(
        int(index)
        for index, local in enumerate(board)
        if local.status == STATUS_OPEN and bool(immediate_winning_cells(local.cells, str(player)))
    )


def sample_macro_threat_board_count(
    rng,
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    player: str,
    namespace: str,
    branch_count: int,
) -> UltimateSample:
    """Sample an Ultimate board with an exact count of immediate-win local boards."""

    target_answer, target_probabilities = sample_integer_axis(
        instance_seed=int(instance_seed),
        params=inner_cycle_params(params, branch_count=int(branch_count)),
        support_key="macro_threat_board_count_support",
        explicit_key="target_answer",
        fallback_support=DEFAULTS.macro_threat_board_count_support,
        namespace=f"{str(namespace)}.target_answer",
        balanced_flag_key="balanced_target_answer_sampling",
    )
    boards: list[UltimateLocalBoard] = []
    for _index in range(int(target_answer)):
        cells, _answer_cell, _support_cells, _threat_player = make_tactic_cells(
            rng,
            acting_player=str(player),
            threat_player=str(player),
            blocking=False,
        )
        boards.append(UltimateLocalBoard(cells=tuple(cells), status=STATUS_OPEN, winning_line=None))
    while len(boards) < 9:
        boards.append(sample_non_threat_board(rng, player=str(player)))
    rng.shuffle(boards)
    board = tuple(boards)
    matching = matching_macro_threat_indices(board, str(player))
    if len(matching) != int(target_answer):
        raise ValueError("macro-threat target count mismatch")
    annotation_ids = tuple(board_entity_id(index) for index in matching)
    return UltimateSample(
        board=tuple(board),
        answer=int(target_answer),
        answer_type="integer",
        target_answer=int(target_answer),
        highlighted_board_index=None,
        option_cells=(),
        answer_cell=None,
        support_cells=(),
        annotation_entity_ids=tuple(annotation_ids),
        metadata={
            "target_answer": int(target_answer),
            "target_answer_probabilities": dict(target_probabilities),
            "threat_player": str(player),
            "matching_small_boards": [MACRO_LABELS[index] for index in matching],
        },
    )


def status_support(target_status: str) -> tuple[str, tuple[int, ...]]:
    """Return support config ownership for one status-count branch."""

    if str(target_status) in {STATUS_X_WON, STATUS_O_WON}:
        return "won_board_count_support", DEFAULTS.won_board_count_support
    if str(target_status) == STATUS_DRAWN:
        return "drawn_board_count_support", DEFAULTS.drawn_board_count_support
    if str(target_status) == STATUS_NEITHER_WON:
        return "neither_won_board_count_support", DEFAULTS.neither_won_board_count_support
    raise ValueError(f"unsupported target status: {target_status}")
