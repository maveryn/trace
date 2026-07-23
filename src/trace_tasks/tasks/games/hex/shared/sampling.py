"""Identity-free sampling helpers for Hex scenes."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from trace_tasks.core.sampling import uniform_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.games.shared.layout import (
    attach_games_unit_size_jitter,
    resolve_games_layout_jitter,
    resolve_games_unit_size_scale,
    scale_games_px,
)
from trace_tasks.tasks.games.shared.sampling import resolve_games_named_axis
from trace_tasks.tasks.games.shared.style import SUPPORTED_HEX_STYLE_VARIANTS
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.font_assets import sample_font_family
from trace_tasks.tasks.shared.support_sampling import resolve_integer_choice, resolve_integer_support

from .rules import (
    BLUE,
    EMPTY,
    HEX_CANDIDATE_LABELS,
    HEX_MODE_CONNECTION_GAP,
    HEX_MODE_NEIGHBOR_COUNT,
    HEX_MODE_WINNING_MOVE,
    RED,
    SUPPORTED_HEX_PLAYER_COLORS,
    SUPPORTED_HEX_SCENE_VARIANTS,
    Board,
    Coord,
    HexCandidateSpec,
    HexSample,
    all_coords,
    board_from_rows,
    color_value,
    coord_to_cell_id,
    immediate_winning_moves,
    make_connection_path,
    minimum_connection_gap_sets,
    minimum_connection_path,
    neighbors,
    sorted_coords,
    validate_hex_sample,
    winning_path_after_move,
)
from .rendering import HexRenderParams
from .state import DEFAULTS, HEX_NAMESPACE, HexIntegerAxis, HexSceneAxes, HexStringAxis


HEX_NEIGHBOR_STATE_TO_VALUE: Dict[str, int] = {
    "red": RED,
    "blue": BLUE,
    "empty": EMPTY,
}


def _string_support(
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    *,
    key: str,
    fallback: Sequence[str],
) -> Tuple[str, ...]:
    raw = params.get(str(key), group_default(gen_defaults, str(key), tuple(fallback)))
    if raw is None:
        raw = tuple(fallback)
    if isinstance(raw, str):
        values = (raw,)
    else:
        values = tuple(str(value) for value in raw)
    values = tuple(value for value in values if value)
    if not values:
        raise ValueError(f"{key} must contain at least one label")
    return values


def resolve_hex_string_choice(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    support_key: str,
    explicit_key: str,
    fallback_support: Sequence[str],
    namespace: str,
    balanced_flag_key: str,
) -> HexStringAxis:
    """Resolve one string-valued choice with optional finite-support cycling."""

    support = _string_support(
        params,
        gen_defaults,
        key=str(support_key),
        fallback=fallback_support,
    )
    explicit = params.get(str(explicit_key))
    if explicit is not None:
        value = str(explicit)
        if value not in support:
            raise ValueError(f"{explicit_key}={value!r} is not in {support_key}")
        return HexStringAxis(
            value=value,
            support=support,
            probabilities={str(item): (1.0 if str(item) == value else 0.0) for item in support},
        )

    probabilities = {str(item): 1.0 / float(len(support)) for item in support}
    rng = spawn_rng(int(instance_seed), str(namespace))
    return HexStringAxis(
        value=str(uniform_choice(rng, support)),
        support=support,
        probabilities=probabilities,
    )


def resolve_hex_integer_axis(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    support_key: str,
    explicit_key: str,
    fallback_support: Sequence[int],
    namespace: str,
    balanced_flag_key: str,
) -> HexIntegerAxis:
    """Resolve one integer-valued task axis."""

    value, probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        support_key=str(support_key),
        explicit_key=str(explicit_key),
        fallback_support=tuple(int(value) for value in fallback_support),
        namespace=str(namespace),
        balanced_flag_key=str(balanced_flag_key),
        namespace_support_permutation=True,
    )
    support = resolve_integer_support(
        params,
        gen_defaults=gen_defaults,
        key=str(support_key),
        fallback=tuple(int(value) for value in fallback_support),
    )
    return HexIntegerAxis(
        value=int(value),
        support=tuple(int(item) for item in support),
        probabilities=dict(probabilities),
    )


def resolve_hex_scene_axes(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    namespace: str = HEX_NAMESPACE,
) -> HexSceneAxes:
    """Resolve scene-level Hex axes shared by all objectives."""

    scene_variant, scene_variant_probabilities = resolve_games_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        namespace=f"{namespace}.scene_variant",
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        supported_variants=SUPPORTED_HEX_SCENE_VARIANTS,
    )
    style_variant, style_variant_probabilities = resolve_games_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        namespace=f"{namespace}.style_variant",
        explicit_key="style_variant",
        weights_key="style_variant_weights",
        balance_flag_key="balanced_style_variant_sampling",
        supported_variants=SUPPORTED_HEX_STYLE_VARIANTS,
    )
    player_color, player_color_probabilities = resolve_games_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        namespace=f"{namespace}.player_color",
        explicit_key="player_color",
        weights_key="player_color_weights",
        balance_flag_key="balanced_player_color_sampling",
        supported_variants=SUPPORTED_HEX_PLAYER_COLORS,
    )
    board_size_axis = resolve_hex_integer_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        support_key="board_size_support",
        explicit_key="board_size",
        fallback_support=DEFAULTS.board_size_support,
        namespace=f"{namespace}.board_size",
        balanced_flag_key="balanced_board_size_sampling",
    )
    return HexSceneAxes(
        scene_variant=str(scene_variant),
        style_variant=str(style_variant),
        player_color=str(player_color),
        board_size=int(board_size_axis.value),
        scene_variant_probabilities=dict(scene_variant_probabilities),
        style_variant_probabilities=dict(style_variant_probabilities),
        player_color_probabilities=dict(player_color_probabilities),
        board_size_probabilities=dict(board_size_axis.probabilities),
    )


def resolve_hex_candidate_count_axis(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    namespace: str,
) -> HexIntegerAxis:
    return resolve_hex_integer_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        support_key="candidate_count_support",
        explicit_key="candidate_count",
        fallback_support=DEFAULTS.candidate_count_support,
        namespace=str(namespace),
        balanced_flag_key="balanced_candidate_count_sampling",
    )


def resolve_hex_render_params(
    params: Mapping[str, Any],
    *,
    render_defaults: Mapping[str, Any],
    instance_seed: int,
) -> HexRenderParams:
    """Resolve Hex rendering parameters from config/defaults."""

    font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace=f"{HEX_NAMESPACE}.text_font",
        params=params,
    )
    unit_scale, unit_scale_meta = resolve_games_unit_size_scale(
        params,
        render_defaults,
        instance_seed=int(instance_seed),
        namespace=f"{HEX_NAMESPACE}.unit_size",
    )
    layout_jitter = attach_games_unit_size_jitter(
        resolve_games_layout_jitter(
            params,
            render_defaults,
            instance_seed=int(instance_seed),
            namespace=f"{HEX_NAMESPACE}.layout",
        ),
        unit_scale_meta,
    )
    base_canvas_width = int(params.get("canvas_width", group_default(render_defaults, "canvas_width", DEFAULTS.canvas_width)))
    base_canvas_height = int(params.get("canvas_height", group_default(render_defaults, "canvas_height", DEFAULTS.canvas_height)))
    max_board_width_px = scale_games_px(
        params.get("max_board_width_px", group_default(render_defaults, "max_board_width_px", DEFAULTS.max_board_width_px)),
        unit_scale,
        min_px=410,
    )
    max_board_height_px = scale_games_px(
        params.get("max_board_height_px", group_default(render_defaults, "max_board_height_px", DEFAULTS.max_board_height_px)),
        unit_scale,
        min_px=380,
    )
    dynamic_canvas_enabled = bool(
        params.get(
            "dynamic_canvas_size_enabled",
            group_default(render_defaults, "dynamic_canvas_size_enabled", DEFAULTS.dynamic_canvas_size_enabled),
        )
    )
    canvas_width = int(base_canvas_width)
    canvas_height = int(base_canvas_height)
    if dynamic_canvas_enabled and params.get("canvas_width") is None:
        canvas_width = min(
            int(base_canvas_width),
            max(
                int(params.get("canvas_min_width_px", group_default(render_defaults, "canvas_min_width_px", DEFAULTS.canvas_min_width_px))),
                int(
                    round(
                        float(max_board_width_px)
                        + (
                            2.0
                            * float(
                                params.get(
                                    "canvas_side_padding_px",
                                    group_default(render_defaults, "canvas_side_padding_px", DEFAULTS.canvas_side_padding_px),
                                )
                            )
                        )
                    )
                ),
            ),
        )
    if dynamic_canvas_enabled and params.get("canvas_height") is None:
        canvas_height = min(
            int(base_canvas_height),
            max(
                int(params.get("canvas_min_height_px", group_default(render_defaults, "canvas_min_height_px", DEFAULTS.canvas_min_height_px))),
                int(
                    round(
                        float(max_board_height_px)
                        + (
                            2.0
                            * float(
                                params.get(
                                    "canvas_vertical_padding_px",
                                    group_default(render_defaults, "canvas_vertical_padding_px", DEFAULTS.canvas_vertical_padding_px),
                                )
                            )
                        )
                    )
                ),
            ),
        )
    return HexRenderParams(
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        panel_margin_px=int(params.get("panel_margin_px", group_default(render_defaults, "panel_margin_px", DEFAULTS.panel_margin_px))),
        max_board_width_px=int(max_board_width_px),
        max_board_height_px=int(max_board_height_px),
        hex_border_width_px=scale_games_px(params.get("hex_border_width_px", group_default(render_defaults, "hex_border_width_px", DEFAULTS.hex_border_width_px)), unit_scale, min_px=1),
        stone_radius_fraction=float(params.get("stone_radius_fraction", group_default(render_defaults, "stone_radius_fraction", DEFAULTS.stone_radius_fraction))),
        candidate_label_font_size_px=scale_games_px(params.get("candidate_label_font_size_px", group_default(render_defaults, "candidate_label_font_size_px", DEFAULTS.candidate_label_font_size_px)), unit_scale, min_px=15),
        side_band_width_px=scale_games_px(params.get("side_band_width_px", group_default(render_defaults, "side_band_width_px", DEFAULTS.side_band_width_px)), unit_scale, min_px=4),
        layout_jitter_meta=layout_jitter,
        font_family=str(font_family),
    )


def _extra_count_bounds(params: Mapping[str, Any], gen_defaults: Mapping[str, Any], *, own: bool) -> Tuple[int, int]:
    if bool(own):
        low_key, high_key = "min_extra_own_stones", "max_extra_own_stones"
        fallback_low, fallback_high = DEFAULTS.min_extra_own_stones, DEFAULTS.max_extra_own_stones
    else:
        low_key, high_key = "min_extra_opponent_stones", "max_extra_opponent_stones"
        fallback_low, fallback_high = DEFAULTS.min_extra_opponent_stones, DEFAULTS.max_extra_opponent_stones
    low = int(params.get(low_key, group_default(gen_defaults, low_key, fallback_low)))
    high = int(params.get(high_key, group_default(gen_defaults, high_key, fallback_high)))
    if low > high:
        raise ValueError(f"{low_key} must be <= {high_key}")
    return max(0, low), max(0, high)


def _add_scene_clutter(
    *,
    rng,
    rows: list[list[int]],
    player_value: int,
    protected: set[Coord],
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    scene_variant: str,
) -> None:
    size = len(rows)
    empties = [coord for coord in all_coords(size) if coord not in protected and rows[coord[0]][coord[1]] == EMPTY]
    rng.shuffle(empties)
    own_low, own_high = _extra_count_bounds(params, gen_defaults, own=True)
    opp_low, opp_high = _extra_count_bounds(params, gen_defaults, own=False)
    own_count = int(rng.randint(own_low, own_high))
    opp_count = int(rng.randint(opp_low, opp_high))
    if str(scene_variant) == "open_board":
        own_count = max(0, int(round(0.65 * own_count)))
        opp_count = max(0, int(round(0.65 * opp_count)))
    for coord in empties[:own_count]:
        rows[coord[0]][coord[1]] = int(player_value)
    opponent_value = int(BLUE if int(player_value) == int(RED) else RED)
    for coord in empties[own_count:own_count + opp_count]:
        rows[coord[0]][coord[1]] = opponent_value


def sample_winning_move_scene(
    *,
    rng,
    scene_axes: HexSceneAxes,
    target_label: str,
    candidate_count: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
) -> HexSample:
    """Sample a Hex position with exactly one immediate winning move."""

    size = int(scene_axes.board_size)
    player_value = int(color_value(scene_axes.player_color))
    target_index = HEX_CANDIDATE_LABELS.index(str(target_label))
    candidate_count = max(int(candidate_count), int(target_index) + 1)
    for _attempt in range(240):
        path = tuple(make_connection_path(rng=rng, board_size=size, player_value=player_value))
        gap_candidates = list(path)
        rng.shuffle(gap_candidates)
        winning_coord = gap_candidates[0]
        rows = [[EMPTY for _col in range(size)] for _row in range(size)]
        for coord in path:
            if coord != winning_coord:
                rows[coord[0]][coord[1]] = int(player_value)
        _add_scene_clutter(
            rng=rng,
            rows=rows,
            player_value=player_value,
            protected=set(path),
            params=params,
            gen_defaults=gen_defaults,
            scene_variant=str(scene_axes.scene_variant),
        )
        board = board_from_rows(rows)
        winning_moves = immediate_winning_moves(board, player_value=player_value)
        if tuple(winning_moves) != (tuple(winning_coord),):
            continue
        empty_distractors = [
            coord
            for coord in all_coords(size)
            if coord != winning_coord and int(board[coord[0]][coord[1]]) == EMPTY
        ]
        if len(empty_distractors) < candidate_count - 1:
            continue
        rng.shuffle(empty_distractors)
        candidate_coords = list(empty_distractors[:candidate_count - 1])
        candidate_coords.insert(target_index, winning_coord)
        candidate_specs = tuple(
            HexCandidateSpec(
                label=str(HEX_CANDIDATE_LABELS[index]),
                coord=tuple(coord),
                is_answer=bool(tuple(coord) == tuple(winning_coord)),
            )
            for index, coord in enumerate(candidate_coords)
        )
        annotation_coords = winning_path_after_move(
            board,
            player_value=player_value,
            move_coord=winning_coord,
        )
        sample = HexSample(
            board_size=size,
            mode=HEX_MODE_WINNING_MOVE,
            scene_variant=str(scene_axes.scene_variant),
            player_color=str(scene_axes.player_color),
            player_value=player_value,
            board=board,
            answer=str(target_label),
            target_answer=str(target_label),
            candidate_specs=candidate_specs,
            annotation_coords=tuple(annotation_coords),
            winning_move_coord=tuple(winning_coord),
            min_gap_path=tuple(),
            min_gap_empty_coords=tuple(),
            construction_mode="one_empty_cell_completes_player_connection",
        )
        validate_hex_sample(sample)
        return sample
    raise ValueError("failed to sample Hex winning move scene")


def sample_gap_count_scene(
    *,
    rng,
    scene_axes: HexSceneAxes,
    target_answer: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
) -> HexSample:
    """Sample a Hex position whose shortest connection gap equals target_answer."""

    size = int(scene_axes.board_size)
    player_value = int(color_value(scene_axes.player_color))
    target = int(target_answer)
    if target > size:
        raise ValueError("Hex connection gap target cannot exceed board size")
    for _attempt in range(260):
        path = tuple(make_connection_path(rng=rng, board_size=size, player_value=player_value))
        rows = [[EMPTY for _col in range(size)] for _row in range(size)]
        gap_coords = list(path)
        rng.shuffle(gap_coords)
        gap_set = set(gap_coords[:target])
        for coord in path:
            if coord not in gap_set:
                rows[coord[0]][coord[1]] = int(player_value)
        _add_scene_clutter(
            rng=rng,
            rows=rows,
            player_value=player_value,
            protected=set(path),
            params=params,
            gen_defaults=gen_defaults,
            scene_variant=str(scene_axes.scene_variant),
        )
        board = board_from_rows(rows)
        gap_count, min_path = minimum_connection_path(board, player_value=player_value)
        if int(gap_count) != int(target):
            continue
        empty_on_path = tuple(coord for coord in min_path if int(board[coord[0]][coord[1]]) == EMPTY)
        gap_search = minimum_connection_gap_sets(board, player_value=player_value, max_sets=2)
        if not bool(gap_search.exhaustive):
            continue
        if int(gap_search.gap_count) != int(target) or len(gap_search.gap_sets) != 1:
            continue
        if tuple(gap_search.gap_sets[0]) != sorted_coords(empty_on_path):
            continue
        sample = HexSample(
            board_size=size,
            mode=HEX_MODE_CONNECTION_GAP,
            scene_variant=str(scene_axes.scene_variant),
            player_color=str(scene_axes.player_color),
            player_value=player_value,
            board=board,
            answer=int(target),
            target_answer=int(target),
            candidate_specs=tuple(),
            annotation_coords=tuple(gap_search.gap_sets[0]),
            winning_move_coord=None,
            min_gap_path=tuple(min_path),
            min_gap_empty_coords=tuple(gap_search.gap_sets[0]),
            construction_mode="unique_minimum_gap_empty_cell_set",
        )
        validate_hex_sample(sample)
        return sample
    raise ValueError("failed to sample Hex connection gap scene")


def sample_neighbor_count_scene(
    *,
    rng,
    scene_axes: HexSceneAxes,
    target_answer: int,
    target_state: str,
) -> HexSample:
    """Sample a Hex position with an exact state count around one reference cell."""

    size = int(scene_axes.board_size)
    target = int(target_answer)
    if target < 0 or target > 6:
        raise ValueError("Hex neighbor count target must be in 0..6")
    state = str(target_state).lower()
    if state not in HEX_NEIGHBOR_STATE_TO_VALUE:
        raise ValueError(f"unsupported Hex neighbor target state: {target_state}")
    target_value = int(HEX_NEIGHBOR_STATE_TO_VALUE[state])
    player_value = int(color_value(scene_axes.player_color))
    rows = [[EMPTY for _col in range(size)] for _row in range(size)]
    interior_coords = tuple((row, col) for row in range(1, size - 1) for col in range(1, size - 1))
    if not interior_coords:
        raise ValueError("Hex neighbor-count board_size must provide interior cells")
    reference_coord = tuple(rng.choice(interior_coords))
    adjacent = list(neighbors(reference_coord, board_size=size))
    if len(adjacent) != 6:
        raise ValueError("Hex neighbor-count reference cell must have six neighbors")
    rng.shuffle(adjacent)
    matching = set(adjacent[:target])
    non_target_values = tuple(value for value in (EMPTY, RED, BLUE) if int(value) != int(target_value))

    rows[reference_coord[0]][reference_coord[1]] = EMPTY
    for coord in adjacent:
        if coord in matching:
            rows[coord[0]][coord[1]] = int(target_value)
        else:
            rows[coord[0]][coord[1]] = int(rng.choice(non_target_values))

    protected = set(adjacent)
    protected.add(reference_coord)
    clutter_values = (EMPTY, EMPTY, EMPTY, RED, BLUE) if str(scene_axes.scene_variant) == "open_board" else (EMPTY, RED, BLUE, RED, BLUE)
    for coord in all_coords(size):
        if coord in protected:
            continue
        rows[coord[0]][coord[1]] = int(rng.choice(clutter_values))

    board = board_from_rows(rows)
    annotation_coords = sorted_coords(
        coord
        for coord in neighbors(reference_coord, board_size=size)
        if int(board[coord[0]][coord[1]]) == int(target_value)
    )
    sample = HexSample(
        board_size=size,
        mode=HEX_MODE_NEIGHBOR_COUNT,
        scene_variant=str(scene_axes.scene_variant),
        player_color=str(scene_axes.player_color),
        player_value=player_value,
        board=board,
        answer=int(len(annotation_coords)),
        target_answer=int(target),
        candidate_specs=tuple(),
        annotation_coords=tuple(annotation_coords),
        winning_move_coord=None,
        min_gap_path=tuple(),
        min_gap_empty_coords=tuple(),
        construction_mode="reference_cell_adjacent_state_count",
        reference_coord=tuple(reference_coord),
        neighbor_target_state=str(state),
        neighbor_match_coords=tuple(annotation_coords),
    )
    validate_hex_sample(sample)
    return sample


__all__ = [
    "HEX_NEIGHBOR_STATE_TO_VALUE",
    "resolve_hex_candidate_count_axis",
    "resolve_hex_integer_axis",
    "resolve_hex_render_params",
    "resolve_hex_scene_axes",
    "resolve_hex_string_choice",
    "sample_gap_count_scene",
    "sample_neighbor_count_scene",
    "sample_winning_move_scene",
]
