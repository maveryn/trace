"""Shared Bubble-shooter axis and board-state sampling helpers."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import cycle
from typing import Any, Dict, Mapping, Sequence, Tuple

from trace_tasks.core.sampling import shuffled_support, uniform_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.games.shared.layout import (
    attach_games_unit_size_jitter,
    resolve_games_layout_jitter,
    resolve_games_unit_size_scale,
    scale_games_px,
)
from trace_tasks.tasks.shared.config_defaults import (
    group_default,
    load_scene_generation_rendering_prompt_defaults,
)
from trace_tasks.tasks.shared.font_assets import sample_font_family
from trace_tasks.tasks.shared.support_sampling import (
    resolve_integer_choice,
    resolve_integer_support,
)
from trace_tasks.tasks.shared.variant_sampling import (
    apply_balanced_variant_sampling,
    resolve_variant,
)

from .state import (
    BUBBLE_COLOR_KEYS,
    BUBBLE_OPTION_LABELS,
    SUPPORTED_BUBBLE_SHOOTER_SCENE_VARIANTS,
    SUPPORTED_BUBBLE_SHOOTER_STYLE_VARIANTS,
    Board,
    BubbleShooterLandingOption,
    BubbleShooterOption,
    BubbleShooterState,
    Coord,
    bubble_entity_id,
)
from .rules import (
    board_from_mapping,
    bubble_neighbors,
    compute_shot_outcome,
    occupied_coords,
    playable_landing_coords,
    sorted_coords,
    top_connected_occupied,
    validate_bubble_shooter_state,
)
from .defaults import RENDER_FALLBACKS, SCENE_ID
from .rendering import BubbleShooterRenderParams

_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS_UNUSED = (
    load_scene_generation_rendering_prompt_defaults(
        "games",
        SCENE_ID,
    )
)


@dataclass(frozen=True)
class ResolvedBubbleShooterSceneAxes:
    """Resolved scene/style axes shared by Bubble-shooter tasks."""

    scene_variant: str
    style_variant: str
    scene_variant_probabilities: Dict[str, float]
    style_variant_probabilities: Dict[str, float]


@dataclass(frozen=True)
class ResolvedBubbleShooterIntegerAxis:
    """Resolved integer axis and support metadata."""

    value: int
    support: Tuple[int, ...]
    probabilities: Dict[str, float]


@dataclass(frozen=True)
class ResolvedBubbleShooterBoardAxes:
    """Resolved Bubble-shooter board dimensions."""

    rows: ResolvedBubbleShooterIntegerAxis
    cols: ResolvedBubbleShooterIntegerAxis


def _resolve_named_axis(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    namespace: str,
    explicit_key: str,
    weights_key: str,
    balance_flag_key: str,
    supported: Sequence[str],
) -> tuple[str, Dict[str, float]]:
    rng = spawn_rng(int(instance_seed), f"games.bubble_shooter.{namespace}")
    selected, probabilities = resolve_variant(
        rng,
        params=params,
        gen_defaults=_GEN_DEFAULTS,
        supported_variants=tuple(str(value) for value in supported),
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
    )
    selected = apply_balanced_variant_sampling(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=_GEN_DEFAULTS,
        selected_variant=str(selected),
        variant_probabilities=probabilities,
        supported_variants=tuple(str(value) for value in supported),
        balance_flag_key=str(balance_flag_key),
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
        sampling_namespace=f"games.bubble_shooter.{namespace}",
    )
    return str(selected), dict(probabilities)


def resolve_bubble_shooter_scene_axes(
    instance_seed: int,
    *,
    params: Mapping[str, Any],
) -> ResolvedBubbleShooterSceneAxes:
    """Resolve scene/style axes for one Bubble-shooter scene."""

    scene_variant, scene_variant_probabilities = _resolve_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        namespace="scene_variant",
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        supported=SUPPORTED_BUBBLE_SHOOTER_SCENE_VARIANTS,
    )
    style_variant, style_variant_probabilities = _resolve_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        namespace="style_variant",
        explicit_key="style_variant",
        weights_key="style_variant_weights",
        balance_flag_key="balanced_style_variant_sampling",
        supported=SUPPORTED_BUBBLE_SHOOTER_STYLE_VARIANTS,
    )
    return ResolvedBubbleShooterSceneAxes(
        scene_variant=str(scene_variant),
        style_variant=str(style_variant),
        scene_variant_probabilities=dict(scene_variant_probabilities),
        style_variant_probabilities=dict(style_variant_probabilities),
    )


def resolve_bubble_shooter_integer_axis(
    instance_seed: int,
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    support_key: str,
    explicit_key: str,
    fallback_support: Sequence[int],
    namespace: str,
    balanced_flag_key: str,
) -> ResolvedBubbleShooterIntegerAxis:
    """Resolve one task-owned Bubble-shooter integer axis."""

    value, probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        support_key=str(support_key),
        explicit_key=str(explicit_key),
        fallback_support=tuple(int(item) for item in fallback_support),
        namespace=str(namespace),
        balanced_flag_key=str(balanced_flag_key),
        namespace_support_permutation=True,
    )
    support = resolve_integer_support(
        params,
        gen_defaults=gen_defaults,
        key=str(support_key),
        fallback=tuple(int(item) for item in fallback_support),
    )
    return ResolvedBubbleShooterIntegerAxis(
        value=int(value),
        support=tuple(int(item) for item in support),
        probabilities=dict(probabilities),
    )


def resolve_bubble_shooter_board_axes(
    instance_seed: int,
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    row_count_support: Sequence[int],
    col_count_support: Sequence[int],
) -> ResolvedBubbleShooterBoardAxes:
    """Resolve common Bubble-shooter board dimension axes."""

    return ResolvedBubbleShooterBoardAxes(
        rows=resolve_bubble_shooter_integer_axis(
            int(instance_seed),
            params=params,
            gen_defaults=gen_defaults,
            support_key="row_count_support",
            explicit_key="row_count",
            fallback_support=row_count_support,
            namespace="row_count",
            balanced_flag_key="balanced_row_count_sampling",
        ),
        cols=resolve_bubble_shooter_integer_axis(
            int(instance_seed),
            params=params,
            gen_defaults=gen_defaults,
            support_key="col_count_support",
            explicit_key="col_count",
            fallback_support=col_count_support,
            namespace="col_count",
            balanced_flag_key="balanced_col_count_sampling",
        ),
    )


def resolve_bubble_shooter_render_params(
    params: Mapping[str, Any],
    *,
    instance_seed: int,
) -> BubbleShooterRenderParams:
    """Resolve Bubble-shooter rendering parameters from config/defaults."""

    font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace="games.bubble_shooter.text_font",
        params=params,
    )
    unit_scale, unit_scale_meta = resolve_games_unit_size_scale(
        params,
        _RENDER_DEFAULTS,
        instance_seed=int(instance_seed),
        namespace="games.bubble_shooter.unit_size",
    )
    layout_jitter = attach_games_unit_size_jitter(
        resolve_games_layout_jitter(
            params,
            _RENDER_DEFAULTS,
            instance_seed=int(instance_seed),
            namespace="games.bubble_shooter.layout",
        ),
        unit_scale_meta,
    )
    base_canvas_width = int(
        params.get(
            "canvas_width",
            group_default(
                _RENDER_DEFAULTS, "canvas_width", RENDER_FALLBACKS.canvas_width
            ),
        )
    )
    base_canvas_height = int(
        params.get(
            "canvas_height",
            group_default(
                _RENDER_DEFAULTS, "canvas_height", RENDER_FALLBACKS.canvas_height
            ),
        )
    )
    playfield_width_px = scale_games_px(
        params.get(
            "playfield_width_px",
            group_default(
                _RENDER_DEFAULTS,
                "playfield_width_px",
                RENDER_FALLBACKS.playfield_width_px,
            ),
        ),
        unit_scale,
        min_px=430,
    )
    playfield_height_px = scale_games_px(
        params.get(
            "playfield_height_px",
            group_default(
                _RENDER_DEFAULTS,
                "playfield_height_px",
                RENDER_FALLBACKS.playfield_height_px,
            ),
        ),
        unit_scale,
        min_px=360,
    )
    dynamic_canvas_enabled = bool(
        params.get(
            "dynamic_canvas_size_enabled",
            group_default(
                _RENDER_DEFAULTS,
                "dynamic_canvas_size_enabled",
                RENDER_FALLBACKS.dynamic_canvas_size_enabled,
            ),
        )
    )
    canvas_width = base_canvas_width
    canvas_height = base_canvas_height
    if dynamic_canvas_enabled and params.get("canvas_width") is None:
        canvas_width = min(
            int(base_canvas_width),
            max(
                int(
                    params.get(
                        "canvas_min_width_px",
                        group_default(
                            _RENDER_DEFAULTS,
                            "canvas_min_width_px",
                            RENDER_FALLBACKS.canvas_min_width_px,
                        ),
                    )
                ),
                int(
                    round(
                        float(playfield_width_px)
                        + (
                            2.0
                            * float(
                                params.get(
                                    "canvas_side_padding_px",
                                    group_default(
                                        _RENDER_DEFAULTS,
                                        "canvas_side_padding_px",
                                        RENDER_FALLBACKS.canvas_side_padding_px,
                                    ),
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
                int(
                    params.get(
                        "canvas_min_height_px",
                        group_default(
                            _RENDER_DEFAULTS,
                            "canvas_min_height_px",
                            RENDER_FALLBACKS.canvas_min_height_px,
                        ),
                    )
                ),
                int(
                    round(
                        float(playfield_height_px)
                        + (
                            2.0
                            * float(
                                params.get(
                                    "canvas_vertical_padding_px",
                                    group_default(
                                        _RENDER_DEFAULTS,
                                        "canvas_vertical_padding_px",
                                        RENDER_FALLBACKS.canvas_vertical_padding_px,
                                    ),
                                )
                            )
                        )
                    )
                ),
            ),
        )
    return BubbleShooterRenderParams(
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        panel_margin_px=int(
            params.get(
                "panel_margin_px",
                group_default(
                    _RENDER_DEFAULTS,
                    "panel_margin_px",
                    RENDER_FALLBACKS.panel_margin_px,
                ),
            )
        ),
        playfield_width_px=int(playfield_width_px),
        playfield_height_px=int(playfield_height_px),
        playfield_border_width_px=scale_games_px(
            params.get(
                "playfield_border_width_px",
                group_default(
                    _RENDER_DEFAULTS,
                    "playfield_border_width_px",
                    RENDER_FALLBACKS.playfield_border_width_px,
                ),
            ),
            unit_scale,
            min_px=2,
        ),
        board_top_px=scale_games_px(
            params.get(
                "board_top_px",
                group_default(
                    _RENDER_DEFAULTS, "board_top_px", RENDER_FALLBACKS.board_top_px
                ),
            ),
            unit_scale,
            min_px=19,
        ),
        board_height_px=scale_games_px(
            params.get(
                "board_height_px",
                group_default(
                    _RENDER_DEFAULTS,
                    "board_height_px",
                    RENDER_FALLBACKS.board_height_px,
                ),
            ),
            unit_scale,
            min_px=250,
        ),
        bubble_gap_px=scale_games_px(
            params.get(
                "bubble_gap_px",
                group_default(
                    _RENDER_DEFAULTS, "bubble_gap_px", RENDER_FALLBACKS.bubble_gap_px
                ),
            ),
            unit_scale,
            min_px=1,
        ),
        path_width_px=scale_games_px(
            params.get(
                "path_width_px",
                group_default(
                    _RENDER_DEFAULTS, "path_width_px", RENDER_FALLBACKS.path_width_px
                ),
            ),
            unit_scale,
            min_px=2,
        ),
        shooter_radius_px=scale_games_px(
            params.get(
                "shooter_radius_px",
                group_default(
                    _RENDER_DEFAULTS,
                    "shooter_radius_px",
                    RENDER_FALLBACKS.shooter_radius_px,
                ),
            ),
            unit_scale,
            min_px=11,
        ),
        option_radius_px=scale_games_px(
            params.get(
                "option_radius_px",
                group_default(
                    _RENDER_DEFAULTS,
                    "option_radius_px",
                    RENDER_FALLBACKS.option_radius_px,
                ),
            ),
            unit_scale,
            min_px=8,
        ),
        option_label_font_size_px=scale_games_px(
            params.get(
                "option_label_font_size_px",
                group_default(
                    _RENDER_DEFAULTS,
                    "option_label_font_size_px",
                    RENDER_FALLBACKS.option_label_font_size_px,
                ),
            ),
            unit_scale,
            min_px=11,
        ),
        font_family=str(font_family),
        layout_jitter_meta=layout_jitter,
    )


def _colors_except(excluded: Sequence[str]) -> Tuple[str, ...]:
    excluded_set = {str(value) for value in excluded}
    values = tuple(color for color in BUBBLE_COLOR_KEYS if color not in excluded_set)
    return values or tuple(BUBBLE_COLOR_KEYS)


def _make_connected_shape(
    *,
    rng,
    rows: int,
    cols: int,
    start: Coord,
    count: int,
    blocked: set[Coord],
    min_row: int = 1,
    max_row: int | None = None,
) -> Tuple[Coord, ...]:
    limit_row = int(rows) - 1 if max_row is None else int(max_row)
    shape: list[Coord] = [tuple(start)]
    seen = {tuple(start)}
    frontier = [tuple(start)]
    while len(shape) < int(count) and frontier:
        rng.shuffle(frontier)
        source = frontier[0]
        candidates = [
            neighbor
            for neighbor in bubble_neighbors(source, rows=rows, cols=cols)
            if neighbor not in seen
            and neighbor not in blocked
            and int(min_row) <= int(neighbor[0]) <= int(limit_row)
        ]
        if not candidates:
            frontier.pop(0)
            continue
        rng.shuffle(candidates)
        coord = tuple(candidates[0])
        seen.add(coord)
        shape.append(coord)
        frontier.append(coord)
    if len(shape) != int(count):
        raise ValueError("failed to construct connected bubble shape")
    return sorted_coords(shape)


def _landing_candidates_for_shape(
    *,
    board: Board,
    shape: Sequence[Coord],
    min_row: int = 1,
    max_row: int | None = None,
) -> Tuple[Coord, ...]:
    """Return exposed landing slots that touch the target shape."""

    rows = len(board)
    cols = len(board[0]) if rows else 0
    limit_row = int(rows) - 2 if max_row is None else int(max_row)
    shape_set = {tuple(coord) for coord in shape}
    return sorted_coords(
        coord
        for coord in playable_landing_coords(board)
        if int(min_row) <= int(coord[0]) <= int(limit_row)
        and any(
            neighbor in shape_set
            for neighbor in bubble_neighbors(coord, rows=rows, cols=cols)
        )
    )


def _support_path_to_top(
    *, rng, rows: int, cols: int, start: Coord, blocked: set[Coord]
) -> Tuple[Coord, ...]:
    current = tuple(start)
    path: list[Coord] = []
    guard = 0
    while int(current[0]) > 0 and guard < int(rows) * 3:
        guard += 1
        candidates = [
            neighbor
            for neighbor in bubble_neighbors(current, rows=rows, cols=cols)
            if int(neighbor[0]) == int(current[0]) - 1 and neighbor not in blocked
        ]
        if not candidates:
            raise ValueError("failed to construct support path to top")
        rng.shuffle(candidates)
        current = tuple(candidates[0])
        path.append(current)
        blocked.add(current)
    if not path or int(path[-1][0]) != 0:
        raise ValueError("support path did not reach top row")
    return tuple(path)


def _add_top_clutter(
    *,
    rng,
    rows: int,
    cols: int,
    values: Dict[Coord, str],
    protected: set[Coord],
    excluded_colors: Sequence[str],
    scene_variant: str,
) -> None:
    colors = _colors_except(excluded_colors)
    fill_rows = 3 if str(scene_variant) == "dense_pack" else 2
    for row in range(min(int(rows), int(fill_rows))):
        base_probability = (
            0.92 if row == 0 else 0.72 if str(scene_variant) == "dense_pack" else 0.52
        )
        for col in range(int(cols)):
            coord = (int(row), int(col))
            if coord in protected or coord in values:
                continue
            if row == 0 or rng.random() < float(base_probability):
                values[coord] = str(rng.choice(colors))


def _assert_top_supported(board: Board) -> None:
    if set(occupied_coords(board)) != set(top_connected_occupied(board)):
        raise ValueError(
            "bubble shooter board contains unsupported bubbles before the shot"
        )


def _make_no_pop_board(
    *,
    rng,
    scene_variant: str,
    rows: int,
    cols: int,
    color_key: str,
) -> Tuple[Board, Coord]:
    values: Dict[Coord, str] = {}
    _add_top_clutter(
        rng=rng,
        rows=rows,
        cols=cols,
        values=values,
        protected=set(),
        excluded_colors=(str(color_key),),
        scene_variant=str(scene_variant),
    )
    board = board_from_mapping(rows=rows, cols=cols, values=values)
    _assert_top_supported(board)
    candidates = [
        coord
        for coord in playable_landing_coords(board)
        if 1 <= int(coord[0]) <= int(rows) - 2
    ]
    if not candidates:
        raise ValueError("no landing candidate for no-pop construction")
    rng.shuffle(candidates)
    for landing in candidates:
        outcome = compute_shot_outcome(
            board, landing_coord=landing, color_key=str(color_key)
        )
        if not outcome.popped_coords and not outcome.dropped_coords:
            return board, tuple(landing)
    raise ValueError("constructed no-pop board unexpectedly changed state")


def _make_pop_board(
    *,
    rng,
    scene_variant: str,
    rows: int,
    cols: int,
    target: int,
    color_key: str,
) -> Tuple[Board, Coord]:
    """Construct a supported board where the placed shot pops exactly target bubbles."""

    if int(target) == 0:
        return _make_no_pop_board(
            rng=rng,
            scene_variant=str(scene_variant),
            rows=int(rows),
            cols=int(cols),
            color_key=str(color_key),
        )

    start = (
        int(rng.randint(2, max(2, rows - 3))),
        int(rng.randint(2, max(2, cols - 3))),
    )
    component = _make_connected_shape(
        rng=rng,
        rows=rows,
        cols=cols,
        start=start,
        count=int(target),
        blocked=set(),
        min_row=1,
        max_row=max(2, rows - 2),
    )
    protected = set(component)
    values: Dict[Coord, str] = {coord: str(color_key) for coord in component}
    support_blocked = set(protected)
    support = _support_path_to_top(
        rng=rng, rows=rows, cols=cols, start=component[0], blocked=support_blocked
    )
    support_colors = _colors_except((str(color_key),))
    support_color_cycle = cycle(shuffled_support(rng, support_colors))
    for coord in support:
        values[coord] = str(next(support_color_cycle))
    protected.update(support)
    _add_top_clutter(
        rng=rng,
        rows=rows,
        cols=cols,
        values=values,
        protected=protected,
        excluded_colors=(str(color_key),),
        scene_variant=str(scene_variant),
    )
    board = board_from_mapping(rows=rows, cols=cols, values=values)
    _assert_top_supported(board)
    landing_candidates = list(
        _landing_candidates_for_shape(board=board, shape=component)
    )
    if not landing_candidates:
        raise ValueError("no exposed landing candidate for pop component")
    rng.shuffle(landing_candidates)
    for landing in landing_candidates:
        outcome = compute_shot_outcome(
            board, landing_coord=landing, color_key=str(color_key)
        )
        if len(outcome.popped_coords) == int(target):
            return board, tuple(landing)
    raise ValueError("constructed pop board did not hit target pop count")


def _make_drop_board(
    *,
    rng,
    scene_variant: str,
    rows: int,
    cols: int,
    target: int,
    color_key: str,
) -> Tuple[Board, Coord]:
    """Construct a supported board where a small pop disconnects exactly target bubbles."""

    if int(target) > max(1, (int(rows) - 4) * 2):
        raise ValueError("drop target is too large for this board")
    start = (
        int(rng.randint(2, max(2, rows - 4))),
        int(rng.randint(2, max(2, cols - 3))),
    )
    pop_component = _make_connected_shape(
        rng=rng,
        rows=rows,
        cols=cols,
        start=start,
        count=2,
        blocked=set(),
        min_row=1,
        max_row=max(2, rows - 3),
    )
    below_candidates = [
        neighbor
        for coord in pop_component
        for neighbor in bubble_neighbors(coord, rows=rows, cols=cols)
        if int(neighbor[0]) > max(int(item[0]) for item in pop_component)
        and neighbor not in set(pop_component)
    ]
    if not below_candidates:
        raise ValueError("drop construction needs tail candidates")
    rng.shuffle(below_candidates)
    blocked = set(pop_component)
    if int(target) > 0:
        tail_seed = next(iter(below_candidates), None)
        if tail_seed is None:
            raise ValueError("drop construction has no tail seed")
        tail = _make_connected_shape(
            rng=rng,
            rows=rows,
            cols=cols,
            start=tail_seed,
            count=int(target),
            blocked=blocked,
            min_row=int(tail_seed[0]),
            max_row=int(rows) - 1,
        )
    else:
        tail = tuple()
    protected = set(pop_component) | set(tail)
    support_blocked = set(protected)
    support = _support_path_to_top(
        rng=rng, rows=rows, cols=cols, start=pop_component[0], blocked=support_blocked
    )
    protected.update(support)
    values: Dict[Coord, str] = {coord: str(color_key) for coord in pop_component}
    non_shot_colors = _colors_except((str(color_key),))
    tail_color = str(uniform_choice(rng, non_shot_colors))
    for coord in tail:
        values[coord] = tail_color
    support_color_cycle = cycle(shuffled_support(rng, non_shot_colors))
    for coord in support:
        values[coord] = str(next(support_color_cycle))
    _add_top_clutter(
        rng=rng,
        rows=rows,
        cols=cols,
        values=values,
        protected=protected,
        excluded_colors=(str(color_key),),
        scene_variant=str(scene_variant),
    )
    board = board_from_mapping(rows=rows, cols=cols, values=values)
    _assert_top_supported(board)
    landing_candidates = list(
        _landing_candidates_for_shape(board=board, shape=pop_component)
    )
    if not landing_candidates:
        raise ValueError("no exposed landing candidate for drop component")
    rng.shuffle(landing_candidates)
    for landing in landing_candidates:
        outcome = compute_shot_outcome(
            board, landing_coord=landing, color_key=str(color_key)
        )
        if len(outcome.popped_coords) == 2 and len(outcome.dropped_coords) == int(
            target
        ):
            return board, tuple(landing)
    raise ValueError("constructed drop board did not hit target drop count")


def sample_pop_state(
    *,
    rng,
    scene_axes: ResolvedBubbleShooterSceneAxes,
    board_axes: ResolvedBubbleShooterBoardAxes,
    target_pop_count: int,
) -> BubbleShooterState:
    """Construct a Bubble-shooter state with a target pop count."""

    color = str(rng.choice(BUBBLE_COLOR_KEYS))
    board, landing = _make_pop_board(
        rng=rng,
        scene_variant=str(scene_axes.scene_variant),
        rows=int(board_axes.rows.value),
        cols=int(board_axes.cols.value),
        target=int(target_pop_count),
        color_key=color,
    )
    outcome = compute_shot_outcome(board, landing_coord=landing, color_key=color)
    state = BubbleShooterState(
        row_count=int(board_axes.rows.value),
        col_count=int(board_axes.cols.value),
        scene_variant=str(scene_axes.scene_variant),
        style_variant=str(scene_axes.style_variant),
        board=board,
        landing_coord=landing,
        shooter_color_key=color,
        option_specs=tuple(),
        outcome=outcome,
        construction_mode="single_shot_pop_component",
    )
    validate_bubble_shooter_state(state)
    if len(state.outcome.popped_coords) != int(target_pop_count):
        raise ValueError("pop state target mismatch")
    return state


def sample_drop_state(
    *,
    rng,
    scene_axes: ResolvedBubbleShooterSceneAxes,
    board_axes: ResolvedBubbleShooterBoardAxes,
    target_drop_count: int,
) -> BubbleShooterState:
    """Construct a Bubble-shooter state with a target drop count."""

    color = str(rng.choice(BUBBLE_COLOR_KEYS))
    board, landing = _make_drop_board(
        rng=rng,
        scene_variant=str(scene_axes.scene_variant),
        rows=int(board_axes.rows.value),
        cols=int(board_axes.cols.value),
        target=int(target_drop_count),
        color_key=color,
    )
    outcome = compute_shot_outcome(board, landing_coord=landing, color_key=color)
    state = BubbleShooterState(
        row_count=int(board_axes.rows.value),
        col_count=int(board_axes.cols.value),
        scene_variant=str(scene_axes.scene_variant),
        style_variant=str(scene_axes.style_variant),
        board=board,
        landing_coord=landing,
        shooter_color_key=color,
        option_specs=tuple(),
        outcome=outcome,
        construction_mode="pop_bridge_then_drop_floating_tail",
    )
    validate_bubble_shooter_state(state)
    if len(state.outcome.dropped_coords) != int(target_drop_count):
        raise ValueError("drop state target mismatch")
    return state


def sample_pop_color_state(
    *,
    rng,
    scene_axes: ResolvedBubbleShooterSceneAxes,
    board_axes: ResolvedBubbleShooterBoardAxes,
    target_option_label: str,
    option_count: int,
) -> BubbleShooterState:
    """Construct a Bubble-shooter state with one popping displayed color option."""

    target_label = str(target_option_label)
    option_count = int(option_count)
    target_index = BUBBLE_OPTION_LABELS.index(target_label)
    option_count = max(option_count, target_index + 1)
    colors = list(BUBBLE_COLOR_KEYS)
    rng.shuffle(colors)
    target_color = str(colors[0])
    negative_colors = [color for color in colors[1:] if color != target_color]
    if len(negative_colors) < option_count - 1:
        raise ValueError("not enough distinct option colors")
    target_pop_count = int(rng.randint(2, 5))
    board, landing = _make_pop_board(
        rng=rng,
        scene_variant=str(scene_axes.scene_variant),
        rows=int(board_axes.rows.value),
        cols=int(board_axes.cols.value),
        target=target_pop_count,
        color_key=target_color,
    )
    option_colors = list(negative_colors[: option_count - 1])
    option_colors.insert(target_index, target_color)
    option_specs = tuple(
        BubbleShooterOption(
            label=str(BUBBLE_OPTION_LABELS[index]),
            color_key=str(color),
            is_answer=bool(index == target_index),
        )
        for index, color in enumerate(option_colors)
    )
    outcome = compute_shot_outcome(board, landing_coord=landing, color_key=target_color)
    state = BubbleShooterState(
        row_count=int(board_axes.rows.value),
        col_count=int(board_axes.cols.value),
        scene_variant=str(scene_axes.scene_variant),
        style_variant=str(scene_axes.style_variant),
        board=board,
        landing_coord=landing,
        shooter_color_key=None,
        option_specs=option_specs,
        outcome=outcome,
        construction_mode="one_displayed_color_reaches_pop_threshold",
    )
    validate_bubble_shooter_state(state)
    return state


def sample_pop_target_state(
    *,
    rng,
    scene_axes: ResolvedBubbleShooterSceneAxes,
    board_axes: ResolvedBubbleShooterBoardAxes,
    target_option_label: str,
    option_labels: Sequence[str],
    target_pop_count: int | None = None,
) -> BubbleShooterState:
    """Construct a board with one displayed landing target that pops bubbles."""

    labels = tuple(str(label) for label in option_labels)
    if not labels:
        raise ValueError("bubble shooter target-option labels must not be empty")
    target_label = str(target_option_label)
    if target_label not in labels:
        raise ValueError("target landing-option label must be displayed")

    color = str(rng.choice(BUBBLE_COLOR_KEYS))
    pop_count = (
        int(target_pop_count)
        if target_pop_count is not None
        else int(rng.choice((2, 3, 4, 5)))
    )
    board, answer_landing = _make_pop_board(
        rng=rng,
        scene_variant=str(scene_axes.scene_variant),
        rows=int(board_axes.rows.value),
        cols=int(board_axes.cols.value),
        target=int(pop_count),
        color_key=color,
    )
    negative_candidates = [
        coord
        for coord in playable_landing_coords(board)
        if tuple(coord) != tuple(answer_landing)
        and not compute_shot_outcome(
            board,
            landing_coord=tuple(coord),
            color_key=color,
        ).popped_coords
    ]
    if len(negative_candidates) < len(labels) - 1:
        raise ValueError("not enough non-popping landing targets")
    rng.shuffle(negative_candidates)
    negative_iter = iter(negative_candidates)
    landing_options: list[BubbleShooterLandingOption] = []
    for label in labels:
        is_answer = str(label) == target_label
        coord = tuple(answer_landing) if is_answer else tuple(next(negative_iter))
        landing_options.append(
            BubbleShooterLandingOption(
                label=str(label),
                landing_coord=coord,
                is_answer=bool(is_answer),
            )
        )

    outcome = compute_shot_outcome(board, landing_coord=answer_landing, color_key=color)
    if len(outcome.popped_coords) != int(pop_count):
        raise ValueError("pop-target state target mismatch")
    state = BubbleShooterState(
        row_count=int(board_axes.rows.value),
        col_count=int(board_axes.cols.value),
        scene_variant=str(scene_axes.scene_variant),
        style_variant=str(scene_axes.style_variant),
        board=board,
        landing_coord=tuple(answer_landing),
        shooter_color_key=color,
        option_specs=tuple(),
        outcome=outcome,
        construction_mode="one_displayed_landing_target_reaches_pop_threshold",
        landing_option_specs=tuple(landing_options),
    )
    validate_bubble_shooter_state(state)
    return state


def bubble_entity_ids_for_coords(coords: Sequence[Coord]) -> Tuple[str, ...]:
    """Return board-bubble entity ids for coordinates."""

    return tuple(bubble_entity_id(coord) for coord in coords)


__all__ = [
    "ResolvedBubbleShooterBoardAxes",
    "ResolvedBubbleShooterIntegerAxis",
    "ResolvedBubbleShooterSceneAxes",
    "bubble_entity_ids_for_coords",
    "resolve_bubble_shooter_board_axes",
    "resolve_bubble_shooter_integer_axis",
    "resolve_bubble_shooter_render_params",
    "resolve_bubble_shooter_scene_axes",
    "sample_drop_state",
    "sample_pop_color_state",
    "sample_pop_target_state",
    "sample_pop_state",
]
