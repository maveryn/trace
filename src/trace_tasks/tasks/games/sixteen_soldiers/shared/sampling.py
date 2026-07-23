"""Scene-neutral sampling helpers for Sixteen Soldiers tasks."""

from __future__ import annotations

from typing import Any, Mapping, Sequence, Tuple

from trace_tasks.tasks.games.shared.sampling import resolve_games_named_axis
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.support_sampling import resolve_integer_choice, resolve_integer_support

from .defaults import DEFAULTS
from .rules import (
    NEIGHBORS,
    all_point_ids,
    board_to_dict,
    capturable_opponent_points,
    freeze_board,
    jump_specs_from,
    legal_destinations,
    opponent,
    point_coord,
)
from .state import (
    BLUE,
    EMPTY,
    RED,
    Board,
    JumpSpec,
    PointId,
    SCENE_NAMESPACE,
    SUPPORTED_SCENE_VARIANTS,
    SUPPORTED_STYLE_VARIANTS,
    SixteenSoldiersSample,
    SixteenSoldiersTargetAxis,
    SixteenSoldiersVisualAxes,
)


def _resolve_named_axis(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    namespace: str,
    explicit_key: str,
    weights_key: str,
    balance_flag_key: str,
    supported: Sequence[str],
) -> tuple[str, dict[str, float]]:
    """Resolve one balanced Sixteen Soldiers visual/setup axis."""

    return resolve_games_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        namespace=str(namespace),
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
        balance_flag_key=str(balance_flag_key),
        supported_variants=tuple(str(value) for value in supported),
    )


def resolve_sixteen_soldiers_visual_axes(
    instance_seed: int,
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    namespace_root: str,
) -> SixteenSoldiersVisualAxes:
    """Resolve non-objective visual/setup axes shared by all scene tasks."""

    scene_variant, scene_variant_probabilities = _resolve_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        namespace=f"{namespace_root}.scene_variant",
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        supported=SUPPORTED_SCENE_VARIANTS,
    )
    style_variant, style_variant_probabilities = _resolve_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        namespace=f"{namespace_root}.style_variant",
        explicit_key="style_variant",
        weights_key="style_variant_weights",
        balance_flag_key="balanced_style_variant_sampling",
        supported=SUPPORTED_STYLE_VARIANTS,
    )
    marked_color_name, marked_piece_color_probabilities = _resolve_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        namespace=f"{namespace_root}.marked_piece_color",
        explicit_key="marked_piece_color",
        weights_key="marked_piece_color_weights",
        balance_flag_key="balanced_marked_piece_color_sampling",
        supported=("red", "blue"),
    )
    piece_count, piece_count_probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        support_key="piece_count_per_side_support",
        explicit_key="piece_count_per_side",
        fallback_support=DEFAULTS.piece_count_per_side_support,
        namespace=f"{namespace_root}.piece_count_per_side",
        balanced_flag_key="balanced_piece_count_per_side_sampling",
        namespace_support_permutation=True,
    )
    return SixteenSoldiersVisualAxes(
        scene_variant=str(scene_variant),
        style_variant=str(style_variant),
        marked_piece_color=RED if str(marked_color_name) == "red" else BLUE,
        piece_count_per_side=int(piece_count),
        scene_variant_probabilities=dict(scene_variant_probabilities),
        style_variant_probabilities=dict(style_variant_probabilities),
        marked_piece_color_probabilities=dict(marked_piece_color_probabilities),
        piece_count_per_side_probabilities=dict(piece_count_probabilities),
    )


def resolve_sixteen_soldiers_target_axis(
    instance_seed: int,
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    support_key: str,
    fallback_support: Sequence[int],
    namespace: str,
) -> SixteenSoldiersTargetAxis:
    """Resolve a task-owned target answer axis."""

    target_answer, target_answer_probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        support_key=str(support_key),
        explicit_key="target_answer",
        fallback_support=tuple(int(value) for value in fallback_support),
        namespace=str(namespace),
        balanced_flag_key="balanced_target_answer_sampling",
        namespace_support_permutation=True,
    )
    target_support = resolve_integer_support(
        params,
        gen_defaults=gen_defaults,
        key=str(support_key),
        fallback=tuple(int(value) for value in fallback_support),
    )
    return SixteenSoldiersTargetAxis(
        target_answer=int(target_answer),
        target_answer_support=tuple(int(value) for value in target_support),
        target_answer_probabilities=dict(target_answer_probabilities),
    )


def marked_candidate_points(scene_variant: str) -> Tuple[PointId, ...]:
    """Return preferred marked-piece points for one scene variant."""

    all_points = all_point_ids()
    if str(scene_variant) == "center_crossroads_midgame":
        candidates = [
            point_id
            for point_id in all_points
            if 2 <= int(point_coord(point_id)[0]) <= 6
            and 1 <= int(point_coord(point_id)[1]) <= 3
            and len(NEIGHBORS[str(point_id)]) >= 5
        ]
        return tuple(candidates) or all_points
    if str(scene_variant) == "triangle_wing_midgame":
        candidates = [
            point_id
            for point_id in all_points
            if int(point_coord(point_id)[0]) in {0, 1, 2, 6, 7, 8}
        ]
        return tuple(candidates) or all_points
    return all_points


def _build_board_from_values(values: Mapping[PointId, int]) -> Board:
    """Build a frozen board from point values."""

    return freeze_board({point_id: int(values.get(point_id, EMPTY)) for point_id in all_point_ids()})


def fill_board_to_piece_counts(
    *,
    rng: Any,
    forced_values: Mapping[PointId, int],
    piece_count_per_side: int,
) -> Board | None:
    """Fill unspecified points while preserving forced local values."""

    values = {point_id: EMPTY for point_id in all_point_ids()}
    for point_id, value in forced_values.items():
        values[str(point_id)] = int(value)
    red_count = sum(1 for value in values.values() if int(value) == RED)
    blue_count = sum(1 for value in values.values() if int(value) == BLUE)
    if red_count > int(piece_count_per_side) or blue_count > int(piece_count_per_side):
        return None

    fillable = [point_id for point_id in all_point_ids() if str(point_id) not in forced_values]
    rng.shuffle(fillable)
    while red_count < int(piece_count_per_side) and fillable:
        point_id = str(fillable.pop())
        values[point_id] = RED
        red_count += 1
    while blue_count < int(piece_count_per_side) and fillable:
        point_id = str(fillable.pop())
        values[point_id] = BLUE
        blue_count += 1
    if red_count != int(piece_count_per_side) or blue_count != int(piece_count_per_side):
        return None
    return _build_board_from_values(values)


def sample_marked_destination_scene(
    *,
    rng: Any,
    axes: SixteenSoldiersVisualAxes,
    target_axis: SixteenSoldiersTargetAxis,
) -> SixteenSoldiersSample:
    """Construct a board where the marked piece has exactly target destinations."""

    target = int(target_axis.target_answer)
    target_color = int(axes.marked_piece_color)
    candidates = [
        point_id
        for point_id in marked_candidate_points(str(axes.scene_variant))
        if len(NEIGHBORS[str(point_id)]) >= target
    ]
    rng.shuffle(candidates)
    for marked_point_id in candidates:
        neighbors = list(NEIGHBORS[str(marked_point_id)])
        rng.shuffle(neighbors)
        empty_destinations = set(neighbors[:target])
        forced: dict[PointId, int] = {str(marked_point_id): target_color}
        for blocker in neighbors[target:]:
            red_remaining = int(axes.piece_count_per_side) - sum(1 for value in forced.values() if int(value) == RED)
            blue_remaining = int(axes.piece_count_per_side) - sum(1 for value in forced.values() if int(value) == BLUE)
            if red_remaining <= 0 and blue_remaining <= 0:
                break
            if red_remaining <= 0:
                forced[str(blocker)] = BLUE
            elif blue_remaining <= 0:
                forced[str(blocker)] = RED
            elif rng.random() < 0.5:
                forced[str(blocker)] = RED
            else:
                forced[str(blocker)] = BLUE
        for destination in empty_destinations:
            forced[str(destination)] = EMPTY

        board = fill_board_to_piece_counts(
            rng=rng,
            forced_values=forced,
            piece_count_per_side=int(axes.piece_count_per_side),
        )
        if board is None:
            continue
        annotation = tuple(sorted(legal_destinations(board, marked_point_id), key=lambda point_id: point_coord(point_id)))
        if len(annotation) != target:
            continue
        return SixteenSoldiersSample(
            scene_variant=str(axes.scene_variant),
            style_variant=str(axes.style_variant),
            board=board,
            answer=int(len(annotation)),
            target_answer=int(target),
            annotation_point_ids=tuple(annotation),
            target_color=int(target_color),
            marked_point_id=str(marked_point_id),
            construction_mode="marked_piece_adjacent_destination_template",
        )
    raise ValueError("failed to construct marked-destination Sixteen Soldiers scene")


def _select_capture_specs(
    *,
    rng: Any,
    specs: Sequence[JumpSpec],
    target: int,
) -> Tuple[JumpSpec, ...] | None:
    """Select target capture lines with unique opponent and landing points."""

    if int(target) == 0:
        return tuple()
    shuffled = list(specs)
    rng.shuffle(shuffled)
    selected: list[JumpSpec] = []
    used_middles: set[PointId] = set()
    used_landings: set[PointId] = set()
    for spec in shuffled:
        if spec.middle_id in used_middles or spec.landing_id in used_landings:
            continue
        selected.append(spec)
        used_middles.add(spec.middle_id)
        used_landings.add(spec.landing_id)
        if len(selected) == int(target):
            return tuple(selected)
    return None


def sample_marked_capture_scene(
    *,
    rng: Any,
    axes: SixteenSoldiersVisualAxes,
    target_axis: SixteenSoldiersTargetAxis,
) -> SixteenSoldiersSample:
    """Construct a board where the marked piece has exactly target captures."""

    target = int(target_axis.target_answer)
    target_color = int(axes.marked_piece_color)
    opponent_color = opponent(target_color)
    candidates = [
        point_id
        for point_id in marked_candidate_points(str(axes.scene_variant))
        if len(jump_specs_from(point_id)) >= target
    ]
    rng.shuffle(candidates)
    for marked_point_id in candidates:
        candidate_specs = list(jump_specs_from(marked_point_id))
        selected_specs = _select_capture_specs(rng=rng, specs=candidate_specs, target=target)
        if selected_specs is None:
            continue

        forced: dict[PointId, int] = {str(marked_point_id): target_color}
        selected_middles = {spec.middle_id for spec in selected_specs}
        selected_landings = {spec.landing_id for spec in selected_specs}
        for spec in selected_specs:
            forced[str(spec.middle_id)] = opponent_color
            forced[str(spec.landing_id)] = EMPTY

        conflict = False
        for spec in candidate_specs:
            if spec in selected_specs:
                continue
            if spec.middle_id in selected_middles:
                if forced.get(str(spec.landing_id), None) == EMPTY:
                    conflict = True
                    break
                forced.setdefault(str(spec.landing_id), target_color)
            elif spec.landing_id in selected_landings:
                if forced.get(str(spec.middle_id), None) == opponent_color:
                    conflict = True
                    break
                forced.setdefault(str(spec.middle_id), target_color)
            else:
                forced.setdefault(str(spec.middle_id), target_color)
        if conflict:
            continue

        board = fill_board_to_piece_counts(
            rng=rng,
            forced_values=forced,
            piece_count_per_side=int(axes.piece_count_per_side),
        )
        if board is None:
            continue
        annotation = tuple(sorted(capturable_opponent_points(board, marked_point_id), key=lambda point_id: point_coord(point_id)))
        if len(annotation) != target:
            continue
        return SixteenSoldiersSample(
            scene_variant=str(axes.scene_variant),
            style_variant=str(axes.style_variant),
            board=board,
            answer=int(len(annotation)),
            target_answer=int(target),
            annotation_point_ids=tuple(annotation),
            target_color=int(target_color),
            marked_point_id=str(marked_point_id),
            construction_mode="marked_piece_immediate_capture_template",
        )
    raise ValueError("failed to construct marked-capture Sixteen Soldiers scene")


def piece_count_support(params: Mapping[str, Any], gen_defaults: Mapping[str, Any]) -> tuple[int, ...]:
    """Return the configured piece-count support for smoke tests and traces."""

    raw = params.get("piece_count_per_side_support", group_default(gen_defaults, "piece_count_per_side_support", DEFAULTS.piece_count_per_side_support))
    return tuple(int(value) for value in raw)


__all__ = [
    "fill_board_to_piece_counts",
    "marked_candidate_points",
    "piece_count_support",
    "resolve_sixteen_soldiers_target_axis",
    "resolve_sixteen_soldiers_visual_axes",
    "sample_marked_capture_scene",
    "sample_marked_destination_scene",
]
