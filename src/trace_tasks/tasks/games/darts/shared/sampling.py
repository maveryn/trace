"""Identity-free semantic sampling primitives for simplified darts scene tasks."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from trace_tasks.tasks.games.shared.layout import apply_games_layout_jitter_to_bbox, resolve_games_layout_jitter
from trace_tasks.tasks.games.shared.sampling import resolve_games_integer_axis, resolve_games_named_axis
from trace_tasks.tasks.games.shared.style import SUPPORTED_DARTS_STYLE_VARIANTS
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.font_assets import sample_font_family
from trace_tasks.tasks.shared.support_sampling import resolve_integer_support

from .defaults import (
    DARTS_NAMESPACE,
    DEFAULTS,
    STANDARD_DART_SECTORS,
    SUPPORTED_DARTS_SCENE_VARIANTS,
)
from .rendering import (
    DARTBOARD_SAMPLE_RADIUS_FRACTIONS,
    DartboardRenderParams,
    polar_to_xy,
)
from .rules import BULLSEYE_SLOT, SCORE_SLOTS, SECTOR_SLOTS, slots_for_score
from .state import (
    DartInstance,
    DartsIntegerAxis,
    DartsSampledScene,
    DartsSceneAxes,
    DartsScoreSlot,
)


_SECTOR_TO_INDEX = {int(value): int(index) for index, value in enumerate(STANDARD_DART_SECTORS)}


def sector_angle_width_deg() -> float:
    """Return the angular width of one simplified dartboard sector."""

    return 360.0 / float(max(1, len(STANDARD_DART_SECTORS)))


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
) -> Tuple[str, Dict[str, float]]:
    """Resolve one named semantic or visual axis without task identity."""

    return resolve_games_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        namespace=f"{DARTS_NAMESPACE}.{str(namespace)}",
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
        balance_flag_key=str(balance_flag_key),
        supported_variants=[str(item) for item in supported],
    )


def resolve_darts_scene_axes(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
) -> DartsSceneAxes:
    """Resolve scene and style axes common to darts tasks."""

    scene_variant, scene_variant_probabilities = _resolve_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        namespace="scene_variant",
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        supported=SUPPORTED_DARTS_SCENE_VARIANTS,
    )
    style_variant, style_variant_probabilities = _resolve_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        namespace="style_variant",
        explicit_key="style_variant",
        weights_key="style_variant_weights",
        balance_flag_key="balanced_style_variant_sampling",
        supported=SUPPORTED_DARTS_STYLE_VARIANTS,
    )
    return DartsSceneAxes(
        scene_variant=str(scene_variant),
        style_variant=str(style_variant),
        scene_variant_probabilities=dict(scene_variant_probabilities),
        style_variant_probabilities=dict(style_variant_probabilities),
    )


def resolve_darts_integer_axis(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    support_key: str,
    explicit_key: str,
    fallback_support: Sequence[int],
    namespace: str,
    balanced_flag_key: str,
) -> DartsIntegerAxis:
    """Resolve one task-owned integer axis."""

    value, support, probabilities = resolve_games_integer_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        support_key=str(support_key),
        explicit_key=str(explicit_key),
        fallback_support=tuple(int(value) for value in fallback_support),
        namespace=f"{DARTS_NAMESPACE}.{str(namespace)}",
        balanced_flag_key=str(balanced_flag_key),
    )
    return DartsIntegerAxis(
        value=int(value),
        support=tuple(int(item) for item in support),
        probabilities=dict(probabilities),
    )


def feasible_count_support(*, dart_count: int, raw_support: Sequence[int]) -> Tuple[int, ...]:
    """Return answer-count support feasible for the active dart count."""

    return tuple(int(value) for value in raw_support if 0 <= int(value) <= int(dart_count))


def resolve_darts_count_target_axis(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    dart_count: int,
    namespace: str,
) -> DartsIntegerAxis:
    """Resolve the answer-count target for a count-style darts task."""

    raw_support = resolve_integer_support(
        params,
        gen_defaults=gen_defaults,
        key="count_target_answer_support",
        fallback=DEFAULTS.count_target_answer_support,
    )
    support = feasible_count_support(dart_count=int(dart_count), raw_support=raw_support)
    target_params = dict(params)
    target_params["count_target_answer_support"] = [int(value) for value in support]
    return resolve_darts_integer_axis(
        instance_seed=int(instance_seed),
        params=target_params,
        gen_defaults=gen_defaults,
        support_key="count_target_answer_support",
        explicit_key="target_answer",
        fallback_support=support,
        namespace=str(namespace),
        balanced_flag_key="balanced_target_answer_sampling",
    )


def resolve_darts_score_axis(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
) -> DartsIntegerAxis:
    """Resolve the marked-dart target score for score-value tasks."""

    return resolve_darts_integer_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        support_key="score_value_support",
        explicit_key="target_score",
        fallback_support=DEFAULTS.score_value_support,
        namespace="score_value.target_score",
        balanced_flag_key="balanced_score_value_sampling",
    )


def _sample_slot(rng, slots: Sequence[DartsScoreSlot]) -> DartsScoreSlot:
    """Return one random score slot from a non-empty pool."""

    if not slots:
        raise ValueError("cannot sample from an empty dart score-slot pool")
    return slots[int(rng.randrange(len(slots)))]


def _slot_position(rng, *, slot: DartsScoreSlot, params: DartboardRenderParams) -> Tuple[float, float]:
    """Sample a visible point inside one simplified scoring slot."""

    board_radius = float(params.board_radius_px)
    if str(slot.area_kind) == "bullseye":
        angle = float(rng.uniform(-180.0, 180.0))
        radius_min, radius_max = DARTBOARD_SAMPLE_RADIUS_FRACTIONS["bullseye"]
        radius = float(rng.uniform(float(radius_min), float(radius_max)) * board_radius)
        return polar_to_xy(
            cx=float(params.board_center_x_px),
            cy=float(params.board_center_y_px),
            radius=float(radius),
            angle_deg=float(angle),
        )

    sector_index = _SECTOR_TO_INDEX[int(slot.sector_value or 0)]
    sector_width = sector_angle_width_deg()
    center_angle = float(sector_index) * float(sector_width)
    angle_margin = float(sector_width) * 0.36
    angle = float(rng.uniform(center_angle - angle_margin, center_angle + angle_margin))
    radius_min, radius_max = DARTBOARD_SAMPLE_RADIUS_FRACTIONS["sector"]
    radius = float(rng.uniform(float(radius_min), float(radius_max)) * board_radius)
    return polar_to_xy(
        cx=float(params.board_center_x_px),
        cy=float(params.board_center_y_px),
        radius=float(radius),
        angle_deg=float(angle),
    )


def _sample_position_without_overlap(
    rng,
    *,
    slot: DartsScoreSlot,
    params: DartboardRenderParams,
    existing_points: Sequence[Tuple[float, float]],
) -> Tuple[float, float]:
    """Sample one marker center with a minimum distance from existing darts."""

    if str(slot.area_kind) == "bullseye":
        min_distance = float(max(24, int(params.marker_radius_px) * 2))
    else:
        min_distance = float(max(24, int(params.marker_radius_px) * 2 + 8))
    for _ in range(96):
        point = _slot_position(rng, slot=slot, params=params)
        if all(((point[0] - x) ** 2 + (point[1] - y) ** 2) ** 0.5 >= min_distance for x, y in existing_points):
            return point
    return _slot_position(rng, slot=slot, params=params)


def sample_darts_for_count(
    rng,
    *,
    dart_count: int,
    target_answer: int,
    render_params: DartboardRenderParams,
    qualifying_slots: Sequence[DartsScoreSlot],
    nonqualifying_slots: Sequence[DartsScoreSlot],
) -> DartsSampledScene:
    """Sample darts with an exact count of qualifying score slots."""

    selected_slots = [
        _sample_slot(rng, qualifying_slots) for _ in range(int(target_answer))
    ] + [
        _sample_slot(rng, nonqualifying_slots) for _ in range(max(0, int(dart_count) - int(target_answer)))
    ]
    annotation_flags = [True for _ in range(int(target_answer))] + [
        False for _ in range(max(0, int(dart_count) - int(target_answer)))
    ]
    combined = list(zip(selected_slots, annotation_flags))
    rng.shuffle(combined)
    return _sample_darts_from_slots(
        rng,
        render_params=render_params,
        selected_slots=[slot for slot, _ in combined],
        annotation_flags=[flag for _, flag in combined],
        marked_flags=[False for _ in combined],
    )


def sample_darts_for_score_value(
    rng,
    *,
    target_score: int,
    render_params: DartboardRenderParams,
) -> DartsSampledScene:
    """Sample one unmarked dart with the requested score."""

    matching_slots = slots_for_score(int(target_score))
    if not matching_slots:
        raise ValueError(f"unsupported darts score: {target_score}")
    selected_slots = [_sample_slot(rng, matching_slots)]
    return _sample_darts_from_slots(
        rng,
        render_params=render_params,
        selected_slots=selected_slots,
        annotation_flags=[True],
        marked_flags=[False],
        target_score=int(target_score),
    )


def sample_darts_for_highest_scoring_label(
    rng,
    *,
    target_label_index: int,
    option_labels: Sequence[str],
    render_params: DartboardRenderParams,
) -> DartsSampledScene:
    """Sample four labeled sector darts with a unique highest-scoring option."""

    labels = tuple(str(label) for label in option_labels)
    if len(labels) != 4:
        raise ValueError("highest-scoring dart label task requires exactly four labels")
    target_index = int(target_label_index)
    if target_index < 0 or target_index >= len(labels):
        raise ValueError(f"unsupported highest-scoring dart label index: {target_index}")

    winner_score = int(rng.choice([score for score in STANDARD_DART_SECTORS if int(score) >= 8]))
    lower_scores = [int(score) for score in STANDARD_DART_SECTORS if int(score) < int(winner_score)]
    if len(lower_scores) < len(labels) - 1:
        raise ValueError("not enough lower-scoring darts to construct unique maximum")
    rng.shuffle(lower_scores)
    label_scores: list[int | None] = [None] * len(labels)
    label_scores[target_index] = int(winner_score)
    lower_iter = iter(lower_scores)
    for label_index in range(len(labels)):
        if label_scores[label_index] is None:
            label_scores[label_index] = int(next(lower_iter))

    selected_slots = []
    for score in label_scores:
        matching_slots = slots_for_score(int(score))
        if not matching_slots:
            raise ValueError(f"unsupported darts score: {score}")
        selected_slots.append(_sample_slot(rng, matching_slots))

    return _sample_darts_from_slots(
        rng,
        render_params=render_params,
        selected_slots=selected_slots,
        annotation_flags=[int(index) == int(target_index) for index in range(len(labels))],
        marked_flags=[False for _ in labels],
        labels=labels,
        target_score=int(winner_score),
    )


def _sample_darts_from_slots(
    rng,
    *,
    render_params: DartboardRenderParams,
    selected_slots: Sequence[DartsScoreSlot],
    annotation_flags: Sequence[bool],
    marked_flags: Sequence[bool],
    labels: Sequence[str | None] | None = None,
    target_score: int | None = None,
) -> DartsSampledScene:
    """Project sampled score slots to visible dart markers."""

    darts: List[DartInstance] = []
    annotation_ids: List[str] = []
    points: List[Tuple[float, float]] = []
    dart_labels = tuple(labels or [None for _ in selected_slots])
    for index, slot in enumerate(selected_slots):
        dart_id = f"dart_{index + 1:02d}"
        x_px, y_px = _sample_position_without_overlap(rng, slot=slot, params=render_params, existing_points=points)
        points.append((float(x_px), float(y_px)))
        is_annotation = bool(annotation_flags[index])
        if is_annotation:
            annotation_ids.append(str(dart_id))
        darts.append(
            DartInstance(
                dart_id=str(dart_id),
                label=None if dart_labels[index] is None else str(dart_labels[index]),
                area_kind=str(slot.area_kind),
                sector_value=None if slot.sector_value is None else int(slot.sector_value),
                score=int(slot.score),
                x_px=float(x_px),
                y_px=float(y_px),
                is_marked=bool(marked_flags[index]),
            )
        )
    return DartsSampledScene(
        darts=tuple(darts),
        annotation_dart_ids=tuple(annotation_ids),
        total_score=int(sum(int(slot.score) for slot in selected_slots)),
        target_score=None if target_score is None else int(target_score),
    )


def resolve_darts_render_params(
    params: Mapping[str, Any],
    *,
    render_defaults: Mapping[str, Any],
    instance_seed: int,
) -> DartboardRenderParams:
    """Resolve darts rendering parameters from config/defaults."""

    canvas_width = int(params.get("canvas_width", group_default(render_defaults, "canvas_width", DEFAULTS.canvas_width)))
    canvas_height = int(params.get("canvas_height", group_default(render_defaults, "canvas_height", DEFAULTS.canvas_height)))
    board_center_x = int(
        params.get("board_center_x_px", group_default(render_defaults, "board_center_x_px", DEFAULTS.board_center_x_px))
    )
    board_center_y = int(
        params.get("board_center_y_px", group_default(render_defaults, "board_center_y_px", DEFAULTS.board_center_y_px))
    )
    board_radius = int(params.get("board_radius_px", group_default(render_defaults, "board_radius_px", DEFAULTS.board_radius_px)))
    font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace=f"{DARTS_NAMESPACE}.font_family",
        params=params,
    )
    requested_jitter = resolve_games_layout_jitter(
        params,
        render_defaults,
        instance_seed=int(instance_seed),
        namespace=f"{DARTS_NAMESPACE}.layout",
    )
    _board_bbox, dx, dy, layout_jitter = apply_games_layout_jitter_to_bbox(
        bbox_px=(
            float(board_center_x - board_radius),
            float(board_center_y - board_radius),
            float(board_center_x + board_radius),
            float(board_center_y + board_radius),
        ),
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        jitter=requested_jitter,
    )
    return DartboardRenderParams(
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        board_center_x_px=int(round(float(board_center_x + dx))),
        board_center_y_px=int(round(float(board_center_y + dy))),
        board_radius_px=int(board_radius),
        marker_radius_px=int(
            params.get("marker_radius_px", group_default(render_defaults, "marker_radius_px", DEFAULTS.marker_radius_px))
        ),
        number_font_size_px=int(
            params.get("number_font_size_px", group_default(render_defaults, "number_font_size_px", DEFAULTS.number_font_size_px))
        ),
        font_family=str(font_family),
        layout_jitter_meta=dict(layout_jitter),
    )


__all__ = [
    "BULLSEYE_SLOT",
    "SCORE_SLOTS",
    "SECTOR_SLOTS",
    "feasible_count_support",
    "resolve_darts_count_target_axis",
    "resolve_darts_integer_axis",
    "resolve_darts_render_params",
    "resolve_darts_scene_axes",
    "resolve_darts_score_axis",
    "sample_darts_for_count",
    "sample_darts_for_highest_scoring_label",
    "sample_darts_for_score_value",
]
