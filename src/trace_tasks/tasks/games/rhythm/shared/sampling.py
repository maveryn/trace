"""Scene-neutral sampling helpers for Rhythm tasks."""

from __future__ import annotations

from typing import Any, List, Mapping, Sequence

from trace_tasks.tasks.games.shared.sampling import resolve_games_named_axis
from trace_tasks.tasks.shared.support_sampling import resolve_integer_choice, resolve_integer_support

from .defaults import DEFAULTS
from .rules import lane_label, note_entity_id, occupied_cells, validate_rhythm_scene_basic
from .state import (
    SCENE_NAMESPACE,
    SUPPORTED_COLOR_KEYS,
    SUPPORTED_SCENE_VARIANTS,
    SUPPORTED_STYLE_VARIANTS,
    RhythmCountTargetAxis,
    RhythmNote,
    RhythmVisualAxes,
    SampledRhythmScene,
)


class RhythmNoteBuilder:
    """Mutable helper that places non-overlapping notes in lane/row cells."""

    def __init__(self, *, lane_count: int, row_count: int, rng: Any) -> None:
        self.lane_count = int(lane_count)
        self.row_count = int(row_count)
        self.rng = rng
        self.notes: list[RhythmNote] = []
        self.occupied: dict[int, set[int]] = {lane: set() for lane in range(int(lane_count))}

    def can_place(self, *, lane: int, bottom_row: int, length: int) -> bool:
        if not (0 <= int(lane) < int(self.lane_count)):
            return False
        if int(bottom_row) < 1 or int(length) < 1:
            return False
        if int(bottom_row) + int(length) - 1 > int(self.row_count):
            return False
        rows = set(range(int(bottom_row), int(bottom_row) + int(length)))
        return not bool(rows & self.occupied[int(lane)])

    def add_note(self, *, lane: int, bottom_row: int, length: int, color_key: str) -> RhythmNote:
        if not self.can_place(lane=int(lane), bottom_row=int(bottom_row), length=int(length)):
            raise ValueError("cannot place rhythm note")
        note = RhythmNote(
            note_id=note_entity_id(len(self.notes)),
            lane_index=int(lane),
            bottom_row=int(bottom_row),
            length=int(length),
            color_key=str(color_key),
            kind="hold" if int(length) > 1 else "tap",
        )
        for row in occupied_cells(note):
            self.occupied[int(lane)].add(int(row))
        self.notes.append(note)
        return note

    def add_random_note(
        self,
        *,
        lanes: Sequence[int],
        bottom_rows: Sequence[int],
        colors: Sequence[str],
        lengths: Sequence[int] = (1, 1, 1, 2, 2, 3),
    ) -> RhythmNote | None:
        """Place the first shuffled note candidate that fits."""

        lane_values = [int(lane) for lane in lanes if 0 <= int(lane) < int(self.lane_count)]
        row_values = [int(row) for row in bottom_rows if 1 <= int(row) <= int(self.row_count)]
        color_values = [str(color) for color in colors if str(color) in SUPPORTED_COLOR_KEYS]
        length_values = [max(1, int(length)) for length in lengths]
        self.rng.shuffle(lane_values)
        self.rng.shuffle(row_values)
        self.rng.shuffle(color_values)
        self.rng.shuffle(length_values)
        for lane in lane_values:
            for row in row_values:
                for length in length_values:
                    if not self.can_place(lane=int(lane), bottom_row=int(row), length=int(length)):
                        continue
                    color = str(color_values[0] if color_values else self.rng.choice(SUPPORTED_COLOR_KEYS))
                    return self.add_note(lane=int(lane), bottom_row=int(row), length=int(length), color_key=color)
        return None


def resolve_rhythm_visual_axes(
    instance_seed: int,
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    namespace_root: str = SCENE_NAMESPACE,
) -> RhythmVisualAxes:
    """Resolve scene, style, lane, row, and beat-window axes."""

    scene_variant, scene_variant_probabilities = resolve_games_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        namespace="scene_variant",
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        supported_variants=[str(value) for value in SUPPORTED_SCENE_VARIANTS],
    )
    style_variant, style_variant_probabilities = resolve_games_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        namespace="style_variant",
        explicit_key="style_variant",
        weights_key="style_variant_weights",
        balance_flag_key="balanced_style_variant_sampling",
        supported_variants=[str(value) for value in SUPPORTED_STYLE_VARIANTS],
    )
    lane_count, lane_count_probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        support_key="lane_count_support",
        explicit_key="lane_count",
        fallback_support=DEFAULTS.lane_count_support,
        namespace=f"{SCENE_NAMESPACE}.lane_count",
        balanced_flag_key="balanced_lane_count_sampling",
        namespace_support_permutation=True,
    )
    row_count, row_count_probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        support_key="row_count_support",
        explicit_key="row_count",
        fallback_support=DEFAULTS.row_count_support,
        namespace=f"{SCENE_NAMESPACE}.row_count",
        balanced_flag_key="balanced_row_count_sampling",
        namespace_support_permutation=True,
    )
    beat_window, beat_window_probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        support_key="beat_window_support",
        explicit_key="beat_window",
        fallback_support=DEFAULTS.beat_window_support,
        namespace=f"{SCENE_NAMESPACE}.beat_window",
        balanced_flag_key="balanced_beat_window_sampling",
        namespace_support_permutation=True,
    )
    if int(beat_window) > int(row_count):
        raise ValueError("beat_window cannot exceed row_count")
    return RhythmVisualAxes(
        scene_variant=str(scene_variant),
        style_variant=str(style_variant),
        lane_count=int(lane_count),
        row_count=int(row_count),
        beat_window=int(beat_window),
        scene_variant_probabilities=dict(scene_variant_probabilities),
        style_variant_probabilities=dict(style_variant_probabilities),
        lane_count_probabilities=dict(lane_count_probabilities),
        row_count_probabilities=dict(row_count_probabilities),
        beat_window_probabilities=dict(beat_window_probabilities),
    )


def resolve_rhythm_count_target_axis(
    instance_seed: int,
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    namespace: str,
    support_key: str,
    explicit_key: str,
    fallback_support: Sequence[int],
    balanced_flag_key: str,
) -> RhythmCountTargetAxis:
    """Resolve an integer target for count-style Rhythm objectives."""

    target_count, target_count_probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        support_key=str(support_key),
        explicit_key=str(explicit_key),
        fallback_support=tuple(int(value) for value in fallback_support),
        namespace=str(namespace),
        balanced_flag_key=str(balanced_flag_key),
        use_instance_seed_cycle=True,
        namespace_support_permutation=True,
    )
    target_count_support = resolve_integer_support(
        params,
        gen_defaults=gen_defaults,
        key=str(support_key),
        fallback=tuple(int(value) for value in fallback_support),
    )
    return RhythmCountTargetAxis(
        target_count=int(target_count),
        target_count_support=tuple(int(value) for value in target_count_support),
        target_count_probabilities=dict(target_count_probabilities),
    )


def resolve_selected_lane(rng: Any, *, lane_count: int, params: Mapping[str, Any], key: str = "selected_lane_index") -> int:
    """Resolve a zero-indexed lane, honoring explicit scene params."""

    explicit = params.get(str(key), params.get("target_lane_index"))
    if explicit is not None:
        lane = int(explicit)
        if not (0 <= lane < int(lane_count)):
            raise ValueError(f"{key} out of range")
        return lane
    return int(rng.randrange(int(lane_count)))


def _fill_late_distractors(builder: RhythmNoteBuilder, *, rng: Any, axes: RhythmVisualAxes, minimum_total: int) -> None:
    """Add extra notes below the answer predicate so the grid remains populated."""

    rows = tuple(range(int(axes.beat_window) + 1, int(axes.row_count) + 1))
    if not rows:
        return
    attempts = 0
    while len(builder.notes) < int(minimum_total) and attempts < 240:
        attempts += 1
        builder.add_random_note(
            lanes=(int(rng.randrange(int(axes.lane_count))),),
            bottom_rows=rows,
            colors=SUPPORTED_COLOR_KEYS,
        )


def _add_note_objects(
    builder: RhythmNoteBuilder,
    *,
    rng: Any,
    lane: int,
    target_count: int,
    colors: Sequence[str],
) -> list[RhythmNote]:
    """Place a requested number of note objects in one lane."""

    placed: list[RhythmNote] = []
    rows = list(range(1, int(builder.row_count) + 1))
    rng.shuffle(rows)
    lengths = [1, 1, 1, 2, 2, 3]
    for _ in range(int(target_count)):
        note = builder.add_random_note(
            lanes=(int(lane),),
            bottom_rows=rows,
            colors=colors,
            lengths=lengths,
        )
        if note is None:
            raise ValueError("could not place requested rhythm note object count")
        placed.append(note)
    return placed


def _fill_other_lane_distractors(
    builder: RhythmNoteBuilder,
    *,
    rng: Any,
    axes: RhythmVisualAxes,
    excluded_lane: int,
    per_lane_max: int = 5,
    colors: Sequence[str] = SUPPORTED_COLOR_KEYS,
) -> None:
    """Populate non-target lanes without changing the target lane answer."""

    color_choices = tuple(str(color) for color in colors)
    rows = tuple(range(1, int(axes.row_count) + 1))
    for lane in range(int(axes.lane_count)):
        if int(lane) == int(excluded_lane):
            continue
        count = int(rng.randrange(1, max(2, int(per_lane_max) + 1)))
        try:
            _add_note_objects(
                builder,
                rng=rng,
                lane=int(lane),
                target_count=count,
                colors=color_choices,
            )
        except ValueError:
            continue
    attempts = 0
    minimum_total = int(rng.randrange(max(12, int(axes.lane_count) * 2), max(13, int(axes.lane_count) * 4)))
    while len(builder.notes) < minimum_total and attempts < 180:
        attempts += 1
        lane = int(rng.randrange(int(axes.lane_count)))
        if lane == int(excluded_lane):
            continue
        builder.add_random_note(lanes=(lane,), bottom_rows=rows, colors=color_choices)


def sample_lane_note_count_scene(*, rng: Any, axes: RhythmVisualAxes, selected_lane: int, target_count: int) -> SampledRhythmScene:
    """Construct a lane-specific note-object count scene."""

    if int(target_count) > int(axes.row_count):
        raise ValueError("target_count cannot exceed row_count")
    builder = RhythmNoteBuilder(lane_count=int(axes.lane_count), row_count=int(axes.row_count), rng=rng)
    target_notes = _add_note_objects(
        builder,
        rng=rng,
        lane=int(selected_lane),
        target_count=int(target_count),
        colors=SUPPORTED_COLOR_KEYS,
    )
    _fill_other_lane_distractors(
        builder,
        rng=rng,
        axes=axes,
        excluded_lane=int(selected_lane),
        per_lane_max=max(2, int(target_count) + 1),
    )
    sample = SampledRhythmScene(
        lane_count=int(axes.lane_count),
        row_count=int(axes.row_count),
        beat_window=int(axes.beat_window),
        scene_variant=str(axes.scene_variant),
        selected_lane_index=int(selected_lane),
        selected_lane_label=lane_label(int(selected_lane)),
        target_color_key=None,
        answer=int(target_count),
        notes=tuple(builder.notes),
        annotation_entity_ids=tuple(str(note.note_id) for note in target_notes),
        construction_mode="target_lane_note_count",
    )
    validate_rhythm_scene_basic(sample)
    return sample


def _score_terms_for_total(total: int, values: Sequence[int], *, max_terms: int = 6) -> tuple[int, ...]:
    """Return one short multiset of score values that sums to total."""

    sorted_values = tuple(sorted({int(value) for value in values}, reverse=True))

    def search(remaining: int, terms_left: int, prefix: tuple[int, ...]) -> tuple[int, ...] | None:
        if remaining == 0 and prefix:
            return prefix
        if remaining < 0 or terms_left <= 0:
            return None
        for value in sorted_values:
            found = search(int(remaining) - int(value), int(terms_left) - 1, (*prefix, int(value)))
            if found is not None:
                return found
        return None

    result = search(int(total), int(max_terms), tuple())
    if result is None:
        raise ValueError(f"cannot compose rhythm score total: {total}")
    return result


def sample_lane_note_score_scene(
    *,
    rng: Any,
    axes: RhythmVisualAxes,
    selected_lane: int,
    target_score: int,
    score_values_by_color: Mapping[str, int],
    max_target_notes: int = 4,
) -> SampledRhythmScene:
    """Construct a lane score scene where color scores sum to the target."""

    color_by_score = {int(value): str(color) for color, value in score_values_by_color.items()}
    score_terms = _score_terms_for_total(
        int(target_score),
        tuple(color_by_score.keys()),
        max_terms=int(max_target_notes),
    )
    builder = RhythmNoteBuilder(lane_count=int(axes.lane_count), row_count=int(axes.row_count), rng=rng)
    target_notes: list[RhythmNote] = []
    rows = list(range(1, int(axes.row_count) + 1))
    rng.shuffle(rows)
    for score_value in score_terms:
        note = builder.add_random_note(
            lanes=(int(selected_lane),),
            bottom_rows=rows,
            colors=(str(color_by_score[int(score_value)]),),
            lengths=(1, 1, 2, 3),
        )
        if note is None:
            raise ValueError("could not place rhythm score note")
        target_notes.append(note)
    _fill_other_lane_distractors(
        builder,
        rng=rng,
        axes=axes,
        excluded_lane=int(selected_lane),
        per_lane_max=max(2, len(target_notes) + 1),
        colors=tuple(str(color) for color in score_values_by_color),
    )
    sample = SampledRhythmScene(
        lane_count=int(axes.lane_count),
        row_count=int(axes.row_count),
        beat_window=int(axes.beat_window),
        scene_variant=str(axes.scene_variant),
        selected_lane_index=int(selected_lane),
        selected_lane_label=lane_label(int(selected_lane)),
        target_color_key=None,
        answer=int(target_score),
        notes=tuple(builder.notes),
        annotation_entity_ids=tuple(str(note.note_id) for note in target_notes),
        construction_mode="target_lane_note_score",
        score_values_by_color={str(color): int(value) for color, value in score_values_by_color.items()},
    )
    validate_rhythm_scene_basic(sample)
    return sample


def sample_most_notes_lane_scene(*, rng: Any, axes: RhythmVisualAxes, target_lane: int, target_count: int) -> SampledRhythmScene:
    """Construct a scene with one lane uniquely maximizing note-object count."""

    builder = RhythmNoteBuilder(lane_count=int(axes.lane_count), row_count=int(axes.row_count), rng=rng)
    target_notes = _add_note_objects(
        builder,
        rng=rng,
        lane=int(target_lane),
        target_count=int(target_count),
        colors=SUPPORTED_COLOR_KEYS,
    )
    for lane in range(int(axes.lane_count)):
        if int(lane) == int(target_lane):
            continue
        lane_count = int(rng.randrange(0, max(1, int(target_count))))
        if lane_count > 0:
            _add_note_objects(builder, rng=rng, lane=int(lane), target_count=lane_count, colors=SUPPORTED_COLOR_KEYS)
    sample = SampledRhythmScene(
        lane_count=int(axes.lane_count),
        row_count=int(axes.row_count),
        beat_window=int(axes.beat_window),
        scene_variant=str(axes.scene_variant),
        selected_lane_index=None,
        selected_lane_label=None,
        target_color_key=None,
        answer=int(lane_label(int(target_lane))),
        notes=tuple(builder.notes),
        annotation_entity_ids=tuple(str(note.note_id) for note in target_notes),
        construction_mode="unique_most_notes_lane",
    )
    validate_rhythm_scene_basic(sample)
    return sample


def sample_earliest_hit_lane_scene(*, rng: Any, axes: RhythmVisualAxes, target_lane: int) -> SampledRhythmScene:
    """Construct a scene with one uniquely earliest note."""

    target_time = int(rng.randrange(1, min(3, int(axes.beat_window)) + 1))
    builder = RhythmNoteBuilder(lane_count=int(axes.lane_count), row_count=int(axes.row_count), rng=rng)
    target_note = builder.add_note(
        lane=int(target_lane),
        bottom_row=int(target_time),
        length=1,
        color_key=str(rng.choice(SUPPORTED_COLOR_KEYS)),
    )
    later_rows = tuple(range(int(target_time) + 1, int(axes.beat_window) + 1))
    for lane in range(int(axes.lane_count)):
        if int(lane) == int(target_lane) or not later_rows or rng.random() < 0.20:
            continue
        builder.add_random_note(lanes=(lane,), bottom_rows=later_rows, colors=SUPPORTED_COLOR_KEYS, lengths=(1, 1, 2))
    _fill_late_distractors(
        builder,
        rng=rng,
        axes=axes,
        minimum_total=int(rng.randrange(max(13, int(axes.lane_count) * 3), max(14, int(axes.lane_count) * 5 + 2))),
    )
    sample = SampledRhythmScene(
        lane_count=int(axes.lane_count),
        row_count=int(axes.row_count),
        beat_window=int(axes.beat_window),
        scene_variant=str(axes.scene_variant),
        selected_lane_index=None,
        selected_lane_label=None,
        target_color_key=None,
        answer=int(lane_label(int(target_lane))),
        notes=tuple(builder.notes),
        annotation_entity_ids=(str(target_note.note_id),),
        construction_mode="unique_earliest_hit_lane",
    )
    validate_rhythm_scene_basic(sample)
    return sample


__all__ = [
    "RhythmNoteBuilder",
    "resolve_rhythm_count_target_axis",
    "resolve_rhythm_visual_axes",
    "resolve_selected_lane",
    "sample_earliest_hit_lane_scene",
    "sample_lane_note_count_scene",
    "sample_lane_note_score_scene",
    "sample_most_notes_lane_scene",
]
