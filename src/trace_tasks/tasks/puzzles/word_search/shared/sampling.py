"""Sampling primitives for word-search puzzle tasks."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.core.sampling import (
    integer_range_choice,
    uniform_choice,
    uniform_choice_with_probabilities,
    weighted_support_choice,
)
from trace_tasks.tasks.puzzles.shared.word_grid import (
    WORD_DIRECTIONS,
    Cell,
    WordPlacement,
    cell_key,
    choose_words,
    fill_random_letters,
    place_word,
    scan_word,
)

from .defaults import get_int_range
from .state import OPTION_LABELS, SCENE_VARIANTS, WordSearchDataset, WordSearchOption


def resolve_scene_variant(
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    rng,
) -> tuple[str, dict[str, float]]:
    """Sample or honor the visual word-search scene variant."""

    explicit = params.get("scene_variant")
    if explicit is not None:
        selected = str(explicit)
        if selected not in SCENE_VARIANTS:
            raise ValueError(f"unsupported word-search scene_variant: {selected}")
        return selected, {
            key: (1.0 if key == selected else 0.0) for key in SCENE_VARIANTS
        }
    weights = defaults.get("scene_variant_weights")
    if isinstance(weights, Mapping):
        return weighted_support_choice(rng, SCENE_VARIANTS, weights=weights)
    return uniform_choice_with_probabilities(rng, SCENE_VARIANTS)


def resolve_grid_size(
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    rng,
) -> tuple[int, int, tuple[int, int]]:
    """Resolve a square word-search grid size."""

    size_min, size_max = get_int_range(
        params,
        defaults,
        min_key="grid_size_min",
        max_key="grid_size_max",
        fallback_min=5,
        fallback_max=8,
    )
    explicit_size = params.get("grid_size")
    if explicit_size is None:
        size, _probabilities = integer_range_choice(rng, size_min, size_max)
    else:
        size = int(explicit_size)
    if not int(size_min) <= int(size) <= int(size_max):
        raise ValueError("grid_size outside configured range")
    return int(size), int(size), (int(size_min), int(size_max))


def sample_location_dataset(
    *,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    rng,
    scene_variant: str,
    scene_variant_probabilities: Mapping[str, float],
    option_count: int,
    answer_label: str,
) -> WordSearchDataset:
    """Build a target-word location option dataset."""

    rows, cols, size_range = resolve_grid_size(params, generation_defaults, rng)
    word_min, word_max = get_int_range(
        params,
        generation_defaults,
        min_key="word_length_min",
        max_key="word_length_max",
        fallback_min=3,
        fallback_max=4,
    )
    for _attempt in range(250):
        word = choose_words(rng, count=1, min_len=word_min, max_len=word_max)[0]
        grid = [["" for _ in range(cols)] for _ in range(rows)]
        placed = place_word(grid, word, rng)
        fill_random_letters(grid, rng)
        hits = scan_word(grid, word)
        if len(hits) != 1:
            continue
        placement = hits[0]
        options = _build_location_options(
            placement=placement,
            rows=rows,
            cols=cols,
            option_count=int(option_count),
            answer_label=str(answer_label),
            rng=rng,
        )
        return WordSearchDataset(
            rows=int(rows),
            cols=int(cols),
            grid_size_range=tuple(size_range),
            grid=tuple(tuple(str(value) for value in row) for row in grid),
            scene_variant=str(scene_variant),
            scene_variant_probabilities=dict(scene_variant_probabilities),
            target_word=str(word),
            target_letter="",
            answer_value=str(answer_label),
            answer_support=tuple(OPTION_LABELS[: int(option_count)]),
            option_specs=tuple(options),
            word_bank=tuple(),
            present_words=tuple(),
            placements=(placement,),
            target_cells=tuple(placement.cells),
        )
    raise RuntimeError("failed to build word-search location dataset")


def sample_present_word_option_dataset(
    *,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    rng,
    scene_variant: str,
    scene_variant_probabilities: Mapping[str, float],
    option_count: int,
    answer_label: str,
) -> WordSearchDataset:
    """Build an option dataset with exactly one listed word present."""

    rows, cols, size_range = resolve_grid_size(params, generation_defaults, rng)
    word_min, word_max = get_int_range(
        params,
        generation_defaults,
        min_key="word_length_min",
        max_key="word_length_max",
        fallback_min=3,
        fallback_max=4,
    )
    labels = tuple(OPTION_LABELS[: int(option_count)])
    if str(answer_label) not in labels:
        raise ValueError("answer_label must be one of the visible option labels")
    answer_index = labels.index(str(answer_label))
    for _attempt in range(400):
        option_words = tuple(
            choose_words(
                rng,
                count=int(option_count),
                min_len=int(word_min),
                max_len=int(word_max),
            )
        )
        present_word = str(option_words[int(answer_index)])
        absent_words = tuple(
            str(word) for index, word in enumerate(option_words) if index != answer_index
        )
        grid = [["" for _ in range(cols)] for _ in range(rows)]
        try:
            placement = place_word(grid, present_word, rng)
        except RuntimeError:
            continue
        fill_random_letters(grid, rng)
        if any(scan_word(grid, str(word)) for word in absent_words):
            continue
        hits = scan_word(grid, present_word)
        if len(hits) != 1:
            continue
        options = tuple(
            WordSearchOption(
                label=str(label),
                display_text=str(word),
                word=str(word),
                is_correct=(str(label) == str(answer_label)),
            )
            for label, word in zip(labels, option_words, strict=True)
        )
        return WordSearchDataset(
            rows=int(rows),
            cols=int(cols),
            grid_size_range=tuple(size_range),
            grid=tuple(tuple(str(value) for value in row) for row in grid),
            scene_variant=str(scene_variant),
            scene_variant_probabilities=dict(scene_variant_probabilities),
            target_word=str(present_word),
            target_letter="",
            answer_value=str(answer_label),
            answer_support=tuple(labels),
            option_specs=tuple(options),
            word_bank=tuple(str(word) for word in option_words),
            present_words=(present_word,),
            placements=(hits[0],),
            target_cells=tuple(hits[0].cells),
        )
    raise RuntimeError("failed to build present-word-option dataset")


def cell_ids_for_target_cells(dataset: WordSearchDataset) -> tuple[str, ...]:
    """Return ordered render ids for the dataset's target cells."""

    return tuple(cell_key(cell) for cell in dataset.target_cells)


def present_word_segments(dataset: WordSearchDataset) -> tuple[tuple[Cell, Cell], ...]:
    """Return start/end cells for each present word placement."""

    segments: list[tuple[Cell, Cell]] = []
    for placement in dataset.placements:
        if not placement.cells:
            continue
        segments.append((placement.cells[0], placement.cells[-1]))
    return tuple(segments)


def _build_location_options(
    *,
    placement: WordPlacement,
    rows: int,
    cols: int,
    option_count: int,
    answer_label: str,
    rng,
) -> tuple[WordSearchOption, ...]:
    """Create one correct placement option plus unique distractors."""

    labels = tuple(OPTION_LABELS[: int(option_count)])
    if str(answer_label) not in labels:
        raise ValueError("answer_label must be one of the visible option labels")
    correct = (int(placement.row) + 1, int(placement.col) + 1, str(placement.direction))
    distractors: list[tuple[int, int, str]] = []
    for direction, _dr, _dc in WORD_DIRECTIONS:
        candidate = (correct[0], correct[1], str(direction))
        if candidate != correct and candidate not in distractors:
            distractors.append(candidate)
    for delta_r, delta_c in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1)):
        row = max(1, min(int(rows), correct[0] + int(delta_r)))
        col = max(1, min(int(cols), correct[1] + int(delta_c)))
        candidate = (int(row), int(col), correct[2])
        if candidate != correct and candidate not in distractors:
            distractors.append(candidate)
    for row in range(1, int(rows) + 1):
        for col in range(1, int(cols) + 1):
            for direction, _dr, _dc in WORD_DIRECTIONS:
                candidate = (int(row), int(col), str(direction))
                if candidate != correct and candidate not in distractors:
                    distractors.append(candidate)
                if len(distractors) >= int(option_count) * 4:
                    break
            if len(distractors) >= int(option_count) * 4:
                break
        if len(distractors) >= int(option_count) * 4:
            break
    rng.shuffle(distractors)
    specs: list[WordSearchOption] = []
    distractor_index = 0
    for label in labels:
        if str(label) == str(answer_label):
            row, col, direction = correct
            specs.append(
                WordSearchOption(
                    label=str(label),
                    row_1based=int(row),
                    col_1based=int(col),
                    direction=str(direction),
                    is_correct=True,
                )
            )
        else:
            row, col, direction = distractors[distractor_index]
            distractor_index += 1
            specs.append(
                WordSearchOption(
                    label=str(label),
                    row_1based=int(row),
                    col_1based=int(col),
                    direction=str(direction),
                    is_correct=False,
                )
            )
    return tuple(specs)


def option_text(spec: WordSearchOption) -> str:
    """Return prompt-facing text for one visible option card."""

    if spec.display_text:
        return f"{spec.label}: {spec.display_text}"
    if spec.row_1based is None or spec.col_1based is None or spec.direction is None:
        raise ValueError("location option requires row, column, and direction")
    return (
        f"{spec.label}: row {int(spec.row_1based)}, "
        f"col {int(spec.col_1based)}, {spec.direction}"
    )


__all__ = [
    "cell_ids_for_target_cells",
    "option_text",
    "present_word_segments",
    "resolve_scene_variant",
    "sample_location_dataset",
    "sample_present_word_option_dataset",
]
