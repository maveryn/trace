"""Sampling primitives for paper-fold result puzzles."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from trace_tasks.core.sampling import integer_range_choice, uniform_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.shared.config_defaults import group_default

from .constraints import resolve_correct_option_index
from .defaults import PaperFoldGenerationDefaults, resolve_generation_defaults
from .state import FOLD_MARK_TYPES, PaperFoldDataset


def folded_result_grid_dimensions(*, axis: str, grid_size: int) -> tuple[int, int]:
    """Return folded result dimensions for the active fold axis."""

    half = int(grid_size) // 2
    if str(axis) == "vertical":
        return int(half), int(grid_size)
    if str(axis) == "horizontal":
        return int(grid_size), int(half)
    raise ValueError(f"unsupported fold axis: {axis}")


def _canonical_mark_specs(
    mark_specs: Iterable[Mapping[str, Any]],
    *,
    include_source_side: bool,
) -> list[dict[str, Any]]:
    """Canonicalize mark dictionaries into deterministic row-major order."""

    items: list[dict[str, Any]] = []
    for raw_mark in mark_specs:
        cell = raw_mark["cell"]
        record: dict[str, Any] = {
            "mark_id": str(raw_mark["mark_id"]),
            "object_type": str(raw_mark["object_type"]),
            "cell": [int(cell[0]), int(cell[1])],
        }
        if include_source_side and "source_side" in raw_mark:
            record["source_side"] = str(raw_mark["source_side"])
        items.append(record)
    items.sort(
        key=lambda item: (
            int(item["cell"][1]),
            int(item["cell"][0]),
            str(item["object_type"]),
            str(item["mark_id"]),
        )
    )
    return items


def _mark_signature(mark_specs: Iterable[Mapping[str, Any]]) -> tuple[tuple[str, int, int], ...]:
    """Return one hashable signature for folded-result marks."""

    return tuple(
        sorted(
            (
                str(mark["object_type"]),
                int(mark["cell"][0]),
                int(mark["cell"][1]),
            )
            for mark in mark_specs
        )
    )


def _sample_source_sides(rng, *, mark_count: int) -> list[str]:
    """Sample kept-versus-folded provenance while ensuring folding matters."""

    if int(mark_count) <= 1:
        return ["folded"]
    folded_count = int(rng.randint(1, int(mark_count) - 1))
    sides = ["folded"] * int(folded_count) + ["kept"] * int(mark_count - folded_count)
    rng.shuffle(sides)
    return [str(item) for item in sides]


def _map_local_to_original(
    *,
    axis: str,
    direction: str,
    source_side: str,
    cell_x: int,
    cell_y: int,
    grid_size: int,
) -> tuple[int, int]:
    """Map one folded-result local cell back to the original full-sheet cell."""

    half = int(grid_size) // 2
    if str(axis) == "vertical":
        if str(direction) == "left_to_right":
            if str(source_side) == "kept":
                return int(half + cell_x), int(cell_y)
            return int((half - 1) - cell_x), int(cell_y)
        if str(direction) == "right_to_left":
            if str(source_side) == "kept":
                return int(cell_x), int(cell_y)
            return int((grid_size - 1) - cell_x), int(cell_y)
        raise ValueError(f"unsupported vertical fold direction: {direction}")
    if str(axis) == "horizontal":
        if str(direction) == "top_to_bottom":
            if str(source_side) == "kept":
                return int(cell_x), int(half + cell_y)
            return int(cell_x), int((half - 1) - cell_y)
        if str(direction) == "bottom_to_top":
            if str(source_side) == "kept":
                return int(cell_x), int(cell_y)
            return int(cell_x), int((grid_size - 1) - cell_y)
        raise ValueError(f"unsupported horizontal fold direction: {direction}")
    raise ValueError(f"unsupported fold axis: {axis}")


def _reflect_result_marks(
    mark_specs: Iterable[Mapping[str, Any]],
    *,
    axis: str,
    cols: int,
    rows: int,
) -> list[dict[str, Any]]:
    """Reflect folded-result marks across the kept packet."""

    reflected: list[dict[str, Any]] = []
    for index, mark in enumerate(mark_specs, start=1):
        cell_x = int(mark["cell"][0])
        cell_y = int(mark["cell"][1])
        if str(axis) == "vertical":
            target = [int((cols - 1) - cell_x), int(cell_y)]
        elif str(axis) == "horizontal":
            target = [int(cell_x), int((rows - 1) - cell_y)]
        else:
            raise ValueError(f"unsupported fold axis: {axis}")
        reflected.append(
            {
                "mark_id": f"candidate_mark_{int(index)}",
                "object_type": str(mark["object_type"]),
                "cell": target,
            }
        )
    return _canonical_mark_specs(reflected, include_source_side=False)


def _shift_candidate(
    mark_specs: Sequence[Mapping[str, Any]],
    *,
    mark_index: int,
    dx: int,
    dy: int,
    cols: int,
    rows: int,
) -> list[dict[str, Any]] | None:
    """Shift one mark inside the folded packet if the target cell is free."""

    occupied = {
        (int(mark["cell"][0]), int(mark["cell"][1]))
        for i, mark in enumerate(mark_specs)
        if int(i) != int(mark_index)
    }
    candidate: list[dict[str, Any]] = []
    for index, mark in enumerate(mark_specs, start=1):
        cell_x = int(mark["cell"][0])
        cell_y = int(mark["cell"][1])
        if int(index - 1) == int(mark_index):
            new_x = int(cell_x + dx)
            new_y = int(cell_y + dy)
            if not (0 <= int(new_x) < int(cols) and 0 <= int(new_y) < int(rows)):
                return None
            if (int(new_x), int(new_y)) in occupied:
                return None
            cell = [int(new_x), int(new_y)]
        else:
            cell = [int(cell_x), int(cell_y)]
        candidate.append(
            {
                "mark_id": f"candidate_mark_{int(index)}",
                "object_type": str(mark["object_type"]),
                "cell": cell,
            }
        )
    return _canonical_mark_specs(candidate, include_source_side=False)


def _replace_type_candidate(
    mark_specs: Sequence[Mapping[str, Any]],
    *,
    mark_index: int,
    object_type: str,
) -> list[dict[str, Any]]:
    """Replace one mark type while keeping positions fixed."""

    candidate: list[dict[str, Any]] = []
    for index, mark in enumerate(mark_specs, start=1):
        candidate.append(
            {
                "mark_id": f"candidate_mark_{int(index)}",
                "object_type": str(
                    object_type if int(index - 1) == int(mark_index) else mark["object_type"]
                ),
                "cell": [int(mark["cell"][0]), int(mark["cell"][1])],
            }
        )
    return _canonical_mark_specs(candidate, include_source_side=False)


def _remove_mark_candidate(
    mark_specs: Sequence[Mapping[str, Any]],
    *,
    mark_index: int,
) -> list[dict[str, Any]] | None:
    """Remove one mark if the result still contains at least two marks."""

    if len(mark_specs) <= 2:
        return None
    candidate = [
        {
            "mark_id": f"candidate_mark_{int(index)}",
            "object_type": str(mark["object_type"]),
            "cell": [int(mark["cell"][0]), int(mark["cell"][1])],
        }
        for index, mark in enumerate(mark_specs, start=1)
        if int(index - 1) != int(mark_index)
    ]
    return _canonical_mark_specs(candidate, include_source_side=False)


def _add_mark_candidate(
    mark_specs: Sequence[Mapping[str, Any]],
    *,
    object_type: str,
    cell: tuple[int, int],
) -> list[dict[str, Any]] | None:
    """Add one mark at a free location if the type and cell are both unused."""

    occupied = {(int(mark["cell"][0]), int(mark["cell"][1])) for mark in mark_specs}
    used_types = {str(mark["object_type"]) for mark in mark_specs}
    if tuple(int(value) for value in cell) in occupied or str(object_type) in used_types:
        return None
    candidate = [
        {
            "mark_id": f"candidate_mark_{int(index)}",
            "object_type": str(mark["object_type"]),
            "cell": [int(mark["cell"][0]), int(mark["cell"][1])],
        }
        for index, mark in enumerate(mark_specs, start=1)
    ]
    candidate.append(
        {
            "mark_id": f"candidate_mark_{int(len(candidate) + 1)}",
            "object_type": str(object_type),
            "cell": [int(cell[0]), int(cell[1])],
        }
    )
    return _canonical_mark_specs(candidate, include_source_side=False)


def _build_distractor_candidates(
    correct_result_marks: Sequence[Mapping[str, Any]],
    *,
    axis: str,
    cols: int,
    rows: int,
) -> list[list[dict[str, Any]]]:
    """Generate structured distractor candidates for a fold-result puzzle."""

    candidates: list[list[dict[str, Any]]] = [
        _reflect_result_marks(
            correct_result_marks,
            axis=str(axis),
            cols=int(cols),
            rows=int(rows),
        )
    ]
    for mark_index in range(len(correct_result_marks)):
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            shifted = _shift_candidate(
                correct_result_marks,
                mark_index=int(mark_index),
                dx=int(dx),
                dy=int(dy),
                cols=int(cols),
                rows=int(rows),
            )
            if shifted is not None:
                candidates.append(shifted)
    used_types = {str(mark["object_type"]) for mark in correct_result_marks}
    unused_types = [
        mark_type for mark_type in FOLD_MARK_TYPES if str(mark_type) not in used_types
    ]
    for mark_index in range(len(correct_result_marks)):
        for object_type in unused_types[:2]:
            candidates.append(
                _replace_type_candidate(
                    correct_result_marks,
                    mark_index=int(mark_index),
                    object_type=str(object_type),
                )
            )
    for mark_index in range(len(correct_result_marks)):
        removed = _remove_mark_candidate(
            correct_result_marks,
            mark_index=int(mark_index),
        )
        if removed is not None:
            candidates.append(removed)
    if unused_types:
        occupied = {
            (int(mark["cell"][0]), int(mark["cell"][1]))
            for mark in correct_result_marks
        }
        for cell_y in range(int(rows)):
            for cell_x in range(int(cols)):
                if (int(cell_x), int(cell_y)) in occupied:
                    continue
                added = _add_mark_candidate(
                    correct_result_marks,
                    object_type=str(unused_types[0]),
                    cell=(int(cell_x), int(cell_y)),
                )
                if added is not None:
                    candidates.append(added)
                if len(candidates) >= 12:
                    return candidates
    return candidates


def _random_result_marks(
    rng,
    *,
    cols: int,
    rows: int,
    mark_count: int,
) -> list[dict[str, Any]]:
    """Sample a fallback random folded-result mark set."""

    cells = rng.sample(
        [(int(x), int(y)) for y in range(int(rows)) for x in range(int(cols))],
        int(mark_count),
    )
    object_types = rng.sample(list(FOLD_MARK_TYPES), int(mark_count))
    raw = [
        {
            "mark_id": f"candidate_mark_{int(index)}",
            "object_type": str(object_type),
            "cell": [int(cell[0]), int(cell[1])],
        }
        for index, (cell, object_type) in enumerate(
            zip(cells, object_types),
            start=1,
        )
    ]
    return _canonical_mark_specs(raw, include_source_side=False)


def _int_param(
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    key: str,
    fallback: int,
) -> int:
    """Resolve one integer param from explicit params, config, or fallback."""

    if params.get(str(key)) is not None:
        return int(params[str(key)])
    return int(group_default(defaults, str(key), int(fallback)))


def sample_paper_fold_dataset(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    generation_defaults: Mapping[str, Any],
    fold_axis: str,
    namespace: str,
) -> PaperFoldDataset:
    """Build one single-fold paper puzzle with a unique correct folded result."""

    defaults: PaperFoldGenerationDefaults = resolve_generation_defaults(generation_defaults)
    rng = spawn_rng(int(instance_seed), f"{namespace}.dataset")
    option_count_min = _int_param(
        params,
        generation_defaults,
        "option_count_min",
        defaults.option_count_min,
    )
    option_count_max = _int_param(
        params,
        generation_defaults,
        "option_count_max",
        defaults.option_count_max,
    )
    option_count, option_count_probabilities = integer_range_choice(
        rng,
        option_count_min,
        option_count_max,
    )
    grid_size = _int_param(params, generation_defaults, "grid_size", defaults.grid_size)
    if int(grid_size) % 2 != 0:
        raise ValueError("paper-fold result puzzles require an even grid_size")
    mark_count_min = _int_param(
        params,
        generation_defaults,
        "mark_count_min",
        defaults.mark_count_min,
    )
    mark_count_max = _int_param(
        params,
        generation_defaults,
        "mark_count_max",
        defaults.mark_count_max,
    )
    mark_count = int(rng.randint(mark_count_min, max(mark_count_min, mark_count_max)))
    if int(mark_count) > len(FOLD_MARK_TYPES):
        raise ValueError("mark_count cannot exceed available mark types")

    if str(fold_axis) == "vertical":
        direction = str(uniform_choice(rng, ("left_to_right", "right_to_left")))
    elif str(fold_axis) == "horizontal":
        direction = str(uniform_choice(rng, ("top_to_bottom", "bottom_to_top")))
    else:
        raise ValueError(f"unsupported fold axis: {fold_axis}")
    result_cols, result_rows = folded_result_grid_dimensions(
        axis=str(fold_axis),
        grid_size=int(grid_size),
    )
    all_local_cells = [
        (int(x), int(y))
        for y in range(int(result_rows))
        for x in range(int(result_cols))
    ]

    original_mark_specs: list[dict[str, Any]] = []
    result_mark_specs: list[dict[str, Any]] = []
    for _attempt in range(80):
        local_cells = rng.sample(all_local_cells, int(mark_count))
        object_types = rng.sample(list(FOLD_MARK_TYPES), int(mark_count))
        source_sides = _sample_source_sides(rng, mark_count=int(mark_count))
        raw_result_marks: list[dict[str, Any]] = []
        raw_original_marks: list[dict[str, Any]] = []
        for mark_index, (cell, object_type, source_side) in enumerate(
            zip(local_cells, object_types, source_sides),
            start=1,
        ):
            cell_x, cell_y = int(cell[0]), int(cell[1])
            mark_id = f"mark_{int(mark_index)}"
            raw_result_marks.append(
                {
                    "mark_id": str(mark_id),
                    "object_type": str(object_type),
                    "cell": [int(cell_x), int(cell_y)],
                    "source_side": str(source_side),
                }
            )
            original_x, original_y = _map_local_to_original(
                axis=str(fold_axis),
                direction=str(direction),
                source_side=str(source_side),
                cell_x=int(cell_x),
                cell_y=int(cell_y),
                grid_size=int(grid_size),
            )
            raw_original_marks.append(
                {
                    "mark_id": str(mark_id),
                    "object_type": str(object_type),
                    "cell": [int(original_x), int(original_y)],
                    "source_side": str(source_side),
                }
            )
        candidate_result = _canonical_mark_specs(
            raw_result_marks,
            include_source_side=True,
        )
        mirrored_signature = _mark_signature(
            _reflect_result_marks(
                candidate_result,
                axis=str(fold_axis),
                cols=int(result_cols),
                rows=int(result_rows),
            )
        )
        if _mark_signature(candidate_result) == mirrored_signature:
            continue
        original_mark_specs = _canonical_mark_specs(
            raw_original_marks,
            include_source_side=True,
        )
        result_mark_specs = candidate_result
        break
    else:
        raise ValueError("failed to sample a non-symmetric fold-result layout")

    correct_option_marks = _canonical_mark_specs(
        result_mark_specs,
        include_source_side=False,
    )
    distractor_candidates = _build_distractor_candidates(
        correct_option_marks,
        axis=str(fold_axis),
        cols=int(result_cols),
        rows=int(result_rows),
    )
    chosen_distractors: list[list[dict[str, Any]]] = []
    seen_signatures = {_mark_signature(correct_option_marks)}
    for candidate in distractor_candidates:
        signature = _mark_signature(candidate)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        chosen_distractors.append(candidate)
        if len(chosen_distractors) >= int(option_count) - 1:
            break
    while len(chosen_distractors) < int(option_count) - 1:
        fallback = _random_result_marks(
            rng,
            cols=int(result_cols),
            rows=int(result_rows),
            mark_count=int(mark_count),
        )
        signature = _mark_signature(fallback)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        chosen_distractors.append(fallback)

    correct_index, correct_index_probabilities, correct_index_sampling_mode = (
        resolve_correct_option_index(
            params,
            option_count=int(option_count),
            rng=rng,
        )
    )
    option_mark_sets: list[list[dict[str, Any]]] = []
    distractor_iter = iter(chosen_distractors)
    for option_index in range(int(option_count)):
        if int(option_index) == int(correct_index):
            option_mark_sets.append(correct_option_marks)
        else:
            option_mark_sets.append(next(distractor_iter))

    option_specs: list[dict[str, Any]] = []
    for option_index, mark_specs in enumerate(option_mark_sets):
        option_label = chr(ord("A") + int(option_index))
        option_choice_id = f"option_choice_{option_label}"
        option_specs.append(
            {
                "option_label": str(option_label),
                "option_choice_id": str(option_choice_id),
                "mark_specs": [
                    {
                        "mark_id": f"{str(option_choice_id)}_mark_{int(index)}",
                        "object_type": str(mark["object_type"]),
                        "cell": [int(mark["cell"][0]), int(mark["cell"][1])],
                    }
                    for index, mark in enumerate(mark_specs, start=1)
                ],
                "is_correct": bool(int(option_index) == int(correct_index)),
            }
        )

    folded_mark_count = sum(
        1 for mark in result_mark_specs if str(mark["source_side"]) == "folded"
    )
    kept_mark_count = int(mark_count - folded_mark_count)
    return PaperFoldDataset(
        grid_size=int(grid_size),
        result_grid_cols=int(result_cols),
        result_grid_rows=int(result_rows),
        option_count=int(option_count),
        option_count_range=(int(option_count_min), int(option_count_max)),
        mark_count=int(mark_count),
        mark_count_range=(int(mark_count_min), int(mark_count_max)),
        fold_axis=str(fold_axis),
        fold_direction=str(direction),
        original_mark_specs=tuple(original_mark_specs),
        folded_result_mark_specs=tuple(result_mark_specs),
        option_specs=tuple(option_specs),
        answer_option_label=str(option_specs[int(correct_index)]["option_label"]),
        correct_option_index=int(correct_index),
        correct_option_choice_id=str(option_specs[int(correct_index)]["option_choice_id"]),
        folded_mark_count=int(folded_mark_count),
        kept_mark_count=int(kept_mark_count),
        option_count_probabilities=dict(option_count_probabilities),
        correct_option_index_probabilities=dict(correct_index_probabilities),
        correct_option_index_sampling_mode=str(correct_index_sampling_mode),
    )


def solver_trace(dataset: PaperFoldDataset) -> dict[str, Any]:
    """Return solver-facing symbolic facts for trace payloads."""

    return {
        "correct_option_label": str(dataset.answer_option_label),
        "correct_option_index": int(dataset.correct_option_index),
        "folded_result_mark_specs": [
            dict(item) for item in dataset.folded_result_mark_specs
        ],
    }


__all__ = [
    "folded_result_grid_dimensions",
    "sample_paper_fold_dataset",
    "solver_trace",
]
