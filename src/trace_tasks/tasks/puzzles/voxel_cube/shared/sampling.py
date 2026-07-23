"""Scene-local sampling primitives for voxel-cube puzzle tasks."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from trace_tasks.core.sampling import (
    integer_range_choice,
    sample_without_replacement,
    uniform_choice,
)

from .defaults import (
    int_bounds,
    resolve_answer_value,
    resolve_axis_choice,
    resolve_option_count,
)
from .rules import (
    complete_cuboid,
    corrupted_projection,
    cube_count,
    projection_grid,
    projection_signature,
    remove_top_cubes,
)
from .state import (
    CHANGE_TYPES,
    VIEW_DIRECTIONS,
    ChangeDataset,
    CountDataset,
    CubeStack,
    ProjectionCountDataset,
    ProjectionGrid,
    ProjectionMatchDataset,
    ProjectionOption,
)

_OPTION_LABELS = ("A", "B", "C", "D", "E", "F")


def sample_cube_count_dataset(
    *,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    rng,
) -> CountDataset:
    """Construct one cube-count case with a target count in support."""

    target, support = resolve_answer_value(
        params,
        generation_defaults,
        rng=rng,
        fallback_min=4,
        fallback_max=14,
    )
    stack = _random_stack_with_count(rng, int(target))
    return CountDataset(
        stack=stack,
        semantic_params={"answer_schema": "integer_count"},
        answer_support=tuple(range(int(support[0]), int(support[1]) + 1)),
        answer_value=int(target),
    )


def sample_structure_change_dataset(
    *,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    rng,
) -> ChangeDataset:
    """Construct a before/after structure-change count case."""

    change_type = resolve_axis_choice(
        params,
        key="change_type",
        support=CHANGE_TYPES,
        rng=rng,
    )
    target, support = resolve_answer_value(
        params,
        generation_defaults,
        rng=rng,
        fallback_min=1,
        fallback_max=5,
    )
    if str(change_type) == "missing_to_complete":
        reference, changed = _sample_missing_to_complete_pair(rng, int(target))
    else:
        reference, changed = _sample_removed_pair(rng, int(target))
    return ChangeDataset(
        stack=reference,
        reference_stack=reference,
        changed_stack=changed,
        semantic_params={
            "answer_schema": "integer_count",
            "change_type": str(change_type),
        },
        answer_support=tuple(range(int(support[0]), int(support[1]) + 1)),
        answer_value=int(target),
    )


def sample_visible_projection_dataset(
    *,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    rng,
) -> ProjectionCountDataset:
    """Construct one orthographic projection-cell count case."""

    direction = resolve_axis_choice(
        params,
        key="view_direction",
        support=VIEW_DIRECTIONS,
        rng=rng,
    )
    support = int_bounds(
        params,
        generation_defaults,
        min_key="answer_min",
        max_key="answer_max",
        fallback_min=2,
        fallback_max=9,
    )
    stack, projection = _sample_projection_count_case(
        rng,
        direction=str(direction),
        support=support,
    )
    return ProjectionCountDataset(
        stack=stack,
        projection=projection,
        semantic_params={
            "answer_schema": "integer_count",
            "view_direction": str(direction),
        },
        answer_support=tuple(range(int(support[0]), int(support[1]) + 1)),
        answer_value=len(projection.filled_cells),
    )


def sample_projection_match_dataset(
    *,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    rng,
) -> ProjectionMatchDataset:
    """Construct projection options with one uniquely correct matching panel."""

    direction = resolve_axis_choice(
        params,
        key="view_direction",
        support=VIEW_DIRECTIONS,
        rng=rng,
    )
    option_count = resolve_option_count(
        params,
        generation_defaults,
        fallback=4,
    )
    stack = _random_stack(rng)
    correct_projection = projection_grid(stack, str(direction))
    labels = _ordered_labels(option_count)
    correct_label = _select_option_label(rng, params=params, labels=labels)
    options = _projection_options(
        rng,
        labels=labels,
        correct_label=correct_label,
        correct_projection=correct_projection,
    )
    return ProjectionMatchDataset(
        stack=stack,
        query_projection=correct_projection,
        options=tuple(options),
        answer_label=correct_label,
        semantic_params={
            "answer_schema": "option_letter",
            "view_direction": str(direction),
            "option_count": int(option_count),
        },
        answer_support=tuple(labels),
    )


def _random_stack_with_count(rng, target_count: int) -> CubeStack:
    """Sample a compact height grid with exactly ``target_count`` cubes."""

    target = int(target_count)
    if target <= 0:
        raise ValueError("voxel cube count must be positive")
    for _attempt in range(200):
        rows = int(rng.choice((2, 3)))
        cols = int(rng.choice((2, 3)))
        max_height_value = int(rng.choice((2, 3)))
        if not rows * cols <= target <= rows * cols * max_height_value:
            continue
        heights = [1 for _cell in range(rows * cols)]
        remaining = target - len(heights)
        while remaining > 0:
            candidates = [
                index
                for index, value in enumerate(heights)
                if int(value) < max_height_value
            ]
            if not candidates:
                break
            selected = int(rng.choice(candidates))
            heights[selected] += 1
            remaining -= 1
        if sum(heights) == target:
            rows_out = []
            for row in range(rows):
                start = row * cols
                rows_out.append(
                    tuple(int(value) for value in heights[start : start + cols])
                )
            return CubeStack(tuple(rows_out))
    raise ValueError(f"could not sample voxel stack with count {target_count}")


def _random_stack(rng) -> CubeStack:
    """Sample a small compact stack for projection-oriented tasks."""

    target, _probabilities = integer_range_choice(rng, 5, 14)
    return _random_stack_with_count(rng, int(target))


def _sample_missing_to_complete_pair(
    rng,
    missing_count: int,
) -> tuple[CubeStack, CubeStack]:
    """Return a complete cuboid and a version missing top cubes."""

    target = int(missing_count)
    for _attempt in range(100):
        rows = int(rng.choice((2, 3)))
        cols = int(rng.choice((2, 3)))
        height = int(rng.choice((2, 3)))
        if rows * cols < target:
            continue
        reference = complete_cuboid(rows, cols, height)
        cells = tuple((row, col) for row in range(rows) for col in range(cols))
        removals = sample_without_replacement(rng, cells, target)
        changed = remove_top_cubes(reference, removals)
        return reference, changed
    raise ValueError("could not sample missing-to-complete voxel pair")


def _sample_removed_pair(rng, removed_count: int) -> tuple[CubeStack, CubeStack]:
    """Return a stack and a version with selected top cubes removed."""

    target = int(removed_count)
    for _attempt in range(100):
        reference = _random_stack(rng)
        occupied_cells = tuple(
            (row, col)
            for row, values in enumerate(reference.heights)
            for col, value in enumerate(values)
            if int(value) > 0
        )
        if len(occupied_cells) < target:
            continue
        removals = sample_without_replacement(rng, occupied_cells, target)
        changed = remove_top_cubes(reference, removals)
        return reference, changed
    raise ValueError("could not sample removed-cube voxel pair")


def _sample_projection_count_case(
    rng,
    *,
    direction: str,
    support: tuple[int, int],
) -> tuple[CubeStack, ProjectionGrid]:
    """Sample a stack whose selected projection count lies in support."""

    lower, upper = int(support[0]), int(support[1])
    for _attempt in range(300):
        stack = _random_stack(rng)
        projection = projection_grid(stack, str(direction))
        answer = len(projection.filled_cells)
        if lower <= int(answer) <= upper:
            return stack, projection
    raise ValueError("could not sample projection count case inside support")


def _projection_options(
    rng,
    *,
    labels: Sequence[str],
    correct_label: str,
    correct_projection: ProjectionGrid,
) -> tuple[ProjectionOption, ...]:
    """Build unique projection options around one correct projection."""

    signatures = {projection_signature(correct_projection)}
    options: list[ProjectionOption] = []
    for label in labels:
        if str(label) == str(correct_label):
            projection = correct_projection
            correct = True
        else:
            projection = _unique_corrupted_projection(
                rng,
                correct_projection,
                signatures=signatures,
            )
            correct = False
        signatures.add(projection_signature(projection))
        options.append(
            ProjectionOption(
                label=str(label),
                projection=projection,
                is_correct=bool(correct),
            )
        )
    return tuple(options)


def _unique_corrupted_projection(
    rng,
    projection: ProjectionGrid,
    *,
    signatures: set[tuple[object, ...]],
) -> ProjectionGrid:
    """Return a corrupted projection not already used by another option."""

    for _attempt in range(80):
        candidate = corrupted_projection(projection, rng=rng)
        signature = projection_signature(candidate)
        if signature not in signatures:
            return candidate
    raise ValueError("could not produce a unique projection distractor")


def _ordered_labels(option_count: int) -> tuple[str, ...]:
    """Return the first N canonical option labels."""

    count = int(option_count)
    if not 1 <= count <= len(_OPTION_LABELS):
        raise ValueError("unsupported option label count")
    return tuple(_OPTION_LABELS[:count])


def _select_option_label(
    rng,
    *,
    params: Mapping[str, Any],
    labels: Sequence[str],
) -> str:
    """Choose a correct option label from the available labels."""

    options = tuple(str(label) for label in labels)
    explicit = params.get("answer_option_label")
    if explicit is not None:
        selected = str(explicit)
        if selected not in options:
            raise ValueError(f"unsupported answer_option_label: {selected}")
        return selected
    return str(uniform_choice(rng, options))
