"""Surface-fixture dataset builders for count-style objectives."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.three_d.shared.semantic_colors import (
    COLOR_CONFUSION_EXCLUSIONS,
    colors_conflict as shared_colors_conflict,
    compatible_color_names,
    confusable_color_names as shared_confusable_color_names,
)

from .layout import grid_for_total, layout_cells, resolve_repeated_layout_style, target_ids_from_indices
from .rendering import layout_surface_element_grid
from .sampling import configured_int, resolve_color, resolve_int_support, sample_indices, uniform_int_probability_map
from .state import (
    ELEMENT_DISPLAY_NAME,
    ELEMENT_PLURAL,
    SEMANTIC_COLOR_SUPPORT,
    SEMANTIC_COLOR_RGB,
    SURFACE_FIXTURE_DISPLAY_NAME,
    semantic_color_label,
)


def _confusable_color_names(color_name: str) -> Tuple[str, ...]:
    """Return named colors that should not distract from a semantic target."""

    return shared_confusable_color_names(str(color_name))


def _colors_conflict(left: str, right: str) -> bool:
    """Return whether two semantic colors are too close for generated color readout."""

    return shared_colors_conflict(str(left), str(right))


def _readout_color_support(
    *,
    anchors: Sequence[str] = (),
    exclude: Sequence[str] = (),
) -> Tuple[str, ...]:
    """Resolve generated color choices after removing target-confusable names."""

    return compatible_color_names(
        SEMANTIC_COLOR_SUPPORT,
        anchors=tuple(str(color) for color in anchors),
        exclude=tuple(str(color) for color in exclude),
    )


def _sample_nonconflicting_readout_colors(
    *,
    rng: Any,
    count: int,
    required: Sequence[str] = (),
) -> Tuple[str, ...]:
    """Sample a small active palette with no pairwise close-color conflicts."""

    selected: list[str] = []
    for color in required:
        color_name = str(color)
        if color_name not in set(SEMANTIC_COLOR_SUPPORT):
            raise ValueError(f"unsupported required color: {color_name}")
        if color_name in selected:
            continue
        if any(_colors_conflict(color_name, existing) for existing in selected):
            raise ValueError(f"required colors are visually confusable: {required}")
        selected.append(color_name)

    pool = [str(color) for color in SEMANTIC_COLOR_SUPPORT if str(color) not in set(selected)]
    rng.shuffle(pool)
    for color in pool:
        if any(_colors_conflict(str(color), existing) for existing in selected):
            continue
        selected.append(str(color))
        if len(selected) >= int(count):
            return tuple(selected[: int(count)])
    raise ValueError(f"could not sample {count} non-conflicting semantic colors")


def base_surface_data(
    *,
    scene_variant: str,
    element_type: str,
    answer_value: int,
    target_element_ids: Sequence[str],
    surface_cells: Sequence[Mapping[str, Any]],
    rows: int,
    cols: int,
    layout_style: str,
    solver_trace: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build the scene dataset consumed by the renderer and output assembly."""

    data: Dict[str, Any] = {
        "scene_variant": str(scene_variant),
        "fixture_display_name": str(SURFACE_FIXTURE_DISPLAY_NAME[str(scene_variant)]),
        "target_element_type": str(element_type),
        "target_element_name": str(ELEMENT_DISPLAY_NAME[str(element_type)]),
        "target_element_plural": str(ELEMENT_PLURAL[str(element_type)]),
        "answer_value": int(answer_value),
        "target_element_ids": list(str(element_id) for element_id in target_element_ids),
        "surface_cells": [dict(cell) for cell in surface_cells],
        "layout_rows": int(rows),
        "layout_columns": int(cols),
        "layout_style": str(layout_style),
        "surface_world_corners": [
            [-2.0, 1.35, 2.55],
            [2.0, 1.35, 2.55],
            [2.0, 1.35, 0.15],
            [-2.0, 1.35, 0.15],
        ],
        "solver_trace": dict(solver_trace),
    }
    if extra:
        data.update(dict(extra))
    data["target_count"] = int(len([cell for cell in surface_cells if bool(cell.get("present", True))]))
    return data


def _resolve_visual_color_names(
    *,
    params: Mapping[str, Any],
    rng: Any,
) -> Tuple[str, ...]:
    """Choose visible object colors for non-color count tasks.

    These colors are a visual factor only. They are intentionally not part of
    the count predicate for repeated-element and extremum tasks.
    """

    explicit_colors = params.get("visual_color_names")
    if explicit_colors is None:
        explicit_colors = params.get("active_color_names", params.get("color_names"))
    if explicit_colors is not None:
        if not isinstance(explicit_colors, Sequence) or isinstance(explicit_colors, (str, bytes)):
            raise ValueError("visual_color_names must be a sequence of color names")
        colors = tuple(str(color) for color in explicit_colors)
        if len(colors) != len(set(colors)):
            raise ValueError("visual_color_names must be unique")
        if len(colors) < 2 or len(colors) > 4:
            raise ValueError("visual_color_names must contain 2 to 4 colors")
        unsupported = [color for color in colors if color not in set(SEMANTIC_COLOR_SUPPORT)]
        if unsupported:
            raise ValueError(f"unsupported visual color names: {unsupported}")
        return tuple(colors)

    color_count = int(
        params.get(
            "visual_color_count",
            params.get("active_color_count", params.get("color_count", 2 + int(rng.randrange(3)))),
        )
    )
    if color_count < 2 or color_count > 4:
        raise ValueError(f"visual_color_count must be in 2..4, got {color_count}")
    colors = list(str(color) for color in SEMANTIC_COLOR_SUPPORT)
    rng.shuffle(colors)
    return tuple(colors[: int(color_count)])


def _sample_visual_color_by_index(
    *,
    indices: Sequence[int],
    params: Mapping[str, Any],
    rng: Any,
) -> Tuple[Dict[int, str], Dict[str, int], Tuple[str, ...]]:
    """Assign canonical colors to each visible element for visual variety."""

    active_colors = _resolve_visual_color_names(params=params, rng=rng)
    index_sequence = [int(index) for index in indices]
    color_sequence: list[str] = []
    guaranteed_color_count = min(len(index_sequence), len(active_colors))
    color_sequence.extend(str(color) for color in active_colors[:guaranteed_color_count])
    for _ in range(max(0, len(index_sequence) - guaranteed_color_count)):
        color_sequence.append(str(active_colors[int(rng.randrange(len(active_colors)))]))
    rng.shuffle(color_sequence)
    color_by_index = {
        int(index): str(color_sequence[offset])
        for offset, index in enumerate(index_sequence)
    }
    color_counts = {str(color): 0 for color in active_colors}
    for color in color_by_index.values():
        color_counts[str(color)] = int(color_counts.get(str(color), 0)) + 1
    return color_by_index, color_counts, tuple(str(color) for color in active_colors)


def build_repeated_surface_data(
    *,
    namespace: str,
    scene_variant: str,
    element_type: str,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """Create a fixture with one repeated element type to count."""

    count, probabilities = resolve_int_support(
        params=params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.target_count",
        min_key="target_count_min",
        max_key="target_count_max",
        default_min=8,
        default_max=24,
        explicit_keys=("target_count", "element_count"),
        lower_bound=4,
        upper_bound=32,
    )
    rows, cols = layout_surface_element_grid(int(count))
    layout_family, layout_style, layout_family_probabilities, layout_style_probabilities = resolve_repeated_layout_style(
        scene_variant=str(scene_variant),
        rng=spawn_rng(int(instance_seed), f"{namespace}.layout_style"),
        params=params,
    )
    rng = spawn_rng(int(instance_seed), f"{namespace}.cells")
    indices = list(range(int(count)))
    color_by_index, visual_color_counts, visual_color_names = _sample_visual_color_by_index(
        indices=indices,
        params=params,
        rng=rng,
    )
    cells = layout_cells(
        scene_variant=str(scene_variant),
        element_type=str(element_type),
        rows=int(rows),
        cols=int(cols),
        present_indices=indices,
        target_indices=indices,
        rng=rng,
        layout_style=layout_style,
        color_by_index=color_by_index,
    )
    dataset = base_surface_data(
        scene_variant=str(scene_variant),
        element_type=str(element_type),
        answer_value=int(count),
        target_element_ids=target_ids_from_indices(cells, indices),
        surface_cells=cells,
        rows=int(rows),
        cols=int(cols),
        layout_style=layout_style,
        solver_trace={
            "count_predicate": "element_type == target_element_type",
            "target_element_type": str(element_type),
            "target_element_plural": str(ELEMENT_PLURAL[str(element_type)]),
            "target_count": int(count),
            "element_count": int(count),
            "layout_family": str(layout_family),
            "layout_family_probabilities": dict(layout_family_probabilities),
            "layout_style": str(layout_style),
            "layout_style_probabilities": dict(layout_style_probabilities),
            "visual_color_names": list(visual_color_names),
            "visual_color_counts": dict(visual_color_counts),
            "color_role": "non_semantic_visual_variation",
            "unique_integer_answer": True,
        },
        extra={
            "layout_family": str(layout_family),
            "layout_family_probabilities": dict(layout_family_probabilities),
            "layout_style_probabilities": dict(layout_style_probabilities),
            "visual_color_names": list(visual_color_names),
            "visual_color_counts": dict(visual_color_counts),
            "color_role": "non_semantic_visual_variation",
        },
    )
    return dataset, dict(probabilities)


def build_color_surface_data(
    *,
    namespace: str,
    scene_variant: str,
    element_type: str,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """Create a fixture where only a target color subset is counted."""

    target_count, target_probabilities = resolve_int_support(
        params=params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.target_count",
        min_key="target_count_min",
        max_key="target_count_max",
        default_min=3,
        default_max=10,
        explicit_keys=("target_count", "answer_count"),
        lower_bound=1,
        upper_bound=16,
    )
    distractor_count, _ = resolve_int_support(
        params=params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.distractor_count",
        min_key="distractor_count_min",
        max_key="distractor_count_max",
        default_min=5,
        default_max=12,
        explicit_keys=("distractor_count",),
        lower_bound=1,
        upper_bound=24,
    )
    total = int(target_count) + int(distractor_count)
    rows, cols = grid_for_total(total)
    total_slots = int(rows) * int(cols)
    rng = spawn_rng(int(instance_seed), f"{namespace}.cells")
    all_indices = list(range(total_slots))
    present_indices = sample_indices(rng, all_indices, total)
    target_indices = sample_indices(rng, present_indices, target_count)
    target_color = resolve_color(
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.target_color",
        explicit_key="target_color_name",
    )
    other_colors = list(_readout_color_support(anchors=(target_color,)))
    color_by_index: Dict[int, str] = {}
    target_set = set(target_indices)
    for index in present_indices:
        color_by_index[int(index)] = str(target_color if int(index) in target_set else other_colors[int(rng.randrange(len(other_colors)))])
    layout_style = "variable_grid" if str(scene_variant) in {"brick_wall", "paver_floor", "mailbox_bank"} else "uniform_grid"
    cells = layout_cells(
        scene_variant=str(scene_variant),
        element_type=str(element_type),
        rows=int(rows),
        cols=int(cols),
        present_indices=present_indices,
        target_indices=target_indices,
        rng=rng,
        layout_style=layout_style,
        color_by_index=color_by_index,
        semantic_color=True,
    )
    dataset = base_surface_data(
        scene_variant=str(scene_variant),
        element_type=str(element_type),
        answer_value=int(target_count),
        target_element_ids=target_ids_from_indices(cells, target_indices),
        surface_cells=cells,
        rows=int(rows),
        cols=int(cols),
        layout_style=layout_style,
        solver_trace={
            "count_predicate": "element_type == target_element_type and color_name == target_color_name",
            "target_color_name": str(target_color),
            "target_count": int(target_count),
            "distractor_count": int(distractor_count),
            "unique_integer_answer": True,
        },
        extra={"target_color_name": str(target_color)},
    )
    return dataset, dict(target_probabilities)


COLOR_FREQUENCY_OPTION_LABELS: Tuple[str, ...] = ("A", "B", "C", "D", "E", "F")
COLOR_FREQUENCY_MAXIMUM_PROGRAM = "option_color_count_maximum"
COLOR_FREQUENCY_ZERO_PROGRAM = "option_color_count_zero"


def _resolve_color_option_names(
    *,
    params: Mapping[str, Any],
    answer_label: str,
    rng: Any,
) -> Tuple[str, ...]:
    raw_options = params.get("option_color_names", params.get("color_option_names"))
    if raw_options is not None:
        if not isinstance(raw_options, Sequence) or isinstance(raw_options, (str, bytes)):
            raise ValueError("option_color_names must be a sequence of six color names")
        colors = tuple(str(color) for color in raw_options)
        if len(colors) != len(COLOR_FREQUENCY_OPTION_LABELS):
            raise ValueError("option_color_names must contain exactly six colors")
        if len(colors) != len(set(colors)):
            raise ValueError("option_color_names must be unique")
        unsupported = [color for color in colors if color not in set(SEMANTIC_COLOR_SUPPORT)]
        if unsupported:
            raise ValueError(f"unsupported option color names: {unsupported}")
        return tuple(colors)

    answer_color = params.get("answer_color_name")
    if answer_color is not None:
        answer = str(answer_color)
        if answer not in set(SEMANTIC_COLOR_SUPPORT):
            raise ValueError(f"unsupported answer_color_name: {answer}")
        colors = list(_readout_color_support(anchors=(answer,)))
        rng.shuffle(colors)
        option_colors = colors[: len(COLOR_FREQUENCY_OPTION_LABELS) - 1]
        option_colors.insert(int(COLOR_FREQUENCY_OPTION_LABELS.index(str(answer_label))), str(answer))
        return tuple(option_colors)

    colors = list(str(color) for color in SEMANTIC_COLOR_SUPPORT)
    rng.shuffle(colors)
    answer = str(colors[0])
    option_colors = list(_readout_color_support(anchors=(answer,)))
    rng.shuffle(option_colors)
    option_colors = option_colors[: len(COLOR_FREQUENCY_OPTION_LABELS) - 1]
    option_colors.insert(int(COLOR_FREQUENCY_OPTION_LABELS.index(str(answer_label))), str(answer))
    return tuple(option_colors)


def _resolve_option_color_counts(
    *,
    frequency_program: str,
    option_colors: Sequence[str],
    answer_label: str,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
    rng: Any,
) -> Tuple[Dict[str, int], Dict[str, float]]:
    """Construct option-color counts for the text-option frequency program.

    The shared metric layer deliberately works with internal program names
    instead of public query ids. The invariant is one unique answer option:
    either exactly one visible color has the maximum count, or exactly one
    listed color is absent from the fixture.
    """

    labels = COLOR_FREQUENCY_OPTION_LABELS
    answer_color = str(option_colors[int(labels.index(str(answer_label)))])
    raw_counts = params.get("color_counts_by_name", params.get("color_counts"))
    if raw_counts is not None:
        if not isinstance(raw_counts, Mapping):
            raise ValueError("color_counts_by_name must be a mapping from option color name to count")
        counts = {str(color): int(raw_counts.get(str(color), 0)) for color in option_colors}
        if str(frequency_program) == COLOR_FREQUENCY_ZERO_PROGRAM:
            if int(counts[str(answer_color)]) != 0:
                raise ValueError("answer color must have zero count for zero-count option program")
            if sum(1 for count in counts.values() if int(count) == 0) != 1:
                raise ValueError("zero-count option program requires exactly one zero-count option color")
        else:
            max_count = max(int(count) for count in counts.values())
            if int(counts[str(answer_color)]) != int(max_count):
                raise ValueError("answer color must be the unique maximum for maximum-count option program")
            if sum(1 for count in counts.values() if int(count) == int(max_count)) != 1:
                raise ValueError("maximum-count option program requires a unique maximum color count")
            if any(int(count) <= 0 for count in counts.values()):
                raise ValueError("maximum-count option program requires every option color to appear")
        total = int(sum(int(count) for count in counts.values()))
        return counts, uniform_int_probability_map(range(max(1, total), max(1, total) + 1), selected=total)

    total_count, total_probabilities = resolve_int_support(
        params=params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.total_count",
        min_key="total_count_min",
        max_key="total_count_max",
        default_min=15,
        default_max=24,
        explicit_keys=("total_count",),
        lower_bound=8,
        upper_bound=36,
    )
    counts = {str(color): 0 for color in option_colors}
    if str(frequency_program) == COLOR_FREQUENCY_ZERO_PROGRAM:
        present_colors = [str(color) for color in option_colors if str(color) != str(answer_color)]
        if int(total_count) < len(present_colors):
            raise ValueError("total_count too small for zero-count option present colors")
        for color in present_colors:
            counts[color] = 1
        remaining = int(total_count) - len(present_colors)
        for _ in range(int(remaining)):
            color = present_colors[int(rng.randrange(len(present_colors)))]
            counts[str(color)] += 1
        counts[str(answer_color)] = 0
        return counts, dict(total_probabilities)

    target_min = max(4, configured_int(params, gen_defaults, "max_color_count_min", 5))
    target_max = max(target_min, configured_int(params, gen_defaults, "max_color_count_max", 9))
    target_max = min(int(target_max), int(total_count) - len(option_colors) + 1)
    if target_max < target_min:
        raise ValueError("total_count too small for maximum-count option support")
    max_count = int(target_min + int(rng.randrange(int(target_max - target_min + 1))))
    other_colors = [str(color) for color in option_colors if str(color) != str(answer_color)]
    counts[str(answer_color)] = int(max_count)
    for color in other_colors:
        counts[color] = 1
    remaining = int(total_count) - int(max_count) - len(other_colors)
    for _ in range(max(0, int(remaining))):
        eligible = [color for color in other_colors if int(counts[str(color)]) < int(max_count) - 1]
        if not eligible:
            raise ValueError("could not allocate option counts with a unique maximum")
        color = str(eligible[int(rng.randrange(len(eligible)))])
        counts[color] += 1
    if int(counts[str(answer_color)]) <= max(int(counts[color]) for color in other_colors):
        raise ValueError("failed to construct unique most frequent color")
    return counts, dict(total_probabilities)


def build_color_frequency_option_surface_data(
    *,
    namespace: str,
    scene_variant: str,
    element_type: str,
    frequency_program: str,
    answer_label: str,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """Create one colored fixture plus six visual color-answer options."""

    if str(frequency_program) not in {COLOR_FREQUENCY_MAXIMUM_PROGRAM, COLOR_FREQUENCY_ZERO_PROGRAM}:
        raise ValueError(f"unsupported color-frequency program: {frequency_program}")
    if str(answer_label) not in set(COLOR_FREQUENCY_OPTION_LABELS):
        raise ValueError(f"unsupported answer_label: {answer_label}")
    rng = spawn_rng(int(instance_seed), f"{namespace}.cells")
    option_colors = _resolve_color_option_names(
        params=params,
        answer_label=str(answer_label),
        rng=rng,
    )
    color_counts, answer_probabilities = _resolve_option_color_counts(
        frequency_program=str(frequency_program),
        option_colors=option_colors,
        answer_label=str(answer_label),
        params=params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        namespace=str(namespace),
        rng=rng,
    )
    total = int(sum(int(count) for count in color_counts.values()))
    rows, cols = grid_for_total(total)
    total_slots = int(rows) * int(cols)
    present_indices = sample_indices(rng, list(range(total_slots)), total)
    answer_color = str(option_colors[int(COLOR_FREQUENCY_OPTION_LABELS.index(str(answer_label)))])
    color_sequence: list[str] = []
    for color in option_colors:
        color_sequence.extend([str(color)] * int(color_counts[str(color)]))
    if len(color_sequence) != len(present_indices):
        raise ValueError("color-frequency count vector does not match present indices")
    rng.shuffle(color_sequence)
    color_by_index = {int(index): str(color_sequence[offset]) for offset, index in enumerate(present_indices)}
    target_indices = [int(index) for index in present_indices if str(color_by_index[int(index)]) == str(answer_color)]
    layout_style = "variable_grid" if str(scene_variant) in {"brick_wall", "paver_floor", "mailbox_bank"} else "uniform_grid"
    cells = layout_cells(
        scene_variant=str(scene_variant),
        element_type=str(element_type),
        rows=int(rows),
        cols=int(cols),
        present_indices=present_indices,
        target_indices=target_indices,
        rng=rng,
        layout_style=layout_style,
        color_by_index=color_by_index,
        semantic_color=True,
    )
    option_records = [
        {
            "label": str(label),
            "color_name": str(color),
            "fill_rgb": list(SEMANTIC_COLOR_RGB[str(color)]),
            "visible_count": int(color_counts[str(color)]),
        }
        for label, color in zip(COLOR_FREQUENCY_OPTION_LABELS, option_colors)
    ]
    dataset = base_surface_data(
        scene_variant=str(scene_variant),
        element_type=str(element_type),
        answer_value=int(color_counts[str(answer_color)]),
        target_element_ids=target_ids_from_indices(cells, target_indices),
        surface_cells=cells,
        rows=int(rows),
        cols=int(cols),
        layout_style=layout_style,
        solver_trace={
            "program": str(frequency_program),
            "option_color_counts": dict(color_counts),
            "answer_label": str(answer_label),
            "answer_color_name": str(answer_color),
            "answer_color_count": int(color_counts[str(answer_color)]),
            "unique_answer": True,
        },
        extra={
            "option_labels": list(COLOR_FREQUENCY_OPTION_LABELS),
            "option_records": list(option_records),
            "option_color_names": [str(color) for color in option_colors],
            "option_color_counts": dict(color_counts),
            "answer_label": str(answer_label),
            "answer_color_name": str(answer_color),
            "answer_color_count": int(color_counts[str(answer_color)]),
            "target_color_name": str(answer_color),
            "frequency_program": str(frequency_program),
        },
    )
    return dataset, dict(answer_probabilities)


def _resolve_active_colors(
    *,
    params: Mapping[str, Any],
    target_color: str,
    rng: Any,
) -> Tuple[str, ...]:
    explicit_colors = params.get("active_color_names", params.get("color_names"))
    if explicit_colors is not None:
        if not isinstance(explicit_colors, Sequence) or isinstance(explicit_colors, (str, bytes)):
            raise ValueError("active_color_names must be a sequence of color names")
        colors = tuple(str(color) for color in explicit_colors)
        if len(colors) != len(set(colors)):
            raise ValueError("active_color_names must be unique")
        if len(colors) < 2 or len(colors) > 4:
            raise ValueError("active_color_names must contain 2 to 4 colors")
        if str(target_color) not in set(colors):
            raise ValueError("active_color_names must include target_color_name")
        unsupported = [color for color in colors if color not in set(SEMANTIC_COLOR_SUPPORT)]
        if unsupported:
            raise ValueError(f"unsupported active color names: {unsupported}")
        return tuple(colors)

    color_count = int(params.get("active_color_count", params.get("color_count", 2 + int(rng.randrange(3)))))
    if color_count < 2 or color_count > 4:
        raise ValueError(f"active_color_count must be in 2..4, got {color_count}")
    return _sample_nonconflicting_readout_colors(
        rng=rng,
        count=int(color_count),
        required=(str(target_color),),
    )


def _resolve_initial_color_counts(
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    active_colors: Sequence[str],
    target_color: str,
    instance_seed: int,
    namespace: str,
    rng: Any,
) -> Tuple[Dict[str, int], Dict[str, float]]:
    """Bind the visible starting multicolor state while preserving target annotation.

    The key invariant for operation tasks is that `target_color` has a known
    positive initial count and every active distractor color is visible at least
    once, so later text operations can change color counts without changing the
    visual witness set.
    """

    raw_counts = params.get("initial_color_counts")
    if raw_counts is not None:
        if not isinstance(raw_counts, Mapping):
            raise ValueError("initial_color_counts must be a mapping from color name to count")
        counts = {str(color): int(raw_counts.get(str(color), 0)) for color in active_colors}
        if counts[str(target_color)] <= 0:
            raise ValueError("initial target color count must be positive")
        if any(int(count) <= 0 for count in counts.values()):
            raise ValueError("each active color must have at least one initial object")
        return counts, uniform_int_probability_map(range(1, 13), selected=int(counts[str(target_color)]))

    target_count, target_probabilities = resolve_int_support(
        params=params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.initial_target_count",
        min_key="initial_target_count_min",
        max_key="initial_target_count_max",
        default_min=2,
        default_max=5,
        explicit_keys=("initial_target_count",),
        lower_bound=1,
        upper_bound=10,
    )
    minimum_total = int(target_count) + len(active_colors) - 1
    initial_total, _ = resolve_int_support(
        params=params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.initial_total_count",
        min_key="initial_total_count_min",
        max_key="initial_total_count_max",
        default_min=8,
        default_max=14,
        explicit_keys=("initial_total_count", "total_count"),
        lower_bound=minimum_total,
        upper_bound=20,
    )
    counts = {str(color): 0 for color in active_colors}
    counts[str(target_color)] = int(target_count)
    other_colors = [str(color) for color in active_colors if str(color) != str(target_color)]
    for color in other_colors:
        counts[color] = 1
    remaining = int(initial_total) - int(target_count) - len(other_colors)
    for _ in range(max(0, int(remaining))):
        color = other_colors[int(rng.randrange(len(other_colors)))]
        counts[str(color)] += 1
    return counts, dict(target_probabilities)


def _coerce_operation_records(raw_operations: Any) -> list[Dict[str, Any]]:
    if not isinstance(raw_operations, Sequence) or isinstance(raw_operations, (str, bytes)):
        raise ValueError("operations must be a sequence")
    operations: list[Dict[str, Any]] = []
    for raw in raw_operations:
        if isinstance(raw, Mapping):
            action = str(raw.get("action", "")).lower()
            color_name = str(raw.get("color_name", raw.get("color", "")))
            count = int(raw.get("count", raw.get("amount", 0)))
        elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) and len(raw) == 3:
            action = str(raw[0]).lower()
            color_name = str(raw[1])
            count = int(raw[2])
        else:
            raise ValueError("each operation must be a mapping or (action, color, count) triple")
        operations.append({"action": action, "color_name": color_name, "count": int(count)})
    return operations


def _apply_operation_sequence(
    *,
    initial_counts: Mapping[str, int],
    active_colors: Sequence[str],
    operations: Sequence[Mapping[str, Any]],
) -> Dict[str, int]:
    counts = {str(color): int(initial_counts[str(color)]) for color in active_colors}
    active_set = set(str(color) for color in active_colors)
    for operation in operations:
        action = str(operation["action"])
        color_name = str(operation["color_name"])
        amount = int(operation["count"])
        if action not in {"add", "remove"}:
            raise ValueError(f"unsupported operation action: {action}")
        if color_name not in active_set:
            raise ValueError(f"operation color {color_name!r} is not an active fixture color")
        if amount <= 0:
            raise ValueError("operation counts must be positive")
        if action == "add":
            counts[color_name] += int(amount)
        else:
            if counts[color_name] < int(amount):
                raise ValueError(f"cannot remove {amount} {color_name} elements from current count {counts[color_name]}")
            counts[color_name] -= int(amount)
    return counts


def _sample_operation_sequence(
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    active_colors: Sequence[str],
    target_color: str,
    initial_counts: Mapping[str, int],
    instance_seed: int,
    namespace: str,
) -> Tuple[list[Dict[str, Any]], Dict[str, int], Dict[str, float]]:
    """Sample or validate the three text operations for the final-count program.

    The sequence is symbolic, not rendered. It must include a target-color
    operation, keep all color counts nonnegative, and change the final target
    count so this task remains distinct from direct colored-element counting.
    """

    amount_min = max(1, configured_int(params, gen_defaults, "operation_amount_min", 1))
    amount_max = max(amount_min, min(5, configured_int(params, gen_defaults, "operation_amount_max", 3)))
    final_min = max(0, configured_int(params, gen_defaults, "final_count_min", 1))
    final_max = max(final_min, min(20, configured_int(params, gen_defaults, "final_count_max", 9)))
    answer_probabilities = uniform_int_probability_map(range(int(final_min), int(final_max) + 1))
    if "operations" in params:
        operations = _coerce_operation_records(params["operations"])
        if len(operations) != 3:
            raise ValueError("surface-fixture operation tasks require exactly three operations")
        final_counts = _apply_operation_sequence(
            initial_counts=initial_counts,
            active_colors=active_colors,
            operations=operations,
        )
        final_target_count = int(final_counts[str(target_color)])
        if final_target_count == int(initial_counts[str(target_color)]):
            raise ValueError("target-color operations must change the target-color count")
        if final_target_count < int(final_min) or final_target_count > int(final_max):
            raise ValueError(f"final target count {final_target_count} outside configured support {final_min}..{final_max}")
        return operations, final_counts, uniform_int_probability_map(range(int(final_min), int(final_max) + 1), selected=final_target_count)

    rng = spawn_rng(int(instance_seed), f"{namespace}.operations")
    non_target_colors = [str(color) for color in active_colors if str(color) != str(target_color)]
    for _attempt in range(200):
        target_op_count = 2 if rng.random() < 0.45 else 1
        positions = list(range(3))
        rng.shuffle(positions)
        target_positions = set(positions[:target_op_count])
        colors = [
            str(target_color) if index in target_positions else non_target_colors[int(rng.randrange(len(non_target_colors)))]
            for index in range(3)
        ]
        current_counts = {str(color): int(initial_counts[str(color)]) for color in active_colors}
        operations: list[Dict[str, Any]] = []
        for color_name in colors:
            action = ("add", "remove")[int(rng.randrange(2))]
            if action == "remove" and int(current_counts[color_name]) < int(amount_min):
                action = "add"
            if action == "remove":
                max_amount = min(int(amount_max), int(current_counts[color_name]))
                amount = int(amount_min + int(rng.randrange(max_amount - amount_min + 1)))
                current_counts[color_name] -= int(amount)
            else:
                amount = int(amount_min + int(rng.randrange(amount_max - amount_min + 1)))
                current_counts[color_name] += int(amount)
            operations.append({"action": str(action), "color_name": str(color_name), "count": int(amount)})
        final_target_count = int(current_counts[str(target_color)])
        if final_target_count == int(initial_counts[str(target_color)]):
            continue
        if final_target_count < int(final_min) or final_target_count > int(final_max):
            continue
        return operations, current_counts, dict(answer_probabilities)
    raise ValueError("could not sample valid color operation sequence")


def _operation_phrase(
    operations: Sequence[Mapping[str, Any]],
    *,
    element_name: str,
    element_plural: str,
) -> str:
    phrases = []
    for operation in operations:
        amount = int(operation["count"])
        noun = str(element_name if amount == 1 else element_plural)
        color_label = semantic_color_label(str(operation["color_name"]))
        phrases.append(f"{operation['action']} {amount} {color_label} {noun}")
    return "; ".join(phrases)


def build_color_operation_surface_data(
    *,
    namespace: str,
    scene_variant: str,
    element_type: str,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """Create a colored fixture plus three hypothetical add/remove operations."""

    target_color = resolve_color(
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.target_color",
        explicit_key="target_color_name",
    )
    rng = spawn_rng(int(instance_seed), f"{namespace}.cells")
    active_colors = _resolve_active_colors(params=params, target_color=str(target_color), rng=rng)
    initial_counts, _initial_target_probabilities = _resolve_initial_color_counts(
        params=params,
        gen_defaults=gen_defaults,
        active_colors=active_colors,
        target_color=str(target_color),
        instance_seed=int(instance_seed),
        namespace=str(namespace),
        rng=rng,
    )
    operations, final_counts, answer_probabilities = _sample_operation_sequence(
        params=params,
        gen_defaults=gen_defaults,
        active_colors=active_colors,
        target_color=str(target_color),
        initial_counts=initial_counts,
        instance_seed=int(instance_seed),
        namespace=str(namespace),
    )
    initial_total = int(sum(int(count) for count in initial_counts.values()))
    rows, cols = grid_for_total(initial_total)
    total_slots = int(rows) * int(cols)
    present_indices = sample_indices(rng, list(range(total_slots)), initial_total)
    color_sequence: list[str] = []
    for color in active_colors:
        color_sequence.extend([str(color)] * int(initial_counts[str(color)]))
    rng.shuffle(color_sequence)
    color_by_index = {int(index): str(color_sequence[offset]) for offset, index in enumerate(present_indices)}
    target_indices = [int(index) for index in present_indices if str(color_by_index[int(index)]) == str(target_color)]
    layout_style = "variable_grid" if str(scene_variant) in {"brick_wall", "paver_floor", "mailbox_bank"} else "uniform_grid"
    cells = layout_cells(
        scene_variant=str(scene_variant),
        element_type=str(element_type),
        rows=int(rows),
        cols=int(cols),
        present_indices=present_indices,
        target_indices=target_indices,
        rng=rng,
        layout_style=layout_style,
        color_by_index=color_by_index,
        semantic_color=True,
    )
    initial_target_count = int(initial_counts[str(target_color)])
    final_target_count = int(final_counts[str(target_color)])
    operation_phrase = _operation_phrase(
        operations,
        element_name=str(ELEMENT_DISPLAY_NAME[str(element_type)]),
        element_plural=str(ELEMENT_PLURAL[str(element_type)]),
    )
    dataset = base_surface_data(
        scene_variant=str(scene_variant),
        element_type=str(element_type),
        answer_value=int(final_target_count),
        target_element_ids=target_ids_from_indices(cells, target_indices),
        surface_cells=cells,
        rows=int(rows),
        cols=int(cols),
        layout_style=layout_style,
        solver_trace={
            "count_program": "initial_target_color_count plus signed matching-color operation deltas",
            "visual_count_predicate": "element_type == target_element_type and color_name == target_color_name",
            "target_color_name": str(target_color),
            "initial_target_count": int(initial_target_count),
            "initial_color_counts": dict(initial_counts),
            "operations": [dict(operation) for operation in operations],
            "final_color_counts": dict(final_counts),
            "final_target_count": int(final_target_count),
            "target_color_delta": int(final_target_count - initial_target_count),
            "unique_integer_answer": True,
        },
        extra={
            "target_color_name": str(target_color),
            "active_color_names": list(active_colors),
            "initial_target_count": int(initial_target_count),
            "initial_color_counts": dict(initial_counts),
            "operations": [dict(operation) for operation in operations],
            "final_color_counts": dict(final_counts),
            "operation_phrase": str(operation_phrase),
        },
    )
    return dataset, dict(answer_probabilities)


def _resolve_recolor_active_colors(
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
) -> Tuple[str, ...]:
    """Resolve the color vocabulary for one recolor-board matching instance."""

    explicit_colors = params.get("active_color_names", params.get("color_names"))
    if explicit_colors is not None:
        if not isinstance(explicit_colors, Sequence) or isinstance(explicit_colors, (str, bytes)):
            raise ValueError("active_color_names must be a sequence of color names")
        colors = tuple(str(color) for color in explicit_colors)
        if len(colors) != len(set(colors)):
            raise ValueError("active_color_names must be unique")
        if len(colors) < 3 or len(colors) > 4:
            raise ValueError("active_color_names must contain 3 or 4 colors")
        unsupported = [color for color in colors if color not in set(SEMANTIC_COLOR_SUPPORT)]
        if unsupported:
            raise ValueError(f"unsupported active color names: {unsupported}")
        return tuple(colors)

    minimum = max(3, configured_int(params, gen_defaults, "active_color_count_min", 3))
    maximum = max(minimum, min(4, configured_int(params, gen_defaults, "active_color_count_max", 4)))
    explicit_count = params.get("active_color_count")
    if explicit_count is not None:
        color_count = int(explicit_count)
        if color_count < int(minimum) or color_count > int(maximum):
            raise ValueError(f"active_color_count must be in {minimum}..{maximum}, got {color_count}")
    else:
        rng_count = spawn_rng(int(instance_seed), f"{namespace}.active_color_count")
        color_count = int(minimum + int(rng_count.randrange(int(maximum - minimum + 1))))
    rng = spawn_rng(int(instance_seed), f"{namespace}.active_colors")
    return _sample_nonconflicting_readout_colors(rng=rng, count=int(color_count))


def _resolve_recolor_rule(
    *,
    params: Mapping[str, Any],
    active_colors: Sequence[str],
    instance_seed: int,
    namespace: str,
) -> Tuple[str, str]:
    """Choose the single source->destination recolor rule over active colors."""

    active = tuple(str(color) for color in active_colors)
    source = params.get("source_color_name", params.get("source_color"))
    destination = params.get("destination_color_name", params.get("target_color_name", params.get("destination_color")))
    rng = spawn_rng(int(instance_seed), f"{namespace}.recolor_rule")
    if source is None:
        source_color = str(active[int(rng.randrange(len(active)))])
    else:
        source_color = str(source)
        if source_color not in set(active):
            raise ValueError(f"source_color_name must be one of active colors: {source_color}")
    destination_support = [str(color) for color in active if str(color) != str(source_color)]
    if destination is None:
        destination_color = str(destination_support[int(rng.randrange(len(destination_support)))])
    else:
        destination_color = str(destination)
        if destination_color not in set(destination_support):
            raise ValueError("destination_color_name must be an active color different from source_color_name")
    return str(source_color), str(destination_color)


def _resolve_recolor_initial_counts(
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    active_colors: Sequence[str],
    source_color: str,
    instance_seed: int,
    namespace: str,
) -> Tuple[Dict[str, int], Dict[str, float]]:
    """Build the original board color-count vector for recolor matching.

    The source color is forced to have at least two objects by default, making
    the single recolor visually consequential while every other active color
    remains present for option-board distractors.
    """

    raw_counts = params.get("initial_color_counts")
    active = tuple(str(color) for color in active_colors)
    if raw_counts is not None:
        if not isinstance(raw_counts, Mapping):
            raise ValueError("initial_color_counts must be a mapping from color name to count")
        counts = {str(color): int(raw_counts.get(str(color), 0)) for color in active}
        if any(int(count) <= 0 for count in counts.values()):
            raise ValueError("each active color must have at least one initial object")
        if int(counts[str(source_color)]) <= 0:
            raise ValueError("source color must have at least one initial object")
        total = int(sum(counts.values()))
        return counts, uniform_int_probability_map(range(1, max(2, total + 1)), selected=total)

    source_count, _source_probabilities = resolve_int_support(
        params=params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.source_count",
        min_key="initial_source_count_min",
        max_key="initial_source_count_max",
        default_min=2,
        default_max=5,
        explicit_keys=("initial_source_count", "source_count"),
        lower_bound=1,
        upper_bound=10,
    )
    minimum_total = int(source_count) + len(active) - 1
    initial_total, total_probabilities = resolve_int_support(
        params=params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.initial_total_count",
        min_key="initial_total_count_min",
        max_key="initial_total_count_max",
        default_min=8,
        default_max=14,
        explicit_keys=("initial_total_count", "total_count"),
        lower_bound=minimum_total,
        upper_bound=20,
    )
    counts = {str(color): 1 for color in active}
    counts[str(source_color)] = int(source_count)
    remaining = int(initial_total) - int(source_count) - (len(active) - 1)
    rng = spawn_rng(int(instance_seed), f"{namespace}.initial_count_allocation")
    for _ in range(max(0, int(remaining))):
        color = str(active[int(rng.randrange(len(active)))])
        counts[color] += 1
    return counts, dict(total_probabilities)


def _apply_single_recolor(
    *,
    initial_counts: Mapping[str, int],
    source_color: str,
    destination_color: str,
) -> Dict[str, int]:
    counts = {str(color): int(count) for color, count in initial_counts.items()}
    moved = int(counts[str(source_color)])
    counts[str(source_color)] = 0
    counts[str(destination_color)] += int(moved)
    return counts


def _count_vector_key(counts: Mapping[str, int], colors: Sequence[str]) -> Tuple[int, ...]:
    return tuple(int(counts[str(color)]) for color in colors)


def _with_color_transfer(
    counts: Mapping[str, int],
    *,
    from_color: str,
    to_color: str,
    amount: int = 1,
) -> Dict[str, int] | None:
    if str(from_color) == str(to_color) or int(counts[str(from_color)]) < int(amount):
        return None
    updated = {str(color): int(count) for color, count in counts.items()}
    updated[str(from_color)] -= int(amount)
    updated[str(to_color)] += int(amount)
    return updated


def _sample_recolor_option_counts(
    *,
    params: Mapping[str, Any],
    active_colors: Sequence[str],
    initial_counts: Mapping[str, int],
    final_counts: Mapping[str, int],
    source_color: str,
    destination_color: str,
    answer_label: str,
    instance_seed: int,
    namespace: str,
) -> Dict[str, Dict[str, int]]:
    """Create four unique answer-option count vectors for the visual MCQ.

    The correct vector is the full source->destination recolor. Distractors
    preserve total object count but encode common mistakes: unchanged board,
    reversed recolor, partial recolor, wrong destination, or one-object transfer.
    """

    labels = ("A", "B", "C", "D")
    raw = params.get("option_color_counts_by_label")
    active = tuple(str(color) for color in active_colors)
    final_key = _count_vector_key(final_counts, active)
    if raw is not None:
        if not isinstance(raw, Mapping):
            raise ValueError("option_color_counts_by_label must be a mapping")
        options = {
            str(label): {str(color): int(raw[str(label)][str(color)]) for color in active}
            for label in labels
        }
        if _count_vector_key(options[str(answer_label)], active) != final_key:
            raise ValueError("answer_label option must match final recolor counts")
        if len({_count_vector_key(counts, active) for counts in options.values()}) != len(labels):
            raise ValueError("option color count vectors must be unique")
        return options

    candidates: list[Dict[str, int]] = []
    seen = {final_key}

    def add_candidate(candidate: Mapping[str, int] | None) -> None:
        if candidate is None:
            return
        normalized = {str(color): int(candidate[str(color)]) for color in active}
        if any(int(value) < 0 for value in normalized.values()):
            return
        if int(sum(normalized.values())) != int(sum(int(value) for value in initial_counts.values())):
            return
        key = _count_vector_key(normalized, active)
        if key in seen:
            return
        seen.add(key)
        candidates.append(normalized)

    add_candidate(initial_counts)
    reverse = {str(color): int(initial_counts[str(color)]) for color in active}
    reverse[str(source_color)] += int(reverse[str(destination_color)])
    reverse[str(destination_color)] = 0
    add_candidate(reverse)
    if int(initial_counts[str(source_color)]) > 1:
        partial = {str(color): int(final_counts[str(color)]) for color in active}
        partial[str(source_color)] = 1
        partial[str(destination_color)] -= 1
        add_candidate(partial)
    for color in active:
        if str(color) not in {str(source_color), str(destination_color)}:
            wrong_destination = {str(name): int(initial_counts[str(name)]) for name in active}
            wrong_destination[str(color)] += int(wrong_destination[str(source_color)])
            wrong_destination[str(source_color)] = 0
            add_candidate(wrong_destination)

    rng = spawn_rng(int(instance_seed), f"{namespace}.distractor_counts")
    for _attempt in range(200):
        from_options = [str(color) for color in active if int(final_counts[str(color)]) > 0]
        from_color = str(from_options[int(rng.randrange(len(from_options)))])
        to_options = [str(color) for color in active if str(color) != str(from_color)]
        to_color = str(to_options[int(rng.randrange(len(to_options)))])
        add_candidate(
            _with_color_transfer(
                final_counts,
                from_color=from_color,
                to_color=to_color,
            )
        )
        if len(candidates) >= 3:
            break
    if len(candidates) < 3:
        raise ValueError("could not build three unique recolor distractor options")

    rng.shuffle(candidates)
    options: Dict[str, Dict[str, int]] = {}
    distractor_iter = iter(candidates[:3])
    for label in labels:
        if str(label) == str(answer_label):
            options[str(label)] = {str(color): int(final_counts[str(color)]) for color in active}
        else:
            options[str(label)] = dict(next(distractor_iter))
    return options


def _colored_dataset_from_counts(
    *,
    scene_variant: str,
    element_type: str,
    rows: int,
    cols: int,
    present_indices: Sequence[int],
    color_counts: Mapping[str, int],
    rng: Any,
    layout_style: str,
    option_label: str | None = None,
) -> Dict[str, Any]:
    """Materialize one board dataset from a color-count vector.

    The geometry is fixed by the shared rows/columns/present indices; only the
    color assignment is shuffled, which lets recolor options rearrange objects
    while preserving the same fixture grammar and object rendering path.
    """

    color_sequence: list[str] = []
    for color, count in color_counts.items():
        color_sequence.extend([str(color)] * int(count))
    if len(color_sequence) != len(present_indices):
        raise ValueError("color_counts total must equal present index count")
    rng.shuffle(color_sequence)
    color_by_index = {int(index): str(color_sequence[offset]) for offset, index in enumerate(present_indices)}
    cells = layout_cells(
        scene_variant=str(scene_variant),
        element_type=str(element_type),
        rows=int(rows),
        cols=int(cols),
        present_indices=present_indices,
        target_indices=[],
        rng=rng,
        layout_style=str(layout_style),
        color_by_index=color_by_index,
        semantic_color=True,
    )
    extra: Dict[str, Any] = {
        "color_counts": {str(color): int(count) for color, count in color_counts.items()},
        "active_color_names": [str(color) for color in color_counts.keys()],
    }
    if option_label is not None:
        extra["option_label"] = str(option_label)
    return base_surface_data(
        scene_variant=str(scene_variant),
        element_type=str(element_type),
        answer_value=0,
        target_element_ids=[],
        surface_cells=cells,
        rows=int(rows),
        cols=int(cols),
        layout_style=str(layout_style),
        solver_trace={
            "count_predicate": "color_counts_by_visible_surface_elements",
            "color_counts": {str(color): int(count) for color, count in color_counts.items()},
            "total_count": int(sum(int(count) for count in color_counts.values())),
        },
        extra=extra,
    )


def _visible_color_by_index(cells: Sequence[Mapping[str, Any]]) -> Dict[int, str]:
    return {
        int(cell["flat_index"]): str(cell["color_name"])
        for cell in cells
        if bool(cell.get("present", True))
    }


def _color_state_key(color_by_index: Mapping[int, str]) -> Tuple[Tuple[int, str], ...]:
    return tuple((int(index), str(color_by_index[int(index)])) for index in sorted(int(index) for index in color_by_index.keys()))


def _recolor_by_index(
    color_by_index: Mapping[int, str],
    *,
    source_color: str,
    destination_color: str,
    indices: Sequence[int] | None = None,
) -> Dict[int, str]:
    allowed = None if indices is None else {int(index) for index in indices}
    updated = {int(index): str(color) for index, color in color_by_index.items()}
    for index, color in list(updated.items()):
        if str(color) == str(source_color) and (allowed is None or int(index) in allowed):
            updated[int(index)] = str(destination_color)
    return updated


def _counts_from_color_by_index(
    color_by_index: Mapping[int, str],
    *,
    active_colors: Sequence[str],
) -> Dict[str, int]:
    counts = {str(color): 0 for color in active_colors}
    for color in color_by_index.values():
        counts[str(color)] = int(counts.get(str(color), 0)) + 1
    return counts


def _sample_recolor_option_color_maps(
    *,
    params: Mapping[str, Any],
    active_colors: Sequence[str],
    original_color_by_index: Mapping[int, str],
    final_color_by_index: Mapping[int, str],
    source_color: str,
    destination_color: str,
    answer_label: str,
    instance_seed: int,
    namespace: str,
) -> Dict[str, Dict[int, str]]:
    """Create four unique fixed-position candidate boards for the recolor MCQ."""

    labels = ("A", "B", "C", "D")
    active = tuple(str(color) for color in active_colors)
    active_set = set(active)
    final_key = _color_state_key(final_color_by_index)
    raw = params.get("option_color_by_index_by_label")
    if raw is not None:
        if not isinstance(raw, Mapping):
            raise ValueError("option_color_by_index_by_label must be a mapping")
        options: Dict[str, Dict[int, str]] = {}
        original_indices = {int(index) for index in original_color_by_index.keys()}
        for label in labels:
            raw_for_label = raw[str(label)]
            if not isinstance(raw_for_label, Mapping):
                raise ValueError("each option_color_by_index_by_label value must be a mapping")
            option_map = {int(index): str(color) for index, color in raw_for_label.items()}
            if set(option_map.keys()) != original_indices:
                raise ValueError("option color maps must use exactly the original visible cell indices")
            if not set(option_map.values()).issubset(active_set):
                raise ValueError("option color maps must use active colors only")
            options[str(label)] = option_map
        if _color_state_key(options[str(answer_label)]) != final_key:
            raise ValueError("answer_label option must match the fixed-position final recolor state")
        if len({_color_state_key(option_map) for option_map in options.values()}) != len(labels):
            raise ValueError("option fixed-position color states must be unique")
        return options

    rng = spawn_rng(int(instance_seed), f"{namespace}.distractor_color_maps")
    source_indices = sorted(
        int(index)
        for index, color in original_color_by_index.items()
        if str(color) == str(source_color)
    )
    destination_indices = sorted(
        int(index)
        for index, color in original_color_by_index.items()
        if str(color) == str(destination_color)
    )
    candidates: list[Dict[int, str]] = []
    seen = {final_key}

    def add_candidate(candidate: Mapping[int, str] | None) -> None:
        if candidate is None:
            return
        normalized = {int(index): str(color) for index, color in candidate.items()}
        if set(normalized.keys()) != set(int(index) for index in original_color_by_index.keys()):
            return
        if not set(normalized.values()).issubset(active_set):
            return
        key = _color_state_key(normalized)
        if key in seen:
            return
        seen.add(key)
        candidates.append(normalized)

    add_candidate(original_color_by_index)
    if len(source_indices) > 1:
        keep_source = int(source_indices[int(rng.randrange(len(source_indices)))])
        recolor_indices = [int(index) for index in source_indices if int(index) != keep_source]
        add_candidate(
            _recolor_by_index(
                original_color_by_index,
                source_color=str(source_color),
                destination_color=str(destination_color),
                indices=recolor_indices,
            )
        )
    for color in active:
        if str(color) not in {str(source_color), str(destination_color)}:
            add_candidate(
                _recolor_by_index(
                    original_color_by_index,
                    source_color=str(source_color),
                    destination_color=str(color),
                )
            )
    if destination_indices:
        changed = dict(final_color_by_index)
        index = int(destination_indices[int(rng.randrange(len(destination_indices)))])
        changed[index] = str(source_color)
        add_candidate(changed)

    visible_indices = sorted(int(index) for index in original_color_by_index.keys())
    for _attempt in range(300):
        candidate = dict(final_color_by_index)
        index = int(visible_indices[int(rng.randrange(len(visible_indices)))])
        choices = [str(color) for color in active if str(color) != str(candidate[index])]
        candidate[index] = str(choices[int(rng.randrange(len(choices)))])
        add_candidate(candidate)
        if len(candidates) >= 3:
            break
    if len(candidates) < 3:
        raise ValueError("could not build three unique fixed-position recolor distractor options")

    rng.shuffle(candidates)
    options: Dict[str, Dict[int, str]] = {}
    distractor_iter = iter(candidates[:3])
    for label in labels:
        if str(label) == str(answer_label):
            options[str(label)] = {int(index): str(color) for index, color in final_color_by_index.items()}
        else:
            options[str(label)] = dict(next(distractor_iter))
    return options


def _fixed_position_dataset_from_color_map(
    *,
    scene_variant: str,
    element_type: str,
    template_cells: Sequence[Mapping[str, Any]],
    color_by_index: Mapping[int, str],
    active_colors: Sequence[str],
    rows: int,
    cols: int,
    layout_style: str,
    option_label: str | None = None,
) -> Dict[str, Any]:
    """Materialize one board by recoloring the original visible cells in place."""

    cells: list[Dict[str, Any]] = []
    for cell in template_cells:
        updated = dict(cell)
        if bool(updated.get("present", True)):
            index = int(updated["flat_index"])
            color = str(color_by_index[index])
            updated["color_name"] = str(color)
            updated["fill_rgb"] = list(SEMANTIC_COLOR_RGB[str(color)])
            updated["semantic_color"] = True
            updated["count_role"] = "distractor"
        cells.append(updated)
    color_counts = _counts_from_color_by_index(color_by_index, active_colors=active_colors)
    extra: Dict[str, Any] = {
        "color_counts": dict(color_counts),
        "color_by_flat_index": {str(index): str(color) for index, color in sorted(color_by_index.items())},
        "active_color_names": [str(color) for color in active_colors],
    }
    if option_label is not None:
        extra["option_label"] = str(option_label)
    return base_surface_data(
        scene_variant=str(scene_variant),
        element_type=str(element_type),
        answer_value=0,
        target_element_ids=[],
        surface_cells=cells,
        rows=int(rows),
        cols=int(cols),
        layout_style=str(layout_style),
        solver_trace={
            "count_predicate": "fixed_position_color_state_by_visible_surface_elements",
            "color_counts": dict(color_counts),
            "color_by_flat_index": {str(index): str(color) for index, color in sorted(color_by_index.items())},
            "total_count": int(len(color_by_index)),
        },
        extra=extra,
    )


def build_recolor_board_match_surface_data(
    *,
    namespace: str,
    scene_variant: str,
    element_type: str,
    answer_label: str,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
) -> Dict[str, Any]:
    """Create original and candidate fixture boards for one recolor MCQ."""

    active_colors = _resolve_recolor_active_colors(
        params=params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        namespace=str(namespace),
    )
    source_color, destination_color = _resolve_recolor_rule(
        params=params,
        active_colors=active_colors,
        instance_seed=int(instance_seed),
        namespace=str(namespace),
    )
    initial_counts, initial_total_probabilities = _resolve_recolor_initial_counts(
        params=params,
        gen_defaults=gen_defaults,
        active_colors=active_colors,
        source_color=str(source_color),
        instance_seed=int(instance_seed),
        namespace=str(namespace),
    )
    total = int(sum(int(count) for count in initial_counts.values()))
    rows, cols = grid_for_total(total)
    total_slots = int(rows) * int(cols)
    rng = spawn_rng(int(instance_seed), f"{namespace}.cells")
    present_indices = sample_indices(rng, list(range(total_slots)), total)
    layout_style = "variable_grid" if str(scene_variant) in {"brick_wall", "paver_floor", "mailbox_bank"} else "uniform_grid"
    original_dataset = _colored_dataset_from_counts(
        scene_variant=str(scene_variant),
        element_type=str(element_type),
        rows=int(rows),
        cols=int(cols),
        present_indices=present_indices,
        color_counts=initial_counts,
        rng=spawn_rng(int(instance_seed), f"{namespace}.original_cells"),
        layout_style=str(layout_style),
    )
    original_color_by_index = _visible_color_by_index(original_dataset["surface_cells"])
    final_color_by_index = _recolor_by_index(
        original_color_by_index,
        source_color=str(source_color),
        destination_color=str(destination_color),
    )
    final_counts = _counts_from_color_by_index(final_color_by_index, active_colors=active_colors)
    option_color_by_index_by_label = _sample_recolor_option_color_maps(
        params=params,
        active_colors=active_colors,
        original_color_by_index=original_color_by_index,
        final_color_by_index=final_color_by_index,
        source_color=str(source_color),
        destination_color=str(destination_color),
        answer_label=str(answer_label),
        instance_seed=int(instance_seed),
        namespace=str(namespace),
    )
    option_counts_by_label = {
        str(label): _counts_from_color_by_index(color_map, active_colors=active_colors)
        for label, color_map in option_color_by_index_by_label.items()
    }
    option_datasets = {
        str(label): _fixed_position_dataset_from_color_map(
            scene_variant=str(scene_variant),
            element_type=str(element_type),
            template_cells=original_dataset["surface_cells"],
            color_by_index=color_map,
            active_colors=active_colors,
            rows=int(rows),
            cols=int(cols),
            layout_style=str(layout_style),
            option_label=str(label),
        )
        for label, color_map in option_color_by_index_by_label.items()
    }
    recolor_phrase = (
        f"every {semantic_color_label(str(source_color))} {ELEMENT_DISPLAY_NAME[str(element_type)]} "
        f"becomes {semantic_color_label(str(destination_color))}"
    )
    return {
        "original_dataset": dict(original_dataset),
        "option_datasets": {str(label): dict(dataset) for label, dataset in option_datasets.items()},
        "answer_label": str(answer_label),
        "active_color_names": list(active_colors),
        "source_color_name": str(source_color),
        "destination_color_name": str(destination_color),
        "source_color_label": semantic_color_label(str(source_color)),
        "destination_color_label": semantic_color_label(str(destination_color)),
        "recolor_phrase": str(recolor_phrase),
        "initial_color_counts": dict(initial_counts),
        "final_color_counts": dict(final_counts),
        "option_color_counts_by_label": {str(label): dict(counts) for label, counts in option_counts_by_label.items()},
        "original_color_by_flat_index": {str(index): str(color) for index, color in sorted(original_color_by_index.items())},
        "final_color_by_flat_index": {str(index): str(color) for index, color in sorted(final_color_by_index.items())},
        "option_color_by_flat_index_by_label": {
            str(label): {str(index): str(color) for index, color in sorted(color_map.items())}
            for label, color_map in option_color_by_index_by_label.items()
        },
        "initial_total_count_probabilities": dict(initial_total_probabilities),
        "layout_rows": int(rows),
        "layout_columns": int(cols),
        "layout_style": str(layout_style),
        "present_indices": [int(index) for index in present_indices],
    }


def build_scoped_color_surface_data(
    *,
    namespace: str,
    scene_variant: str,
    element_type: str,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """Create a row/column scoped color-count fixture."""

    target_count, target_probabilities = resolve_int_support(
        params=params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.target_count",
        min_key="target_count_min",
        max_key="target_count_max",
        default_min=1,
        default_max=6,
        explicit_keys=("target_count", "answer_count"),
        lower_bound=1,
        upper_bound=10,
    )
    rng = spawn_rng(int(instance_seed), f"{namespace}.cells")
    rows = int(params.get("layout_rows", 4 + int(rng.randrange(3))))
    cols = int(params.get("layout_columns", 4 + int(rng.randrange(3))))
    axis = str(params.get("scope_axis", ("row", "column")[int(rng.randrange(2))]))
    if axis not in {"row", "column"}:
        raise ValueError(f"unsupported scope_axis: {axis}")
    scope_len = cols if axis == "row" else rows
    if int(target_count) > int(scope_len):
        raise ValueError(f"target_count {target_count} exceeds {axis} length {scope_len}")
    scope_index = int(params.get("scope_index", int(rng.randrange(rows if axis == "row" else cols))))
    total_slots = int(rows) * int(cols)
    all_indices = list(range(total_slots))
    if axis == "row":
        scope_indices = [scope_index * cols + col for col in range(cols)]
        scope_phrase = f"row {scope_index + 1}"
    else:
        scope_indices = [row * cols + scope_index for row in range(rows)]
        scope_phrase = f"column {scope_index + 1}"
    target_indices = sample_indices(rng, scope_indices, int(target_count))
    target_color = resolve_color(
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.target_color",
        explicit_key="target_color_name",
    )
    other_colors = list(_readout_color_support(anchors=(target_color,)))
    outside_same_color_count = min(int(rng.randrange(1, 4)), max(0, total_slots - len(scope_indices)))
    outside_indices = [index for index in all_indices if index not in set(scope_indices)]
    outside_same_color = set(sample_indices(rng, outside_indices, outside_same_color_count))
    target_set = set(target_indices)
    color_by_index: Dict[int, str] = {}
    for index in all_indices:
        if int(index) in target_set or int(index) in outside_same_color:
            color_by_index[int(index)] = str(target_color)
        else:
            color_by_index[int(index)] = str(other_colors[int(rng.randrange(len(other_colors)))])
    layout_style = "variable_grid" if str(scene_variant) in {"brick_wall", "paver_floor"} else "uniform_grid"
    cells = layout_cells(
        scene_variant=str(scene_variant),
        element_type=str(element_type),
        rows=int(rows),
        cols=int(cols),
        present_indices=all_indices,
        target_indices=target_indices,
        rng=rng,
        layout_style=layout_style,
        color_by_index=color_by_index,
        semantic_color=True,
    )
    dataset = base_surface_data(
        scene_variant=str(scene_variant),
        element_type=str(element_type),
        answer_value=int(target_count),
        target_element_ids=target_ids_from_indices(cells, target_indices),
        surface_cells=cells,
        rows=int(rows),
        cols=int(cols),
        layout_style=layout_style,
        solver_trace={
            "count_predicate": "scope_match and color_name == target_color_name",
            "scope_axis": str(axis),
            "scope_index": int(scope_index),
            "scope_phrase": str(scope_phrase),
            "target_color_name": str(target_color),
            "target_count": int(target_count),
            "outside_same_color_distractor_count": int(len(outside_same_color)),
            "unique_integer_answer": True,
        },
        extra={
            "scope_axis": str(axis),
            "scope_index": int(scope_index),
            "scope_phrase": str(scope_phrase),
            "target_color_name": str(target_color),
        },
    )
    return dataset, dict(target_probabilities)


__all__ = [
    "base_surface_data",
    "build_color_frequency_option_surface_data",
    "build_color_operation_surface_data",
    "build_color_surface_data",
    "build_recolor_board_match_surface_data",
    "build_repeated_surface_data",
    "build_scoped_color_surface_data",
    "COLOR_CONFUSION_EXCLUSIONS",
    "COLOR_FREQUENCY_MAXIMUM_PROGRAM",
    "COLOR_FREQUENCY_OPTION_LABELS",
    "COLOR_FREQUENCY_ZERO_PROGRAM",
]
