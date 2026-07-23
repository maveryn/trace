"""Sampling primitives for sequence-strip icon scenes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple

from .....core.sampling import uniform_choice
from .....core.seed import spawn_rng
from ...shared.icon_assets import resolve_icon_pool
from ...shared.icon_scene import IconInstanceSpec
from ...shared.icon_task_rendering import resolve_icon_cell_render_params, sample_icon_instance_noise
from ...shared.icon_style import sample_single_icon_tint
from ....shared.config_defaults import group_default
from ....shared.deterministic_sampling import uniform_probability_map

from .rendering import IconSequenceCellSpec, resolve_completion_canvas_size


@dataclass(frozen=True)
class SequenceIconAppearanceSample:
    """Scene-level icon, tint, cell, and canvas choices for one sequence row."""

    sequence_icon_id: str
    tint_rgb: Tuple[int, int, int]
    sampled_palette_rgb: Tuple[Tuple[int, int, int], ...]
    cell_box_width_px: int
    cell_box_height_px: int
    canvas_width: int
    canvas_height: int


@dataclass(frozen=True)
class SequenceCompletionDefaults:
    """Fallback defaults shared by sequence-completion tasks."""

    sequence_length: int = 4
    option_count: int = 4
    missing_index_min: int = 0
    missing_index_max: int = 3
    canvas_width: int = 760
    canvas_height: int = 440
    outer_margin_px: int = 24
    panel_padding_px: int = 20
    panel_corner_radius_px: int = 18
    scene_icon_size_min_px: int = 24
    scene_icon_size_max_px: int = 72
    cell_box_width_min_px: int = 128
    cell_box_width_max_px: int = 148
    cell_box_height_min_px: int = 132
    cell_box_height_max_px: int = 154
    row_gap_px: int = 20
    scene_max_overlap_fraction: float = 0.0
    scene_placement_max_attempts: int = 1
    scene_size_shrink_rounds: int = 0
    scene_size_shrink_factor: float = 1.0
    panel_title_font_size_px: int = 24
    pool_manifest: str = "all_icons.txt"
    palette_size_min: int = 1
    palette_size_max: int = 1
    color_channel_min: int = 24
    color_channel_max: int = 220
    min_color_distance: float = 40.0
    color_distance_space: str = "lab"
    background_color_rgb: Tuple[int, int, int] = (247, 248, 251)
    panel_fill_rgb: Tuple[int, int, int] = (255, 255, 255)
    panel_border_rgb: Tuple[int, int, int] = (205, 212, 224)
    header_text_rgb: Tuple[int, int, int] = (70, 78, 96)
    cell_padding_px: int = 10
    cell_icon_padding_px: int = 8
    cell_corner_radius_px: int = 12
    cell_border_rgb: Tuple[int, int, int] = (218, 223, 233)
    cell_label_font_size_px: int = 22
    cell_label_color_rgb: Tuple[int, int, int] = (52, 60, 77)
    missing_mark_font_size_px: int = 56
    missing_mark_color_rgb: Tuple[int, int, int] = (84, 96, 118)
    icon_noise_edit_types: Tuple[str, ...] = ()
    icon_noise_edit_count_range: Tuple[int, int] = (0, 0)
    icon_noise_value_ranges: Dict[str, Dict[str, Tuple[float, float]]] = field(default_factory=dict)


@dataclass(frozen=True)
class SequenceCompletionPlan:
    """Task-resolved content for one sequence-completion scene."""

    attribute_id: str
    sequence_rule: str
    sequence_icon_id: str
    full_sequence_values: Tuple[Any, ...]
    missing_index: int
    correct_option_label: str
    correct_option_value: Any
    option_values_by_label: Dict[str, Any]
    sequence_cells: Tuple[IconSequenceCellSpec, ...]
    option_cells: Tuple[IconSequenceCellSpec, ...]
    sampled_palette_rgb: Tuple[Tuple[int, int, int], ...]
    support_probabilities: Dict[str, Any]
    extra_trace: Dict[str, Any]


@dataclass(frozen=True)
class CyclicProgressionSample:
    """Resolved four-step cyclic progression over an integer support."""

    answer_value: int
    start_value: int
    step_value: int
    sequence_values: Tuple[int, ...]
    value_support: Tuple[int, ...]
    step_support: Tuple[int, ...]
    support_probabilities: Dict[str, Any]


def sample_sequence_icon_appearance(
    rng,
    *,
    pool_manifest: str,
    render_params: Mapping[str, Any],
    sequence_length: int,
    empty_pool_message: str,
) -> SequenceIconAppearanceSample:
    """Sample reusable visual choices for a single horizontal sequence row."""

    pool = list(resolve_icon_pool(str(pool_manifest)))
    if not pool:
        raise ValueError(str(empty_pool_message))
    sequence_icon_id = str(rng.choice(pool))
    tint_rgb, sampled_palette_rgb = sample_single_icon_tint(
        rng,
        channel_min=int(render_params["color_channel_min"]),
        channel_max=int(render_params["color_channel_max"]),
        anchor_colors=(
            tuple(int(v) for v in render_params["background_color_rgb"]),
            tuple(int(v) for v in render_params["panel_fill_rgb"]),
            tuple(int(v) for v in render_params["panel_border_rgb"]),
            tuple(int(v) for v in render_params["header_text_rgb"]),
        ),
        min_color_distance=float(render_params["min_color_distance"]),
        distance_space=str(render_params["color_distance_space"]),
    )
    cell_box_width_px = int(
        rng.randint(
            int(render_params["cell_box_width_min_px"]),
            int(render_params["cell_box_width_max_px"]),
        )
    )
    cell_box_height_px = int(
        rng.randint(
            int(render_params["cell_box_height_min_px"]),
            int(render_params["cell_box_height_max_px"]),
        )
    )
    canvas_width, canvas_height = resolve_completion_canvas_size(
        cell_count=int(sequence_length),
        cell_box_width_px=int(cell_box_width_px),
        cell_box_height_px=int(cell_box_height_px),
        row_gap_px=int(render_params.get("row_gap_px", 0)),
        render_params=render_params,
    )
    return SequenceIconAppearanceSample(
        sequence_icon_id=str(sequence_icon_id),
        tint_rgb=tuple(int(value) for value in tint_rgb),
        sampled_palette_rgb=tuple(tuple(int(channel) for channel in color) for color in sampled_palette_rgb),
        cell_box_width_px=int(cell_box_width_px),
        cell_box_height_px=int(cell_box_height_px),
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
    )


def resolve_rotation_candidates(
    *,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    fallback_candidates: Sequence[int],
    key: str = "rotation_candidates_degrees",
) -> Tuple[int, ...]:
    """Resolve supported icon rotations for a sequence-strip scene."""

    raw = params.get(str(key), generation_defaults.get(str(key), list(fallback_candidates)))
    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"{key} must be a sequence")
    rotations = tuple(int(value) % 360 for value in raw)
    if not rotations:
        raise ValueError(f"{key} must contain at least one rotation")
    return rotations


def int_support_from_bounds(
    *,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    min_key: str,
    max_key: str,
    fallback_min: int,
    fallback_max: int,
) -> Tuple[int, ...]:
    """Resolve an inclusive integer support from params/defaults."""

    lo = int(params.get(str(min_key), group_default(defaults, str(min_key), int(fallback_min))))
    hi = int(params.get(str(max_key), group_default(defaults, str(max_key), int(fallback_max))))
    if lo > hi:
        raise ValueError(f"{min_key} must be <= {max_key}")
    return tuple(range(lo, hi + 1))


def int_tuple_from_defaults(
    *,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    key: str,
    fallback: Sequence[int],
    modulo: int | None = None,
) -> Tuple[int, ...]:
    """Read a de-duplicated integer tuple from params/defaults."""

    raw = params.get(str(key), group_default(defaults, str(key), tuple(fallback)))
    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"{key} must be a sequence")
    if modulo is None:
        values = tuple(int(value) for value in raw)
    else:
        values = tuple(int(value) % int(modulo) for value in raw)
    if not values:
        raise ValueError(f"{key} must contain at least one value")
    return tuple(dict.fromkeys(values))


def resolve_cyclic_progression_sample(
    *,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    instance_seed: int,
    missing_index: int,
    value_key: str,
    step_key: str,
    answer_key: str,
    start_key: str,
    fallback_values: Sequence[int],
    fallback_steps: Sequence[int],
    selection_namespace: str,
    probability_value_key: str,
    probability_step_key: str,
    modulo: int = 360,
) -> CyclicProgressionSample:
    """Resolve a cyclic arithmetic progression with one hidden value."""

    value_support = int_tuple_from_defaults(
        params=params,
        defaults=defaults,
        key=value_key,
        fallback=fallback_values,
        modulo=int(modulo),
    )
    step_support = int_tuple_from_defaults(
        params=params,
        defaults=defaults,
        key=step_key,
        fallback=fallback_steps,
        modulo=int(modulo),
    )
    explicit_answer = params.get(str(answer_key))
    explicit_start = params.get(str(start_key))
    explicit_step = params.get(str(step_key).replace("candidates_", ""))
    support_set = set(int(value) for value in value_support)
    feasible: list[tuple[int, int, int, Tuple[int, ...]]] = []
    for start in value_support:
        if explicit_start is not None and int(start) != int(explicit_start) % int(modulo):
            continue
        for step in step_support:
            if explicit_step is not None and int(step) != int(explicit_step) % int(modulo):
                continue
            sequence = tuple(int((int(start) + (index * int(step))) % int(modulo)) for index in range(4))
            if len(set(sequence)) != 4 or any(int(value) not in support_set for value in sequence):
                continue
            answer = int(sequence[int(missing_index)])
            if explicit_answer is not None and int(answer) != int(explicit_answer) % int(modulo):
                continue
            feasible.append((int(answer), int(start), int(step), sequence))
    if not feasible:
        raise ValueError("no feasible cyclic progression for requested parameters")
    answer, start, step, sequence = uniform_choice(
        spawn_rng(int(instance_seed), str(selection_namespace)),
        tuple(feasible),
    )
    return CyclicProgressionSample(
        answer_value=int(answer),
        start_value=int(start),
        step_value=int(step),
        sequence_values=tuple(int(value) for value in sequence),
        value_support=tuple(int(value) for value in value_support),
        step_support=tuple(int(value) for value in step_support),
        support_probabilities={
            str(probability_value_key): uniform_probability_map(
                value_support,
                selected=int(answer) if explicit_answer is not None else None,
            ),
            str(probability_step_key): uniform_probability_map(
                step_support,
                selected=int(step) if explicit_step is not None else None,
            ),
        },
    )


def sequence_missing_index(*, params: Mapping[str, Any], generation_defaults: Mapping[str, Any], instance_seed: int) -> int:
    """Resolve the top-row question-mark index for a four-cell sequence."""

    lo = int(params.get("missing_index_min", group_default(generation_defaults, "missing_index_min", 0)))
    hi = int(params.get("missing_index_max", group_default(generation_defaults, "missing_index_max", 3)))
    if lo < 0 or hi > 3 or lo > hi:
        raise ValueError("missing index bounds must be within 0..3")
    if params.get("missing_index") is not None:
        value = int(params["missing_index"])
        if value < lo or value > hi:
            raise ValueError("explicit missing_index is outside configured support")
        return value
    support = tuple(range(lo, hi + 1))
    return int(
        uniform_choice(
            spawn_rng(int(instance_seed), "icons_sequence_strip_missing_index"),
            support,
        )
    )


def sample_sequence_icon_id(rng, *, params: Mapping[str, Any], generation_defaults: Mapping[str, Any], fallback: str) -> str:
    """Sample one icon id from a configured manifest."""

    pool_manifest = str(params.get("pool_manifest", group_default(generation_defaults, "pool_manifest", fallback)))
    pool = list(resolve_icon_pool(pool_manifest))
    if not pool:
        raise ValueError(f"sequence-strip icon pool is empty: {pool_manifest}")
    return str(rng.choice(pool))


def sample_sequence_tint(
    rng,
    *,
    render_params: Mapping[str, Any],
) -> Tuple[Tuple[int, int, int], Tuple[Tuple[int, int, int], ...]]:
    """Sample one readable icon tint and palette trace."""

    tint, palette = sample_single_icon_tint(
        rng,
        channel_min=int(render_params["color_channel_min"]),
        channel_max=int(render_params["color_channel_max"]),
        anchor_colors=(
            tuple(int(value) for value in render_params["background_color_rgb"]),
            tuple(int(value) for value in render_params["panel_fill_rgb"]),
            tuple(int(value) for value in render_params["panel_border_rgb"]),
            tuple(int(value) for value in render_params["header_text_rgb"]),
        ),
        min_color_distance=float(render_params["min_color_distance"]),
        distance_space=str(render_params["color_distance_space"]),
    )
    return tuple(int(value) for value in tint), tuple(tuple(int(channel) for channel in color) for color in palette)


def icon_instance(
    *,
    instance_seed: int,
    noise_namespace: str,
    icon_id: str,
    tint_rgb: Sequence[int],
    render_params: Mapping[str, Any],
    nominal_size_px: int | None = None,
    rotation_degrees: int = 0,
) -> IconInstanceSpec:
    """Create one icon instance with deterministic per-icon noise."""

    edits, noise_seed = sample_icon_instance_noise(
        instance_seed=int(instance_seed),
        namespace=str(noise_namespace),
        render_params=render_params,
    )
    return IconInstanceSpec(
        icon_id=str(icon_id),
        nominal_size_px=None if nominal_size_px is None else int(nominal_size_px),
        rotation_degrees=int(rotation_degrees),
        tint_rgb=tuple(int(value) for value in tint_rgb),
        noise_edits=tuple(edits),
        noise_seed=int(noise_seed),
    )


def single_icon_completion_cells(
    *,
    instance_seed: int,
    noise_stem: str,
    sequence_values: Sequence[int],
    option_values_by_label: Mapping[str, int],
    missing_index: int,
    icon_id: str,
    render_params: Mapping[str, Any],
    tint_for_value: Callable[[int], Tuple[int, int, int]],
    size_for_value: Callable[[int], int],
    rotation_for_value: Callable[[int], int] | None = None,
) -> Tuple[Tuple[IconSequenceCellSpec, ...], Tuple[IconSequenceCellSpec, ...]]:
    """Create one-icon sequence and option cells for scalar visual progressions."""

    def cell_for_value(value: int, *, namespace: str, label: str | None = None, missing: bool = False) -> IconSequenceCellSpec:
        if missing:
            return IconSequenceCellSpec(is_missing=True)
        return IconSequenceCellSpec(
            icon_instances=(
                icon_instance(
                    instance_seed=int(instance_seed),
                    noise_namespace=f"{noise_stem}:{namespace}",
                    icon_id=str(icon_id),
                    tint_rgb=tint_for_value(int(value)),
                    render_params=render_params,
                    nominal_size_px=int(size_for_value(int(value))),
                    rotation_degrees=int(rotation_for_value(int(value))) if rotation_for_value is not None else 0,
                ),
            ),
            cell_label_text=label,
        )

    sequence_cells = tuple(
        cell_for_value(
            int(value),
            namespace=f"sequence_{index}",
            missing=int(index) == int(missing_index),
        )
        for index, value in enumerate(sequence_values)
    )
    option_cells = tuple(
        cell_for_value(int(option_values_by_label[label]), namespace=f"option_{label}", label=str(label))
        for label in ("A", "B", "C", "D")
    )
    return sequence_cells, option_cells


def option_values(
    rng,
    *,
    correct_value: Any,
    distractor_values: Sequence[Any],
) -> Tuple[str, Dict[str, Any]]:
    """Shuffle one correct option among three distractors into fixed A-D positions."""

    distractors = list(distractor_values)
    if len(distractors) < 3:
        raise ValueError("sequence completion requires at least three distractor values")
    rng.shuffle(distractors)
    raw_options = [correct_value, *distractors[:3]]
    rng.shuffle(raw_options)
    labels = ("A", "B", "C", "D")
    mapping = {label: value for label, value in zip(labels, raw_options)}
    correct_label = next(label for label, value in mapping.items() if value == correct_value)
    return str(correct_label), mapping


def resolve_completion_render_params(
    *,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    fallback_defaults: SequenceCompletionDefaults,
    instance_seed: int,
) -> Dict[str, Any]:
    """Resolve geometry and style params for a two-row completion scene."""

    render_params = resolve_icon_cell_render_params(
        params=params,
        render_defaults=render_defaults,
        fallback_defaults=fallback_defaults,
        instance_seed=int(instance_seed),
    )
    width_rng = spawn_rng(int(instance_seed), "sequence_strip_cell_width")
    height_rng = spawn_rng(int(instance_seed), "sequence_strip_cell_height")
    render_params["cell_box_width_px"] = int(
        params.get(
            "cell_box_width_px",
            width_rng.randint(
                int(render_params["cell_box_width_min_px"]),
                int(render_params["cell_box_width_max_px"]),
            ),
        )
    )
    render_params["cell_box_height_px"] = int(
        params.get(
            "cell_box_height_px",
            height_rng.randint(
                int(render_params["cell_box_height_min_px"]),
                int(render_params["cell_box_height_max_px"]),
            ),
        )
    )
    render_params["row_gap_px"] = int(params.get("row_gap_px", group_default(render_defaults, "row_gap_px", fallback_defaults.row_gap_px)))
    canvas_width, canvas_height = resolve_completion_canvas_size(
        cell_count=4,
        cell_box_width_px=int(render_params["cell_box_width_px"]),
        cell_box_height_px=int(render_params["cell_box_height_px"]),
        row_gap_px=int(render_params["row_gap_px"]),
        render_params=render_params,
    )
    render_params["canvas_width"] = int(params.get("canvas_width", canvas_width))
    render_params["canvas_height"] = int(params.get("canvas_height", canvas_height))
    return render_params


__all__ = [
    "CyclicProgressionSample",
    "SequenceIconAppearanceSample",
    "SequenceCompletionDefaults",
    "SequenceCompletionPlan",
    "icon_instance",
    "int_support_from_bounds",
    "int_tuple_from_defaults",
    "option_values",
    "resolve_completion_render_params",
    "resolve_cyclic_progression_sample",
    "resolve_rotation_candidates",
    "sample_sequence_icon_id",
    "sample_sequence_icon_appearance",
    "sample_sequence_tint",
    "sequence_missing_index",
    "single_icon_completion_cells",
    "uniform_probability_map",
]
