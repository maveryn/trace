"""Sampling and color-rule helpers for color-gradient puzzle scenes."""

from __future__ import annotations

import colorsys
from typing import Any, List, Mapping, Sequence, Tuple

from trace_tasks.core.sampling import (
    support_probability_map,
    uniform_choice_with_probabilities,
)
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.shared.color_distance import color_distance
from trace_tasks.tasks.shared.config_defaults import group_default

from .defaults import sample_weighted_variant
from .state import (
    COMPLETION_LENGTH_VARIANTS,
    COMPLETION_OPTION_COUNT_VARIANTS,
    COMPLETION_RULE_VARIANTS,
    DEFAULTS,
    GRID_SIZE_VARIANTS,
    LABELS,
    RULE_VARIANTS,
    CellSpec,
    CompletionCellSpec,
    CompletionDataset,
    CompletionOptionSpec,
    ViolationDataset,
)


def clamp_unit(value: float) -> float:
    """Clamp a float into the unit interval."""

    return max(0.0, min(1.0, float(value)))


def hsl_to_rgb(
    hue_degrees: float,
    saturation: float,
    lightness: float,
) -> Tuple[int, int, int]:
    """Convert HSL-like values to RGB using Python's HLS convention."""

    red, green, blue = colorsys.hls_to_rgb(
        (float(hue_degrees) % 360.0) / 360.0,
        clamp_unit(float(lightness)),
        clamp_unit(float(saturation)),
    )
    return (
        max(0, min(255, int(round(red * 255.0)))),
        max(0, min(255, int(round(green * 255.0)))),
        max(0, min(255, int(round(blue * 255.0)))),
    )


def luminance(rgb: Sequence[int]) -> float:
    """Return approximate sRGB luminance for text contrast."""

    red, green, blue = (float(max(0, min(255, int(value)))) for value in rgb[:3])
    return ((0.2126 * red) + (0.7152 * green) + (0.0722 * blue)) / 255.0


def rule_hsl(
    *,
    row: int,
    col: int,
    rule_variant: str,
    rule_params: Mapping[str, Any],
) -> Tuple[float, float, float]:
    """Return expected HSL for one grid coordinate and rule variant."""

    hue_base = float(rule_params["hue_base"])
    hue_step = float(rule_params["hue_step"])
    lightness_base = float(rule_params["lightness_base"])
    lightness_step = float(rule_params["lightness_step"])
    saturation_base = float(rule_params["saturation_base"])
    saturation_step = float(rule_params["saturation_step"])

    if str(rule_variant) == "column_hue_row_lightness":
        hue = hue_base + (float(col) * hue_step)
        saturation = saturation_base
        lightness = lightness_base + (float(row) * lightness_step)
    elif str(rule_variant) == "row_hue_column_lightness":
        hue = hue_base + (float(row) * hue_step)
        saturation = saturation_base
        lightness = lightness_base + (float(col) * lightness_step)
    elif str(rule_variant) == "column_hue_row_saturation":
        hue = hue_base + (float(col) * hue_step)
        saturation = saturation_base + (float(row) * saturation_step)
        lightness = lightness_base
    else:
        raise ValueError(f"unsupported color-gradient rule_variant: {rule_variant}")
    return (float(hue % 360.0), clamp_unit(saturation), clamp_unit(lightness))


def sample_rule_params(
    rng,
    *,
    generation_defaults: Mapping[str, Any],
    rule_variant: str,
) -> dict[str, Any]:
    """Sample one smooth two-dimensional color-progression rule."""

    hue_step_min = float(
        group_default(generation_defaults, "hue_step_min", DEFAULTS.hue_step_min)
    )
    hue_step_max = float(
        group_default(generation_defaults, "hue_step_max", DEFAULTS.hue_step_max)
    )
    lightness_step_min = float(
        group_default(
            generation_defaults,
            "lightness_step_min",
            DEFAULTS.lightness_step_min,
        )
    )
    lightness_step_max = float(
        group_default(
            generation_defaults,
            "lightness_step_max",
            DEFAULTS.lightness_step_max,
        )
    )
    saturation_step_min = float(
        group_default(
            generation_defaults,
            "saturation_step_min",
            DEFAULTS.saturation_step_min,
        )
    )
    saturation_step_max = float(
        group_default(
            generation_defaults,
            "saturation_step_max",
            DEFAULTS.saturation_step_max,
        )
    )

    hue_direction = -1.0 if bool(rng.randint(0, 1)) else 1.0
    lightness_direction = -1.0 if bool(rng.randint(0, 1)) else 1.0
    saturation_direction = -1.0 if bool(rng.randint(0, 1)) else 1.0
    lightness_step = (
        float(rng.uniform(lightness_step_min, lightness_step_max)) * lightness_direction
    )
    saturation_step = (
        float(rng.uniform(saturation_step_min, saturation_step_max))
        * saturation_direction
    )

    lightness_base = (
        float(rng.uniform(0.36, 0.46))
        if lightness_direction > 0
        else float(rng.uniform(0.64, 0.74))
    )
    saturation_base = (
        float(rng.uniform(0.44, 0.54))
        if saturation_direction > 0
        else float(rng.uniform(0.76, 0.88))
    )
    if str(rule_variant) == "column_hue_row_saturation":
        lightness_base = float(rng.uniform(0.54, 0.64))
    else:
        saturation_base = float(rng.uniform(0.66, 0.84))

    return {
        "hue_base": round(float(rng.uniform(0.0, 360.0)), 6),
        "hue_step": round(
            float(rng.uniform(hue_step_min, hue_step_max) * hue_direction),
            6,
        ),
        "lightness_base": round(float(lightness_base), 6),
        "lightness_step": round(float(lightness_step), 6),
        "saturation_base": round(float(saturation_base), 6),
        "saturation_step": round(float(saturation_step), 6),
    }


def build_violation_dataset(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    generation_defaults: Mapping[str, Any],
    namespace: str,
) -> ViolationDataset:
    """Construct the grid, inject one violation, and bind trace probabilities."""

    grid_size_variant, grid_size_probabilities = sample_weighted_variant(
        params=params,
        generation_defaults=generation_defaults,
        instance_seed=int(instance_seed),
        support=GRID_SIZE_VARIANTS,
        explicit_key="grid_size_variant",
        weights_key="grid_size_variant_weights",
        namespace=f"{namespace}.grid_size_variant",
    )
    rows = int(str(grid_size_variant).split("x", maxsplit=1)[0])
    cols = int(str(grid_size_variant).split("x", maxsplit=1)[1])

    rule_variant, rule_probabilities = sample_weighted_variant(
        params=params,
        generation_defaults=generation_defaults,
        instance_seed=int(instance_seed),
        support=RULE_VARIANTS,
        explicit_key="rule_variant",
        weights_key="rule_variant_weights",
        namespace=f"{namespace}.rule_variant",
    )

    labels = tuple(LABELS[index] for index in range(int(rows * cols)))
    explicit_label = (
        str(params.get("answer_label", "") or params.get("violation_label", ""))
        .strip()
        .upper()
    )
    if explicit_label:
        if explicit_label not in labels:
            raise ValueError(
                f"answer_label {explicit_label!r} is outside visible labels"
            )
        violation_index = int(labels.index(explicit_label))
        label_probabilities = support_probability_map(
            labels,
            selected=explicit_label,
        )
    else:
        label_rng = spawn_rng(int(instance_seed), f"{namespace}.answer_label")
        answer_label, label_probabilities = uniform_choice_with_probabilities(
            label_rng,
            labels,
        )
        violation_index = int(labels.index(str(answer_label)))

    rng = spawn_rng(int(instance_seed), f"{namespace}.dataset")
    rule_params = sample_rule_params(
        rng,
        generation_defaults=generation_defaults,
        rule_variant=str(rule_variant),
    )
    expected_hsl: List[Tuple[float, float, float]] = []
    expected_rgb: List[Tuple[int, int, int]] = []
    for row in range(int(rows)):
        for col in range(int(cols)):
            hsl = rule_hsl(
                row=int(row),
                col=int(col),
                rule_variant=str(rule_variant),
                rule_params=rule_params,
            )
            expected_hsl.append(hsl)
            expected_rgb.append(hsl_to_rgb(*hsl))

    violation_row = int(violation_index // cols)
    violation_col = int(violation_index % cols)
    candidate_indices = [
        int(index)
        for index in range(int(rows * cols))
        if int(index) != int(violation_index)
        and abs(int(index // cols) - violation_row)
        + abs(int(index % cols) - violation_col)
        >= 2
    ]
    if not candidate_indices:
        candidate_indices = [
            int(index)
            for index in range(int(rows * cols))
            if int(index) != int(violation_index)
        ]
    borrowed_index = int(rng.choice(candidate_indices))

    observed_hsl = list(expected_hsl)
    observed_rgb = list(expected_rgb)
    observed_hsl[int(violation_index)] = tuple(expected_hsl[int(borrowed_index)])
    observed_rgb[int(violation_index)] = tuple(expected_rgb[int(borrowed_index)])

    cells: List[CellSpec] = []
    for index, label in enumerate(labels):
        row = int(index // cols)
        col = int(index % cols)
        cells.append(
            CellSpec(
                cell_id=f"cell_{label}",
                label=str(label),
                row=int(row),
                col=int(col),
                expected_hsl=tuple(float(value) for value in expected_hsl[index]),
                observed_hsl=tuple(float(value) for value in observed_hsl[index]),
                expected_rgb=tuple(int(value) for value in expected_rgb[index]),
                observed_rgb=tuple(int(value) for value in observed_rgb[index]),
                is_violation=bool(index == int(violation_index)),
            )
        )

    return ViolationDataset(
        rows=int(rows),
        cols=int(cols),
        grid_size_variant=str(grid_size_variant),
        grid_size_variant_probabilities=dict(grid_size_probabilities),
        rule_variant=str(rule_variant),
        rule_variant_probabilities=dict(rule_probabilities),
        rule_params=dict(rule_params),
        cells=tuple(cells),
        answer_label=str(labels[int(violation_index)]),
        answer_label_probabilities=dict(label_probabilities),
        violation_cell_id=f"cell_{labels[int(violation_index)]}",
        violation_index=int(violation_index),
        borrowed_from_label=str(labels[int(borrowed_index)]),
    )


def _parse_completion_length_variant(value: str) -> int:
    """Return sequence length from a completion length variant key."""

    if str(value) not in COMPLETION_LENGTH_VARIANTS:
        raise ValueError(f"unsupported sequence_length_variant: {value}")
    return int(str(value).split("_", maxsplit=1)[0])


def _parse_completion_option_count_variant(value: str) -> int:
    """Return option count from a completion option-count variant key."""

    if str(value) not in COMPLETION_OPTION_COUNT_VARIANTS:
        raise ValueError(f"unsupported option_count_variant: {value}")
    return int(str(value).split("_", maxsplit=1)[0])


def completion_rule_hsl(
    *,
    index: int,
    rule_variant: str,
    rule_params: Mapping[str, Any],
) -> Tuple[float, float, float]:
    """Return expected HSL for one position in a linear gradient."""

    hue_base = float(rule_params["hue_base"])
    hue_step = float(rule_params["hue_step"])
    lightness_base = float(rule_params["lightness_base"])
    lightness_step = float(rule_params["lightness_step"])
    saturation_base = float(rule_params["saturation_base"])

    if str(rule_variant) == "hue_gradient":
        hue = hue_base + (float(index) * hue_step)
        lightness = lightness_base
    elif str(rule_variant) == "lightness_gradient":
        hue = hue_base
        lightness = lightness_base + (float(index) * lightness_step)
    elif str(rule_variant) == "hue_lightness_gradient":
        hue = hue_base + (float(index) * hue_step)
        lightness = lightness_base + (float(index) * lightness_step)
    else:
        raise ValueError(f"unsupported completion rule_variant: {rule_variant}")
    return (
        float(hue % 360.0),
        clamp_unit(float(saturation_base)),
        clamp_unit(lightness),
    )


def sample_completion_rule_params(
    rng,
    *,
    sequence_length: int,
    generation_defaults: Mapping[str, Any],
    rule_variant: str,
) -> dict[str, Any]:
    """Sample one linearly progressing color rule without hue wraparound."""

    hue_step_min = float(
        group_default(
            generation_defaults,
            "completion_hue_step_min",
            group_default(generation_defaults, "hue_step_min", 34.0),
        )
    )
    hue_step_max = float(
        group_default(
            generation_defaults,
            "completion_hue_step_max",
            min(58.0, max(34.0, hue_step_min + 20.0)),
        )
    )
    lightness_step_min = float(
        group_default(generation_defaults, "completion_lightness_step_min", 0.060)
    )
    lightness_step_max = float(
        group_default(generation_defaults, "completion_lightness_step_max", 0.088)
    )
    hue_step_min = max(20.0, hue_step_min)
    hue_step_max = max(hue_step_min, hue_step_max)

    hue_direction = -1.0 if bool(rng.randint(0, 1)) else 1.0
    lightness_direction = -1.0 if bool(rng.randint(0, 1)) else 1.0
    uses_hue = str(rule_variant) in {"hue_gradient", "hue_lightness_gradient"}
    uses_lightness = str(rule_variant) in {
        "lightness_gradient",
        "hue_lightness_gradient",
    }

    hue_step = 0.0
    if uses_hue:
        max_no_wrap = max(float(hue_step_min), 320.0 / max(1.0, sequence_length - 1))
        hue_step = float(rng.uniform(hue_step_min, min(hue_step_max, max_no_wrap)))
        hue_span = abs(hue_step) * max(1.0, float(sequence_length - 1))
        if hue_direction > 0:
            hue_base = float(rng.uniform(6.0, max(6.0, 354.0 - hue_span)))
        else:
            hue_base = float(rng.uniform(min(354.0, 6.0 + hue_span), 354.0))
        hue_step *= hue_direction
    else:
        hue_base = float(rng.uniform(0.0, 360.0))

    lightness_step = 0.0
    if uses_lightness:
        lightness_step = (
            float(rng.uniform(lightness_step_min, lightness_step_max))
            * lightness_direction
        )
        lightness_span = abs(lightness_step) * max(1.0, float(sequence_length - 1))
        if lightness_direction > 0:
            lightness_base = float(rng.uniform(0.34, max(0.34, 0.76 - lightness_span)))
        else:
            lightness_base = float(rng.uniform(min(0.76, 0.34 + lightness_span), 0.76))
    else:
        lightness_base = float(rng.uniform(0.55, 0.66))

    return {
        "hue_base": round(float(hue_base), 6),
        "hue_step": round(float(hue_step), 6),
        "lightness_base": round(float(lightness_base), 6),
        "lightness_step": round(float(lightness_step), 6),
        "saturation_base": round(float(rng.uniform(0.68, 0.84)), 6),
    }


def _append_distinct_rgb(
    candidates: List[Tuple[int, int, int]],
    rgb: Tuple[int, int, int],
    *,
    min_lab_distance: float,
) -> bool:
    """Append one RGB candidate when it is visually distinct enough."""

    normalized = tuple(int(value) for value in rgb)
    if any(
        color_distance(normalized, existing, distance_space="lab") < min_lab_distance
        for existing in candidates
    ):
        return False
    candidates.append(normalized)
    return True


def _build_completion_distractor_rgbs(
    *,
    rng,
    option_count: int,
    correct_rgb: Tuple[int, int, int],
    missing_index: int,
    rule_variant: str,
    rule_params: Mapping[str, Any],
) -> Tuple[Tuple[int, int, int], ...]:
    """Build visually distinct but plausible color-option distractors."""

    candidate_rgbs: List[Tuple[int, int, int]] = [
        tuple(int(value) for value in correct_rgb)
    ]
    for offset in (-2, -1, 1, 2, -3, 3):
        if len(candidate_rgbs) >= int(option_count):
            break
        hsl = completion_rule_hsl(
            index=int(missing_index) + int(offset),
            rule_variant=str(rule_variant),
            rule_params=rule_params,
        )
        _append_distinct_rgb(candidate_rgbs, hsl_to_rgb(*hsl), min_lab_distance=18.0)

    correct_h, correct_s, correct_l = completion_rule_hsl(
        index=int(missing_index),
        rule_variant=str(rule_variant),
        rule_params=rule_params,
    )
    attempts = 0
    while len(candidate_rgbs) < int(option_count) and attempts < 200:
        attempts += 1
        hue_jitter = float(rng.choice([-1.0, 1.0])) * float(rng.uniform(30.0, 80.0))
        lightness_jitter = float(rng.choice([-1.0, 1.0])) * float(
            rng.uniform(0.06, 0.16)
        )
        saturation_jitter = float(rng.choice([-1.0, 1.0])) * float(
            rng.uniform(0.03, 0.10)
        )
        hsl = (
            float(correct_h + hue_jitter),
            clamp_unit(float(correct_s + saturation_jitter)),
            clamp_unit(float(correct_l + lightness_jitter)),
        )
        _append_distinct_rgb(candidate_rgbs, hsl_to_rgb(*hsl), min_lab_distance=18.0)

    fallback_index = 0
    while len(candidate_rgbs) < int(option_count):
        fallback_index += 1
        fallback_hsl = (
            float(correct_h + (45.0 * fallback_index)),
            clamp_unit(float(correct_s)),
            clamp_unit(float(correct_l)),
        )
        if not _append_distinct_rgb(
            candidate_rgbs,
            hsl_to_rgb(*fallback_hsl),
            min_lab_distance=8.0,
        ):
            candidate_rgbs.append(
                hsl_to_rgb(
                    float(correct_h + (31.0 * fallback_index)), correct_s, correct_l
                )
            )

    return tuple(
        tuple(int(channel) for channel in rgb)
        for rgb in candidate_rgbs[1 : int(option_count)]
    )


def build_completion_dataset(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    generation_defaults: Mapping[str, Any],
    namespace: str,
) -> CompletionDataset:
    """Construct one linear color-gradient completion dataset."""

    length_variant, length_probabilities = sample_weighted_variant(
        params=params,
        generation_defaults=generation_defaults,
        instance_seed=int(instance_seed),
        support=COMPLETION_LENGTH_VARIANTS,
        explicit_key="sequence_length_variant",
        weights_key="sequence_length_variant_weights",
        namespace=f"{namespace}.sequence_length_variant",
    )
    option_count_variant, option_count_probabilities = sample_weighted_variant(
        params=params,
        generation_defaults=generation_defaults,
        instance_seed=int(instance_seed),
        support=COMPLETION_OPTION_COUNT_VARIANTS,
        explicit_key="option_count_variant",
        weights_key="option_count_variant_weights",
        namespace=f"{namespace}.option_count_variant",
    )
    rule_variant, rule_probabilities = sample_weighted_variant(
        params=params,
        generation_defaults=generation_defaults,
        instance_seed=int(instance_seed),
        support=COMPLETION_RULE_VARIANTS,
        explicit_key="rule_variant",
        weights_key="rule_variant_weights",
        namespace=f"{namespace}.rule_variant",
    )
    sequence_length = _parse_completion_length_variant(str(length_variant))
    option_count = _parse_completion_option_count_variant(str(option_count_variant))
    option_labels = tuple(LABELS[index] for index in range(int(option_count)))

    explicit_label = (
        str(params.get("answer_label", "") or params.get("correct_option_label", ""))
        .strip()
        .upper()
    )
    if explicit_label:
        if explicit_label not in option_labels:
            raise ValueError(f"answer_label {explicit_label!r} is outside options")
        correct_option_index = int(option_labels.index(explicit_label))
        label_probabilities = support_probability_map(
            option_labels,
            selected=explicit_label,
        )
    else:
        label_rng = spawn_rng(int(instance_seed), f"{namespace}.answer_label")
        answer_label, label_probabilities = uniform_choice_with_probabilities(
            label_rng,
            option_labels,
        )
        correct_option_index = int(option_labels.index(str(answer_label)))

    interior_positions = tuple(range(1, max(1, int(sequence_length) - 1)))
    if "missing_index" in params and params.get("missing_index") is not None:
        missing_index = int(params.get("missing_index"))
        if missing_index not in interior_positions:
            raise ValueError("missing_index must be an interior position")
        missing_probabilities = support_probability_map(
            interior_positions,
            selected=missing_index,
            sort_keys=True,
        )
    else:
        missing_rng = spawn_rng(int(instance_seed), f"{namespace}.missing_index")
        selected_missing, missing_probabilities = uniform_choice_with_probabilities(
            missing_rng,
            interior_positions,
            sort_keys=True,
        )
        missing_index = int(selected_missing)

    rng = spawn_rng(int(instance_seed), f"{namespace}.dataset")
    rule_params = sample_completion_rule_params(
        rng,
        sequence_length=int(sequence_length),
        generation_defaults=generation_defaults,
        rule_variant=str(rule_variant),
    )
    cells: List[CompletionCellSpec] = []
    expected_rgbs: List[Tuple[int, int, int]] = []
    expected_hsls: List[Tuple[float, float, float]] = []
    for index in range(int(sequence_length)):
        hsl = completion_rule_hsl(
            index=int(index),
            rule_variant=str(rule_variant),
            rule_params=rule_params,
        )
        rgb = hsl_to_rgb(*hsl)
        expected_hsls.append(tuple(float(value) for value in hsl))
        expected_rgbs.append(tuple(int(value) for value in rgb))
        cells.append(
            CompletionCellSpec(
                cell_id=f"sequence_cell_{index}",
                index=int(index),
                expected_hsl=tuple(float(value) for value in hsl),
                expected_rgb=tuple(int(value) for value in rgb),
                is_missing=bool(index == int(missing_index)),
            )
        )

    correct_rgb = tuple(int(value) for value in expected_rgbs[int(missing_index)])
    distractor_rgbs = list(
        _build_completion_distractor_rgbs(
            rng=rng,
            option_count=int(option_count),
            correct_rgb=correct_rgb,
            missing_index=int(missing_index),
            rule_variant=str(rule_variant),
            rule_params=rule_params,
        )
    )
    options: List[CompletionOptionSpec] = []
    distractor_index = 0
    correct_option_id = ""
    for option_index, label in enumerate(option_labels):
        is_correct = bool(option_index == int(correct_option_index))
        if is_correct:
            rgb = correct_rgb
            hsl = tuple(float(value) for value in expected_hsls[int(missing_index)])
            correct_option_id = f"option_{label}"
        else:
            rgb = tuple(int(value) for value in distractor_rgbs[int(distractor_index)])
            distractor_index += 1
            hsl = (0.0, 0.0, 0.0)
        options.append(
            CompletionOptionSpec(
                option_id=f"option_{label}",
                label=str(label),
                hsl=tuple(float(value) for value in hsl),
                rgb=tuple(int(value) for value in rgb),
                is_correct=bool(is_correct),
            )
        )

    answer_label = str(option_labels[int(correct_option_index)])
    return CompletionDataset(
        sequence_length=int(sequence_length),
        sequence_length_variant=str(length_variant),
        sequence_length_variant_probabilities=dict(length_probabilities),
        option_count=int(option_count),
        option_count_variant=str(option_count_variant),
        option_count_variant_probabilities=dict(option_count_probabilities),
        rule_variant=str(rule_variant),
        rule_variant_probabilities=dict(rule_probabilities),
        rule_params=dict(rule_params),
        cells=tuple(cells),
        options=tuple(options),
        missing_index=int(missing_index),
        missing_index_probabilities=dict(missing_probabilities),
        missing_cell_id=f"sequence_cell_{missing_index}",
        answer_label=str(answer_label),
        answer_label_probabilities=dict(label_probabilities),
        correct_option_id=str(correct_option_id),
    )


__all__ = [
    "build_completion_dataset",
    "build_violation_dataset",
    "completion_rule_hsl",
    "hsl_to_rgb",
    "luminance",
]
