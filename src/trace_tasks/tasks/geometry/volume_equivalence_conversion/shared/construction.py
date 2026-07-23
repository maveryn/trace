"""Solid-volume construction formulas for volume-equivalence scenes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from trace_tasks.core.seed import spawn_rng

from .sampling import support_probabilities
from .state import OptionSpec, ResolvedProblem, SolidSpec

OPTION_LABELS: tuple[str, ...] = ("A", "B", "C", "D", "E", "F")
CYLINDER_OPTION_BASE_AREAS: tuple[int, ...] = tuple(range(4, 25))
CONE_OPTION_BASE_AREAS: tuple[int, ...] = tuple(range(9, 43))
OPTION_HEIGHT_MIN = 2
OPTION_HEIGHT_MAX = 32


def _set_best_case(
    best: dict[int, tuple[tuple[int, ...], tuple[int, ...]]],
    *,
    answer: int,
    score: tuple[int, ...],
    case: tuple[int, ...],
) -> None:
    if int(answer) not in best or tuple(score) < best[int(answer)][0]:
        best[int(answer)] = (tuple(score), tuple(case))


def _spread_answer_cases(
    best: Mapping[int, tuple[tuple[int, ...], tuple[int, ...]]],
    *,
    answer_min: int,
    answer_max: int,
    target_count: int,
) -> tuple[tuple[int, ...], ...]:
    answer_keys = [
        answer for answer in sorted(best) if answer_min <= int(answer) <= answer_max
    ]
    if len(answer_keys) < int(target_count):
        raise ValueError(
            f"not enough generated volume-equivalence cases in [{answer_min}, {answer_max}]: "
            f"{len(answer_keys)} < {target_count}"
        )
    selected: list[int] = []
    used: set[int] = set()
    for index in range(int(target_count)):
        target = float(answer_min) + (
            float(answer_max - answer_min) * float(index)
        ) / float(max(1, target_count - 1))
        answer = min(
            (candidate for candidate in answer_keys if candidate not in used),
            key=lambda value: (abs(value - target), value),
        )
        used.add(int(answer))
        selected.append(int(answer))
    return tuple(tuple(best[answer][1]) for answer in sorted(selected))


def _generate_cuboid_to_cylinder_length_cases() -> tuple[tuple[int, ...], ...]:
    best: dict[int, tuple[tuple[int, ...], tuple[int, ...]]] = {}
    for length in range(5, 45):
        for width in range(4, 37):
            for height in range(3, 31):
                source_volume = int(length) * int(width) * int(height)
                for target_base_area in range(8, 301):
                    if source_volume % int(target_base_area) != 0:
                        continue
                    answer = source_volume // int(target_base_area)
                    if not (6 <= int(answer) <= 300):
                        continue
                    if int(answer) in (
                        int(length),
                        int(width),
                        int(height),
                        int(target_base_area),
                    ):
                        continue
                    _set_best_case(
                        best,
                        answer=int(answer),
                        score=(
                            abs(int(target_base_area) - 84),
                            abs(int(length) - int(width)),
                            abs(int(height) - 14),
                            length + width + height + target_base_area,
                        ),
                        case=(
                            int(length),
                            int(width),
                            int(height),
                            int(target_base_area),
                        ),
                    )
    return _spread_answer_cases(best, answer_min=121, answer_max=240, target_count=70)


def _generate_cylinder_to_cone_height_cases() -> tuple[tuple[int, ...], ...]:
    best: dict[int, tuple[tuple[int, ...], tuple[int, ...]]] = {}
    for source_base_area in range(8, 421):
        for source_height in range(4, 61):
            numerator = 3 * int(source_base_area) * int(source_height)
            for target_base_area in range(8, 361):
                if numerator % int(target_base_area) != 0:
                    continue
                answer = numerator // int(target_base_area)
                if not (8 <= int(answer) <= 500):
                    continue
                if int(answer) in (
                    int(source_base_area),
                    int(source_height),
                    int(target_base_area),
                ):
                    continue
                if int(target_base_area) in (
                    int(source_base_area),
                    3 * int(source_base_area),
                ):
                    continue
                _set_best_case(
                    best,
                    answer=int(answer),
                    score=(
                        abs(int(target_base_area) - 96),
                        abs(int(source_base_area) - 72),
                        abs(int(source_height) - 24),
                        source_base_area + source_height + target_base_area,
                    ),
                    case=(
                        int(source_base_area),
                        int(source_height),
                        int(target_base_area),
                    ),
                )
    return _spread_answer_cases(best, answer_min=241, answer_max=420, target_count=100)


def _generate_cone_to_cuboid_height_cases() -> tuple[tuple[int, ...], ...]:
    best: dict[int, tuple[tuple[int, ...], tuple[int, ...]]] = {}
    for source_base_area in range(12, 421):
        for source_height in range(6, 73):
            if (int(source_base_area) * int(source_height)) % 3 != 0:
                continue
            source_volume = (int(source_base_area) * int(source_height)) // 3
            for target_length in range(3, 37):
                if source_volume % int(target_length) != 0:
                    continue
                width_factor = source_volume // int(target_length)
                for target_width in range(3, 37):
                    if width_factor % int(target_width) != 0:
                        continue
                    answer = width_factor // int(target_width)
                    if not (2 <= int(answer) <= 200):
                        continue
                    if int(answer) in (
                        int(source_base_area),
                        int(source_height),
                        int(target_length),
                        int(target_width),
                    ):
                        continue
                    _set_best_case(
                        best,
                        answer=int(answer),
                        score=(
                            abs(int(source_base_area) - 120),
                            abs(int(source_height) - 30),
                            abs(int(target_length) - int(target_width)),
                            source_base_area
                            + source_height
                            + target_length
                            + target_width,
                        ),
                        case=(
                            int(source_base_area),
                            int(source_height),
                            int(target_length),
                            int(target_width),
                        ),
                    )
    return _spread_answer_cases(best, answer_min=2, answer_max=120, target_count=100)


CUBOID_TO_CYLINDER_LENGTH_CASES: tuple[tuple[int, ...], ...] = (
    _generate_cuboid_to_cylinder_length_cases()
)
CYLINDER_TO_CONE_HEIGHT_CASES: tuple[tuple[int, ...], ...] = (
    _generate_cylinder_to_cone_height_cases()
)
CONE_TO_CUBOID_HEIGHT_CASES: tuple[tuple[int, ...], ...] = (
    _generate_cone_to_cuboid_height_cases()
)
CONE_SOURCE_OPTION_CASES: tuple[tuple[int, int], ...] = (
    (42, 9),
    (66, 8),
    (84, 6),
    (48, 12),
    (54, 10),
    (60, 12),
    (72, 9),
)
CYLINDER_SOURCE_OPTION_CASES: tuple[tuple[int, int], ...] = (
    (24, 6),
    (26, 8),
    (27, 7),
    (28, 9),
    (30, 8),
    (32, 9),
    (35, 6),
)
CUBOID_SOURCE_OPTION_CASES: tuple[tuple[int, int, int], ...] = (
    (9, 8, 6),
    (10, 9, 4),
    (12, 7, 5),
    (13, 7, 4),
    (14, 8, 3),
    (11, 6, 5),
    (16, 5, 5),
    (13, 8, 4),
)


def solid_volume(spec: SolidSpec) -> int:
    if spec.shape == "cuboid":
        return int(spec.length) * int(spec.width) * int(spec.height)
    if spec.shape == "cylinder":
        return int(spec.base_area) * int(spec.height)
    if spec.shape == "cone":
        numerator = int(spec.base_area) * int(spec.height)
        if numerator % 3 != 0:
            raise ValueError("cone base_area * height must be divisible by 3")
        return numerator // 3
    raise ValueError(f"unsupported solid shape: {spec.shape}")


def _solid_with_base_height(shape: str, *, base_area: int, height: int) -> SolidSpec:
    if str(shape) == "cylinder":
        return SolidSpec("cylinder", base_area=int(base_area), height=int(height))
    if str(shape) == "cone":
        return SolidSpec("cone", base_area=int(base_area), height=int(height))
    raise ValueError(f"unsupported option shape: {shape}")


def _matching_option_solids(
    *,
    shape: str,
    source_volume: int,
    base_areas: Sequence[int],
    min_height: int = OPTION_HEIGHT_MIN,
    max_height: int = OPTION_HEIGHT_MAX,
) -> tuple[SolidSpec, ...]:
    matches: list[SolidSpec] = []
    for base_area in base_areas:
        for height in range(int(min_height), int(max_height) + 1):
            solid = _solid_with_base_height(
                shape, base_area=int(base_area), height=int(height)
            )
            try:
                volume = solid_volume(solid)
            except ValueError:
                continue
            if int(volume) == int(source_volume):
                matches.append(solid)
    if not matches:
        raise ValueError(
            f"no equal-volume {shape} options available for volume={source_volume}"
        )
    return tuple(matches)


def _select_correct_option_solid(
    *,
    shape: str,
    source_volume: int,
    base_areas: Sequence[int],
    instance_seed: int,
    namespace: str,
) -> SolidSpec:
    matches = _matching_option_solids(
        shape=str(shape),
        source_volume=int(source_volume),
        base_areas=base_areas,
    )
    rng = spawn_rng(int(instance_seed), str(namespace))
    return matches[int(rng.randrange(len(matches)))]


def _near_volume_distractors(
    *,
    shape: str,
    source_volume: int,
    correct: SolidSpec,
    base_areas: Sequence[int],
    count: int,
    min_height: int = OPTION_HEIGHT_MIN,
    max_height: int = OPTION_HEIGHT_MAX,
) -> tuple[SolidSpec, ...]:
    candidates: list[tuple[tuple[int, ...], SolidSpec]] = []
    for base_area in base_areas:
        for height in range(int(min_height), int(max_height) + 1):
            solid = _solid_with_base_height(
                shape, base_area=int(base_area), height=int(height)
            )
            try:
                volume = solid_volume(solid)
            except ValueError:
                continue
            if solid == correct or int(volume) == int(source_volume):
                continue
            candidates.append(
                (
                    (
                        abs(int(volume) - int(source_volume)),
                        abs(int(base_area) - int(correct.base_area)),
                        abs(int(height) - int(correct.height)),
                        int(base_area),
                        int(height),
                    ),
                    solid,
                )
            )
    selected: list[SolidSpec] = []
    seen_volumes: set[int] = set()
    for _score, solid in sorted(candidates, key=lambda item: item[0]):
        volume = solid_volume(solid)
        if int(volume) in seen_volumes:
            continue
        seen_volumes.add(int(volume))
        selected.append(solid)
        if len(selected) >= int(count):
            return tuple(selected)
    raise ValueError(
        f"not enough near-volume {shape} distractors for volume={source_volume}: "
        f"{len(selected)} < {count}"
    )


def resolve_cuboid_to_cylinder_length(case: Sequence[int]) -> ResolvedProblem:
    length, width, height, target_base_area = [int(value) for value in case]
    source = SolidSpec("cuboid", height=height, length=length, width=width)
    answer = solid_volume(source) // int(target_base_area)
    if int(target_base_area) * int(answer) != solid_volume(source):
        raise ValueError("cuboid-to-cylinder case must yield integer target length")
    target = SolidSpec("cylinder", base_area=int(target_base_area), height=int(answer))
    return ResolvedProblem(
        source=source,
        target=target,
        answer=int(answer),
        answer_schema="integer",
        formula_family="volume_equivalence_missing_dimension",
        formula="target_height = source_cuboid_volume / cylinder_base_area",
        target_unknown_role="cylinder_height",
    )


def resolve_cylinder_to_cone_height(case: Sequence[int]) -> ResolvedProblem:
    source_base_area, source_height, target_base_area = [int(value) for value in case]
    source = SolidSpec(
        "cylinder", base_area=int(source_base_area), height=int(source_height)
    )
    answer = (3 * solid_volume(source)) // int(target_base_area)
    if int(target_base_area) * int(answer) != 3 * solid_volume(source):
        raise ValueError("cylinder-to-cone case must yield integer target height")
    target = SolidSpec("cone", base_area=int(target_base_area), height=int(answer))
    return ResolvedProblem(
        source=source,
        target=target,
        answer=int(answer),
        answer_schema="integer",
        formula_family="volume_equivalence_missing_dimension",
        formula="target_height = 3 * source_cylinder_volume / cone_base_area",
        target_unknown_role="cone_height",
    )


def resolve_cone_to_cuboid_height(case: Sequence[int]) -> ResolvedProblem:
    source_base_area, source_height, target_length, target_width = [
        int(value) for value in case
    ]
    source = SolidSpec(
        "cone", base_area=int(source_base_area), height=int(source_height)
    )
    target_base = int(target_length) * int(target_width)
    answer = solid_volume(source) // target_base
    if target_base * int(answer) != solid_volume(source):
        raise ValueError("cone-to-cuboid case must yield integer target height")
    target = SolidSpec(
        "cuboid", height=int(answer), length=int(target_length), width=int(target_width)
    )
    return ResolvedProblem(
        source=source,
        target=target,
        answer=int(answer),
        answer_schema="integer",
        formula_family="volume_equivalence_missing_dimension",
        formula="target_height = source_cone_volume / (cuboid_length * cuboid_width)",
        target_unknown_role="cuboid_height",
    )


def bind_case_metadata(
    problem: ResolvedProblem,
    *,
    case_probabilities: Mapping[str, float],
    answer_support: Sequence[int | str],
) -> ResolvedProblem:
    return ResolvedProblem(
        source=problem.source,
        target=problem.target,
        answer=problem.answer,
        answer_schema=problem.answer_schema,
        formula_family=problem.formula_family,
        formula=problem.formula,
        target_unknown_role=problem.target_unknown_role,
        option_specs=tuple(problem.option_specs),
        selected_option_label=str(problem.selected_option_label),
        case_probabilities=dict(case_probabilities),
        answer_support_probabilities=support_probabilities(answer_support),
        option_count_probabilities=dict(problem.option_count_probabilities),
    )


def _rotated_option_specs(
    *,
    source: SolidSpec,
    correct: SolidSpec,
    distractors: Sequence[SolidSpec],
    option_count: int,
    instance_seed: int,
    params: Mapping[str, Any],
    shuffle_namespace: str,
    label_namespace: str,
) -> tuple[tuple[OptionSpec, ...], str]:
    source_volume = solid_volume(source)
    candidates = [correct, *list(distractors)]
    rng = spawn_rng(int(instance_seed), str(shuffle_namespace))
    rng.shuffle(candidates[1:])
    candidates = [candidates[0], *candidates[1 : max(1, int(option_count))]]
    rng = spawn_rng(int(instance_seed), str(label_namespace))
    offset = int(rng.randrange(len(candidates)))
    rotated = candidates[-offset:] + candidates[:-offset] if offset else candidates
    option_specs = tuple(
        OptionSpec(label=str(label), solid=solid, volume=solid_volume(solid))
        for label, solid in zip(OPTION_LABELS[: int(option_count)], rotated)
    )
    selected_label = next(
        option.label
        for option in option_specs
        if option.volume == source_volume and option.solid == correct
    )
    return option_specs, str(selected_label)


def _option_problem(
    *,
    source: SolidSpec,
    correct: SolidSpec,
    distractors: Sequence[SolidSpec],
    option_count: int,
    instance_seed: int,
    params: Mapping[str, Any],
    shuffle_namespace: str,
    label_namespace: str,
) -> ResolvedProblem:
    option_specs, selected_label = _rotated_option_specs(
        source=source,
        correct=correct,
        distractors=distractors,
        option_count=int(option_count),
        instance_seed=int(instance_seed),
        params=params,
        shuffle_namespace=str(shuffle_namespace),
        label_namespace=str(label_namespace),
    )
    selected = next(
        option.solid for option in option_specs if option.label == selected_label
    )
    source_volume = solid_volume(source)
    matching_options = [
        option for option in option_specs if int(option.volume) == int(source_volume)
    ]
    if len(matching_options) != 1 or matching_options[0].solid != correct:
        raise ValueError(
            "equal-volume option task must expose exactly one matching visual option"
        )
    labels = tuple(option.label for option in option_specs)
    return ResolvedProblem(
        source=source,
        target=selected,
        answer=str(selected_label),
        answer_schema="option_letter",
        formula_family="volume_equivalence_option_match",
        formula="select option whose solid volume equals the source solid volume",
        target_unknown_role="equal_volume_option",
        option_specs=tuple(option_specs),
        selected_option_label=str(selected_label),
        answer_support_probabilities=support_probabilities(labels),
    )


def resolve_cone_matching_cylinder_option(
    case: Sequence[int],
    *,
    option_count: int,
    instance_seed: int,
    params: Mapping[str, Any],
    shuffle_namespace: str,
    label_namespace: str,
) -> ResolvedProblem:
    source = SolidSpec("cone", base_area=int(case[0]), height=int(case[1]))
    source_volume = solid_volume(source)
    correct = _select_correct_option_solid(
        shape="cylinder",
        source_volume=int(source_volume),
        base_areas=CYLINDER_OPTION_BASE_AREAS,
        instance_seed=int(instance_seed),
        namespace=f"{shuffle_namespace}.correct_option",
    )
    distractors = _near_volume_distractors(
        shape="cylinder",
        source_volume=int(source_volume),
        correct=correct,
        base_areas=CYLINDER_OPTION_BASE_AREAS,
        count=5,
    )
    return _option_problem(
        source=source,
        correct=correct,
        distractors=distractors,
        option_count=int(option_count),
        instance_seed=int(instance_seed),
        params=params,
        shuffle_namespace=str(shuffle_namespace),
        label_namespace=str(label_namespace),
    )


def resolve_cylinder_matching_cone_option(
    case: Sequence[int],
    *,
    option_count: int,
    instance_seed: int,
    params: Mapping[str, Any],
    shuffle_namespace: str,
    label_namespace: str,
) -> ResolvedProblem:
    source = SolidSpec("cylinder", base_area=int(case[0]), height=int(case[1]))
    source_volume = solid_volume(source)
    correct = _select_correct_option_solid(
        shape="cone",
        source_volume=int(source_volume),
        base_areas=CONE_OPTION_BASE_AREAS,
        instance_seed=int(instance_seed),
        namespace=f"{shuffle_namespace}.correct_option",
    )
    distractors = _near_volume_distractors(
        shape="cone",
        source_volume=int(source_volume),
        correct=correct,
        base_areas=CONE_OPTION_BASE_AREAS,
        count=5,
        min_height=3,
    )
    return _option_problem(
        source=source,
        correct=correct,
        distractors=distractors,
        option_count=int(option_count),
        instance_seed=int(instance_seed),
        params=params,
        shuffle_namespace=str(shuffle_namespace),
        label_namespace=str(label_namespace),
    )


def resolve_cuboid_matching_cylinder_option(
    case: Sequence[int],
    *,
    option_count: int,
    instance_seed: int,
    params: Mapping[str, Any],
    shuffle_namespace: str,
    label_namespace: str,
) -> ResolvedProblem:
    source = SolidSpec(
        "cuboid", height=int(case[2]), length=int(case[0]), width=int(case[1])
    )
    source_volume = solid_volume(source)
    correct = _select_correct_option_solid(
        shape="cylinder",
        source_volume=int(source_volume),
        base_areas=CYLINDER_OPTION_BASE_AREAS,
        instance_seed=int(instance_seed),
        namespace=f"{shuffle_namespace}.correct_option",
    )
    distractors = _near_volume_distractors(
        shape="cylinder",
        source_volume=int(source_volume),
        correct=correct,
        base_areas=CYLINDER_OPTION_BASE_AREAS,
        count=5,
    )
    return _option_problem(
        source=source,
        correct=correct,
        distractors=distractors,
        option_count=int(option_count),
        instance_seed=int(instance_seed),
        params=params,
        shuffle_namespace=str(shuffle_namespace),
        label_namespace=str(label_namespace),
    )


def bind_option_metadata(
    problem: ResolvedProblem,
    *,
    case_probabilities: Mapping[str, float],
    option_count_probabilities: Mapping[str, float],
) -> ResolvedProblem:
    return ResolvedProblem(
        source=problem.source,
        target=problem.target,
        answer=problem.answer,
        answer_schema=problem.answer_schema,
        formula_family=problem.formula_family,
        formula=problem.formula,
        target_unknown_role=problem.target_unknown_role,
        option_specs=tuple(problem.option_specs),
        selected_option_label=str(problem.selected_option_label),
        case_probabilities=dict(case_probabilities),
        answer_support_probabilities=dict(problem.answer_support_probabilities),
        option_count_probabilities=dict(option_count_probabilities),
    )


__all__ = [
    "CONE_SOURCE_OPTION_CASES",
    "CONE_TO_CUBOID_HEIGHT_CASES",
    "CUBOID_SOURCE_OPTION_CASES",
    "CUBOID_TO_CYLINDER_LENGTH_CASES",
    "CYLINDER_SOURCE_OPTION_CASES",
    "CYLINDER_TO_CONE_HEIGHT_CASES",
    "OPTION_LABELS",
    "bind_case_metadata",
    "bind_option_metadata",
    "resolve_cone_matching_cylinder_option",
    "resolve_cone_to_cuboid_height",
    "resolve_cuboid_matching_cylinder_option",
    "resolve_cuboid_to_cylinder_length",
    "resolve_cylinder_matching_cone_option",
    "resolve_cylinder_to_cone_height",
    "solid_volume",
]
