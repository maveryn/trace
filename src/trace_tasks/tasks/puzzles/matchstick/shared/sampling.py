"""Sampling helpers for matchstick puzzle scenes."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.shared.config_defaults import (
    resolve_required_int_bounds,
)
from trace_tasks.tasks.shared.mcq import option_label_for_index

from .rules import (
    changed_digit_index,
    completed_lattice_squares,
    digit_segment_keys,
    digit_source_candidates_for_removal,
    equation_is_true,
    equation_text,
    lattice_edges,
    optimal_lattice_square_additions,
    remove_equation_stick,
    number_segment_keys,
    number_transition_allowed,
)
from .state import (
    EquationRepairDataset,
    NumberDataset,
    OPTION_LABELS,
    OptionSpec,
    SquareCompletionDataset,
)


def resolve_option_count(
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    *,
    rng,
    force_count: int | None = None,
) -> tuple[int, tuple[int, int]]:
    """Resolve the visual answer-option count for matchstick tasks."""

    if force_count is not None:
        count = max(4, min(len(OPTION_LABELS), int(force_count)))
        return int(count), (int(count), int(count))
    if "option_count" in params:
        count = max(4, min(len(OPTION_LABELS), int(params["option_count"])))
        return int(count), (int(count), int(count))
    low, high = resolve_required_int_bounds(
        params,
        generation_defaults,
        min_key="option_count_min",
        max_key="option_count_max",
        fallback_min=4,
        fallback_max=6,
        context="matchstick option count",
    )
    low = max(4, min(len(OPTION_LABELS), int(low)))
    high = max(int(low), min(len(OPTION_LABELS), int(high)))
    return int(rng.randint(int(low), int(high))), (int(low), int(high))


def build_number_dataset(
    *,
    stick_delta: int,
    scene_variant: str,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    namespace: str,
    instance_seed: int,
) -> NumberDataset:
    """Build one source number plus six candidate transformed numbers."""

    rng = spawn_rng(int(instance_seed), f"{namespace}.number_dataset")
    option_count, _option_range = resolve_option_count(
        params,
        generation_defaults,
        rng=rng,
        force_count=6,
    )
    answer_index = int(rng.randrange(int(option_count)))

    numbers = list(range(10, 100))
    rng.shuffle(numbers)
    for source_number in numbers:
        reachable = [
            target_number
            for target_number in range(10, 100)
            if int(target_number) != int(source_number)
            and number_transition_allowed(
                int(source_number),
                int(target_number),
                stick_delta=int(stick_delta),
            )
        ]
        if not reachable:
            continue
        answer_number = int(reachable[int(rng.randrange(len(reachable)))])
        distractors = [
            target_number
            for target_number in range(10, 100)
            if int(target_number) != int(source_number)
            and int(target_number) != int(answer_number)
            and not number_transition_allowed(
                int(source_number),
                int(target_number),
                stick_delta=int(stick_delta),
            )
        ]
        rng.shuffle(distractors)
        if len(distractors) < int(option_count) - 1:
            continue
        options = [int(value) for value in distractors[: int(option_count) - 1]]
        options.insert(int(answer_index), int(answer_number))
        option_specs = tuple(
            OptionSpec(
                label=str(option_label_for_index(int(index))),
                is_correct=bool(index == int(answer_index)),
                value=int(number),
                metric_value=None,
            )
            for index, number in enumerate(options)
        )
        source_keys = number_segment_keys(int(source_number))
        answer_keys = number_segment_keys(int(answer_number))
        return NumberDataset(
            scene_variant=str(scene_variant),
            source_number=int(source_number),
            answer_number=int(answer_number),
            answer_label=str(option_label_for_index(int(answer_index))),
            option_count=int(option_count),
            option_specs=tuple(option_specs),
            changed_digit_index=int(
                changed_digit_index(int(source_number), int(answer_number))
            ),
            removed_segment_keys=tuple(sorted(source_keys - answer_keys)),
            added_segment_keys=tuple(sorted(answer_keys - source_keys)),
        )
    raise RuntimeError("failed to sample matchstick number dataset")


def build_equation_repair_dataset(
    *,
    scene_variant: str,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    namespace: str,
    instance_seed: int,
) -> EquationRepairDataset:
    """Build one false equation with exactly one labeled removable repair stick."""

    rng = spawn_rng(int(instance_seed), f"{namespace}.equation_repair_dataset")
    option_count, _option_range = resolve_option_count(
        params,
        generation_defaults,
        rng=rng,
    )
    answer_index = int(rng.randrange(int(option_count)))
    true_equations = list(_true_single_digit_equations())
    rng.shuffle(true_equations)

    for operator, repaired_digits in true_equations:
        digit_indices = [0, 1, 2]
        rng.shuffle(digit_indices)
        for digit_index in digit_indices:
            candidates = list(
                digit_source_candidates_for_removal(
                    int(repaired_digits[int(digit_index)])
                )
            )
            rng.shuffle(candidates)
            for source_digit, repair_segment_key in candidates:
                source_digits = list(int(value) for value in repaired_digits)
                source_digits[int(digit_index)] = int(source_digit)
                source_tuple = tuple(int(value) for value in source_digits)
                if equation_is_true(source_tuple, str(operator)):
                    continue
                removal_outcomes = _digit_removal_outcomes(
                    source_tuple,
                    operator=str(operator),
                )
                true_repairs = [
                    outcome for outcome in removal_outcomes if bool(outcome["is_true"])
                ]
                if len(true_repairs) != 1:
                    continue
                repair_stick_id = str(true_repairs[0]["stick_id"])
                if repair_stick_id != f"digit{int(digit_index)}:{repair_segment_key}":
                    continue
                distractors = [
                    str(outcome["stick_id"])
                    for outcome in removal_outcomes
                    if str(outcome["stick_id"]) != repair_stick_id
                ]
                rng.shuffle(distractors)
                if len(distractors) < int(option_count) - 1:
                    continue
                option_sticks = distractors[: int(option_count) - 1]
                option_sticks.insert(int(answer_index), repair_stick_id)
                option_specs = tuple(
                    OptionSpec(
                        label=str(option_label_for_index(int(index))),
                        is_correct=bool(str(stick_id) == repair_stick_id),
                        value=str(stick_id),
                        metric_value=None,
                    )
                    for index, stick_id in enumerate(option_sticks)
                )
                return EquationRepairDataset(
                    scene_variant=str(scene_variant),
                    operator=str(operator),
                    source_digits=source_tuple,
                    repaired_digits=tuple(int(value) for value in repaired_digits),
                    answer_label=str(option_label_for_index(int(answer_index))),
                    option_count=int(option_count),
                    option_specs=tuple(option_specs),
                    repair_stick_id=repair_stick_id,
                    repair_digit_index=int(digit_index),
                    repair_segment_key=str(repair_segment_key),
                    all_removal_outcomes=tuple(removal_outcomes),
                )
    raise RuntimeError("failed to sample matchstick equation-repair dataset")


def build_square_completion_dataset(
    *,
    scene_variant: str,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    namespace: str,
    instance_seed: int,
) -> SquareCompletionDataset:
    """Build one lattice where adding K sticks has a unique final square set."""

    rng = spawn_rng(int(instance_seed), f"{namespace}.square_completion_dataset")
    row_min, row_max = resolve_required_int_bounds(
        params,
        generation_defaults,
        min_key="row_count_min",
        max_key="row_count_max",
        fallback_min=3,
        fallback_max=3,
        context="matchstick square-completion rows",
    )
    col_min, col_max = resolve_required_int_bounds(
        params,
        generation_defaults,
        min_key="col_count_min",
        max_key="col_count_max",
        fallback_min=3,
        fallback_max=3,
        context="matchstick square-completion columns",
    )
    add_min, add_max = resolve_required_int_bounds(
        params,
        generation_defaults,
        min_key="add_count_min",
        max_key="add_count_max",
        fallback_min=1,
        fallback_max=2,
        context="matchstick square-completion add count",
    )
    answer_min, answer_max = resolve_required_int_bounds(
        params,
        generation_defaults,
        min_key="answer_count_min",
        max_key="answer_count_max",
        fallback_min=1,
        fallback_max=6,
        context="matchstick square-completion answer count",
    )
    max_attempts = int(params.get("square_completion_sample_attempts", 1200))

    for _attempt_index in range(max(1, max_attempts)):
        rows = int(rng.randint(int(row_min), int(row_max)))
        cols = int(rng.randint(int(col_min), int(col_max)))
        add_count = int(rng.randint(int(add_min), int(add_max)))
        answer_upper = min(int(answer_max), rows * cols)
        target_answer = int(rng.randint(int(answer_min), int(answer_upper)))
        all_edges = tuple(lattice_edges(rows, cols))
        if len(all_edges) < add_count:
            continue

        present_probability = rng.uniform(0.48, 0.76)
        present_edges = frozenset(
            str(edge) for edge in all_edges if rng.random() < present_probability
        )
        missing_edges = tuple(sorted(frozenset(all_edges) - present_edges))
        if len(missing_edges) < int(add_count):
            continue

        initial_squares = completed_lattice_squares(
            present_edges,
            rows=int(rows),
            cols=int(cols),
        )
        optimal = optimal_lattice_square_additions(
            present_edges,
            rows=int(rows),
            cols=int(cols),
            add_count=int(add_count),
        )
        best_count = int(optimal["best_count"])
        best_square_sets = tuple(optimal["best_square_sets"])  # type: ignore[arg-type]
        best_added_sets = tuple(optimal["best_added_sets"])  # type: ignore[arg-type]
        if best_count != int(target_answer):
            continue
        if best_count <= len(initial_squares):
            continue
        if len(best_square_sets) != 1 or not best_added_sets:
            continue

        completed_square_ids = tuple(str(square) for square in best_square_sets[0])
        optimal_added_edges = tuple(str(edge) for edge in best_added_sets[0])
        return SquareCompletionDataset(
            scene_variant=str(scene_variant),
            rows=int(rows),
            cols=int(cols),
            add_count=int(add_count),
            answer_value=int(best_count),
            answer_label=str(best_count),
            option_count=0,
            present_edges=tuple(sorted(str(edge) for edge in present_edges)),
            missing_edges=tuple(sorted(str(edge) for edge in missing_edges)),
            initial_completed_square_ids=tuple(str(square) for square in initial_squares),
            completed_square_ids=completed_square_ids,
            optimal_added_edges=optimal_added_edges,
            optimal_added_edge_sets=tuple(
                tuple(str(edge) for edge in edge_set) for edge_set in best_added_sets
            ),
        )
    raise RuntimeError("failed to sample unique matchstick square-completion dataset")


def _true_single_digit_equations() -> tuple[tuple[str, tuple[int, int, int]], ...]:
    """Enumerate true one-digit addition and subtraction equations."""

    equations: list[tuple[str, tuple[int, int, int]]] = []
    for left in range(10):
        for right in range(10):
            plus_result = int(left + right)
            if plus_result <= 9:
                equations.append(("+", (int(left), int(right), int(plus_result))))
            minus_result = int(left - right)
            if 0 <= minus_result <= 9:
                equations.append(("-", (int(left), int(right), int(minus_result))))
    return tuple(equations)


def _digit_removal_outcomes(
    digits: tuple[int, int, int],
    *,
    operator: str,
) -> list[dict[str, object]]:
    """Return the result of removing every visible digit stick."""

    outcomes: list[dict[str, object]] = []
    for digit_index, digit in enumerate(digits):
        for segment_key in sorted(
            digit_segment_keys(int(digit), digit_index=int(digit_index))
        ):
            local_segment = str(segment_key).split(":", 1)[1]
            stick_id = f"digit{int(digit_index)}:{local_segment}"
            result_digits, is_true = remove_equation_stick(
                digits,
                operator=str(operator),
                stick_id=stick_id,
            )
            outcomes.append(
                {
                    "stick_id": stick_id,
                    "result_digits": None
                    if result_digits is None
                    else [int(value) for value in result_digits],
                    "result_equation": None
                    if result_digits is None
                    else equation_text(result_digits, str(operator)),
                    "is_valid_digit": result_digits is not None,
                    "is_true": bool(is_true),
                }
            )
    return outcomes


__all__ = [
    "build_equation_repair_dataset",
    "build_number_dataset",
    "build_square_completion_dataset",
    "resolve_option_count",
]
