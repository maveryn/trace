"""Sampling helpers for symbolic truth-table tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .....core.sampling import (
    sample_without_replacement,
    uniform_choice,
    uniform_choice_with_probabilities,
)
from ...shared.common import resolve_symbolic_axis_variant
from .rules import (
    all_expression_specs,
    distinct_truth_patterns,
    expression_by_id,
    expressions_by_true_count,
    truth_rows,
)
from .state import (
    EXPRESSION_OPTION_LABELS,
    OPTION_LABELS,
    SUPPORTED_TRUTH_TABLE_SCENE_VARIANTS,
    TruthExpressionSpec,
    TruthRowSpec,
)


@dataclass(frozen=True)
class TruthCountDataset:
    scene_variant: str
    scene_variant_probabilities: dict[str, float]
    rows: tuple[TruthRowSpec, ...]
    expression: TruthExpressionSpec
    answer_value: int
    target_answer_support: tuple[int, ...]
    true_cell_ids: tuple[str, ...]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class TruthPatternDataset:
    scene_variant: str
    scene_variant_probabilities: dict[str, float]
    rows: tuple[TruthRowSpec, ...]
    expression: TruthExpressionSpec
    answer_value: str
    target_answer_support: tuple[str, ...]
    options: tuple[tuple[str, str], ...]
    selected_option_id: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class TruthExpressionFromRowsDataset:
    scene_variant: str
    scene_variant_probabilities: dict[str, float]
    rows: tuple[TruthRowSpec, ...]
    expression: TruthExpressionSpec
    answer_value: str
    target_answer_support: tuple[str, ...]
    candidates: tuple[tuple[str, TruthExpressionSpec], ...]
    selected_option_id: str
    metadata: dict[str, Any]


def resolve_truth_table_scene_variant(
    params: Mapping[str, Any],
    *,
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
) -> tuple[str, dict[str, float]]:
    """Resolve the non-semantic truth-table style axis."""

    return resolve_symbolic_axis_variant(
        params=params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        supported_variants=SUPPORTED_TRUTH_TABLE_SCENE_VARIANTS,
        task_id=str(namespace),
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        axis_namespace="scene_variant",
    )


def build_with_retries(
    factory,
    *,
    instance_seed: int,
    max_attempts: int,
    failure_message: str,
):
    """Call a deterministic dataset factory with sequential retry seeds."""

    last_error: Exception | None = None
    for attempt_index in range(max(1, int(max_attempts))):
        try:
            return factory(int(instance_seed) + int(attempt_index))
        except Exception as exc:
            last_error = exc
    raise RuntimeError(str(failure_message)) from last_error


def _label_support(
    params: Mapping[str, Any], gen_defaults: Mapping[str, Any]
) -> tuple[str, ...]:
    labels = tuple(
        str(label)
        for label in params.get(
            "option_label_support",
            gen_defaults.get("option_label_support", OPTION_LABELS),
        )
    )
    if labels != OPTION_LABELS:
        raise ValueError("truth-table option tasks require labels A, B, C, D, E, F")
    return tuple(labels)


def _expression_label_support(
    params: Mapping[str, Any], gen_defaults: Mapping[str, Any]
) -> tuple[str, ...]:
    labels = tuple(
        str(label)
        for label in params.get(
            "expression_option_label_support",
            gen_defaults.get(
                "expression_option_label_support", EXPRESSION_OPTION_LABELS
            ),
        )
    )
    if labels != EXPRESSION_OPTION_LABELS:
        raise ValueError(
            "truth-table expression-option tasks require labels W, X, Y, Z"
        )
    return tuple(labels)


def _count_support(
    params: Mapping[str, Any], gen_defaults: Mapping[str, Any]
) -> tuple[int, ...]:
    raw_support = params.get(
        "target_count_support",
        gen_defaults.get("target_count_support", tuple(range(1, 8))),
    )
    support = tuple(int(value) for value in raw_support)
    if not support or any(value < 1 or value > 7 for value in support):
        raise ValueError("truth-table count support must stay within 1..7")
    return tuple(dict.fromkeys(support))


def _sample_expression_with_count(rng: Any, *, count: int) -> TruthExpressionSpec:
    expressions = expressions_by_true_count(int(count))
    if not expressions:
        raise ValueError(f"no curated truth-table expression has true count {count}")
    return uniform_choice(rng, expressions, sort_keys=True)


def build_count_dataset(
    *,
    rng: Any,
    scene_variant: str,
    scene_variant_probabilities: Mapping[str, float],
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
) -> TruthCountDataset:
    """Build one row-count sample with feasible non-binary answers."""

    support = _count_support(params, gen_defaults)
    if "expression_id" in params:
        expression = expression_by_id(str(params["expression_id"]))
        if int(expression.true_count) not in set(support):
            raise ValueError("expression true count is outside configured support")
        probabilities = {
            str(value): (1.0 if int(value) == int(expression.true_count) else 0.0)
            for value in support
        }
    else:
        if "target_count" in params:
            target_count = int(params["target_count"])
            if target_count not in set(support):
                raise ValueError("target_count is outside configured support")
            probabilities = {
                str(value): (1.0 if int(value) == target_count else 0.0)
                for value in support
            }
        else:
            target_count, probabilities = uniform_choice_with_probabilities(
                rng, support, sort_keys=True
            )
        expression = _sample_expression_with_count(rng, count=int(target_count))
    true_cell_ids = tuple(
        f"output_P_row_{index + 1}"
        for index, value in enumerate(expression.pattern)
        if int(value) == 1
    )
    return TruthCountDataset(
        scene_variant=str(scene_variant),
        scene_variant_probabilities={
            str(key): float(value) for key, value in scene_variant_probabilities.items()
        },
        rows=truth_rows(),
        expression=expression,
        answer_value=int(expression.true_count),
        target_answer_support=tuple(support),
        true_cell_ids=tuple(true_cell_ids),
        metadata={
            "task_mode": "truth_count",
            "expression_id": str(expression.expression_id),
            "expression_display": str(expression.display),
            "truth_pattern": str(expression.pattern_string),
            "true_row_labels": [
                str(row.row_label)
                for row, value in zip(truth_rows(), expression.pattern)
                if int(value) == 1
            ],
            "target_count_probabilities": {
                str(key): float(value) for key, value in probabilities.items()
            },
        },
    )


def _bind_option_values(
    *,
    rng: Any,
    labels: Sequence[str],
    correct_label: str,
    correct_value: str,
    distractors: Sequence[str],
) -> tuple[tuple[str, str], ...]:
    label_tuple = tuple(str(label) for label in labels)
    if str(correct_label) not in label_tuple:
        raise ValueError("correct label is outside option labels")
    distractor_values = [
        str(value) for value in distractors if str(value) != str(correct_value)
    ]
    rng.shuffle(distractor_values)
    if len(distractor_values) < len(label_tuple) - 1:
        raise ValueError("not enough distinct distractors for truth-table options")
    bound: list[tuple[str, str]] = []
    cursor = 0
    for label in label_tuple:
        if str(label) == str(correct_label):
            value = str(correct_value)
        else:
            value = str(distractor_values[cursor])
            cursor += 1
        bound.append((str(label), str(value)))
    return tuple(bound)


def build_pattern_dataset(
    *,
    rng: Any,
    scene_variant: str,
    scene_variant_probabilities: Mapping[str, float],
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
) -> TruthPatternDataset:
    """Build one pattern-option sample with a unique correct card."""

    labels = _label_support(params, gen_defaults)
    expressions = tuple(
        spec for spec in all_expression_specs() if 1 <= int(spec.true_count) <= 7
    )
    expression = (
        expression_by_id(str(params["expression_id"]))
        if "expression_id" in params
        else uniform_choice(rng, expressions, sort_keys=True)
    )
    if not 1 <= int(expression.true_count) <= 7:
        raise ValueError("truth-pattern task requires a non-constant expression")
    correct_label = str(
        params.get("correct_label", uniform_choice(rng, labels, sort_keys=False))
    )
    pattern_support = distinct_truth_patterns()
    distractors = sample_without_replacement(
        rng,
        tuple(
            pattern
            for pattern in pattern_support
            if pattern != expression.pattern_string
        ),
        len(labels) - 1,
    )
    options = _bind_option_values(
        rng=rng,
        labels=labels,
        correct_label=correct_label,
        correct_value=str(expression.pattern_string),
        distractors=distractors,
    )
    return TruthPatternDataset(
        scene_variant=str(scene_variant),
        scene_variant_probabilities={
            str(key): float(value) for key, value in scene_variant_probabilities.items()
        },
        rows=truth_rows(),
        expression=expression,
        answer_value=str(correct_label),
        target_answer_support=tuple(labels),
        options=tuple(options),
        selected_option_id=f"pattern_option_{correct_label}",
        metadata={
            "task_mode": "truth_pattern_match",
            "expression_id": str(expression.expression_id),
            "expression_display": str(expression.display),
            "truth_pattern": str(expression.pattern_string),
            "option_patterns": {str(label): str(pattern) for label, pattern in options},
            "correct_label": str(correct_label),
        },
    )


def _sample_expression_for_pattern(
    rng: Any, pattern_string: str
) -> TruthExpressionSpec:
    expressions = tuple(
        spec
        for spec in all_expression_specs()
        if spec.pattern_string == str(pattern_string)
    )
    if not expressions:
        raise ValueError(f"no curated expression has truth pattern {pattern_string!r}")
    return uniform_choice(rng, expressions, sort_keys=True)


def build_expression_from_rows_dataset(
    *,
    rng: Any,
    scene_variant: str,
    scene_variant_probabilities: Mapping[str, float],
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
) -> TruthExpressionFromRowsDataset:
    """Build one completed-output-column sample with expression options."""

    labels = _expression_label_support(params, gen_defaults)
    expressions = tuple(
        spec for spec in all_expression_specs() if 1 <= int(spec.true_count) <= 7
    )
    target = (
        expression_by_id(str(params["expression_id"]))
        if "expression_id" in params
        else uniform_choice(rng, expressions, sort_keys=True)
    )
    if not 1 <= int(target.true_count) <= 7:
        raise ValueError("expression-from-rows task requires a non-constant pattern")

    correct_expression = (
        expression_by_id(str(params["correct_expression_id"]))
        if "correct_expression_id" in params
        else target
    )
    if tuple(correct_expression.pattern) != tuple(target.pattern):
        raise ValueError("correct_expression_id must match the completed P column")

    correct_label = str(
        params.get("correct_label", uniform_choice(rng, labels, sort_keys=False))
    )
    distractor_patterns = sample_without_replacement(
        rng,
        tuple(
            pattern
            for pattern in distinct_truth_patterns()
            if pattern != str(target.pattern_string)
        ),
        len(labels) - 1,
    )
    distractors = tuple(
        _sample_expression_for_pattern(rng, pattern) for pattern in distractor_patterns
    )
    candidate_values = _bind_option_values(
        rng=rng,
        labels=labels,
        correct_label=correct_label,
        correct_value=str(correct_expression.expression_id),
        distractors=tuple(spec.expression_id for spec in distractors),
    )
    candidates = tuple(
        (label, expression_by_id(expression_id))
        for label, expression_id in candidate_values
    )
    return TruthExpressionFromRowsDataset(
        scene_variant=str(scene_variant),
        scene_variant_probabilities={
            str(key): float(value) for key, value in scene_variant_probabilities.items()
        },
        rows=truth_rows(),
        expression=target,
        answer_value=str(correct_label),
        target_answer_support=tuple(labels),
        candidates=tuple(candidates),
        selected_option_id=f"expression_option_{correct_label}",
        metadata={
            "task_mode": "expression_from_rows",
            "expression_id": str(target.expression_id),
            "expression_display": str(target.display),
            "truth_pattern": str(target.pattern_string),
            "candidate_expressions": {
                str(label): {
                    "expression_id": str(spec.expression_id),
                    "display": str(spec.display),
                    "truth_pattern": str(spec.pattern_string),
                    "matches_output_column": tuple(spec.pattern)
                    == tuple(target.pattern),
                }
                for label, spec in candidates
            },
            "correct_label": str(correct_label),
            "correct_expression_id": str(correct_expression.expression_id),
        },
    )


__all__ = [
    "TruthCountDataset",
    "TruthExpressionFromRowsDataset",
    "TruthPatternDataset",
    "build_count_dataset",
    "build_expression_from_rows_dataset",
    "build_pattern_dataset",
    "build_with_retries",
    "resolve_truth_table_scene_variant",
]
