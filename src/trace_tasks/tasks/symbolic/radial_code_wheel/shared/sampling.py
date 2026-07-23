"""Sampling helpers for radial code-wheel scene packages."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence, TypeVar

from .....core.sampling import uniform_choice
from ...shared.common import resolve_symbolic_axis_variant

from .rules import all_codes, code_to_index, terminal_label_pool, validate_terminal_labels
from .state import (
    CODE_SYMBOLS,
    OPTION_LABELS,
    RadialMissingSymbolDataset,
    SUPPORTED_RADIAL_SCENE_VARIANTS,
    RadialChoiceDataset,
    RadialOptionSpec,
    RadialReferenceSpec,
    RadialTerminalSpec,
)


T = TypeVar("T")


def build_with_retries(
    factory: Callable[[int], T],
    *,
    instance_seed: int,
    max_attempts: int,
    failure_message: str,
) -> T:
    """Call a deterministic dataset factory with sequential retry seeds."""

    last_error: Exception | None = None
    for attempt_index in range(max(1, int(max_attempts))):
        try:
            return factory(int(instance_seed) + int(attempt_index))
        except Exception as exc:
            last_error = exc
    raise RuntimeError(str(failure_message)) from last_error


def resolve_radial_scene_variant(
    params: Mapping[str, Any],
    *,
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
) -> tuple[str, dict[str, float]]:
    """Resolve the non-semantic radial wheel style axis."""

    return resolve_symbolic_axis_variant(
        params=params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        supported_variants=SUPPORTED_RADIAL_SCENE_VARIANTS,
        task_id=str(namespace),
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        axis_namespace="scene_variant",
    )


def resolve_option_count(params: Mapping[str, Any], gen_defaults: Mapping[str, Any]) -> int:
    """Return the fixed radial wheel option count."""

    option_count = int(params.get("option_count", gen_defaults.get("option_count", len(OPTION_LABELS))))
    if int(option_count) != len(OPTION_LABELS):
        raise ValueError("radial code-wheel tasks require exactly six visual options")
    return int(option_count)


def sample_terminal_assignment(rng: Any, params: Mapping[str, Any]) -> tuple[str, ...]:
    """Sample a complete unique terminal-label assignment."""

    if "terminal_labels" in params:
        return validate_terminal_labels(params["terminal_labels"])
    labels = list(terminal_label_pool())
    rng.shuffle(labels)
    return validate_terminal_labels(labels)


def sample_target_code(rng: Any, params: Mapping[str, Any]) -> str:
    """Sample or validate one three-symbol target code."""

    if "target_code" in params:
        code = str(params["target_code"]).strip().upper()
        code_to_index(code)
        return code
    return str(rng.choice(all_codes(CODE_SYMBOLS)))


def bind_option_values(
    *,
    rng: Any,
    labels: Sequence[str],
    correct_label: str,
    correct_value: str,
    distractor_values: Sequence[str],
) -> tuple[RadialOptionSpec, ...]:
    """Bind one correct value and unique distractors to fixed visible option labels."""

    label_tuple = tuple(str(label) for label in labels)
    if str(correct_label) not in label_tuple:
        raise ValueError(f"correct_label must be one of {label_tuple}")
    distractors = [str(value) for value in distractor_values if str(value) != str(correct_value)]
    if len(set(distractors)) < len(label_tuple) - 1:
        raise ValueError("not enough unique radial code-wheel distractors")
    selected = list(rng.sample(sorted(set(distractors)), len(label_tuple) - 1))
    out: list[RadialOptionSpec] = []
    cursor = 0
    for label in label_tuple:
        if str(label) == str(correct_label):
            value = str(correct_value)
            role = "correct_option"
        else:
            value = str(selected[int(cursor)])
            role = "distractor_option"
            cursor += 1
        out.append(RadialOptionSpec(item_id=f"option_{label}", label=str(label), value=str(value), role=str(role)))
    return tuple(out)


def make_terminal_specs(terminal_labels: Sequence[str]) -> tuple[RadialTerminalSpec, ...]:
    """Create terminal specs in code/index order."""

    labels = validate_terminal_labels(terminal_labels)
    return tuple(
        RadialTerminalSpec(
            item_id=f"terminal_{index}",
            code=str(code),
            output_label=str(label),
            terminal_index=int(index),
        )
        for index, (code, label) in enumerate(zip(all_codes(CODE_SYMBOLS), labels))
    )


_MISSING_POSITION_ALIASES = {
    "0": (0, "inner_ring_symbol"),
    "first": (0, "inner_ring_symbol"),
    "inner": (0, "inner_ring_symbol"),
    "inner_ring": (0, "inner_ring_symbol"),
    "1": (1, "middle_ring_symbol"),
    "second": (1, "middle_ring_symbol"),
    "middle": (1, "middle_ring_symbol"),
    "middle_ring": (1, "middle_ring_symbol"),
    "2": (2, "outer_ring_symbol"),
    "third": (2, "outer_ring_symbol"),
    "outer": (2, "outer_ring_symbol"),
    "outer_ring": (2, "outer_ring_symbol"),
}


def normalize_missing_position(value: Any) -> tuple[int, str]:
    """Normalize one missing-code position to index and annotation role."""

    key = str(value).strip().lower()
    try:
        return _MISSING_POSITION_ALIASES[key]
    except KeyError as exc:
        raise ValueError(f"unsupported radial missing-code position: {value}") from exc


def resolve_missing_position(rng: Any, params: Mapping[str, Any], gen_defaults: Mapping[str, Any]) -> tuple[int, str]:
    """Sample or validate the hidden code-symbol position."""

    explicit = params.get("missing_position", params.get("missing_position_index"))
    if explicit is not None:
        return normalize_missing_position(explicit)
    support_raw = params.get(
        "missing_position_support",
        gen_defaults.get("missing_position_support", ("inner", "middle", "outer")),
    )
    support = tuple(str(value) for value in support_raw)
    if not support:
        raise ValueError("missing_position_support must not be empty")
    selected = uniform_choice(rng, support, sort_keys=True)
    return normalize_missing_position(selected)


def resolve_missing_answer_symbol(
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
) -> tuple[str, dict[str, float]]:
    """Resolve the missing symbol as a balanced semantic answer axis."""

    return resolve_symbolic_axis_variant(
        params=params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        supported_variants=CODE_SYMBOLS,
        task_id=str(namespace),
        explicit_key="answer_symbol",
        weights_key="answer_symbol_weights",
        balance_flag_key="balanced_answer_symbol_sampling",
        axis_namespace="answer_symbol",
    )


def build_missing_code_symbol_choice(
    *,
    rng: Any,
    instance_seed: int,
    namespace: str,
    scene_variant: str,
    scene_variant_probabilities: Mapping[str, float],
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
) -> RadialMissingSymbolDataset:
    """Build a dataset where one symbol is missing from the target-output code."""

    terminal_labels = sample_terminal_assignment(rng, params)
    terminal_specs = make_terminal_specs(terminal_labels)
    missing_index, missing_role = resolve_missing_position(rng, params, gen_defaults)
    if "target_code" in params:
        target_code = sample_target_code(rng, params)
        answer_symbol = str(target_code[int(missing_index)])
        explicit_answer_symbol = params.get("answer_symbol")
        if explicit_answer_symbol is not None and str(explicit_answer_symbol).strip().upper() != str(answer_symbol):
            raise ValueError("explicit answer_symbol does not match target_code at the missing position")
        answer_symbol_probabilities = {str(symbol): (1.0 if str(symbol) == str(answer_symbol) else 0.0) for symbol in CODE_SYMBOLS}
    else:
        answer_symbol, answer_symbol_probabilities = resolve_missing_answer_symbol(
            params=params,
            gen_defaults=gen_defaults,
            instance_seed=int(instance_seed),
            namespace=str(namespace),
        )
        code_symbols = [str(rng.choice(CODE_SYMBOLS)) for _ in range(3)]
        code_symbols[int(missing_index)] = str(answer_symbol)
        target_code = "".join(code_symbols)
    target_index = code_to_index(target_code)
    target_output = str(terminal_labels[int(target_index)])
    partial_symbols = list(str(target_code))
    partial_symbols[int(missing_index)] = "?"
    partial_code = " ".join(str(symbol) for symbol in partial_symbols)
    return RadialMissingSymbolDataset(
        scene_variant=str(scene_variant),
        answer_value=str(answer_symbol),
        target_answer_support=tuple(str(symbol) for symbol in CODE_SYMBOLS),
        target_output_label=str(target_output),
        target_code=str(target_code),
        partial_code=str(partial_code),
        missing_position_index=int(missing_index),
        missing_ring_role=str(missing_role),
        terminal_specs=tuple(terminal_specs),
        annotation_item_id=str(missing_role),
        metadata={
            "task_mode": "missing_code_symbol",
            "symbols": list(CODE_SYMBOLS),
            "code_length": 3,
            "target_code": str(target_code),
            "partial_code": str(partial_code),
            "missing_position_index": int(missing_index),
            "missing_ring_role": str(missing_role),
            "answer_symbol": str(answer_symbol),
            "answer_symbol_probabilities": {str(key): float(value) for key, value in answer_symbol_probabilities.items()},
            "target_terminal_index": int(target_index),
            "target_output_label": str(target_output),
            "candidate_symbols": list(CODE_SYMBOLS),
            "terminal_labels_by_code": {spec.code: spec.output_label for spec in terminal_specs},
        },
        scene_variant_probabilities={str(key): float(value) for key, value in scene_variant_probabilities.items()},
    )


def build_code_output_choice(
    *,
    rng: Any,
    scene_variant: str,
    scene_variant_probabilities: Mapping[str, float],
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    labels: Sequence[str],
) -> RadialChoiceDataset:
    """Build a choice dataset where the source is a three-symbol code."""

    resolve_option_count(params, gen_defaults)
    terminal_labels = sample_terminal_assignment(rng, params)
    terminal_specs = make_terminal_specs(terminal_labels)
    target_code = sample_target_code(rng, params)
    target_index = code_to_index(target_code)
    target_output = str(terminal_labels[int(target_index)])
    label_tuple = tuple(str(label) for label in labels)
    correct_label = str(params.get("correct_label", uniform_choice(rng, label_tuple, sort_keys=False)))
    options = bind_option_values(
        rng=rng,
        labels=label_tuple,
        correct_label=str(correct_label),
        correct_value=str(target_output),
        distractor_values=tuple(terminal_labels),
    )
    option_map = {str(option.label): str(option.value) for option in options}
    return RadialChoiceDataset(
        scene_variant=str(scene_variant),
        answer_value=str(correct_label),
        target_answer_support=tuple(label_tuple),
        reference=RadialReferenceSpec(item_id="source_code", title="Code", value=str(target_code), role="source_code"),
        options=tuple(options),
        terminal_specs=tuple(terminal_specs),
        target_code=str(target_code),
        target_output_label=str(target_output),
        target_terminal_index=int(target_index),
        annotation_item_ids=("inner_ring_symbol", "middle_ring_symbol", "outer_ring_symbol"),
        metadata={
            "task_mode": "code_to_output",
            "symbols": list(CODE_SYMBOLS),
            "code_length": 3,
            "target_code": str(target_code),
            "target_terminal_index": int(target_index),
            "target_output_label": str(target_output),
            "correct_option_label": str(correct_label),
            "correct_option_id": f"option_{correct_label}",
            "option_values": dict(option_map),
            "terminal_labels_by_code": {spec.code: spec.output_label for spec in terminal_specs},
        },
        scene_variant_probabilities={str(key): float(value) for key, value in scene_variant_probabilities.items()},
    )


def build_output_code_match_choice(
    *,
    rng: Any,
    scene_variant: str,
    scene_variant_probabilities: Mapping[str, float],
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    labels: Sequence[str],
) -> RadialChoiceDataset:
    """Build a choice dataset where the source is a terminal output label."""

    resolve_option_count(params, gen_defaults)
    terminal_labels = sample_terminal_assignment(rng, params)
    terminal_specs = make_terminal_specs(terminal_labels)
    target_code = sample_target_code(rng, params)
    target_index = code_to_index(target_code)
    target_output = str(terminal_labels[int(target_index)])
    label_tuple = tuple(str(label) for label in labels)
    correct_label = str(params.get("correct_label", uniform_choice(rng, label_tuple, sort_keys=False)))
    all_code_values = all_codes(CODE_SYMBOLS)
    options = bind_option_values(
        rng=rng,
        labels=label_tuple,
        correct_label=str(correct_label),
        correct_value=str(target_code),
        distractor_values=all_code_values,
    )
    option_map = {str(option.label): str(option.value) for option in options}
    return RadialChoiceDataset(
        scene_variant=str(scene_variant),
        answer_value=str(correct_label),
        target_answer_support=tuple(label_tuple),
        reference=RadialReferenceSpec(item_id="target_output", title="Output", value=str(target_output), role="target_output"),
        options=tuple(options),
        terminal_specs=tuple(terminal_specs),
        target_code=str(target_code),
        target_output_label=str(target_output),
        target_terminal_index=int(target_index),
        annotation_item_ids=("inner_ring_symbol", "middle_ring_symbol", "outer_ring_symbol"),
        metadata={
            "task_mode": "output_to_code",
            "symbols": list(CODE_SYMBOLS),
            "code_length": 3,
            "target_code": str(target_code),
            "target_terminal_index": int(target_index),
            "target_output_label": str(target_output),
            "correct_option_label": str(correct_label),
            "correct_option_id": f"option_{correct_label}",
            "option_values": dict(option_map),
            "terminal_labels_by_code": {spec.code: spec.output_label for spec in terminal_specs},
        },
        scene_variant_probabilities={str(key): float(value) for key, value in scene_variant_probabilities.items()},
    )


__all__ = [
    "build_code_output_choice",
    "build_missing_code_symbol_choice",
    "build_output_code_match_choice",
    "build_with_retries",
    "bind_option_values",
    "make_terminal_specs",
    "normalize_missing_position",
    "resolve_missing_answer_symbol",
    "resolve_option_count",
    "resolve_missing_position",
    "resolve_radial_scene_variant",
    "sample_target_code",
    "sample_terminal_assignment",
]
