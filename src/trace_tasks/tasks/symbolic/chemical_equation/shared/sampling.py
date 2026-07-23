"""Sampling helpers for symbolic chemical-equation tasks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Mapping

from .....core.sampling import (
    sample_without_replacement,
    uniform_choice_with_probabilities,
)
from ....shared.config_defaults import group_default
from ...shared.common import resolve_symbolic_axis_variant
from .rules import (
    REACTION_BANK,
    formulas_for_reaction,
    is_balanced_coefficients,
    matching_coefficient_slots,
    parsed_terms_for_reaction,
    reaction_by_id,
    reaction_metadata,
    side_totals_for_coefficients,
)
from .state import (
    COEFFICIENT_SUPPORT,
    OPTION_LABELS,
    SCENE_VARIANTS,
    ChemicalEquationDataset,
    ChemicalOptionSpec,
    ChemicalReactionDef,
    ChemicalTermSpec,
)


def resolve_chemical_scene_variant(
    params: Mapping[str, Any],
    *,
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    sampling_scope: str,
) -> tuple[str, dict[str, float]]:
    """Resolve one non-semantic chemical-equation visual variant."""

    return resolve_symbolic_axis_variant(
        params=params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        supported_variants=SCENE_VARIANTS,
        task_id=str(sampling_scope),
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        axis_namespace="scene_variant",
    )


def _support_from_defaults(
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    *,
    key: str,
    fallback: Sequence[int | str],
) -> tuple[int | str, ...]:
    raw = params.get(str(key), group_default(gen_defaults, str(key), list(fallback)))
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError(f"{key} must be a sequence")
    values: list[int | str] = []
    for value in raw:
        if isinstance(value, int):
            values.append(int(value))
        else:
            text = str(value)
            values.append(int(text) if text.isdigit() else text)
    if not values:
        raise ValueError(f"{key} cannot be empty")
    if len({str(value) for value in values}) != len(values):
        raise ValueError(f"{key} values must be unique")
    return tuple(values)


def _coefficient_support(
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
) -> tuple[int, ...]:
    values = _support_from_defaults(
        params,
        gen_defaults,
        key="coefficient_answer_support",
        fallback=COEFFICIENT_SUPPORT,
    )
    support = tuple(int(value) for value in values)
    if any(value < 1 or value > 5 for value in support):
        raise ValueError("chemical-equation v1 coefficients must stay within 1..5")
    return support


def _option_label_support(
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
) -> tuple[str, ...]:
    values = _support_from_defaults(
        params,
        gen_defaults,
        key="option_label_support",
        fallback=OPTION_LABELS,
    )
    support = tuple(str(value) for value in values)
    if len(support) != 4:
        raise ValueError("chemical-equation v1 option tasks require four option labels")
    return support


def _term_specs(
    reaction: ChemicalReactionDef,
    *,
    hidden_slot_indices: Sequence[int],
) -> tuple[ChemicalTermSpec, ...]:
    parsed_terms = parsed_terms_for_reaction(reaction)
    hidden_slots = {int(value) for value in hidden_slot_indices}
    terms: list[ChemicalTermSpec] = []
    left_count = len(reaction.left_formulas)
    for term_index, term_def in enumerate(parsed_terms):
        side = "left" if term_index < left_count else "right"
        side_index = term_index if side == "left" else term_index - left_count
        terms.append(
            ChemicalTermSpec(
                item_id=f"term_{term_index + 1}",
                coefficient_slot_id=f"coefficient_slot_{term_index + 1}",
                molecule_card_id=f"molecule_card_{term_index + 1}",
                side=side,
                side_index=int(side_index),
                term_index=int(term_index),
                formula=str(term_def.formula),
                coefficient=int(reaction.coefficients[term_index]),
                hidden_coefficient=int(term_index) in hidden_slots,
                atom_counts=dict(term_def.atom_counts),
                element_order=tuple(term_def.element_order),
            )
        )
    return tuple(terms)


def _reaction_choice(
    rng,
    params: Mapping[str, Any],
) -> ChemicalReactionDef:
    explicit = params.get("reaction_id")
    if explicit is not None:
        return reaction_by_id(str(explicit))
    reaction, _probabilities = uniform_choice_with_probabilities(
        rng,
        REACTION_BANK,
        sort_keys=False,
    )
    return reaction


def build_missing_coefficient_dataset(
    *,
    rng,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    scene_variant: str,
    scene_variant_probabilities: Mapping[str, float],
) -> ChemicalEquationDataset:
    """Build one missing-coefficient instance from answer-first sampling."""

    support = _coefficient_support(params, gen_defaults)
    explicit_reaction_id = params.get("reaction_id")
    explicit_slot = params.get("hidden_slot_index", params.get("hidden_coefficient_slot"))
    explicit_answer = params.get("answer_value")

    if explicit_reaction_id is not None and explicit_slot is not None:
        reaction = reaction_by_id(str(explicit_reaction_id))
        hidden_slot_index = int(explicit_slot)
        if not 0 <= hidden_slot_index < len(reaction.coefficients):
            raise ValueError("hidden_slot_index is outside reaction term range")
        answer_value = int(reaction.coefficients[hidden_slot_index])
        if explicit_answer is not None and int(explicit_answer) != int(answer_value):
            raise ValueError("answer_value conflicts with selected hidden coefficient")
        answer_probabilities = {str(answer_value): 1.0}
    else:
        if explicit_answer is not None:
            answer_value = int(explicit_answer)
            if answer_value not in support:
                raise ValueError("answer_value is outside coefficient support")
            answer_probabilities = {str(answer_value): 1.0}
        else:
            selected_answer, answer_probabilities = uniform_choice_with_probabilities(
                rng,
                support,
                sort_keys=True,
            )
            answer_value = int(selected_answer)
        matches = matching_coefficient_slots(
            int(answer_value),
            reaction_id=None if explicit_reaction_id is None else str(explicit_reaction_id),
        )
        if not matches:
            raise ValueError("no reaction slot matches requested coefficient answer")
        if explicit_slot is not None:
            slot_index = int(explicit_slot)
            matches = tuple(
                (reaction, index)
                for reaction, index in matches
                if int(index) == int(slot_index)
            )
            if not matches:
                raise ValueError("hidden_slot_index does not match requested answer")
        reaction, hidden_slot_index = sample_without_replacement(rng, matches, 1)[0]

    terms = _term_specs(reaction, hidden_slot_indices=(int(hidden_slot_index),))
    metadata = {
        "task_kind": "coefficient_blank",
        "hidden_slot_index": int(hidden_slot_index),
        "hidden_slot_number": int(hidden_slot_index) + 1,
        "coefficient_answer_support": [int(value) for value in support],
        "reaction": reaction_metadata(reaction),
        "term_formulas": list(formulas_for_reaction(reaction)),
        "visible_coefficients": {
            str(index + 1): (
                None
                if int(index) == int(hidden_slot_index)
                else int(coefficient)
            )
            for index, coefficient in enumerate(reaction.coefficients)
        },
    }
    return ChemicalEquationDataset(
        task_kind="coefficient_blank",
        scene_variant=str(scene_variant),
        scene_variant_probabilities=dict(scene_variant_probabilities),
        reaction=reaction,
        terms=terms,
        hidden_slot_index=int(hidden_slot_index),
        answer_value=int(answer_value),
        target_answer_support=tuple(int(value) for value in support),
        target_answer_probabilities={str(key): float(value) for key, value in answer_probabilities.items()},
        metadata=metadata,
    )


def _distractor_coefficients(
    rng,
    reaction: ChemicalReactionDef,
    *,
    option_count: int,
    coefficient_support: Sequence[int],
) -> tuple[tuple[int, ...], ...]:
    """Return unique non-balancing coefficient tuples."""

    correct = tuple(int(value) for value in reaction.coefficients)
    support = tuple(int(value) for value in coefficient_support)
    candidates: list[tuple[int, ...]] = []
    attempts = 0
    while len(candidates) < int(option_count) - 1 and attempts < 1000:
        attempts += 1
        candidate = list(correct)
        edit_count = 1 if rng.random() < 0.65 else 2
        edit_indices = rng.sample(range(len(candidate)), k=min(edit_count, len(candidate)))
        for index in edit_indices:
            alternatives = [value for value in support if int(value) != int(candidate[index])]
            if alternatives:
                candidate[index] = int(rng.choice(alternatives))
        item = tuple(int(value) for value in candidate)
        if item == correct:
            continue
        if item in candidates:
            continue
        if is_balanced_coefficients(reaction, item):
            continue
        candidates.append(item)
    if len(candidates) < int(option_count) - 1:
        raise RuntimeError("failed to generate unique unbalanced coefficient options")
    return tuple(candidates)


def build_balanced_option_dataset(
    *,
    rng,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    scene_variant: str,
    scene_variant_probabilities: Mapping[str, float],
) -> ChemicalEquationDataset:
    """Build one option-card instance with exactly one balanced coefficient tuple."""

    coefficient_support = _coefficient_support(params, gen_defaults)
    label_support = _option_label_support(params, gen_defaults)
    reaction = _reaction_choice(rng, params)
    if any(int(value) not in coefficient_support for value in reaction.coefficients):
        raise ValueError("selected reaction has coefficients outside configured support")

    explicit_label = params.get("correct_label", params.get("answer_value"))
    if explicit_label is not None:
        correct_label = str(explicit_label)
        if correct_label not in set(label_support):
            raise ValueError("correct_label is outside option label support")
        label_probabilities = {
            label: (1.0 if label == correct_label else 0.0)
            for label in label_support
        }
    else:
        correct_label, label_probabilities = uniform_choice_with_probabilities(
            rng,
            label_support,
            sort_keys=False,
        )
        correct_label = str(correct_label)

    distractors = list(
        _distractor_coefficients(
            rng,
            reaction,
            option_count=len(label_support),
            coefficient_support=coefficient_support,
        )
    )
    options: list[ChemicalOptionSpec] = []
    for label in label_support:
        if str(label) == str(correct_label):
            coefficients = tuple(int(value) for value in reaction.coefficients)
            balances = True
        else:
            coefficients = tuple(int(value) for value in distractors.pop(0))
            balances = False
        options.append(
            ChemicalOptionSpec(
                item_id=f"option_{label}",
                label=str(label),
                coefficients=coefficients,
                balances_equation=bool(balances),
            )
        )

    terms = _term_specs(reaction, hidden_slot_indices=range(len(reaction.coefficients)))
    metadata = {
        "task_kind": "coefficient_option_selection",
        "option_label_support": list(label_support),
        "coefficient_answer_support": [int(value) for value in coefficient_support],
        "reaction": reaction_metadata(reaction),
        "term_formulas": list(formulas_for_reaction(reaction)),
        "options": {
            str(option.label): {
                "coefficients": [int(value) for value in option.coefficients],
                "balances_equation": bool(option.balances_equation),
                "atom_totals": side_totals_for_coefficients(reaction, option.coefficients),
            }
            for option in options
        },
    }
    return ChemicalEquationDataset(
        task_kind="coefficient_option_selection",
        scene_variant=str(scene_variant),
        scene_variant_probabilities=dict(scene_variant_probabilities),
        reaction=reaction,
        terms=terms,
        hidden_slot_index=None,
        answer_value=str(correct_label),
        target_answer_support=tuple(str(value) for value in label_support),
        target_answer_probabilities={
            str(key): float(value) for key, value in label_probabilities.items()
        },
        options=tuple(options),
        correct_option_label=str(correct_label),
        correct_option_label_probabilities={
            str(key): float(value) for key, value in label_probabilities.items()
        },
        metadata=metadata,
    )


__all__ = [
    "build_balanced_option_dataset",
    "build_missing_coefficient_dataset",
    "resolve_chemical_scene_variant",
]
