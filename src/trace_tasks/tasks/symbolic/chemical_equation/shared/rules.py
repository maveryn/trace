"""Rule helpers for symbolic chemical-equation balancing scenes."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Mapping, Sequence

from .state import ChemicalReactionDef, ChemicalTermDef

_FORMULA_TOKEN_RE = re.compile(r"([A-Z][a-z]?)([0-9]*)")


def parse_formula(formula: str) -> ChemicalTermDef:
    """Parse a simple chemistry formula with element symbols and numeric suffixes."""

    text = str(formula).strip()
    if not text:
        raise ValueError("formula cannot be empty")
    position = 0
    counts: dict[str, int] = {}
    order: list[str] = []
    for match in _FORMULA_TOKEN_RE.finditer(text):
        if int(match.start()) != position:
            raise ValueError(f"unsupported formula syntax: {formula}")
        element = str(match.group(1))
        count_text = str(match.group(2) or "")
        count = int(count_text) if count_text else 1
        if count <= 0:
            raise ValueError(f"formula atom counts must be positive: {formula}")
        if element not in counts:
            order.append(element)
            counts[element] = 0
        counts[element] += int(count)
        position = int(match.end())
    if position != len(text):
        raise ValueError(f"unsupported formula syntax: {formula}")
    return ChemicalTermDef(
        formula=text,
        atom_counts=dict(counts),
        element_order=tuple(order),
    )


def _reaction(
    reaction_id: str,
    left_formulas: Sequence[str],
    right_formulas: Sequence[str],
    coefficients: Sequence[int],
    *,
    family: str,
) -> ChemicalReactionDef:
    left = tuple(str(value) for value in left_formulas)
    right = tuple(str(value) for value in right_formulas)
    coeffs = tuple(int(value) for value in coefficients)
    if len(coeffs) != len(left) + len(right):
        raise ValueError(f"coefficient count does not match terms for {reaction_id}")
    if not all(1 <= int(value) <= 5 for value in coeffs):
        raise ValueError(f"v1 reaction coefficients must stay within 1..5: {reaction_id}")
    return ChemicalReactionDef(
        reaction_id=str(reaction_id),
        left_formulas=left,
        right_formulas=right,
        coefficients=coeffs,
        family=str(family),
    )


REACTION_BANK: tuple[ChemicalReactionDef, ...] = (
    _reaction("water_synthesis", ("H2", "O2"), ("H2O",), (2, 1, 2), family="synthesis"),
    _reaction("ammonia_synthesis", ("N2", "H2"), ("NH3",), (1, 3, 2), family="synthesis"),
    _reaction("methane_combustion", ("CH4", "O2"), ("CO2", "H2O"), (1, 2, 1, 2), family="combustion"),
    _reaction("iron_oxide", ("Fe", "O2"), ("Fe2O3",), (4, 3, 2), family="synthesis"),
    _reaction("potassium_chlorate", ("KClO3",), ("KCl", "O2"), (2, 2, 3), family="decomposition"),
    _reaction("phosphorus_oxide", ("P4", "O2"), ("P2O5",), (1, 5, 2), family="synthesis"),
    _reaction("propane_combustion", ("C3H8", "O2"), ("CO2", "H2O"), (1, 5, 3, 4), family="combustion"),
    _reaction("sodium_chloride", ("Na", "Cl2"), ("NaCl",), (2, 1, 2), family="synthesis"),
    _reaction("magnesium_chloride", ("Mg", "HCl"), ("MgCl2", "H2"), (1, 2, 1, 1), family="replacement"),
    _reaction("aluminum_oxide", ("Al", "O2"), ("Al2O3",), (4, 3, 2), family="synthesis"),
)


def reaction_by_id(reaction_id: str) -> ChemicalReactionDef:
    """Return one curated reaction by id."""

    target = str(reaction_id)
    for reaction in REACTION_BANK:
        if str(reaction.reaction_id) == target:
            return reaction
    raise ValueError(f"unsupported reaction_id: {reaction_id}")


def formulas_for_reaction(reaction: ChemicalReactionDef) -> tuple[str, ...]:
    """Return all formulas in left-to-right equation order."""

    return tuple(reaction.left_formulas) + tuple(reaction.right_formulas)


def parsed_terms_for_reaction(reaction: ChemicalReactionDef) -> tuple[ChemicalTermDef, ...]:
    """Return parsed formula terms in equation order."""

    return tuple(parse_formula(formula) for formula in formulas_for_reaction(reaction))


def side_totals_for_coefficients(
    reaction: ChemicalReactionDef,
    coefficients: Sequence[int],
) -> dict[str, dict[str, int]]:
    """Return left and right atom totals for one coefficient tuple."""

    coeffs = tuple(int(value) for value in coefficients)
    if len(coeffs) != len(reaction.coefficients):
        raise ValueError("coefficient tuple length does not match reaction terms")
    parsed_terms = parsed_terms_for_reaction(reaction)
    totals = {"left": defaultdict(int), "right": defaultdict(int)}
    left_count = len(reaction.left_formulas)
    for index, term in enumerate(parsed_terms):
        side = "left" if index < left_count else "right"
        coefficient = int(coeffs[index])
        for element, count in term.atom_counts.items():
            totals[side][str(element)] += int(coefficient) * int(count)
    return {
        "left": {str(key): int(value) for key, value in totals["left"].items()},
        "right": {str(key): int(value) for key, value in totals["right"].items()},
    }


def is_balanced_coefficients(
    reaction: ChemicalReactionDef,
    coefficients: Sequence[int],
) -> bool:
    """Return whether a coefficient tuple balances every visible element."""

    totals = side_totals_for_coefficients(reaction, coefficients)
    elements = set(totals["left"]) | set(totals["right"])
    return all(
        int(totals["left"].get(element, 0)) == int(totals["right"].get(element, 0))
        for element in elements
    )


for _reaction_spec in REACTION_BANK:
    if not is_balanced_coefficients(_reaction_spec, _reaction_spec.coefficients):
        raise ValueError(f"reaction is not balanced: {_reaction_spec.reaction_id}")


def matching_coefficient_slots(
    coefficient_value: int,
    *,
    reaction_id: str | None = None,
) -> tuple[tuple[ChemicalReactionDef, int], ...]:
    """Return all reaction slots whose balanced coefficient has the requested value."""

    matches: list[tuple[ChemicalReactionDef, int]] = []
    for reaction in REACTION_BANK:
        if reaction_id is not None and str(reaction.reaction_id) != str(reaction_id):
            continue
        for index, coefficient in enumerate(reaction.coefficients):
            if int(coefficient) == int(coefficient_value):
                matches.append((reaction, int(index)))
    return tuple(matches)


def reaction_metadata(reaction: ChemicalReactionDef) -> dict[str, object]:
    """Serialize reaction facts for trace metadata."""

    return {
        "reaction_id": str(reaction.reaction_id),
        "family": str(reaction.family),
        "left_formulas": list(reaction.left_formulas),
        "right_formulas": list(reaction.right_formulas),
        "balanced_coefficients": [int(value) for value in reaction.coefficients],
        "atom_totals": side_totals_for_coefficients(reaction, reaction.coefficients),
    }


__all__ = [
    "REACTION_BANK",
    "formulas_for_reaction",
    "is_balanced_coefficients",
    "matching_coefficient_slots",
    "parse_formula",
    "parsed_terms_for_reaction",
    "reaction_by_id",
    "reaction_metadata",
    "side_totals_for_coefficients",
]
