"""Choose the relation of a queried balance-scale comparison."""

from __future__ import annotations

from itertools import product
from typing import Any, Dict, Mapping, Sequence

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import (
    load_scene_generation_rendering_prompt_defaults,
)
from trace_tasks.tasks.shared.fixed_query import DEFAULT_QUERY_ID, select_task_query_id

from ._lifecycle import run_balance_scene_lifecycle
from .shared.constraints import UNKNOWN_LABELS
from .shared.rules import (
    expressions_match,
    make_panel,
    maybe_swap_sides,
    side_total,
)
from .shared.sampling import object_specs_for_labels, resolve_scene_axes
from .shared.state import SCENE_ID, SIDE_RELATION_ROW_KIND

TASK_ID = "task_puzzles__balance_scale__query_side_relation_label"
QUERY_ID = DEFAULT_QUERY_ID
SUPPORTED_QUERY_IDS = (QUERY_ID,)
PROMPT_QUERY_KEY = "query_side_relation_label"
TASK_PROMPT_KEY = "balance_scale_query"
OPTION_LABELS = ("A", "B", "C", "D")
RELATION_BY_OPTION = {
    "A": "left",
    "B": "right",
    "C": "balanced",
    "D": "not_determined",
}
OPTION_BY_RELATION = {
    relation: option_label
    for option_label, relation in RELATION_BY_OPTION.items()
}
RELATION_DISPLAY_TEXT = {
    "left": "Left",
    "right": "Right",
    "balanced": "Balanced",
    "not_determined": "Cannot determine",
}
TARGET_RELATIONS = ("left", "right", "balanced", "not_determined")
NEUTRAL_FREE_LABEL_FAMILIES = (
    "shared_single_helpers",
    "shared_repeated_helpers",
    "repeated_free_single_helpers",
    "numeric_helper_balance",
    "compound_equal_helper_balance",
)
WEIGHT_SUPPORT = tuple(range(1, 21))

_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS_UNUSED = (
    load_scene_generation_rendering_prompt_defaults(
        "puzzles",
        SCENE_ID,
        task_id=TASK_ID,
    )
)


def _object_term(label: str, count: int = 1) -> dict[str, Any]:
    return {
        "kind": "object",
        "object_label": str(label),
        "count": int(count),
    }


def _numeric_term(value: int) -> dict[str, Any]:
    return {"kind": "numeric", "value": int(value)}


def _numeric_terms_for_sum(value: int, *, rng) -> list[dict[str, Any]]:
    """Represent a visible numeric total as one or two weight chips."""

    total = int(value)
    if total <= 0:
        raise RuntimeError("numeric balance token total must be positive")
    if total >= 5 and rng.random() < 0.35:
        first = int(rng.randint(1, int(total) - 1))
        return [_numeric_term(first), _numeric_term(int(total) - first)]
    return [_numeric_term(total)]


def _equation_record(
    *,
    equation_id: str,
    left_terms: Sequence[Mapping[str, Any]],
    right_terms: Sequence[Mapping[str, Any]],
    family: str,
) -> dict[str, Any]:
    """Create a traceable reference-equation record for this objective."""

    return {
        "equation_id": str(equation_id),
        "equation_family": str(family),
        "left_terms": [dict(term) for term in left_terms],
        "right_terms": [dict(term) for term in right_terms],
    }


def _direct_value_panel(
    *,
    panel_index: int,
    label: str,
    weights: Mapping[str, int],
    rng,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build one balanced scale that directly determines an object value."""

    weight = int(weights[str(label)])
    offset = int(rng.randint(2, 8))
    object_side = [
        _object_term(str(label)),
        *_numeric_terms_for_sum(int(offset), rng=rng),
    ]
    numeric_side = _numeric_terms_for_sum(int(weight) + int(offset), rng=rng)
    left_terms, right_terms = maybe_swap_sides(object_side, numeric_side, rng=rng)
    if expressions_match(left_terms, right_terms):
        raise RuntimeError("direct value panel cannot use identical expressions")
    panel = make_panel(
        panel_index=int(panel_index),
        left_terms=left_terms,
        right_terms=right_terms,
        weights=weights,
    )
    equation = _equation_record(
        equation_id=f"direct_value_constraint_{int(panel_index)}",
        left_terms=left_terms,
        right_terms=right_terms,
        family="direct_single_unknown_value",
    )
    return panel, equation


def _neutral_free_label_panel(
    *,
    panel_index: int,
    free_label: str,
    left_helper_label: str,
    right_helper_label: str,
    family: str,
    weights: Mapping[str, int],
    rng,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Show the free label in a balanced equation without determining it."""

    helper_weight = int(weights[str(left_helper_label)])
    family_name = str(family)
    if family_name == "shared_single_helpers":
        left_terms = [
            _object_term(str(free_label)),
            _object_term(str(left_helper_label)),
        ]
        right_terms = [
            _object_term(str(free_label)),
            _object_term(str(right_helper_label)),
        ]
    elif family_name == "shared_repeated_helpers":
        left_terms = [
            _object_term(str(free_label)),
            _object_term(str(left_helper_label), count=2),
        ]
        right_terms = [
            _object_term(str(free_label)),
            _object_term(str(right_helper_label), count=2),
        ]
    elif family_name == "repeated_free_single_helpers":
        left_terms = [
            _object_term(str(free_label), count=2),
            _object_term(str(left_helper_label)),
        ]
        right_terms = [
            _object_term(str(free_label), count=2),
            _object_term(str(right_helper_label)),
        ]
    elif family_name == "numeric_helper_balance":
        left_terms = [
            _object_term(str(free_label)),
            _object_term(str(left_helper_label)),
        ]
        right_terms = [
            _object_term(str(free_label)),
            *_numeric_terms_for_sum(int(helper_weight), rng=rng),
        ]
    elif family_name == "compound_equal_helper_balance":
        left_terms = [
            _object_term(str(free_label)),
            _object_term(str(left_helper_label)),
            _object_term(str(right_helper_label)),
        ]
        right_terms = [
            _object_term(str(free_label)),
            _object_term(str(left_helper_label), count=2),
        ]
    else:
        raise RuntimeError(f"unknown neutral free-label family: {family_name}")
    left_terms, right_terms = maybe_swap_sides(left_terms, right_terms, rng=rng)
    if expressions_match(left_terms, right_terms):
        raise RuntimeError("neutral panel cannot use identical expressions")
    panel = make_panel(
        panel_index=int(panel_index),
        left_terms=left_terms,
        right_terms=right_terms,
        weights=weights,
    )
    equation = _equation_record(
        equation_id=f"neutral_free_label_constraint_{int(panel_index)}",
        left_terms=left_terms,
        right_terms=right_terms,
        family=f"neutral_free_label_{family_name}",
    )
    return panel, equation


def _relation_name(
    left_terms: Sequence[Mapping[str, Any]],
    right_terms: Sequence[Mapping[str, Any]],
    weights: Mapping[str, int],
) -> str:
    """Return left/right/balanced for one concrete assignment."""

    left_value = side_total(left_terms, weights)
    right_value = side_total(right_terms, weights)
    if int(left_value) > int(right_value):
        return "left"
    if int(right_value) > int(left_value):
        return "right"
    return "balanced"


def _consistent_assignments(
    *,
    equations: Sequence[Mapping[str, Any]],
    labels: Sequence[str],
    support: Sequence[int],
) -> list[dict[str, int]]:
    """Enumerate hidden-weight assignments consistent with shown references."""

    label_list = [str(label) for label in labels]
    assignments: list[dict[str, int]] = []
    for values in product([int(value) for value in support], repeat=len(label_list)):
        weights = {label: int(value) for label, value in zip(label_list, values)}
        if all(
            side_total(equation["left_terms"], weights)
            == side_total(equation["right_terms"], weights)
            for equation in equations
        ):
            assignments.append(weights)
    return assignments


def _query_relation_outcomes(
    *,
    equations: Sequence[Mapping[str, Any]],
    labels: Sequence[str],
    support: Sequence[int],
    left_terms: Sequence[Mapping[str, Any]],
    right_terms: Sequence[Mapping[str, Any]],
) -> tuple[list[str], int]:
    """Return all query-side relations allowed by the reference scales."""

    assignments = _consistent_assignments(
        equations=equations,
        labels=labels,
        support=support,
    )
    outcomes = {
        _relation_name(left_terms, right_terms, weights)
        for weights in assignments
    }
    return sorted(outcomes), len(assignments)


def _relation_options() -> list[dict[str, str]]:
    """Return fixed visual options for the four relation answer classes."""

    return [
        {
            "option_label": str(option_label),
            "relation": str(RELATION_BY_OPTION[str(option_label)]),
            "display_text": str(
                RELATION_DISPLAY_TEXT[RELATION_BY_OPTION[str(option_label)]]
            ),
        }
        for option_label in OPTION_LABELS
    ]


def _determined_relation_dataset(
    *,
    target_relation: str,
    rng,
    instance_seed: int,
    scene_variant: str,
) -> dict[str, Any]:
    """Build fully determined references for left/right/balanced relations."""

    labels = list(UNKNOWN_LABELS)
    rng.shuffle(labels)
    lighter_label, heavier_label = labels[0], labels[1]
    third_label = labels[2]
    lighter_weight = int(rng.randint(4, 11))
    difference = int(rng.randint(3, 7))
    heavier_weight = int(lighter_weight) + int(difference)
    weights = {
        str(lighter_label): int(lighter_weight),
        str(heavier_label): int(heavier_weight),
        str(third_label): int(rng.randint(3, 18)),
    }
    panels: list[dict[str, Any]] = []
    equations: list[dict[str, Any]] = []
    for panel_index, label in enumerate(UNKNOWN_LABELS, start=1):
        panel, equation = _direct_value_panel(
            panel_index=int(panel_index),
            label=str(label),
            weights=weights,
            rng=rng,
        )
        panels.append(panel)
        equations.append(equation)

    if str(target_relation) == "left":
        query_left_terms = [_object_term(str(heavier_label))]
        query_right_terms = [_object_term(str(lighter_label))]
    elif str(target_relation) == "right":
        query_left_terms = [_object_term(str(lighter_label))]
        query_right_terms = [_object_term(str(heavier_label))]
    else:
        query_left_terms = [
            _object_term(str(lighter_label)),
            _numeric_term(int(difference)),
        ]
        query_right_terms = [_object_term(str(heavier_label))]
        query_left_terms, query_right_terms = maybe_swap_sides(
            query_left_terms,
            query_right_terms,
            rng=rng,
        )

    return _finalize_dataset(
        instance_seed=int(instance_seed),
        scene_variant=str(scene_variant),
        weights=weights,
        panels=panels,
        equations=equations,
        query_left_terms=query_left_terms,
        query_right_terms=query_right_terms,
        target_relation=str(target_relation),
        construction_mode=f"determined_{target_relation}_query_relation",
    )


def _not_determined_dataset(
    *,
    rng,
    instance_seed: int,
    scene_variant: str,
) -> dict[str, Any]:
    """Build references that intentionally leave the query relation ambiguous."""

    equal_weight = int(rng.randint(8, 12))
    free_weight = int(rng.choice([3, 5, 15, 18]))
    free_label = str(rng.choice(UNKNOWN_LABELS))
    fixed_labels = [str(label) for label in UNKNOWN_LABELS if str(label) != free_label]
    rng.shuffle(fixed_labels)
    fixed_a, fixed_b = fixed_labels
    neutral_family = str(rng.choice(NEUTRAL_FREE_LABEL_FAMILIES))
    weights = {
        str(fixed_a): int(equal_weight),
        str(fixed_b): int(equal_weight),
        str(free_label): int(free_weight),
    }
    panels: list[dict[str, Any]] = []
    equations: list[dict[str, Any]] = []
    for panel_index, label in enumerate((fixed_a, fixed_b), start=1):
        panel, equation = _direct_value_panel(
            panel_index=int(panel_index),
            label=str(label),
            weights=weights,
            rng=rng,
        )
        panels.append(panel)
        equations.append(equation)
    panel, equation = _neutral_free_label_panel(
        panel_index=3,
        free_label=str(free_label),
        left_helper_label=str(fixed_a),
        right_helper_label=str(fixed_b),
        family=str(neutral_family),
        weights=weights,
        rng=rng,
    )
    panels.append(panel)
    equations.append(equation)

    return _finalize_dataset(
        instance_seed=int(instance_seed),
        scene_variant=str(scene_variant),
        weights=weights,
        panels=panels,
        equations=equations,
        query_left_terms=[_object_term(str(free_label))],
        query_right_terms=[_object_term(str(fixed_a))],
        target_relation="not_determined",
        construction_mode=f"ambiguous_free_label_{neutral_family}",
    )


def _finalize_dataset(
    *,
    instance_seed: int,
    scene_variant: str,
    weights: Mapping[str, int],
    panels: Sequence[Mapping[str, Any]],
    equations: Sequence[Mapping[str, Any]],
    query_left_terms: Sequence[Mapping[str, Any]],
    query_right_terms: Sequence[Mapping[str, Any]],
    target_relation: str,
    construction_mode: str,
) -> Dict[str, Any]:
    """Validate relation uniqueness or ambiguity and bind answer/annotation."""

    relation_outcomes, assignment_count = _query_relation_outcomes(
        equations=equations,
        labels=UNKNOWN_LABELS,
        support=WEIGHT_SUPPORT,
        left_terms=query_left_terms,
        right_terms=query_right_terms,
    )
    if str(target_relation) == "not_determined":
        if len(relation_outcomes) <= 1:
            raise RuntimeError("not-determined query must allow multiple relations")
    elif relation_outcomes != [str(target_relation)]:
        raise RuntimeError("determined query relation is not unique")

    object_type_offset = int(
        spawn_rng(int(instance_seed), f"{TASK_ID}.object_type_offset").randrange(11)
    )
    answer_label = str(OPTION_BY_RELATION[str(target_relation)])
    return {
        "prompt_query_key": PROMPT_QUERY_KEY,
        "scene_id": SCENE_ID,
        "scene_variant": str(scene_variant),
        "query_row_kind": SIDE_RELATION_ROW_KIND,
        "target_cue_mode": "query_row_only",
        "target_label": answer_label,
        "object_labels": list(UNKNOWN_LABELS),
        "object_specs": object_specs_for_labels(
            UNKNOWN_LABELS,
            offset=int(object_type_offset),
        ),
        "object_weights": {str(key): int(value) for key, value in weights.items()},
        "panels": [dict(panel) for panel in panels],
        "equations": [dict(equation) for equation in equations],
        "query_left_terms": [dict(term) for term in query_left_terms],
        "query_right_terms": [dict(term) for term in query_right_terms],
        "query_relation_outcomes": list(relation_outcomes),
        "consistent_assignment_count": int(assignment_count),
        "target_relation": str(target_relation),
        "relation_options": _relation_options(),
        "construction_mode": str(construction_mode),
        "answer_value": answer_label,
        "answer_labels": list(OPTION_LABELS),
        "answer_range": list(OPTION_LABELS),
        "target_answer_support": list(OPTION_LABELS),
        "annotation_item_id": f"option_{answer_label}",
        "supporting_role_item_ids": {
            "selected_option": f"option_{answer_label}",
        },
    }


def _build_query_side_relation_dataset(
    *,
    instance_seed: int,
    scene_variant: str,
) -> Dict[str, Any]:
    """Construct three reference scales and a four-option relation query."""

    rng = spawn_rng(int(instance_seed), f"{TASK_ID}.dataset")
    target_relation = str(rng.choice(TARGET_RELATIONS))
    if str(target_relation) == "not_determined":
        return _not_determined_dataset(
            rng=rng,
            instance_seed=int(instance_seed),
            scene_variant=str(scene_variant),
        )
    return _determined_relation_dataset(
        target_relation=str(target_relation),
        rng=rng,
        instance_seed=int(instance_seed),
        scene_variant=str(scene_variant),
    )


@register_task
class PuzzlesBalanceScaleQuerySideRelationLabelTask:
    """Choose which side of the query comparison is implied heavier."""

    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation', 'matching')
    domain = "puzzles"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed, *, params, max_attempts):
        """Generate a balanced four-way query-relation option task."""

        selected_query, branch_probs, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=params,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=QUERY_ID,
            task_id=TASK_ID,
            namespace=f"{TASK_ID}.branch",
        )
        axes = resolve_scene_axes(
            instance_seed=int(instance_seed),
            params=task_params,
            gen_defaults=_GEN_DEFAULTS,
            namespace=TASK_ID,
            include_target_cue=False,
        )
        return run_balance_scene_lifecycle(
            task_name=TASK_ID,
            domain=self.domain,
            selected_query=str(selected_query),
            branch_probabilities=branch_probs,
            task_params=task_params,
            render_defaults=_RENDER_DEFAULTS,
            axes=axes,
            instance_seed=int(instance_seed),
            max_attempts=int(max_attempts),
            dataset_builder=lambda seed: _build_query_side_relation_dataset(
                instance_seed=int(seed),
                scene_variant=str(axes.scene_variant),
            ),
            task_prompt_key=TASK_PROMPT_KEY,
            answer_type="string",
            extra_query_params={
                "answer_labels": list(OPTION_LABELS),
                "answer_range": list(OPTION_LABELS),
            },
        )


__all__ = [
    "OPTION_LABELS",
    "PROMPT_QUERY_KEY",
    "PuzzlesBalanceScaleQuerySideRelationLabelTask",
    "QUERY_ID",
    "RELATION_BY_OPTION",
    "SUPPORTED_QUERY_IDS",
    "NEUTRAL_FREE_LABEL_FAMILIES",
    "TARGET_RELATIONS",
    "TASK_ID",
    "WEIGHT_SUPPORT",
]
