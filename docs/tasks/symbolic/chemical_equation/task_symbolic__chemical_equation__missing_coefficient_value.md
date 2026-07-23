# `task_symbolic__chemical_equation__missing_coefficient_value`

## 1) Identity
1. Domain: `symbolic`
2. Scene id: `chemical_equation`
3. Task id: `task_symbolic__chemical_equation__missing_coefficient_value`
4. Objective: solve the one missing coefficient in a displayed chemical equation.

## Program Contract

Program: `chemical_equation.missing_coefficient_value(scene=chemical_equation, scope=one_blank_coefficient_plus_molecule_cards, output=integer)`

Candidate set: the visible symbolic notation, tokens, rows, columns, cards, labels, components, and target markers inside the `one_blank_coefficient_plus_molecule_cards` objective scope.

Operands: the visible atom chips grouped into molecule cards, side/operator layout, and visible coefficients.
Operation: count atoms per molecule card and apply coefficients so both sides match.
Output binding: `answer` is the missing integer coefficient.
Annotation witnesses: a `bbox_set` around the missing coefficient slot and each molecule card.
Query ids: `single`.

## Reasoning Operations

Families: `formula_evaluation`

## 2) Scene + Task Contract
1. Every coefficient, including `1`, is explicitly drawn unless it is the one hidden `?` slot.
2. Molecule cards show repeated atom chips only; formula strings are not drawn.
3. A coefficient multiplies the full adjacent molecule-card term.
4. Supported generated coefficients are integers from `1` through `5`.
5. `answer_gt.type`: `integer`
6. `annotation_gt.type`: `bbox_set`
7. Annotation schema: `bbox_set` with multiple boxes: the blank coefficient slot box plus one molecule-card box for every term.
8. The scene contains a single hidden coefficient slot.

## 3) Prompt Contract
1. Bundle: `symbolic_chemical_equation_v1`
2. `scene_key`: `chemical_equation`
3. `task_key`: `missing_coefficient_value`
4. Modes: `answer_only`, `answer_and_annotation`
5. Prompt text comes from the external prompt bundle.

## 4) Trace Contract
1. `execution_trace.chemical_equation_metadata.reaction` records formulas, balanced coefficients, and atom totals.
2. `execution_trace.terms` records term formulas, coefficients, hidden status, and atom counts.
3. `render_map.item_bboxes_px` contains coefficient-slot and molecule-card projections.
4. `projected_annotation` mirrors the public `bbox_set` annotation.

## 5) Determinism + Tests
1. Deterministic generation and rendering from `instance_seed`.
2. The hidden coefficient is sampled answer-first from the configured support unless pinned by params.
3. Behavior tests: `tests/test_symbolic_chemical_equation_tasks.py`
