# `task_symbolic__chemical_equation__balanced_option_label`

## 1) Identity
1. Domain: `symbolic`
2. Scene id: `chemical_equation`
3. Task id: `task_symbolic__chemical_equation__balanced_option_label`
4. Objective: select the coefficient option that balances the displayed chemical equation.

## Program Contract
Program: `chemical_equation.balanced_option_label(scene=chemical_equation, scope=blank_coefficient_slots_plus_option_cards, output=option_letter)`

Candidate set: the four visible coefficient option cards labeled `A..D`.
Operands: the visible atom chips grouped into molecule cards, side/operator layout, and option coefficient tuples.
Operation: fill coefficient slots from left to right using each option and compare atom totals on both sides.
Output binding: `answer` is the selected option label.
Annotation witnesses: a scalar `bbox` around the selected option card.
Query ids: `single`.

## Reasoning Operations

Families: `formula_evaluation`

## 2) Scene + Task Contract
1. All coefficient slots in the equation are hidden as `?` boxes.
2. Molecule cards show repeated atom chips only; formula strings are not drawn.
3. Each option card contains a left-to-right coefficient tuple for the equation terms.
4. Exactly one option balances all atom counts.
5. `answer_gt.type`: `string`
6. `annotation_gt.type`: `bbox`
7. Annotation schema: `bbox`
8. Candidate labels are `A`, `B`, `C`, and `D`.

## 3) Prompt Contract
1. Bundle: `symbolic_chemical_equation_v1`
2. `scene_key`: `chemical_equation`
3. `task_key`: `balanced_option_label`
4. Modes: `answer_only`, `answer_and_annotation`
5. Prompt text comes from the external prompt bundle.

## 4) Trace Contract
1. `execution_trace.chemical_equation_metadata.reaction` records formulas, balanced coefficients, and atom totals.
2. `execution_trace.chemical_equation_metadata.options` records each candidate tuple and whether it balances the equation.
3. `render_map.option_bboxes_px` contains option-card projections.
4. `projected_annotation` mirrors the selected option `bbox`.

## 5) Determinism + Tests
1. Deterministic generation and rendering from `instance_seed`.
2. Distractors are unique and rejected if they accidentally balance the equation.
3. Behavior tests: `tests/test_symbolic_chemical_equation_tasks.py`
