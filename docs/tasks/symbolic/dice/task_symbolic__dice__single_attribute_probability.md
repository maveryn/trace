# `task_symbolic__dice__single_attribute_probability`

## Public Taxonomy
1. Domain: `symbolic`
2. Scene id: `dice`
3. Task id: `task_symbolic__dice__single_attribute_probability`
4. Objective: single-tray attribute probability.

## Program Contract
Program: `dice.single_attribute_probability(scene=dice, scope=single_visible_tray, predicate=parity|value_set|color_and_value|color_or_value, output=probability_option_letter)`

Candidate set: the visible dice in the single tray.
Operands: each die's color and top-face value, plus the resolved parity, value-set, color-and-value, or color-or-value predicate.
Operation: count dice satisfying the selected predicate and reduce `matching_dice / total_dice`.
Output binding: `answer` is the letter of the visible A-F option whose reduced fraction is the requested probability.
Annotation schema: `bbox`.
Annotation witnesses: the scalar bbox of the single visible dice tray.
Query ids: `single_parity_probability`, `single_value_set_probability`, `single_color_and_value_probability`, `single_color_or_value_probability`.

## Reasoning Operations

Families: `filtering`, `counting`, `logical_composition`, `formula_evaluation`

## Generation And Trace
1. The task samples one semantic predicate branch and asks for the probability that a uniformly selected die from the tray satisfies it.
2. The execution trace records tray specs, die colors, die values, favorable outcome counts, total outcome counts, the exact reduced fraction, visible option fractions, and the selected answer label.
3. Scene variants are `dice_tray_clean`, `dice_tray_felt`, and `dice_tray_notebook`; dice visual style is non-semantic rendering metadata.
4. Annotation is projected from finalized tray geometry, not from pixels.
5. Scalar annotation checked: true.

## Implementation
1. Registered class: `trace_tasks.tasks.symbolic.dice.single_attribute_probability.SymbolicProbabilityDiceSingleAttributeProbabilityTask`
2. Config: `src/trace_tasks/resources/configs/domains/symbolic/dice.yaml`
3. Prompt asset: `src/trace_tasks/resources/prompts/symbolic/dice/symbolic_dice_probability_v1.json`
4. Focused tests: `tests/test_symbolic_probability_dice_tasks.py`
