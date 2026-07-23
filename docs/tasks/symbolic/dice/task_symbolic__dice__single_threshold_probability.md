# `task_symbolic__dice__single_threshold_probability`

## Public Taxonomy
1. Domain: `symbolic`
2. Scene id: `dice`
3. Task id: `task_symbolic__dice__single_threshold_probability`
4. Objective: single-tray value-threshold probability.

## Program Contract
Program: `dice.single_threshold_probability(scene=dice, scope=single_visible_tray, predicate=value_at_least_threshold|value_at_most_threshold, output=probability_option_letter)`

Candidate set: the visible dice in the single tray.
Operands: each die's top-face value, the sampled threshold, and the at-least or at-most predicate.
Operation: count dice whose value satisfies the threshold predicate and reduce `matching_dice / total_dice`.
Output binding: `answer` is the letter of the visible A-F option whose reduced fraction is the requested probability.
Annotation schema: `bbox`.
Annotation witnesses: the scalar bbox of the single visible dice tray.
Query ids: `single_value_at_least_probability`, `single_value_at_most_probability`.

## Reasoning Operations

Families: `filtering`, `counting`, `comparison`, `formula_evaluation`

## Generation And Trace
1. The task asks for a probability over a sampled visible top-value threshold; `query_id` selects at-least versus at-most semantics.
2. The execution trace records tray specs, die colors, die values, favorable outcome counts, total outcome counts, the exact reduced fraction, visible option fractions, and the selected answer label.
3. Scene variants are `dice_tray_clean`, `dice_tray_felt`, and `dice_tray_notebook`; dice visual style is non-semantic rendering metadata.
4. Annotation is projected from finalized tray geometry, not from pixels.
5. Scalar annotation checked: true.

## Implementation
1. Registered class: `trace_tasks.tasks.symbolic.dice.single_threshold_probability.SymbolicProbabilityDiceSingleThresholdProbabilityTask`
2. Config: `src/trace_tasks/resources/configs/domains/symbolic/dice.yaml`
3. Prompt asset: `src/trace_tasks/resources/prompts/symbolic/dice/symbolic_dice_probability_v1.json`
4. Focused tests: `tests/test_symbolic_probability_dice_tasks.py`
