# `task_symbolic__dice__pair_difference_probability`

## Public Taxonomy
1. Domain: `symbolic`
2. Scene id: `dice`
3. Task id: `task_symbolic__dice__pair_difference_probability`
4. Objective: two-tray absolute-difference probability.

## Program Contract
Program: `dice.pair_difference_probability(scene=dice, scope=two_visible_trays, predicate=absolute_difference, output=probability_option_letter)`

Candidate set: the cartesian product of one die from tray `A` and one die from tray `B`.
Operands: each die's top-face value in both trays and the sampled absolute-difference target.
Operation: count ordered tray-pair outcomes whose absolute value difference equals the target and reduce `favorable / total_outcomes`.
Output binding: `answer` is the letter of the visible A-F option whose reduced fraction is the requested probability.
Annotation schema: `bbox_map`.
Annotation witnesses: a `bbox_map` with `tray_a` and `tray_b` bboxes.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `formula_evaluation`

## Generation And Trace
1. The task asks for the probability that the absolute difference between the two selected visible values equals a sampled value.
2. The execution trace records tray specs, die colors, die values, favorable outcome counts, total outcome counts, the exact reduced fraction, visible option fractions, and the selected answer label.
3. Scene variants are `dice_tray_clean`, `dice_tray_felt`, and `dice_tray_notebook`; dice visual style is non-semantic rendering metadata.
4. Annotation is projected from finalized tray geometry, not from pixels.
5. Scalar annotation checked: true.

## Implementation
1. Registered class: `trace_tasks.tasks.symbolic.dice.pair_difference_probability.SymbolicProbabilityDicePairDifferenceProbabilityTask`
2. Config: `src/trace_tasks/resources/configs/domains/symbolic/dice.yaml`
3. Prompt asset: `src/trace_tasks/resources/prompts/symbolic/dice/symbolic_dice_probability_v1.json`
4. Focused tests: `tests/test_symbolic_probability_dice_tasks.py`
