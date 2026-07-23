# `task_symbolic__dice__pair_sum_threshold_probability`

## Public Taxonomy
1. Domain: `symbolic`
2. Scene id: `dice`
3. Task id: `task_symbolic__dice__pair_sum_threshold_probability`
4. Objective: two-tray sum-threshold probability.

## Program Contract
Program: `dice.pair_sum_threshold_probability(scene=dice, scope=two_visible_trays, predicate=sum_at_least_threshold|sum_at_most_threshold, output=probability_option_letter)`

Candidate set: the cartesian product of one die from tray `A` and one die from tray `B`.
Operands: each die's top-face value in both trays, the sampled sum threshold, and the at-least or at-most predicate.
Operation: count ordered tray-pair outcomes whose value sum satisfies the threshold predicate and reduce `favorable / total_outcomes`.
Output binding: `answer` is the letter of the visible A-F option whose reduced fraction is the requested probability.
Annotation schema: `bbox_map`.
Annotation witnesses: a `bbox_map` with `tray_a` and `tray_b` bboxes.
Query ids: `pair_sum_at_least_probability`, `pair_sum_at_most_probability`.

## Reasoning Operations

Families: `filtering`, `counting`, `comparison`, `formula_evaluation`

## Generation And Trace
1. The task asks for the probability that one uniformly selected die from each tray satisfies a sampled sum threshold; `query_id` selects at-least versus at-most semantics.
2. The execution trace records tray specs, die colors, die values, favorable outcome counts, total outcome counts, the exact reduced fraction, visible option fractions, and the selected answer label.
3. Scene variants are `dice_tray_clean`, `dice_tray_felt`, and `dice_tray_notebook`; dice visual style is non-semantic rendering metadata.
4. Annotation is projected from finalized tray geometry, not from pixels.
5. Scalar annotation checked: true.

## Implementation
1. Registered class: `trace_tasks.tasks.symbolic.dice.pair_sum_threshold_probability.SymbolicProbabilityDicePairSumThresholdProbabilityTask`
2. Config: `src/trace_tasks/resources/configs/domains/symbolic/dice.yaml`
3. Prompt asset: `src/trace_tasks/resources/prompts/symbolic/dice/symbolic_dice_probability_v1.json`
4. Focused tests: `tests/test_symbolic_probability_dice_tasks.py`
