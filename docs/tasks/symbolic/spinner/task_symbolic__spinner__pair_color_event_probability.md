# `task_symbolic__spinner__pair_color_event_probability`

## Public Taxonomy
1. Domain: `symbolic`
2. Scene id: `spinner`
3. Task id: `task_symbolic__spinner__pair_color_event_probability`
4. Objective: two-spinner color-event probability.

## Program Contract
Program: `spinner.pair_color_event_probability(scene=spinner, scope=two_independent_equal_sector_spinners, event=both_target_color|at_least_one_target_color|same_color, sample_space=cartesian_product_of_visible_sectors, output=probability_option_letter)`

Candidate set: the cartesian product of visible sectors from Spinner A and Spinner B.
Operands: the sector colors on both spinners and the query event predicate.
Operation: count ordered two-spinner outcomes satisfying the event predicate and reduce `matching_outcomes / total_outcomes`.
Output binding: `answer` is the letter of the visible A-F option whose reduced fraction is the requested probability.
Annotation schema: `bbox_map`.
Annotation witnesses: a `bbox_map` with `spinner_a` and `spinner_b` panel bboxes.
Query ids: `pair_both_target_color_probability`, `pair_at_least_one_target_color_probability`, `pair_same_color_probability`.

## Reasoning Operations

Families: `filtering`, `counting`, `comparison`, `formula_evaluation`

## Query Contract
- `pair_both_target_color_probability`: both spinners must land on the same resolved target color.
- `pair_at_least_one_target_color_probability`: at least one spinner must land on the resolved target color.
- `pair_same_color_probability`: the two selected sectors must have the same color.

## Generation And Trace
1. The task samples one two-spinner color event with a nontrivial favorable count.
2. The execution trace records sector specs for both spinners, favorable outcome counts, total outcome counts, the exact reduced fraction, visible option fractions, and the selected answer label.
3. Scene variants are `spinner_clean`, `spinner_card`, and `spinner_notebook`.
4. Annotation is projected from finalized spinner-panel geometry, not from pixels.

## Implementation
1. Registered class: `trace_tasks.tasks.symbolic.spinner.pair_color_event_probability.SymbolicSpinnerPairColorEventProbabilityTask`
2. Config: `src/trace_tasks/resources/configs/domains/symbolic/spinner.yaml`
3. Prompt asset: `src/trace_tasks/resources/prompts/symbolic/spinner/symbolic_spinner_v1.json`
4. Focused tests: `tests/test_symbolic_probability_spinner_tasks.py`
